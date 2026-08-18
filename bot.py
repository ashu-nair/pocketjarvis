"""
Pocket Jarvis - Phase 4
Telegram long-polling bot that triggers real device actions via actions.py.
Literal commands (/open, /type) stay as a fast/power-user path. Anything
else is handled by the multi-step act-observe loop (continue_agent_loop):
Gemini sees the actual screen state and decides one action at a time,
pausing to ask for explicit confirmation before anything consequential
(installing, sending, buying, agreeing to something) rather than acting
on its own judgment.

--- Vision fallback (Known Issues #1) -----------------------------------
When the tree-based decision names a tap "target" that
screen.find_target_bounds() can't resolve to real coordinates — usually
an unlabeled icon (Play Store's search bar) or a UI that changed since it
was last mapped (Skyscanner) — continue_agent_loop falls back to a real
screenshot + reasoning.decide_next_step_vision() instead of failing the
step outright. The vision decision replaces the tree decision for that
turn and flows through the exact same confirmation-gate and execution
code below it, since both paths return the same
{action, args, requires_confirmation, confirmation_reason} shape.
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
import actions
import reasoning
import screen


BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.environ.get("TELEGRAM_USER_ID")
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "3"))
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "10"))
# Brief pause after each action before re-capturing the screen. This is a
# known-temporary stand-in for real readiness detection (waiting for the
# target app's window to actually be foregrounded) — flagged in the
# project roadmap as something to replace, not something to trust long
# term. Fine for now since every action we have is fast on a real device.
STEP_SETTLE_SECONDS = float(os.environ.get("STEP_SETTLE_SECONDS", "1.5"))

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pocket-jarvis")

# Per-chat paused-loop state, keyed by chat_id. When an agent loop hits an
# action Gemini flagged requires_confirmation=True, it stores its progress
# here (goal + history so far + the exact action awaiting approval) and
# returns control to Telegram instead of executing. The next message from
# that chat is checked against this dict first — if there's a pending
# entry, the message is treated as a yes/no reply rather than a new
# command. Lives only in memory: a bot restart clears any paused loop.
AGENT_SESSIONS = {}

_AFFIRMATIVE_REPLIES = {
    "yes", "y", "yeah", "yep", "yup", "confirm", "ok", "okay",
    "proceed", "go", "go ahead", "do it", "approved", "approve",
}


def is_affirmative(text: str) -> bool:
    return text.strip().lower() in _AFFIRMATIVE_REPLIES


def validate_config():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Check your .env file.")
    if not ALLOWED_USER_ID:
        raise RuntimeError("TELEGRAM_USER_ID is not set. Check your .env file.")


def get_updates(offset=None, timeout=30):
    """Long-poll Telegram for new messages."""
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(f"{API_BASE}/getUpdates", params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json().get("result", [])


def send_message(chat_id, text):
    requests.post(
        f"{API_BASE}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )


def handle_message(message):
    chat_id = message["chat"]["id"]
    sender_id = str(message["from"]["id"])
    text = message.get("text", "")

    # Security gate: only respond to the configured owner.
    if sender_id != str(ALLOWED_USER_ID):
        log.warning("Ignored message from unauthorized user_id=%s", sender_id)
        return

    log.info("Received: %s", text)

    try:
        if chat_id in AGENT_SESSIONS:
            reply = handle_confirmation_reply(chat_id, text)
        else:
            reply = dispatch_command(text, chat_id)
    except Exception as e:
        log.exception("Action failed: %s", e)
        reply = f"⚠️ Action failed: {e}"

    send_message(chat_id, reply)
    log.info("Replied.")


def handle_confirmation_reply(chat_id, text: str) -> str:
    """Called when a message arrives while this chat has a paused loop
    waiting on a yes/no. Anything not recognized as affirmative is
    treated as a decline — a paused, consequential action should never
    proceed on an ambiguous reply."""
    session = AGENT_SESSIONS.pop(chat_id)

    if not is_affirmative(text):
        log.info("Confirmation declined for chat_id=%s", chat_id)
        return "❌ Cancelled — that step was not carried out."

    log.info("Confirmation approved for chat_id=%s", chat_id)
    pending = session["pending"]
    session["history"].append(
        {"action": pending["action"], "args": pending["args"], "result": "approved by owner"}
    )
    return continue_agent_loop(chat_id, session, execute_first=pending)


def dispatch_command(text, chat_id):
    """
    Literal /open and /type stay as a fast, unambiguous power-user path.
    Anything else falls through to the multi-step act-observe loop. Every
    failure mode along the way (screen capture, reasoning, action
    execution) fails safely — no action is executed on a bad parse, and
    the loop stops and reports rather than guessing.
    """
    text = text.strip()

    if text.startswith("/open"):
        app_hint = text[len("/open"):].strip() or "default"
        actions.open_app(app_hint)
        return f"✅ Opened: {app_hint}"

    if text.startswith("/type"):
        content = text[len("/type"):].strip()
        if not content:
            return "Usage: /type <text to type>"
        actions.type_text(content)
        return f"✅ Typed: {content}"

    if text in ("/help", "/start"):
        return (
            "Pocket Jarvis (Phase 4)\n\n"
            "/open <app>   - open an app directly\n"
            "/type <text>  - type into the currently focused field\n"
            "Anything else - handled via the agent loop (Gemini sees the\n"
            "                screen and decides one step at a time; you'll\n"
            "                be asked to confirm anything consequential)\n"
        )

    # Phase 4: route anything non-literal through the multi-step
    # act-observe loop instead of a single one-shot Gemini call.
    session = {"goal": text, "history": []}
    return continue_agent_loop(chat_id, session)


def continue_agent_loop(chat_id, session, execute_first=None) -> str:
    """
    Runs (or resumes) the act-observe loop for one chat's session until
    it finishes ("done"/"none"), pauses again on another confirmation,
    or hits MAX_AGENT_STEPS.

    session: {"goal": str, "history": list}. Mutated/extended in place
    as the loop progresses; if the loop pauses, this exact dict (with
    "pending" added) is what gets stored in AGENT_SESSIONS for the reply
    to resume from.

    execute_first: when resuming after an approved confirmation, the
    exact action dict that was awaiting approval — executed before
    capturing a fresh screen and asking Gemini for the next step, so the
    approved action doesn't need to be re-decided.
    """
    goal = session["goal"]
    history = session["history"]

    if execute_first is not None:
        ok, result = _execute_action(execute_first["action"], execute_first["args"])
        # Overwrite the "approved by owner" placeholder appended just
        # before this call with the real outcome, so history reflects
        # what actually happened, not just that it was approved.
        history[-1]["result"] = result
        if not ok:
            log.warning("Approved action failed: %s", result)
        time.sleep(STEP_SETTLE_SECONDS)

    remaining = MAX_AGENT_STEPS - len(history)
    if remaining <= 0:
        # Total steps across this whole session (including earlier
        # confirmation rounds) has hit the cap. Stop here rather than
        # continuing to pause-and-resume indefinitely — without this
        # check, a task that keeps needing confirmation (e.g. because a
        # tap isn't actually landing where intended and the same dialog
        # keeps reappearing) could ask for approval forever, since each
        # resumed call would otherwise still attempt "at least 1" more
        # step no matter how large history had already grown.
        return (
            f"⚠️ Stopped after {MAX_AGENT_STEPS} steps without finishing. "
            "Try breaking the request into smaller pieces."
        )

    for _ in range(remaining):
        try:
            current_screen = actions.get_screen_state()
        except Exception as e:
            log.exception("Failed to capture screen")
            return f"⚠️ Couldn't read the screen: {e}"

        try:
            decision = reasoning.decide_next_step(goal, current_screen, history)
        except reasoning.ReasoningError as e:
            # Recoverable, not fatal: a single malformed decision (e.g. an
            # empty tap target) shouldn't throw away everything the loop
            # already accomplished this turn (like having opened an app).
            # Record it and let the loop try again on a fresh decision,
            # same as any other failed step — still bounded by the
            # for-loop's step budget, so a persistently broken Gemini
            # response can't loop forever, it just eventually falls
            # through to the "stopped after N steps" message below.
            log.warning("Reasoning failed: %s", e)
            history.append(
                {"action": "invalid_decision", "args": {}, "result": f"failed: {e}"}
            )
            time.sleep(STEP_SETTLE_SECONDS)
            continue

        action, args = decision["action"], decision["args"]
        log.info("Decision: %s(%s)", action, args)

        if action == "done":
            return f"✅ {args.get('message', 'Done.')}"

        if action == "none":
            return f"🤔 {args.get('reason', 'Not sure how to proceed from here.')}"

        if action == "tap":
            # Resolve the target text to real pixel coordinates NOW, using
            # this exact screen capture — not deferred to whenever the
            # owner replies to a confirmation, since the screen could look
            # completely different by then. args gets the resolved x/y
            # baked in, so both the immediate-execute and the
            # pause-for-confirmation paths below use the same coordinates.
            target = args["target"]
            coords = screen.find_target_bounds(current_screen, target)

            if coords is None:
                # Tree-based matching failed — usually an unlabeled icon
                # (no text/desc/resourceId to match on) or a UI that
                # changed since last mapped. Fall back to a real
                # screenshot + vision-based reasoning instead of just
                # failing the step. This REPLACES the tree decision for
                # this turn: decision/action/args below all become the
                # vision call's output, which shares the exact same
                # {action, args, requires_confirmation,
                # confirmation_reason} shape as the tree path, so the
                # confirmation-gate and execution code further down
                # doesn't need to know which path produced it.
                log.warning(
                    "Tap target not found via tree: %r — trying vision fallback", target
                )
                try:
                    screenshot = actions.get_screenshot()
                    decision = reasoning.decide_next_step_vision(goal, screenshot, history)
                except (RuntimeError, reasoning.ReasoningError) as e:
                    log.warning("Vision fallback failed: %s", e)
                    history.append(
                        {
                            "action": action,
                            "args": args,
                            "result": (
                                "failed: target not found on screen; "
                                f"vision fallback also failed: {e}"
                            ),
                        }
                    )
                    time.sleep(STEP_SETTLE_SECONDS)
                    continue

                action, args = decision["action"], decision["args"]
                log.info("Vision decision: %s(%s)", action, args)

                if action == "done":
                    return f"✅ {args.get('message', 'Done.')}"

                if action == "none":
                    return f"🤔 {args.get('reason', 'Not sure how to proceed from here.')}"

                # If vision also decided "tap", args["x"]/args["y"] are
                # already real device pixel coordinates — rescaled inside
                # reasoning.decide_next_step_vision() from the downscaled
                # screenshot's coordinate space, so no further resolution
                # is needed here. Any other action (open_app, type_text,
                # scroll) needs no resolution at all and falls straight
                # through to the confirmation/execution code below.
            else:
                args = {**args, "x": coords[0], "y": coords[1]}

        if decision["requires_confirmation"]:
            reason = decision["confirmation_reason"] or f"{action}({args})"
            session["pending"] = {"action": action, "args": args}
            AGENT_SESSIONS[chat_id] = session
            log.info("Pausing for confirmation: %s", reason)
            return (
                f"⏸️ Before I continue: {reason}\n\n"
                "Reply yes to proceed, or anything else to cancel."
            )

        ok, result = _execute_action(action, args)
        history.append({"action": action, "args": args, "result": result})
        if not ok:
            log.warning("Step failed: %s(%s) -> %s", action, args, result)

        time.sleep(STEP_SETTLE_SECONDS)

    return (
        f"⚠️ Stopped after {MAX_AGENT_STEPS} steps without finishing. "
        "Try breaking the request into smaller pieces."
    )


def _execute_action(action: str, args: dict) -> tuple:
    """
    Executes a single already-decided action. Returns (ok, result_str).
    Never raises — any failure (device rejects the action, tap target
    isn't on screen) is reported back as a result string rather than an
    exception, so the caller can feed it into history for Gemini to see
    and adjust on the next turn, the same recoverable-by-design approach
    as before the confirmation gate existed.
    """
    try:
        if action == "open_app":
            actions.open_app(args["package"])
            return True, "ok"

        if action == "type_text":
            actions.type_text(args["text"])
            return True, "ok"

        if action == "scroll":
            actions.scroll(args["direction"])
            return True, "ok"

        if action == "tap":
            # x/y are always pre-resolved by continue_agent_loop before
            # this is called (tree match, vision fallback, or a resumed
            # approved confirmation) — this function only ever sends real
            # coordinates to the device, never a target string.
            if "x" in args and "y" in args:
                actions.tap(args["x"], args["y"])
                return True, "ok"
            return False, "failed: tap target was not resolved to coordinates"

    except Exception as e:
        return False, f"failed: {e}"

    return False, f"failed: unknown action {action!r}"


def main():
    validate_config()
    log.info("Pocket Jarvis Phase 4 starting. Polling for messages...")

    offset = None
    while True:
        try:
            updates = get_updates(offset=offset)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message")
                if message:
                    handle_message(message)
        except requests.exceptions.RequestException as e:
            log.error("Network error: %s. Retrying in %ss...", e, POLL_INTERVAL_SECONDS)
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            log.exception("Unexpected error: %s", e)
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()