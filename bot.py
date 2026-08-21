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

import base64
import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
import actions
import reasoning
import screen
import voice


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


_TAP_REPEAT_TOLERANCE_PX = 25
_TAP_REPEAT_THRESHOLD = 2  # this many consecutive near-identical taps triggers a stop


def _is_stuck_repeating_tap(x: int, y: int, history: list) -> bool:
    """
    Returns True if the last _TAP_REPEAT_THRESHOLD executed tap actions
    in history all landed within _TAP_REPEAT_TOLERANCE_PX pixels of the
    newly proposed (x, y). Gemini's system prompt already instructs it to
    check history and not repeat a dead action, but this isn't reliably
    followed in practice — observed directly in testing, where a
    content-rating dialog's OK button was tapped ~8 times in a row at
    near-identical coordinates with no progress, each time proposed as a
    nominally "different" tree target (index:4, index:2, index:4...) that
    the vision fallback resolved to essentially the same pixel every
    time. Enforcing this in code rather than trusting the model to
    self-regulate stops a runaway loop — and the repeated confirmation
    prompts that come with it — automatically instead of relying on the
    owner to notice and kill the process.
    """
    recent_taps = [
        h for h in history[-_TAP_REPEAT_THRESHOLD:]
        if h.get("action") == "tap" and "x" in h.get("args", {}) and "y" in h.get("args", {})
    ]
    if len(recent_taps) < _TAP_REPEAT_THRESHOLD:
        return False
    return all(
        abs(h["args"]["x"] - x) <= _TAP_REPEAT_TOLERANCE_PX
        and abs(h["args"]["y"] - y) <= _TAP_REPEAT_TOLERANCE_PX
        for h in recent_taps
    )


def is_affirmative(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!.?")
    if cleaned in _AFFIRMATIVE_REPLIES:
        return True
    # Tolerate simple elongated typos like "yess"/"yesss" — bounded to a
    # short length so this can't accidentally match a longer, unrelated
    # sentence that merely starts with "yes". Stays conservative: this
    # does NOT add new words to the affirmative set, just forgives minor
    # typing slips on the existing ones most likely to occur ("yes").
    if cleaned.startswith("yes") and len(cleaned) <= 6:
        return True
    return False


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


def send_photo(chat_id, image_b64: str, caption: str = "", mime_type: str = "image/jpeg"):
    """Push a base64-encoded image to Telegram as a photo message."""
    photo_bytes = base64.b64decode(image_b64)
    # Guess a sensible filename from the MIME type so Telegram's client
    # renders it as an image rather than a generic document.
    ext = "jpg" if "jpeg" in (mime_type or "") else "png" if "png" in (mime_type or "") else "jpg"
    files = {"photo": (f"screenshot.{ext}", photo_bytes, mime_type or "image/jpeg")}
    data = {"chat_id": str(chat_id)}
    if caption:
        data["caption"] = caption
    resp = requests.post(f"{API_BASE}/sendPhoto", data=data, files=files, timeout=30)
    resp.raise_for_status()


def _send_current_screenshot(chat_id, caption: str = "Current screen") -> None:
    """
    Capture via actions.get_screenshot() and push to Telegram. Shared by
    the explicit send_screenshot agent action and the stuck-detection
    path — same plumbing, two triggers.
    """
    screenshot = actions.get_screenshot()
    send_photo(
        chat_id,
        screenshot["image"],
        caption=caption or "Current screen",
        mime_type=screenshot.get("mimeType", "image/jpeg"),
    )


def handle_message(message):
    chat_id = message["chat"]["id"]
    sender_id = str(message["from"]["id"])

    # Security gate: only respond to the configured owner.
    if sender_id != str(ALLOWED_USER_ID):
        log.warning("Ignored message from unauthorized user_id=%s", sender_id)
        return

    # Push-to-talk: a voice note is transcribed offline (voice.py, Vosk +
    # ffmpeg) into plain text, then dropped into the exact same
    # dispatch_command / handle_confirmation_reply path as a typed
    # message below — voice is purely an input adapter, nothing
    # downstream needs to know the command didn't arrive as text.
    voice_note = message.get("voice")
    if voice_note:
        try:
            text = voice.transcribe_voice_message(voice_note["file_id"])
        except voice.VoiceError as e:
            log.warning("Voice transcription failed: %s", e)
            send_message(chat_id, f"🎤⚠️ Couldn't understand that: {e}")
            return
        log.info("Transcribed voice: %s", text)
        # Echo the transcript back before acting on it. STT isn't
        # perfect, and this bot executes real taps/purchases — the owner
        # should see what was heard rather than trust it blindly, the
        # same spirit as the confirmation gate further down the pipeline.
        send_message(chat_id, f"🎤 Heard: \u201c{text}\u201d")
    else:
        text = message.get("text", "")
        if not text:
            return  # no text, no voice (e.g. a sticker) — nothing to act on

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
        ok, result = _execute_action(
            execute_first["action"], execute_first["args"], chat_id
        )
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
                # Before falling back to vision, retry the tree match once
                # with extra settle time. Confirmed via a real /screen
                # capture during testing: a target like "Install" can be
                # genuinely present in the tree with an exact text match —
                # it just wasn't rendered yet at the original capture time
                # (e.g. an animated bottom sheet still settling), and
                # STEP_SETTLE_SECONDS wasn't long enough to catch it. A
                # cheap retry resolves that timing case via the tree
                # (fast, exact) instead of falling through to vision,
                # which is both slower and — also confirmed via testing —
                # meaningfully less precise (observed landing ~100px off
                # a real button's position).
                log.info(
                    "Tap target %r not found on first try — retrying tree "
                    "match after extra settle time before vision fallback",
                    target,
                )
                time.sleep(1.5)
                try:
                    retry_screen = actions.get_screen_state()
                    coords = screen.find_target_bounds(retry_screen, target)
                except Exception as e:
                    log.warning("Retry screen capture failed: %s", e)
                    coords = None
                if coords is not None:
                    current_screen = retry_screen

            if coords is None:
                # Tree still doesn't have it after the retry — a genuine
                # vision fallback case (e.g. a truly unlabeled icon the
                # tree can never describe), not just a timing issue. Fall
                # back to a real screenshot + vision-based reasoning
                # instead of just failing the step. This REPLACES the
                # tree decision for this turn: decision/action/args below
                # all become the vision call's output, which shares the
                # exact same {action, args, requires_confirmation,
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
                # scroll, send_screenshot) needs no resolution at all and
                # falls straight through to the confirmation/execution
                # code below.
            else:
                args = {**args, "x": coords[0], "y": coords[1]}

        if action == "tap" and _is_stuck_repeating_tap(args["x"], args["y"], history):
            log.warning(
                "Detected repeated tap at (%s, %s) with no progress — stopping instead of looping",
                args["x"], args["y"],
            )
            history.append(
                {
                    "action": action,
                    "args": args,
                    "result": "skipped: repeated tap at same location without progress — stopped to avoid a loop",
                }
            )
            # Same send-photo plumbing as the explicit send_screenshot
            # action — stuck-detection is just a different trigger.
            try:
                _send_current_screenshot(
                    chat_id,
                    caption="I got stuck — here's what I see on the screen.",
                )
            except Exception as e:
                log.warning("Failed to send stuck-state screenshot: %s", e)
            return (
                "🤔 I tapped the same spot a couple of times without anything changing, "
                "so I stopped instead of repeating it further. Screenshot of what I see "
                "is above — a dialog may need a different response than expected."
            )

        if decision["requires_confirmation"]:
            reason = decision["confirmation_reason"] or f"{action}({args})"
            session["pending"] = {"action": action, "args": args}
            AGENT_SESSIONS[chat_id] = session
            log.info("Pausing for confirmation: %s", reason)
            return (
                f"⏸️ Before I continue: {reason}\n\n"
                "Reply yes to proceed, or anything else to cancel."
            )

        ok, result = _execute_action(action, args, chat_id)
        history.append({"action": action, "args": args, "result": result})
        if not ok:
            log.warning("Step failed: %s(%s) -> %s", action, args, result)

        time.sleep(STEP_SETTLE_SECONDS)

    return (
        f"⚠️ Stopped after {MAX_AGENT_STEPS} steps without finishing. "
        "Try breaking the request into smaller pieces."
    )


def _execute_action(action: str, args: dict, chat_id) -> tuple:
    """
    Executes a single already-decided action. Returns (ok, result_str).
    Never raises — any failure (device rejects the action, tap target
    isn't on screen) is reported back as a result string rather than an
    exception, so the caller can feed it into history for Gemini to see
    and adjust on the next turn, the same recoverable-by-design approach
    as before the confirmation gate existed.

    chat_id is needed for send_screenshot (Telegram sendPhoto); ignored
    by every other action.
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

        if action == "send_screenshot":
            caption = (args.get("caption") or "").strip() or "Current screen"
            _send_current_screenshot(chat_id, caption=caption)
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