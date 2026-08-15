"""
Pocket Jarvis - Phase 4
Telegram long-polling bot that triggers real device actions via actions.py.
Literal commands (/open, /type) stay as a fast/power-user path. Anything
else is handled by run_agent_loop(): a multi-step act-observe loop where
Gemini sees the actual screen state and decides one action at a time,
rather than committing to a whole plan up front.
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
        reply = dispatch_command(text)
    except Exception as e:
        log.exception("Action failed: %s", e)
        reply = f"⚠️ Action failed: {e}"

    send_message(chat_id, reply)
    log.info("Replied.")


def dispatch_command(text):
    """
    Literal /open and /type stay as a fast, unambiguous power-user path.
    Anything else falls through to run_agent_loop(), the multi-step
    act-observe loop. Every failure mode along the way (screen capture,
    reasoning, action execution) fails safely — no action is executed on
    a bad parse, and the loop stops and reports rather than guessing.
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
            "                screen and decides one step at a time)\n"
        )

    # Phase 4: route anything non-literal through the multi-step
    # act-observe loop instead of a single one-shot Gemini call.
    return run_agent_loop(text)


def run_agent_loop(goal: str) -> str:
    """
    Capture screen -> ask Gemini for one next action -> execute it ->
    re-capture -> repeat, until Gemini says "done" (goal achieved), "none"
    (stuck/ambiguous), a step fails outright, or MAX_AGENT_STEPS is hit.

    Deliberately does NOT try to plan ahead — every decision is made from
    a fresh, real screen capture, so the loop stays correct even when an
    app takes a moment to load or a screen looks different than expected.
    """
    history = []

    for step_num in range(1, MAX_AGENT_STEPS + 1):
        try:
            current_screen = actions.get_screen_state()
        except Exception as e:
            log.exception("Step %s: failed to capture screen", step_num)
            return f"⚠️ Couldn't read the screen at step {step_num}: {e}"

        try:
            decision = reasoning.decide_next_step(goal, current_screen, history)
        except reasoning.ReasoningError as e:
            log.warning("Step %s: reasoning failed: %s", step_num, e)
            return "🤔 Got stuck figuring out the next step — try rephrasing the request."

        action, args = decision["action"], decision["args"]
        log.info("Step %s: %s(%s)", step_num, action, args)

        if action == "done":
            return f"✅ {args.get('message', 'Done.')}"

        if action == "none":
            return f"🤔 {args.get('reason', 'Not sure how to proceed from here.')}"

        try:
            if action == "open_app":
                actions.open_app(args["package"])
                history.append({"action": action, "args": args, "result": "ok"})

            elif action == "type_text":
                actions.type_text(args["text"])
                history.append({"action": action, "args": args, "result": "ok"})

            elif action == "scroll":
                actions.scroll(args["direction"])
                history.append({"action": action, "args": args, "result": "ok"})

            elif action == "tap":
                target = args["target"]
                coords = screen.find_target_bounds(current_screen, target)
                if coords is None:
                    # Not a hard failure — tell Gemini via history so it
                    # can try scrolling or a different target next turn,
                    # rather than aborting the whole task over one miss.
                    log.warning("Step %s: tap target not found: %r", step_num, target)
                    history.append(
                        {
                            "action": action,
                            "args": args,
                            "result": "failed: target not found on screen",
                        }
                    )
                    continue
                actions.tap(*coords)
                history.append({"action": action, "args": args, "result": "ok"})

        except Exception as e:
            log.exception("Step %s: action failed", step_num)
            return f"⚠️ Action failed at step {step_num} ({action}): {e}"

        time.sleep(STEP_SETTLE_SECONDS)

    return (
        f"⚠️ Stopped after {MAX_AGENT_STEPS} steps without finishing. "
        "Try breaking the request into smaller pieces."
    )


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