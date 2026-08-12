"""
Pocket Jarvis - Phase 3
Telegram long-polling bot that triggers real device actions via actions.py.
Literal commands (/open, /type) still work as a fast/power-user path.
Anything else is routed through reasoning.py, which calls the Gemini API
to turn natural language into a structured action call.
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
import actions
import reasoning


BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.environ.get("TELEGRAM_USER_ID")
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "3"))

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
    Anything else falls through to reasoning.py (Gemini) for natural
    language parsing. If Gemini's output doesn't parse into a valid,
    fully-specified action, this fails safely — no action is executed
    and the user is told rather than guessing.
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
            "Pocket Jarvis (Phase 3)\n\n"
            "/open <app>   - open an app directly\n"
            "/type <text>  - type into the currently focused field\n"
            "Anything else - handled via natural language (Gemini)\n"
        )

    # Phase 3: route anything non-literal through the reasoning layer.
    try:
        result = reasoning.parse_command(text)
    except reasoning.ReasoningError as e:
        log.warning("Reasoning failed for %r: %s", text, e)
        return "🤔 Sorry, I didn't understand that — try a literal /open or /type command."

    action, args = result["action"], result["args"]

    if action == "open_app":
        actions.open_app(args["package"])
        return f"✅ Opened: {args['package']}"

    if action == "type_text":
        actions.type_text(args["text"])
        return f"✅ Typed: {args['text']}"

    if action == "tap":
        actions.tap(args["x"], args["y"])
        return f"✅ Tapped: ({args['x']}, {args['y']})"

    # action == "none"
    return "🤔 Not sure what you meant — try being more specific."


def main():
    validate_config()
    log.info("Pocket Jarvis Phase 3 starting. Polling for messages...")

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
