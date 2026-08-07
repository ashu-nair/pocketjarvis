"""
Pocket Jarvis - Phase 2
Telegram long-polling bot that can now trigger real device actions via
actions.py (MacroDroid webhooks in Path A). Commands are still literal
for now, e.g. "/open playstore" or "/type hello world" - natural language
parsing arrives in Phase 3 with PocketClaw.

Every command is confirmed back to you after execution, and every action
is written to the audit log by actions.py before it runs.
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

import actions

load_dotenv()

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
    Phase 2: literal command parsing only.
    Phase 3 replaces this whole function with PocketClaw reasoning -
    natural language in, action calls out. Keeping it this dumb for now
    makes it easy to verify the control layer works before adding an LLM
    on top of it.
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
            "Pocket Jarvis (Phase 2)\n\n"
            "/open <app>  - open an app (bound in MacroDroid)\n"
            "/type <text> - type into the currently focused field\n"
        )

    # Fallback: still echo, same as Phase 1, for anything unrecognized.
    return f"Pocket Jarvis received: {text}\n(unrecognized command - try /help)"


def main():
    validate_config()
    log.info("Pocket Jarvis Phase 1 starting. Polling for messages...")

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
