"""
Pocket Jarvis - action execution layer

This module is the ONLY place that knows how actions actually get executed
on the device. Path B: it talks to LocalHttpServer.kt running inside the
Accessibility Service app on the phone itself (127.0.0.1:8765).

Every action is logged (see logs/audit.log) before it's sent, per the
audit-log principle from the security design.
"""

import os
import logging
import requests
from urllib.parse import quote

log = logging.getLogger("pocket-jarvis.actions")

# Backend URLs, loaded from .env — now pointing at the on-device
# LocalHttpServer (Path B), not MacroDroid webhooks.
OPEN_APP_URL = os.environ.get("POCKETJARVIS_OPEN_APP_URL")
TYPE_TEXT_URL = os.environ.get("POCKETJARVIS_TYPE_TEXT_URL")
TAP_URL = os.environ.get("POCKETJARVIS_TAP_URL")

AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "logs/audit.log")

# Friendly names -> real Android package names. Add more as you need them.
# Look up any app's package name via its Play Store URL, e.g.
# play.google.com/store/apps/details?id=<this is the package name>
APP_PACKAGES = {
    "playstore": "com.android.vending",
    "play store": "com.android.vending",
    "chrome": "com.android.chrome",
    "gmail": "com.google.android.gm",
    "whatsapp": "com.whatsapp",
    "youtube": "com.google.android.youtube",
    "settings": "com.android.settings",
}


def _audit(action_name, detail):
    """Append every executed action to a local audit log."""
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    import datetime
    line = f"{datetime.datetime.now().isoformat()} | {action_name} | {detail}\n"
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(line)


def _call(url_base, label):
    if not url_base:
        raise RuntimeError(
            f"No URL configured for '{label}'. "
            f"Set it in .env (see POCKETJARVIS_OPEN_APP_URL / "
            f"POCKETJARVIS_TYPE_TEXT_URL / POCKETJARVIS_TAP_URL)."
        )
    resp = requests.get(url_base, timeout=10)
    resp.raise_for_status()
    if resp.text.strip() != "OK":
        raise RuntimeError(f"Device returned unexpected response: {resp.text}")
    return resp


def open_app(app_hint: str):
    """
    Resolve app_hint (a friendly name or a raw package name) to a real
    Android package name, then tell the phone to open it.
    """
    package = APP_PACKAGES.get(app_hint.lower(), app_hint)

    log.info("Action: open_app(%s) -> package=%s", app_hint, package)
    _audit("open_app", f"{app_hint} -> {package}")

    url = f"{OPEN_APP_URL}{quote(package)}"
    _call(url, "open_app")


def type_text(text: str):
    """Tell the phone to type into whatever field currently has focus."""
    log.info("Action: type_text(%s)", text)
    _audit("type_text", text)

    url = f"{TYPE_TEXT_URL}{quote(text)}"
    _call(url, "type_text")


def tap(x: int, y: int):
    """
    Tell the phone to tap a specific screen coordinate. Idle since Phase 2
    (LocalHttpServer/JarvisAccessibilityService already support it) — this
    is the first caller, wired up for Phase 3's reasoning layer.
    """
    log.info("Action: tap(x=%s, y=%s)", x, y)
    _audit("tap", f"x={x}, y={y}")

    if not TAP_URL:
        raise RuntimeError(
            "No URL configured for 'tap'. Set POCKETJARVIS_TAP_URL in .env "
            "(e.g. http://127.0.0.1:8765/tap)."
        )
    url = f"{TAP_URL}?x={x}&y={y}"
    _call(url, "tap")
