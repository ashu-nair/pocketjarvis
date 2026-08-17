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
SCROLL_URL = os.environ.get("POCKETJARVIS_SCROLL_URL")
SCREEN_STATE_URL = os.environ.get("POCKETJARVIS_SCREEN_STATE_URL")
SCREENSHOT_URL = os.environ.get("POCKETJARVIS_SCREENSHOT_URL")

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
    "brave": "com.brave.browser",
    "flipkart": "com.flipkart.android",
    "termux": "com.termux",
    # "gallery" and "phone"/"dialer" are OEM-specific — stock Android uses
    # Google Photos / Google Dialer, but Xiaomi/Realme/Vivo-family skins
    # often ship their own with a different package name. Confirm via
    # `pm list packages | grep -i gallery` / `grep -i dialer` in Termux
    # before uncommenting/adjusting these:
    # "gallery": "com.google.android.apps.photos",
    # "phone": "com.google.android.dialer",
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
    Tell the phone to tap a specific screen coordinate. Used both by the
    tree-based reasoning path (coordinates resolved via screen.py) and the
    vision fallback path (coordinates returned directly by Gemini, already
    rescaled to real device pixels by reasoning.decide_next_step_vision).
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


def scroll(direction: str):
    """Tell the phone to scroll up or down via a synthetic swipe gesture."""
    log.info("Action: scroll(%s)", direction)
    _audit("scroll", direction)

    if not SCROLL_URL:
        raise RuntimeError(
            "No URL configured for 'scroll'. Set POCKETJARVIS_SCROLL_URL in "
            ".env (e.g. http://127.0.0.1:8765/scroll)."
        )
    url = f"{SCROLL_URL}?direction={direction}"
    _call(url, "scroll")


def get_screen_state() -> dict:
    """
    Fetch a snapshot of what's currently on screen: a flat list of nodes
    with text/desc/resourceId/bounds. Unlike the other actions, /screen
    always returns 200 with a JSON body (even an {"error": ...} one)
    rather than the OK/"Action failed" text pattern, so this doesn't go
    through _call() — it parses the JSON directly instead.

    Every node also gets an "index" attached here. Some genuinely have no
    text, desc, or resourceId at all (e.g. an icon-only button) — real
    example found in testing: Play Store's search bar is a wide,
    unlabeled clickable region, while the notification bell and account
    avatar right next to it both have full descriptions, so without an
    index Gemini could only ever "see" and target the two labeled icons,
    never the actual search bar. The index gives it something to point at
    even when there's nothing to read.
    """
    log.info("Action: get_screen_state()")

    if not SCREEN_STATE_URL:
        raise RuntimeError(
            "No URL configured for 'screen state'. Set "
            "POCKETJARVIS_SCREEN_STATE_URL in .env "
            "(e.g. http://127.0.0.1:8765/screen)."
        )
    resp = requests.get(SCREEN_STATE_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Device reported: {data['error']}")

    data["nodes"] = _dedupe_by_bounds(data.get("nodes", []))
    for i, node in enumerate(data["nodes"]):
        node["index"] = i
    return data


def get_screenshot() -> dict:
    """
    Fetch a downscaled screenshot for the vision fallback path (see
    Known Issues #1 — unlabeled/changed UI the accessibility tree can't
    describe well). Like /screen, /screenshot always returns 200 with a
    JSON body (even an {"error": ...} one), so this parses the JSON
    directly rather than going through _call().

    Returns the raw dict from the device:
        image        - base64-encoded JPEG
        mimeType     - "image/jpeg"
        width/height - dimensions of the *scaled* image actually sent to
                        Gemini (max 1024px long edge)
        deviceWidth/deviceHeight - real, unscaled screen pixels

    reasoning.decide_next_step_vision() uses deviceWidth/deviceHeight to
    rescale any coordinates Gemini returns back to real device pixels
    before they're passed to actions.tap() — callers of this function
    don't need to do that math themselves.
    """
    log.info("Action: get_screenshot()")

    if not SCREENSHOT_URL:
        raise RuntimeError(
            "No URL configured for 'screenshot'. Set "
            "POCKETJARVIS_SCREENSHOT_URL in .env "
            "(e.g. http://127.0.0.1:8765/screenshot)."
        )
    resp = requests.get(SCREENSHOT_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Device reported: {data['error']}")
    return data


def _dedupe_by_bounds(nodes: list) -> list:
    """
    Collapse nodes that share the exact same bounds down to one. Testing
    showed real duplication of this kind — e.g. Play Store's "Install"
    label appearing as two separate nodes at identical coordinates, one
    via its "desc" and one via its "text" — which pads out the node count
    a model has to precisely scan without adding any real information.
    Conservative by design: only merges EXACT bounds matches, so nested
    elements with different (even overlapping) bounds are all preserved.
    When duplicates exist, keeps whichever has the most identifying info
    (a non-blank desc, then text, then resourceId), so nothing useful is
    lost in the merge.
    """
    best_by_bounds = {}
    order = []

    def _richness(node):
        return (
            bool(node.get("desc", "").strip()),
            bool(node.get("text", "").strip()),
            bool(node.get("resourceId", "").strip()),
            bool(node.get("clickable")),
        )

    for node in nodes:
        bounds = node.get("bounds") or {}
        key = (bounds.get("left"), bounds.get("top"), bounds.get("right"), bounds.get("bottom"))
        if key not in best_by_bounds:
            best_by_bounds[key] = node
            order.append(key)
        elif _richness(node) > _richness(best_by_bounds[key]):
            best_by_bounds[key] = node

    return [best_by_bounds[key] for key in order]