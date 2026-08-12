"""
reasoning.py — Phase 3 natural-language reasoning layer for Pocket Jarvis.

Takes raw natural-language text (from Telegram) and asks the Gemini API to
translate it into exactly one structured action call. Fails safely
(raises ReasoningError) rather than ever guessing at an action — a wrong
guess here executes for real on the phone.

Design contract:
    parse_command(text) -> {"action": "open_app"|"type_text"|"tap"|"none",
                             "args": {...}}
    Raises ReasoningError on any failure: missing key, network error,
    malformed JSON, unknown action, missing/invalid args.

Kept as a plain requests-based REST call (no google-generativeai SDK) to
match the project's existing "no extra dependency" style.
"""

import json
import os

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# "-latest" alias tracks whatever model Google currently designates as the
# free-tier Flash-Lite model, so this doesn't need updating every time
# Google renames/retires a specific version. Override via .env if needed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# open_app's "package" arg is deliberately a friendly name (whatsapp, chrome,
# gmail, ...) — actions.py's existing APP_PACKAGES dict already resolves
# those, so this layer doesn't need to know real Android package names.
SYSTEM_PROMPT = """You are the reasoning layer for Pocket Jarvis, a personal \
Android automation agent controlled via Telegram by its owner.

Given a natural-language instruction from the owner, respond with ONLY a \
JSON object (no markdown, no code fences, no explanation, no extra text) \
describing exactly one action to take.

Valid actions:
- {"action": "open_app", "args": {"package": "<friendly app name, e.g. \
whatsapp, chrome, gmail, youtube, playstore, settings, or a raw Android \
package name if you're confident>"}}
- {"action": "type_text", "args": {"text": "<literal text to type into \
the currently focused field>"}}
- {"action": "tap", "args": {"x": <int>, "y": <int>}} — only if the \
instruction gives explicit screen coordinates or clearly implies a \
specific point; otherwise prefer open_app/type_text
- {"action": "none", "args": {}} — use this if the instruction doesn't map \
to any available action, is ambiguous, or you are not confident

Rules:
- Always return exactly one JSON object, nothing else.
- Never invent an action outside this list.
- If the instruction is a compound request (e.g. "open whatsapp and text \
mom"), handle only the FIRST actionable step — compound handling is not \
supported yet.
- Prefer "none" over guessing wrong. A wrong guess executes a real action \
on a real phone.
"""

_VALID_ACTIONS = {
    "open_app": {"package"},
    "type_text": {"text"},
    "tap": {"x", "y"},
    "none": set(),
}


class ReasoningError(Exception):
    """Raised whenever Gemini's response can't be trusted to execute.
    Callers should catch this and fail safely (tell the user, don't act)."""


def parse_command(user_text: str) -> dict:
    if not GEMINI_API_KEY:
        raise ReasoningError("GEMINI_API_KEY not set in .env")

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ReasoningError(f"Gemini API request failed: {e}") from e

    try:
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        raise ReasoningError(f"Unexpected Gemini response shape: {e}") from e

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ReasoningError(
            f"Gemini did not return valid JSON: {raw_text!r}"
        ) from e

    return _validate(parsed)


def _validate(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ReasoningError(f"Gemini response was not a JSON object: {parsed!r}")

    action = parsed.get("action")
    args = parsed.get("args", {})

    if action not in _VALID_ACTIONS:
        raise ReasoningError(f"Unknown action from Gemini: {action!r}")

    if not isinstance(args, dict):
        raise ReasoningError(f"'args' was not an object: {args!r}")

    required = _VALID_ACTIONS[action]
    missing = required - set(args.keys())
    if missing:
        raise ReasoningError(f"Missing args {missing} for action {action!r}")

    if action == "tap":
        try:
            args["x"] = int(args["x"])
            args["y"] = int(args["y"])
        except (TypeError, ValueError) as e:
            raise ReasoningError(f"tap args not valid integers: {args}") from e

    return {"action": action, "args": args}
