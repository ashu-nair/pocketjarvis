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

class ReasoningError(Exception):
    """Raised whenever Gemini's response can't be trusted to execute.
    Callers should catch this and fail safely (tell the user, don't act)."""


# --- Multi-step act-observe loop contract -----------------------------
#
# Unlike parse_command() above (one command -> one action, used for the
# literal-command fast path), decide_next_step() is called repeatedly by
# bot.py's agent loop: each call gets the original goal, what's actually
# on screen right now, and a log of what's been tried so far, and returns
# exactly ONE next action. The loop re-captures the screen and calls this
# again after every action, so Gemini never has to plan multiple steps
# blind — it only ever decides the next single step given real, current
# screen state.

SYSTEM_PROMPT_LOOP = """You are the reasoning layer for Pocket Jarvis, a \
personal Android automation agent controlled via Telegram by its owner. \
You control the phone ONE STEP AT A TIME in a loop: you'll be called \
again after every action with a fresh snapshot of the screen, so you \
never need to plan more than the next single step.

You will be given:
1. The owner's original goal (in their own words).
2. "screen": a JSON snapshot of everything currently visible on screen. \
Each entry has class, text, desc, resourceId, clickable, editable, and \
bounds (pixel rectangle).
3. "history": the actions already taken this turn and their results.

Respond with ONLY a JSON object (no markdown, no code fences, no extra \
text) describing exactly ONE next action.

Valid actions:
- {"action": "open_app", "args": {"package": "<friendly app name, e.g. \
whatsapp, chrome, gmail, youtube, playstore, settings, or a raw Android \
package name if you're confident>"}}
- {"action": "tap", "args": {"target": "<copied VERBATIM from a node's \
text, desc, or resourceId field in the current "screen" JSON>"}}
- {"action": "type_text", "args": {"text": "<text to type into whatever \
is currently focused>"}}
- {"action": "scroll", "args": {"direction": "up" or "down"}}
- {"action": "done", "args": {"message": "<short summary of what was \
accomplished, will be shown to the owner>"}} — use only once the goal is \
ACTUALLY achieved, based on what "screen" shows right now, not before
- {"action": "none", "args": {"reason": "<why you can't proceed, will be \
shown to the owner>"}} — use when stuck, ambiguous, or not confident

Any action (open_app, tap, type_text) can ALSO include two extra \
top-level fields when the action is consequential and hard to undo:
  "requires_confirmation": true
  "confirmation_reason": "<one short sentence explaining what this does \
and why it needs the owner's own judgment, e.g. \"This installs Instagram \
from the Play Store.\" or \"This sends the typed message to the contact.\">"
When you set requires_confirmation, the action will NOT run yet — the \
owner will be asked to approve it first over Telegram, then you'll be \
called again to continue from there. Use this instead of "none" for \
consequential actions (see rules below) — don't refuse the task outright \
just because a step needs approval.

Rules:
- Always return exactly one JSON object, nothing else.
- Never invent an action outside this list.
- For "tap": the "target" value MUST be copied verbatim from a node's \
text, desc, or resourceId field in the current "screen" JSON — never \
leave it empty and never invent one that isn't present. If the element \
you want (e.g. an icon-only button with no label) has no usable text, \
desc, or resourceId to copy, you cannot reliably target it — use \
"scroll" or "none" instead of guessing or leaving target blank. If what \
you need isn't visible yet at all, scroll first, then tap on the next \
turn once it's visible.
- "type_text" only works on a field that is CURRENTLY FOCUSED — check \
the current "screen" JSON for a node with "focused": true before typing. \
If the field you want to type into is not focused yet (e.g. right after \
opening an app, nothing is focused by default), you MUST "tap" that \
field first and wait for your next turn — never "type_text" in the same \
turn you intend to focus a field, since it won't be focused until after \
the tap actually happens.
- Set requires_confirmation: true on any action that agrees to, \
activates, pays for, sends, installs, deletes, or otherwise commits to \
something — e.g. tapping "Accept"/"Agree"/"Allow"/"Sign in"/"Subscribe"/\
"Buy"/"Pay"/"Install"/"Send"/"Delete"/"Uninstall"/"Confirm order", or \
typing then sending a message to a real contact. A plain dismiss/skip \
button (e.g. "Skip", "No thanks", "Not now", "Got it", "Continue", a \
close/X icon on a promo screen) does NOT need confirmation — it doesn't \
commit to anything, just proceed normally.
- Prefer "none" over guessing. A wrong tap or type executes for real on \
a real phone, right now — but for anything consequential, prefer \
requires_confirmation over "none" so the owner gets a real choice \
instead of the task just failing.
- Check "history" before repeating yourself: if the same action was just \
tried and didn't move things forward, don't just try it again — either \
try something different or use "none" and explain you're stuck.
- If a previous step in "history" shows result "failed: ...", read why it \
failed and adjust — e.g. a failed type_text usually means you need to \
tap the field first; a failed tap usually means the target isn't \
actually on screen and you may need to scroll.
- Keep "reason" (for "none") and "message" (for "done") to ONE short, \
plain sentence for the owner. Never include your own working-through-it \
narration — no "let's check", "wait", "hmm", or similar. Decide first, \
then state only the clean conclusion.
"""

_VALID_ACTIONS = {
    "open_app": {"package"},
    "type_text": {"text"},
    "tap": {"x", "y"},
    "none": set(),
}

_VALID_LOOP_ACTIONS = {
    "open_app": {"package"},
    "type_text": {"text"},
    "tap": {"target"},
    "scroll": {"direction"},
    "done": set(),
    "none": set(),
}


def decide_next_step(goal: str, screen_state: dict, history: list) -> dict:
    """
    One turn of the act-observe loop: given the goal, current screen, and
    what's been tried so far, ask Gemini for exactly one next action.
    Raises ReasoningError on any failure — callers should stop the loop
    and report to the user rather than guess.
    """
    user_content = json.dumps(
        {"goal": goal, "screen": screen_state, "history": history}
    )
    parsed = _call_gemini(SYSTEM_PROMPT_LOOP, user_content)
    return _validate_loop_step(parsed)


def _validate_loop_step(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ReasoningError(f"Gemini response was not a JSON object: {parsed!r}")

    action = parsed.get("action")
    args = parsed.get("args", {})

    if action not in _VALID_LOOP_ACTIONS:
        raise ReasoningError(f"Unknown action from Gemini: {action!r}")

    if not isinstance(args, dict):
        raise ReasoningError(f"'args' was not an object: {args!r}")

    required = _VALID_LOOP_ACTIONS[action]
    missing = required - set(args.keys())
    if missing:
        raise ReasoningError(f"Missing args {missing} for action {action!r}")

    if action == "scroll" and args["direction"] not in ("up", "down"):
        raise ReasoningError(f"Invalid scroll direction: {args['direction']!r}")

    if action == "tap" and not str(args["target"]).strip():
        raise ReasoningError("tap target was empty")

    requires_confirmation = bool(parsed.get("requires_confirmation", False))
    confirmation_reason = str(parsed.get("confirmation_reason", "")).strip()

    # "done" and "none" don't execute anything, so confirmation doesn't
    # apply to them — ignore the flag rather than erroring, in case a
    # model response sets it inconsistently.
    if action in ("done", "none"):
        requires_confirmation = False

    return {
        "action": action,
        "args": args,
        "requires_confirmation": requires_confirmation,
        "confirmation_reason": confirmation_reason,
    }


def _call_gemini(system_prompt: str, user_content: str) -> dict:
    """Shared plumbing: send a system prompt + user content to Gemini,
    parse the JSON it returns. Raises ReasoningError on any failure."""
    if not GEMINI_API_KEY:
        raise ReasoningError("GEMINI_API_KEY not set in .env")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
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
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ReasoningError(
            f"Gemini did not return valid JSON: {raw_text!r}"
        ) from e


def parse_command(user_text: str) -> dict:
    parsed = _call_gemini(SYSTEM_PROMPT, user_text)
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