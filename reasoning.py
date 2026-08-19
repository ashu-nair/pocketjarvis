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

--- Vision fallback (added for Known Issues #1) -----------------------
decide_next_step() (tree-based) is the default path for every turn of the
act-observe loop. When screen.find_target_bounds() can't resolve a tap
target — usually an unlabeled icon (Play Store's search bar) or a UI that
changed since it was last mapped (Skyscanner) — the caller falls back to
decide_next_step_vision(), which sends an actual screenshot instead of the
accessibility tree and asks Gemini to point at pixel coordinates directly.
Both paths share the same action schema and confirmation-gate behavior;
only the input Gemini sees (tree vs image) and the tap arg shape differ
along the way, and decide_next_step_vision() normalizes tap args back to
{"target": ...}-free real x,y before returning, so bot.py's execution code
doesn't need two different tap-handling branches.
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
Each entry has index, class, text, desc, resourceId, clickable, \
editable, focused, and bounds (pixel rectangle).
3. "history": the actions already taken this turn and their results.

Respond with ONLY a JSON object (no markdown, no code fences, no extra \
text) describing exactly ONE next action.

Valid actions:
- {"action": "open_app", "args": {"package": "<friendly app name, e.g. \
whatsapp, chrome, gmail, youtube, playstore, settings, or a raw Android \
package name if you're confident>"}}
- {"action": "tap", "args": {"target": "<EITHER copied VERBATIM from a \
node's text, desc, or resourceId field, OR "index:N" using that node's \
index number — use index:N when the element you want has no usable \
text, desc, or resourceId to quote (e.g. an icon-only button); you can \
still identify such a node by its class and bounds relative to other \
nodes even with nothing to read>"}}
- {"action": "type_text", "args": {"text": "<text to type into whatever \
is currently focused>"}}
- {"action": "scroll", "args": {"direction": "up", "down", "left", or \
"right"}} — "up"/"down" for vertical scrolling (lists, feeds, long \
pages). "left"/"right" for horizontal navigation (calendar/date-picker \
month arrows, image carousels, horizontally-scrolling tab bars). Use \
"right" to move to content further right/forward (e.g. the next month \
in a date picker), "left" to move back/previous.
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
- For "tap": "target" MUST be either copied verbatim from a node's text, \
desc, or resourceId, OR be "index:N" for that node's index — never leave \
it empty and never invent a text/desc/resourceId value that isn't \
present. When an element has no usable text, desc, or resourceId (an \
icon-only button), use its index instead, reasoning from its class and \
bounds relative to labeled neighbors — e.g. several equal-width, \
evenly-spaced unlabeled clickable regions together usually form a \
navigation bar (a row across the top, or a row across the bottom), and \
you can often tell which one is likely what you need from its position \
in that row plus any hint given in the goal itself. Don't assume a \
fixed layout — different apps put navigation and search in different \
places (a top bar in one app, a bottom tab in another). If what you \
need isn't visible on screen at all yet, scroll first, then tap on the \
next turn once it's visible.
- Be careful with short, non-unique text like a bare day number (e.g. \
"26" on a calendar/date picker). If a calendar or list could plausibly \
show more than one element with that same short text (e.g. day 26 of \
more than one visible month, or the same number in two different \
places), a plain text match on "26" is ambiguous and might tap the \
wrong one silently. When you can see multiple candidates like this, \
either use "index:N" for the SPECIFIC one you mean (never a generic \
text match), or confirm from context (visible month/year label, \
position in the grid) which one is actually correct before choosing.
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
- When using "index:N", copy the exact number from a node you can \
actually see in the current "screen" JSON — don't guess a nearby number \
if you're not sure (e.g. don't try index:29 then index:30 then index:31 \
hoping one lands). If you can't confidently identify which index is \
right, use "scroll" or "none" instead of guessing — a wrong tap is a \
real action on a real phone.
- If a previous step in "history" shows result "failed: ...", read why it \
failed and adjust — e.g. a failed type_text usually means you need to \
tap the field first; a failed tap usually means the target isn't \
actually on screen and you may need to scroll.
- Keep "reason" (for "none") and "message" (for "done") to ONE short, \
plain sentence for the owner. Never include your own working-through-it \
narration — no "let's check", "wait", "hmm", or similar. Decide first, \
then state only the clean conclusion.
"""

# --- Vision fallback contract -------------------------------------------
#
# Called instead of decide_next_step() when screen.find_target_bounds()
# returns None for a tap target — i.e. the tree-based reasoning named a
# target that couldn't be resolved to real coordinates, usually because
# the element has no usable text/desc/resourceId (unlabeled icon) or the
# app's UI changed since it was last seen. Sends an actual screenshot
# instead of the accessibility tree and asks Gemini to point at pixel
# coordinates directly, which sidesteps the tree-matching problem
# entirely rather than trying to prompt-engineer around it further (see
# Known Issues #1 — this was tried extensively and hit a real ceiling).

SYSTEM_PROMPT_VISION = """You are the reasoning layer for Pocket Jarvis, a \
personal Android automation agent controlled via Telegram by its owner. \
You are being called as a FALLBACK: the accessibility-tree-based reasoning \
couldn't find a reliable text/desc/resourceId match for the next tap \
(usually an unlabeled icon, or a UI that changed since the app was last \
mapped). You will be shown an actual screenshot instead.

You will be given:
1. The owner's original goal.
2. An image: the current screen.
3. "history": actions already taken this turn and their results.

Respond with ONLY a JSON object (no markdown, no extra text) describing \
exactly ONE next action.

Valid actions:
- {"action": "tap", "args": {"x": <int 0-1000>, "y": <int 0-1000>}} — \
coordinates NORMALIZED to a 0-1000 scale, where x=0 is the left edge of \
the image, x=1000 is the right edge, y=0 is the top edge, y=1000 is the \
bottom edge — regardless of the image's actual pixel dimensions
- {"action": "open_app", "args": {"package": "<friendly app name>"}}
- {"action": "type_text", "args": {"text": "<text to type>"}}
- {"action": "scroll", "args": {"direction": "up", "down", "left", or \
"right"}} — "up"/"down" for vertical scrolling, "left"/"right" for \
horizontal navigation (calendar month arrows, carousels)
- {"action": "done", "args": {"message": "<short summary for the owner>"}}
- {"action": "none", "args": {"reason": "<why you can't proceed>"}}

Any action can also include:
  "requires_confirmation": true
  "confirmation_reason": "<one short sentence>"
Use this for anything consequential (agrees/pays/sends/installs/deletes/\
confirms), same as the normal reasoning layer.

Rules:
- For "tap": pick the normalized (0-1000, 0-1000) position of the \
CENTER of the element you actually see in the image. Look carefully — \
this is being called specifically because the element has no label, so \
read it visually (icon shape, position, nearby text) rather than \
guessing.
- Before proposing a tap that installs, buys, or otherwise commits to a \
specific app or item named in the goal (e.g. "install BGMI"), you MUST \
be able to see that exact app's name or clearly identifiable icon/\
branding somewhere in the image first. Do NOT assume any visible \
"Install"/"Buy"/"Get" button satisfies the goal just because it exists \
— a screen can show unrelated apps, recommendations, or ads with their \
own buttons that have nothing to do with what the owner asked for. If \
you don't see the target app's name or icon anywhere in the image, do \
NOT tap an unrelated button — instead work toward finding it (e.g. tap \
a visible search icon/bar) or use "none" if you're not sure how.
- Sponsored or ad content (labeled "Sponsored"/"Ad", or presented as a \
promoted card) never satisfies the goal unless the owner's goal \
explicitly names that exact product or service.
- When you do propose a consequential tap, "confirmation_reason" must \
name the SPECIFIC app/item text you can actually read in the image, not \
just repeat the goal's wording back. If you can't read enough on-screen \
text to name it specifically, that itself is a sign you may be \
targeting the wrong element — reconsider before proposing the tap.
- If you can't confidently identify the element visually either, use \
"none" rather than guessing — a wrong tap is a real action on a real \
phone.
- Keep "reason"/"message" to one short plain sentence, no narration.
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

_VALID_VISION_ACTIONS = {
    "open_app": {"package"},
    "type_text": {"text"},
    "tap": {"x", "y"},
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


def decide_next_step_vision(goal: str, screenshot: dict, history: list) -> dict:
    """
    Vision fallback for one turn of the act-observe loop. `screenshot` is
    the dict returned by actions.get_screenshot() (image/mimeType/width/
    height/deviceWidth/deviceHeight). Tap coordinates are rescaled from
    the downscaled image's pixel space back to real device pixels before
    returning, so callers can pass the returned args straight into
    actions.tap() exactly like the tree-based path — no extra
    coordinate math needed at the call site.

    Raises ReasoningError on any failure, same contract as
    decide_next_step().
    """
    img_w = screenshot["width"]
    img_h = screenshot["height"]
    prompt = SYSTEM_PROMPT_VISION

    user_text = json.dumps({"goal": goal, "history": history})
    parsed = _call_gemini_vision(
        prompt, user_text, screenshot["image"], screenshot.get("mimeType", "image/jpeg")
    )
    step = _validate_vision_step(parsed)

    if step["action"] == "tap":
        # Gemini's vision output uses a normalized 0-1000 coordinate space
        # by convention (trained behavior for pointing/bbox tasks) rather
        # than following arbitrary pixel-range instructions in the prompt
        # — confirmed via testing, where returned coordinates like
        # {'x': 500, 'y': 900} were consistently out of the actual scaled
        # image's pixel bounds but made sense as 0-1000 normalized values.
        # Rescale directly against the real device dimensions; the scaled
        # screenshot's own width/height aren't needed for this step at
        # all now.
        device_w = screenshot["deviceWidth"]
        device_h = screenshot["deviceHeight"]
        step["args"]["x"] = int(step["args"]["x"] / 1000 * device_w)
        step["args"]["y"] = int(step["args"]["y"] / 1000 * device_h)

    return step


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

    if action == "scroll" and args["direction"] not in ("up", "down", "left", "right"):
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


def _validate_vision_step(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ReasoningError(f"Gemini vision response was not a JSON object: {parsed!r}")

    action = parsed.get("action")
    args = parsed.get("args", {})

    if action not in _VALID_VISION_ACTIONS:
        raise ReasoningError(f"Unknown vision action from Gemini: {action!r}")
    if not isinstance(args, dict):
        raise ReasoningError(f"'args' was not an object: {args!r}")

    required = _VALID_VISION_ACTIONS[action]
    missing = required - set(args.keys())
    if missing:
        raise ReasoningError(f"Missing args {missing} for vision action {action!r}")

    if action == "tap":
        try:
            x, y = int(args["x"]), int(args["y"])
        except (TypeError, ValueError) as e:
            raise ReasoningError(f"tap args not valid integers: {args}") from e
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ReasoningError(f"tap coordinates out of normalized 0-1000 bounds: {args}")
        args["x"], args["y"] = x, y

    if action == "scroll" and args["direction"] not in ("up", "down", "left", "right"):
        raise ReasoningError(f"Invalid scroll direction: {args['direction']!r}")

    requires_confirmation = bool(parsed.get("requires_confirmation", False))
    confirmation_reason = str(parsed.get("confirmation_reason", "")).strip()
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
    except json.JSONDecodeError:
        # Observed in testing: Gemini occasionally appends one stray extra
        # "}" despite response_mime_type=json requesting clean JSON. Try
        # once to repair that specific, narrow case before giving up —
        # this still requires the repaired text to parse as valid JSON,
        # so it can't accept genuine garbage, it just recovers a known
        # cheap mistake instead of wasting the whole turn on it.
        if raw_text.rstrip().endswith("}}"):
            try:
                return json.loads(raw_text.rstrip()[:-1])
            except json.JSONDecodeError:
                pass
        raise ReasoningError(f"Gemini did not return valid JSON: {raw_text!r}")


def _call_gemini_vision(system_prompt: str, user_text: str, image_b64: str, mime_type: str) -> dict:
    """Same contract as _call_gemini(), but attaches an inline image part
    to the request alongside the text content. Kept separate from
    _call_gemini() rather than adding an optional image param there, so
    the plain text-only path (used by parse_command and decide_next_step)
    can't accidentally regress if this path changes."""
    if not GEMINI_API_KEY:
        raise ReasoningError("GEMINI_API_KEY not set in .env")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ],
        }],
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
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ReasoningError(f"Gemini vision API request failed: {e}") from e

    try:
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        raise ReasoningError(f"Unexpected Gemini vision response shape: {e}") from e

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        if raw_text.rstrip().endswith("}}"):
            try:
                return json.loads(raw_text.rstrip()[:-1])
            except json.JSONDecodeError:
                pass
        raise ReasoningError(f"Gemini vision did not return valid JSON: {raw_text!r}")


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