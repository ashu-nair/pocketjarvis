"""
screen.py — turns a screen-state snapshot (from actions.get_screen_state())
into something usable for tapping. Kept separate from reasoning.py (which
only knows how to talk to Gemini) and actions.py (which only knows how to
send device commands) — this module's only job is interpreting screen JSON.
"""


def find_target_bounds(screen_state: dict, target: str):
    """
    Find the node whose text or desc matches `target` and return the pixel
    center of its bounds as (x, y), or None if nothing matches.

    Tries an exact (case-insensitive) match first, then falls back to a
    substring match. Gemini is instructed (see reasoning.py's system
    prompt) to copy `target` verbatim from a node it was actually shown,
    so an exact match should be the common case — the substring fallback
    just adds tolerance for minor whitespace/casing drift.
    """
    nodes = screen_state.get("nodes", [])
    target_lower = target.strip().lower()
    if not target_lower:
        return None

    for node in nodes:
        if (
            node.get("text", "").strip().lower() == target_lower
            or node.get("desc", "").strip().lower() == target_lower
        ):
            return _center(node.get("bounds"))

    for node in nodes:
        if (
            target_lower in node.get("text", "").lower()
            or target_lower in node.get("desc", "").lower()
        ):
            return _center(node.get("bounds"))

    return None


def _center(bounds):
    if not bounds:
        return None
    x = (bounds["left"] + bounds["right"]) // 2
    y = (bounds["top"] + bounds["bottom"]) // 2
    return x, y