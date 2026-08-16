"""
screen.py — turns a screen-state snapshot (from actions.get_screen_state())
into something usable for tapping. Kept separate from reasoning.py (which
only knows how to talk to Gemini) and actions.py (which only knows how to
send device commands) — this module's only job is interpreting screen JSON.
"""


def find_target_bounds(screen_state: dict, target: str):
    """
    Find the node whose text, desc, or resourceId matches `target` and
    return the pixel center of its bounds as (x, y), or None if nothing
    matches.

    Tries an exact (case-insensitive) match first, then falls back to a
    substring match. Gemini is instructed (see reasoning.py's system
    prompt) to copy `target` verbatim from a node it was actually shown,
    so an exact match should be the common case — the substring fallback
    just adds tolerance for minor whitespace/casing drift. resourceId is
    checked alongside text/desc because some icon-only buttons (e.g. an
    unlabeled search icon) have neither text nor a content description,
    and resourceId is the only thing Gemini can actually name them by.
    """
    nodes = screen_state.get("nodes", [])
    target_lower = target.strip().lower()
    if not target_lower:
        return None

    def _fields(node):
        return (
            node.get("text", ""),
            node.get("desc", ""),
            node.get("resourceId", ""),
        )

    for node in nodes:
        if any(f.strip().lower() == target_lower for f in _fields(node)):
            return _center(node.get("bounds"))

    for node in nodes:
        if any(target_lower in f.lower() for f in _fields(node)):
            return _center(node.get("bounds"))

    return None


def _center(bounds):
    if not bounds:
        return None
    x = (bounds["left"] + bounds["right"]) // 2
    y = (bounds["top"] + bounds["bottom"]) // 2
    return x, y