"""Small Freeciv helpers shared across civmarsh: the focal player of a position
and the visible score read back from a rendered observation."""

from __future__ import annotations

import re

# In the rendered value input the first `"score":N,` field is the focal
# player's own visible score, taken from the digest metrics.
SCORE_RE = re.compile(r'"score":(\d+),')


def focal_player(state) -> str:
    """Name of the focal player of a position: the player with player_id 0.

    `state` is a `civharness.position(...)` result (anything with a
    ``players`` mapping of name -> {"player_id": int, ...})."""
    for name, p in state.players.items():
        if p["player_id"] == 0:
            return name
    raise ValueError("no player_id 0 in position")


def score_from_rendering(text: str) -> int | None:
    """The focal player's visible score in a rendered observation, or None
    when the rendering carries no score field."""
    m = SCORE_RE.search(text)
    return int(m.group(1)) if m else None
