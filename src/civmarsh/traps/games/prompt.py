"""Zero-shot pairwise prompt for cross-game position pairs.

The Freeciv pairwise prompt with the game named instead of Freeciv; the
template text reaches the model verbatim and must not be edited.
"""

from __future__ import annotations

CROSS_GAME_PAIRWISE_PROMPT = """You will see two positions from two different games of {game}, \
snapshotted at similar points in the game. For each position, evaluate the named \
player's LONG-RUN prospects through to the end of the game — not just the current \
visible score.

=== Position A (evaluate player '{focal_a}') ===
{rendering_a}

=== Position B (evaluate player '{focal_b}') ===
{rendering_b}

Question: which evaluated player is better positioned for long-run success by the \
end of the game — player '{focal_a}' in Position A, or player '{focal_b}' in \
Position B?

Reply with exactly one line: "ANSWER: A" or "ANSWER: B"."""


def cross_game_prompt(game: str, rec_a: dict, rec_b: dict) -> str:
    """The prompt for bank records `rec_a` (shown as A) and `rec_b` (as B)."""
    return CROSS_GAME_PAIRWISE_PROMPT.format(
        game=game,
        focal_a=rec_a["focal_player"],
        rendering_a=rec_a["rendering"],
        focal_b=rec_b["focal_player"],
        rendering_b=rec_b["rendering"],
    )
