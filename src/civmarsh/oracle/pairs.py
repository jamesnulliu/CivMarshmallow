"""Decidable position pairs: the cross-game pair rule behind every trap rate.

Two labeled positions form a pair when they come from different games and
their turns differ by at most `max_turn_gap`. A pair is decidable when the
replay oracle separates them clearly: the gap between the two oracle values
exceeds twice their pooled standard error, hypot(se_a, se_b). The oracle
winner is the position with the higher oracle value; the pair is a trap when
the current (visible) score orders the two positions the other way.

Ties on the visible score are an explicit policy. By default they are dropped
(the scoreboard cannot pick a side). With ``drop_ties=False`` they are kept
and both orders are resolved with ">=": position A wins whenever its value is
greater than or equal to B's, on the oracle and on the scoreboard alike.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import combinations


def decidable_pairs(
    records: Iterable[dict],
    *,
    max_turn_gap: int,
    value_key: str = "mean_end",
    se_key: str = "se_end",
    now_key: str = "score_now",
    drop_ties: bool = True,
    skip_zero_se: bool = False,
    id_key: str = "position_id",
    game_key: str = "game_id",
    turn_key: str = "turn",
) -> list[dict]:
    """All decidable cross-game pairs of `records`, in `combinations` order.

    records       labeled positions carrying an id, a game id, a turn, the
                  oracle value (`value_key`), its standard error (`se_key`)
                  and the visible score now (`now_key`).
    max_turn_gap  largest allowed |turn_a - turn_b|.
    drop_ties     drop pairs whose visible scores are equal; when False they
                  are kept with ``score_tied=True`` and ">=" resolves them.
    skip_zero_se  treat a pair with zero pooled SE as undecidable. By default
                  such a pair is decidable whenever the two oracle values
                  differ (the strict ``gap > 2 * se`` rule).

    Each pair: a, b, game_a, game_b, turn_a, turn_b, score_a, score_b,
    winner ("A" or "B", the oracle winner), margin (oracle gap), pooled_se,
    margin_z, is_trap, score_tied.
    """
    records = list(records)
    pairs = []
    for a, b in combinations(records, 2):
        if a[game_key] == b[game_key]:
            continue
        if abs(a[turn_key] - b[turn_key]) > max_turn_gap:
            continue
        va, vb = a[value_key], b[value_key]
        gap = abs(va - vb)
        se = math.hypot(a[se_key], b[se_key])
        if skip_zero_se and not se:
            continue
        if not gap > 2 * se:
            continue
        na, nb = a[now_key], b[now_key]
        tied = na == nb
        if tied and drop_ties:
            continue
        winner = "A" if va >= vb else "B"
        scoreboard = "A" if na >= nb else "B"
        pairs.append(
            {
                "a": a[id_key],
                "b": b[id_key],
                "game_a": a[game_key],
                "game_b": b[game_key],
                "turn_a": a[turn_key],
                "turn_b": b[turn_key],
                "score_a": na,
                "score_b": nb,
                "winner": winner,
                "margin": round(gap, 4),
                "pooled_se": round(se, 4),
                "margin_z": round(gap / se, 2) if se else None,
                "is_trap": winner != scoreboard,
                "score_tied": tied,
            }
        )
    return pairs


def trap_summary(pairs: list[dict]) -> dict:
    """Counts and trap rate of a pair list (as built, ties included or not):
    decidable, trap, trap_rate, scoreboard_accuracy."""
    n = len(pairs)
    trap = sum(p["is_trap"] for p in pairs)
    return {
        "decidable": n,
        "trap": trap,
        "trap_rate": round(trap / n, 4) if n else None,
        "scoreboard_accuracy": round(1 - trap / n, 4) if n else None,
    }
