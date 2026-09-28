"""Game-length curve: how well the scoreboard and the replay oracle predict the
real game's terminal score, for games of different lengths.

Input is a labeled bank of `civmarsh.oracle.bank` (all positions of short games,
oracle branches to the game's end turn), whose records carry the real game's
final focal score (`game_end_score`). Per sampled turn, across games, the
Pearson correlation of the terminal score with the current score and with the
oracle value; the oracle wins a turn when its correlation exceeds the
scoreboard's by more than `TURN_WIN_MARGIN`.

The trap rate here is taken over every decidable pair, visible-score ties
included and resolved with ">=", and counts only pairs with a positive pooled
standard error.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from civmarsh.oracle.pairs import decidable_pairs

TURN_WIN_MARGIN = 0.05
MAX_TURN_GAP = 10


def pearson(xs: list[float], ys: list[float]) -> float:
    """Population Pearson correlation; NaN for fewer than three points or a
    constant series."""
    if len(xs) < 3 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def turn_correlations(
    labels: list[dict], *, margin: float = TURN_WIN_MARGIN
) -> list[dict]:
    """One row per sampled turn: n games, correlation of the real terminal score
    with the scoreboard and with the oracle value (None when undefined), and
    which predictor wins."""
    by_turn = defaultdict(list)
    for r in labels:
        by_turn[r["turn"]].append(r)
    rows = []
    for turn in sorted(by_turn):
        recs = sorted(by_turn[turn], key=lambda r: r["game_id"])
        actual = [r["game_end_score"] for r in recs]
        cs = pearson([r["score_now"] for r in recs], actual)
        cv = pearson([r["mean_end"] for r in recs], actual)
        if cv > cs + margin:
            winner = "oracle"
        elif cs > cv + margin:
            winner = "scoreboard"
        else:
            winner = "tie"
        rows.append(
            {
                "turn": turn,
                "horizon": recs[0]["until"] - turn,
                "n": len(recs),
                "corr_scoreboard": None if math.isnan(cs) else cs,
                "corr_oracle": None if math.isnan(cv) else cv,
                "winner": winner,
            }
        )
    return rows


def _defined_mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def game_length_summary(
    labels: list[dict],
    *,
    margin: float = TURN_WIN_MARGIN,
    max_turn_gap: int = MAX_TURN_GAP,
) -> dict:
    """Mean correlations over turns (turns with an undefined correlation are
    left out), turns won by the oracle, and the trap rate."""
    per_turn = turn_correlations(labels, margin=margin)
    pairs = decidable_pairs(
        labels, max_turn_gap=max_turn_gap, drop_ties=False, skip_zero_se=True
    )
    trap = sum(p["is_trap"] for p in pairs)
    return {
        "games": len({r["game_id"] for r in labels}),
        "positions": len(labels),
        "corr_scoreboard": _defined_mean([t["corr_scoreboard"] for t in per_turn]),
        "corr_oracle": _defined_mean([t["corr_oracle"] for t in per_turn]),
        "turns_won_by_oracle": sum(t["winner"] == "oracle" for t in per_turn),
        "n_turns": len(per_turn),
        "decidable": len(pairs),
        "trap": trap,
        "trap_rate": round(trap / len(pairs), 4) if pairs else None,
        "per_turn": per_turn,
    }
