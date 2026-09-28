"""Game-length curve: per-turn correlations with the real terminal score, turns
won by the oracle, and the trap rate with ties kept and zero-SE pairs skipped."""

import math

import pytest

from civmarsh.traps.game_length import game_length_summary, pearson, turn_correlations

ENDS = {"g1": 10, "g2": 20, "g3": 30, "g4": 40}


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(pearson([1, 2], [1, 2]))
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))


def lab(game, turn, now, value, end, se=1.0):
    return {
        "position_id": f"{game}_T{turn}",
        "game_id": game,
        "turn": turn,
        "score_now": now,
        "mean_end": value,
        "se_end": se,
        "until": 60,
        "game_end_score": end,
    }


def labels():
    rows = []
    # turn 10: the scoreboard is anti-correlated with the end, the oracle exact
    for g, now in zip(ENDS, [40, 30, 20, 10]):
        rows.append(lab(g, 10, now, ENDS[g], ENDS[g]))
    # turn 30 (no pairs with turn 10): both exact
    for g, end in ENDS.items():
        rows.append(lab(g, 30, end / 2, end, end))
    return rows


def test_turn_correlations_and_winner():
    rows = turn_correlations(labels())
    assert [r["turn"] for r in rows] == [10, 30]
    assert rows[0]["winner"] == "oracle" and rows[0]["horizon"] == 50
    assert rows[0]["corr_scoreboard"] == pytest.approx(-1.0)
    assert rows[1]["winner"] == "tie"


def test_summary_counts_ties_and_skips_zero_se():
    rows = labels()
    # a zero-SE pair at turn 30, ahead of every other position on both counts
    rows += [
        lab("g5", 30, 100, 100.0, 100, se=0.0),
        lab("g6", 30, 101, 101.0, 101, se=0.0),
    ]
    # a tied visible score at turn 50: kept, ">=" makes it a trap
    rows += [lab("g7", 50, 50, 60.0, 60), lab("g8", 50, 50, 70.0, 70)]
    s = game_length_summary(rows)
    assert s["turns_won_by_oracle"] == 1 and s["n_turns"] == 3
    # turn 10: 6 pairs, all traps; turn 30: 6 + 2 x 4 pairs, none a trap
    # (g5-g6 skipped: pooled SE 0); turn 50: 1 tied pair, a trap
    assert s["decidable"] == 6 + 14 + 1
    assert s["trap"] == 7
    assert s["trap_rate"] == round(7 / 21, 4)
