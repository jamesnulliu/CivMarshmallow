"""Trap rates: progress binning (edges, clamping, blank sparse bins) and the
Freeciv / cross-game summaries built on the decidable-pair rule."""

from civmarsh.traps.rates import (
    bin_edges,
    freeciv_trap_rates,
    game_trap_rates,
    pair_items,
    progress_bins,
    untied_summary,
)


def test_bin_edges():
    assert bin_edges() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_progress_bins_blank_below_min_pairs():
    items = [(12.0, True)] * 30 + [(36.0, False)] * 29 + [(200.0, True)] * 31
    out = progress_bins(items, 120.0)
    bins = out["bins"]
    assert bins[1]["n"] == 30 and bins[1]["trap_rate"] == 1.0  # 0.1 <= p < 0.2
    assert bins[3]["n"] == 29 and bins[3]["trap_rate"] is None  # p = 0.3 exactly
    assert bins[9]["n"] == 31 and bins[9]["trap_rate"] == 1.0  # clamped to < 1
    assert out["n_untied"] == 90
    assert out["overall"] == round(61 / 90, 4)
    assert out["progress_range"] == [0.1, round(200 / 120, 3)]


def test_progress_bins_min_pairs_argument_and_empty():
    out = progress_bins([(5.0, True)], 100.0, min_pairs=1)
    assert out["bins"][0]["trap_rate"] == 1.0
    empty = progress_bins([], 100.0)
    assert empty["overall"] is None and empty["progress_range"] is None


def pair(turn_a, turn_b, trap, tied):
    return {"turn_a": turn_a, "turn_b": turn_b, "is_trap": trap, "score_tied": tied}


def test_untied_summary_and_items():
    pairs = [
        pair(10, 20, True, False),
        pair(10, 10, True, True),
        pair(30, 40, False, False),
    ]
    s = untied_summary(pairs)
    assert (s["decidable"], s["untied"], s["trap"], s["trap_rate"]) == (3, 2, 1, 0.5)
    assert pair_items(pairs) == [(15.0, True), (35.0, False)]


def label(pid, game, turn, now, value, se=1.0):
    return {
        "position_id": pid,
        "game_id": game,
        "turn": turn,
        "score_now": now,
        "mean_end": value,
        "se_end": se,
        "until": 120,
    }


def test_freeciv_trap_rates_drops_ties_from_the_rate():
    labels = [
        label("a", "g1", 60, 10, 200),
        label("b", "g2", 60, 20, 100),  # a-b: trap
        label("c", "g3", 60, 20, 300),  # b-c tied now: counted decidable only
        label("d", "g4", 100, 0, 0),  # 40 turns away from the rest
    ]
    out = freeciv_trap_rates(labels, length=120.0, min_pairs=1)
    assert (out["decidable"], out["untied"], out["trap"]) == (3, 2, 1)
    assert out["trap_rate"] == 0.5
    assert out["bins"][5]["n"] == 2 and out["bins"][5]["trap_rate"] == 0.5
    assert out["games"] == 4 and out["max_turn_gap"] == 10


def test_game_trap_rates_uses_game_gap_and_progress_bank():
    def g(pid, game, turn, now, value):
        return {
            "position_id": pid,
            "game_id": game,
            "turn": turn,
            "score_now": now,
            "oracle_mean": value,
            "oracle_se": 0.0,
            "oracle_n": 20,
        }

    bank = [
        g("a", "c1", 24, 1, 0.9),
        g("b", "c2", 32, 2, 0.1),
        g("c", "c3", 60, 0, 0.5),
    ]
    extra = [g("x", "c4", 60, 3, 0.0)]
    out = game_trap_rates("chess", bank, length=100.0, progress_bank=extra, min_pairs=1)
    assert out["max_turn_gap"] == 8 and out["replays"] == 20
    assert (out["decidable"], out["untied"], out["trap"]) == (1, 1, 1)
    assert out["n_untied_bins"] == 2  # a-b plus c-x
    assert sum(b["n"] for b in out["bins"]) == 2
