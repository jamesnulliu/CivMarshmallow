"""Tests for civmarsh.eval.diagnostics on synthetic value and reward logs."""

import json
import math

import pytest

from civmarsh.eval import diagnostics as dg


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _render(score):
    digest = {
        "turn": 5,
        "metrics": {"score": score, "cities": 2},
        "other": {"score": 999},
    }
    return 'SPATIAL {"score":1,}\nDIGEST ' + json.dumps(digest) + "\nMAP ..."


def test_score_now_from_rendering_reads_digest_metrics():
    assert dg.score_now_from_rendering(_render(42)) == 42.0
    assert dg.score_now_from_rendering("no digest here") is None
    assert dg.score_now_from_rendering("DIGEST {broken") is None


def test_turn_spectrum_buckets_errors_and_spearman(tmp_path):
    rows = [
        {"turns": [12, 35, 36], "values": [10.0, 50.0, 40.0], "score_end": 20.0},
        {"turns": [15, 40], "values": [30.0, 60.0], "score_end": 60.0},
        {"turns": [20, 45], "values": [None, 70.0], "score_end": 80.0},
        {"turns": [25], "values": [5.0], "score_end": None},
    ]
    parsed = dg.read_value_log(_jsonl(tmp_path / "value_log.jsonl", rows))
    assert len(parsed) == 6
    spec = dg.turn_spectrum(parsed)
    early = spec["t10-29"]
    assert early["n"] == 2
    # absolute errors 10 and 30: median 20, p90 by linear interpolation 28
    assert early["median_abs_err"] == pytest.approx(20.0)
    assert early["p90_abs_err"] == pytest.approx(28.0)
    assert early["spearman"] is None  # fewer than three points
    mid = spec["t30-49"]
    assert mid["n"] == 4
    # values 50, 40, 60, 70 against score end 20, 20, 60, 80
    assert mid["spearman"] == pytest.approx(4.5 / math.sqrt(22.5))
    assert spec["t110-129"] == {
        "n": 0,
        "median_abs_err": None,
        "p90_abs_err": None,
        "spearman": None,
    }


def test_delta_r2_matches_least_squares(tmp_path):
    turns = list(range(10, 20))
    value_rows, harvest_rows = [], []
    for ep in range(12):
        end = 50.0 + 7 * ep + (ep % 3) * 5
        values = [0.5 * end + 3 * math.sin(ep + t) for t in turns]
        value_rows.append({"turns": turns, "values": values, "score_end": end})
        # the harvest keeps every other decision
        harvest_rows.append(
            {
                "score_end": end,
                "decisions": [
                    {"turn": t, "rendering": _render(0.3 * end + (t * ep) % 7)}
                    for t in turns[::2]
                ],
            }
        )
    triplets, counts = dg.value_triplets(
        _jsonl(tmp_path / "value_log.jsonl", value_rows),
        _jsonl(tmp_path / "value_harvest.jsonl", harvest_rows),
    )
    assert counts["matched"] == len(triplets) == 12 * 5
    out = dg.delta_r2(triplets)

    import numpy as np

    phi, now, end = (np.array(col) for col in zip(*triplets))

    def r2(*cols):
        x = np.column_stack([np.ones(len(end)), *cols])
        coef, *_ = np.linalg.lstsq(x, end, rcond=None)
        resid = end - x @ coef
        return 1 - resid.var() / end.var()

    assert out["r2_score_now"] == pytest.approx(r2(now))
    assert out["r2_score_now_and_potential"] == pytest.approx(r2(now, phi))
    assert out["delta_r2"] == pytest.approx(r2(now, phi) - r2(now))


def _group(ts, pid, scores, ok=True):
    return {
        "ts": ts,
        "episodes": [
            {"position_id": pid, "score_end": s, "ok": ok and s is not None}
            for s in scores
        ],
    }


def test_consumed_groups_take_all_ok_groups_in_time_order():
    rows = [
        _group(3.0, "b", [2, 2]),
        _group(1.0, "a", [1, 3]),
        _group(2.0, "a", [5, None]),  # a stalled episode: group not consumed
        _group(4.0, "a", [7, 9]),
        _group(5.0, "c", [0, 0]),
    ]
    groups = dg.consumed_groups(rows, n_iterations=2, groups_per_iteration=1)
    assert groups == [(0, "a", 2.0), (1, "b", 2.0)]


def test_training_slope_per_start_ols_then_mean(tmp_path):
    rows = []
    ts = 0.0
    # 4 iterations x 2 groups; start "a" rises 2/iteration, "b" falls 1/iteration,
    # "c" appears in one iteration only and is not fit
    plan = [("a", "b"), ("a", "c"), ("a", "b"), ("a", "b")]
    for it, pids in enumerate(plan):
        for pid in pids:
            ts += 1
            base = {"a": 10 + 2 * it, "b": 30 - it, "c": 5}[pid]
            rows.append(_group(ts, pid, [base - 1, base + 1]))
    log = _jsonl(tmp_path / "reward_log.jsonl", rows)
    out = dg.training_slope(log, n_iterations=4, groups_per_iteration=2)
    assert out["per_start"] == pytest.approx({"a": 2.0, "b": -1.0})
    assert out["n"] == 2 and out["mean"] == pytest.approx(0.5)
    assert out["se"] == pytest.approx(1.5)
    diff = dg.paired_slope_difference(out["per_start"], {"a": 1.0, "b": -2.0, "z": 0})
    assert diff["n"] == 2 and diff["mean"] == pytest.approx(1.0)
    assert diff["se"] == 0.0


def test_per_start_slopes_needs_span_of_two():
    groups = [(0, "a", 1.0), (1, "a", 2.0), (0, "b", 1.0), (2, "b", 5.0)]
    assert dg.per_start_slopes(groups) == {"b": 2.0}
