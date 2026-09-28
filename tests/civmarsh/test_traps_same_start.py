"""Same-start trap rate: value-log parsing and exclusions, the pair rule, the
cross-start sample, and the cluster bootstrap."""

import json

import pytest

from civmarsh.traps.same_start import (
    GAP_MIN,
    load_value_log,
    pair_counts,
    per_turn_rates,
    same_start_bootstrap,
    same_start_traps,
)


def episode(start, end, scores, **flags):
    return {
        "position_id": f"ep_{start}",
        "score_end": end,
        "decisions": [
            {"turn": t, "rendering": f'DIGEST {{"metrics":{{"score":{s},"gold":1}}}}'}
            for t, s in scores.items()
        ],
        **flags,
    }


def write_log(path, eps):
    path.write_text("".join(json.dumps(e) + "\n" for e in eps))
    return path


def test_load_value_log_exclusions(tmp_path):
    log = write_log(
        tmp_path / "log.jsonl",
        [
            episode("s1", 50, {10: 5, 20: 9}),
            episode("s1", 60, {10: 4}, eliminated=True),
            episode("s1", 60, {10: 4}, control_lost=True),
            episode("s2", None, {10: 4}),
            {
                "position_id": "s3",
                "score_end": 10,
                "decisions": [{"turn": 10, "rendering": "no score here"}],
            },
        ],
    )
    eps, n_excluded, n_no_score = load_value_log(log)
    assert eps == [("s1", 50, {10: 5, 20: 9})]
    assert (n_excluded, n_no_score) == (3, 1)


def test_pair_counts_rule():
    items = [
        ("s", 10, 100.0),
        ("s", 5, 110.0),  # vs 0: trap (ahead now, behind at the end)
        ("s", 10, 104.0),  # vs 0: |d end| < GAP_MIN -> not decidable
        ("s", 10, 120.0),  # vs 0: tied now -> ge only
    ]
    assert GAP_MIN == 6.0
    n_dec, n_strict, n_trap_strict, n_trap_ge = pair_counts(
        items, [(0, 1), (0, 2), (0, 3)]
    )
    assert (n_dec, n_strict, n_trap_strict) == (2, 1, 1)
    # tie on the visible score, B finishes ahead: ">=" calls A the scoreboard
    # leader, so the tied pair also counts as a trap
    assert n_trap_ge == 2


def test_per_turn_rates_same_and_cross_start():
    eps = [
        ("s1", 100, {10: 10}),
        ("s1", 80, {10: 20}),  # same-start trap
        ("s2", 200, {10: 30}),
        ("s2", 150, {10: 5}),  # same-start non-trap
    ]
    out = per_turn_rates(eps)
    d = out[10]
    assert d["n_episodes"] == 4 and d["n_starts"] == 2
    assert d["same_start"]["n_pairs"] == 2 and d["same_start"]["n_strict"] == 2
    assert d["same_start"]["trap_strict"] == 0.5
    assert d["cross_start"]["n_pairs"] == 1000  # max(#same-start, 1000)


def test_bootstrap_interval_contains_rate():
    eps = []
    for s in range(20):
        eps.append((f"s{s}", 100, {30: 10}))
        eps.append((f"s{s}", 50, {30: 20 if s % 4 == 0 else 5}))
    b = same_start_bootstrap(eps, 30, 49)
    assert b["rate"] == 0.25 and b["n_strict"] == 20 and b["n_starts"] == 20
    assert b["ci95"][0] <= b["rate"] <= b["ci95"][1]
    assert same_start_bootstrap(eps, 50, 69) is None


def test_same_start_traps_pools_arm_start_clusters(tmp_path):
    a = write_log(
        tmp_path / "a.jsonl",
        [episode("s1", 100, {50: 10}), episode("s1", 80, {50: 20})],
    )
    b = write_log(
        tmp_path / "b.jsonl", [episode("s1", 100, {50: 30}), episode("s1", 50, {50: 5})]
    )
    out = same_start_traps({"arm_a": a, "arm_b": b})
    assert set(out["arms"]) == {"arm_a", "arm_b"}
    pooled = out["pooled_same_start_by_bucket"]["t50-69"]
    assert pooled["n_starts"] == 2 and pooled["rate"] == 0.5
    assert out["pooled_per_turn"]["50"]["same_start"]["n_pairs"] == 2
    assert out["arms"]["arm_a"]["buckets"]["t50-69"]["same_start_trap_strict"] == 1.0
    with pytest.raises(KeyError):
        out["arms"]["arm_a"]["per_turn"]["50"]["same_start_counts"]
