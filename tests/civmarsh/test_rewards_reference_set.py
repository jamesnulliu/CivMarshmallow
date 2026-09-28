"""Reference-set refresh (served model mocked).

Pins the harvest-window split, the rebuild recipe (N_POLICY policy + N_AI AI
positions per bucket), validation with the runtime value configuration, atomic
promotion picked up through the mtime-keyed cache, the non-fatal gate (a
failing candidate is never promoted, the run continues), and the cadence.
"""

import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import civmarsh.rewards.reference_set as rs
from civmarsh.civtelescope import client
from civmarsh.rewards import potential

BUCKETS = list(range(10, 70, 5))


def _episode(score, rng, buckets=BUCKETS, last_turn=69):
    decs = [
        {"turn": t, "focal": "p", "rendering": f"r{score}t{t}x{rng.random():.3f}"}
        for t in buckets
    ]
    decs.append(
        {
            "turn": last_turn,
            "focal": "p",
            "rendering": f"r{score}last{rng.random():.3f}",
        }
    )
    return {
        "score_end": score,
        "end_turn": last_turn + 1,
        "eliminated": False,
        "decisions": decs,
    }


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def _last_line(path):
    return json.loads(Path(path).read_text().strip().split("\n")[-1])


def _write_ai_pool(path, buckets):
    with open(path, "w") as f:
        for t in buckets:
            f.writelines(
                json.dumps(
                    {
                        "turn": t,
                        "focal": f"ai{j}",
                        "rendering": f"air{score}t{t}",
                        "end_score": score,
                        "position_id": "ai",
                    }
                )
                + "\n"
                for j, score in enumerate([50, 90, 120])
            )


def _args(tmp_path, harvest, ai):
    refset = tmp_path / "refset.jsonl"
    refset.write_text("")  # replaced on promotion
    return SimpleNamespace(
        civ_value_url="http://mock/generate",
        civ_value_refset=str(refset),
        civ_value_concurrency=8,
        civ_value_refset_idx=[0, 2, 4, 7],
        civ_value_single_order=True,
        civ_value_harvest=str(harvest),
        civ_value_ai_refset=str(ai),
        civ_value_refresh_log=str(tmp_path / "refresh_log.jsonl"),
        civ_value_refresh_every=10,
    )


@pytest.fixture()
def workdir(tmp_path):
    rng = random.Random(0)
    harvest = tmp_path / "harvest.jsonl"
    # scores 0..99, newest window = 20..99
    _write_jsonl(harvest, (_episode(i, rng) for i in range(100)))
    ai = tmp_path / "ai.jsonl"
    _write_ai_pool(ai, BUCKETS)
    return _args(tmp_path, harvest, ai)


def _oracle(url, prompt):
    """Graded P(A wins) from the true scores embedded in both renderings (a
    hard 0/1 oracle would give every validation episode the same winrate)."""
    a, b = (float(m) for m in re.findall(r"r(\d+)(?:t\d|last)", prompt)[:2])
    return a / (a + b) if (a + b) else 0.5


def test_refresh_promotes_on_good_model(workdir, monkeypatch):
    monkeypatch.setattr(client, "prompt_preference", _oracle)
    potential._load_at.cache_clear()
    report = rs.refresh(workdir, step=10)
    assert report["promoted"]
    assert report["40"]["spearman"] > 0.9
    assert report["last"]["spearman"] > 0.9
    refset = potential.load_reference_set(workdir.civ_value_refset)
    assert set(refset) == set(BUCKETS)
    for rows in refset.values():
        assert len(rows) == rs.N_POLICY + rs.N_AI
        assert [r["end_score"] for r in rows[-2:]] == [90, 120]  # 2 strongest AI
    assert _last_line(workdir.civ_value_refresh_log)["promoted"]


def test_refresh_keeps_the_set_on_a_random_model(workdir, monkeypatch):
    rng = random.Random(1)
    monkeypatch.setattr(client, "prompt_preference", lambda u, p: rng.random())
    potential._load_at.cache_clear()
    report = rs.refresh(workdir, step=10)
    assert report["promoted"] is False and "warning" in report
    assert Path(workdir.civ_value_refset).read_text() == ""
    assert not os.path.exists(workdir.civ_value_refset + ".candidate")
    assert _last_line(workdir.civ_value_refresh_log)["promoted"] is False


def test_refresh_never_promotes_a_constant_model(workdir, monkeypatch):
    # constant Phi -> Spearman NaN, which must fail the gate
    monkeypatch.setattr(client, "prompt_preference", lambda u, p: 0.5)
    potential._load_at.cache_clear()
    report = rs.refresh(workdir, step=10)
    assert report["promoted"] is False
    assert Path(workdir.civ_value_refset).read_text() == ""


def test_refresh_skips_and_logs_on_thin_harvest(workdir, tmp_path):
    thin = tmp_path / "thin.jsonl"
    with open(thin, "w") as f:
        f.writelines(
            json.dumps(_episode(i, random.Random(2))) + "\n" for i in range(10)
        )
    workdir.civ_value_harvest = str(thin)
    report = rs.refresh(workdir, step=10)
    assert "skipped" in report
    assert "skipped" in _last_line(workdir.civ_value_refresh_log)


def test_maybe_refresh_cadence(workdir, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "refresh", lambda a, step: calls.append(step))
    rs._step_counter["n"] = 0
    for _ in range(25):
        rs.maybe_refresh(workdir)
    assert calls == [10, 20]
    # off without an AI reference pool
    rs._step_counter["n"] = 0
    workdir.civ_value_ai_refset = None
    for _ in range(15):
        assert rs.maybe_refresh(workdir) is None
    assert calls == [10, 20]


def test_maybe_refresh_default_cadence_is_four(workdir, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "refresh", lambda a, step: calls.append(step))
    rs._step_counter["n"] = 0
    del workdir.civ_value_refresh_every
    for _ in range(9):
        rs.maybe_refresh(workdir)
    assert calls == [4, 8]


def test_mtime_cache_picks_up_replacement(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps({"turn": 10, "focal": "x", "rendering": "r", "end_score": 1}) + "\n"
    )
    assert potential.load_reference_set(str(path))[10][0]["end_score"] == 1
    tmp = tmp_path / "a.jsonl.candidate"
    tmp.write_text(
        json.dumps({"turn": 10, "focal": "x", "rendering": "r", "end_score": 2}) + "\n"
    )
    now = os.stat(path).st_mtime_ns
    os.replace(tmp, path)
    os.utime(path, ns=(now + 10**6, now + 10**6))  # make sure the mtime moves
    assert potential.load_reference_set(str(path))[10][0]["end_score"] == 2


def test_refresh_buckets_follow_the_ai_pool(tmp_path, monkeypatch):
    # end turn 120: buckets 10..110; probes = every bucket divisible by 20 + last
    buckets = list(range(10, 120, 10))
    rng = random.Random(3)
    harvest = tmp_path / "harvest.jsonl"
    with open(harvest, "w") as f:
        f.writelines(
            json.dumps(_episode(i, rng, buckets, 119)) + "\n" for i in range(100)
        )
    ai = tmp_path / "ai.jsonl"
    _write_ai_pool(ai, buckets)
    args = _args(tmp_path, harvest, ai)
    monkeypatch.setattr(client, "prompt_preference", _oracle)
    potential._load_at.cache_clear()
    report = rs.refresh(args, step=10)
    assert report["promoted"]
    refset = potential.load_reference_set(args.civ_value_refset)
    assert set(refset) == set(buckets)
    for rows in refset.values():
        assert len(rows) == rs.N_POLICY + rs.N_AI
    for tag in ("20", "40", "60", "80", "100", "last"):
        assert report[tag]["spearman"] > 0.9, tag


def test_refresh_rebuilds_reachable_buckets_of_mid_game_starts(tmp_path, monkeypatch):
    # episodes that start at turn 40 never reach the early buckets: those keep
    # their AI positions and the reachable ones are rebuilt
    buckets = list(range(10, 120, 10))
    rng = random.Random(4)
    harvest = tmp_path / "harvest.jsonl"
    with open(harvest, "w") as f:
        f.writelines(
            json.dumps(_episode(i, rng, [t for t in buckets if t >= 40], 119)) + "\n"
            for i in range(100)
        )
    ai = tmp_path / "ai.jsonl"
    _write_ai_pool(ai, buckets)
    args = _args(tmp_path, harvest, ai)
    monkeypatch.setattr(client, "prompt_preference", _oracle)
    potential._load_at.cache_clear()
    report = rs.refresh(args, step=10)
    assert report["promoted"], report
    assert report["rebuilt_buckets"] == [t for t in buckets if t >= 40]
    refset = potential.load_reference_set(args.civ_value_refset)
    assert set(refset) == set(buckets)
    for t, rows in refset.items():
        if t < 40:
            assert len(rows) == 3 and all(r["position_id"] == "ai" for r in rows)
        else:
            assert len(rows) == rs.N_POLICY + rs.N_AI
