"""Bank stages with the engine replaced by fakes: position selection, labels
from K branches, resume, and branch-directory cleanup."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import civharness
from civmarsh.oracle import bank
from civmarsh.oracle.bank import (
    PRESETS,
    label_positions,
    position_id,
    select_positions,
)


def test_presets():
    short = PRESETS["short_game"]
    assert short.ruleset is None and short.endturn == 70
    assert (short.min_turn, short.max_turn) == (10, 65)
    assert short.seeds == [*range(4000, 4240), *range(5000, 5030)]
    assert PRESETS["civ2civ3_120"].seeds == list(range(12000, 12040))
    assert PRESETS["classic_120"].max_turn == 115
    assert PRESETS["classic_transfer"].max_turn == 65
    lengths = {n: PRESETS[f"game_length_{n}"] for n in (60, 70, 80)}
    assert [p.seeds[0] for p in lengths.values()] == [7000, 3000, 7100]
    assert all(
        len(p.seeds) == 10 and p.max_turn == p.endturn - 5 for p in lengths.values()
    )
    for p in PRESETS.values():
        assert (p.k, p.skill, p.every, p.game_skill, p.aifill, p.size) == (
            8,
            "normal",
            5,
            "hard",
            4,
            1,
        )


def test_position_id_width_follows_end_turn():
    assert position_id("g4000", 10, 70) == "g4000_T10"
    assert position_id("g9", 10, 120) == "g9_T010"


@dataclass
class FakeState:
    turn: int
    players: dict


@dataclass
class FakeTraj:
    scores: dict


def fake_game(pool: Path, gid: str, turns, focal="Ada"):
    saves = []
    for t in turns:
        p = pool / gid / "saves" / f"freeciv-T{t:04d}-auto.sav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(t))
        saves.append({"turn": t, "path": str(p.relative_to(pool))})
    return {
        "game_id": gid,
        "seed": int(gid[1:]),
        "endturn": 70,
        "focal_player": focal,
        "end_turn": 71,
        "end_score": 99,
        "saves": saves,
        "split": "train",
    }


@pytest.fixture
def engine(monkeypatch):
    """position() reads the turn written in the fake save; branch() returns K
    trajectories whose focal terminal score is turn + branch index."""
    calls = []

    def position(save):
        t = int(Path(save).read_text())
        return FakeState(turn=t, players={"Ada": {"player_id": 0, "score": t // 5}})

    def branch(save, wd, *, until, k, skill):
        calls.append((Path(save).name, until, k, skill))
        Path(wd).mkdir(parents=True, exist_ok=True)
        t = int(Path(save).read_text())
        return [FakeTraj(scores={"Ada": {"total": t + i}}) for i in range(k)]

    monkeypatch.setattr(civharness, "position", position)
    monkeypatch.setattr(civharness, "branch", branch)
    return calls


def test_select_positions_window_and_fields(tmp_path, engine):
    games = [fake_game(tmp_path, "g1", [5, 10, 15, 17, 65, 70, 71])]
    rows = select_positions(games, min_turn=10, max_turn=65, every=5, pool_dir=tmp_path)
    assert [r["turn"] for r in rows] == [10, 15, 65]
    r = rows[0]
    assert r["position_id"] == "g1_T10"
    assert r["score_now"] == 2 and r["focal_player"] == "Ada"
    assert r["game_end_score"] == 99 and r["split"] == "train"
    assert not Path(r["save_path"]).is_absolute()


def test_label_positions_resume_and_cleanup(tmp_path, engine):
    pool, work, out = (
        tmp_path / "pool",
        tmp_path / "branches",
        tmp_path / "labels.jsonl",
    )
    games = [fake_game(pool, "g1", [10, 15]), fake_game(pool, "g2", [10])]
    positions = select_positions(
        games, min_turn=10, max_turn=65, every=5, pool_dir=pool
    )
    res = label_positions(
        positions, pool_dir=pool, until=70, k=4, work_dir=work, out_path=out, workers=2
    )
    assert res["n_labeled"] == 3 and res["n_new"] == 3 and not res["failures"]
    labels = {
        r["position_id"]: r for r in map(json.loads, out.read_text().splitlines())
    }
    lab = labels["g1_T15"]
    assert lab["end_scores"] == [15, 16, 17, 18]
    assert lab["mean_end"] == 16.5 and lab["n"] == 4
    assert lab["se_end"] == pytest.approx(lab["sd_end"] / 2)
    assert lab["until"] == 70 and lab["skill"] == "normal"
    assert lab["score_now"] == 3 and lab["game_end_score"] == 99
    assert {c[1:] for c in engine} == {(70, 4, "normal")}
    assert list(work.iterdir()) == []  # branch dirs deleted once labeled

    res = label_positions(
        positions, pool_dir=pool, until=70, k=4, work_dir=work, out_path=out, workers=2
    )
    assert res["n_new"] == 0 and len(engine) == 3


def test_label_requires_all_k_branches(tmp_path, engine, monkeypatch):
    def partial(save, wd, *, until, k, skill):
        return [FakeTraj(scores={"Ada": {"total": 1}})] + [
            FakeTraj(scores={}) for _ in range(k - 1)
        ]

    monkeypatch.setattr(civharness, "branch", partial)
    pool = tmp_path / "pool"
    positions = select_positions(
        [fake_game(pool, "g1", [10])], min_turn=10, max_turn=65, every=5, pool_dir=pool
    )
    res = label_positions(
        positions,
        pool_dir=pool,
        until=70,
        k=8,
        work_dir=tmp_path / "w",
        out_path=tmp_path / "labels.jsonl",
        workers=1,
    )
    assert res["n_labeled"] == 0
    assert "1/8" in res["failures"][0]["error"]


def test_launch_retry_on_server_crash(monkeypatch):
    from civharness.server import ServerCrash

    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ServerCrash("bind failure")
        return "ok"

    assert bank._with_retry(flaky, "g1") == "ok" and len(attempts) == 3
