"""The branch API: K=16 branches from a mid-game save yield 16 trajectories;
same-seed replicas are bit-identical; the different-seed spread is measured;
a per-turn scorelog series is extractable; position() returns a structured
state; and a snapshot series can be reloaded without replaying the game."""

import pytest

from civharness import (
    GameConfig,
    branch,
    load_snapshot_series,
    noise_floor,
    position,
    snapshot_series,
)

pytestmark = pytest.mark.server


def test_branch_api(tmp_path):
    # Position library from one pass: autosave every 10 turns to T30.
    refs = snapshot_series(
        GameConfig(aifill=3, endturn=30, mapseed=11, gameseed=11),
        tmp_path / "seedgame",
        every=10,
    )
    assert len(refs) >= 3, f"expected >=3 snapshots, got {len(refs)}"
    assert load_snapshot_series(tmp_path / "seedgame") == refs
    mid = next(r for r in refs if r.turn >= 20)

    # K=16 distinct seeds, 15 turns forward.
    until = mid.turn + 15
    trajs = branch(mid, tmp_path / "branches", until=until, k=16, scorelog=True)
    assert len(trajs) == 16
    shas = {t.save_sha256_normalized for t in trajs}
    assert len(shas) > 1, "16 seeds produced no spread at all — suspicious"

    # Same-seed determinism at the API level.
    again = branch(mid, tmp_path / "replay", until=until, seeds=[1], scorelog=False)
    assert again[0].save_sha256_normalized == trajs[0].save_sha256_normalized, (
        "same seed, different outcome — branch determinism BROKEN"
    )

    # Noise floor: replay identity + different-seed spread, per player.
    floor = noise_floor(mid, tmp_path / "floor", until=until)
    assert floor.replicas_identical, "stored-stream replicas differ"
    assert all("stdev" in s for s in floor.score_spread.values())

    # Per-turn readout from the scorelog.
    with_log = next(t for t in trajs if t.scorelog_path)
    player = next(iter(with_log.scores))
    series = with_log.series("settlers", player)
    assert series, "no settler series in scorelog"

    # Structured position parse.
    state = position(mid)
    assert state.turn == mid.turn
    assert state.players, "no players parsed"
    _name, info = next(iter(state.players.items()))
    assert "gold" in info and "cities" in info
