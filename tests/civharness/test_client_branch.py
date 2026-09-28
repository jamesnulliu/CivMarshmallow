"""The external client makes genuine construction/research decisions through
the real protocol (what server Lua cannot), and a client-driven branch is
deterministic — same seed -> byte-identical."""

from pathlib import Path

import pytest

from civharness import GameConfig, position, snapshot_series
from civharness.branch import client_branch
from civharness.policy.client import (
    BuildImprovement,
    ClientPolicy,
    SetResearch,
    SetTechGoal,
    SetWorklist,
)

pytestmark = pytest.mark.server


def test_client_branch(tmp_path):
    # A mid-game position with at least one city to control.
    refs = snapshot_series(
        GameConfig(aifill=3, endturn=20, mapseed=7, gameseed=7),
        tmp_path / "seedgame",
        every=10,
    )
    save = next(r for r in refs if r.turn >= 10)
    pos = position(save)
    focal = next(n for n, p in pos.players.items() if p["cities"])
    city = pos.players[focal]["cities"][0]
    city_id, original = city["id"], city["building"]

    # A real decision: build a Barracks (queue Warriors, Phalanx) and research.
    policy = ClientPolicy(
        orders={
            save.turn: [
                BuildImprovement(city_id, "Barracks"),
                SetWorklist(city_id, (("unit", "Warriors"), ("unit", "Phalanx"))),
                SetResearch("Bronze Working"),
                SetTechGoal("Currency"),
            ],
        }
    )
    until = save.turn + 6

    # Decision takes effect: the city is no longer building what it was.
    trajs = client_branch(
        save, tmp_path / "b1", until=until, focal_player=focal, policy=policy
    )
    assert len(trajs) == 1
    after = position(Path(trajs[0].save_path)).players[focal]["cities"]
    building = next(c["building"] for c in after if c["id"] == city_id)
    assert building != original or original == "Barracks", (
        f"production decision ignored: still building {building!r}"
    )

    # Determinism: same policy, same seed -> byte-identical continuation.
    again = client_branch(
        save, tmp_path / "b2", until=until, focal_player=focal, policy=policy
    )
    assert again[0].save_sha256_normalized == trajs[0].save_sha256_normalized, (
        "same-seed client-driven branch is NOT deterministic"
    )

    # A different seed gives a different (still valid) continuation.
    other = client_branch(
        save, tmp_path / "b3", until=until, focal_player=focal, policy=policy, seeds=[7]
    )
    assert len(other) == 1
