"""A server-Lua scripted action sequence loaded into a branch produces the
intended, engine-verified game-state change — deterministically (two runs,
bit-identical saves)."""

from pathlib import Path

import pytest

from civharness import GameConfig, branch, position, run_game
from civharness.policy import (
    CreateBuilding,
    ScriptedPolicy,
    UnitAction,
)


@pytest.mark.server
def test_scripted_policy(tmp_path):
    root = tmp_path

    seed_result = run_game(
        GameConfig(aifill=3, endturn=40, mapseed=21, gameseed=21, scorelog=False),
        root / "seedgame",
    )
    save = Path(seed_result.save_path)
    state = position(save)

    # Choose a player with a city and a unit to command. "Disband Unit" is
    # legal for any unit type, engine-reported, and save-verifiable by the
    # unit's subsequent absence.
    player = unit = city = None
    for info in state.players.values():
        if info["is_alive"] and info["cities"] and info["units"]:
            player, unit, city = info, info["units"][0], info["cities"][0]
            break
    assert player, "no player with city + unit"

    act_turn = state.turn + 1
    policy = ScriptedPolicy(
        orders={
            act_turn: [
                UnitAction(player["player_id"], unit["id"], "Disband Unit"),
                CreateBuilding(player["player_id"], city["id"], "Barracks"),
            ]
        }
    )

    t1 = branch(
        save,
        root / "run1",
        until=state.turn + 4,
        seeds=[5],
        scorelog=False,
        policy=policy,
    )[0]
    t2 = branch(
        save,
        root / "run2",
        until=state.turn + 4,
        seeds=[5],
        scorelog=False,
        policy=policy,
    )[0]

    log = (Path(t1.workdir) / "server.log").read_text()
    assert "CIVHARNESS policy loaded" in log, "lua file was not executed"
    assert f"CIVHARNESS executing turn {act_turn}" in log, (
        "turn_begin handler never fired for the order turn"
    )
    assert f"CIVHARNESS action Disband Unit unit={unit['id']} ok=true" in log, (
        "Disband order failed:\n"
        + "\n".join(line for line in log.splitlines() if "CIVHARNESS" in line)
    )
    assert f"CIVHARNESS build Barracks city={city['id']} has=true" in log, (
        "Barracks was not created (engine-verified has_building failed)"
    )
    assert t1.save_sha256_normalized == t2.save_sha256_normalized, (
        "same policy, same seed, different outcome — determinism BROKEN"
    )

    # The disbanded unit must be gone from the final position.
    end_state = position(Path(t1.save_path))
    end_ids = {u["id"] for p in end_state.players.values() for u in p["units"]}
    assert unit["id"] not in end_ids, (
        f"unit {unit['id']} still exists after Disband order"
    )

    # Control: without the policy, the unit survives — the change is caused
    # by the order, not by the passage of turns.
    ctrl = branch(
        save, root / "control", until=state.turn + 4, seeds=[5], scorelog=False
    )[0]
    ctrl_ids = {
        u["id"]
        for p in position(Path(ctrl.save_path)).players.values()
        for u in p["units"]
    }
    assert unit["id"] in ctrl_ids, (
        "control invalid: unit vanished even without the disband order"
    )
    assert ctrl.save_sha256_normalized != t1.save_sha256_normalized
