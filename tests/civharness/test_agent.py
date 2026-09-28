"""Agents act from per-turn observations (not hardcoded state), one or many
players are controlled in a shared game, and the result is deterministic
(same seed -> byte-identical)."""

from pathlib import Path

import pytest

from civharness import (
    BuildImprovement,
    BuildUnit,
    FunctionAgent,
    GameConfig,
    Observation,
    SetResearch,
    agent_noise_floor,
    position,
    run_agent,
    run_agents,
    snapshot_series,
)
from civharness.agent import DEFAULT_PHASE_BUDGET, drive_timeout
from civharness.runner import default_timeout


def _mid_save(root: Path):
    refs = snapshot_series(
        GameConfig(aifill=3, endturn=20, mapseed=7, gameseed=7), root / "seed", every=10
    )
    return next(r for r in refs if r.turn >= 10)


@pytest.mark.server
def test_single_agent_reacts_to_observation(tmp_path):
    save = _mid_save(tmp_path)
    focal = next(n for n, p in position(save).players.items() if p["cities"])

    seen = {}

    def agent(obs: Observation):
        # The agent knows NOTHING a priori — it discovers its city from the
        # observation and reacts. Records what it saw, to prove obs is live.
        seen[obs.turn] = [(c["id"], c["building"]) for c in obs.my_cities]
        assert obs.me == focal and obs.my is obs.players[focal]
        orders = []
        for c in obs.my_cities:
            if c["building"] != "Barracks":
                orders.append(BuildImprovement(c["id"], "Barracks"))
            orders.append(SetResearch("Bronze Working"))
        return orders

    trajs = run_agent(
        save,
        tmp_path / "a",
        focal_player=focal,
        agent=FunctionAgent(agent),
        until=save.turn + 6,
    )
    assert seen, "agent never observed a turn"
    assert seen[save.turn], "first observation had no cities"
    city_id = seen[save.turn][0][0]
    assert any(
        c["id"] == city_id
        for c in position(Path(trajs[0].save_path)).players[focal]["cities"]
    )

    # same seed -> byte-identical (agent is a deterministic function of obs)
    again = run_agent(
        save,
        tmp_path / "a2",
        focal_player=focal,
        agent=FunctionAgent(agent),
        until=save.turn + 6,
    )
    assert again[0].save_sha256_normalized == trajs[0].save_sha256_normalized, (
        "same-seed agent run is not deterministic"
    )


@pytest.mark.server
def test_multi_agent_controls_two_players(tmp_path):
    save = _mid_save(tmp_path)
    pos = position(save)
    owners = [n for n, p in pos.players.items() if p["cities"]][:2]
    assert len(owners) == 2, f"need two city-owning players, got {owners}"

    def builder(unit_name):
        def act(obs: Observation):
            return [BuildUnit(c["id"], unit_name) for c in obs.my_cities]

        return FunctionAgent(act)

    # Two DISTINCT no-tech units (an agent choosing a unit it lacks the tech for
    # would be correctly rejected by the engine — that's faithful behavior).
    agents = {owners[0]: builder("Warriors"), owners[1]: builder("Workers")}
    trajs = run_agents(save, tmp_path / "m", agents=agents, until=save.turn + 6)

    final = position(Path(trajs[0].save_path))
    b0 = [c["building"] for c in final.players[owners[0]]["cities"]]
    b1 = [c["building"] for c in final.players[owners[1]]["cities"]]
    assert "Warriors" in b0, f"{owners[0]} did not build Warriors: {b0}"
    assert "Workers" in b1, f"{owners[1]} did not build Workers: {b1}"

    again = run_agents(save, tmp_path / "m2", agents=agents, until=save.turn + 6)
    assert again[0].save_sha256_normalized == trajs[0].save_sha256_normalized, (
        "same-seed multi-agent run is not deterministic"
    )


@pytest.mark.server
def test_agent_noise_floor(tmp_path):
    save = _mid_save(tmp_path)
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    agents = {
        focal: FunctionAgent(
            lambda obs: [BuildUnit(c["id"], "Warriors") for c in obs.my_cities]
        )
    }

    floor = agent_noise_floor(
        save, tmp_path / "floor", agents=agents, until=save.turn + 6, seeds=[101, 102]
    )
    assert floor.replicas_identical, "agent replicas differ (nondeterministic)"
    assert floor.n_seeds == 2


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_drive_timeout_scaling(scale):
    """The drive deadline is default_timeout scaled by timeout_scale and by the
    per-phase budget relative to the 60 s bot-speed budget."""
    base = default_timeout(120)
    assert drive_timeout(120, timeout_scale=scale, phase_budget=60.0) == scale * base
    assert drive_timeout(120, timeout_scale=scale) == (
        scale * base * (DEFAULT_PHASE_BUDGET / 60.0)
    )
