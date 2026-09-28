"""Expanded action set and legal-action discovery against the live server.

- Legal actions: with the EnvSpec's legal_actions enabled, an agent can ask the
  engine what each unit may legally do (PACKET_UNIT_GET_ACTIONS -> UNIT_ACTIONS),
  getting real, context-sensitive answers.
- Rates / worked-tile packets: their exact wire bytes are asserted in
  test_client_proto.py; here they go through the live pipeline without
  breaking the delta stream, i.e. a run issuing them stays byte-identical on
  the same seed.
"""

from civharness import (
    DoAction,
    EnvSpec,
    FunctionAgent,
    MakeWorker,
    SetRates,
    position,
    run_agents,
)


def test_legal_action_discovery(save, root):
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    collected: dict[str, list] = {}

    def probe(obs):
        if obs.legal_actions and obs.my_units and not collected:
            xs = obs.terrain["xsize"] if obs.terrain else 0
            for u in obs.my_units:
                tile = u["y"] * xs + u["x"]
                la = obs.legal_actions(u["id"], target_tile=tile)
                if la:
                    collected[u["type"]] = sorted(la)
        return []

    env = EnvSpec.from_dict(
        {"name": "probe", "observations": {"terrain": True, "legal_actions": True}}
    )
    run_agents(
        save,
        root / "legal",
        agents={focal: FunctionAgent(probe)},
        until=save.turn + 2,
        seeds=[1],
        env=env,
    )
    assert collected, "no legal actions returned for any unit"
    all_actions = {a for acts in collected.values() for a in acts}
    # Disband Unit is available to essentially any unit — a reliable anchor.
    assert "Disband Unit" in all_actions, all_actions


def test_do_action_executes(save, root):
    """DoAction performs a discovered action end-to-end: disbanding a unit
    removes it, which is unambiguous in the resulting save. 'Disband Unit' is
    a self action available to essentially any unit."""
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    before = position(save).players[focal]["units"]
    assert before, "need at least one unit to disband"
    target = before[0]["id"]

    def act(obs):
        return [DoAction(target, "Disband Unit")] if obs.turn == save.turn else []

    trajs = run_agents(
        save,
        root / "doact",
        agents={focal: FunctionAgent(act)},
        until=save.turn + 2,
        seeds=[1],
    )
    after = {u["id"] for u in position(trajs[0].save_path).players[focal]["units"]}
    assert target not in after, (
        f"unit {target} still present after 'Disband Unit' — DoAction did not execute"
    )


def test_new_actions_pipeline_deterministic(save, root):
    focal = next(n for n, p in position(save).players.items() if p["cities"])

    def act(obs):
        if obs.turn != save.turn or not obs.my_cities:
            return []
        c = obs.my_cities[0]
        xs = obs.terrain["xsize"] if obs.terrain else 0
        # A valid rate split under Despotism (max 60), plus a worked-tile packet
        # (rejected-or-applied faithfully; either way the stream must stay sane).
        return [SetRates(60, 0, 40), MakeWorker(c["id"], c["y"] * xs + c["x"])]

    env = EnvSpec.from_dict({"observations": {"terrain": True}})
    a = run_agents(
        save,
        root / "a",
        agents={focal: FunctionAgent(act)},
        until=save.turn + 4,
        seeds=[1],
        env=env,
    )
    b = run_agents(
        save,
        root / "b",
        agents={focal: FunctionAgent(act)},
        until=save.turn + 4,
        seeds=[1],
        env=env,
    )
    assert a[0].save_sha256_normalized == b[0].save_sha256_normalized, (
        "issuing the new action packets broke same-seed determinism"
    )
