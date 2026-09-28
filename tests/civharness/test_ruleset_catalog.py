"""The ruleset catalog decoded from a live connection's login burst
(PACKET_RULESET_EXTRA / PACKET_RULESET_ACTION) reaches the agent's
observation and carries the worker extras and extra-sub-target actions."""

from pathlib import Path

import pytest

from civharness import (
    EnvSpec,
    FunctionAgent,
    GameConfig,
    position,
    run_agents,
    snapshot_series,
)

ENV = EnvSpec.from_dict(
    {
        "name": "ruleset-catalog-test",
        "observations": {"terrain": True, "legal_actions": True},
    }
)


def _mid_save(root: Path):
    cfg = GameConfig(aifill=3, endturn=25, mapseed=5, gameseed=5)
    refs = snapshot_series(cfg, root / "seed", every=10)
    return next(r for r in refs if r.turn >= 10)


@pytest.mark.server
def test_ruleset_catalog_decoded_from_login(tmp_path):
    save = _mid_save(tmp_path)
    pos = position(save)
    focal = next(
        n
        for n, p in pos.players.items()
        if any(u.get("type") == "Workers" for u in p["units"])
    )

    seen = {}

    def act(obs):
        if obs.turn != save.turn or "catalog" in seen:
            return []
        seen["catalog"] = obs.rulesets
        seen["rule_ids"] = obs.rule_ids
        return []

    run_agents(
        save,
        tmp_path / "d",
        agents={focal: FunctionAgent(act)},
        until=save.turn + 2,
        env=ENV,
    )

    catalog = seen["catalog"]
    assert catalog is not None and catalog.extras, (
        "RULESET_EXTRA packets were not captured"
    )
    assert catalog.actions, "RULESET_ACTION packets were not captured"
    rn = catalog.extra_rule_names()
    assert "Irrigation" in rn and "Mine" in rn and "Road" in rn, (
        f"civ2civ3 worker extras missing from catalog: {sorted(rn)[:10]}"
    )
    worker = {rec["rule_name"] for rec in catalog.worker_target_extras()}
    assert {"Irrigation", "Mine", "Road"} <= worker
    needing = catalog.actions_needing_extra()
    assert needing, "no extra-sub-target actions decoded"
    # every extra-sub-target action id resolves to a rule name via the Lua dump
    id2action = {v: k for k, v in seen["rule_ids"].actions.items()}
    assert all(aid in id2action for aid in needing), sorted(needing)
