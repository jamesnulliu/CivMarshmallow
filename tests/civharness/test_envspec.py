"""EnvSpec: a declared observation + action space, and its enforcement.

The no-server tests cover loading (including the bundled presets), validation,
and order gating; the server test proves a forbidden action is rejected
end-to-end inside a real run (strict), and dropped when lenient.
"""

import pytest

from civharness import (
    EnvSpec,
    ForbiddenAction,
    FunctionAgent,
    GameConfig,
    position,
    run_agents,
    snapshot_series,
)
from civharness.envspec import all_action_keys, preset_names
from civharness.policy.client import BuildUnit, SetResearch


# -- no-server ---------------------------------------------------------------
def test_full_allows_all():
    full = EnvSpec.full()
    assert full.actions == frozenset(all_action_keys())
    for key in all_action_keys():
        assert full.allows(key)


def test_subset_strict_forbids():
    env = EnvSpec.from_dict({"name": "r", "actions": ["set_research"]})
    assert env.check_order(SetResearch("Pottery")) is True
    raised = False
    try:
        env.check_order(BuildUnit(1, "Warriors"))
    except ForbiddenAction:
        raised = True
    assert raised, "strict EnvSpec did not reject a disallowed order"


def test_lenient_drops():
    env = EnvSpec.from_dict({"actions": ["set_research"], "strict": False})
    assert env.check_order(BuildUnit(1, "Warriors")) is False  # dropped, no raise


def test_unknown_keys_rejected():
    for bad in (
        {"actions": ["teleport_the_capital"]},
        {"observations": {"weather": True}},
    ):
        raised = False
        try:
            EnvSpec.from_dict(bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad}"


def test_no_action_key_fails_closed_when_restricted():
    """An order without an action_key is denied under a restricted action space
    (fail-closed), so a custom order type can't slip a forbidden packet through.
    An unrestricted env still allows it."""

    class _Custom:  # a custom order that forgot to declare action_key
        pass

    restricted = EnvSpec.from_dict({"actions": ["set_research"]})
    assert restricted.restricts_actions() is True
    raised = False
    try:
        restricted.check_order(_Custom())
    except ForbiddenAction:
        raised = True
    assert raised, "a no-action_key order slipped through a restricted strict env"

    lenient = EnvSpec.from_dict({"actions": ["set_research"], "strict": False})
    assert lenient.check_order(_Custom()) is False  # dropped, not raised

    assert EnvSpec.full().restricts_actions() is False
    assert EnvSpec.full().check_order(_Custom()) is True  # unrestricted: allowed


def test_custom_order_cannot_spoof_action_key():
    """Gating is by canonical order class, not the object's self-declared
    action_key string: a custom class that claims an allowed key but does
    something else in apply() is still denied under a restricted env."""
    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass
    class _EvilSetResearch:  # labels itself 'set_research' but isn't the real class
        action_key: ClassVar[str] = "set_research"

        def apply(self, client):
            client.declare_war(2)  # would smuggle a forbidden action

    env = EnvSpec.from_dict({"actions": ["set_research"]})
    raised = False
    try:
        env.check_order(_EvilSetResearch())
    except ForbiddenAction:
        raised = True
    assert raised, "a spoofed action_key slipped past canonical-class gating"
    # the genuine registered order type is still allowed.
    assert env.check_order(SetResearch("Pottery")) is True


def test_full_information_gating():
    """Only a full-information env exposes the raw save; dialing down any world
    field marks the view restricted. terrain/legal_actions are additive
    extras, not restrictions."""
    assert EnvSpec.full().full_information() is True
    for restricting in (
        {"players": "self"},
        {"techs": "count"},
        {"techs": False},
        {"cities": ["id"]},
        {"units": False},
        {"government": False},
        {"rates": False},
        {"diplomacy": False},
    ):
        env = EnvSpec.from_dict({"observations": restricting})
        assert env.full_information() is False, restricting
    extras = EnvSpec.from_dict(
        {"observations": {"terrain": True, "legal_actions": True}}
    )
    assert extras.full_information() is True

    # shipped presets: full exposes the save; the restricted ones withhold it.
    pytest.importorskip("yaml")
    assert EnvSpec.preset("full").full_information() is True
    for name in ("economy", "research_only", "military"):
        assert EnvSpec.preset(name).full_information() is False, name


def test_malformed_selectors_rejected():
    """Observation selector VALUES are validated, not just keys: a typo or a
    wrong type is rejected rather than silently *widening* what the agent sees."""
    for bad in (
        {
            "observations": {"government": "false"}
        },  # string reads truthy -> exposes save
        {"observations": {"players": "slef"}},  # typo -> would fall through to 'all'
        {"observations": {"players": True}},
        {"observations": {"techs": "all"}},  # not in the enum
        {"observations": {"techs": True}},
        {"observations": {"terrain": "yes"}},
        {"observations": {"legal_actions": "on"}},
        {"observations": {"rates": "yes"}},
        {"observations": {"diplomacy": 1}},
        {"observations": {"cities": "id,name"}},  # string, not a list
        {"observations": {"cities": [1, 2]}},  # non-string entries
    ):
        raised = False
        try:
            EnvSpec.from_dict(bad)
        except ValueError:
            raised = True
        assert raised, f"malformed spec accepted (fail-open): {bad}"

    # every valid form still loads.
    EnvSpec.from_dict(
        {
            "observations": {
                "government": False,
                "players": "self",
                "techs": "count",
                "cities": ["id"],
                "units": False,
                "terrain": True,
                "legal_actions": False,
            }
        }
    )


def test_presets_load(tmp_path):
    pytest.importorskip("yaml")
    assert preset_names() == [
        "economy",
        "full",
        "military",
        "research_only",
        "saga_fullinfo",
        "saga_visible",
    ]
    for name in preset_names():
        env = EnvSpec.preset(name)
        assert env.actions <= frozenset(all_action_keys())
        assert set(env.observations) <= {
            "techs",
            "cities",
            "units",
            "government",
            "rates",
            "diplomacy",
            "players",
            "terrain",
            "legal_actions",
            "visibility",
        }
    assert EnvSpec.preset("full") == EnvSpec.full()
    # from_yaml reads the same format from any path
    (tmp_path / "r.yaml").write_text("name: r\nactions: [set_research]\n")
    assert EnvSpec.from_yaml(tmp_path / "r.yaml").actions == {"set_research"}
    with pytest.raises(ValueError):
        EnvSpec.preset("no_such_preset")


# -- server ------------------------------------------------------------------
@pytest.mark.server
def test_forbidden_action_enforced(root):
    refs = snapshot_series(
        GameConfig(aifill=3, endturn=20, mapseed=11, gameseed=11),
        root / "seed",
        every=10,
    )
    save = next(r for r in refs if r.turn >= 10)
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    build = FunctionAgent(
        lambda obs: (
            [BuildUnit(obs.my_cities[0]["id"], "Warriors")] if obs.my_cities else []
        )
    )
    until = save.turn + 3

    # strict: the disallowed build raises out of run_agents.
    strict = EnvSpec.from_dict({"name": "research-only", "actions": ["set_research"]})
    raised = False
    try:
        run_agents(
            save,
            root / "strict",
            agents={focal: build},
            until=until,
            seeds=[1],
            env=strict,
        )
    except ForbiddenAction:
        raised = True
    assert raised, "a forbidden action was not rejected inside run_agents (strict)"

    # lenient: the same order is dropped and the run completes.
    lenient = EnvSpec.from_dict({"actions": ["set_research"], "strict": False})
    trajs = run_agents(
        save,
        root / "lenient",
        agents={focal: build},
        until=until,
        seeds=[1],
        env=lenient,
    )
    assert trajs and trajs[0].turns >= save.turn
