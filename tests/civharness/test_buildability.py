"""City-level buildability evaluator: ruleset facts and a server contrast.

Ruleset half: parse the pinned installation's civ2civ3 ruleset and check
hand-verified facts (no game is run).  Server half: positive/negative
contrast — the evaluator's True verdicts must be accepted by the pinned engine
and its False verdicts must be engine-rejected (production unchanged), using
the mutating path ONLY inside this test as ground truth, never in discovery.
Both halves need the pinned installation (its ruleset files).
"""

from pathlib import Path

import pytest

from civharness import FunctionAgent, GameConfig, position, run_agents, snapshot_series
from civharness.buildability import (
    CityContext,
    Ruleset,
    can_city_build_improvement_now,
    can_city_build_unit_now,
    pinned_ruleset_dir,
)
from civharness.config import DEFAULT_BINARY
from civharness.policy.client import BuildImprovement, BuildUnit

pytestmark = pytest.mark.server


def _rs() -> Ruleset:
    return Ruleset.load(pinned_ruleset_dir())


def _ctx(**kw) -> CityContext:
    base = {
        "techs_known": {"Alphabet"},
        "city_buildings": [],
        "city_size": 4,
        "all_own_city_buildings": [],
        "adjacent_terrains": {"Grassland", "Plains"},
        "adjacent_extras": set(),
        "center_terrain": "Grassland",
        "center_extras": set(),
    }
    base.update(kw)
    return CityContext(**base)


def test_ruleset_loads_pinned_civ2civ3():
    assert pinned_ruleset_dir(binary=DEFAULT_BINARY) == pinned_ruleset_dir()
    rs = _rs()
    assert "Warriors" in rs.units and "Aqueduct" in rs.buildings
    assert "Ocean" in rs.oceanic_terrains and "Lake" in rs.oceanic_terrains
    assert rs.units["Engineers"]["reqs"], "Engineers must carry a Tech req"


def test_tech_gate_positive_and_negative():
    rs = _rs()
    # Warriors: no reqs -> always buildable at city level
    assert can_city_build_unit_now(rs, "Warriors", _ctx()) is True
    # Phalanx requires Bronze Working
    assert can_city_build_unit_now(rs, "Phalanx", _ctx()) is False
    assert (
        can_city_build_unit_now(rs, "Phalanx", _ctx(techs_known={"Bronze Working"}))
        is True
    )


def test_obsoleted_by_chain():
    rs = _rs()
    # Workers are obsoleted by Engineers (Explosives); knowing Explosives
    # makes Workers unbuildable
    assert can_city_build_unit_now(rs, "Workers", _ctx()) is True
    assert (
        can_city_build_unit_now(rs, "Workers", _ctx(techs_known={"Explosives"}))
        is False
    )


def test_naval_needs_adjacent_ocean():
    rs = _rs()
    ctx_inland = _ctx(techs_known={"Map Making"})
    assert can_city_build_unit_now(rs, "Trireme", ctx_inland) is False
    ctx_coastal = _ctx(
        techs_known={"Map Making"}, adjacent_terrains={"Grassland", "Ocean"}
    )
    assert can_city_build_unit_now(rs, "Trireme", ctx_coastal) is True
    # unknown adjacency -> unknown, never True
    ctx_unknown = _ctx(techs_known={"Map Making"}, adjacent_terrains=None)
    assert can_city_build_unit_now(rs, "Trireme", ctx_unknown) is None


def test_improvement_city_reqs_and_duplicates():
    rs = _rs()
    v = can_city_build_improvement_now(rs, "Barracks", _ctx())
    assert v in (True, False, None)  # must at least evaluate, not crash
    # already present -> False, regardless of anything else
    assert (
        can_city_build_improvement_now(
            rs, "Barracks", _ctx(city_buildings=["Barracks"])
        )
        is False
    )
    # Aqueduct: Construction + River/Lake adjacency interplay (present=FALSE
    # rows) — with Construction, no river adjacent, no lake adjacent, the
    # negated reqs hold -> True
    v = can_city_build_improvement_now(
        rs, "Aqueduct", _ctx(techs_known={"Construction"})
    )
    assert v is True
    # without the tech -> False
    assert can_city_build_improvement_now(rs, "Aqueduct", _ctx()) is False


def test_unknown_requirement_fails_closed():
    rs = _rs()
    # inject a fake building with an unimplemented req type
    rs.buildings["FakeThing"] = {
        "genus": "Improvement",
        "reqs": [
            {"type": "Nation", "name": "Roman", "range": "Player", "present": True}
        ],
        "obsolete_by": [],
    }
    assert can_city_build_improvement_now(rs, "FakeThing", _ctx()) is None


def test_great_wonder_is_unknown():
    rs = _rs()
    # GreatWonders need world knowledge; the evaluator must not claim legality
    gw = next(n for n, b in rs.buildings.items() if b["genus"] == "GreatWonder")
    v = can_city_build_improvement_now(
        rs, gw, _ctx(techs_known=set(rs.units) | {"Pottery"})
    )
    assert v in (None, False)


# --- server contrast fixture -------------------------------------------------


def test_server_accepts_true_rejects_false(tmp_path):
    """Ground-truth contrast on the pinned engine: for a real mid-game city,
    every evaluator-True unit/building must be accepted by CITY_CHANGE, and
    every evaluator-False must be rejected (production unchanged)."""
    root = tmp_path
    cfg = GameConfig(aifill=3, endturn=25, mapseed=5, gameseed=5)
    refs = snapshot_series(cfg, root / "seed", every=10)
    save = next(r for r in refs if r.turn >= 10)
    pos = position(save)
    focal = next(n for n, p in pos.players.items() if p["cities"])
    me = pos.players[focal]
    city = me["cities"][0]

    rs = _rs()
    ctx = CityContext(
        techs_known=set(me.get("techs_known", set())),
        city_buildings=list(city.get("buildings", [])),
        city_size=city.get("size", 1),
        all_own_city_buildings=[
            b for c in me["cities"] for b in c.get("buildings", [])
        ],
        adjacent_terrains=None,  # not derivable from save here -> those
        adjacent_extras=None,  # verdicts become None and are not tested
        center_terrain=None,
        center_extras=None,
    )

    units_true = [
        n for n in sorted(rs.units) if can_city_build_unit_now(rs, n, ctx) is True
    ][:3]
    units_false = [
        n for n in sorted(rs.units) if can_city_build_unit_now(rs, n, ctx) is False
    ][:3]
    imprs_true = [
        n
        for n in sorted(rs.buildings)
        if can_city_build_improvement_now(rs, n, ctx) is True
    ][:3]
    imprs_false = [
        n
        for n in sorted(rs.buildings)
        if can_city_build_improvement_now(rs, n, ctx) is False
        and n not in ctx.city_buildings
    ][:2]
    assert units_true and units_false, (
        f"contrast needs both verdicts: true={units_true} false={units_false}"
    )

    results = {}

    # each candidate needs its own run: production is one slot. Run one
    # probe per name to observe accept/reject individually.
    for name, is_unit, expect in (
        [(n, True, True) for n in units_true]
        + [(n, True, False) for n in units_false]
        + [(n, False, True) for n in imprs_true]
        + [(n, False, False) for n in imprs_false]
    ):

        def probe(obs, _n=name, _u=is_unit):
            if obs.turn != save.turn:
                return []
            return [
                BuildUnit(city["id"], _n) if _u else BuildImprovement(city["id"], _n)
            ]

        wd = root / f"p-{name.replace(' ', '_')}"
        trajs = run_agents(
            save, wd, agents={focal: FunctionAgent(probe)}, until=save.turn + 2
        )
        final = position(Path(trajs[0].save_path))
        target = None
        for c in final.players[focal]["cities"]:
            if c["id"] == city["id"]:
                target = c
        # accepted iff the city is now building the probe name
        accepted = target is not None and target.get("building") == name
        results[name] = (expect, accepted)

    mismatches = {n: r for n, r in results.items() if r[0] != r[1]}
    assert not mismatches, (
        f"evaluator vs engine mismatches (expected, engine-accepted): {mismatches}"
    )
