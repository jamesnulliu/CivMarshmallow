"""Env v4.1: hex geometry, deterministic bounded state text, strict menu and parser, the merged turn prompt and the
civm41 production rules.

Offline: a synthetic empire, plus real player-visible observations saved as JSON files when CIVMARSH_V41_OBS_DIR
points at a directory of them.  The civm41 test needs the ruleset installed (scripts/install_freeciv_server.sh).
"""

import copy
import json
import os
import random
from pathlib import Path

import pytest

from civmarsh.env.v41.hexmap import HexMap
from civmarsh.env.v41.menu import (
    BundleValidationError,
    obs_block_v4,
    parse_action_bundle_v4,
)
from civmarsh.env.v41.observation import StateV4

OBS_DIR = Path(os.environ.get("CIVMARSH_V41_OBS_DIR", "/nonexistent"))
SAVED_OBS = sorted(OBS_DIR.glob("*.json")) if OBS_DIR.is_dir() else []


def test_hex_neighbors_symmetric_and_distance_matches_bfs():
    m = HexMap(36, 72)
    for idx in range(0, 36 * 72, 7):
        assert m.index(*m.map_to_nat(*m.map_pos(idx))) == idx
        nb = m.neighbors(idx)
        assert len(nb) == 6
        for _d, n in nb:
            assert m.distance(idx, n) == 1
            assert idx in [x for _, x in m.neighbors(n)]
    rng = random.Random(0)
    for _ in range(300):
        a, b = rng.randrange(36 * 72), rng.randrange(36 * 72)
        assert len(m.path(a, b, lambda t: True)) == m.distance(a, b)
    assert len(m.within(1000, 2)) == 19  # radius_sq 5 work area on iso-hex


def _synthetic(n_cities=25, n_units=120):
    """A large empire on a fully known grassland map, for the length bound."""
    HexMap(40, 80)
    tiles = {
        i: {
            "tile": i,
            "terrain": "Grassland",
            "owner": 0,
            "extras": [],
            "resource_id": None,
            "known": 2,
        }
        for i in range(40 * 80)
    }
    rng = random.Random(1)
    city_tiles = rng.sample(range(40 * 80), n_cities)
    cities = [
        {
            "id": 100 + i,
            "name": f"c{i}",
            "tile": t,
            "x": t % 40,
            "y": t // 40,
            "size": 3,
            "mine": True,
            "building": "Warriors",
            "food_stock": 5,
            "shield_stock": 5,
            "shield_surplus": 2,
            "buildings": [],
        }
        for i, t in enumerate(city_tiles)
    ]
    units = [
        {
            "id": 1000 + i,
            "type": ["Warriors", "Settlers", "Workers"][i % 3],
            "tile": t,
            "x": t % 40,
            "y": t // 40,
            "hp": 10,
            "moves": 6,
            "activity": "Idle",
            "mine": True,
            "owner": 0,
        }
        for i, t in enumerate(rng.sample(range(40 * 80), n_units))
    ]
    return {
        "turn": 90,
        "me": "me",
        "rates": {},
        "diplomacy": {},
        "my": {
            "player_id": 0,
            "score": 100,
            "gold": 50,
            "government": "Monarchy",
            "techs_known": ["Alphabet"],
            "cities": cities,
            "units": units,
        },
        "player_view": {
            "map": {"xsize": 40, "ysize": 80, "tiles": tiles},
            "cities": cities,
            "units": units,
        },
    }


def test_state_length_is_bounded_for_a_large_empire():
    d = _synthetic()
    s = StateV4(d)
    txt = s.render()
    assert len(s.sections["CITIES"]) <= s.caps.cities + 1
    idle_lines = [x for x in s.sections["UNITS"] if x.startswith("U")]
    assert len(idle_lines) == s.caps.idle_units
    assert any(x.startswith("idle, not listed") for x in s.sections["UNITS"])
    assert len(txt) < 12000, len(txt)


@pytest.mark.parametrize("path", SAVED_OBS, ids=[p.stem for p in SAVED_OBS])
def test_saved_state_is_deterministic(path):
    d = json.loads(path.read_text())
    a = StateV4(copy.deepcopy(d)).render()
    b = StateV4(copy.deepcopy(d)).render()
    assert a == b
    for sec in ("[OVERVIEW]", "[RIVALS]", "[CITIES]", "[UNITS]", "[MAP]", "[EVENTS]"):
        assert sec in a


@pytest.mark.parametrize("path", SAVED_OBS, ids=[p.stem for p in SAVED_OBS])
def test_every_own_city_and_listed_unit_gets_actions(path):
    d = json.loads(path.read_text())
    _text, menu, state = obs_block_v4(d)
    ents = {c.entity for c in menu.candidates}
    for c in state.cities:
        assert f"city:{c['id']}" in ents
    for u in state.idle_units()[: state.caps.idle_units]:
        assert f"unit:{u['id']}" in ents, u


def test_parser_is_strict():
    d = _synthetic(n_cities=3, n_units=6)
    _text, menu, _state = obs_block_v4(d)
    c = next(x for x in menu.candidates if x.domain == "city")
    ok = parse_action_bundle_v4(json.dumps({"city": [c.key]}), menu)
    assert ok.keys == (c.key,)
    with pytest.raises(BundleValidationError):
        parse_action_bundle_v4(
            json.dumps({"city": ["city:999/build:warriors"]}), menu
        )  # not offered
    with pytest.raises(BundleValidationError):
        parse_action_bundle_v4(json.dumps({"research": []}), menu)  # unknown domain
    same = [x.key for x in menu.candidates if x.entity == c.entity][:2]
    with pytest.raises(BundleValidationError):
        parse_action_bundle_v4(json.dumps({"city": same}), menu)  # one key per entity
    with pytest.raises(BundleValidationError):
        parse_action_bundle_v4("not json", menu)


def test_settle_orders_end_with_found_city_on_the_site():
    d = _synthetic(n_cities=2, n_units=9)
    _text, menu, state = obs_block_v4(d)
    settle = [c for c in menu.candidates if "/settle:" in c.key]
    assert settle, "a settler on an open map must get settle keys"
    o = settle[0].orders[0]
    site = state.refs[settle[0].key.rsplit(":", 1)[1]].tile
    assert o.dest_tile == site
    assert o.orders[-1]["order"] == 4 and o.orders[-1]["target"] == site
    assert len(o.orders) - 1 == state.map.distance(o.src_tile, site)


# -- v4.1: merged prompt, refs for every unit, civm41 production lists ----------------------------------------------


def test_v41_prompt_lists_every_commandable_entity_once_without_target_lists():
    from civmarsh.env.v41.menu import TURN_SCHEMA

    d = _synthetic(n_cities=6, n_units=30)
    text, menu, state = obs_block_v4(d, fmt="entity")
    assert text.startswith(f"TURN {TURN_SCHEMA}")
    ents = set(menu.entity_schema()["properties"])
    for e in ents:
        if e.startswith(("unit:", "city:")):
            assert text.count(e + " ") + text.count(e + ",") >= 1, e
    assert (
        "goto:C" not in text and "settle:S1," not in text
    )  # targets are not listed per unit
    old = state.render() + "\n" + menu.render()
    assert len(text) < len(old)


def test_v41_any_ref_resolves_for_any_unit():
    from civmarsh.env.v41.menu import parse_entity_bundle

    d = _synthetic(n_cities=2, n_units=9)
    _text, menu, state = obs_block_v4(d, fmt="entity")
    worker = next(u for u in state.units if u["type"] == "Workers")
    far = max(
        (r for r in state.refs.values() if r.kind == "site"),
        key=lambda r: state.map.distance(worker["tile"], r.tile),
    )
    b = parse_entity_bundle(
        json.dumps({f"unit:{worker['id']}": f"goto:{far.label}"}), menu, lenient=False
    )
    assert len(b.orders) == 1 and b.orders[0].src_tile == worker["tile"]
    settler = next(u for u in state.units if u["type"] == "Settlers")
    b = parse_entity_bundle(
        json.dumps({f"unit:{settler['id']}": f"settle:{far.label}"}),
        menu,
        lenient=False,
    )
    assert len(b.orders) == 1
    with pytest.raises(BundleValidationError):  # settle needs a site ref
        parse_entity_bundle(
            json.dumps({f"unit:{settler['id']}": "settle:C1"}), menu, lenient=False
        )


def test_v41_schema_allows_refs_and_coordinates():
    import re as _re

    d = _synthetic(n_cities=2, n_units=9)
    _text, menu, state = obs_block_v4(d, fmt="entity")
    props = menu.entity_schema()["properties"]
    settler = next(u for u in state.units if u["type"] == "Settlers")
    worker = next(u for u in state.units if u["type"] == "Workers")
    pats = [
        a["pattern"] for a in props[f"unit:{settler['id']}"]["anyOf"] if "pattern" in a
    ]
    assert any(_re.match(p, "settle:S1") for p in pats) and any(
        _re.match(p, "goto:@3,4") for p in pats
    )
    assert not any(_re.match(p, "settle:C1") for p in pats)
    wpats = [
        a["pattern"] for a in props[f"unit:{worker['id']}"]["anyOf"] if "pattern" in a
    ]
    assert any(_re.match(p, "goto:C1") for p in wpats) and not any(
        _re.match(p, "settle:S1") for p in wpats
    )


def test_civm41_bans_wonders_coinage_ships_diplomats():
    from civharness.buildability import (
        CityContext,
        can_city_build_improvement_now,
        can_city_build_unit_now,
    )
    from civmarsh.env.v41.menu import _ruleset

    try:
        rs = _ruleset("civm41")
    except FileNotFoundError:
        pytest.skip("civm41 not installed (scripts/install_freeciv_server.sh)")
    ctx = CityContext(
        techs_known={
            "Bronze Working",
            "Writing",
            "Map Making",
            "Pottery",
            "Masonry",
            "Currency",
            "Alphabet",
            "Ceremonial Burial",
        },
        city_buildings=[],
        city_size=8,
        all_own_city_buildings=[],
        adjacent_terrains={"Ocean"},
        adjacent_extras=set(),
        center_terrain="Grassland",
        center_extras=set(),
    )
    for unit in ("Trireme", "Diplomat", "Caravan", "Migrants"):
        assert can_city_build_unit_now(rs, unit, ctx) is False, unit
    for b in ("Coinage", "Colossus", "Pyramids", "Great Wall"):
        assert can_city_build_improvement_now(rs, b, ctx) is not True, b
    assert can_city_build_unit_now(rs, "Settlers", ctx) is True
    assert can_city_build_improvement_now(rs, "Granary", ctx) is True
