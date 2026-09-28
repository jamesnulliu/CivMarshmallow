"""Server/GPU-free tests for the policy observation and action interface."""

import json

import pytest

from civharness import Observation
from civmarsh.env.menu import (
    MENU_FORMAT,
    BundleValidationError,
    MenuConfig,
    build_menu,
    obs_block,
    parse_action_bundle,
    ruleset_dir,
)
from civmarsh.env.prompts import SYSTEM_PROMPT, event_digest
from civmarsh.env.snapshot import FULLSTATE_FORMAT, STATE_FORMAT, SpatialSnapshot


def _has_ruleset():
    try:
        return (ruleset_dir() / "buildings.ruleset").exists()
    except FileNotFoundError:
        return False


needs_ruleset = pytest.mark.skipif(
    not _has_ruleset(),
    reason="no freeciv ruleset (set CIVMARSH_RULESET_DIR or CIVHARNESS_SERVER)",
)


def _obs(player_view=None):
    mine = {
        "player_id": 0,
        "name": "Alice",
        "nation": "Romans",
        "is_alive": True,
        "score": 80,
        "gold": 140,
        "ncities": 1,
        "nunits": 2,
        "techs": 2,
        "techs_known": {"Alphabet", "Monarchy"},
        "researching": "Alphabet",
        "research_goal": None,
        "government": "Despotism",
        "rates": {"tax": 60, "luxury": 0, "science": 40},
        "diplomacy": {1: {"player_id": 1, "player": "Bob", "state": "peace"}},
        "cities": [
            {
                "id": 10,
                "name": "Rome",
                "x": 5,
                "y": 5,
                "size": 3,
                "building": "Warriors",
                "food_stock": 10,
                "shield_stock": 4,
                "shield_surplus": 2,
                "specialists": 0,
                "buildings": ["Palace"],
            }
        ],
        "units": [
            {
                "id": 20,
                "type": "Workers",
                "x": 5,
                "y": 6,
                "hp": 10,
                "moves": 3,
                "veteran": 0,
                "activity": "Idle",
                "homecity": 10,
            },
            {
                "id": 21,
                "type": "Warriors",
                "x": 6,
                "y": 5,
                "hp": 10,
                "moves": 3,
                "veteran": 0,
                "activity": "Fortified",
                "homecity": 10,
            },
        ],
    }
    enemy = {
        "player_id": 1,
        "name": "Bob",
        "nation": "Greeks",
        "is_alive": True,
        "score": 75,
        "ncities": 1,
        "nunits": 1,
        "cities": [{"id": 11, "name": "Athens", "x": 13, "y": 5, "size": 2}],
        "units": [
            {
                "id": 30,
                "type": "Warriors",
                "x": 10,
                "y": 5,
                "hp": 8,
                "moves": 1,
                "activity": "Idle",
            }
        ],
    }
    own_tiles = {20: 6 * 20 + 5, 21: 5 * 20 + 6}

    def legal(unit_id, *, target_tile=-1, **_kwargs):
        if target_tile == own_tiles[unit_id]:
            return (
                {"Found City": (100, 100), "Build Road": (100, 100)}
                if unit_id == 20
                else {"Fortify": (100, 100)}
            )
        return {"Unit Move": (75, 100)}

    return Observation(
        turn=30,
        me="Alice",
        my=mine,
        players={"Alice": mine, "Bob": enemy},
        save_path=None,
        terrain={
            "xsize": 20,
            "ysize": 12,
            "grid": ["g" * 20] * 12,
            "legend": {"g": "Grassland"},
        },
        rates=mine["rates"],
        diplomacy=mine["diplomacy"],
        legal_actions=legal,
        player_view=player_view,
    )


TECH_REQS = {
    "Writing": ("Alphabet", None),
    "Code of Laws": ("Alphabet", None),
    "Future Tech": ("Writing", "Code of Laws"),
}


def test_model_visible_format_tags():
    assert STATE_FORMAT == "civm-saga-spatial-v1-playervisible"
    assert MENU_FORMAT == "civm-action-menu-v3"


def test_actor_and_value_input_share_canonical_facts():
    snapshot = SpatialSnapshot.from_observation(
        _obs(), available_techs=["Writing", "Code of Laws"]
    )
    actor = snapshot.actor_global_view()
    value = snapshot.value_input_view()
    assert actor["fact_sha"] == value["fact_sha"] == snapshot.fact_sha
    assert actor["digest"] == value["digest"]
    assert actor["graph_sha"] == value["graph_sha"]
    assert "grid" not in snapshot.render_actor_global()
    assert value["army_roster"]["units"][0]["id"] == 20
    assert snapshot.entity_view("unit", 20)["scene"]["edges"]
    # no player view -> omniscient format tag, in both renderings
    assert snapshot.format_name == FULLSTATE_FORMAT
    assert snapshot.render_actor_global().startswith(
        f"STATE {FULLSTATE_FORMAT} fact_sha="
    )
    assert snapshot.render_value_input().startswith(
        f"Freeciv state value input ({FULLSTATE_FORMAT}) fact_sha={snapshot.fact_sha}\nDIGEST "
    )
    assert '"score":80,' in snapshot.render_value_input()


@needs_ruleset
def test_grouped_menu_covers_six_domains_and_eight_way_movement():
    obs = _obs()
    menu = build_menu(obs, MenuConfig(max_units=2), tech_requirements=TECH_REQS)
    assert set(menu.grouped) == {
        "technology",
        "government",
        "diplomacy",
        "city",
        "civilian",
        "military",
    }
    assert all(menu.grouped[d] for d in menu.grouped)
    worker_moves = [
        c for c in menu.grouped["civilian"] if c.key.startswith("unit:20/move:")
    ]
    assert len(worker_moves) == 8
    # every offered production item is checked against the ruleset
    city_prod = [c for c in menu.grouped["city"] if "/build-" in c.key]
    assert city_prod, "no production candidates offered"
    assert all(c.source == "ruleset_city_buildability_checked" for c in city_prod)
    text = obs_block(obs, menu)
    assert menu.sha in text and "ACTION_JSON" in text and "grid" not in text
    assert f"ACTION_MENU {MENU_FORMAT} sha={menu.sha}" in text


@needs_ruleset
def test_action_bundle_is_multi_order_and_strict():
    menu = build_menu(_obs(), tech_requirements=TECH_REQS)
    payload = {
        "technology": [menu.grouped["technology"][0].key],
        "government": [
            next(c.key for c in menu.grouped["government"] if "/rates:" in c.key)
        ],
        "city": [menu.grouped["city"][0].key],
        "civilian": [
            next(c.key for c in menu.grouped["civilian"] if "/move:" in c.key)
        ],
    }
    bundle = parse_action_bundle(json.dumps(payload), menu)
    assert len(bundle.orders) == 4
    assert bundle.candidate_sha == menu.sha
    assert len(bundle.as_dict()["orders"]) == 4

    same_unit = [c.key for c in menu.grouped["civilian"] if "/move:" in c.key][:2]
    with pytest.raises(BundleValidationError, match="same entity"):
        parse_action_bundle(json.dumps({"civilian": same_unit}), menu)
    with pytest.raises(BundleValidationError, match="not offered"):
        parse_action_bundle('{"city":["not-offered"]}', menu)
    with pytest.raises(BundleValidationError, match="unknown domains"):
        parse_action_bundle('{"research":[]}', menu)


def test_system_prompt_carries_start_turn():
    s = SYSTEM_PROMPT.format(focal="p", endturn=120, startturn=1)
    assert "from turn 1 to turn 120" in s
    s = SYSTEM_PROMPT.format(focal="p", endturn=120, startturn=100)
    assert "from turn 100 to turn 120" in s
    assert "highest score at turn 120" in s
    assert "Use {} to pass." in s


def test_event_digest_window_and_wording():
    assert event_digest([], 5) == "Recent actions: none (first decision).\n"
    records = [
        {"turn": 1, "keys": ["a"]},
        {"turn": 2, "keys": []},
        {"turn": 3, "error": "x" * 100},
        {"turn": 4, "keys": ["b", "c"]},
    ]
    assert event_digest(records, 3) == (
        "Recent actions:\n"
        "- turn 2: passed\n"
        f"- turn 3: no-op ({'x' * 80})\n"
        "- turn 4: executed b, c\n"
    )
