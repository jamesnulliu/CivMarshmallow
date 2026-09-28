"""Pure, server-free tests for the SAGA-aligned observation substrate."""

from civharness import (
    EnvSpec,
    Observation,
    ObservationTools,
    SceneGraphConfig,
    build_bounded_digest,
    build_scene_graph,
    discover_unit_action_candidates,
)
from civharness.candidates import ACTION_CANDIDATE_SCHEMA
from civharness.observe import observe_state
from civharness.spatial import DIGEST_SCHEMA, SCENE_SCHEMA


def _observation(*, reverse=False):
    own = {
        "player_id": 0,
        "name": "Alice",
        "is_alive": True,
        "score": 100,
        "gold": 42,
        "ncities": 1,
        "nunits": 2,
        "techs": 2,
        "techs_known": {"Alphabet", "Bronze Working"},
        "researching": "Writing",
        "research_goal": "The Republic",
        "government": "Despotism",
        "rates": {"tax": 60, "luxury": 0, "science": 40},
        "cities": [
            {
                "id": 10,
                "name": "Home",
                "x": 2,
                "y": 2,
                "size": 3,
                "building": "Granary",
                "food_stock": 8,
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
                "x": 3,
                "y": 2,
                "hp": 10,
                "moves": 3,
                "veteran": 0,
                "activity": "Idle",
                "homecity": 10,
            },
            {
                "id": 21,
                "type": "Warriors",
                "x": 4,
                "y": 2,
                "hp": 10,
                "moves": 3,
                "veteran": 1,
                "activity": "Fortified",
                "homecity": 10,
            },
        ],
    }
    enemy = {
        "player_id": 1,
        "name": "Bob",
        "is_alive": True,
        "cities": [{"id": 11, "name": "Far", "x": 15, "y": 2, "size": 2}],
        "units": [
            {
                "id": 30,
                "type": "Warriors",
                "x": 9,
                "y": 2,
                "hp": 8,
                "moves": 0,
                "veteran": 0,
                "activity": "Idle",
            }
        ],
    }
    if reverse:
        own["units"] = list(reversed(own["units"]))
        players = {"Bob": enemy, "Alice": own}
    else:
        players = {"Alice": own, "Bob": enemy}

    def legal(unit_id, *, target_tile=-1, **_kwargs):
        if target_tile == 2 * 20 + 3 and unit_id == 20:
            return {
                "Found City": (100, 100),
                "Build Road": (100, 100),
                "Fortify": (100, 100),
            }
        return {"Unit Move": (80, 100), "Attack": (50, 75)}

    return Observation(
        turn=40,
        me="Alice",
        my=own,
        players=players,
        save_path=None,
        terrain={
            "xsize": 20,
            "ysize": 12,
            "grid": ["g" * 20] * 12,
            "legend": {"g": "Grassland"},
        },
        rates=own["rates"],
        diplomacy={},
        legal_actions=legal,
    )


def test_scene_graph_relations_and_permutation_stability():
    config = SceneGraphConfig.saga(proximity_radius=3, influence_radius=3)
    graph = build_scene_graph(_observation(), config)
    relations = {e.relation for e in graph.edges}
    assert {"anchor", "proximity", "attack", "threat"} <= relations
    assert any(
        e.source == "unit:0:20" and e.target == "city:0:10"
        for e in graph.edges
        if e.relation == "anchor"
    )
    assert any(
        e.source == "unit:0:21" and e.target == "city:1:11"
        for e in graph.edges
        if e.relation == "attack"
    )
    assert any(e.source == "unit:1:30" and e.relation == "threat" for e in graph.edges)
    assert graph.sha == build_scene_graph(_observation(reverse=True), config).sha
    assert graph.entity_view("unit:0:21")["edges"]
    assert graph.as_dict()["schema"] == SCENE_SCHEMA == "civharness-scene-v1"
    assert graph.observation_scope == "save_derived_full_state"


def test_bounded_digest_and_tools_report_real_absence():
    obs = _observation()
    config = SceneGraphConfig.saga(
        proximity_radius=3, influence_radius=3, max_city_status=1, max_alerts=1
    )
    digest = build_bounded_digest(obs, build_scene_graph(obs, config))
    assert digest.metrics["rates"]["science"] == 40
    assert digest.as_dict()["schema"] == DIGEST_SCHEMA == "civharness-bounded-digest-v1"
    assert len(digest.metrics["city_status"]) <= 1
    assert len(digest.alerts) <= 1
    assert set(digest.expansion["anchored_units_by_direction"]) == {
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
        "HERE",
    }
    assert digest.expansion["terrain"]["source"] == "save_full_state_terrain"
    assert (
        sum(x["tile_count"] for x in digest.expansion["terrain"]["directions"].values())
        == 20 * 12
    )

    tools = ObservationTools(obs, available_techs=["Code of Laws", "Writing"])
    city = tools.city_metrics(10)["cities"][0]
    assert city["unit_burden"] == 2
    assert city["gold_output"] is None
    assert tools.army_roster()["units"][1]["fortified"] is True
    assert tools.tech_status()["next_available"] == ["Code of Laws", "Writing"]


def test_eight_way_action_candidates_are_serializable_and_safe():
    obs = _observation()
    found = discover_unit_action_candidates(obs, unit_limit=1)
    moves = [c for c in found.candidates if c.action == "Unit Move"]
    assert len(moves) == 8
    assert {m.target["direction"] for m in moves} == {
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    }
    assert {c.action for c in found.candidates} >= {"Found City", "Fortify"}
    assert any(
        s.action == "Build Road" and "sub_target" in s.reason for s in found.skipped
    )
    records = found.as_dict()
    assert records["schema"] == ACTION_CANDIDATE_SCHEMA
    assert all("order" in c and "key" in c for c in records["candidates"])


def test_rates_and_diplomacy_parse_without_inventing_missing_values(tmp_path):
    save = tmp_path / "tiny.sav"
    save.write_text("""[game]
turn=7
[savefile]
technology_vector="A_NONE","Alphabet"
improvement_vector="Palace"
[player0]
name="Alice"
gold=12
is_alive=TRUE
ncities=0
nunits=0
team_no=0
government_name="Despotism"
tax=60
luxury=0
science=40
diplstate1.type=3
diplstate1.max_state=4
diplstate1.first_contact_turn=2
diplstate1.turns_left=0
[player1]
name="Bob"
gold=9
is_alive=TRUE
ncities=0
nunits=0
team_no=1
government_name="Despotism"
[research]
""")
    _turn, players, _terrain = observe_state(save, EnvSpec.full())
    assert players["Alice"]["rates"] == {"tax": 60, "luxury": 0, "science": 40}
    relation = players["Alice"]["diplomacy"][1]
    assert relation["player"] == "Bob" and relation["state"] == "peace"
    assert relation["contact_turns_left"] is None
    assert players["Bob"]["rates"]["science"] is None
