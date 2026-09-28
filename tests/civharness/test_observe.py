"""The save-derived observation carries the known-tech SET (not just a
count), government, city economy, unit hp/moves, and an opt-in terrain grid —
all read from the per-turn autosave, all EnvSpec-gated.

Generates one save, then asserts against it (no live server interaction beyond
the initial snapshot).
"""

from civharness import EnvSpec
from civharness.observe import observe_state


def _a_player_with_a_city(players):
    return next(p for p in players.values() if p.get("cities"))


def test_rich_fields_present_and_consistent(save):
    _, players, terrain = observe_state(save.path, EnvSpec.full())
    p = _a_player_with_a_city(players)

    # known-tech SET, and it agrees with the count.
    assert isinstance(p["techs_known"], set)
    assert p["techs"] == len(p["techs_known"]), (p["techs"], p["techs_known"])
    assert "A_NONE" not in p["techs_known"]  # null tech excluded
    assert isinstance(p["government"], str) and p["government"]

    # city economy fields.
    c = p["cities"][0]
    for key in ("size", "food_stock", "shield_stock", "shield_surplus", "specialists"):
        assert isinstance(c[key], int), (key, c[key])
    assert isinstance(c["buildings"], list)  # names, from the improvements bitstring

    # unit fields.
    if p.get("units"):
        u = p["units"][0]
        for key in ("hp", "moves", "veteran", "fuel"):
            assert isinstance(u[key], int), (key, u[key])
        assert u["type"]

    # full() leaves terrain OFF (opt-in).
    assert terrain is None


def test_terrain_opt_in(save):
    env = EnvSpec.from_dict({"observations": {"terrain": True}})
    _, players, terrain = observe_state(save.path, env)
    assert terrain is not None
    assert terrain["ysize"] == len(terrain["grid"])
    assert terrain["xsize"] == len(terrain["grid"][0])
    assert terrain["legend"]  # ident -> name
    c = _a_player_with_a_city(players)["cities"][0]
    ident = terrain["grid"][c["y"]][c["x"]]
    assert ident in terrain["legend"], ident


def test_envspec_subsets_observation(save):
    # cities restricted to a field list; units/techs disabled entirely.
    env = EnvSpec.from_dict(
        {
            "observations": {
                "cities": ["id", "name", "size"],
                "units": False,
                "techs": False,
            }
        }
    )
    _, players, _ = observe_state(save.path, env)
    p = _a_player_with_a_city(players)
    assert set(p["cities"][0]) == {"id", "name", "size"}
    assert "units" not in p and "techs" not in p and "techs_known" not in p

    # self vs all is applied by the driver; here confirm the raw players map has
    # every player so the driver can choose.
    assert len(players) >= 2
