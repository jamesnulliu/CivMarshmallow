"""Server-backed visibility tests — the fog-of-war gate.

Two controlled players on one game: each client's PlayerViewCache is fed
only by its own connection, so the same server position must yield distinct
views, and facts the server never sent a client must be absent from its
observation.
"""

from pathlib import Path

import pytest

from civharness import (
    EnvSpec,
    FunctionAgent,
    GameConfig,
    Observation,
    position,
    run_agents,
    snapshot_series,
)
from civharness.playerview import TILE_KNOWN_SEEN

FOG_ENV = EnvSpec.from_dict(
    {
        "name": "visibility-test",
        "observations": {"visibility": "player_visible"},
    }
)


def _mid_save(root: Path):
    cfg = GameConfig(aifill=3, endturn=20, mapseed=5, gameseed=5)
    refs = snapshot_series(cfg, root / "seed", every=10)
    return next(r for r in refs if r.turn >= 10)


@pytest.mark.server
def test_two_clients_see_different_worlds(tmp_path):
    root = tmp_path
    save = _mid_save(root)
    pos = position(save)
    owners = [n for n, p in pos.players.items() if p["cities"]][:2]
    assert len(owners) == 2, f"need two city-owning players, got {owners}"

    captured: dict[str, list[Observation]] = {o: [] for o in owners}

    def capture(pname):
        def act(obs: Observation):
            captured[pname].append(obs)
            return []

        return FunctionAgent(act)

    agents = {o: capture(o) for o in owners}
    run_agents(save, root / "fog", agents=agents, until=save.turn + 3, env=FOG_ENV)

    a_name, b_name = owners
    obs_a = captured[a_name][0]
    obs_b = captured[b_name][0]

    # 0. both observations are the fog form: no raw save, no global terrain
    for obs in (obs_a, obs_b):
        assert obs.player_view is not None
        assert obs.player_view["visibility"] == "player_visible"
        assert obs.save_path is None, "player_visible must withhold the save"
        assert obs.terrain is None, "global terrain grid must not leak"
        assert list(obs.players) == [obs.me], "other players' dicts must be absent"

    # 1. distinct views of the same position
    tiles_a = set(obs_a.player_view["map"]["tiles"])
    tiles_b = set(obs_b.player_view["map"]["tiles"])
    assert tiles_a != tiles_b, (
        "two players' known-tile sets are identical — fog is not per-client"
    )
    only_a = tiles_a - tiles_b
    only_b = tiles_b - tiles_a
    assert only_a and only_b, (
        f"each player should know tiles the other does not "
        f"(only_a={len(only_a)}, only_b={len(only_b)})"
    )

    # 2. unknown tiles do not leak: every tile in the view was actually sent
    #    (known in {seen, remembered}); the full map is strictly larger
    xsize = obs_a.player_view["map"]["xsize"]
    ysize = obs_a.player_view["map"]["ysize"]
    assert xsize and ysize
    assert len(tiles_a) < xsize * ysize, "player A knows the whole map?"
    assert len(tiles_b) < xsize * ysize, "player B knows the whole map?"

    # 3. own entities present and consistent with the omniscient save
    save_pos = position(save)
    a_units_true = {u["id"] for u in save_pos.players[a_name]["units"]}
    a_units_view = {u["id"] for u in obs_a.player_view["units"] if u["mine"]}
    assert a_units_view == a_units_true, (
        f"own units mismatch: view={a_units_view} save={a_units_true}"
    )
    a_cities_true = {c["id"] for c in save_pos.players[a_name]["cities"]}
    a_cities_view = {c["id"] for c in obs_a.player_view["cities"] if c["mine"]}
    assert a_cities_view == a_cities_true

    # 4. hidden enemy units/cities do not leak: anything foreign in A's view
    #    must stand on a tile A currently sees
    for u in obs_a.player_view["units"]:
        if not u["mine"]:
            t = obs_a.player_view["map"]["tiles"].get(u["tile"])
            assert t is not None and t["known"] == TILE_KNOWN_SEEN, (
                f"foreign unit {u['id']} on unseen tile {u['tile']} leaked"
            )
    # foreign FULL city info must never appear (only own cities are full)
    for c in obs_a.player_view["cities"]:
        if not c["mine"]:
            assert c["form"] == "short", (
                f"foreign city {c['id']} leaked full economy data"
            )
            assert c["food_stock"] is None and c["shield_stock"] is None

    # 5. my.cities / my.units follow the fog view (single source of truth)
    assert {c["id"] for c in obs_a.my_cities} == a_cities_view
    assert {u["id"] for u in obs_a.my_units} == a_units_view


@pytest.mark.server
def test_fog_run_is_deterministic(tmp_path):
    """visibility=player_visible must not perturb the engine: same-seed
    reruns stay byte-identical (fog is read-only on the wire)."""
    root = tmp_path
    save = _mid_save(root)
    pos = position(save)
    focal = next(n for n, p in pos.players.items() if p["cities"])
    agents = {focal: FunctionAgent(lambda obs: [])}
    t1 = run_agents(save, root / "d1", agents=agents, until=save.turn + 3, env=FOG_ENV)
    t2 = run_agents(save, root / "d2", agents=agents, until=save.turn + 3, env=FOG_ENV)
    assert t1[0].save_sha256_normalized == t2[0].save_sha256_normalized
