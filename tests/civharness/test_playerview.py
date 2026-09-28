"""Server-free byte-level tests for the per-client player-view cache.

Wire bytes are hand-encoded per freeciv-3.2.5 packets.def / packets_gen.c
(delta bitvector LSB-first, then key, then only the set fields; BOOLs fold
into the bitvector).  These are the negative/positive fixtures for the
player-visible (fog-of-war) observation — no server needed.
"""

from civharness.playerview import (
    _CITY_SHORT_FIELDS,
    _TILE_FIELDS,
    _UNIT_SHORT_FIELDS,
    CITY_INFO,
    CITY_REMOVE,
    CITY_SHORT_INFO,
    MAP_INFO,
    MAX_EXTRA_TYPES,
    TILE_INFO,
    TILE_KNOWN_SEEN,
    TILE_KNOWN_UNSEEN,
    TILE_UNKNOWN,
    UNIT_REMOVE,
    UNIT_SHORT_INFO,
    PlayerViewCache,
)
from civharness.proto.dataio import Writer


def _bv(nbits, set_bits):
    w = Writer()
    w.put_bitvector([i in set_bits for i in range(nbits)], nbits)
    return w


def tile_info(
    tile,
    *,
    continent=None,
    known=None,
    owner=None,
    terrain=None,
    resource=None,
    extras=None,
):
    """Encode a TILE_INFO body with only the given fields set."""
    set_bits = set()
    if continent is not None:
        set_bits.add(0)
    if known is not None:
        set_bits.add(1)
    if owner is not None:
        set_bits.add(2)
    if terrain is not None:
        set_bits.add(5)
    if resource is not None:
        set_bits.add(6)
    if extras is not None:
        set_bits.add(7)
    w = _bv(len(_TILE_FIELDS), set_bits)
    w.put_sint32(tile)
    if continent is not None:
        w.put_sint16(continent)
    if known is not None:
        w.put_uint8(known)
    if owner is not None:
        w.put_uint16(owner)
    if terrain is not None:
        w.put_uint8(terrain)
    if resource is not None:
        w.put_uint8(resource)
    if extras is not None:
        w.put_bitvector([i in extras for i in range(MAX_EXTRA_TYPES)], MAX_EXTRA_TYPES)
    return w.bytes()


def city_short(
    cid,
    *,
    tile=None,
    owner=None,
    size=None,
    name=None,
    occupied=False,
    improvements=None,
):
    set_bits = set()
    if tile is not None:
        set_bits.add(0)
    if owner is not None:
        set_bits.add(1)
    if size is not None:
        set_bits.add(3)
    if occupied:
        set_bits.add(6)  # BOOL folds into the bitvector
    if improvements is not None:
        set_bits.add(11)
    if name is not None:
        set_bits.add(12)
    w = _bv(len(_CITY_SHORT_FIELDS), set_bits)
    w.put_uint32(cid)
    if tile is not None:
        w.put_sint32(tile)
    if owner is not None:
        w.put_uint16(owner)
    if size is not None:
        w.put_uint8(size)
    if improvements is not None:
        w.put_bitvector([i in improvements for i in range(200)], 200)
    if name is not None:
        w.put_estring(name)
    return w.bytes()


def unit_short(uid, *, owner=None, tile=None, utype=None, hp=None, occupied=False):
    set_bits = set()
    if owner is not None:
        set_bits.add(0)
    if tile is not None:
        set_bits.add(1)
    if utype is not None:
        set_bits.add(3)
    if occupied:
        set_bits.add(5)  # BOOL fold
    if hp is not None:
        set_bits.add(7)
    w = _bv(len(_UNIT_SHORT_FIELDS), set_bits)
    w.put_uint32(uid)
    if owner is not None:
        w.put_uint16(owner)
    if tile is not None:
        w.put_sint32(tile)
    if utype is not None:
        w.put_uint16(utype)
    if hp is not None:
        w.put_uint8(hp)
    return w.bytes()


def remove_body(entity_id):
    w = _bv(1, {0})
    w.put_uint32(entity_id)
    return w.bytes()


def map_info_body(xsize, ysize):
    w = _bv(6, {0, 1})
    w.put_uint16(xsize)
    w.put_uint16(ysize)
    return w.bytes()


def test_tile_delta_carries_unsent_fields_forward():
    c = PlayerViewCache()
    c.feed(TILE_INFO, tile_info(7, known=TILE_KNOWN_SEEN, terrain=3, owner=2))
    assert c.tiles[7]["terrain"] == 3 and c.tiles[7]["owner"] == 2
    # second packet for the same tile: only `known` set -> terrain/owner kept
    c.feed(TILE_INFO, tile_info(7, known=TILE_KNOWN_UNSEEN))
    assert c.tiles[7]["known"] == TILE_KNOWN_UNSEEN
    assert c.tiles[7]["terrain"] == 3
    assert c.tiles[7]["owner"] == 2


def test_unknown_tile_is_absent():
    c = PlayerViewCache()
    c.feed(TILE_INFO, tile_info(7, known=TILE_KNOWN_SEEN, terrain=3))
    assert c.known_state(99) == TILE_UNKNOWN
    assert 99 not in c.tiles


def test_seen_to_remembered_transition_partitions_views():
    c = PlayerViewCache()
    c.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_SEEN, terrain=1))
    assert 5 in c.visible_tiles() and 5 not in c.remembered_tiles()
    c.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_UNSEEN))
    assert 5 not in c.visible_tiles() and 5 in c.remembered_tiles()


def test_city_short_boolfold_and_delta():
    c = PlayerViewCache()
    c.feed(
        CITY_SHORT_INFO,
        city_short(30, tile=12, owner=1, size=4, name="Athens", occupied=True),
    )
    rec = c.cities[30]
    assert rec["name"] == "Athens" and rec["size"] == 4
    assert rec["occupied"] is True and rec["_form"] == "short"
    # next packet: size changes, occupied bit NOT set -> False (bool folds
    # carry the current value in every packet, they never inherit)
    c.feed(CITY_SHORT_INFO, city_short(30, size=5))
    assert c.cities[30]["size"] == 5
    assert c.cities[30]["name"] == "Athens"  # delta carry
    assert c.cities[30]["occupied"] is False  # bool fold semantics


def test_city_remove_deletes_and_wipes_delta_baseline():
    c = PlayerViewCache()
    c.feed(CITY_SHORT_INFO, city_short(30, tile=12, owner=1, size=4, name="Athens"))
    c.feed(CITY_REMOVE, remove_body(30))
    assert 30 not in c.cities
    # a later re-arrival starts from the ZERO packet, not the old baseline
    c.feed(CITY_SHORT_INFO, city_short(30, size=2))
    assert c.cities[30]["name"] == ""  # zero string, not "Athens"
    assert c.cities[30]["size"] == 2


def test_unit_remove_is_the_out_of_sight_signal():
    c = PlayerViewCache()
    c.feed(UNIT_SHORT_INFO, unit_short(101, owner=1, tile=9, utype=2, hp=10))
    assert 101 in c.units
    c.feed(UNIT_REMOVE, remove_body(101))
    assert 101 not in c.units  # hidden enemy unit leaves no trace


def test_full_and_short_forms_cancel_each_other():
    c = PlayerViewCache()
    c.feed(CITY_SHORT_INFO, city_short(30, tile=12, owner=1, size=4, name="Athens"))
    assert c.cities[30]["_form"] == "short"
    # CITY_INFO for the same id must not inherit the short baseline: even a
    # zero-bit full packet rebuilds from the zero packet
    from civharness.playerview import _CITY_FULL_FIELDS

    w = _bv(len(_CITY_FULL_FIELDS), set())
    w.put_uint32(30)
    c.feed(CITY_INFO, w.bytes())
    assert c.cities[30]["_form"] == "full"
    assert c.cities[30]["name"] == ""  # not "Athens" — cancel wiped it


def test_reconnect_reset_clears_everything():
    c = PlayerViewCache()
    c.feed(MAP_INFO, map_info_body(30, 20))
    c.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_SEEN, terrain=1))
    c.feed(UNIT_SHORT_INFO, unit_short(101, owner=1, tile=9, utype=2, hp=10))
    c.reset()
    assert not c.tiles and not c.units and not c.map_info


def test_snapshot_freezes_decision_boundary_view():
    c = PlayerViewCache()
    c.feed(MAP_INFO, map_info_body(30, 20))
    c.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_SEEN, terrain=1))
    snap = c.snapshot()
    # later packets must not leak into the frozen snapshot
    c.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_UNSEEN))
    c.feed(UNIT_SHORT_INFO, unit_short(101, owner=1, tile=9, utype=2, hp=10))
    assert snap.tiles[5]["known"] == TILE_KNOWN_SEEN
    assert 101 not in snap.units
    assert c.tiles[5]["known"] == TILE_KNOWN_UNSEEN


def test_xy_uses_map_info():
    c = PlayerViewCache()
    assert c.xy(7) is None  # no MAP_INFO yet -> unknown geometry, not a guess
    c.feed(MAP_INFO, map_info_body(30, 20))
    assert c.xy(65) == (5, 2)


def test_player_view_builder_hides_what_was_never_sent():
    from types import SimpleNamespace

    from civharness.observe_view import player_view

    rules = SimpleNamespace(
        terrains={"Grassland": 1, "Ocean": 2},
        extras={"Road": 0},
        improvements={"Palace": 0},
        units={"Warriors": 2},
    )
    c = PlayerViewCache()
    c.feed(MAP_INFO, map_info_body(30, 20))
    c.feed(TILE_INFO, tile_info(65, known=TILE_KNOWN_SEEN, terrain=1, extras={0}))
    c.feed(TILE_INFO, tile_info(66, known=TILE_KNOWN_UNSEEN, terrain=2))
    c.feed(UNIT_SHORT_INFO, unit_short(101, owner=1, tile=65, utype=2, hp=10))
    v = player_view(c, rules, my_player_id=0)
    assert v["visibility"] == "player_visible"
    tiles = v["map"]["tiles"]
    assert tiles[65]["terrain"] == "Grassland"
    assert tiles[65]["extras"] == ["Road"]
    assert tiles[66]["known_label"] == "remembered"
    assert 67 not in tiles  # never sent -> absent, not defaulted
    (u,) = v["units"]
    assert u["type"] == "Warriors" and u["mine"] is False
    assert u["moves"] is None  # short form: moves genuinely unknown


def test_two_caches_are_independent_views():
    """Two clients on the same game hold different views: what one learns
    never appears in the other (per-connection state, no globals)."""
    a, b = PlayerViewCache(), PlayerViewCache()
    a.feed(TILE_INFO, tile_info(5, known=TILE_KNOWN_SEEN, terrain=1))
    b.feed(TILE_INFO, tile_info(9, known=TILE_KNOWN_SEEN, terrain=2))
    assert 5 in a.tiles and 5 not in b.tiles
    assert 9 in b.tiles and 9 not in a.tiles
