"""Per-client player-view cache — genuine fog of war from the wire.

Freeciv's server already computes per-player visibility: each connected
client receives only what its player may know, via TILE_INFO / CITY_INFO /
CITY_SHORT_INFO / UNIT_INFO / UNIT_SHORT_INFO and the matching REMOVE
packets.  This module decodes those server->client packets incrementally, so
a ``player_visible`` observation can be built from what the client actually
received — never by filtering the omniscient autosave.

Wire facts verified against freeciv-3.2.5 sources (common/networking/
packets.def field order; common/packets_gen.c generated delta receive code;
client/packhand.c semantics):

* Every one of these packets is delta-encoded per connection: a leading
  LSB-first field bitvector, then the key field(s), then only the fields
  whose bit is set.  Unsent fields keep the value from the *previous packet
  of the same type with the same key* (genhash keyed on the key field) —
  that is exactly the per-key dict this cache keeps.  A first-seen key
  starts from the all-zero packet, not from garbage.
* BOOL fields are folded into the bitvector itself (the bit IS the value;
  no byte in the body).  Which fields fold is fixed by the generated code
  and mirrored in the field tables below.
* ``known``: 0 = TILE_UNKNOWN, 1 = TILE_KNOWN_UNSEEN (remembered),
  2 = TILE_KNOWN_SEEN (currently visible) — common/tile.h.
* tile ``resource`` is only meaningful when != MAX_EXTRA_TYPES (250) AND the
  corresponding extra bit is present in ``extras`` (packhand.c keeps the
  resource pointer NULL otherwise; we store the raw id plus a validity flag).
* ``owner`` fields are uint16 player numbers; MAX_UINT16 (65535) never
  arrives for a valid player — player_by_number() returning NULL in the
  client corresponds to "no owner" here (owner id kept verbatim, name
  resolution happens at observation-build time).
* CITY_INFO cancels CITY_SHORT_INFO for the same id and vice versa (a city
  entering/leaving full view replaces the other form); both REMOVE packets
  drop the entity outright.  The server sends UNIT_REMOVE when a unit goes
  out of sight (server/unittools.c unit_goes_out_of_sight) and CITY_SHORT_INFO
  refreshes the remembered "dumb" city (server/maphand.c reality_check_city),
  so remove/short-info arrival IS the visible->remembered/hidden transition;
  no extra inference is needed.
* Variant selection: the pinned 3.2.5 server and this client both advertise
  the ``hap2clnt`` capability (NETWORK_CAPSTRING), so CITY_INFO arrives as
  variant 101 (54 bits, anarchy/rapture at bits 42/43).  The other four fog
  packets have a single variant (100).

Everything decoded is stored typed-but-raw (ids, not names): name resolution
and Observation assembly live in civharness.observe / the caller, keeping
this module a pure protocol mirror that is unit-testable from byte strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from civharness.proto.dataio import Reader

# --- packet type ids (server -> client) ------------------------------------
TILE_INFO = 15
MAP_INFO = 17
CITY_REMOVE = 30
CITY_INFO = 31
CITY_SHORT_INFO = 32
PLAYER_REMOVE = 50
UNIT_REMOVE = 62
UNIT_INFO = 63
UNIT_SHORT_INFO = 64

# --- protocol constants (common/fc_types.h, common/tile.h) -----------------
MAX_EXTRA_TYPES = 250  # bv_extras size; also the "no resource" id
MAX_NUM_BUILDINGS = 200  # bv_imprs size (B_LAST)
O_LAST = 6  # output types: food/shield/trade/gold/lux/sci
FEELING_LAST = 6
SP_MAX = 20
CITYO_LAST = 3  # bv_city_options size

TILE_UNKNOWN = 0
TILE_KNOWN_UNSEEN = 1  # remembered, not currently seen
TILE_KNOWN_SEEN = 2  # currently visible


# --- delta field tables -----------------------------------------------------
# One row per non-key field, in bitvector order:  (name, getter, fold/array).
#   getter: a Reader method name, or a callable(Reader) for compound types.
#   kind:   "" plain; "bool" folded into the bitvector; "arrN" fixed array of
#           N scalars; callables handle everything else.


def _bv_extras(r: Reader):
    return tuple(i for i, b in enumerate(r.get_bitvector(MAX_EXTRA_TYPES)) if b)


def _bv_imprs(r: Reader):
    return tuple(i for i, b in enumerate(r.get_bitvector(MAX_NUM_BUILDINGS)) if b)


def _bv_city_options(r: Reader):
    return tuple(i for i, b in enumerate(r.get_bitvector(CITYO_LAST)) if b)


def _worklist(r: Reader):
    n = r.get_uint8()
    return tuple((r.get_uint8(), r.get_uint8()) for _ in range(n))


def _unit_orders_array(length):
    def read(r: Reader):
        out = []
        for _ in range(length):
            out.append(
                {
                    "order": r.get_uint8(),
                    "activity": r.get_uint8(),
                    "target": r.get_sint32(),
                    "sub_target": r.get_sint16(),
                    "action": r.get_uint8(),
                    "dir": r.get_sint8(),
                }
            )
        return tuple(out)

    return read


def _arr(getter, n):
    def read(r: Reader):
        return tuple(getattr(r, getter)() for _ in range(n))

    return read


# PACKET_TILE_INFO = 15 — key: tile (sint32); 12 delta bits.
_TILE_FIELDS = (
    ("continent", "get_sint16", ""),
    ("known", "get_uint8", ""),
    ("owner", "get_uint16", ""),
    ("extras_owner", "get_uint16", ""),
    ("worked", "get_uint32", ""),
    ("terrain", "get_uint8", ""),
    ("resource", "get_uint8", ""),
    ("extras", _bv_extras, ""),
    ("placing", "get_sint8", ""),
    ("place_turn", "get_sint16", ""),
    ("spec_sprite", "get_string", ""),
    ("label", "get_string", ""),
)

# PACKET_CITY_SHORT_INFO = 32 — key: id (uint32); 13 delta bits.
_CITY_SHORT_FIELDS = (
    ("tile", "get_sint32", ""),
    ("owner", "get_uint16", ""),
    ("original", "get_uint16", ""),
    ("size", "get_uint8", ""),
    ("style", "get_uint8", ""),
    ("capital", "get_uint8", ""),
    ("occupied", None, "bool"),
    ("walls", "get_uint8", ""),
    ("happy", None, "bool"),
    ("unhappy", None, "bool"),
    ("city_image", "get_sint8", ""),
    ("improvements", _bv_imprs, ""),
    ("name", "get_estring", ""),
)

# PACKET_CITY_INFO = 31, variant 101 (hap2clnt) — key: id (uint32); 54 bits.
_CITY_FULL_FIELDS = (
    ("tile", "get_sint32", ""),
    ("owner", "get_uint16", ""),
    ("original", "get_uint16", ""),
    ("size", "get_uint8", ""),
    ("city_radius_sq", "get_uint8", ""),
    ("style", "get_uint8", ""),
    ("capital", "get_uint8", ""),
    ("ppl_happy", _arr("get_uint8", FEELING_LAST), ""),
    ("ppl_content", _arr("get_uint8", FEELING_LAST), ""),
    ("ppl_unhappy", _arr("get_uint8", FEELING_LAST), ""),
    ("ppl_angry", _arr("get_uint8", FEELING_LAST), ""),
    ("specialists_size", "get_uint8", ""),
    ("specialists", None, "specialists"),  # length = specialists_size
    ("history", "get_uint32", ""),
    ("culture", "get_uint32", ""),
    ("buy_cost", "get_uint32", ""),
    ("surplus", _arr("get_sint16", O_LAST), ""),
    ("waste", _arr("get_uint16", O_LAST), ""),
    ("unhappy_penalty", _arr("get_sint16", O_LAST), ""),
    ("prod", _arr("get_uint16", O_LAST), ""),
    ("citizen_base", _arr("get_sint16", O_LAST), ""),
    ("usage", _arr("get_sint16", O_LAST), ""),
    ("food_stock", "get_sint16", ""),
    ("shield_stock", "get_uint16", ""),
    ("trade_route_count", "get_uint8", ""),
    ("pollution", "get_uint16", ""),
    ("illness_trade", "get_uint16", ""),
    ("production_kind", "get_uint8", ""),
    ("production_value", "get_uint8", ""),
    ("turn_founded", "get_sint16", ""),
    ("turn_last_built", "get_sint16", ""),
    ("changed_from_kind", "get_uint8", ""),
    ("changed_from_value", "get_uint8", ""),
    ("before_change_shields", "get_uint16", ""),
    ("disbanded_shields", "get_uint16", ""),
    ("caravan_shields", "get_uint16", ""),
    ("last_turns_shield_surplus", "get_uint16", ""),
    ("airlift", "get_uint8", ""),
    ("did_buy", None, "bool"),
    ("did_sell", None, "bool"),
    ("was_happy", None, "bool"),
    ("had_famine", None, "bool"),
    ("anarchy", "get_uint16", ""),  # variant 101 (hap2clnt)
    ("rapture", "get_uint16", ""),  # variant 101 (hap2clnt)
    ("diplomat_investigate", None, "bool"),
    ("walls", "get_uint8", ""),
    ("city_image", "get_sint8", ""),
    ("steal", "get_uint16", ""),
    ("worklist", _worklist, ""),
    ("improvements", _bv_imprs, ""),
    ("city_options", _bv_city_options, ""),
    ("wl_cb", "get_uint8", ""),
    ("acquire_type", "get_uint8", ""),
    ("name", "get_estring", ""),
)

# PACKET_UNIT_SHORT_INFO = 64 — key: id (uint32); 13 delta bits.
_UNIT_SHORT_FIELDS = (
    ("owner", "get_uint16", ""),
    ("tile", "get_sint32", ""),
    ("facing", "get_sint8", ""),
    ("type", "get_uint16", ""),
    ("veteran", "get_uint8", ""),
    ("occupied", None, "bool"),
    ("transported", None, "bool"),
    ("hp", "get_uint8", ""),
    ("activity", "get_uint8", ""),
    ("activity_tgt", "get_sint8", ""),
    ("transported_by", "get_uint32", ""),
    ("packet_use", "get_uint8", ""),
    ("info_city_id", "get_uint32", ""),
)

# PACKET_UNIT_INFO = 63 — key: id (uint32); 37 delta bits.
_UNIT_FULL_FIELDS = (
    ("owner", "get_uint16", ""),
    ("nationality", "get_uint16", ""),
    ("tile", "get_sint32", ""),
    ("facing", "get_sint8", ""),
    ("homecity", "get_uint32", ""),
    ("upkeep", _arr("get_uint8", O_LAST), ""),
    ("veteran", "get_uint8", ""),
    ("ssa_controller", "get_uint8", ""),
    ("paradropped", None, "bool"),
    ("occupied", None, "bool"),
    ("transported", None, "bool"),
    ("done_moving", None, "bool"),
    ("stay", None, "bool"),
    ("birth_turn", "get_sint16", ""),
    ("current_form_turn", "get_sint16", ""),
    ("type", "get_uint16", ""),
    ("transported_by", "get_uint32", ""),
    ("carrying", "get_sint8", ""),
    ("movesleft", "get_uint32", ""),
    ("hp", "get_uint8", ""),
    ("fuel", "get_uint8", ""),
    ("activity_count", "get_uint16", ""),
    ("changed_from_count", "get_uint16", ""),
    ("goto_tile", "get_sint32", ""),
    ("activity", "get_uint8", ""),
    ("activity_tgt", "get_sint8", ""),
    ("changed_from", "get_uint8", ""),
    ("changed_from_tgt", "get_sint8", ""),
    ("battlegroup", "get_sint8", ""),
    ("has_orders", None, "bool"),
    ("orders_length", "get_uint16", ""),
    ("orders_index", "get_uint16", ""),
    ("orders_repeat", None, "bool"),
    ("orders_vigilant", None, "bool"),
    ("orders", None, "orders"),  # length = orders_length
    ("action_decision_want", "get_uint8", ""),
    ("action_decision_tile", "get_sint32", ""),
)


def _zero_value(name, getter, kind):
    if kind == "bool":
        return False
    if kind in ("specialists", "orders"):
        return ()
    if callable(getter):
        return ()
    if getter in ("get_string", "get_estring"):
        return ""
    return 0


def _decode_delta(
    body: bytes,
    fields,
    key_getter: str,
    old: dict | None,
    *,
    dynamic: dict | None = None,
) -> dict:
    """Decode one delta packet: bitvector, key, then set fields only.
    ``old`` is the previous decoded dict for this key (None -> zero packet).
    ``dynamic`` maps kind -> callable(Reader, out_dict) for length-dependent
    arrays that need an already-decoded count field."""
    r = Reader(body)
    bits = r.get_bitvector(len(fields))
    key = getattr(r, key_getter)()
    if old is None:
        out = {name: _zero_value(name, getter, kind) for name, getter, kind in fields}
    else:
        out = dict(old)
    for i, (name, getter, kind) in enumerate(fields):
        if kind == "bool":
            out[name] = bits[i]
            continue
        if not bits[i]:
            continue
        if kind in ("specialists", "orders"):
            out[name] = dynamic[kind](r, out)
        elif callable(getter):
            out[name] = getter(r)
        else:
            out[name] = getattr(r, getter)()
    out["_key"] = key
    return out


def decode_tile_info(body: bytes, old: dict | None) -> dict:
    return _decode_delta(body, _TILE_FIELDS, "get_sint32", old)


def decode_city_short_info(body: bytes, old: dict | None) -> dict:
    return _decode_delta(body, _CITY_SHORT_FIELDS, "get_uint32", old)


def decode_city_info(body: bytes, old: dict | None) -> dict:
    return _decode_delta(
        body,
        _CITY_FULL_FIELDS,
        "get_uint32",
        old,
        dynamic={
            "specialists": lambda r, out: tuple(
                r.get_uint8() for _ in range(out["specialists_size"])
            )
        },
    )


def decode_unit_short_info(body: bytes, old: dict | None) -> dict:
    return _decode_delta(body, _UNIT_SHORT_FIELDS, "get_uint32", old)


def decode_unit_info(body: bytes, old: dict | None) -> dict:
    return _decode_delta(
        body,
        _UNIT_FULL_FIELDS,
        "get_uint32",
        old,
        dynamic={"orders": lambda r, out: _unit_orders_array(out["orders_length"])(r)},
    )


def decode_remove(body: bytes) -> int:
    """CITY_REMOVE / UNIT_REMOVE: one delta bit + the id field.  The single
    field always changes, so the bit is always set on the wire; decode both
    ways defensively (an unset bit means 'same id as last remove', which the
    caller's delta dict supplies)."""
    r = Reader(body)
    bits = r.get_bitvector(1)
    if bits[0]:
        return r.get_uint32()
    return -1  # unchanged-from-last; caller resolves via its own last-id


def decode_map_info(body: bytes, old: dict | None) -> dict:
    fields = (
        ("xsize", "get_uint16", ""),
        ("ysize", "get_uint16", ""),
        ("topology_id", "get_uint8", ""),
        ("wrap_id", "get_uint8", ""),
        ("north_latitude", "get_sint16", ""),
        ("south_latitude", "get_sint16", ""),
    )
    # MAP_INFO has no key field: bitvector then set fields.
    r = Reader(body)
    bits = r.get_bitvector(len(fields))
    out = dict(old) if old else {n: 0 for n, _g, _k in fields}
    for i, (name, getter, _kind) in enumerate(fields):
        if bits[i]:
            out[name] = getattr(r, getter)()
    return out


@dataclass
class PlayerViewCache:
    """The incremental state one client's connection has actually received.

    Mirrors the server's per-connection send model:
      * ``tiles[tile_index]``    — last TILE_INFO for that tile;
      * ``cities[city_id]``      — last CITY_INFO/CITY_SHORT_INFO, tagged
                                   ``_form: "full"|"short"`` (the two forms
                                   cancel each other, per packets.def);
      * ``units[unit_id]``       — last UNIT_INFO/UNIT_SHORT_INFO, same
                                   ``_form`` tag and mutual cancellation;
      * REMOVE packets delete the entry (visible -> gone/remembered-empty).

    The delta baseline is per packet TYPE and key, exactly like the engine's
    genhash: a CITY_INFO after a CITY_SHORT_INFO for the same city starts
    from the zero packet (the cancel wipes the received-hash entry), not from
    the short form's fields.
    """

    tiles: dict[int, dict] = field(default_factory=dict)
    cities: dict[int, dict] = field(default_factory=dict)
    units: dict[int, dict] = field(default_factory=dict)
    map_info: dict = field(default_factory=dict)
    # separate delta baselines per packet type (cancel wipes these, not the
    # user-facing dicts above)
    _tile_old: dict[int, dict] = field(default_factory=dict)
    _city_full_old: dict[int, dict] = field(default_factory=dict)
    _city_short_old: dict[int, dict] = field(default_factory=dict)
    _unit_full_old: dict[int, dict] = field(default_factory=dict)
    _unit_short_old: dict[int, dict] = field(default_factory=dict)
    _last_city_remove: int = -1
    _last_unit_remove: int = -1

    # -- packet ingestion ----------------------------------------------------
    def handles(self, ptype: int) -> bool:
        return ptype in (
            TILE_INFO,
            MAP_INFO,
            CITY_INFO,
            CITY_SHORT_INFO,
            CITY_REMOVE,
            UNIT_INFO,
            UNIT_SHORT_INFO,
            UNIT_REMOVE,
        )

    def feed(self, ptype: int, body: bytes) -> None:
        if ptype == TILE_INFO:
            peek = Reader(body)
            peek.get_bitvector(len(_TILE_FIELDS))
            key = peek.get_sint32()
            rec = decode_tile_info(body, self._tile_old.get(key))
            self._tile_old[key] = rec
            self.tiles[key] = rec
        elif ptype == MAP_INFO:
            self.map_info = decode_map_info(body, self.map_info or None)
        elif ptype == CITY_INFO:
            peek = Reader(body)
            peek.get_bitvector(len(_CITY_FULL_FIELDS))
            key = peek.get_uint32()
            rec = decode_city_info(body, self._city_full_old.get(key))
            rec["_form"] = "full"
            self._city_full_old[key] = rec
            self._city_short_old.pop(key, None)  # cancel(CITY_SHORT_INFO)
            self.cities[key] = rec
        elif ptype == CITY_SHORT_INFO:
            peek = Reader(body)
            peek.get_bitvector(len(_CITY_SHORT_FIELDS))
            key = peek.get_uint32()
            rec = decode_city_short_info(body, self._city_short_old.get(key))
            rec["_form"] = "short"
            self._city_short_old[key] = rec
            self._city_full_old.pop(key, None)  # cancel(CITY_INFO)
            self.cities[key] = rec
        elif ptype == CITY_REMOVE:
            cid = decode_remove(body)
            if cid == -1:
                cid = self._last_city_remove
            self._last_city_remove = cid
            self.cities.pop(cid, None)
            self._city_full_old.pop(cid, None)  # cancel(CITY_INFO)
            self._city_short_old.pop(cid, None)  # cancel(CITY_SHORT_INFO)
        elif ptype == UNIT_INFO:
            peek = Reader(body)
            peek.get_bitvector(len(_UNIT_FULL_FIELDS))
            key = peek.get_uint32()
            rec = decode_unit_info(body, self._unit_full_old.get(key))
            rec["_form"] = "full"
            self._unit_full_old[key] = rec
            self._unit_short_old.pop(key, None)  # cancel(UNIT_SHORT_INFO)
            self.units[key] = rec
        elif ptype == UNIT_SHORT_INFO:
            peek = Reader(body)
            peek.get_bitvector(len(_UNIT_SHORT_FIELDS))
            key = peek.get_uint32()
            rec = decode_unit_short_info(body, self._unit_short_old.get(key))
            rec["_form"] = "short"
            self._unit_short_old[key] = rec
            self._unit_full_old.pop(key, None)  # cancel(UNIT_INFO)
            self.units[key] = rec
        elif ptype == UNIT_REMOVE:
            uid = decode_remove(body)
            if uid == -1:
                uid = self._last_unit_remove
            self._last_unit_remove = uid
            self.units.pop(uid, None)
            self._unit_full_old.pop(uid, None)  # cancel(UNIT_INFO)
            self._unit_short_old.pop(uid, None)  # cancel(UNIT_SHORT_INFO)

    def reset(self) -> None:
        """Reconnect/resync: the server resends full state on a new
        connection, whose delta stream starts from zero packets again."""
        self.tiles.clear()
        self.cities.clear()
        self.units.clear()
        self.map_info = {}
        self._tile_old.clear()
        self._city_full_old.clear()
        self._city_short_old.clear()
        self._unit_full_old.clear()
        self._unit_short_old.clear()
        self._last_city_remove = -1
        self._last_unit_remove = -1

    # -- views -----------------------------------------------------------
    def xy(self, tile_index: int) -> tuple[int, int] | None:
        xsize = self.map_info.get("xsize")
        if not xsize:
            return None
        return tile_index % xsize, tile_index // xsize

    def known_state(self, tile_index: int) -> int:
        rec = self.tiles.get(tile_index)
        return rec["known"] if rec is not None else TILE_UNKNOWN

    def visible_tiles(self) -> dict[int, dict]:
        return {k: v for k, v in self.tiles.items() if v["known"] == TILE_KNOWN_SEEN}

    def remembered_tiles(self) -> dict[int, dict]:
        return {k: v for k, v in self.tiles.items() if v["known"] == TILE_KNOWN_UNSEEN}

    def snapshot(self) -> PlayerViewCache:
        """Deep-enough copy for a frozen decision-boundary view: the per-key
        record dicts are immutable by convention (feed() replaces, never
        mutates them), so copying the outer maps freezes the view."""
        c = PlayerViewCache()
        c.tiles = dict(self.tiles)
        c.cities = dict(self.cities)
        c.units = dict(self.units)
        c.map_info = dict(self.map_info)
        return c
