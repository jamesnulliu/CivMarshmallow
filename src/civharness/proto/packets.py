"""Packet ids and (de)serializers for the handful of packets a controlling
client needs — grounded in freeciv-3.2.5 `packets.def` / `packets_gen.c`.

The delta protocol (the crux): every cs packet except those flagged
`no-delta` is sent as a leading bitvector (one bit per field, LSB-first,
ceil(nfields/8) bytes) followed by only the fields that differ from the
*previous* packet of that type on this connection. Both sides start from an
all-zero baseline and update it after every packet, so `DeltaEncoder._old`
is a mirror of the server's receive-side delta cache — kept in lockstep by
TCP's ordered, reliable delivery. Get that mirror wrong and every later
packet of that type desynchronizes.
"""

from civharness.proto.dataio import Reader, Writer

# --- packet type ids -------------------------------------------------------
PROCESSING_STARTED = 0
PROCESSING_FINISHED = 1
SERVER_JOIN_REQ = 4
SERVER_JOIN_REPLY = 5
ENDGAME_REPORT = 12  # sent once when the game ends (S_S_OVER)
CHAT_MSG = 25
CHAT_MSG_REQ = 26
CITY_SELL = 33
CITY_BUY = 34
CITY_CHANGE = 35
CITY_WORKLIST = 36
CITY_MAKE_SPECIALIST = 37  # pull a worker off a tile -> specialist
CITY_MAKE_WORKER = 38  # put a citizen to work a tile
CITY_CHANGE_SPECIALIST = 39
PLAYER_RATES = 53  # tax / luxury / science percents
PLAYER_CHANGE_GOVERNMENT = 54
PLAYER_PHASE_DONE = 52
PLAYER_RESEARCH = 55
PLAYER_TECH_GOAL = 56
UNIT_ORDERS = 73
UNIT_DO_ACTION = 84  # cs: perform one action by id (delta; single packet executes)
UNIT_GET_ACTIONS = 87  # cs: ask what an actor may do to a target
CONN_PING = 88
CONN_PONG = 89
UNIT_ACTIONS = 90  # sc: reply — legal-action probabilities (decode)
DIPLOMACY_CANCEL_PACT = 105  # declare war / step a treaty down one level
END_PHASE = 125
START_PHASE = 126
NEW_YEAR = 127
BEGIN_TURN = 128
END_TURN = 129

# --- universals kinds (enum universals_n, common/fc_types.h) ---------------
VUT_IMPROVEMENT = 3  # city building / wonder
VUT_UTYPE = 6  # unit

# --- diplomacy clause types (common/diptreaty.h) ---------------------------
CLAUSE_CEASEFIRE = 5
CLAUSE_PEACE = 6

# --- capability / version (gen_headers/version_gen.h, fc_version) ----------
NETWORK_CAPSTRING = "+Freeciv-3.2-network ownernull16 unignoresync tu32 hap2clnt"
MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = 3, 2, 5

# --- delta field specs for the cs packets we send --------------------------
# Ordered list of dataio types per field; order defines the bit index.
_SPECS: dict[int, list[str]] = {
    CITY_CHANGE: ["uint32", "uint8", "uint8"],  # city_id, prod_kind, prod_value
    CITY_WORKLIST: ["uint32", "worklist"],  # city_id, worklist
    CITY_BUY: ["uint32"],  # city_id
    CITY_SELL: ["uint32", "uint8"],  # city_id, build_id
    CITY_CHANGE_SPECIALIST: ["uint32", "uint8", "uint8"],  # city_id, from, to
    CITY_MAKE_SPECIALIST: ["uint32", "sint32"],  # city_id, tile_id
    CITY_MAKE_WORKER: ["uint32", "sint32"],  # city_id, tile_id
    PLAYER_RATES: ["uint8", "uint8", "uint8"],  # tax, luxury, science (percent)
    PLAYER_CHANGE_GOVERNMENT: ["sint8"],  # government id
    PLAYER_RESEARCH: ["uint16"],  # tech
    PLAYER_TECH_GOAL: ["uint16"],  # tech
    PLAYER_PHASE_DONE: ["sint16"],  # turn
    # actor, target_id, sub_target, name, action_type (PACKET_UNIT_DO_ACTION=84).
    # target_id is a unit id (self/unit actions), a city id, or a tile index
    # y*xsize+x (tile actions); sub_target is an extra/building id (0 = none,
    # -1 = let the server pick for a flexible action). name is "" except for
    # actions that carry a string (e.g. founding a named city).
    UNIT_DO_ACTION: ["uint32", "sint32", "sint16", "estring", "uint8"],
    # actor, target_unit, target_tile, target_extra, request_kind
    UNIT_GET_ACTIONS: ["uint32", "uint32", "sint32", "sint8", "uint8"],
    DIPLOMACY_CANCEL_PACT: ["uint16", "uint8"],  # other_player_id, clause
    CHAT_MSG_REQ: ["estring"],  # message
}


def _zero(dio_type: str):
    if dio_type in ("string", "estring"):
        return ""
    if dio_type == "worklist":
        return []
    return 0


class DeltaEncoder:
    """Per-connection delta state for the packets this client sends. One
    instance per connection; `_old[ptype]` mirrors the server's receive cache
    for that packet type."""

    def __init__(self):
        self._old: dict[int, list] = {}

    def encode(self, ptype: int, values: list) -> bytes:
        spec = _SPECS[ptype]
        assert len(values) == len(spec), (ptype, values)
        old = self._old.get(ptype)
        if old is None:
            old = [_zero(t) for t in spec]
        bits = [values[i] != old[i] for i in range(len(spec))]
        w = Writer()
        w.put_bitvector(bits, len(spec))
        for i, on in enumerate(bits):
            if on:
                w.put_type(spec[i], values[i])
        self._old[ptype] = list(values)
        return w.bytes()

    def encode_unit_orders(
        self,
        unit_id: int,
        src_tile: int,
        orders: list[dict],
        dest_tile: int,
        repeat: bool = False,
        vigilant: bool = False,
    ) -> bytes:
        """PACKET_UNIT_ORDERS (7 delta fields; bools 'repeat'/'vigilant' are
        *folded* into the bitvector — the bit is the value, no field byte).
        Each order dict: order, activity, target, sub_target, action, dir
        (dio_put_unit_order_raw)."""
        length = len(orders)
        old = self._old.get(UNIT_ORDERS) or {
            "unit_id": 0,
            "src_tile": 0,
            "length": 0,
            "orders": [],
            "dest_tile": 0,
        }
        bits = [
            unit_id != old["unit_id"],
            src_tile != old["src_tile"],
            length != old["length"],
            repeat,  # folded boolean
            vigilant,  # folded boolean
            length != old["length"] or orders != old["orders"],
            dest_tile != old["dest_tile"],
        ]
        w = Writer()
        w.put_bitvector(bits, 7)
        if bits[0]:
            w.put_uint32(unit_id)
        if bits[1]:
            w.put_sint32(src_tile)
        if bits[2]:
            w.put_uint16(length)
        if bits[5]:
            for o in orders:
                w.put_uint8(o["order"])
                w.put_uint8(o["activity"])
                w.put_sint32(o["target"])
                w.put_sint16(o["sub_target"])
                w.put_uint8(o["action"])
                w.put_sint8(o["dir"])
        if bits[6]:
            w.put_sint32(dest_tile)
        self._old[UNIT_ORDERS] = {
            "unit_id": unit_id,
            "src_tile": src_tile,
            "length": length,
            "orders": list(orders),
            "dest_tile": dest_tile,
        }
        return w.bytes()


def server_join_req(username: str) -> bytes:
    """PACKET_SERVER_JOIN_REQ (no-delta): all fields, in order."""
    w = Writer()
    w.put_string(username)
    w.put_string(NETWORK_CAPSTRING)
    w.put_string("")  # version_label
    w.put_uint32(MAJOR_VERSION)
    w.put_uint32(MINOR_VERSION)
    w.put_uint32(PATCH_VERSION)
    return w.bytes()


def decode_join_reply(body: bytes) -> dict:
    """PACKET_SERVER_JOIN_REPLY (no-delta): sequential fields."""
    r = Reader(body)
    return {
        "you_can_join": r.get_bool8(),
        "message": r.get_string(),
        "capability": r.get_string(),
        "challenge_file": r.get_string(),
        "conn_id": r.get_sint16(),
    }


def decode_new_year(body: bytes) -> int | None:
    """PACKET_NEW_YEAR: year(sint32), fragments(uint16), turn(sint16), delta.
    Returns the turn (its bit is set every turn since the value always
    changes), or None if absent."""
    r = Reader(body)
    bits = r.get_bitvector(3)
    if bits[0]:
        r.get_sint32()
    if bits[1]:
        r.get_uint16()
    return r.get_sint16() if bits[2] else None


_UNIT_ACTIONS_ZERO = {
    "actor_unit_id": 0,
    "target_unit_id": 0,
    "target_city_id": 0,
    "target_tile_id": 0,
    "target_extra_id": 0,
    "request_kind": 0,
    "action_probabilities": [],
}


def decode_unit_actions(body: bytes, last: dict | None = None) -> dict:
    """PACKET_UNIT_ACTIONS (sc): the server's legal-action reply. 7 delta fields;
    bit 6 is `action_probabilities[ACTION_COUNT]`, a fixed array (no length
    prefix) of 2-byte (min,max) entries indexed by action id. The server deltas
    against the previous unit_actions it sent, so pass the `last` decoded dict to
    carry unchanged fields forward. Array length is read to end of body.

    An action `a` is legal-to-offer when action_probabilities[a] != (0, 0)
    (the impossible / not-applicable sentinel, common/fc_types.h)."""
    last = last or _UNIT_ACTIONS_ZERO
    out = dict(last)
    r = Reader(body)
    bits = r.get_bitvector(7)
    if bits[0]:
        out["actor_unit_id"] = r.get_uint32()
    if bits[1]:
        out["target_unit_id"] = r.get_uint32()
    if bits[2]:
        out["target_city_id"] = r.get_uint32()
    if bits[3]:
        out["target_tile_id"] = r.get_sint32()
    if bits[4]:
        out["target_extra_id"] = r.get_sint8()
    if bits[5]:
        out["request_kind"] = r.get_uint8()
    if bits[6]:
        probs = []
        while r.remaining() >= 2:
            probs.append((r.get_uint8(), r.get_uint8()))
        out["action_probabilities"] = probs
    return out


# --- outgoing helpers that don't need delta state --------------------------
def empty() -> bytes:
    """Body for a fieldless packet (e.g. CONN_PONG)."""
    return b""
