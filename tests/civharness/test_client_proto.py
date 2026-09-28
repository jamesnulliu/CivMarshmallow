"""Wire-protocol unit tests — exact-byte assertions for the encoders, so a
regression in the delta/dataio layer is caught without a server. The expected
bytes match what freeciv-server 3.2.5 accepts on the wire.
"""

import types

from civharness.client import FreecivClient, LegalActionsUnavailable
from civharness.proto import packets as P
from civharness.proto.dataio import Reader, Writer
from civharness.proto.framing import ProtocolError


def test_dataio_primitives():
    assert Writer().put_uint8(0x12).bytes() == b"\x12"
    assert Writer().put_uint16(0x1234).bytes() == b"\x12\x34"  # big-endian
    assert Writer().put_uint32(120).bytes() == b"\x00\x00\x00\x78"
    assert Writer().put_sint16(-1).bytes() == b"\xff\xff"
    assert Writer().put_sint8(-2).bytes() == b"\xfe"
    assert Writer().put_bool8(True).bytes() == b"\x01"
    assert (
        Writer().put_string("hi").bytes() == b"hi\x00"
    )  # NUL-terminated, no len prefix
    assert Writer().put_string("").bytes() == b"\x00"


def test_bitvector_lsb_first():
    # bits 0,2,8 set across 9 fields -> 2 bytes, LSB-first.
    bits = [True, False, True, False, False, False, False, False, True]
    assert Writer().put_bitvector(bits, 9).bytes() == b"\x05\x01"
    assert Writer().put_bitvector([True, True, True], 3).bytes() == b"\x07"
    # round-trip
    r = Reader(b"\x05\x01")
    assert r.get_bitvector(9) == bits


def test_worklist_encoding():
    # uint8 length, then (kind, value) uint8 pairs.
    assert Writer().put_worklist([(6, 4), (3, 5)]).bytes() == b"\x02\x06\x04\x03\x05"
    assert Writer().put_worklist([]).bytes() == b"\x00"


def test_city_change_delta_first_send():
    enc = P.DeltaEncoder()
    # city 120, unit (VUT_UTYPE=6), value 4 -> all three fields differ from 0.
    body = enc.encode(P.CITY_CHANGE, [120, P.VUT_UTYPE, 4])
    assert body == b"\x07" + b"\x00\x00\x00\x78" + b"\x06" + b"\x04"


def test_delta_state_mirrors_server():
    enc = P.DeltaEncoder()
    enc.encode(P.CITY_CHANGE, [120, 6, 4])
    # Re-sending an identical packet: every field matches old -> empty bitvector.
    assert enc.encode(P.CITY_CHANGE, [120, 6, 4]) == b"\x00"
    # Changing only production_value (bit 2) -> only that field on the wire.
    assert enc.encode(P.CITY_CHANGE, [120, 6, 5]) == b"\x04" + b"\x05"


def test_scalar_control_packets():
    enc = P.DeltaEncoder()
    assert enc.encode(P.PLAYER_RESEARCH, [1]) == b"\x01\x00\x01"
    assert enc.encode(P.PLAYER_TECH_GOAL, [7]) == b"\x01\x00\x07"
    assert enc.encode(P.CITY_BUY, [120]) == b"\x01\x00\x00\x00\x78"
    assert enc.encode(P.CITY_SELL, [120, 5]) == b"\x03\x00\x00\x00\x78\x05"
    assert enc.encode(P.PLAYER_PHASE_DONE, [10]) == b"\x01\x00\x0a"


def test_chat_msg_req_delta_string():
    enc = P.DeltaEncoder()
    assert enc.encode(P.CHAT_MSG_REQ, ["/start"]) == b"\x01" + b"/start\x00"


def test_worklist_packet():
    enc = P.DeltaEncoder()
    body = enc.encode(P.CITY_WORKLIST, [120, [(6, 4), (6, 5)]])
    # both fields differ from (0, []) -> bitvector 0x03, city_id, then worklist.
    assert body == b"\x03" + b"\x00\x00\x00\x78" + b"\x02\x06\x04\x06\x05"


def test_unit_orders_folds_bools():
    enc = P.DeltaEncoder()
    orders = [
        {
            "order": 0,
            "activity": 0,
            "target": -1,
            "sub_target": -1,
            "action": 255,
            "dir": 2,
        }
    ]
    body = enc.encode_unit_orders(
        unit_id=42,
        src_tile=100,
        orders=orders,
        dest_tile=105,
        repeat=False,
        vigilant=True,
    )
    r = Reader(body)
    bits = r.get_bitvector(7)
    assert bits[0] and bits[1] and bits[2]  # unit_id, src_tile, length present
    assert bits[3] is False and bits[4] is True  # repeat=False, vigilant=True (folded)
    assert bits[5] and bits[6]  # orders + dest_tile present
    assert r.get_uint32() == 42
    assert r.get_sint32() == 100
    assert r.get_uint16() == 1  # length
    assert r.get_uint8() == 0 and r.get_uint8() == 0  # order, activity
    assert r.get_sint32() == -1 and r.get_sint16() == -1  # target, sub_target
    assert r.get_uint8() == 255 and r.get_sint8() == 2  # action, dir
    assert r.get_sint32() == 105  # dest_tile


def test_server_join_req_no_delta():
    body = P.server_join_req("civharness")
    r = Reader(body)
    assert r.get_string() == "civharness"
    assert r.get_string() == P.NETWORK_CAPSTRING
    assert r.get_string() == ""  # version_label
    assert (r.get_uint32(), r.get_uint32(), r.get_uint32()) == (3, 2, 5)


def test_expanded_action_encoders():
    """Exact bytes for the rates/government/worked-tile/diplomacy/legal-action
    cs packets (first send: delta baseline is all-zero, so every non-zero
    field's bit is set). Specs verified against freeciv-3.2.5 packets.def."""
    enc = lambda pt, vals: P.DeltaEncoder().encode(pt, vals)
    # PLAYER_RATES: 3 uint8; all non-zero -> bits 0b111.
    assert enc(P.PLAYER_RATES, [30, 10, 60]) == b"\x07\x1e\x0a\x3c"
    # PLAYER_CHANGE_GOVERNMENT: one sint8.
    assert enc(P.PLAYER_CHANGE_GOVERNMENT, [5]) == b"\x01\x05"
    # CITY_MAKE_WORKER: uint32 city, sint32 tile.
    assert (
        enc(P.CITY_MAKE_WORKER, [120, 250]) == b"\x03\x00\x00\x00\x78\x00\x00\x00\xfa"
    )
    assert (
        enc(P.CITY_MAKE_SPECIALIST, [120, 3]) == b"\x03\x00\x00\x00\x78\x00\x00\x00\x03"
    )
    # DIPLOMACY_CANCEL_PACT: uint16 player, uint8 clause.
    assert enc(P.DIPLOMACY_CANCEL_PACT, [2, 6]) == b"\x03\x00\x02\x06"
    # UNIT_GET_ACTIONS: actor, target_unit(0->bit clear), target_tile, extra(-1), kind.
    assert enc(P.UNIT_GET_ACTIONS, [5, 0, 100, -1, 1]) == (
        b"\x1d\x00\x00\x00\x05\x00\x00\x00\x64\xff\x01"
    )


def test_do_action_encoder():
    """PACKET_UNIT_DO_ACTION (84): delta bitvector over
    [uint32 actor, sint32 target, sint16 sub_tgt, estring name, uint8 action].
    Field order/widths verified against packets.def and packets_gen.c of the
    pinned source; a single such packet executes the action, no negotiation."""
    # Fortify unit 7 (ATK_SELF -> target=own id, sub=0, name="", action=61):
    # non-zero fields are actor, target, action -> bits 0,1,4 = 0x13.
    assert P.DeltaEncoder().encode(P.UNIT_DO_ACTION, [7, 7, 0, "", 61]) == (
        b"\x13\x00\x00\x00\x07\x00\x00\x00\x07\x3d"
    )
    # A tile action with an extra sub-target (bits 0,1,2,4 = 0x17):
    # actor, tile index 42, extra id 5, action 66 (Build Irrigation).
    assert P.DeltaEncoder().encode(P.UNIT_DO_ACTION, [7, 42, 5, "", 66]) == (
        b"\x17\x00\x00\x00\x07\x00\x00\x00\x2a\x00\x05\x42"
    )
    # A named action (bits 0,1,3,4 = 0x1B): the estring `name` sits between the
    # (omitted) sub_tgt and the action byte — locks the field position.
    assert P.DeltaEncoder().encode(P.UNIT_DO_ACTION, [7, 7, 0, "Rome", 30]) == (
        b"\x1b\x00\x00\x00\x07\x00\x00\x00\x07Rome\x00\x1e"
    )


def test_unit_actions_decode_and_delta_merge():
    """PACKET_UNIT_ACTIONS (sc): full decode, then a delta packet that only
    changes request_kind must carry the array/actor forward."""
    w = Writer().put_bitvector([True] * 7, 7)
    w.put_uint32(5).put_uint32(0).put_uint32(0).put_sint32(-1).put_sint8(-1).put_uint8(
        1
    )
    for mn, mx in [(0, 0), (200, 200), (0, 255)]:
        w.put_uint8(mn).put_uint8(mx)
    ua = P.decode_unit_actions(w.bytes())
    assert ua["actor_unit_id"] == 5
    assert ua["action_probabilities"] == [(0, 0), (200, 200), (0, 255)]

    w2 = Writer().put_bitvector([False] * 5 + [True, False], 7).put_uint8(9)
    ua2 = P.decode_unit_actions(w2.bytes(), ua)
    assert ua2["request_kind"] == 9  # updated
    assert ua2["actor_unit_id"] == 5  # carried forward
    assert ua2["action_probabilities"] == [(0, 0), (200, 200), (0, 255)]  # carried


class _ScriptedFraming:
    """A stand-in for PacketFraming that plays a fixed script of reads: the
    string 'timeout' raises TimeoutError (an idle socket read), 'drop' raises
    ProtocolError (a disconnect), a (ptype, body) tuple is delivered."""

    def __init__(self, script):
        self._script = list(script)
        self.reads = 0

    def send(self, ptype, body):
        pass

    def next_packet(self, timeout):
        self.reads += 1
        if not self._script:
            raise TimeoutError()
        item = self._script.pop(0)
        if item == "timeout":
            raise TimeoutError()
        if item == "drop":
            raise ProtocolError("socket closed")
        return item


def _client_with(script, actions):
    cli = FreecivClient("127.0.0.1", 0)  # __init__ opens no socket
    cli.fr = _ScriptedFraming(script)
    cli.rules = types.SimpleNamespace(actions=actions)
    return cli


def _unit_actions_body(actor, probs):
    w = Writer().put_bitvector([True] * 7, 7)
    w.put_uint32(actor).put_uint32(0).put_uint32(0).put_sint32(-1).put_sint8(-1)
    w.put_uint8(1)
    for mn, mx in probs:
        w.put_uint8(mn).put_uint8(mx)
    return w.bytes()


def test_get_unit_actions_timeout_honors_full_deadline():
    """Two idle 1s reads must NOT abort a 5s query; only when the whole deadline
    passes (or the socket drops) does it give up."""
    body = _unit_actions_body(7, [(0, 0), (200, 200)])  # action id 1 legal
    cli = _client_with(["timeout", "timeout", (P.UNIT_ACTIONS, body)], {"Fortify": 1})
    legal = cli.get_unit_actions(7, timeout=5.0)
    assert legal == {"Fortify": (200, 200)}, legal
    assert cli.fr.reads == 3  # it kept reading past the two idle timeouts


def test_get_unit_actions_protocol_error_breaks_immediately():
    """A disconnect ends the query at once with LegalActionsUnavailable — that
    is distinct from an idle timeout, and from an empty 'nothing legal' reply."""
    cli = _client_with(["drop"], {"Fortify": 1})
    raised = False
    try:
        cli.get_unit_actions(7, timeout=5.0)
    except LegalActionsUnavailable:
        raised = True
    assert raised and cli.fr.reads == 1


def test_join_reply_and_new_year_round_trip():
    w = Writer()
    w.put_bool8(True).put_string("Welcome").put_string(P.NETWORK_CAPSTRING)
    w.put_string("challenge").put_sint16(1)
    reply = P.decode_join_reply(w.bytes())
    assert reply["you_can_join"] and reply["conn_id"] == 1

    ny = Writer().put_bitvector([True, False, True], 3).put_sint32(-3000).put_sint16(42)
    assert P.decode_new_year(ny.bytes()) == 42
