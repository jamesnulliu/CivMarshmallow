"""Freeciv 3.2.5 client wire protocol — the faithful path to external control.

Server-side Lua can issue rule-checked unit actions and god-mode grants,
but it has *no* setter for city production or research target: those are
genuine player decisions the engine only accepts as client packets
(PACKET_CITY_CHANGE, PACKET_PLAYER_RESEARCH, ...). This package speaks the
real Freeciv network protocol so an external policy can make those decisions
through the normal pipeline — economy and RNG untouched — which is what makes
a branched position's value faithful rather than distorted by cheats.

Layout
------
- dataio    : byte-level put/get primitives (verified against dataio_raw.c)
- framing   : packet header + compression framing over a socket
- packets   : packet-type ids and the delta-aware (de)serializers
- rulesets  : decoders for the extra/action ruleset packets
The high-level driver lives in civharness.client.

Everything here is grounded in freeciv-3.2.5 source (common/networking):
packet header is [uint16 len][uint8 type] during login, [uint16 len]
[uint16 type] after; big-endian; strings are bytes + NUL; the delta protocol
sends a leading LSB-first bitvector then only the changed fields.
"""

from civharness.proto import packets
from civharness.proto.dataio import Reader, Writer
from civharness.proto.framing import PacketFraming

__all__ = ["PacketFraming", "Reader", "Writer", "packets"]
