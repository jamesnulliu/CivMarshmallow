"""Byte-level encode/decode primitives — a faithful port of the parts of
freeciv-3.2.5 `common/networking/dataio_raw.c` this client exercises.

Wire facts (verified in source):
  * All multi-byte integers are big-endian (`htons`/`htonl`).
  * sint* are two's-complement in the same width as their unsigned form.
  * bool8 is one byte, 0 or 1.
  * string/estring are the raw bytes followed by a single NUL terminator —
    there is NO length prefix (`dio_put_string_raw` -> memory + strlen+1;
    estring == string in the non-JSON build).
  * a bitvector is `ceil(nbits/8)` bytes, LSB-first: bit i lives in byte
    i//8 at mask 1 << (i%8)  (utility/bitvector.h).
  * a worklist is uint8 length then, per entry, uint8 kind + uint8 value
    (`dio_put_worklist_raw`).
  * a unit_order is order:u8, activity:u8, target:sint32, sub_target:sint16,
    action:u8, dir:sint8 (`dio_put_unit_order_raw`).
"""


class Writer:
    """Accumulates the body of one packet, in freeciv wire encoding."""

    def __init__(self):
        self._buf = bytearray()

    # -- primitives ---------------------------------------------------------
    def put_uint8(self, v: int) -> "Writer":
        self._buf.append(v & 0xFF)
        return self

    def put_uint16(self, v: int) -> "Writer":
        self._buf += (v & 0xFFFF).to_bytes(2, "big")
        return self

    def put_uint32(self, v: int) -> "Writer":
        self._buf += (v & 0xFFFFFFFF).to_bytes(4, "big")
        return self

    def put_sint8(self, v: int) -> "Writer":
        return self.put_uint8(v + 0x100 if v < 0 else v)

    def put_sint16(self, v: int) -> "Writer":
        return self.put_uint16(v + 0x10000 if v < 0 else v)

    def put_sint32(self, v: int) -> "Writer":
        return self.put_uint32(v + 0x100000000 if v < 0 else v)

    def put_bool8(self, v: bool) -> "Writer":
        return self.put_uint8(1 if v else 0)

    def put_string(self, s: str) -> "Writer":
        self._buf += s.encode("utf-8") + b"\x00"
        return self

    put_estring = put_string  # identical in the raw (non-JSON) protocol

    def put_bitvector(self, bits: list[bool], nfields: int) -> "Writer":
        """LSB-first, ceil(nfields/8) bytes."""
        nbytes = (nfields + 7) // 8
        raw = bytearray(nbytes)
        for i, on in enumerate(bits):
            if on:
                raw[i // 8] |= 1 << (i % 8)
        self._buf += raw
        return self

    def put_worklist(self, entries: list[tuple[int, int]]) -> "Writer":
        """entries: [(kind, value), ...] — kind is universals_n, value the id."""
        self.put_uint8(len(entries))
        for kind, value in entries:
            self.put_uint8(kind)
            self.put_uint8(value)
        return self

    def put_type(self, dio_type: str, v) -> "Writer":
        return getattr(self, "put_" + dio_type)(v)

    def bytes(self) -> bytes:
        return bytes(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


class Reader:
    """Sequentially decodes a packet body already sliced from the stream."""

    def __init__(self, data: bytes, pos: int = 0):
        self._d = data
        self.pos = pos

    def remaining(self) -> int:
        return len(self._d) - self.pos

    def get_uint8(self) -> int:
        v = self._d[self.pos]
        self.pos += 1
        return v

    def get_uint16(self) -> int:
        v = int.from_bytes(self._d[self.pos : self.pos + 2], "big")
        self.pos += 2
        return v

    def get_uint32(self) -> int:
        v = int.from_bytes(self._d[self.pos : self.pos + 4], "big")
        self.pos += 4
        return v

    def get_sint8(self) -> int:
        v = self.get_uint8()
        return v - 0x100 if v > 0x7F else v

    def get_sint16(self) -> int:
        v = self.get_uint16()
        return v - 0x10000 if v > 0x7FFF else v

    def get_sint32(self) -> int:
        v = self.get_uint32()
        return v - 0x100000000 if v > 0x7FFFFFFF else v

    def get_bool8(self) -> bool:
        return self.get_uint8() != 0

    def get_string(self) -> str:
        end = self._d.index(0, self.pos)
        s = self._d[self.pos : end].decode("utf-8", "replace")
        self.pos = end + 1
        return s

    get_estring = get_string

    def get_bitvector(self, nfields: int) -> list[bool]:
        nbytes = (nfields + 7) // 8
        raw = self._d[self.pos : self.pos + nbytes]
        self.pos += nbytes
        return [bool(raw[i // 8] & (1 << (i % 8))) for i in range(nfields)]

    def get_type(self, dio_type: str):
        return getattr(self, "get_" + dio_type)()
