"""Packet framing over a TCP socket — the header + compression envelope from
freeciv-3.2.5 `common/networking/packets.c`.

Header (big-endian):
    [uint16 length][uint8 type]     during login
    [uint16 length][uint16 type]    after login
`length` counts the whole packet including the header, and is back-patched
once the body is known (SEND_PACKET_START/END).

Compression (receive side): the length field doubles as a signal.
    len == 0xFFFF (JUMBO_SIZE)          -> next uint32 is the real chunk len;
                                           a zlib blob of packets follows.
    len >= 16385 (COMPRESSION_BORDER)   -> zlib blob, chunk len = len - BORDER.
    otherwise                          -> a plain packet of `len` bytes.
We never compress what we send (our packets are tiny); we only decompress.
"""

import socket
import zlib

from civharness.proto.dataio import Reader

COMPRESSION_BORDER = 16 * 1024 + 1  # 16385
JUMBO_SIZE = 0xFFFF


class ProtocolError(RuntimeError):
    """The connection closed or delivered a malformed packet."""


class PacketFraming:
    """Frames packets on a connected socket. Not thread-safe; one game, one
    client, driven from a single loop."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = bytearray()
        self.login_header = True  # flips to False right after JOIN_REPLY

    def _type_size(self) -> int:
        return 1 if self.login_header else 2

    # -- send ---------------------------------------------------------------
    def send(self, ptype: int, body: bytes) -> None:
        tsize = self._type_size()
        total = 2 + tsize + len(body)
        if total >= COMPRESSION_BORDER:
            raise ProtocolError(f"outgoing packet too large ({total} bytes)")
        header = total.to_bytes(2, "big") + ptype.to_bytes(tsize, "big")
        self.sock.sendall(header + body)

    # -- receive ------------------------------------------------------------
    def _fill(self, timeout: float | None) -> bool:
        """Read one chunk from the socket into the buffer. False on EOF."""
        self.sock.settimeout(timeout)
        chunk = self.sock.recv(65536)
        if not chunk:
            return False
        self._buf += chunk
        return True

    def _try_extract(self) -> tuple[int, bytes] | None:
        """Pull one framed packet from the buffer, decompressing as needed.
        Returns (ptype, body) or None if more bytes are required."""
        while True:
            if len(self._buf) < 2:
                return None
            length = int.from_bytes(self._buf[0:2], "big")

            if length == JUMBO_SIZE:
                if len(self._buf) < 6:
                    return None
                whole = int.from_bytes(self._buf[2:6], "big")
                header_size = 6
                compressed = True
            elif length >= COMPRESSION_BORDER:
                whole = length - COMPRESSION_BORDER
                header_size = 2
                compressed = True
            else:
                whole = length
                compressed = False
                header_size = 2  # length field only; type read below

            if whole < 2 or len(self._buf) < whole:
                return None

            if compressed:
                blob = bytes(self._buf[header_size:whole])
                del self._buf[:whole]
                self._buf[:0] = zlib.decompress(blob)  # prepend decoded packets
                continue  # re-extract from the now-inflated buffer

            tsize = self._type_size()
            if whole < 2 + tsize:
                raise ProtocolError(f"packet length {whole} smaller than header")
            ptype = int.from_bytes(self._buf[2 : 2 + tsize], "big")
            body = bytes(self._buf[2 + tsize : whole])
            del self._buf[:whole]
            return ptype, body

    def next_packet(self, timeout: float | None = 30.0) -> tuple[int, bytes]:
        """Block for the next (ptype, body). Raises on EOF/timeout."""
        while True:
            got = self._try_extract()
            if got is not None:
                return got
            if not self._fill(timeout):
                raise ProtocolError("connection closed by server")

    def reader(self, body: bytes) -> Reader:
        return Reader(body)
