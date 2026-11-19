"""Map geometry of env v4: Freeciv `ISO|HEX` topology with `WRAPX|WRAPY` (civ2civ3 ruleset settings).

Ported from freeciv-3.2.5 common/map.h (NATIVE_TO_MAP_POS / MAP_TO_NATIVE_POS), common/map.c (DIR_DX/DIR_DY,
base_map_distance_vector, map_vector_to_real_distance, is_valid_dir).  Tile indices are native: idx = y*xsize + x.

Iso-hex has six valid directions: NW, N, W, E, S, SE (NE and SW are invalid).  Directions and distances are in map
coordinates; wrapping is done in native coordinates.  Pure functions, no engine access.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

DIR_NAMES = ("nw", "n", "ne", "w", "e", "sw", "s", "se")  # enum direction8 order
DIR_DX = (-1, 0, 1, -1, 1, -1, 0, 1)
DIR_DY = (-1, -1, -1, 0, 0, 1, 1, 1)
HEX_ISO_DIRS = (0, 1, 3, 4, 6, 7)  # valid in ISO|HEX
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_NEIGHBOR_CACHE: dict = {}


@dataclass(frozen=True)
class HexMap:
    xsize: int  # native width
    ysize: int  # native height
    wrap_x: bool = True
    wrap_y: bool = True

    # -- coordinates ------------------------------------------------------
    def native(self, idx: int) -> tuple[int, int]:
        return idx % self.xsize, idx // self.xsize

    def index(self, nx: int, ny: int) -> int | None:
        """Normalized native position -> index, or None if off-map."""
        if self.wrap_x:
            nx %= self.xsize
        if self.wrap_y:
            ny %= self.ysize
        if not (0 <= nx < self.xsize and 0 <= ny < self.ysize):
            return None
        return ny * self.xsize + nx

    def nat_to_map(self, nx: int, ny: int) -> tuple[int, int]:
        mx = (ny + (ny & 1)) // 2 + nx  # exact: ny + (ny & 1) is even
        return mx, ny - mx + self.xsize

    def map_to_nat(self, mx: int, my: int) -> tuple[int, int]:
        ny = mx + my - self.xsize
        return (2 * mx - ny - (ny & 1)) // 2, ny  # exact: 2mx - ny - (ny&1) is even

    def map_pos(self, idx: int) -> tuple[int, int]:
        return self.nat_to_map(*self.native(idx))

    # -- adjacency / distance --------------------------------------------
    def step(self, idx: int, d: int) -> int | None:
        mx, my = self.map_pos(idx)
        return self.index(*self.map_to_nat(mx + DIR_DX[d], my + DIR_DY[d]))

    def neighbors(self, idx: int) -> tuple[tuple[int, int], ...]:
        """((direction8, neighbour index), ...) for the six valid iso-hex directions (cached per map and tile)."""
        key = (self.xsize, self.ysize, self.wrap_x, self.wrap_y, idx)
        hit = _NEIGHBOR_CACHE.get(key)
        if hit is None:
            out = []
            for d in HEX_ISO_DIRS:
                t = self.step(idx, d)
                if t is not None and t != idx:
                    out.append((d, t))
            hit = _NEIGHBOR_CACHE[key] = tuple(out)
        return hit

    def vector(self, a: int, b: int) -> tuple[int, int]:
        """Map-coordinate vector a -> b, shortest under wrapping (base_map_distance_vector)."""
        x0, y0 = self.native(a)
        x1, y1 = self.native(b)
        dx, dy = x1 - x0, y1 - y0
        if self.wrap_x:
            dx = (dx + self.xsize // 2) % self.xsize - self.xsize // 2
        if self.wrap_y:
            dy = (dy + self.ysize // 2) % self.ysize - self.ysize // 2
        m0 = self.nat_to_map(x0, y0)
        m1 = self.nat_to_map(x0 + dx, y0 + dy)
        return m1[0] - m0[0], m1[1] - m0[1]

    @staticmethod
    def _real(dx: int, dy: int) -> int:
        if (dx < 0 < dy) or (dy < 0 < dx):
            return abs(dx) + abs(dy)
        return max(abs(dx), abs(dy))

    def distance(self, a: int, b: int) -> int:
        """Hex step distance = BFS path length.  Freeciv's real_map_distance wraps the native delta into
        [-size/2, size/2) first, which can overestimate far pairs by a few steps; here every wrap image is tried
        (identical for the near pairs that matter: citymindist, work radius, threats)."""
        x0, y0 = self.native(a)
        x1, y1 = self.native(b)
        m0 = self.nat_to_map(x0, y0)
        best = None
        for ox in (-self.xsize, 0, self.xsize) if self.wrap_x else (0,):
            for oy in (-self.ysize, 0, self.ysize) if self.wrap_y else (0,):
                m1 = self.nat_to_map(x1 + ox, y1 + oy)
                d = self._real(m1[0] - m0[0], m1[1] - m0[1])
                best = d if best is None else min(best, d)
        return best

    def within(self, center: int, radius: int) -> list[int]:
        """All tiles at real distance <= radius (BFS over hex steps), sorted."""
        seen, frontier = {center}, [center]
        for _ in range(radius):
            nxt = []
            for t in frontier:
                for _d, n in self.neighbors(t):
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return sorted(seen)

    def sector(self, a: int, b: int) -> str:
        """8-way compass sector of b seen from a, for text only.  Screen axes of an iso map: map +x points
        south-east and map +y south-west, so screen east = dx - dy and screen south = dx + dy."""
        dx, dy = self.vector(a, b)
        ex, sy = dx - dy, dx + dy
        if ex == 0 and sy == 0:
            return "HERE"
        import math

        ang = math.degrees(math.atan2(-sy, ex))  # 0 = east, 90 = north
        k = int(((90 - ang) % 360 + 22.5) // 45) % 8  # 0 = N, clockwise
        return SECTORS[k]

    # -- paths ---------------------------------------------------------------
    def path(self, src: int, dst: int, passable) -> list[int] | None:
        """Shortest hex path src -> dst as direction8 codes (BFS, uniform step cost) over tiles for which
        passable(idx) is true (dst must be passable too).  None if unreachable."""
        if src == dst:
            return []
        prev = {src: None}
        q = deque([src])
        while q:
            t = q.popleft()
            for d, n in self.neighbors(t):
                if n in prev or not passable(n):
                    continue
                prev[n] = (t, d)
                if n == dst:
                    dirs = []
                    cur = n
                    while prev[cur] is not None:
                        p, dd = prev[cur]
                        dirs.append(dd)
                        cur = p
                    return dirs[::-1]
                q.append(n)
        return None

    def bfs_tree(self, src: int, passable, max_depth: int | None = None) -> dict:
        """One BFS from src over passable tiles: {tile: (parent, direction8)} (src -> None).  Paths to many targets
        from one source come from a single tree (a BFS per target is too slow for large empires)."""
        prev = {src: None}
        frontier, depth = [src], 0
        while frontier and (max_depth is None or depth < max_depth):
            depth += 1
            nxt = []
            for t in frontier:
                for d, n in self.neighbors(t):
                    if n not in prev and passable(n):
                        prev[n] = (t, d)
                        nxt.append(n)
            frontier = nxt
        return prev

    @staticmethod
    def path_in_tree(prev: dict, dst: int) -> list[int] | None:
        if dst not in prev:
            return None
        dirs, cur = [], dst
        while prev[cur] is not None:
            p, d = prev[cur]
            dirs.append(d)
            cur = p
        return dirs[::-1]
