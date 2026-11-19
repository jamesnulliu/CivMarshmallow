"""Env v4 / v4.1 observation (schema civm-state-v4).

Input is a player-visible Observation (EnvSpec preset "saga_visible") or a plain dict with the same fields
(turn, me, my, player_view, diplomacy, rates).  Output: StateV4 with six text sections plus the reference
tables (C/S/T/G/F/H) that civmarsh.env.v41.menu turns into actions.  Deterministic: the same input
gives the same bytes; every list is sorted and capped with an explicit "N more" line.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from civmarsh.env.v41.hexmap import HexMap

# v4.1 plays the civm41 ruleset (scripts/make_civm41.py): civ2civ3 without great
# wonders, Coinage, ships, diplomats, caravans, migrants and pillage. Production
# lists and build costs come from it. CIVMARSH_RULESET=civ2civ3 selects the stock
# ruleset instead.
RULESET = os.environ.get("CIVMARSH_RULESET", "civm41")

STATE_SCHEMA = "civm-state-v4"
NO_OWNER = 65535
NON_MILITARY = frozenset(
    {
        "Settlers",
        "Migrants",
        "Workers",
        "Engineers",
        "Explorer",
        "Caravan",
        "Freight",
        "Diplomat",
        "Spy",
        "Leader",
    }
)
SETTLER_TYPES = frozenset({"Settlers", "Migrants"})
WORKER_TYPES = frozenset({"Workers", "Engineers"})
CITYMINDIST = 3
WORK_RADIUS = 2  # radius_sq_city 5 on iso-hex = real distance <= 2 (19 tiles)
MOVE_FRAGS = 6  # civ2civ3 terrain.ruleset move_fragments


@dataclass(frozen=True)
class CapsV4:
    cities: int = 40
    idle_units: int = 14
    sites: int = 6
    threats: int = 6
    targets: int = 6
    events: int = 10
    threat_radius: int = 4
    city_threat_radius: int = 3
    site_search_radius: int = 10
    frontier_radius: int = 8
    site_dist_penalty: float = 2.0  # score -= penalty * (distance to the nearest own city - CITYMINDIST); recall of the bots' own sites: 0 -> .31, 1 -> .39, 2 -> .44, 4 -> .47


def ruleset_file(name: str) -> Path | None:
    """``<ruleset dir>/<name>`` of the v4.1 ruleset, or None when it is not installed."""
    from civmarsh.env.menu import ruleset_dir

    try:
        path = ruleset_dir(RULESET) / name
    except FileNotFoundError:
        return None
    return path if path.exists() else None


@lru_cache(maxsize=1)
def terrain_table() -> dict[str, dict]:
    """{terrain name: {food, shield, trade, oceanic}} from the ruleset's terrain.ruleset."""
    path = ruleset_file("terrain.ruleset")
    out = {}
    if path is None:
        return out
    for sec in re.split(r"\n\[terrain_", path.read_text(errors="replace"))[1:]:
        name = re.search(r'\nname\s*=\s*_\("([^"]+)"\)', sec)
        if not name:
            continue
        val = {
            k: int(m.group(1)) if (m := re.search(rf"\n{k}\s*=\s*(-?\d+)", sec)) else 0
            for k in ("food", "shield", "trade")
        }
        cls = re.search(r'\nclass\s*=\s*"([^"]+)"', sec)
        val["oceanic"] = bool(cls and cls.group(1) == "Oceanic")
        out[name.group(1)] = val
    return out


@lru_cache(maxsize=1)
def build_costs() -> dict[str, int]:
    """{unit or building name: build_cost in shields} from the pinned ruleset (shieldbox 100%; costs are the same in
    civ2civ3 and civm41)."""
    out = {}
    for fname in ("units.ruleset", "buildings.ruleset"):
        path = ruleset_file(fname)
        if path is None:
            continue
        for sec in re.split(
            r"\n\[(?:unit|building)_", path.read_text(errors="replace")
        )[1:]:
            name = re.search(
                r'\nname\s*=\s*_\("(?:\?[^:"]*:)?([^"]+)"\)', sec
            )  # "?unit:Workers" -> Workers
            rule = re.search(r'\nrule_name\s*=\s*"([^"]+)"', sec)
            cost = re.search(r"\nbuild_cost\s*=\s*(\d+)", sec)
            if name and cost:
                out[name.group(1)] = int(cost.group(1))
                # the game reports the rule name: Colosseum (shown "Amphitheater"), Harbour ("Harbor")
                if rule:
                    out[rule.group(1)] = int(cost.group(1))
    return out


def production_text(c) -> str:
    """'Settlers 16/30 (+4, 4 turns)': what the city builds, shields so far / cost, surplus, turns left."""
    item, stock, plus = (
        c.get("building"),
        c.get("shield_stock") or 0,
        c.get("shield_surplus") or 0,
    )
    cost = build_costs().get(item)
    if cost is None:
        return f"{item} {stock} (+{plus})"
    left = max(0, cost - stock)
    n = -(-left // plus) if plus > 0 else 0
    turns = (
        "done next turn"
        if left == 0
        else (f"{n} turn{'s' if n != 1 else ''}" if plus > 0 else "never at +0")
    )
    return f"{item} {stock}/{cost} (+{plus}, {turns})"


def _tech(v) -> str:
    return "none" if v in (None, "", "None", "A_UNSET", "A_NONE") else str(v)


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


@dataclass
class Ref:
    label: str
    kind: str  # city | site | threat | target | hut
    tile: int
    info: dict = field(default_factory=dict)


class StateV4:
    def __init__(
        self,
        obs,
        *,
        endturn: int = 120,
        prev: dict | None = None,
        caps: CapsV4 | None = None,
        wrap_x: bool = True,
        wrap_y: bool = True,
    ):
        self.caps = caps or CapsV4()
        self.turn = _get(obs, "turn")
        self.me = _get(obs, "me")
        self.my = _get(obs, "my") or {}
        self.endturn = endturn
        pv = _get(obs, "player_view") or {}
        mp = pv.get("map") or {}
        self.map = HexMap(int(mp.get("xsize")), int(mp.get("ysize")), wrap_x, wrap_y)
        self.tiles = {int(k): v for k, v in (mp.get("tiles") or {}).items()}
        self.my_id = self.my.get("player_id")
        self.terr = terrain_table()
        self.cities = sorted(
            (c for c in self.my.get("cities") or [] if c.get("id") is not None),
            key=lambda c: c["id"],
        )
        self.units = sorted(
            (u for u in self.my.get("units") or [] if u.get("id") is not None),
            key=lambda u: u["id"],
        )
        for o in self.cities + self.units:
            if o.get("tile") is None and o.get("x") is not None:
                o["tile"] = o["y"] * self.map.xsize + o["x"]
        self.foreign_cities = sorted(
            (c for c in pv.get("cities") or [] if not c.get("mine")),
            key=lambda c: c["id"],
        )
        self.foreign_units = sorted(
            (u for u in pv.get("units") or [] if not u.get("mine")),
            key=lambda u: u["id"],
        )
        self.diplomacy = _get(obs, "diplomacy") or self.my.get("diplomacy") or {}
        self.rates = _get(obs, "rates") or self.my.get("rates") or {}
        self.refs: dict[str, Ref] = {}
        self.city_label = {c["id"]: f"C{i + 1}" for i, c in enumerate(self.cities)}
        for c in self.cities:
            self.refs[self.city_label[c["id"]]] = Ref(
                self.city_label[c["id"]], "city", c["tile"], {"id": c["id"]}
            )
        self.sections: dict[str, list[str]] = {}
        self._build(prev)

    # -- helpers ----------------------------------------------------------------
    def land(self, t: int) -> bool:
        tile = self.tiles.get(t)
        if tile is None:
            return False
        info = self.terr.get(tile.get("terrain"), {})
        return not info.get(
            "oceanic", tile.get("terrain") in ("Ocean", "Deep Ocean", "Lake")
        ) and tile.get("terrain") not in ("Inaccessible",)

    def oceanic(self, t: int) -> bool:
        tile = self.tiles.get(t)
        return bool(tile) and self.terr.get(tile.get("terrain"), {}).get(
            "oceanic", False
        )

    def anchor(self) -> int | None:
        if self.cities:
            return self.cities[0]["tile"]
        return self.units[0]["tile"] if self.units else None

    def nearest_city(self, t: int) -> tuple[str | None, int | None]:
        best = None
        for c in self.cities:
            d = self.map.distance(c["tile"], t)
            if best is None or d < best[1]:
                best = (self.city_label[c["id"]], d, c["tile"])
        if best is None:
            return None, None
        return best[0], best[1]

    def coord(self, t: int) -> str:
        """Native map coordinates of a tile, the form goto:@x,y takes."""
        x, y = self.map.native(t)
        return f"@{x},{y}"

    def where(self, t: int) -> str:
        """'<sector> <dist> of C<k>' relative to the nearest own city (or 'no city')."""
        best = None
        for c in self.cities:
            d = self.map.distance(c["tile"], t)
            if best is None or d < best[0]:
                best = (d, c)
        if best is None:
            return "no own city"
        d, c = best
        if d == 0:
            return f"in {self.city_label[c['id']]}"
        return f"{self.map.sector(c['tile'], t)} {d} of {self.city_label[c['id']]}"

    def tile_yield(self, t: int) -> tuple[int, int, int]:
        tile = self.tiles.get(t)
        if tile is None:
            return 0, 0, 0
        v = self.terr.get(tile.get("terrain"), {})
        f, s, tr = v.get("food", 0), v.get("shield", 0), v.get("trade", 0)
        extras = set(tile.get("extras") or [])
        if "River" in extras:
            tr += 1
        if tile.get("resource_id") is not None:
            f += 1
        return f, s, tr

    @staticmethod
    def _activity(u) -> str:
        """Idle units with pending server-side orders (multi-turn goto/settle) count as busy."""
        a = u.get("activity") or "Idle"
        return "Following-orders" if a == "Idle" and u.get("has_orders") else a

    def _military(self, u) -> bool:
        return u.get("type") not in NON_MILITARY

    # -- sections ---------------------------------------------------------------
    def _build(self, prev):
        my = self.my
        rates = self.rates or {}
        self.sections["OVERVIEW"] = [
            f"turn {self.turn}/{self.endturn} | score {my.get('score')} | gold {my.get('gold')} | "
            f"government {my.get('government')} | "
            + (
                f"rates tax {rates.get('tax')} lux {rates.get('luxury')} sci {rates.get('science')} | "
                if rates.get("tax") is not None
                else ""
            )
            + f"researching {_tech(my.get('researching'))} | goal {_tech(my.get('research_goal'))} | "
            f"techs {len(my.get('techs_known') or [])} | cities {len(self.cities)} | units {len(self.units)}"
        ]
        self._rivals()
        self._cities()
        self._units()
        self._map_facts()
        self._events(prev)

    def _rivals(self):
        owners = sorted(
            {
                o["owner"]
                for o in self.foreign_cities + self.foreign_units
                if o.get("owner") not in (None, NO_OWNER, self.my_id)
            }
        )
        lines = []
        for pid in owners:
            cs = [c for c in self.foreign_cities if c["owner"] == pid]
            us = [u for u in self.foreign_units if u["owner"] == pid]
            near = min(
                (
                    self.map.distance(o["tile"], c["tile"])
                    for o in cs + us
                    for c in self.cities
                ),
                default=None,
            )
            rel = (self.diplomacy.get(str(pid)) or self.diplomacy.get(pid) or {}).get(
                "state"
            )
            mil = sum(self._military(u) for u in us)
            lines.append(
                f"P{pid}: "
                + (f"{rel} | " if rel else "")
                + f"known cities {len(cs)} | "
                + f"nearest to our cities {'unknown' if near is None else near} | "
                f"visible units {len(us)} ({mil} military)"
            )
        self.sections["RIVALS"] = lines or ["none seen"]

    def _cities(self):
        caps, lines = self.caps, []
        mil_on = Counter(u["tile"] for u in self.units if self._military(u))
        for c in self.cities[: caps.cities]:
            lab = self.city_label[c["id"]]
            thr = sorted(
                (self.map.distance(c["tile"], u["tile"]), u["id"])
                for u in self.foreign_units
                if self._military(u)
                and self.map.distance(c["tile"], u["tile"]) <= caps.city_threat_radius
            )
            lines.append(
                f"{lab} #{c['id']} {c.get('name')} {self.coord(c['tile'])} | size {c.get('size')} | building {production_text(c)} | "
                f"food stock {c.get('food_stock')} | "
                f"buildings {len(c.get('buildings') or [])} | defenders {mil_on.get(c['tile'], 0)} | "
                f"threat {('enemy unit at ' + str(thr[0][0])) if thr else 'none'}"
            )
        if len(self.cities) > caps.cities:
            lines.append(f"... {len(self.cities) - caps.cities} more cities")
        self.sections["CITIES"] = lines or ["none"]

    def idle_units(self) -> list[dict]:
        """Units waiting for orders, in priority order (settlers, workers, units next to enemies, others)."""
        foreign_tiles = {o["tile"] for o in self.foreign_units + self.foreign_cities}

        def enemy_adjacent(u):
            return any(n in foreign_tiles for _d, n in self.map.neighbors(u["tile"]))

        idle = [u for u in self.units if self._activity(u) == "Idle"]

        def prio(u):
            t = u.get("type")
            return (
                0
                if t in SETTLER_TYPES
                else 1
                if t in WORKER_TYPES
                else 2
                if enemy_adjacent(u)
                else 3,
                u["id"],
            )

        return sorted(idle, key=prio)

    def busy_split(self) -> tuple[list[dict], list[dict]]:
        """v4.1: busy units (any activity but Idle, or pending orders) split into those next to a
        foreign unit / city (expanded like idle units) and the rest (listed compactly, still re-orderable)."""
        foreign_tiles = {o["tile"] for o in self.foreign_units + self.foreign_cities}
        busy = [u for u in self.units if self._activity(u) != "Idle"]
        near = [
            u
            for u in busy
            if any(n in foreign_tiles for _d, n in self.map.neighbors(u["tile"]))
        ]
        rest = [u for u in busy if u not in near]
        return near, rest

    def _units(self):
        caps = self.caps
        busy = Counter(
            (self._activity(u), u.get("type"))
            for u in self.units
            if self._activity(u) != "Idle"
        )
        lines = []
        if busy:
            lines.append(
                "busy: "
                + ", ".join(
                    f"{n} {t} {a.lower()}" for (a, t), n in sorted(busy.items())
                )
            )
        idle = self.idle_units()
        foreign_tiles = {o["tile"] for o in self.foreign_units + self.foreign_cities}
        for u in idle[: caps.idle_units]:
            flags = []
            if any(n in foreign_tiles for _d, n in self.map.neighbors(u["tile"])):
                flags.append("enemy adjacent")
            if u.get("type") in SETTLER_TYPES and self.valid_site(u["tile"]):
                flags.append("can found city here")
            mv = (u.get("moves") or 0) / MOVE_FRAGS
            lines.append(
                f"U{u['id']} {u.get('type')} {self.coord(u['tile'])} hp {u.get('hp')} moves {mv:g} | {self.where(u['tile'])}"
                + (" | " + ", ".join(flags) if flags else "")
            )
        rest = Counter(u.get("type") for u in idle[caps.idle_units :])
        if rest:
            lines.append(
                "idle, not listed: "
                + ", ".join(f"{n} {t}" for t, n in sorted(rest.items()))
            )
        near, compact = self.busy_split()
        for u in near:
            lines.append(
                f"U{u['id']} {u.get('type')} {self.coord(u['tile'])} hp {u.get('hp')} ({self._activity(u).lower()}) | "
                f"{self.where(u['tile'])} | enemy adjacent"
            )
        groups = {}
        for u in compact:
            groups.setdefault(self.where(u["tile"]), []).append(
                f"U{u['id']} {u.get('type')} {self.coord(u['tile'])} ({self._activity(u).lower()})"
            )
        for loc, us in sorted(groups.items()):
            lines.append(f"busy {loc}: " + ", ".join(us))
        self.sections["UNITS"] = lines or ["none"]

    def valid_site(self, t: int) -> bool:
        if not self.land(t) or self.tiles[t].get("terrain") in ("Glacier",):
            return False
        owner = self.tiles[t].get("owner")
        if owner not in (None, NO_OWNER, self.my_id):
            return False
        for c in self.cities + self.foreign_cities:
            if self.map.distance(c["tile"], t) < CITYMINDIST:
                return False
        return not any(u["tile"] == t for u in self.foreign_units)

    def _map_facts(self):
        caps, lines = self.caps, []
        # sites: reachable over known land from own cities / settlers within the search radius
        starts = [c["tile"] for c in self.cities] + [
            u["tile"] for u in self.units if u.get("type") in SETTLER_TYPES
        ]
        reach, frontier = set(starts), list(starts)
        for _ in range(caps.site_search_radius):
            nxt = []
            for t in frontier:
                for _d, n in self.map.neighbors(t):
                    if n not in reach and self.land(n):
                        reach.add(n)
                        nxt.append(n)
            frontier = nxt
        scored = []
        for t in sorted(reach):
            if not self.valid_site(t):
                continue
            f = s = tr = 0
            for a in self.map.within(t, WORK_RADIUS):
                y = self.tile_yield(a)
                f, s, tr = f + y[0], s + y[1], tr + y[2]
            coastal = any(self.oceanic(n) for _d, n in self.map.neighbors(t))
            near = min(
                (self.map.distance(t, c["tile"]) for c in self.cities),
                default=CITYMINDIST,
            )
            pen = caps.site_dist_penalty * max(0, near - CITYMINDIST)
            scored.append(
                (f + s + tr + (2 if coastal else 0) - pen, t, f, s, tr, coastal)
            )
        self.site_scored = sorted(
            scored, key=lambda x: (-x[0], x[1])
        )  # (score, tile, f, s, tr, coastal)
        ref = self.anchor()
        best_by_sector = {}
        for sc in sorted(scored, key=lambda x: (-x[0], x[1])):
            sec = self.map.sector(ref, sc[1]) if ref is not None else "HERE"
            best_by_sector.setdefault(sec, sc)
        chosen = sorted(best_by_sector.values(), key=lambda x: (-x[0], x[1]))[
            : caps.sites
        ]
        rest = [
            x for x in sorted(scored, key=lambda x: (-x[0], x[1])) if x not in chosen
        ]
        chosen = sorted(
            chosen + rest[: max(0, caps.sites - len(chosen))],
            key=lambda x: (-x[0], x[1]),
        )
        site_lines = []
        for i, (score, t, f, s, tr, coastal) in enumerate(chosen):
            lab = f"S{i + 1}"
            near_foreign = min(
                (self.map.distance(t, c["tile"]) for c in self.foreign_cities),
                default=None,
            )
            self.refs[lab] = Ref(
                lab,
                "site",
                t,
                {"food": f, "shield": s, "trade": tr, "coastal": coastal},
            )
            site_lines.append(
                f"{lab} {self.coord(t)} {self.where(t)} | area food {f} shield {s} trade {tr}"
                + (" | coastal" if coastal else "")
                + (
                    f" | foreign city at {near_foreign}"
                    if near_foreign is not None
                    else ""
                )
            )
        lines.append(
            "SITES "
            + (
                f"({len(scored)} valid known sites)"
                if scored
                else "(no valid known site)"
            )
        )
        lines += site_lines
        # threats
        thr = []
        for u in self.foreign_units:
            if not self._military(u):
                continue
            d = min(
                (self.map.distance(u["tile"], c["tile"]) for c in self.cities),
                default=None,
            )
            if d is not None and d <= caps.threat_radius:
                thr.append((d, u["id"], u))
        thr.sort(key=lambda x: (x[0], x[1]))
        lines.append("THREATS" + ("" if thr else " none"))
        for i, (d, _id, u) in enumerate(thr[: caps.threats]):
            lab = f"T{i + 1}"
            self.refs[lab] = Ref(
                lab, "threat", u["tile"], {"unit_id": u["id"], "owner": u["owner"]}
            )
            lines.append(
                f"{lab} {u.get('type')} of P{u['owner']} {self.coord(u['tile'])} hp {u.get('hp')} | {self.where(u['tile'])}"
            )
        # targets
        own_tiles = [c["tile"] for c in self.cities] + [u["tile"] for u in self.units]
        tg = sorted(
            (
                (
                    min(
                        (self.map.distance(c["tile"], t) for t in own_tiles),
                        default=999,
                    ),
                    c["id"],
                    c,
                )
                for c in self.foreign_cities
            ),
            key=lambda x: (x[0], x[1]),
        )
        lines.append("TARGETS" + ("" if tg else " none known"))
        for i, (d, _id, c) in enumerate(tg[: caps.targets]):
            lab = f"G{i + 1}"
            defenders = sum(
                1
                for u in self.foreign_units
                if u["tile"] == c["tile"] and self._military(u)
            )
            self.refs[lab] = Ref(
                lab, "target", c["tile"], {"city_id": c["id"], "owner": c["owner"]}
            )
            lines.append(
                f"{lab} city {c.get('name')} of P{c['owner']} {self.coord(c['tile'])} size {c.get('size')} | {self.where(c['tile'])}"
                f" | visible defenders {defenders}"
            )
        # frontier: unknown share per sector, plus exploration points F<k> = the nearest known land tile (reachable
        # from own cities/units over known land) that borders unknown tiles, one per sector, most-unknown first
        a = self.anchor()
        if a is not None:
            unknown = Counter()
            area = self.map.within(a, caps.frontier_radius)
            for t in area:
                if t not in self.tiles:
                    unknown[self.map.sector(a, t)] += 1
            huts = sorted(
                t for t, v in self.tiles.items() if "Hut" in (v.get("extras") or [])
            )
            lines.append(
                f"FRONTIER unknown tiles within {caps.frontier_radius} ({len(area) - sum(unknown.values())}"
                f"/{len(area)} known): "
                + ", ".join(
                    f"{s} {unknown.get(s, 0)}"
                    for s in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
                )
            )
            land_reach = self._land_reach(
                [c["tile"] for c in self.cities] + [u["tile"] for u in self.units],
                caps.frontier_radius + 4,
            )
            edge = {}
            for t in sorted(land_reach, key=lambda t: (self.map.distance(a, t), t)):
                if t == a or not any(
                    n not in self.tiles for _d, n in self.map.neighbors(t)
                ):
                    continue
                edge.setdefault(self.map.sector(a, t), t)
            pts = sorted(edge.items(), key=lambda kv: (-unknown.get(kv[0], 0), kv[1]))[
                :4
            ]
            for i, (sec, t) in enumerate(pts):
                lab = f"F{i + 1}"
                self.refs[lab] = Ref(lab, "frontier", t, {"sector": sec})
                lines.append(
                    f"{lab} explore point {self.coord(t)} {self.where(t)} (borders unknown {sec})"
                )
            for i, t in enumerate(huts[:3]):
                lab = f"H{i + 1}"
                self.refs[lab] = Ref(lab, "hut", t, {})
                lines.append(f"{lab} hut {self.coord(t)} {self.where(t)}")
        self.sections["MAP"] = lines

    def _land_reach(self, starts, radius) -> set:
        reach, frontier = set(starts), list(starts)
        for _ in range(radius):
            nxt = []
            for t in frontier:
                for _d, n in self.map.neighbors(t):
                    if n not in reach and self.land(n):
                        reach.add(n)
                        nxt.append(n)
            frontier = nxt
        return reach

    def summary(self) -> dict:
        """What the next turn's EVENTS diff needs."""
        return {
            "cities": {
                c["id"]: {
                    "name": c.get("name"),
                    "size": c.get("size"),
                    "buildings": sorted(c.get("buildings") or []),
                }
                for c in self.cities
            },
            "units": {u["id"]: u.get("type") for u in self.units},
            "techs": sorted(self.my.get("techs_known") or []),
        }

    def _events(self, prev):
        if not prev:
            self.sections["EVENTS"] = ["(first observed turn)"]
            return
        ev = []
        pc, cur = (
            {int(k): v for k, v in prev.get("cities", {}).items()},
            {c["id"]: c for c in self.cities},
        )
        for cid in sorted(set(cur) - set(pc)):
            ev.append(
                f"city founded/acquired: {cur[cid].get('name')} ({self.city_label[cid]})"
            )
        for cid in sorted(set(pc) - set(cur)):
            ev.append(f"city lost: {pc[cid].get('name')}")
        for cid in sorted(set(cur) & set(pc)):
            if (cur[cid].get("size") or 0) != (pc[cid].get("size") or 0):
                ev.append(
                    f"{self.city_label[cid]} size {pc[cid].get('size')} -> {cur[cid].get('size')}"
                )
            new_b = sorted(
                set(cur[cid].get("buildings") or [])
                - set(pc[cid].get("buildings") or [])
            )
            for b in new_b:
                ev.append(f"{self.city_label[cid]} completed {b}")
        pu = {int(k): v for k, v in prev.get("units", {}).items()}
        cu = {u["id"]: u.get("type") for u in self.units}
        for t, n in sorted(Counter(cu[i] for i in set(cu) - set(pu)).items()):
            ev.append(f"new units: {n} {t}")
        for t, n in sorted(Counter(pu[i] for i in set(pu) - set(cu)).items()):
            ev.append(f"units gone: {n} {t}")
        for tech in sorted(
            set(self.my.get("techs_known") or []) - set(prev.get("techs", []))
        ):
            ev.append(f"learned {tech}")
        ev = list(prev.get("notes", [])) + ev
        caps = self.caps
        self.sections["EVENTS"] = (
            ev[: caps.events]
            + ([f"... {len(ev) - caps.events} more"] if len(ev) > caps.events else [])
        ) or ["no change"]

    def render(self) -> str:
        out = [f"STATE {STATE_SCHEMA}"]
        for name in ("OVERVIEW", "RIVALS", "CITIES", "UNITS", "MAP", "EVENTS"):
            out.append(f"[{name}]")
            out.extend(self.sections[name])
        return "\n".join(out)
