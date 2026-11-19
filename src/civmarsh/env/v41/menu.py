"""Env v4 / v4.1 action menu (schema civm-action-menu-v4) and the v4.1 turn prompt (civm-turn-v41).

Every own city and every idle unit listed in the state gets actions; unit actions include multi-turn travel
(goto / settle via PACKET_UNIT_ORDERS on a BFS path over known tiles), attacks on adjacent refs, worker jobs and
self actions.  Production offers the full ruleset unit/improvement lists through civharness.buildability.

Keys are rendered compactly (one line per entity, `unit:<id>/<verb>[:<ref>]`, `city:<id>/build:<item>`); the
parser accepts exactly the offered concrete keys (strict, no repair).  v4 bundles group keys by domain
(as in civmarsh.env.menu); v4.1 bundles map each entity to one action (parse_entity_bundle).

Live mode (an Observation with legal_actions) asks the server for legality of moves, self actions, founding,
attacks and worker jobs of the listed units.  Offline mode (plain dicts) offers by rule, for tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from civharness import (
    BuildImprovement,
    BuildUnit,
    Buy,
    ChangeGovernment,
    DeclareWar,
    SetRates,
    SetResearch,
    SetTechGoal,
)
from civharness.candidates import DoAction
from civharness.episode import order_as_record
from civharness.policy.client import SetServerSideAgent, UnitOrders
from civmarsh.env.menu import (
    GOVERNMENTS,
    RATE_PRESETS,
    load_tech_requirements,
    reachable_unknown,
    ruleset_dir,
)
from civmarsh.env.v41.observation import (
    MOVE_FRAGS,
    NON_MILITARY,
    RULESET,
    SETTLER_TYPES,
    WORKER_TYPES,
    StateV4,
    production_text,
)

MENU_SCHEMA = "civm-action-menu-v4"
DOMAINS = ("technology", "government", "diplomacy", "city", "civilian", "military")
ORDER_MOVE, ORDER_PERFORM_ACTION = 0, 4
NO_TARGET = -1
SEA_CLASSES = frozenset({"Sea", "Trireme"})
COORD_MAX_PATH = 40  # goto:@x,y path cap (refs use MenuV4Config.max_path)
COORD_RE = re.compile(r"^(goto|settle):@(\d{1,3}),(\d{1,3})$")
REF_RE = re.compile(r"^(goto|settle):([CSTGFH]\d{1,2})$")
# screen compass of the six iso-hex map directions (map +x = screen SE, map +y = screen SW)
SCREEN_DIR = {0: "n", 1: "ne", 3: "nw", 4: "se", 6: "sw", 7: "s"}
ATTACK_ACTIONS = (
    "Attack",
    "Suicide Attack",
    "Conquer City",
    "Conquer City 2",
    "Conquer City 3",
    "Conquer City 4",
    "Bombard",
    "Bombard 2",
    "Bombard 3",
    "Capture Units",
)


@dataclass(frozen=True)
class MenuV4Config:
    max_orders: int = 20  # v4.1: at most 20 orders per turn
    max_technology_orders: int = 2
    max_government_orders: int = 2
    max_diplomacy_orders: int = 1
    max_city_orders: int = 10
    max_civilian_orders: int = 10
    max_military_orders: int = 10
    max_path: int = 24
    goto_cities: int = 2
    buy_min_gold: int = 120

    def domain_cap(self, domain: str) -> int:
        return getattr(self, f"max_{domain}_orders")


@dataclass(frozen=True)
class Cand:
    key: str
    domain: str
    entity: str
    orders: tuple = field(repr=False, compare=False)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "domain": self.domain,
            "entity": self.entity,
            "orders": [order_as_record(o) for o in self.orders],
        }


@dataclass(frozen=True)
class MenuV4:
    candidates: tuple
    lines: tuple  # rendered menu body
    config: MenuV4Config
    schema: str = MENU_SCHEMA

    @property
    def by_key(self) -> dict:
        return {c.key: c for c in self.candidates}

    @property
    def sha(self) -> str:
        raw = json.dumps(
            [c.as_dict() for c in self.candidates],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def entity_action(c) -> tuple[str, str]:
        """(entity, action) of a candidate in the v4.1 entity-keyed format: unit/city keys split at the first '/',
        player-level keys keep their entity and use the value after ':' ("technology/research:masonry" ->
        ("player:research", "masonry")); declare-war keeps its verb."""
        if c.key.startswith(("unit:", "city:")):
            ent, action = c.key.split("/", 1)
            return ent, action
        if c.key.startswith("diplomacy/"):
            return c.entity, c.key.split("/", 1)[1].split(":", 1)[0]
        return c.entity, c.key.split(":", 1)[1]

    @property
    def by_entity_action(self) -> dict:
        return {self.entity_action(c): c for c in self.candidates}

    def entity_schema(self) -> dict:
        """v4.1 constrained-decoding schema: an object whose properties are the
        commandable entities, each a string enum of that entity's offered actions.  One action per entity holds by
        construction (JSON object keys are unique); the 20-order total cap is enforced by the parser."""
        props = {}
        for ent, action in sorted(self.by_entity_action):
            props.setdefault(ent, []).append(action)
        b = getattr(self, "_builder", None)
        labels = (
            sorted(b.s.refs, key=lambda x: (x[0], int(x[1:]))) if b is not None else []
        )
        coord = "@[0-9]{1,3},[0-9]{1,3}"
        goto_pat = "^goto:(" + "|".join(labels + [coord]) + ")$"
        sites = [x for x in labels if x.startswith("S")]
        settle_pat = "^settle:(" + "|".join(sites + [coord]) + ")$"
        settlers = b.settler_entities if b is not None else set()

        def prop(e, a):
            if e.startswith("unit:"):
                alts = [
                    {"type": "string", "enum": a},
                    {"type": "string", "pattern": goto_pat},
                ]
                if b is None or e in settlers:
                    alts.append({"type": "string", "pattern": settle_pat})
                return {"anyOf": alts}
            return {"type": "string", "enum": a}

        return {
            "type": "object",
            "properties": {e: prop(e, a) for e, a in props.items()},
            "additionalProperties": False,
        }

    def json_schema(self) -> dict:
        """Constrained-decoding schema: a JSON object whose domain values are arrays of offered
        keys, at most the domain cap each.  Domains without offered keys are absent (additionalProperties false).
        Not expressible here, and enforced by the parser instead: one key per entity and the total cap."""
        props = {}
        for d in DOMAINS:
            keys = sorted(c.key for c in self.candidates if c.domain == d)
            if keys:
                props[d] = {
                    "type": "array",
                    "items": {"type": "string", "enum": keys},
                    "maxItems": self.config.domain_cap(d),
                }
        return {"type": "object", "properties": props, "additionalProperties": False}

    def render(self) -> str:
        cfg = self.config
        out = [f"ACTION_MENU {self.schema} sha={self.sha}"]
        out.extend(self.lines)
        if getattr(self, "fmt", "domain") == "entity":
            out.append(
                "ACTION_JSON: return one JSON object mapping each entity you want to command to ONE action, e.g. "
                '{"unit:40": "settle:S1", "unit:41": "fortify", "city:12": "build:settlers", "player:research": '
                f'"bronze-working"}}. At most {cfg.max_orders} entries. Entities you leave out keep their current orders. '
                "A line `unit:<id> ...: verb, verb:A,B` offers the actions verb, verb:A, verb:B for that unit; "
                "`city:<id>/build:<item>` offers build:<item>; player-level entities are player:research, "
                "player:tech-goal, player:rates, player:government. Any unit also takes goto:@x,y for a known tile "
                "(coordinates are shown as @x,y), and settlers take settle:@x,y (walk there and found a city); an "
                "unreachable target sends the unit as close as possible. Empty object passes."
            )
            return "\n".join(out)
        out.append(
            'ACTION_JSON: return one object such as {"city":["city:12/build:settlers"],'
            '"civilian":["unit:40/settle:S1"],"military":["unit:41/fortify"]} with at most '
            f"{cfg.max_orders} keys (per domain: "
            + ", ".join(f"{d} {cfg.domain_cap(d)}" for d in DOMAINS)
            + "). "
            "A line `unit:<id> ...: verb, verb:A,B` offers the keys unit:<id>/verb and unit:<id>/verb:A, "
            "unit:<id>/verb:B; `city:<id>/build:<item>` takes one item from that city's list. "
            "At most one key per unit/city/entity. Empty object passes. Orders like goto/settle continue over "
            "later turns until done."
        )
        return "\n".join(out)


class BundleValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ActionBundle:
    keys: tuple
    orders: tuple
    candidate_sha: str
    dropped: tuple = ()  # lenient mode only: (key, reason) pairs that were skipped
    notes: tuple = ()  # v4.1: messages for next turn's EVENTS (e.g. an unreachable goto:@x,y target)


_RULESETS: dict = {}


def _ruleset(name: str):
    """The parsed ruleset (civharness.buildability.Ruleset) of the installed server, loaded once per name."""
    if name not in _RULESETS:
        from civharness.buildability import Ruleset

        _RULESETS[name] = Ruleset.load(ruleset_dir(name))
    return _RULESETS[name]


def _slug(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")


def _move_order(d: int, action_none: int) -> dict:
    return {
        "order": ORDER_MOVE,
        "activity": 0,
        "target": 0,
        "sub_target": 0,
        "action": action_none,
        "dir": d,
    }


class _Builder:
    def __init__(self, obs, state: StateV4, config: MenuV4Config):
        self.obs, self.s, self.cfg = obs, state, config
        self.live = callable(getattr(obs, "legal_actions", None))
        rid = getattr(getattr(obs, "rule_ids", None), "actions", None) or {}
        self.action_ids = dict(rid)
        self.action_none = (max(rid.values()) + 1) if rid else 255
        self.rs = _ruleset(RULESET)
        self.cands: list[Cand] = []
        self.lines: list[str] = []
        # structured view for the v4.1 merged rendering (render_turn_v41)
        self.common_build: list[str] = []
        self.city_extra: dict[int, list[str]] = {}
        self.city_buy: dict[int, bool] = {}
        self.unit_verbs: dict[
            int, list[str]
        ] = {}  # expanded units: verbs other than goto / settle
        self.compact: list[
            tuple[dict, list[str]]
        ] = []  # (unit, verbs other than goto / settle)
        self.settler_entities: set[str] = set()
        self.tech_reach: list[str] = []
        self.tech_next: list[str] = []
        self.rate_values: list[str] = []
        self.gov_values: list[str] = []
        self.dip_keys: list[str] = []
        self.foreign = {o["tile"] for o in state.foreign_units + state.foreign_cities}
        self._trees = {}
        self._extra_actions = None

    def legal(self, uid: int, tile: int, **kw) -> dict:
        if not self.live:
            return {}
        try:
            return self.obs.legal_actions(uid, target_tile=tile, **kw) or {}
        except Exception:  # noqa: BLE001 -- treat a failed query as "nothing legal" (fail closed)
            return {}

    def add(self, key, domain, entity, *orders):
        self.cands.append(Cand(key, domain, entity, tuple(orders)))

    # -- cities -------------------------------------------------------------------
    def cities(self):
        from civharness.buildability import (
            CityContext,
            can_city_build_improvement_now,
            can_city_build_unit_now,
        )

        s, rs = self.s, self.rs
        known = set(s.my.get("techs_known") or [])
        all_b = [b for c in s.cities for b in (c.get("buildings") or [])]
        per_city = {}
        for c in s.cities:
            t = c["tile"]
            adj = [s.tiles.get(n) for _d, n in s.map.neighbors(t)]
            center = s.tiles.get(t) or {}
            ctx = CityContext(
                techs_known=known,
                city_buildings=list(c.get("buildings") or []),
                city_size=c.get("size") or 1,
                all_own_city_buildings=all_b,
                adjacent_terrains={a.get("terrain") for a in adj if a}
                if all(adj)
                else None,
                adjacent_extras={e for a in adj if a for e in (a.get("extras") or [])}
                if all(adj)
                else None,
                center_terrain=center.get("terrain"),
                center_extras=set(center.get("extras") or []) if center else None,
            )
            items = []
            for name, u in sorted(rs.units.items()):
                if (
                    not u.get("no_build")
                    and can_city_build_unit_now(rs, name, ctx) is True
                ):
                    items.append(("unit", name))
            for name, b in sorted(rs.buildings.items()):
                if can_city_build_improvement_now(rs, name, ctx) is True:
                    items.append(("building", name))
            per_city[c["id"]] = [it for it in items if it[1] != c.get("building")]
        common = (
            set.intersection(*(set(v) for v in per_city.values()))
            if per_city
            else set()
        )
        self.common_build = sorted(_slug(n) for _k, n in common)
        if common:
            self.lines.append(
                "[city] one key per city, written city:<id>/build:<option> or city:<id>/buy. Options in every city: "
                + ", ".join(
                    _slug(n) for _k, n in sorted(common, key=lambda x: _slug(x[1]))
                )
            )
        for c in s.cities:
            cid, ent = c["id"], f"city:{c['id']}"
            extra = sorted(set(per_city[cid]) - common, key=lambda x: _slug(x[1]))
            for kind, name in per_city[cid]:
                order = (
                    BuildUnit(cid, name)
                    if kind == "unit"
                    else BuildImprovement(cid, name)
                )
                self.add(f"{ent}/build:{_slug(name)}", "city", ent, order)
            buy = (s.my.get("gold") or 0) >= self.cfg.buy_min_gold and c.get("building")
            if buy:
                self.add(f"{ent}/buy", "city", ent, Buy(cid))
            self.city_extra[cid] = [_slug(n) for _k, n in extra]
            self.city_buy[cid] = bool(buy)
            self.lines.append(
                f"{ent} ({s.city_label[cid]}, now {c.get('building')}): build:<option>"
                + (" plus " + ", ".join(_slug(n) for _k, n in extra) if extra else "")
                + (", buy" if buy else "")
            )

    # -- units --------------------------------------------------------------------
    def passable_for(self, utype):
        s = self.s
        cls = (self.rs.units.get(utype) or {}).get("class")
        sea = cls in SEA_CLASSES
        own_cities = {c["tile"] for c in s.cities}

        def ok(t):
            if t in self.foreign:
                return False
            if sea:
                return s.oceanic(t) or t in own_cities
            return s.land(t)

        return ok, sea

    def recover_city(self, u, self_legal) -> dict | None:
        """The own city a disband would add shields to (Disband Unit Recover: in the city or next to it), or None.
        disband then means recover, which is never worse."""
        if not self.live:
            return None
        t = u["tile"]
        here = next((c for c in self.s.cities if c["tile"] == t), None)
        if here is not None:
            return here if "Disband Unit Recover" in self_legal else None
        for c in self.s.cities:
            if self.s.map.distance(
                t, c["tile"]
            ) == 1 and "Disband Unit Recover" in self.legal(u["id"], c["tile"]):
                return c
        return None

    def unit_orders(self, u, dest, final=None) -> UnitOrders | None:
        ok, _sea = self.passable_for(u.get("type"))
        tree = self._trees.get(u["id"])
        if (
            tree is None
        ):  # one BFS per unit, reused for every target (dest must be passable)
            tree = self._trees[u["id"]] = self.s.map.bfs_tree(
                u["tile"], ok, self.cfg.max_path
            )
        path = self.s.map.path_in_tree(tree, dest) if ok(dest) else None
        if path is None or not path or len(path) > self.cfg.max_path:
            return None
        orders = [_move_order(d, self.action_none) for d in path]
        if final is not None:
            orders.append(final)
        return UnitOrders(u["id"], u["tile"], tuple(orders), dest)

    def units(self):
        s, cfg = self.s, self.cfg
        refs = s.refs
        near_busy, compact_busy = s.busy_split()
        idle = s.idle_units()
        listed = (
            idle[: s.caps.idle_units] + near_busy
        )  # busy units next to enemies are expanded
        compact_busy = (
            compact_busy + idle[s.caps.idle_units :]
        )  # idle units over the cap still get verbs
        unit_lines = {"civilian": [], "military": []}
        for u in listed:
            uid, utype, t = u["id"], u.get("type"), u["tile"]
            ent = f"unit:{uid}"
            dom = "civilian" if utype in NON_MILITARY else "military"
            verbs = []
            self_legal = self.legal(uid, t)
            # self actions
            recover_city = self.recover_city(u, self_legal)
            if recover_city is not None:
                self.add(
                    f"{ent}/disband",
                    dom,
                    ent,
                    DoAction(uid, "Disband Unit Recover", target_id=recover_city["id"]),
                )
                verbs.append(
                    f"disband (adds shields to {s.city_label[recover_city['id']]})"
                )
            for verb, action in (
                ("fortify", "Fortify"),
                ("sentry", "Sentry"),
                ("disband", "Disband Unit"),
            ):
                if verb == "disband" and any(
                    c.key == f"{ent}/disband" for c in self.cands
                ):
                    continue
                if self.live and action not in self_legal:
                    continue
                if not self.live and verb == "fortify" and utype in NON_MILITARY:
                    continue
                self.add(
                    f"{ent}/{verb}",
                    dom,
                    ent,
                    DoAction(
                        uid, action, target_id=None if action != "Found City" else t
                    ),
                )
                verbs.append(verb)
            # city-targeted and terrain actions, live only
            if self.live:
                city_here = next((c for c in s.cities if c["tile"] == t), None)
                if city_here is not None:
                    for verb, action in (
                        ("rehome", "Home City"),
                        ("upgrade", "Upgrade Unit"),
                    ):
                        if action in self_legal:
                            self.add(
                                f"{ent}/{verb}",
                                dom,
                                ent,
                                DoAction(uid, action, target_id=city_here["id"]),
                            )
                            verbs.append(verb)
                if utype in SETTLER_TYPES:
                    for c in s.cities:
                        if s.map.distance(
                            t, c["tile"]
                        ) <= 1 and "Join City" in self.legal(uid, c["tile"]):
                            lab = s.city_label[c["id"]]
                            self.add(
                                f"{ent}/join:{lab}",
                                dom,
                                ent,
                                DoAction(uid, "Join City", target_id=c["id"]),
                            )
                            verbs.append(f"join:{lab}")
                if (
                    utype in WORKER_TYPES or utype in SETTLER_TYPES
                ):  # settlers terraform too (civ2civ3)
                    for verb, action in (
                        ("work:plant", "Plant"),
                        ("work:cultivate", "Cultivate"),
                        ("work:transform", "Transform Terrain"),
                    ):
                        if action in self_legal:
                            self.add(
                                f"{ent}/{verb}",
                                dom,
                                ent,
                                DoAction(uid, action, target_id=t),
                            )
                            verbs.append(verb)
            # engine auto-explore: PACKET_UNIT_SERVER_SIDE_AGENT_SET agent 2; workers and
            # settlers excluded (their moves are economic decisions)
            if utype not in SETTLER_TYPES and utype not in WORKER_TYPES:
                self.add(f"{ent}/explore", dom, ent, SetServerSideAgent(uid, 2))
                verbs.append("explore")
            # found here
            if utype in SETTLER_TYPES and (
                (self.live and "Found City" in self_legal)
                or (not self.live and s.valid_site(t))
            ):
                self.add(
                    f"{ent}/found", dom, ent, DoAction(uid, "Found City", target_id=t)
                )
                verbs.append("found")
            # worker jobs on this tile (live only: needs the ruleset catalog for extra ids); military units may
            # build bases (fortress)
            if self.live and getattr(self.obs, "rulesets", None) is not None:
                if utype in WORKER_TYPES or utype in SETTLER_TYPES:
                    verbs += self.worker_jobs(u)
                elif utype not in NON_MILITARY:
                    verbs += self.worker_jobs(u, bases_only=True)
            # one-step moves
            ok, _sea = self.passable_for(utype)
            moves = []
            for d, n in s.map.neighbors(t):
                if not ok(n):
                    continue
                if self.live and "Unit Move" not in self.legal(uid, n):
                    continue
                name = SCREEN_DIR[d]
                self.add(
                    f"{ent}/move:{name}",
                    dom,
                    ent,
                    DoAction(uid, "Unit Move", target_id=n),
                )
                moves.append(name)
            if moves:
                verbs.append("move:" + ",".join(moves))
            # attack any adjacent enemy by direction (enemies away from our cities are not THREATS refs,
            # so ref-only attacks would miss them)
            if utype not in NON_MILITARY:
                atk = []
                foreign_city = {c["tile"] for c in s.foreign_cities}
                for d, n in s.map.neighbors(t):
                    if n not in self.foreign:
                        continue
                    legal = self.legal(uid, n)
                    action = next((a for a in ATTACK_ACTIONS if a in legal), None)
                    if action is None and not self.live:
                        action = "Conquer City" if n in foreign_city else "Attack"
                    if action is None:
                        continue
                    name = SCREEN_DIR[d]
                    self.add(
                        f"{ent}/attack:{name}",
                        "military",
                        ent,
                        DoAction(uid, action, target_id=n),
                    )
                    atk.append(name)
                if atk:
                    verbs.append("attack:" + ",".join(atk))
            # settle at a site (settlers): path + Found City on arrival
            if utype in SETTLER_TYPES:
                fid = self.action_ids.get("Found City")
                st = []
                for lab, r in sorted(refs.items()):
                    if r.kind != "site" or r.tile == t:
                        continue
                    final = {
                        "order": ORDER_PERFORM_ACTION,
                        "activity": 0,
                        "target": r.tile,
                        "sub_target": NO_TARGET,
                        "action": fid if fid is not None else self.action_none,
                        "dir": -1,
                    }
                    o = self.unit_orders(u, r.tile, final)
                    if o is not None:
                        self.add(f"{ent}/settle:{lab}", dom, ent, o)
                        st.append(lab)
                if st:
                    verbs.append("settle:" + ",".join(st))
            # goto: nearest own cities, sites, explore points, huts, approach targets
            go = []
            cities = sorted(
                (r for r in refs.values() if r.kind == "city"),
                key=lambda r: (s.map.distance(t, r.tile), r.label),
            )[: cfg.goto_cities]
            others = [
                r
                for _l, r in sorted(refs.items())
                if r.kind in ("site", "frontier", "hut")
            ]
            for r in cities + others:
                if r.tile == t:
                    continue
                o = self.unit_orders(u, r.tile)
                if o is not None:
                    self.add(f"{ent}/goto:{r.label}", dom, ent, o)
                    go.append(r.label)
            if utype not in NON_MILITARY:
                for lab, r in sorted(refs.items()):
                    if r.kind != "target" or s.map.distance(t, r.tile) <= 1:
                        continue
                    ok, _ = self.passable_for(utype)
                    adj = sorted(
                        (n for _d, n in s.map.neighbors(r.tile) if ok(n)),
                        key=lambda n: (s.map.distance(t, n), n),
                    )
                    for n in adj[:1]:
                        o = self.unit_orders(u, n)
                        if o is not None:
                            self.add(f"{ent}/goto:{lab}", dom, ent, o)
                            go.append(lab)
            if go:
                verbs.append("goto:" + ",".join(go))
            self.unit_verbs[uid] = [
                v for v in verbs if not v.startswith(("goto:", "settle:"))
            ]
            if utype in SETTLER_TYPES:
                self.settler_entities.add(ent)
            unit_lines[dom].append(
                f"{ent} {utype} ({s.where(t)}): "
                + ("; ".join(verbs) if verbs else "(no action)")
            )
        # busy units elsewhere: compact, re-orderable; identical verb sets share one line
        groups = {}
        for u in compact_busy:
            uid, utype, t = u["id"], u.get("type"), u["tile"]
            ent = f"unit:{uid}"
            dom = "civilian" if utype in NON_MILITARY else "military"
            act = s._activity(u)
            verbs = []
            go = []
            cities = sorted(
                (r for r in refs.values() if r.kind == "city"),
                key=lambda r: (s.map.distance(t, r.tile), r.label),
            )[: cfg.goto_cities]
            extra = (
                [
                    r
                    for _l, r in sorted(refs.items())
                    if r.kind in ("target", "frontier", "hut", "site")
                ]
                if dom == "military"
                else []
            )
            for r in cities + extra:
                if r.tile == t:
                    continue
                o = self.unit_orders(u, r.tile)
                if o is not None:
                    self.add(f"{ent}/goto:{r.label}", dom, ent, o)
                    go.append(r.label)
            if go:
                verbs.append("goto:" + ",".join(go))
            if utype in SETTLER_TYPES:
                fid = self.action_ids.get("Found City")
                st = []
                for lab, r in sorted(refs.items()):
                    if r.kind != "site" or r.tile == t:
                        continue
                    final = {
                        "order": ORDER_PERFORM_ACTION,
                        "activity": 0,
                        "target": r.tile,
                        "sub_target": NO_TARGET,
                        "action": fid if fid is not None else self.action_none,
                        "dir": -1,
                    }
                    o = self.unit_orders(u, r.tile, final)
                    if o is not None:
                        self.add(f"{ent}/settle:{lab}", dom, ent, o)
                        st.append(lab)
                if st:
                    verbs.append("settle:" + ",".join(st))
            if dom == "military" and act not in ("Fortified", "Fortifying"):
                self.add(
                    f"{ent}/fortify", dom, ent, DoAction(uid, "Fortify", target_id=None)
                )
                verbs.append("fortify")
            if (
                utype not in SETTLER_TYPES
                and utype not in WORKER_TYPES
                and act != "Explore"
            ):
                self.add(f"{ent}/explore", dom, ent, SetServerSideAgent(uid, 2))
                verbs.append("explore")
            if self.live:  # one self-legality query per compact unit
                self_legal = self.legal(uid, t)
                city_here = next((c for c in s.cities if c["tile"] == t), None)
                if city_here is not None:
                    for verb, action in (
                        ("rehome", "Home City"),
                        ("upgrade", "Upgrade Unit"),
                    ):
                        if action in self_legal:
                            self.add(
                                f"{ent}/{verb}",
                                dom,
                                ent,
                                DoAction(uid, action, target_id=city_here["id"]),
                            )
                            verbs.append(verb)
                if utype in WORKER_TYPES or utype in SETTLER_TYPES:
                    for verb, action in (
                        ("work:plant", "Plant"),
                        ("work:cultivate", "Cultivate"),
                        ("work:transform", "Transform Terrain"),
                    ):
                        if action in self_legal:
                            self.add(
                                f"{ent}/{verb}",
                                dom,
                                ent,
                                DoAction(uid, action, target_id=t),
                            )
                            verbs.append(verb)
                    if getattr(self.obs, "rulesets", None) is not None:
                        verbs += self.worker_jobs(u)
                elif (
                    utype not in NON_MILITARY
                    and getattr(self.obs, "rulesets", None) is not None
                ):
                    verbs += self.worker_jobs(u, bases_only=True)
                rc = self.recover_city(u, self_legal)
                if rc is not None:
                    self.add(
                        f"{ent}/disband",
                        dom,
                        ent,
                        DoAction(uid, "Disband Unit Recover", target_id=rc["id"]),
                    )
            self.add(
                f"{ent}/disband",
                dom,
                ent,
                DoAction(uid, "Disband Unit", target_id=None),
            )
            verbs.append("disband")
            self.compact.append(
                (u, [v for v in verbs if not v.startswith(("goto:", "settle:"))])
            )
            if utype in SETTLER_TYPES:
                self.settler_entities.add(ent)
            key = (dom, s.where(t), tuple(verbs))
            groups.setdefault(key, []).append(f"{ent} {utype} ({act.lower()})")
        for (dom, loc, verbs), members in sorted(groups.items()):
            unit_lines[dom].append(
                f"busy {loc}: "
                + ", ".join(members)
                + " -> each may take: "
                + "; ".join(verbs)
            )
        for dom in ("civilian", "military"):
            if unit_lines[dom]:
                self.lines.append(
                    f"[{dom}] one key per unit, written unit:<id>/<verb> or unit:<id>/<verb>:<ref>"
                )
                self.lines.extend(unit_lines[dom])

    def settle_coord(self, uid: int, x: int, y: int):
        """v4.1 settle:@x,y for settlers: path to the tile and Found City on arrival (one key for any site, so every
        AI founding is expressible).  Unreachable / invalid targets fall back like goto:@x,y (no founding)."""
        s = self.s
        u = next((v for v in s.units if v["id"] == uid), None)
        if u is None or u.get("type") not in SETTLER_TYPES:
            return None, f"U{uid}: settle @{x},{y} needs a settler"
        if not (0 <= x < s.map.xsize and 0 <= y < s.map.ysize):
            return None, f"U{uid}: settle @{x},{y} is outside the map; unit stays"
        tgt = s.map.index(x, y)
        if not s.valid_site(tgt):
            order, note = self.goto_coord(uid, x, y)
            return order, (
                note
                or f"U{uid}: @{x},{y} is not a valid city site (citymindist 3 / terrain / owner); moving there without founding"
            )
        ok, _sea = self.passable_for(u.get("type"))
        path = s.map.path(u["tile"], tgt, ok)
        if path is None or len(path) > COORD_MAX_PATH:
            return self.goto_coord(uid, x, y)
        fid = self.action_ids.get("Found City")
        final = {
            "order": ORDER_PERFORM_ACTION,
            "activity": 0,
            "target": tgt,
            "sub_target": NO_TARGET,
            "action": fid if fid is not None else self.action_none,
            "dir": -1,
        }
        return UnitOrders(
            uid,
            u["tile"],
            tuple(_move_order(d, self.action_none) for d in path) + (final,),
            tgt,
        ), None

    def goto_coord(self, uid: int, x: int, y: int):
        """v4.1 goto:@x,y: a path to any known tile; if the target is unknown, impassable
        or unreachable, head for the reachable known tile nearest to it and say so in a note.
        Returns (UnitOrders | None, note | None)."""
        s = self.s
        u = next((v for v in s.units if v["id"] == uid), None)
        if not (0 <= x < s.map.xsize and 0 <= y < s.map.ysize):
            return (
                None,
                f"U{uid}: goto @{x},{y} is outside the map (x < {s.map.xsize}, y < {s.map.ysize}); unit stays",
            )
        tgt = s.map.index(x, y)
        if u is None or tgt is None:
            return (
                None,
                f"U{uid}: goto @{x},{y} is not a unit of ours or not on the map",
            )
        ok, _sea = self.passable_for(u.get("type"))
        if tgt in s.tiles and ok(tgt):
            path = s.map.path(u["tile"], tgt, ok)
            if path and len(path) <= COORD_MAX_PATH:
                return self._orders(u, path, tgt), None
        reach, frontier, dist = {u["tile"]: 0}, [u["tile"]], 0
        while frontier and dist < COORD_MAX_PATH:
            dist += 1
            nxt = []
            for t in frontier:
                for _d, n in s.map.neighbors(t):
                    if n not in reach and n in s.tiles and ok(n):
                        reach[n] = dist
                        nxt.append(n)
            frontier = nxt
        best = min(reach, key=lambda t: (s.map.distance(t, tgt), reach[t], t))
        if best == u["tile"]:
            return (
                None,
                f"U{uid}: target @{x},{y} unreachable and nothing closer is reachable; unit stays",
            )
        path = s.map.path(u["tile"], best, ok)
        return self._orders(u, path, best), (
            f"U{uid}: target @{x},{y} unreachable; heading to "
            f"{s.coord(best)} (nearest reachable)"
        )

    def _orders(self, u, path, dest):
        return UnitOrders(
            u["id"],
            u["tile"],
            tuple(_move_order(d, self.action_none) for d in path),
            dest,
        )

    def worker_jobs(self, u, bases_only: bool = False) -> list[str]:
        uid, t = u["id"], u["tile"]
        out = []
        rulesets = self.obs.rulesets
        try:
            extras = (
                rulesets.extras_by_cause("Base")
                if bases_only
                else rulesets.worker_target_extras()
            )
        except Exception:  # noqa: BLE001
            return out
        rid = self.action_ids
        # the probe with a target extra also reports actions that ignore the extra (Plant, Cultivate, ...), so
        # each extra only accepts the build action of its own cause (otherwise busy workers would be
        # offered work:railroad / work:maglev that execute Plant)
        if self._extra_actions is None:
            self._extra_actions = {}
            for cause, action in (
                ("Irrigation", "Build Irrigation"),
                ("Mine", "Build Mine"),
                ("Road", "Build Road"),
                ("Base", "Build Base"),
            ):
                try:
                    for rec in rulesets.extras_by_cause(cause):
                        self._extra_actions.setdefault(rec["id"], set()).add(action)
                except Exception:  # noqa: BLE001, S110 -- a cause missing from this ruleset offers no job
                    pass
        for ex in extras:
            allowed = self._extra_actions.get(ex["id"], set())
            if not allowed:
                continue
            legal = self.legal(uid, t, target_extra=ex["id"])
            for action_name in sorted(legal):
                if action_name not in allowed or action_name not in rid:
                    continue
                verb = f"work:{_slug(ex['rule_name'])}"
                key = f"unit:{uid}/{verb}"
                if any(c.key == key for c in self.cands):
                    continue
                dom = "civilian" if u.get("type") in NON_MILITARY else "military"
                self.add(
                    key,
                    dom,
                    f"unit:{uid}",
                    DoAction(uid, action_name, target_id=t, sub_target=ex["id"]),
                )
                out.append(verb)
        return out

    # -- player-level ---------------------------------------------------------------
    def player(self):
        s = self.s
        known = set(s.my.get("techs_known") or [])
        reach = reachable_unknown(known)
        reqs = load_tech_requirements()
        nxt = sorted(
            t
            for t, (a, b) in reqs.items()
            if t not in known
            and t not in reach
            and (a is None or a in known or a in reach)
            and (b is None or b in known or b in reach)
        )
        self.tech_reach = [_slug(x) for x in reach]
        self.tech_next = [_slug(x) for x in nxt]
        for tech in reach:
            self.add(
                f"technology/research:{_slug(tech)}",
                "technology",
                "player:research",
                SetResearch(tech),
            )
        for tech in reach + nxt:
            self.add(
                f"technology/goal:{_slug(tech)}",
                "technology",
                "player:tech-goal",
                SetTechGoal(tech),
            )
        self.lines.append(
            "[technology] technology/research:<tech> (entity player:research) with <tech> in: "
            + ", ".join(_slug(x) for x in reach)
        )
        self.lines.append(
            "[technology] technology/goal:<tech> (entity player:tech-goal) with <tech> in: "
            + ", ".join(_slug(x) for x in reach + nxt)
        )
        gov_keys = []
        for tax, lux, sci in RATE_PRESETS:
            k = f"government/rates:{tax}-{lux}-{sci}"
            self.add(k, "government", "player:rates", SetRates(tax, lux, sci))
            gov_keys.append(k)
            self.rate_values.append(f"{tax}-{lux}-{sci}")
        cur = s.my.get("government")
        for g, req in GOVERNMENTS:
            if req in known and g != cur:
                k = f"government/change:{_slug(g)}"
                self.add(k, "government", "player:government", ChangeGovernment(g))
                gov_keys.append(k)
                self.gov_values.append(_slug(g))
        self.lines.append(
            "[government] (entities player:rates, player:government) "
            + ", ".join(gov_keys)
        )
        dip = []
        for pid_s, rel in sorted((s.diplomacy or {}).items(), key=lambda x: int(x[0])):
            if (rel or {}).get("state") in ("war", "no_contact", "team", None):
                continue
            k = f"diplomacy/declare-war:{int(pid_s)}"
            self.add(
                k, "diplomacy", f"player:diplomacy:{int(pid_s)}", DeclareWar(int(pid_s))
            )
            dip.append(k)
            self.dip_keys.append(f"player:diplomacy:{int(pid_s)} -> declare-war")
        self.lines.append("[diplomacy] " + (", ".join(dip) if dip else "(none)"))


def build_menu_v4(obs, state: StateV4, config: MenuV4Config | None = None) -> MenuV4:
    cfg = config or MenuV4Config()
    b = _Builder(obs, state, cfg)
    b.player()
    b.cities()
    b.units()
    seen, cands = set(), []
    for c in b.cands:
        if c.key not in seen:
            seen.add(c.key)
            cands.append(c)
    menu = MenuV4(tuple(cands), tuple(b.lines), cfg)
    object.__setattr__(menu, "_builder", b)  # resolves goto:@x,y at parse time
    return menu


def parse_action_bundle_v4(
    text: str, menu: MenuV4, *, lenient: bool = False
) -> ActionBundle:
    """Strict JSON parser + conflict validator; never repairs model output.

    lenient=True (data generation): any JSON object is accepted; every string key in any list is
    matched against the offered set regardless of the domain it was filed under; unknown keys, a second key for
    the same entity, and keys beyond the domain / total caps are dropped (recorded in .dropped).  Unparsable JSON
    still raises."""
    if lenient:
        return _parse_lenient(text, menu)
    cfg = menu.config
    try:
        raw = json.loads((text or "").strip())
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise BundleValidationError("bundle must be a JSON object")
    unknown = set(raw) - set(DOMAINS)
    if unknown:
        raise BundleValidationError(f"unknown domains: {sorted(unknown)}")
    selected = []
    for d in DOMAINS:
        keys = raw.get(d, [])
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise BundleValidationError(f"{d} must be a list of keys")
        if len(keys) > cfg.domain_cap(d):
            raise BundleValidationError(
                f"{d} selects {len(keys)} > cap {cfg.domain_cap(d)}"
            )
        selected += [(d, k) for k in keys]
    if len(selected) > cfg.max_orders:
        raise BundleValidationError(
            f"bundle selects {len(selected)} > cap {cfg.max_orders}"
        )
    offered = menu.by_key
    cands = []
    for d, k in selected:
        c = offered.get(k)
        if c is None:
            raise BundleValidationError(f"key was not offered: {k}")
        if c.domain != d:
            raise BundleValidationError(f"key {k} belongs to {c.domain}, not {d}")
        cands.append(c)
    ents = [c.entity for c in cands]
    if len(ents) != len(set(ents)) or len({c.key for c in cands}) != len(cands):
        raise BundleValidationError("more than one key for the same entity")
    return ActionBundle(
        tuple(c.key for c in cands), tuple(o for c in cands for o in c.orders), menu.sha
    )


def obs_block_v4(
    obs,
    *,
    endturn: int = 120,
    prev: dict | None = None,
    config: MenuV4Config | None = None,
    fmt: str = "domain",
):
    """(prompt text, menu, state) for one decision.  fmt "entity" = the v4.1 entity-keyed output format."""
    state = StateV4(obs, endturn=endturn, prev=prev)
    menu = build_menu_v4(obs, state, config)
    object.__setattr__(menu, "fmt", fmt)
    if fmt == "entity":
        return render_turn_v41(state, menu), menu, state
    return state.render() + "\n" + menu.render(), menu, state


TURN_SCHEMA = "civm-turn-v41"


def render_turn_v41(state: StateV4, menu: MenuV4) -> str:
    """v4.1 prompt: state and actions merged, one line per city and per unit.
    goto / settle targets are not listed per unit: any unit takes goto:<any ref>|goto:@x,y and settlers take
    settle:<S ref>|settle:@x,y (unreachable -> nearest reachable tile, decision #19).  Same information as
    StateV4.render() + MenuV4.render(); the constrained-decoding schema (entity_schema) is unchanged in meaning."""
    s, b, cfg = state, menu._builder, menu.config
    sec = s.sections
    out = [f"TURN {TURN_SCHEMA}", "[OVERVIEW]"] + sec["OVERVIEW"]
    out += ["[RIVALS]"] + sec["RIVALS"]
    out += [
        "[MAP] refs usable as targets: C city, S site, T threat, G enemy city, F explore point, H hut"
    ] + sec["MAP"]
    out += ["[EVENTS]"] + sec["EVENTS"]
    # cities
    mil_on = Counter(u["tile"] for u in s.units if s._military(u))
    out.append(
        "[CITIES] each city takes one action: build:<item> or buy. Items every city can build: "
        + (", ".join(b.common_build) or "none")
        + ". Extra items per city after 'also'."
    )
    for c in s.cities[: s.caps.cities]:
        cid = c["id"]
        thr = sorted(
            s.map.distance(c["tile"], u["tile"])
            for u in s.foreign_units
            if s._military(u)
            and s.map.distance(c["tile"], u["tile"]) <= s.caps.city_threat_radius
        )
        line = (
            f"city:{cid} {s.city_label[cid]} {c.get('name')} {s.coord(c['tile'])} | size {c.get('size')} | "
            f"building {production_text(c)} | "
            f"food {c.get('food_stock')} | buildings {len(c.get('buildings') or [])} | "
            f"defenders {mil_on.get(c['tile'], 0)}"
        )
        if thr:
            line += f" | enemy unit at {thr[0]}"
        if b.city_extra.get(cid):
            line += " | also " + ", ".join(b.city_extra[cid])
        if b.city_buy.get(cid):
            line += " | buy"
        out.append(line)
    if len(s.cities) > s.caps.cities:
        out.append(f"... {len(s.cities) - s.caps.cities} more cities")
    if not s.cities:
        out.append("none")
    # units
    out.append(
        "[UNITS] each unit takes one action from its list. Every unit also takes goto:<ref> (enemy refs: "
        "move next to it) or goto:@x,y; settlers also take settle:<S ref> or settle:@x,y (walk there, found "
        "a city). An unreachable target sends the unit as close as possible. Orders last until done."
    )
    foreign_tiles = {o["tile"] for o in s.foreign_units + s.foreign_cities}
    by_id = {u["id"]: u for u in s.units}
    expanded = {}  # units with the same type, tile, status and actions share one line
    for uid, verbs in b.unit_verbs.items():
        u = by_id[uid]
        t = u["tile"]
        act = s._activity(u)
        flags = [] if act == "Idle" else [act.lower()]
        enemy_dirs = [
            SCREEN_DIR[d] for d, n in s.map.neighbors(t) if n in foreign_tiles
        ]
        if enemy_dirs:
            flags.append("enemy at " + ",".join(enemy_dirs))
        mv = (u.get("moves") or 0) / MOVE_FRAGS
        head = (
            f"{u.get('type')} {s.coord(t)} {s.where(t)} | "
            + (f"hp {u.get('hp')} " if s._military(u) else "")
            + f"mv {mv:g}"
            + "".join(f" | {f}" for f in flags)
        )
        expanded.setdefault((head, tuple(verbs)), []).append(f"unit:{uid}")
    for (head, verbs), ids in expanded.items():
        out.append(
            ", ".join(ids) + f" {head}: " + (", ".join(verbs) if verbs else "goto only")
        )
    if b.compact:
        out.append("busy or extra units (same rules; no key = keep current orders):")
        groups = {}
        for u, verbs in b.compact:
            groups.setdefault(
                (u.get("type"), s._activity(u).lower(), tuple(verbs)), []
            ).append(f"unit:{u['id']} {s.coord(u['tile'])}")
        for (utype, act, verbs), members in sorted(groups.items()):
            out.append(
                f"{utype} {act}: " + ", ".join(members) + " | " + ", ".join(verbs)
            )
    if not s.units:
        out.append("none")
    # player
    out.append("[PLAYER] player:research: " + (", ".join(b.tech_reach) or "none"))
    out.append("player:tech-goal: " + (", ".join(b.tech_reach + b.tech_next) or "none"))
    out.append(
        "player:rates (tax-lux-sci): "
        + ", ".join(b.rate_values)
        + (" | player:government: " + ", ".join(b.gov_values) if b.gov_values else "")
    )
    out += b.dip_keys
    out.append(
        f'ANSWER: one JSON object mapping each entity you command to ONE action, e.g. {{"unit:40": "settle:S1", '
        f'"unit:41": "goto:@12,30", "city:12": "build:settlers", "player:research": "bronze-working"}}. At most '
        f"{cfg.max_orders} entries; entities left out keep their current orders; {{}} passes."
    )
    return "\n".join(out)


def _parse_lenient(text: str, menu: MenuV4) -> ActionBundle:
    cfg = menu.config
    raw_text = (text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text[raw_text.find("{") :] if "{" in raw_text else raw_text
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        i, j = raw_text.find("{"), raw_text.rfind("}")
        try:
            raw = json.loads(raw_text[i : j + 1]) if i >= 0 and j > i else None
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        raise BundleValidationError("invalid JSON object")
    offered = menu.by_key
    keys = [
        k for v in raw.values() if isinstance(v, list) for k in v if isinstance(k, str)
    ]
    chosen, dropped, ents, per = [], [], set(), {}
    for k in keys:
        c = offered.get(k) or offered.get(_normalize_key(k))
        if c is None:
            dropped.append((k, "not offered"))
        elif c.entity in ents:
            dropped.append((k, "second key for entity"))
        elif (
            per.get(c.domain, 0) >= cfg.domain_cap(c.domain)
            or len(chosen) >= cfg.max_orders
        ):
            dropped.append((k, "over cap"))
        else:
            chosen.append(c)
            ents.add(c.entity)
            per[c.domain] = per.get(c.domain, 0) + 1
    return ActionBundle(
        tuple(c.key for c in chosen),
        tuple(o for c in chosen for o in c.orders),
        menu.sha,
        tuple(dropped),
        (),  # the lenient domain parser produces no notes
    )


_NAME_HEADS = ("/build", "/research", "/goal", "/change")


def _normalize_key(k: str) -> str:
    """Lenient mode only: fill a missing technology/ or government/ prefix and slug a display name after build /
    research / goal / change ("technology/research:Bronze Working" -> "technology/research:bronze-working").
    Refs (S1, C2, ...) and unit verbs are left as written."""
    k = k.strip()
    if k.startswith(("research:", "goal:")):
        k = "technology/" + k
    elif k.startswith(("rates:", "change:")):
        k = "government/" + k
    head, sep, tail = k.rpartition(":")
    if sep and head.endswith(_NAME_HEADS):
        k = f"{head}:{_slug(tail)}"
    return k


def parse_entity_bundle(
    text: str, menu: MenuV4, *, lenient: bool = True
) -> ActionBundle:
    """v4.1 entity-keyed bundle: {"unit:330": "settle:S1", "city:149": "build:warriors", "player:research":
    "masonry", ...}.  Unknown (entity, action) pairs are dropped (lenient) or raise (strict); at most
    config.max_orders entries are applied, in the order written."""
    raw_text = (text or "").strip()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        i, j = raw_text.find("{"), raw_text.rfind("}")
        try:
            raw = json.loads(raw_text[i : j + 1]) if i >= 0 and j > i else None
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        raise BundleValidationError("invalid JSON object")
    table = menu.by_entity_action
    entities = {e for e, _a in table}
    chosen, dropped, notes = [], [], []
    for ent, action in raw.items():
        m = COORD_RE.match(action) if isinstance(action, str) else None
        if (
            m
            and ent.startswith("unit:")
            and ent in entities
            and getattr(menu, "_builder", None) is not None
        ):
            if len(chosen) >= menu.config.max_orders:
                dropped.append((f"{ent}/{action}", "over cap"))
                continue
            fn = (
                menu._builder.settle_coord
                if m.group(1) == "settle"
                else menu._builder.goto_coord
            )
            order, note = fn(int(ent.split(":")[1]), int(m.group(2)), int(m.group(3)))
            if note:
                notes.append(note)
            if order is not None:
                dom = next(c.domain for (e, _a), c in table.items() if e == ent)
                chosen.append(Cand(f"{ent}/{action}", dom, ent, (order,)))
            continue
        c = table.get((ent, action)) if isinstance(action, str) else None
        r = REF_RE.match(action) if isinstance(action, str) and c is None else None
        if (
            r
            and ent.startswith("unit:")
            and ent in entities
            and getattr(menu, "_builder", None) is not None
        ):
            # v4.1: goto / settle take any ref; refs without a precomputed path resolve like coordinates
            # (nearest reachable tile, with a note for next turn's EVENTS)
            ref = menu._builder.s.refs.get(r.group(2))
            if ref is not None and (r.group(1) == "goto" or ref.kind == "site"):
                if len(chosen) >= menu.config.max_orders:
                    dropped.append((f"{ent}/{action}", "over cap"))
                    continue
                x, y = menu._builder.s.map.native(ref.tile)
                fn = (
                    menu._builder.settle_coord
                    if r.group(1) == "settle"
                    else menu._builder.goto_coord
                )
                order, note = fn(int(ent.split(":")[1]), x, y)
                if note:
                    notes.append(note.replace(f"@{x},{y}", f"{r.group(2)} @{x},{y}"))
                if order is not None:
                    dom = next(cc.domain for (e, _a), cc in table.items() if e == ent)
                    chosen.append(Cand(f"{ent}/{action}", dom, ent, (order,)))
                continue
        if c is None and isinstance(action, str):
            c = table.get(
                (ent, _slug(action))
            )  # display-name tolerance for build / research items
        if c is None:
            if not lenient:
                raise BundleValidationError(f"not offered: {ent} -> {action}")
            dropped.append((f"{ent}:{action}", "not offered"))
        elif len(chosen) >= menu.config.max_orders:
            dropped.append((c.key, "over cap"))
        else:
            chosen.append(c)
    return ActionBundle(
        tuple(c.key for c in chosen),
        tuple(o for c in chosen for o in c.orders),
        menu.sha,
        tuple(dropped),
        tuple(notes),
    )
