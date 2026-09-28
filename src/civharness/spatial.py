"""Deterministic SAGA-style spatial views over :class:`Observation`.

This module deliberately contains no prompt text and makes no LLM calls.  It
turns the facts already present in an observation into three reusable pieces:

* a map-semantic scene graph (cities/units plus anchor, proximity, attack and
  threat edges);
* a bounded ``(m, g, u, alpha)`` digest; and
* the three read-only query tools described by SAGA.

The SAGA paper specifies the distance bands but does not publish numeric values
for its two tuned radii.  They are therefore explicit configuration here and
are stamped into every graph.  ``saga_manhattan`` is the paper-comparison mode;
``wrapped_manhattan`` is available when a caller knows the Freeciv topology.

Visibility boundary: graph construction uses exactly *what the supplied
Observation contains*.  Under ``visibility: player_visible`` the entities come
from the acting client's own player view, so enemies the server never sent are
absent; under ``full_state`` they come from the save-derived full state.  The
terrain summary in the digest reads ``obs.terrain``, the save-derived grid,
which a ``player_visible`` observation does not carry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# Schema identifiers. Both are part of the hashed graph/digest payloads, so
# changing either changes every graph and digest sha derived from them.
SCENE_SCHEMA = "civharness-scene-v1"
DIGEST_SCHEMA = "civharness-bounded-digest-v1"

_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "HERE")


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _content_sha(payload: Mapping) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SceneGraphConfig:
    """All choices that can change graph membership or labels.

    SAGA reports that ``proximity_radius`` and ``influence_radius`` were tuned
    and frozen, but does not state their values in the paper.  The defaults are
    conservative project values, not claimed paper constants.  Experiments
    comparing against released SAGA code should override them with that code's
    values and retain the resulting config in provenance.
    """

    proximity_radius: int = 5
    influence_radius: int = 5
    close_below: int = 3
    medium_below: int = 10
    distance_mode: str = "saga_manhattan"
    wrap_x: bool = False
    wrap_y: bool = False
    civilian_unit_types: tuple[str, ...] = ("Settlers", "Workers", "Engineers")
    max_city_status: int = 12
    max_alerts: int = 8

    def __post_init__(self):
        if self.distance_mode not in {"saga_manhattan", "wrapped_manhattan"}:
            raise ValueError("distance_mode must be saga_manhattan|wrapped_manhattan")
        if min(self.proximity_radius, self.influence_radius) < 0:
            raise ValueError("scene-graph radii must be non-negative")
        if not 0 < self.close_below < self.medium_below:
            raise ValueError("distance bands require 0 < close_below < medium_below")

    @classmethod
    def saga(cls, **overrides) -> SceneGraphConfig:
        """Paper-compatible geometry with explicit, caller-overridable radii."""
        return cls(distance_mode="saga_manhattan", **overrides)

    def as_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))

    @property
    def sha(self) -> str:
        return _content_sha(self.as_dict())


@dataclass(frozen=True)
class SceneNode:
    """A city or unit placed on the map, tagged self/enemy."""

    key: str
    kind: str
    entity_id: int
    owner: str
    relation: str
    x: int
    y: int
    military: bool = False
    attributes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))


@dataclass(frozen=True)
class SceneEdge:
    """A directed relation between two nodes with its distance band."""

    source: str
    target: str
    relation: str
    distance: int
    band: str
    direction: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SceneGraph:
    """Nodes, edges and the config that produced them, content-hashed."""

    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    config: SceneGraphConfig
    schema: str = SCENE_SCHEMA
    observation_scope: str = "save_derived_full_state"

    def as_dict(self) -> dict:
        payload = {
            "schema": self.schema,
            "observation_scope": self.observation_scope,
            "config": self.config.as_dict(),
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
        }
        return {**payload, "sha": _content_sha(payload)}

    @property
    def sha(self) -> str:
        return self.as_dict()["sha"]

    def entity_view(self, key: str) -> dict:
        """One entity plus only its incident relations, as SAGA controllers see."""
        node = next((n for n in self.nodes if n.key == key), None)
        if node is None:
            raise KeyError(key)
        edges = [e for e in self.edges if e.source == key or e.target == key]
        return {
            "schema": self.schema,
            "node": node.as_dict(),
            "edges": [e.as_dict() for e in edges],
        }


def _map_size(obs) -> tuple[int | None, int | None]:
    terrain = getattr(obs, "terrain", None) or {}
    return terrain.get("xsize"), terrain.get("ysize")


def _signed_delta(a: int, b: int, size: int | None, wrap: bool) -> int:
    delta = b - a
    if not wrap or not size:
        return delta
    alternatives = (delta, delta - size, delta + size)
    return min(alternatives, key=lambda d: (abs(d), d))


def _delta(
    a: SceneNode, b: SceneNode, obs, config: SceneGraphConfig
) -> tuple[int, int]:
    xs, ys = _map_size(obs)
    wrapped = config.distance_mode == "wrapped_manhattan"
    return (
        _signed_delta(a.x, b.x, xs, wrapped and config.wrap_x),
        _signed_delta(a.y, b.y, ys, wrapped and config.wrap_y),
    )


def _geometry(a: SceneNode, b: SceneNode, obs, config: SceneGraphConfig):
    dx, dy = _delta(a, b, obs, config)
    distance = abs(dx) + abs(dy)
    if distance < config.close_below:
        band = "close"
    elif distance < config.medium_below:
        band = "medium"
    else:
        band = "far"
    direction = _direction(dx, dy)
    return distance, band, direction


def _direction(dx: int, dy: int) -> str:
    vertical = "N" if dy < 0 else "S" if dy > 0 else ""
    horizontal = "W" if dx < 0 else "E" if dx > 0 else ""
    return vertical + horizontal or "HERE"


def _entity_nodes(obs, config: SceneGraphConfig) -> list[SceneNode]:
    pview = getattr(obs, "player_view", None)
    if pview is not None:
        return _entity_nodes_from_view(obs, pview, config)
    nodes = []
    for owner, player in sorted(obs.players.items()):
        if not player.get("is_alive", True):
            continue
        owner_id = player.get("player_id", owner)
        relation = "self" if owner == obs.me else "enemy"
        for kind, plural in (("city", "cities"), ("unit", "units")):
            entities = sorted(
                player.get(plural) or [],
                key=lambda x: (x.get("id") is None, x.get("id", 0)),
            )
            for entity in entities:
                entity_id = entity.get("id")
                x, y = entity.get("x"), entity.get("y")
                if entity_id is None or x is None or y is None:
                    continue
                unit_type = entity.get("type")
                military = (
                    kind == "unit" and unit_type not in config.civilian_unit_types
                )
                attrs = {k: v for k, v in entity.items() if k not in {"id", "x", "y"}}
                nodes.append(
                    SceneNode(
                        key=f"{kind}:{owner_id}:{entity_id}",
                        kind=kind,
                        entity_id=int(entity_id),
                        owner=owner,
                        relation=relation,
                        x=int(x),
                        y=int(y),
                        military=military,
                        attributes=attrs,
                    )
                )
    return sorted(nodes, key=lambda n: n.key)


def _entity_nodes_from_view(
    obs, pview: dict, config: SceneGraphConfig
) -> list[SceneNode]:
    """Fog path: nodes come from the acting client's own player view —
    enemy entities exist in the graph ONLY if the server sent them to this
    connection.  Owner is the numeric player id (names of unseen players are
    not fog-safe)."""
    my_id = None
    me = (obs.players or {}).get(obs.me) or obs.my or {}
    my_id = me.get("player_id")
    nodes = []
    for kind, plural in (("city", "cities"), ("unit", "units")):
        for entity in pview.get(plural) or []:
            entity_id = entity.get("id")
            x, y = entity.get("x"), entity.get("y")
            if entity_id is None or x is None or y is None:
                continue
            mine = entity.get("mine", entity.get("owner") == my_id)
            unit_type = entity.get("type")
            military = kind == "unit" and unit_type not in config.civilian_unit_types
            attrs = {k: v for k, v in entity.items() if k not in {"id", "x", "y"}}
            nodes.append(
                SceneNode(
                    key=f"{kind}:{entity.get('owner')}:{entity_id}",
                    kind=kind,
                    entity_id=int(entity_id),
                    owner=obs.me if mine else f"player_{entity.get('owner')}",
                    relation="self" if mine else "enemy",
                    x=int(x),
                    y=int(y),
                    military=military,
                    attributes=attrs,
                )
            )
    return sorted(nodes, key=lambda n: n.key)


def _nearest(source: SceneNode, choices: Iterable[SceneNode], obs, config):
    ranked = []
    for target in choices:
        distance, _band, _direction = _geometry(source, target, obs, config)
        ranked.append((distance, target.key, target))
    return min(ranked, default=(None, None, None))[2]


def _edge(source, target, relation, obs, config):
    distance, band, direction = _geometry(source, target, obs, config)
    return SceneEdge(source.key, target.key, relation, distance, band, direction)


def build_scene_graph(obs, config: SceneGraphConfig | None = None) -> SceneGraph:
    """Build the four SAGA relation families from the supplied visible facts."""
    config = config or SceneGraphConfig.saga()
    nodes = _entity_nodes(obs, config)
    allied = [n for n in nodes if n.relation == "self"]
    allied_cities = [n for n in allied if n.kind == "city"]
    allied_units = [n for n in allied if n.kind == "unit"]
    enemy_cities = [n for n in nodes if n.relation == "enemy" and n.kind == "city"]
    enemy_units = [n for n in nodes if n.relation == "enemy" and n.kind == "unit"]
    edges: dict[tuple[str, str, str], SceneEdge] = {}

    def add(source, target, relation):
        edge = _edge(source, target, relation, obs, config)
        edges[(edge.source, edge.target, edge.relation)] = edge

    # Permanent own-unit -> nearest own-city anchor.
    for unit in allied_units:
        city = _nearest(unit, allied_cities, obs, config)
        if city:
            add(unit, city, "anchor")

    # SAGA's set expression is directed: every ordered allied pair in radius.
    for source in allied:
        for target in allied:
            if source.key == target.key:
                continue
            distance, _band, _direction = _geometry(source, target, obs, config)
            if distance <= config.proximity_radius:
                add(source, target, "proximity")

    for unit in allied_units:
        if unit.military:
            for city in enemy_cities:
                add(unit, city, "attack")

    # An enemy unit beyond its own side's nearest city influence is advancing.
    # With no visible own-side city, it is conservatively treated as advancing.
    for enemy in enemy_units:
        own_cities = [c for c in enemy_cities if c.owner == enemy.owner]
        own_city = _nearest(enemy, own_cities, obs, config)
        advancing = own_city is None
        if own_city is not None:
            distance, _band, _direction = _geometry(enemy, own_city, obs, config)
            advancing = distance > config.influence_radius
        if advancing:
            target = _nearest(enemy, allied, obs, config)
            if target:
                add(enemy, target, "threat")

    ordered_edges = tuple(
        sorted(edges.values(), key=lambda e: (e.relation, e.source, e.target))
    )
    scope = (
        "player_visible"
        if getattr(obs, "player_view", None) is not None
        else "save_derived_full_state"
    )
    return SceneGraph(tuple(nodes), ordered_edges, config, observation_scope=scope)


@dataclass(frozen=True)
class GoalProgress:
    """Caller-supplied goal state; Harness never creates strategic goals."""

    key: str
    domain: str
    description: str
    current: int | float | None = None
    target: int | float | None = None
    status: str = "active"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class BoundedDigest:
    """Fixed-domain summary of one observation (see build_bounded_digest)."""

    metrics: dict
    goals: tuple[dict, ...]
    expansion: dict
    alerts: tuple[dict, ...]
    omitted: dict
    graph_sha: str
    schema: str = DIGEST_SCHEMA

    def as_dict(self) -> dict:
        payload = _jsonable(dataclasses.asdict(self))
        return {**payload, "sha": _content_sha(payload)}

    @property
    def sha(self) -> str:
        return self.as_dict()["sha"]


def _city_status(city: Mapping) -> dict:
    return {
        k: city.get(k)
        for k in (
            "id",
            "name",
            "size",
            "building",
            "food_stock",
            "shield_stock",
            "shield_surplus",
            "specialists",
        )
    }


def _directional_expansion(graph: SceneGraph) -> dict:
    counts = {d: 0 for d in _DIRECTIONS}
    farthest = {d: 0 for d in _DIRECTIONS}
    for edge in graph.edges:
        if edge.relation != "anchor":
            continue
        # Anchor points unit -> city.  Expansion direction is city -> unit.
        opposite = {
            "N": "S",
            "NE": "SW",
            "E": "W",
            "SE": "NW",
            "S": "N",
            "SW": "NE",
            "W": "E",
            "NW": "SE",
            "HERE": "HERE",
        }[edge.direction]
        counts[opposite] += 1
        farthest[opposite] = max(farthest[opposite], edge.distance)
    return {
        "anchored_units_by_direction": counts,
        "farthest_unit_distance_by_direction": farthest,
    }


def _terrain_expansion(obs, graph: SceneGraph) -> dict:
    """Bound the terrain modality into directional counts around one own city.

    The terrain grid is an omniscient save layer, not explored/visible
    tiles.  Provenance says so explicitly.  Counts describe terrain only; they
    do not invent resource yield or claim that a direction is a good city site.
    """
    terrain = getattr(obs, "terrain", None)
    if not terrain:
        return {"source": "terrain_not_observed", "anchor": None, "directions": {}}
    own = [n for n in graph.nodes if n.relation == "self"]
    own_cities = [n for n in own if n.kind == "city"]
    anchor = min(own_cities or own, key=lambda n: n.key, default=None)
    if anchor is None:
        return {
            "source": "save_full_state_terrain",
            "anchor": None,
            "directions": {},
        }
    xs, ys = terrain.get("xsize"), terrain.get("ysize")
    wrapped = graph.config.distance_mode == "wrapped_manhattan"
    counts = {d: {} for d in _DIRECTIONS}
    max_distance = {d: 0 for d in _DIRECTIONS}
    legend = terrain.get("legend") or {}
    for y, row in enumerate(terrain.get("grid") or []):
        for x, ident in enumerate(row):
            dx = _signed_delta(anchor.x, x, xs, wrapped and graph.config.wrap_x)
            dy = _signed_delta(anchor.y, y, ys, wrapped and graph.config.wrap_y)
            direction = _direction(dx, dy)
            name = legend.get(ident) or str(ident)
            counts[direction][name] = counts[direction].get(name, 0) + 1
            max_distance[direction] = max(max_distance[direction], abs(dx) + abs(dy))
    directions = {}
    for direction in _DIRECTIONS:
        ranked = sorted(counts[direction].items(), key=lambda item: (-item[1], item[0]))
        top = ranked[:3]
        directions[direction] = {
            "tile_count": sum(counts[direction].values()),
            "max_distance": max_distance[direction],
            "top_terrain": [{"terrain": name, "count": count} for name, count in top],
            "other_terrain_tiles": sum(count for _name, count in ranked[3:]),
        }
    return {
        "source": "save_full_state_terrain",
        "anchor": {"key": anchor.key, "x": anchor.x, "y": anchor.y},
        "directions": directions,
    }


def build_bounded_digest(
    obs,
    graph: SceneGraph | None = None,
    goals: Iterable[GoalProgress] | None = None,
) -> BoundedDigest:
    """Return a fixed-domain digest; city and alert lists have explicit caps."""
    graph = graph or build_scene_graph(obs)
    my = obs.my
    cities = sorted(
        my.get("cities") or [],
        key=lambda c: (c.get("id") is None, c.get("id", 0)),
    )
    city_rows = cities[: graph.config.max_city_status]
    metrics = {
        "turn": obs.turn,
        "player": obs.me,
        "score": my.get("score"),
        "gold": my.get("gold"),
        "city_count": my.get("ncities", len(cities)),
        "unit_count": my.get("nunits", len(my.get("units") or [])),
        "tech_count": my.get("techs"),
        "government": my.get("government"),
        "rates": my.get("rates"),
        "researching": my.get("researching"),
        "research_goal": my.get("research_goal"),
        "city_status": [_city_status(c) for c in city_rows],
    }
    goal_rows = tuple(
        g.as_dict() if isinstance(g, GoalProgress) else _jsonable(g)
        for g in (goals or ())
    )
    threat_edges = [e for e in graph.edges if e.relation == "threat"]
    alerts = tuple(
        {
            "source": e.source,
            "target": e.target,
            "distance": e.distance,
            "band": e.band,
            "direction": e.direction,
        }
        for e in threat_edges[: graph.config.max_alerts]
    )
    omitted = {
        "city_status": max(0, len(cities) - len(city_rows)),
        "alerts": max(0, len(threat_edges) - len(alerts)),
    }
    expansion = _directional_expansion(graph)
    expansion["terrain"] = _terrain_expansion(obs, graph)
    return BoundedDigest(
        metrics=metrics,
        goals=goal_rows,
        expansion=expansion,
        alerts=alerts,
        omitted=omitted,
        graph_sha=graph.sha,
    )


class ObservationTools:
    """SAGA's deterministic read-only tools over one immutable observation.

    Savegames do not contain several live city yields/history fields requested
    by SAGA.  Responses name those unavailable fields instead of fabricating
    zeroes.  A caller may supply engine-derived ``available_techs``.
    """

    def __init__(self, obs, *, available_techs: Iterable[str] | None = None):
        self.obs = obs
        self.available_techs = (
            None if available_techs is None else sorted(set(available_techs))
        )

    def city_metrics(self, city_id: int | None = None) -> dict:
        cities = sorted(
            self.obs.my.get("cities") or [],
            key=lambda c: (c.get("id") is None, c.get("id") or 0),
        )
        if city_id is not None:
            cities = [c for c in cities if c.get("id") == city_id]
            if not cities:
                raise KeyError(f"city {city_id}")
        rows = []
        units = self.obs.my.get("units") or []
        for city in cities:
            rows.append(
                {
                    **_city_status(city),
                    "buildings": sorted(city.get("buildings") or []),
                    "unit_burden": sum(
                        1 for u in units if u.get("homecity") == city.get("id")
                    ),
                    "gold_output": None,
                    "science_output": None,
                    "happiness": None,
                    "recent_production": None,
                }
            )
        return {
            "cities": rows,
            "unavailable_from_save": [
                "gold_output",
                "science_output",
                "happiness",
                "recent_production",
            ],
        }

    def army_roster(self) -> dict:
        cities_at = {
            (c.get("x"), c.get("y")): c.get("id")
            for c in self.obs.my.get("cities") or []
        }
        rows = []
        for unit in sorted(
            self.obs.my.get("units") or [],
            key=lambda u: (u.get("id") is None, u.get("id") or 0),
        ):
            activity = unit.get("activity")
            rows.append(
                {
                    "id": unit.get("id"),
                    "type": unit.get("type"),
                    "x": unit.get("x"),
                    "y": unit.get("y"),
                    "hp": unit.get("hp"),
                    "moves": unit.get("moves"),
                    "veteran": unit.get("veteran"),
                    "activity": activity,
                    "fortified": str(activity).lower() in {"fortified", "fortify"},
                    "garrison_city_id": cities_at.get((unit.get("x"), unit.get("y"))),
                }
            )
        return {"units": rows}

    def tech_status(self) -> dict:
        known = sorted(self.obs.my.get("techs_known") or [])
        return {
            "researched_count": self.obs.my.get("techs", len(known)),
            "current_research": self.obs.my.get("researching"),
            "research_goal": self.obs.my.get("research_goal"),
            "next_available": (
                None if self.available_techs is None else self.available_techs[:3]
            ),
            "next_available_source": (
                "not_available_from_save"
                if self.available_techs is None
                else "caller_supplied"
            ),
        }
