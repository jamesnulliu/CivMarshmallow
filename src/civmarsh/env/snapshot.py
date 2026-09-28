"""State snapshot shared by the policy prompt and the CivTelescope value input.

``SpatialSnapshot`` builds one factual object from a CivHarness observation and
exposes bounded projections of it: the policy's global state block, per-entity
scene views, and the value input that CivTelescope scores.  The projections
differ in detail but share the exact digest and scene hashes, which makes fact
parity between what the policy sees and what is scored testable.

This is an information-compression layer only: it does not create goals, run a
planning loop, schedule specialists or keep memory across games.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from civharness import (
    Observation,
    ObservationTools,
    SceneGraphConfig,
    build_bounded_digest,
    build_scene_graph,
)

# Model-visible format tags (they appear in the policy's STATE line and in the
# value-input header); kept byte-identical to the strings the models saw.
STATE_FORMAT = "civm-saga-spatial-v1-playervisible"
FULLSTATE_FORMAT = "civm-saga-spatial-v1-fullstate"


def _format_for(obs) -> str:
    """The format tag records the visibility mode: an observation built from a
    player view (fog of war) and an omniscient one are different formats."""
    if getattr(obs, "player_view", None) is not None:
        return STATE_FORMAT
    return FULLSTATE_FORMAT


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProjectionConfig:
    """Caps on the value input, so its size is stable under late-game growth."""

    max_value_cities: int = 12
    max_value_units: int = 24
    max_value_graph_edges: int = 24


@dataclass(frozen=True)
class SpatialSnapshot:
    observation: Observation
    graph: object
    digest: object
    city_metrics: dict
    army_roster: dict
    tech_status: dict
    projection: ProjectionConfig = ProjectionConfig()
    format_name: str = FULLSTATE_FORMAT

    @classmethod
    def from_observation(
        cls,
        obs: Observation,
        *,
        graph_config: SceneGraphConfig | None = None,
        available_techs: Iterable[str] | None = None,
        goals=None,
        projection: ProjectionConfig | None = None,
    ) -> SpatialSnapshot:
        graph = build_scene_graph(obs, graph_config)
        digest = build_bounded_digest(obs, graph, goals)
        tools = ObservationTools(obs, available_techs=available_techs)
        return cls(
            observation=obs,
            graph=graph,
            digest=digest,
            city_metrics=tools.city_metrics(),
            army_roster=tools.army_roster(),
            tech_status=tools.tech_status(),
            projection=projection or ProjectionConfig(),
            format_name=_format_for(obs),
        )

    @property
    def shared_facts(self) -> dict:
        return {
            "format": self.format_name,
            "observation_scope": self.graph.observation_scope,
            "graph_sha": self.graph.sha,
            "graph_config_sha": self.graph.config.sha,
            "digest_sha": self.digest.sha,
            "digest": self.digest.as_dict(),
        }

    @property
    def fact_sha(self) -> str:
        return _sha(self.shared_facts)

    def actor_global_view(self) -> dict:
        threats = [e.as_dict() for e in self.graph.edges if e.relation == "threat"]
        return {
            **self.shared_facts,
            "fact_sha": self.fact_sha,
            "threat_relations": threats[: self.graph.config.max_alerts],
            "tools_available": ["city_metrics", "army_roster", "tech_status"],
        }

    def entity_view(self, kind: str, entity_id: int) -> dict:
        node = next(
            (
                n
                for n in self.graph.nodes
                if n.relation == "self" and n.kind == kind and n.entity_id == entity_id
            ),
            None,
        )
        if node is None:
            raise KeyError(f"{kind}:{entity_id}")
        return {
            "format": self.format_name,
            "fact_sha": self.fact_sha,
            "global_digest_sha": self.digest.sha,
            "scene": self.graph.entity_view(node.key),
        }

    def value_input_view(self) -> dict:
        # The value input carries fixed materializations of the same tools the
        # policy can name; the caps keep its size stable as armies grow.
        roster = list(self.army_roster["units"])
        cities = list(self.city_metrics["cities"])
        graph_edges = [e.as_dict() for e in self.graph.edges]
        cap = self.projection
        return {
            **self.shared_facts,
            "fact_sha": self.fact_sha,
            "city_metrics": {
                "cities": cities[: cap.max_value_cities],
                "omitted": max(0, len(cities) - cap.max_value_cities),
                "unavailable_from_save": self.city_metrics["unavailable_from_save"],
            },
            "army_roster": {
                "units": roster[: cap.max_value_units],
                "omitted": max(0, len(roster) - cap.max_value_units),
            },
            "tech_status": self.tech_status,
            "graph_summary": {
                "node_counts": _node_counts(self.graph.nodes),
                "edge_counts": _edge_counts(self.graph.edges),
                "sample": graph_edges[: cap.max_value_graph_edges],
                "omitted": max(0, len(graph_edges) - cap.max_value_graph_edges),
            },
        }

    def render_actor_global(self) -> str:
        """The policy's state block (first half of each decision prompt)."""
        view = self.actor_global_view()
        digest = view["digest"]
        return "\n".join(
            [
                f"STATE {self.format_name} fact_sha={self.fact_sha}",
                "METRICS " + _canonical(digest["metrics"]),
                "GOALS " + _canonical(digest["goals"]),
                "EXPANSION " + _canonical(digest["expansion"]),
                "ALERTS " + _canonical(digest["alerts"]),
                "THREAT_RELATIONS " + _canonical(view["threat_relations"]),
            ]
        )

    def render_value_input(self) -> str:
        """The rendering CivTelescope compares (one side of a pairwise prompt)."""
        view = self.value_input_view()
        return "\n".join(
            [
                f"Freeciv state value input ({self.format_name}) fact_sha={self.fact_sha}",
                "DIGEST " + _canonical(view["digest"]),
                "CITY_METRICS " + _canonical(view["city_metrics"]),
                "ARMY_ROSTER " + _canonical(view["army_roster"]),
                "TECH_STATUS " + _canonical(view["tech_status"]),
                "GRAPH_SUMMARY " + _canonical(view["graph_summary"]),
            ]
        )


def _node_counts(nodes) -> dict:
    counts = {}
    for node in nodes:
        key = f"{node.relation}_{node.kind}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _edge_counts(edges) -> dict:
    counts = {}
    for edge in edges:
        counts[edge.relation] = counts.get(edge.relation, 0) + 1
    return dict(sorted(counts.items()))


def render_sha(text: str) -> str:
    """Short content hash of a rendering, for provenance fields."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
