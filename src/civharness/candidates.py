"""Serializable, executable action candidates derived from engine legality.

The live Freeciv legality reply is target-specific.  This module enumerates the
actor's current tile and eight adjacent tiles, records probability ranges, and
only constructs orders whose target semantics are unambiguous.  Other legal
verbs are returned as explicit ``skipped`` records; in particular, actions that
need an extra/building ``sub_target`` are never guessed.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field

from civharness.episode import order_as_record
from civharness.policy.client import DoAction

ACTION_CANDIDATE_SCHEMA = "civharness-action-candidate"
_NEIGHBORS = (
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
)
_SAFE_SELF_TARGETS = frozenset(
    {
        "Disband Unit",
        "Fortify",
        "Sentry",
        "Convert Unit",
    }
)
_SAFE_TILE_TARGETS = frozenset({"Found City"})


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class ActionCandidate:
    """One executable order with a stable key, a label and its legality."""

    key: str
    domain: str
    actor: str
    target: dict
    action: str
    label: str
    order: object = field(repr=False, compare=False)
    probability: tuple[int, int] | None = None
    schema: str = ACTION_CANDIDATE_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "key": self.key,
            "domain": self.domain,
            "actor": self.actor,
            "target": self.target,
            "action": self.action,
            "label": self.label,
            "probability": list(self.probability) if self.probability else None,
            "order": order_as_record(self.order),
        }


@dataclass(frozen=True)
class SkippedAction:
    """A legal verb that was not offered, and why."""

    unit_id: int
    action: str
    target: dict
    reason: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CandidateDiscovery:
    """The candidates and skipped verbs of one discovery pass."""

    candidates: tuple[ActionCandidate, ...]
    skipped: tuple[SkippedAction, ...]
    schema: str = ACTION_CANDIDATE_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "candidates": [c.as_dict() for c in self.candidates],
            "skipped": [s.as_dict() for s in self.skipped],
        }


def candidate_from_order(
    *,
    key: str,
    domain: str,
    actor: str,
    target: dict,
    action: str,
    label: str,
    order,
    probability: tuple[int, int] | None = None,
) -> ActionCandidate:
    """Wrap another faithful order family in the same serializable contract."""
    return ActionCandidate(
        key, domain, actor, target, action, label, order, probability
    )


def _map_geometry(obs) -> tuple[int, int]:
    """(xsize, ysize) from the full-state terrain grid or, under fog, from
    the player view's MAP_INFO — geometry is public, contents are not."""
    terrain = getattr(obs, "terrain", None)
    if terrain:
        return terrain["xsize"], terrain["ysize"]
    pview = getattr(obs, "player_view", None)
    if pview and pview.get("map", {}).get("xsize"):
        return pview["map"]["xsize"], pview["map"]["ysize"]
    raise ValueError(
        "candidate discovery requires map geometry (terrain observation or "
        "a player view with MAP_INFO)"
    )


def discover_unit_action_candidates(
    obs,
    *,
    unit_limit: int | None = None,
    wrap_x: bool = False,
    wrap_y: bool = False,
    safe_self_actions=frozenset(_SAFE_SELF_TARGETS),
    safe_tile_actions=frozenset(_SAFE_TILE_TARGETS),
) -> CandidateDiscovery:
    """Discover safe self actions plus eight-way adjacent ``Unit Move``.

    Map geometry and ``obs.legal_actions`` are required because target tile
    IDs and live legality cannot be reconstructed safely.  Deterministic caps
    are applied after sorting by engine unit id, never by save-table order.
    """
    if not callable(getattr(obs, "legal_actions", None)):
        raise ValueError(  # noqa: TRY004 -- a missing observation field
            "unit candidate discovery requires legal_actions observation"
        )
    xs, ys = _map_geometry(obs)
    units = sorted(obs.my_units, key=lambda u: (u.get("id") is None, u.get("id", 0)))
    if unit_limit is not None:
        units = units[:unit_limit]
    candidates, skipped = [], []
    seen_skips = set()

    def skip(unit_id, action, target, reason):
        identity = (unit_id, action, tuple(sorted(target.items())), reason)
        if identity not in seen_skips:
            skipped.append(SkippedAction(unit_id, action, target, reason))
            seen_skips.add(identity)

    for unit in units:
        uid, x, y = unit.get("id"), unit.get("x"), unit.get("y")
        if uid is None or x is None or y is None:
            continue
        actor = f"unit:{uid}"
        tile = y * xs + x
        self_legal = obs.legal_actions(uid, target_tile=tile) or {}
        for action, probability in sorted(self_legal.items()):
            target_id = None
            target = {"kind": "unit", "id": uid}
            domain = "military"
            if action in safe_tile_actions:
                target_id = tile
                target = {"kind": "tile", "id": tile, "x": x, "y": y}
                domain = "expansion" if action == "Found City" else "unit"
            elif action not in safe_self_actions:
                skip(
                    uid,
                    action,
                    {"kind": "tile", "id": tile, "x": x, "y": y},
                    "target or mandatory sub_target is not safely resolvable",
                )
                continue
            candidates.append(
                ActionCandidate(
                    key=(
                        f"unit:{uid}/action:{_slug(action)}@"
                        f"{target['kind']}:{target['id']}"
                    ),
                    domain=domain,
                    actor=actor,
                    target=target,
                    action=action,
                    label=f"unit {uid} ({unit.get('type') or '?'}): {action}",
                    order=DoAction(uid, action, target_id=target_id),
                    probability=tuple(probability),
                )
            )

        for direction, dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if wrap_x:
                nx %= xs
            if wrap_y:
                ny %= ys
            if not (0 <= nx < xs and 0 <= ny < ys):
                continue
            target_tile = ny * xs + nx
            legal = obs.legal_actions(uid, target_tile=target_tile) or {}
            if "Unit Move" not in legal:
                # Record other replies on adjacent targets, but never offer
                # their target semantics merely because they appeared legal.
                for action in sorted(legal):
                    skip(
                        uid,
                        action,
                        {"kind": "tile", "id": target_tile, "x": nx, "y": ny},
                        "non-move adjacent action needs a typed target resolver",
                    )
                continue
            candidates.append(
                ActionCandidate(
                    key=f"unit:{uid}/move:{direction.lower()}@tile:{target_tile}",
                    domain="movement",
                    actor=actor,
                    target={
                        "kind": "tile",
                        "id": target_tile,
                        "x": nx,
                        "y": ny,
                        "direction": direction,
                    },
                    action="Unit Move",
                    label=(
                        f"unit {uid} ({unit.get('type') or '?'}): "
                        f"move {direction} to ({nx},{ny})"
                    ),
                    order=DoAction(uid, "Unit Move", target_id=target_tile),
                    probability=tuple(legal["Unit Move"]),
                )
            )

    return CandidateDiscovery(
        tuple(sorted(candidates, key=lambda c: c.key)),
        tuple(sorted(skipped, key=lambda s: (s.unit_id, s.action, str(s.target)))),
    )
