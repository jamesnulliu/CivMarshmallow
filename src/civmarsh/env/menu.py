"""Grouped action menu and strict multi-order action bundles for the policy.

Each decision offers stable action keys in six domains (technology,
government, diplomacy, city, civilian, military); the policy answers with one
JSON object selecting a few of them.  Unit candidates are discovered by the
engine through CivHarness and cover all eight neighbouring tiles.  City
production items are offered only when the pinned ruleset proves the city can
build them now; the other order families are wrapped in the same stable-key
contract and tagged with how they were gated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from civharness import (
    BuildImprovement,
    BuildUnit,
    Buy,
    ChangeGovernment,
    DeclareWar,
    SetRates,
    SetResearch,
    SetTechGoal,
    candidate_from_order,
    discover_unit_action_candidates,
)
from civharness.episode import order_as_record

from .snapshot import SpatialSnapshot

# Model-visible schema tag (printed in the ACTION_MENU header line); kept
# byte-identical to the string the policy saw.
MENU_FORMAT = "civm-action-menu-v3"
BUNDLE_SCHEMA = "civm-action-bundle"
RULESET = "civ2civ3"
DOMAINS = ("technology", "government", "diplomacy", "city", "civilian", "military")
CIVILIAN_TYPES = frozenset({"Settlers", "Workers", "Engineers"})

# Production candidate pool: which items are CONSIDERED per city.  Each is then
# checked by the CivHarness city-level buildability evaluator and offered only
# on a positive verdict; a False or unknown (None) verdict is dropped, so the
# pool bounds the menu size and the evaluator supplies legality.
UNIT_ITEMS = ("Warriors", "Workers", "Settlers", "Phalanx", "Archers", "Horsemen")
BUILDING_ITEMS = (
    "Barracks",
    "Granary",
    "Temple",
    "Library",
    "Marketplace",
    "City Walls",
)
RATE_PRESETS = ((60, 0, 40), (40, 0, 60), (30, 30, 40), (50, 20, 30))
GOVERNMENTS = (
    ("Monarchy", "Monarchy"),
    ("Republic", "The Republic"),
    ("Democracy", "Democracy"),
    ("Communism", "Communism"),
)


def ruleset_dir(ruleset: str = RULESET) -> Path:
    """Ruleset directory: ``CIVMARSH_RULESET_DIR`` if set, else the ruleset of
    the pinned freeciv-server installation."""
    configured = os.environ.get("CIVMARSH_RULESET_DIR")
    if configured:
        return Path(configured)
    from civharness.buildability import pinned_ruleset_dir

    return pinned_ruleset_dir(ruleset)


@lru_cache(maxsize=1)
def _ruleset():
    """The parsed ruleset used by the buildability evaluator, loaded once."""
    from civharness.buildability import Ruleset

    return Ruleset.load(ruleset_dir())


@dataclass(frozen=True)
class MenuConfig:
    max_units: int = 3
    max_cities: int = 4
    unit_production_per_city: int = 4
    building_production_per_city: int = 4
    max_research: int = 4
    max_goals: int = 2
    max_rates: int = 4
    max_governments: int = 2
    max_diplomacy: int = 3
    buy_min_gold: int = 120
    max_candidates: int = 96
    max_orders: int = 4
    max_technology_orders: int = 2
    max_government_orders: int = 2
    max_diplomacy_orders: int = 1
    max_city_orders: int = 2
    max_civilian_orders: int = 2
    max_military_orders: int = 2

    def domain_cap(self, domain: str) -> int:
        return getattr(self, f"max_{domain}_orders")


@dataclass(frozen=True)
class MenuCandidate:
    key: str
    domain: str
    entity: str
    label: str
    orders: tuple = field(repr=False, compare=False)
    source: str = "engine_verified"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "domain": self.domain,
            "entity": self.entity,
            "label": self.label,
            "source": self.source,
            "orders": [order_as_record(o) for o in self.orders],
        }


@dataclass(frozen=True)
class ActionMenu:
    candidates: tuple[MenuCandidate, ...]
    skipped: tuple[dict, ...]
    config: MenuConfig
    schema: str = MENU_FORMAT

    @property
    def by_key(self) -> dict[str, MenuCandidate]:
        return {c.key: c for c in self.candidates}

    @property
    def grouped(self) -> dict[str, tuple[MenuCandidate, ...]]:
        return {d: tuple(c for c in self.candidates if c.domain == d) for d in DOMAINS}

    @property
    def sha(self) -> str:
        payload = [c.as_dict() for c in self.candidates]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def render(self) -> str:
        # The per-domain caps, the valid domain names, the verbatim-key rule
        # and the one-action-per-entity rule are stated in the prompt because
        # they are exactly what an unguided policy gets wrong most often.
        # Text only: the candidate sha and bundle semantics do not depend on it.
        lines = [f"ACTION_MENU {self.schema} sha={self.sha}"]
        for domain in DOMAINS:
            lines.append(
                f"[{domain}] (select at most {self.config.domain_cap(domain)})"
            )
            group = self.grouped[domain]
            if not group:
                lines.append("- (none)")
            else:
                lines.extend(f"- {c.key} :: {c.label} [{c.source}]" for c in group)
        lines.append(
            "ACTION_JSON: return one object such as "
            '{"city":["city:10/build-unit:settlers"],"military":[]} '
            f"with at most {self.config.max_orders} total keys. Empty object passes. "
            f"The only valid domains are: {', '.join(DOMAINS)}. "
            "Only select keys copied exactly as listed above; "
            "at most one action per city/unit/entity."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class ActionBundle:
    keys: tuple[str, ...]
    orders: tuple
    candidate_sha: str
    schema: str = BUNDLE_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "keys": list(self.keys),
            "candidate_sha": self.candidate_sha,
            "orders": [order_as_record(o) for o in self.orders],
        }


class BundleValidationError(ValueError):
    pass


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _wrap(order_candidate, domain, entity) -> MenuCandidate:
    return MenuCandidate(
        key=order_candidate.key,
        domain=domain,
        entity=entity,
        label=order_candidate.label,
        orders=(order_candidate.order,),
        source="engine_verified",
    )


def _order_candidate(key, domain, entity, label, order, *, source="faithful_static"):
    # Built through CivHarness so order serialization stays shared; the menu
    # keeps its own domain/entity metadata for the bundle conflict checks.
    wrapped = candidate_from_order(
        key=key,
        domain=domain,
        actor=entity,
        target={},
        action=type(order).__name__,
        label=label,
        order=order,
    )
    return MenuCandidate(
        key=wrapped.key,
        domain=domain,
        entity=entity,
        label=label,
        orders=(wrapped.order,),
        source=source,
    )


@lru_cache(maxsize=1)
def load_tech_requirements() -> dict[str, tuple[str | None, str | None]]:
    """{tech: (req1, req2)} parsed from the ruleset's techs.ruleset; empty when
    no ruleset directory is available."""
    try:
        path = ruleset_dir() / "techs.ruleset"
    except FileNotFoundError:
        return {}
    if not path.exists():
        return {}
    reqs, name, req1, req2 = {}, None, None, None
    for line in path.read_text(errors="replace").splitlines():
        if re.match(r"\[advance_", line):
            if name:
                reqs[name] = (req1, req2)
            name, req1, req2 = None, None, None
            continue
        match = re.match(r"(name|req1|req2)\s*=\s*(.*)", line.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        value = re.sub(r'^_?\(?"?|"?\)?$', "", value).removeprefix("?tech:")
        value = None if value in {"None", ""} else value
        if key == "name":
            name = value
        elif key == "req1":
            req1 = value
        else:
            req2 = value
    if name:
        reqs[name] = (req1, req2)
    return reqs


def reachable_unknown(known, requirements=None) -> list[str]:
    """Unknown techs whose prerequisites are all known, sorted by name."""
    reqs = load_tech_requirements() if requirements is None else requirements
    return sorted(
        t
        for t, (r1, r2) in reqs.items()
        if t not in known
        and (r1 is None or r1 in known)
        and (r2 is None or r2 in known)
    )


def build_menu(
    obs, config: MenuConfig | None = None, *, tech_requirements=None
) -> ActionMenu:
    from civharness.buildability import (
        CityContext,
        can_city_build_improvement_now,
        can_city_build_unit_now,
    )

    config = config or MenuConfig()
    known = obs.my_techs
    items: list[MenuCandidate] = []
    menu_skips: list[dict] = []

    unit_discovery = discover_unit_action_candidates(obs, unit_limit=config.max_units)
    unit_types = {u.get("id"): u.get("type") for u in obs.my_units}
    for candidate in unit_discovery.candidates:
        uid = int(candidate.actor.split(":", 1)[1])
        domain = "civilian" if unit_types.get(uid) in CIVILIAN_TYPES else "military"
        items.append(_wrap(candidate, domain, f"unit:{uid}"))

    # City production: only items the ruleset proves buildable right now.
    ruleset = _ruleset()
    all_own_buildings = [b for c in obs.my_cities for b in (c.get("buildings") or [])]

    cities = sorted(obs.my_cities, key=lambda c: (c.get("id") is None, c.get("id", 0)))
    for city in cities[: config.max_cities]:
        city_id = city.get("id")
        if city_id is None:
            continue
        entity = f"city:{city_id}"
        name = city.get("name") or f"#{city_id}"
        current = city.get("building")
        ctx = CityContext(
            techs_known=set(known),
            city_buildings=list(city.get("buildings") or []),
            city_size=city.get("size") or 1,
            all_own_city_buildings=all_own_buildings,
            # adjacency is not in the bounded observation, so adjacency-
            # dependent verdicts come back None and those items are not offered
            adjacent_terrains=None,
            adjacent_extras=None,
            center_terrain=None,
            center_extras=None,
        )
        offered_u = 0
        for unit in UNIT_ITEMS:
            if offered_u >= config.unit_production_per_city or unit == current:
                continue
            verdict = can_city_build_unit_now(ruleset, unit, ctx)
            if verdict is True:
                items.append(
                    _order_candidate(
                        f"city:{city_id}/build-unit:{_slug(unit)}",
                        "city",
                        entity,
                        f"city {name}: build {unit}",
                        BuildUnit(city_id, unit),
                        source="ruleset_city_buildability_checked",
                    )
                )
                offered_u += 1
            elif verdict is None:
                menu_skips.append(
                    {
                        "entity": entity,
                        "item": unit,
                        "reason": "buildability unknown (missing context) — fail closed",
                    }
                )
        offered_b = 0
        for building in BUILDING_ITEMS:
            if offered_b >= config.building_production_per_city or building == current:
                continue
            verdict = can_city_build_improvement_now(ruleset, building, ctx)
            if verdict is True:
                items.append(
                    _order_candidate(
                        f"city:{city_id}/build-improvement:{_slug(building)}",
                        "city",
                        entity,
                        f"city {name}: build {building}",
                        BuildImprovement(city_id, building),
                        source="ruleset_city_buildability_checked",
                    )
                )
                offered_b += 1
            elif verdict is None:
                menu_skips.append(
                    {
                        "entity": entity,
                        "item": building,
                        "reason": "buildability unknown (missing context) — fail closed",
                    }
                )
        if (obs.my.get("gold") or 0) >= config.buy_min_gold:
            items.append(
                _order_candidate(
                    f"city:{city_id}/buy",
                    "city",
                    entity,
                    f"city {name}: buy current production ({current})",
                    Buy(city_id),
                    source="faithful_packet_gold_threshold_only",
                )
            )

    reachable = reachable_unknown(known, tech_requirements)
    for tech in reachable[: config.max_research]:
        items.append(
            _order_candidate(
                f"technology/research:{_slug(tech)}",
                "technology",
                "player:research",
                f"switch research to {tech}",
                SetResearch(tech),
                source="ruleset_prerequisite_checked",
            )
        )
    for tech in reachable[: config.max_goals]:
        items.append(
            _order_candidate(
                f"technology/goal:{_slug(tech)}",
                "technology",
                "player:tech-goal",
                f"set long-term research goal {tech}",
                SetTechGoal(tech),
                source="ruleset_prerequisite_checked",
            )
        )

    for tax, luxury, science in RATE_PRESETS[: config.max_rates]:
        items.append(
            _order_candidate(
                f"government/rates:{tax}-{luxury}-{science}",
                "government",
                "player:rates",
                f"rates tax={tax} luxury={luxury} science={science}",
                SetRates(tax, luxury, science),
            )
        )
    current_government = obs.my.get("government")
    available_governments = [
        (g, req) for g, req in GOVERNMENTS if req in known and g != current_government
    ]
    for government, _req in available_governments[: config.max_governments]:
        items.append(
            _order_candidate(
                f"government/change:{_slug(government)}",
                "government",
                "player:government",
                f"change government to {government}",
                ChangeGovernment(government),
                source="known-tech-gated",
            )
        )

    players_by_id = {p.get("player_id"): (name, p) for name, p in obs.players.items()}
    diplomacy = obs.diplomacy or obs.my.get("diplomacy") or {}
    dip_items = []
    for target_id, relation in sorted(diplomacy.items(), key=lambda x: int(x[0])):
        target_id = int(target_id)
        target_name, target_player = players_by_id.get(
            target_id, (f"player {target_id}", {})
        )
        if target_name == obs.me or not target_player.get("is_alive", True):
            continue
        if relation.get("state") in {"war", "no_contact", "team"}:
            continue
        dip_items.append(
            _order_candidate(
                f"diplomacy/declare-war:{target_id}",
                "diplomacy",
                f"player:diplomacy:{target_id}",
                f"declare war on {target_name}",
                DeclareWar(target_id),
                source="save-diplomacy-state-gated",
            )
        )
    items.extend(dip_items[: config.max_diplomacy])

    # Stable domain/key ordering; the total cap is part of the environment.
    domain_order = {name: i for i, name in enumerate(DOMAINS)}
    items = sorted(items, key=lambda c: (domain_order[c.domain], c.key))
    omitted = max(0, len(items) - config.max_candidates)
    items = items[: config.max_candidates]
    skipped = tuple(s.as_dict() for s in unit_discovery.skipped) + tuple(menu_skips)
    if omitted:
        skipped += ({"reason": "max_candidates cap", "omitted": omitted},)
    return ActionMenu(tuple(items), skipped, config)


def parse_action_bundle(
    text: str, menu: ActionMenu, config: MenuConfig | None = None
) -> ActionBundle:
    """Strict JSON parser and conflict validator; never repairs model output."""
    config = config or menu.config
    try:
        raw = json.loads((text or "").strip())
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise BundleValidationError("bundle must be a JSON object")
    unknown_domains = set(raw) - set(DOMAINS)
    if unknown_domains:
        raise BundleValidationError(f"unknown domains: {sorted(unknown_domains)}")
    selected = []
    for domain in DOMAINS:
        keys = raw.get(domain, [])
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise BundleValidationError(f"{domain} must be a list of candidate keys")
        if len(keys) > config.domain_cap(domain):
            raise BundleValidationError(
                f"{domain} selects {len(keys)} > cap {config.domain_cap(domain)}"
            )
        selected.extend((domain, key) for key in keys)
    if len(selected) > config.max_orders:
        raise BundleValidationError(
            f"bundle selects {len(selected)} > cap {config.max_orders}"
        )
    keys = [key for _domain, key in selected]
    if len(keys) != len(set(keys)):
        raise BundleValidationError("duplicate candidate key")
    offered = menu.by_key
    candidates = []
    for domain, key in selected:
        candidate = offered.get(key)
        if candidate is None:
            raise BundleValidationError(f"candidate was not offered: {key}")
        if candidate.domain != domain:
            raise BundleValidationError(
                f"candidate {key} belongs to {candidate.domain}, not {domain}"
            )
        candidates.append(candidate)
    entities = [c.entity for c in candidates]
    if len(entities) != len(set(entities)):
        raise BundleValidationError(
            "more than one action selected for the same entity/subdomain"
        )
    orders = tuple(order for candidate in candidates for order in candidate.orders)
    return ActionBundle(tuple(keys), orders, menu.sha)


def obs_block(obs, menu: ActionMenu, *, snapshot=None) -> str:
    """Bounded current-state facts followed by the grouped action menu."""
    if snapshot is None:
        snapshot = SpatialSnapshot.from_observation(
            obs, available_techs=reachable_unknown(obs.my_techs)
        )
    return snapshot.render_actor_global() + "\n" + menu.render()
