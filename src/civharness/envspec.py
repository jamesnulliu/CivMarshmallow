"""EnvSpec — a declarative observation space + action space for an experiment.

A downstream researcher rarely wants "the whole game". They want a *slice*: an
economy-only agent that may set production and research but not move armies; an
observation that hides the map so a myopia study is clean; a military agent with
the full action set. EnvSpec is that slice, written once as a YAML file and
handed to the drivers:

    env = EnvSpec.preset("economy")  # or EnvSpec.from_yaml(path)
    run_agents(save, wd, agents={p: agent}, until=..., env=env)

The bundled presets live in civharness/configs/*.yaml (see `preset_names()`).

Two axes:
- **actions**: the set of allowed action keys (each order type declares its
  `action_key`). An order whose key is not allowed is rejected — loudly by
  default (`strict`), because an agent emitting a disallowed order is an
  experiment-configuration bug, not something to swallow silently.
- **observations**: which fields the observation carries (consumed by
  `civharness.observe`). Restricting these actually removes information from
  what the agent sees, so the environment is genuinely smaller.

The default, `EnvSpec.full()`, imposes no action restriction and a rich
observation; the terrain grid and the live legal-action mask stay opt-in because
they cost real work. Core stays dependency-free: YAML (including the presets)
loads through a guarded import, and a spec can always be built with
`from_dict` without PyYAML.

**Scope of the guarantee.** EnvSpec is a *correctness scaffold for cooperating
agents*, not a security sandbox. The driver enforces it at the seams it controls
— it gates every returned order by canonical order-class, restricts what the
observation carries, and withholds the raw save under a restricted view — so an
honest agent literally cannot see or do what the spec forbids by mistake. It
does **not** contain adversarial in-process code: a Python agent that *wants* to
escape can reach the engine through language internals (a captured client via a
function's `__closure__`, module globals, `gc`, etc.). Enforcing an untrusted
policy requires running it out-of-process behind a message interface — out of
scope here. EnvSpec closes the easy/accidental holes only.
"""

import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from civharness.policy import client as _order_mod


class ForbiddenAction(RuntimeError):
    """An agent issued an order whose action_key the EnvSpec does not allow.
    Raised in strict mode (the default) so an experiment misconfiguration fails
    loudly instead of the order being silently dropped."""


# The canonical {order_class: action_key} map — the harness's OWN order types,
# discovered from policy.client at import (so a type added there registers
# automatically). check_order gates on `type(order)` against this map, NOT on the
# order's self-declared `action_key` string: a custom class cannot claim an
# allowed key just by labelling itself. Class identity (not
# isinstance) is deliberate — a malicious *subclass* of an allowed order would
# otherwise inherit its key while overriding apply().
_CANONICAL_ACTIONS: dict[type, str] = {
    obj: obj.action_key
    for obj in vars(_order_mod).values()
    if isinstance(obj, type)
    and dataclasses.is_dataclass(obj)
    and hasattr(obj, "action_key")
}


def all_action_keys() -> set[str]:
    """Every action key the harness implements — the values of the canonical
    order-class map. This is *all actions CivHarness ships*, a deliberate core
    subset of Freeciv (treaty negotiation, spaceship, rally points are
    non-goals) — not every packet the engine accepts."""
    return set(_CANONICAL_ACTIONS.values())


def preset_names() -> list[str]:
    """Names accepted by `EnvSpec.preset` (the bundled configs/*.yaml)."""
    configs = resources.files("civharness").joinpath("configs")
    return sorted(
        f.name[: -len(".yaml")] for f in configs.iterdir() if f.name.endswith(".yaml")
    )


# Observation field schema + its generous default (what full() grants). Values:
#   techs:         "count" | "known_set" | False
#   cities/units:  True (all fields) | False | [field names]
#   government:    bool
#   rates:         bool
#   diplomacy:     bool
#   players:       "self" | "all"
#   terrain:       bool   (parsing the whole grid — opt-in)
#   legal_actions: bool   (a live client round-trip per unit — opt-in)
_OBS_DEFAULTS = {
    "techs": "known_set",
    "cities": True,
    "units": True,
    "government": True,
    "rates": True,
    "diplomacy": True,
    "players": "all",
    "terrain": False,
    "legal_actions": False,
    # Which world the agent sees. "full_state" = the save-derived
    # omniscient view (the default). "player_visible" = genuine fog:
    # map/cities/units come from the client's own PlayerViewCache (what the
    # server actually sent this connection) and the raw save is withheld.
    "visibility": "full_state",
}


def _is_field_list(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)


# Allowed VALUES per observation selector, checked in from_dict so a malformed
# spec is rejected rather than silently *widening* what the agent sees.
# Without this, `government: "false"` (a string) reads as truthy and
# exposes the raw save, and `players: "slef"` (a typo) falls through to "all".
# `is True/False` (not truthiness) is deliberate, so 1/0/"false" don't sneak in.
_OBS_VALID = {
    "techs": (
        lambda v: v is False or v in ("count", "known_set"),
        "'count', 'known_set', or false",
    ),
    "cities": (
        lambda v: v is True or v is False or _is_field_list(v),
        "true, false, or a list of field-name strings",
    ),
    "units": (
        lambda v: v is True or v is False or _is_field_list(v),
        "true, false, or a list of field-name strings",
    ),
    "government": (lambda v: v is True or v is False, "true or false"),
    "rates": (lambda v: v is True or v is False, "true or false"),
    "diplomacy": (lambda v: v is True or v is False, "true or false"),
    "players": (lambda v: v in ("self", "all"), "'self' or 'all'"),
    "terrain": (lambda v: v is True or v is False, "true or false"),
    "legal_actions": (lambda v: v is True or v is False, "true or false"),
    "visibility": (
        lambda v: v in ("full_state", "player_visible"),
        "'full_state' or 'player_visible'",
    ),
}


@dataclass
class EnvSpec:
    """An action space (allowed action keys) plus an observation selector."""

    name: str = "full"
    actions: frozenset = field(default_factory=lambda: frozenset(all_action_keys()))
    observations: dict = field(default_factory=lambda: dict(_OBS_DEFAULTS))
    strict: bool = True  # forbidden order raises (True) vs. is dropped (False)

    # -- constructors -------------------------------------------------------
    @classmethod
    def full(cls) -> "EnvSpec":
        """No restriction *within the harness's action set* — every action key
        CivHarness implements is allowed — plus a rich observation (terrain grid
        and live legal-action mask remain opt-in). This is the default for every
        driver. Note `full()` means "all harness-supported actions" (a core
        subset of Freeciv), not literally every engine action."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "EnvSpec":
        known = all_action_keys()
        acts = d.get("actions")
        if acts is None:
            actions = frozenset(known)
        else:
            actions = frozenset(acts)
            unknown = actions - known
            if unknown:
                raise ValueError(
                    f"unknown action keys {sorted(unknown)}; known: {sorted(known)}"
                )
        obs = dict(_OBS_DEFAULTS)
        for k, v in (d.get("observations") or {}).items():
            if k not in _OBS_DEFAULTS:
                raise ValueError(
                    f"unknown observation field {k!r}; known: {sorted(_OBS_DEFAULTS)}"
                )
            ok, allowed = _OBS_VALID[k]
            if not ok(v):
                raise ValueError(
                    f"invalid value {v!r} for observation field {k!r}; "
                    f"allowed: {allowed}"
                )
            obs[k] = v
        return cls(
            name=d.get("name", "env"),
            actions=actions,
            observations=obs,
            strict=d.get("strict", True),
        )

    @classmethod
    def from_yaml(cls, path) -> "EnvSpec":
        """Load a spec from a YAML file (needs PyYAML)."""
        return cls._from_yaml_text(Path(path).read_text())

    @classmethod
    def preset(cls, name: str) -> "EnvSpec":
        """Load a bundled preset by name — one of `preset_names()`: "full",
        "economy", "research_only", "military", "saga_visible",
        "saga_fullinfo" (needs PyYAML)."""
        names = preset_names()
        if name not in names:
            raise ValueError(f"unknown EnvSpec preset {name!r}; known: {names}")
        configs = resources.files("civharness").joinpath("configs")
        return cls._from_yaml_text(configs.joinpath(f"{name}.yaml").read_text())

    @classmethod
    def _from_yaml_text(cls, text: str) -> "EnvSpec":
        try:
            import yaml
        except ImportError as e:  # keep core dependency-free
            raise ImportError(
                "EnvSpec YAML loading needs PyYAML (`pip install pyyaml`); "
                "or build the spec in Python with EnvSpec.from_dict(...)."
            ) from e
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    # -- action gating ------------------------------------------------------
    def allows(self, action_key: str) -> bool:
        return action_key in self.actions

    def restricts_actions(self) -> bool:
        """True when the action space is a strict subset of everything the
        harness can do — i.e. some known action key is disallowed. Used to fail
        *closed*: under a restricted space an unclassifiable order (no
        `action_key`) is a bypass risk, so it is denied rather than waved
        through."""
        return not (all_action_keys() <= self.actions)

    def check_order(self, order) -> bool:
        """True if the order may be applied. A forbidden order raises
        ForbiddenAction in strict mode, else returns False (caller drops it).

        The action key is looked up by the order's **class** in the canonical
        registry, not read off the object — a custom class cannot pass a
        restricted spec by self-declaring an allowed `action_key` and then doing
        something else in apply(). A *recognised* order is allowed iff its
        canonical key is in the action set; an *unrecognised* one (any class
        the harness didn't ship) is allowed only when the env imposes no action
        restriction at all, and denied fail-closed otherwise. This gates
        cooperating agents against misconfiguration; it is not a sandbox
        against adversarial in-process code (see the module note)."""
        key = _CANONICAL_ACTIONS.get(type(order))
        if key is None:
            allowed = not self.restricts_actions()  # unrecognised order: fail closed
        else:
            allowed = key in self.actions
        if allowed:
            return True
        if self.strict:
            what = (
                f"order {type(order).__name__} (action {key!r})"
                if key is not None
                else f"unrecognised order type {type(order).__name__!r} "
                "(not a registered CivHarness order)"
            )
            raise ForbiddenAction(
                f"{what} is not permitted by EnvSpec {self.name!r}; "
                f"allowed: {sorted(self.actions)}"
            )
        return False

    # -- observation gating (consumed by civharness.observe) ----------------
    def full_information(self) -> bool:
        """True when the agent is entitled to the whole world: every player and
        the full tech/city/unit/government detail. Only then does the driver
        hand the agent the raw save path — a restricted env withholds it, so an
        observation restriction cannot be bypassed by re-parsing the god-view
        save. `terrain` and `legal_actions` are *additive*
        conveniences (off in `full()` by default) rather than world-hiding
        levers, so they do not gate this; dialing down players/techs/cities/
        units/government/rates/diplomacy is what marks the view as restricted.
        ``player_visible`` visibility is by definition NOT full information."""
        o = self.observations
        return (
            o.get("visibility", "full_state") == "full_state"
            and o.get("players", "all") == "all"
            and o.get("techs") == "known_set"
            and o.get("cities") is True
            and o.get("units") is True
            and bool(o.get("government"))
            and bool(o.get("rates"))
            and bool(o.get("diplomacy"))
        )

    def wants(self, field_name: str) -> bool:
        """Whether an observation field is enabled at all (truthy and not the
        string 'false')."""
        v = self.observations.get(field_name, False)
        return bool(v) and v != "false"

    def obs(self, field_name: str, default=None):
        """Raw selector for a field (e.g. 'known_set', 'self', a list of
        column names)."""
        return self.observations.get(field_name, default)
