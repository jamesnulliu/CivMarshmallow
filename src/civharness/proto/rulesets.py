"""Ruleset packet decoders — PACKET_RULESET_EXTRA (232) and
PACKET_RULESET_ACTION (246), verified against freeciv-3.2.5 packets.def and
the generated packets_gen.c receive code.

Why the wire and not Lua: 3.2.5's server Lua `find` module has no extra
lookup (api_game_find.h), so extra ids/rule names are ONLY available from the
ruleset packets the server sends every client at login.  These two packets
carry exactly what typed action sub-target (e.g. worker extra) resolution
needs:

  * extras: id, rule_name, causes/rmcauses bitvectors (EC_*/ERM_* in
    fc_types.h), buildable flag, reqs;
  * actions: id, target kind and SUB-TARGET kind (ASTK_* in actres.h) —
    which actions require an extra sub-target at all.

Delta note: neither packet declares a key field, so each deltas against the
previous packet OF THE SAME TYPE on the connection (single rolling baseline,
bit 0 = id).  BOOLs fold into the bitvector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from civharness.proto.dataio import Reader

RULESET_EXTRA = 232
RULESET_ACTION = 246

# bitvector sizes (common/fc_types.h, extras.h, unittype.h, actions.h)
EC_COUNT = 9  # extra_cause: Irrigation Mine Road Base Pollution
#              Fallout Hut Appear Resource
ERM_COUNT = 4  # extra_rmcause: Pillage Clean Disappear Enter
UCL_LAST = 32  # bv_unit_classes
EF_COUNT = 22  # bv_extra_flags
MAX_EXTRA_TYPES = 250  # bv_extras
ACTION_COUNT = 125  # bv_actions (SPECENUM_VALUE124 is the last)
ACT_SUB_RES_COUNT = 4

EXTRA_CAUSE_NAMES = (
    "Irrigation",
    "Mine",
    "Road",
    "Base",
    "Pollution",
    "Fallout",
    "Hut",
    "Appear",
    "Resource",
)
EXTRA_RMCAUSE_NAMES = ("Pillage", "Clean", "Disappear", "Enter")

# enum action_sub_target_kind (common/actres.h)
ASTK_NONE = 0
ASTK_BUILDING = 1
ASTK_TECH = 2
ASTK_EXTRA = 3
ASTK_EXTRA_NOT_THERE = 4

# enum action_target_kind (common/actres.h)
ATK_CITY, ATK_UNIT, ATK_UNITS, ATK_TILE, ATK_EXTRAS, ATK_SELF = range(6)


def _req(r: Reader) -> dict:
    """One REQUIREMENT: dio_get_requirement_raw field order."""
    return {
        "type": r.get_uint8(),
        "value": r.get_sint32(),
        "range": r.get_uint8(),
        "survives": r.get_bool8(),
        "present": r.get_bool8(),
        "quiet": r.get_bool8(),
    }


def _reqs(r: Reader, count: int) -> tuple:
    return tuple(_req(r) for _ in range(count))


def _bv_indices(r: Reader, nbits: int) -> tuple:
    return tuple(i for i, b in enumerate(r.get_bitvector(nbits)) if b)


# (name, reader, kind) in delta-bit order — extracted from the generated
# receive_packet_ruleset_extra_100.  kind "" plain / "bool" bitfold /
# "reqs:<countfield>" requirement array sized by an earlier count field.
_EXTRA_FIELDS = (
    ("id", "get_uint8", ""),
    ("name", "get_string", ""),
    ("rule_name", "get_string", ""),
    ("category", "get_uint8", ""),
    ("causes", lambda r: _bv_indices(r, EC_COUNT), ""),
    ("rmcauses", lambda r: _bv_indices(r, ERM_COUNT), ""),
    ("activity_gfx", "get_string", ""),
    ("act_gfx_alt", "get_string", ""),
    ("act_gfx_alt2", "get_string", ""),
    ("rmact_gfx", "get_string", ""),
    ("rmact_gfx_alt", "get_string", ""),
    ("rmact_gfx_alt2", "get_string", ""),
    ("graphic_str", "get_string", ""),
    ("graphic_alt", "get_string", ""),
    ("reqs_count", "get_uint8", ""),
    ("reqs", None, "reqs:reqs_count"),
    ("rmreqs_count", "get_uint8", ""),
    ("rmreqs", None, "reqs:rmreqs_count"),
    ("appearance_chance", "get_uint16", ""),
    ("appearance_reqs_count", "get_uint8", ""),
    ("appearance_reqs", None, "reqs:appearance_reqs_count"),
    ("disappearance_chance", "get_uint16", ""),
    ("disappearance_reqs_count", "get_uint8", ""),
    ("disappearance_reqs", None, "reqs:disappearance_reqs_count"),
    ("visibility_req", "get_uint16", ""),
    ("buildable", None, "bool"),
    ("generated", None, "bool"),
    ("build_time", "get_uint8", ""),
    ("build_time_factor", "get_uint8", ""),
    ("removal_time", "get_uint8", ""),
    ("removal_time_factor", "get_uint8", ""),
    ("infracost", "get_uint16", ""),
    ("defense_bonus", "get_uint8", ""),
    ("eus", "get_uint8", ""),
    ("native_to", lambda r: _bv_indices(r, UCL_LAST), ""),
    ("flags", lambda r: _bv_indices(r, EF_COUNT), ""),
    ("hidden_by", lambda r: _bv_indices(r, MAX_EXTRA_TYPES), ""),
    ("bridged_over", lambda r: _bv_indices(r, MAX_EXTRA_TYPES), ""),
    ("conflicts", lambda r: _bv_indices(r, MAX_EXTRA_TYPES), ""),
    ("no_aggr_near_city", "get_sint8", ""),
    ("helptext", "get_string", ""),
)

_ACTION_FIELDS = (
    ("id", "get_uint8", ""),
    ("ui_name", "get_string", ""),
    ("quiet", None, "bool"),
    ("result", "get_uint8", ""),
    ("sub_results", lambda r: _bv_indices(r, ACT_SUB_RES_COUNT), ""),
    ("actor_consuming_always", None, "bool"),
    ("act_kind", "get_uint8", ""),
    ("tgt_kind", "get_uint8", ""),
    ("sub_tgt_kind", "get_uint8", ""),
    ("min_distance", "get_sint32", ""),
    ("max_distance", "get_sint32", ""),
    ("blocked_by", lambda r: _bv_indices(r, ACTION_COUNT), ""),
)


def _zero(fields) -> dict:
    out = {}
    for name, getter, kind in fields:
        if kind == "bool":
            out[name] = False
        elif kind.startswith("reqs:") or callable(getter):
            out[name] = ()
        elif getter in ("get_string", "get_estring"):
            out[name] = ""
        else:
            out[name] = 0
    return out


def _decode(body: bytes, fields, old: dict | None) -> dict:
    r = Reader(body)
    bits = r.get_bitvector(len(fields))
    out = dict(old) if old is not None else _zero(fields)
    for i, (name, getter, kind) in enumerate(fields):
        if kind == "bool":
            out[name] = bits[i]
            continue
        if not bits[i]:
            continue
        if kind.startswith("reqs:"):
            out[name] = _reqs(r, out[kind.split(":", 1)[1]])
        elif callable(getter):
            out[name] = getter(r)
        else:
            out[name] = getattr(r, getter)()
    return out


@dataclass
class RulesetCatalog:
    """Extras and actions as this connection received them at login.
    Rolling delta baseline per packet type (these packets have no key)."""

    extras: dict[int, dict] = field(default_factory=dict)  # id -> record
    actions: dict[int, dict] = field(default_factory=dict)  # id -> record
    _extra_old: dict | None = None
    _action_old: dict | None = None

    def handles(self, ptype: int) -> bool:
        return ptype in (RULESET_EXTRA, RULESET_ACTION)

    def feed(self, ptype: int, body: bytes) -> None:
        if ptype == RULESET_EXTRA:
            rec = _decode(body, _EXTRA_FIELDS, self._extra_old)
            self._extra_old = rec
            self.extras[rec["id"]] = rec
        elif ptype == RULESET_ACTION:
            rec = _decode(body, _ACTION_FIELDS, self._action_old)
            self._action_old = rec
            self.actions[rec["id"]] = rec

    def reset(self) -> None:
        self.extras.clear()
        self.actions.clear()
        self._extra_old = None
        self._action_old = None

    # -- typed queries -----------------------------------------------------
    def extras_by_cause(self, cause_name: str) -> list[dict]:
        """Buildable extras whose causes include `cause_name` (EC_* label),
        ordered by id (deterministic)."""
        try:
            cause = EXTRA_CAUSE_NAMES.index(cause_name)
        except ValueError:
            raise ValueError(
                f"unknown extra cause {cause_name!r}; known: {EXTRA_CAUSE_NAMES}"
            ) from None
        return [
            rec
            for _i, rec in sorted(self.extras.items())
            if cause in rec["causes"] and rec["buildable"]
        ]

    def worker_target_extras(self) -> list[dict]:
        """Extras creatable by worker-type actions: causes Irrigation, Mine,
        Road or Base (is_extra_caused_by_worker_action, common/extras.c),
        buildable only."""
        seen, out = set(), []
        for cause in ("Irrigation", "Mine", "Road", "Base"):
            for rec in self.extras_by_cause(cause):
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    out.append(rec)
        return sorted(out, key=lambda r: r["id"])

    def actions_needing_extra(self) -> dict[int, dict]:
        """Action id -> record for actions whose sub-target is an extra
        (ASTK_EXTRA / ASTK_EXTRA_NOT_THERE): the ones that must carry a
        typed sub_tgt_id in UNIT_DO_ACTION."""
        return {
            aid: rec
            for aid, rec in self.actions.items()
            if rec["sub_tgt_kind"] in (ASTK_EXTRA, ASTK_EXTRA_NOT_THERE)
        }

    def extra_rule_names(self) -> dict[str, int]:
        """rule_name -> id (feeds RuleIds.extras, which the 3.2.5 Lua dump
        cannot populate — find.extra does not exist in this version)."""
        return {
            rec["rule_name"]: i for i, rec in self.extras.items() if rec["rule_name"]
        }
