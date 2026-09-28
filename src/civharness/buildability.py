"""City-level buildability — a faithful, fail-closed port of the
freeciv-3.2.5 ``can_city_build_*_now`` checks for the pinned ruleset.

Why local logic: 3.2.5 has NO non-mutating network query for "what can this
city build" (common/city.c can_city_build_* are internal; the only wire path
is sending a mutating CITY_CHANGE and watching it stick, which candidate
discovery must never do).  Server Lua exposes only the player-level
``can_build_direct`` coarse prefilter.  So this module decodes the
requirements and ports the city-level logic — from the exact ruleset files the
pinned server itself loads (same installation directory), which keeps the
facts engine-sourced and auditable.

Faithfulness contract (mirrors common/city.c + common/improvement.c):

  improvement:  can_player_build_improvement_direct (player-range reqs)
              + city_has_building check
              + city/local-range reqs (are_reqs_active)
              + improvement_obsolete (obsolete_by reqs)
  unit:         can_player_build_unit_direct (tech reqs)
              + unit build_reqs at city level
              + native-class-near-tile (naval units inland)
              + obsoleted_by chain

FAIL-CLOSED: any requirement type/range this port does not implement, or any
fact the caller's context cannot supply (e.g. adjacent-tile extras without
tile knowledge), makes the verdict ``None`` (unknown) — the caller must NOT
offer such an item, and should record it as skipped.  ``True`` is only
returned when every requirement was positively evaluated.

Server fixtures (tests/test_buildability.py) validate this evaluator against
real engine accept/reject behavior on the pinned binary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# --- ruleset file parsing ---------------------------------------------------


def _split_cells(line: str) -> list[str]:
    return [c.strip().strip('"') for c in re.findall(r'"[^"]*"|[^,\s][^,]*', line)]


def _parse_req_table(lines: list[str]) -> list[dict]:
    """A `reqs = { "type","name","range"[,"present"] ... }` table."""
    if not lines:
        return []
    header = [h.lower() for h in _split_cells(lines[0].lstrip("{ "))]
    out = []
    for row in lines[1:]:
        cells = _split_cells(row.lstrip("{ "))
        if len(cells) < len(header):
            continue
        req = dict(zip(header, cells))
        req["present"] = req.get("present", "TRUE").upper() != "FALSE"
        out.append(req)
    return out


def _iter_sections(path: Path):
    """Yield (section_name, {key: scalar or table_lines}) from a ruleset file.
    Handles `key = { ... }` tables and comma-terminated continuation lines
    (`flags = "A", "B",` continued on the next line)."""
    section, data = None, {}
    key, table = None, None
    cont_key = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].rstrip() if not raw.lstrip().startswith(";") else ""
        if not line.strip():
            continue
        if table is not None:
            if line.strip() == "}":
                data[key] = table
                key, table = None, None
            else:
                table.append(line.strip())
            continue
        if cont_key is not None:
            data[cont_key] += " " + line.strip()
            if not line.rstrip().endswith(","):
                cont_key = None
            continue
        m = re.match(r"\[(\S+)\]", line.strip())
        if m:
            if section:
                yield section, data
            section, data = m.group(1), {}
            continue
        m = re.match(r"(\w+)\s*=\s*(.*)", line.strip())
        if m:
            k, v = m.group(1), m.group(2).strip()
            if v == "" or v == "{":
                key, table = k, []
            else:
                data[k] = v
                if v.endswith(","):
                    cont_key = k
    if table is not None and key is not None:
        data[key] = table
    if section:
        yield section, data


def _name_of(v: str) -> str:
    v = v.strip()
    v = re.sub(r'^_\(\s*"', "", v)
    v = re.sub(r'"\s*\)$', "", v)
    v = v.strip('"')
    return v.split(":", 1)[1] if v.startswith("?") and ":" in v else v


def pinned_ruleset_dir(
    ruleset: str = "civ2civ3", *, binary: Path | None = None
) -> Path:
    """The ruleset directory of the PINNED server installation (the same
    files the binary itself loads): <prefix>/share/freeciv/<ruleset>, where
    <prefix> is two levels above `binary` (default: CIVHARNESS_SERVER) —
    never a random system copy."""
    from civharness.config import DEFAULT_BINARY

    binary = Path(DEFAULT_BINARY if binary is None else binary)
    prefix = binary.resolve().parent.parent
    d = prefix / "share" / "freeciv" / ruleset
    if not d.is_dir():
        raise FileNotFoundError(f"pinned ruleset dir not found: {d} (from {binary})")
    return d


@dataclass
class Ruleset:
    """Typed slices of the pinned ruleset needed for buildability."""

    buildings: dict[str, dict] = field(default_factory=dict)
    units: dict[str, dict] = field(default_factory=dict)
    oceanic_terrains: set[str] = field(default_factory=set)
    unit_classes: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, ruleset_dir: Path) -> Ruleset:
        rs = cls()

        def rule_name(data: dict) -> str:
            # engine identifiers are rule_name; name is only the display
            # label and MAY differ (e.g. name "Amphitheater", rule_name
            # "Colosseum" in civ2civ3) — RuleIds/BuildImprovement use
            # rule names, so key everything by them
            rn = data.get("rule_name", "").strip('"')
            return rn if rn else _name_of(data.get("name", ""))

        for sec, data in _iter_sections(ruleset_dir / "buildings.ruleset"):
            if not sec.startswith("building_"):
                continue
            rs.buildings[rule_name(data)] = {
                "genus": data.get("genus", "").strip('"'),
                "reqs": _parse_req_table(data.get("reqs", [])),
                "obsolete_by": _parse_req_table(data.get("obsolete_by", [])),
            }
        for sec, data in _iter_sections(ruleset_dir / "units.ruleset"):
            if sec.startswith("unitclass_"):
                rs.unit_classes[_name_of(data.get("name", ""))] = {
                    "flags": [
                        f.strip().strip('"')
                        for f in data.get("flags", "").split(",")
                        if f.strip()
                    ],
                }
            if not sec.startswith("unit_"):
                continue
            flags = (
                ",".join(data.get("flags", ""))
                if isinstance(data.get("flags"), list)
                else data.get("flags", "")
            )
            rs.units[rule_name(data)] = {
                "class": _name_of(data.get("class", "")),
                "reqs": _parse_req_table(data.get("reqs", [])),
                "obsolete_by": _name_of(data.get("obsolete_by", "")),
                "no_build": "NoBuild" in flags,
            }
        for sec, data in _iter_sections(ruleset_dir / "terrain.ruleset"):
            if sec.startswith("terrain_"):
                tclass = data.get("class", "").strip('"')
                if tclass == "Oceanic":
                    rs.oceanic_terrains.add(rule_name(data))
        return rs


@dataclass
class CityContext:
    """The facts one city-level evaluation needs.  Every field the caller
    cannot supply stays None, and any requirement touching it evaluates to
    unknown (fail-closed)."""

    techs_known: set  # player's known techs (rule names)
    city_buildings: list  # this city's buildings (rule names)
    city_size: int
    all_own_city_buildings: list | None = None  # union over own cities (SmallWonder)
    adjacent_terrains: set | None = None  # terrain rule names on adjacent tiles
    adjacent_extras: set | None = None  # extra rule names on adjacent tiles
    center_terrain: str | None = None
    center_extras: set | None = None


def _eval_req(req: dict, ctx: CityContext):
    """One requirement -> True / False / None (unknown).  Implements the
    (type, range) pairs the pinned civ2civ3 build reqs actually use;
    everything else is unknown -> fail closed."""
    rtype = req.get("type", "")
    rname = req.get("name", "")
    rrange = req.get("range", "")
    present = req.get("present", True)

    def known(v: bool):
        return v if present else (None if v is None else not v)

    if rtype == "Tech" and rrange == "Player":
        return known(rname in ctx.techs_known)
    if rtype == "Building":
        if rrange == "City":
            return known(rname in ctx.city_buildings)
        if rrange == "Player":
            if ctx.all_own_city_buildings is None:
                return None
            return known(rname in ctx.all_own_city_buildings)
        return None
    if rtype == "MinSize" and rrange == "City":
        try:
            return known(ctx.city_size >= int(rname))
        except ValueError:
            return None
    if rtype == "Extra" and rrange == "Adjacent":
        if ctx.adjacent_extras is None or ctx.center_extras is None:
            return None
        return known(rname in ctx.adjacent_extras or rname in ctx.center_extras)
    if rtype == "Terrain" and rrange == "Adjacent":
        if ctx.adjacent_terrains is None or ctx.center_terrain is None:
            return None
        return known(rname in ctx.adjacent_terrains or rname == ctx.center_terrain)
    if rtype in ("TerrainClass", "TerrainFlag"):
        # needs the terrain-class/flag tables, which CityContext does not carry
        return None
    if rtype == "None" or not rtype:
        return True
    return None  # unknown requirement type: fail closed


def _eval_reqs(reqs: list[dict], ctx: CityContext):
    """All-of evaluation: False dominates, then None, then True."""
    verdict = True
    for req in reqs:
        r = _eval_req(req, ctx)
        if r is False:
            return False
        if r is None:
            verdict = None
    return verdict


def can_city_build_improvement_now(rs: Ruleset, name: str, ctx: CityContext):
    b = rs.buildings.get(name)
    if b is None:
        return None
    if name in ctx.city_buildings:
        return False  # city_has_building (common/city.c:837)
    genus = b["genus"]
    if genus == "GreatWonder":
        # requires world-level "not built anywhere" knowledge: unknown
        return None
    if genus == "SmallWonder":
        if ctx.all_own_city_buildings is None:
            return None
        if name in ctx.all_own_city_buildings:
            return False
        # a small wonder absent from the visible building lists may still
        # exist (engine tracks small wonders per player, and rebuilding one
        # moves it) — legality cannot be proven either way: fail closed
        return None
    if genus == "Special":
        return None  # coinage-like conversions; never offered
    reqs_ok = _eval_reqs(b["reqs"], ctx)
    if reqs_ok is not True:
        return reqs_ok
    # improvement_obsolete: obsolete when obsolete_by reqs are ACTIVE
    if b["obsolete_by"]:
        obs = _eval_reqs(b["obsolete_by"], ctx)
        if obs is True:
            return False
        if obs is None:
            return None
    return True


def can_city_build_unit_now(rs: Ruleset, name: str, ctx: CityContext):
    u = rs.units.get(name)
    if u is None:
        return None
    if u.get("no_build"):
        return False  # utype_can_be_built / NoBuild flag
    reqs_ok = _eval_reqs(u["reqs"], ctx)
    if reqs_ok is not True:
        return reqs_ok
    # naval units inland (common/city.c:928): a class that is not
    # native-to-land must have oceanic terrain adjacent
    uclass = u["class"]
    if uclass in ("Sea", "Trireme"):
        if ctx.adjacent_terrains is None:
            return None
        if not (ctx.adjacent_terrains & rs.oceanic_terrains):
            return False
    # obsoleted_by chain (common/city.c:955): obsolete when the PLAYER can
    # already build the successor (player-level tech check)
    successor = u["obsolete_by"]
    seen = set()
    while successor and successor != "None" and successor not in seen:
        seen.add(successor)
        s = rs.units.get(successor)
        if s is None:
            return None
        s_ok = _eval_reqs(s["reqs"], ctx)
        if s_ok is True:
            return False  # successor buildable -> this unit is obsolete
        if s_ok is None:
            return None
        successor = s["obsolete_by"]
    return True
