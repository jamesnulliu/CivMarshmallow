"""observe_state(save, env) — an EnvSpec-gated, save-derived observation.

Builds the view a gameplay agent needs from one savegame, and honours the
EnvSpec so an experiment sees only the fields it declared. Besides each
player's identity, gold, score and city/unit counts, that is:

- known-tech **set** (which techs, by name) + current research + goal,
- government (rule name),
- tax/luxury/science rates and save-persisted diplomatic states,
- per-city economy: size, food/shield stocks, last-turn shield surplus,
  specialist count, and the buildings present (by name),
- per-unit hp / moves / veteran / fuel / activity / home city,
- optionally the terrain grid (opt-in — the whole map).

Everything is read from the per-turn autosave (no protocol decoding), and all
id->name mapping comes from the save's own `technology_vector` /
`improvement_vector` / `terrident`, so no ruleset dump is needed here.

Documented limitation: savegames do NOT persist live per-turn food/shield/
trade/gold yields (the engine recomputes them at load), so those are not
available from a save. Stocks, surplus, specialists and buildings are.
"""

import re
from pathlib import Path

from civharness import parse
from civharness.envspec import EnvSpec


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _optional_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _id(v):
    return int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else None


def _city_dict(row: dict, improvement_names: list[str]) -> dict:
    specialists = (
        _int(row.get("nspe0")) + _int(row.get("nspe1")) + _int(row.get("nspe2"))
    )
    return {
        "id": _id(row.get("id")),
        "name": row.get("name"),
        "x": _int(row.get("x")),
        "y": _int(row.get("y")),
        "size": _int(row.get("size")),
        "building": row.get("currently_building_name"),
        "food_stock": _int(row.get("food_stock")),
        "shield_stock": _int(row.get("shield_stock")),
        "shield_surplus": _int(row.get("last_turns_shield_surplus")),
        "specialists": specialists,
        "buildings": parse.bitstring_names(
            row.get("improvements", ""), improvement_names
        ),
    }


def _unit_dict(row: dict) -> dict:
    return {
        "id": _id(row.get("id")),
        "x": _int(row.get("x")),
        "y": _int(row.get("y")),
        "type": row.get("type_by_name"),
        "activity": row.get("activity"),
        "hp": _int(row.get("hp")),
        "moves": _int(row.get("moves")),
        "veteran": _int(row.get("veteran")),
        "fuel": _int(row.get("fuel")),
        "homecity": _id(row.get("homecity")),
    }


def _select(d: dict, sel) -> dict:
    """sel is True (all fields), or a list of field names to keep."""
    if sel is True:
        return d
    return {k: d[k] for k in sel if k in d}


_DIPLSTATE_NAMES = {
    0: "armistice",
    1: "war",
    2: "ceasefire",
    3: "peace",
    4: "alliance",
    5: "no_contact",
    6: "team",
}


def _diplomacy(kv: dict, player_names: dict[int, str]) -> dict[int, dict]:
    """Parse Freeciv's ``diplstateN.*`` save fields without guessing absence.

    Numeric codes are retained alongside their Freeciv 3.x enum labels, so an
    unknown future code remains usable instead of being silently mislabelled.
    """
    target_ids = sorted(
        {
            int(m.group(1))
            for key in kv
            if (m := re.fullmatch(r"diplstate(\d+)\.type", key))
        }
    )
    out = {}
    for target_id in target_ids:
        prefix = f"diplstate{target_id}."
        code = _optional_int(kv.get(prefix + "type"))
        out[target_id] = {
            "player_id": target_id,
            "player": player_names.get(target_id),
            "state_raw": kv.get(prefix + "type"),
            "state_code": code,
            "state": _DIPLSTATE_NAMES.get(
                code, f"unknown_{code}" if code is not None else None
            ),
            "max_state_code": _optional_int(kv.get(prefix + "max_state")),
            "first_contact_turn": _optional_int(kv.get(prefix + "first_contact_turn")),
            "turns_left": _optional_int(kv.get(prefix + "turns_left")),
            "contact_turns_left": _optional_int(kv.get(prefix + "contact_turns_left")),
            "has_reason_to_cancel": _optional_int(
                kv.get(prefix + "has_reason_to_cancel")
            ),
        }
    return out


def observe_state(save, env: EnvSpec | None = None) -> tuple[int, dict, dict | None]:
    """Parse `save` once into (turn, {player_name: fields}, terrain_or_None),
    including only the fields the EnvSpec asks for. `_drive` slices this per
    player and applies the players: self|all visibility knob."""
    env = env or EnvSpec.full()
    sections = parse.parse_save_sections(Path(save))
    scores = parse.save_scores(sections)
    sf = sections.get("savefile", {})
    tech_names = parse.parse_vector(sf.get("technology_vector", ""))
    impr_names = parse.parse_vector(sf.get("improvement_vector", ""))
    research = parse.save_research(sections)

    want_techs = env.wants("techs")
    techs_mode = env.obs("techs")
    want_gov = env.wants("government")
    want_rates = env.wants("rates")
    want_diplomacy = env.wants("diplomacy")
    cities_sel = env.obs("cities")
    units_sel = env.obs("units")

    player_names = parse.save_players(sections)
    players: dict[str, dict] = {}
    for sec, kv in sections.items():
        m = re.fullmatch(r"player(\d+)", sec)
        if not m:
            continue
        name = kv.get("name", "").strip('"')
        if not name:
            continue
        score = scores.get(name, {})
        p = {
            "player_id": int(m.group(1)),
            "name": name,
            "gold": _int(kv.get("gold")),
            "nation": kv.get("nation", "").strip('"'),
            "is_alive": kv.get("is_alive", "").upper().endswith("TRUE"),
            "ncities": _int(kv.get("ncities")),
            "nunits": _int(kv.get("nunits")),
            "score": score.get("total"),
        }
        if want_techs:
            p["techs"] = score.get("techs")  # count
            if techs_mode == "known_set":
                team = _int(kv.get("team_no"))
                r = research.get(team, {})
                # A_NONE (id 0) is the always-set "no tech" sentinel — drop it so
                # techs_known counts real advances (matching the `techs` count).
                p["techs_known"] = set(
                    parse.bitstring_names(r.get("done", ""), tech_names)
                ) - {"A_NONE"}
                p["researching"] = r.get("now")
                p["research_goal"] = r.get("goal")
        if want_gov:
            p["government"] = kv.get("government_name", "").strip('"')
        if want_rates:
            p["rates"] = {
                "tax": _optional_int(kv.get("tax")),
                "luxury": _optional_int(kv.get("luxury")),
                "science": _optional_int(kv.get("science")),
            }
        if want_diplomacy:
            p["diplomacy"] = _diplomacy(kv, player_names)
        if cities_sel:
            rows = parse.parse_table(kv["c"]) if "c" in kv else []
            p["cities"] = [_select(_city_dict(r, impr_names), cities_sel) for r in rows]
        if units_sel:
            rows = parse.parse_table(kv["u"]) if "u" in kv else []
            p["units"] = [_select(_unit_dict(r), units_sel) for r in rows]
        players[name] = p

    terrain = parse.save_terrain(sections) if env.wants("terrain") else None
    return parse.save_turn(sections), players, terrain
