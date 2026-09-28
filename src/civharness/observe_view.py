"""Build a player-visible Observation from a PlayerViewCache.

``observe_state`` (observe.py) reads the omniscient autosave; that remains
the ``full_state`` path.  This module is the ``player_visible`` path: it
materializes an Observation-shaped dict tree from what one client actually
received over its connection, so hidden tiles/cities/units are absent by
construction, not by filtering.

Name resolution comes from RuleIds (the Lua rule dump: unit types, terrain,
improvements, techs, governments).  Fields the wire genuinely does not carry
for a fogged entity (e.g. a remembered city's food stock) are absent or
None — never fabricated.
"""

from __future__ import annotations

from civharness.playerview import (
    MAX_EXTRA_TYPES,
    TILE_KNOWN_SEEN,
    TILE_KNOWN_UNSEEN,
    PlayerViewCache,
)

# enum unit_activity value -> the save-file style label observe.py produces
# (common/fc_types.h SPECENUM; labels match savegame "activity" strings)
_ACTIVITY_NAMES = {
    0: "Idle",
    1: "Cultivate",
    2: "Mine",
    3: "Irrigate",
    4: "Fortified",
    5: "Sentry",
    6: "Pillage",
    7: "Goto",
    8: "Explore",
    9: "Transform",
    10: "Fortifying",
    11: "Clean",
    12: "Base",
    13: "Road",
    14: "Convert",
    15: "Plant",
}

_KNOWN_LABELS = {0: "unknown", 1: "remembered", 2: "visible"}


def _invert(d: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in (d or {}).items()}


def view_tiles(cache: PlayerViewCache, rules, catalog=None) -> dict[int, dict]:
    """Typed tile map: only tiles this connection ever learned about.
    ``known`` keeps the engine's three-state semantics; a remembered tile
    carries its remembered contents (that is what the server last sent while
    it was visible), and an unknown tile is simply absent.

    ``catalog`` (the connection's RulesetCatalog) is required to interpret
    extras: the server's fogged send_tile_info paths copy only the ruleset's
    actual extra count into the 250-bit wire field (dbv_to_bv, maphand.c) and
    never zero the remainder, so bits past the ruleset's extras are
    uninitialized server memory and must be dropped, not decoded."""
    terrain_names = _invert(getattr(rules, "terrains", {}) or {})
    cat_extras = getattr(catalog, "extras", None) or {}
    # 3.2.5 Lua has no find.extra, so RuleIds.extras is empty and the wire
    # catalog is the only extra-name source; fall back to it if ever present.
    extra_names = _invert(getattr(rules, "extras", {}) or {})
    extra_names.update({i: rec["rule_name"] for i, rec in cat_extras.items()})
    valid_extras = set(cat_extras) or None
    out = {}
    for idx, rec in cache.tiles.items():
        if rec["known"] not in (TILE_KNOWN_SEEN, TILE_KNOWN_UNSEEN):
            continue  # an unknown tile exposes nothing
        xy = cache.xy(idx)
        resource_id = rec["resource"]
        extras = [e for e in rec["extras"] if valid_extras is None or e in valid_extras]
        out[idx] = {
            "tile": idx,
            "x": xy[0] if xy else None,
            "y": xy[1] if xy else None,
            "known": rec["known"],
            "known_label": _KNOWN_LABELS[rec["known"]],
            "terrain": terrain_names.get(rec["terrain"], f"terrain_{rec['terrain']}"),
            "terrain_id": rec["terrain"],
            "owner": rec["owner"],
            "extras": [extra_names.get(e, f"extra_{e}") for e in extras],
            "extra_ids": extras,
            "resource_id": (resource_id if resource_id != MAX_EXTRA_TYPES else None),
        }
    return out


def view_cities(cache: PlayerViewCache, rules, *, my_player_id: int) -> list[dict]:
    """Cities as this player sees them.  Own cities arrive as CITY_INFO
    (full economy); foreign visible/remembered ones as CITY_SHORT_INFO
    (name/size/owner/walls only).  Fields the short form does not carry are
    None — the consumer can tell 'unknown' from 'zero'."""
    impr_names = _invert(getattr(rules, "improvements", {}) or {})
    utype_names = _invert(getattr(rules, "units", {}) or {})
    out = []
    for cid, rec in sorted(cache.cities.items()):
        xy = cache.xy(rec["tile"])
        full = rec["_form"] == "full"
        c = {
            "id": cid,
            "name": rec["name"],
            "x": xy[0] if xy else None,
            "y": xy[1] if xy else None,
            "tile": rec["tile"],
            "owner": rec["owner"],
            "mine": rec["owner"] == my_player_id,
            "size": rec["size"],
            "form": rec["_form"],
            "buildings": [impr_names.get(b, f"impr_{b}") for b in rec["improvements"]],
        }
        if full:
            # production_kind: universals_n — VUT_UTYPE(6)=unit,
            # VUT_IMPROVEMENT(3)=building (common/fc_types.h)
            kind, val = rec["production_kind"], rec["production_value"]
            if kind == 6:
                building = utype_names.get(val, f"utype_{val}")
            elif kind == 3:
                building = impr_names.get(val, f"impr_{val}")
            else:
                building = None
            c.update(
                {
                    "building": building,
                    "food_stock": rec["food_stock"],
                    "shield_stock": rec["shield_stock"],
                    "shield_surplus": rec["last_turns_shield_surplus"],
                    "specialists": sum(rec["specialists"]),
                }
            )
        else:
            c.update(
                {
                    "building": None,
                    "food_stock": None,
                    "shield_stock": None,
                    "shield_surplus": None,
                    "specialists": None,
                }
            )
        out.append(c)
    return out


def view_units(cache: PlayerViewCache, rules, *, my_player_id: int) -> list[dict]:
    """Units this player currently sees.  Own units are UNIT_INFO (moves,
    fuel, homecity); foreign ones UNIT_SHORT_INFO (type/hp/activity only).
    A unit that went out of sight was removed by UNIT_REMOVE and is absent —
    the engine does not remember old unit positions and neither do we."""
    utype_names = _invert(getattr(rules, "units", {}) or {})
    out = []
    for uid, rec in sorted(cache.units.items()):
        xy = cache.xy(rec["tile"])
        full = rec["_form"] == "full"
        u = {
            "id": uid,
            "type": utype_names.get(rec["type"], f"utype_{rec['type']}"),
            "x": xy[0] if xy else None,
            "y": xy[1] if xy else None,
            "tile": rec["tile"],
            "owner": rec["owner"],
            "mine": rec["owner"] == my_player_id,
            "hp": rec["hp"],
            "veteran": rec["veteran"],
            "activity": _ACTIVITY_NAMES.get(
                rec["activity"], f"activity_{rec['activity']}"
            ),
            "form": rec["_form"],
        }
        if full:
            u.update(
                {
                    "moves": rec["movesleft"],
                    "fuel": rec["fuel"],
                    "homecity": rec["homecity"],
                }
            )
        else:
            u.update({"moves": None, "fuel": None, "homecity": None})
        out.append(u)
    return out


def player_view(
    cache: PlayerViewCache, rules, *, my_player_id: int, catalog=None
) -> dict:
    """The complete player-visible map/entity view, ready to embed in an
    Observation.  Everything traces to packets this connection received.
    ``catalog`` is the connection's RulesetCatalog (see view_tiles)."""
    return {
        "visibility": "player_visible",
        "map": {
            "xsize": cache.map_info.get("xsize"),
            "ysize": cache.map_info.get("ysize"),
            "tiles": view_tiles(cache, rules, catalog),
        },
        "cities": view_cities(cache, rules, my_player_id=my_player_id),
        "units": view_units(cache, rules, my_player_id=my_player_id),
    }
