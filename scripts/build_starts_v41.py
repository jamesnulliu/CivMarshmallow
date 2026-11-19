#!/usr/bin/env python3
"""Start positions for environment v4.1 (civm41 ruleset), with an audit of every game.

Each seed is one bot-only game (four Hard bots, 200 tiles per player, to turn 120,
a save every turn). The turn-1 save is the start position; the whole game is the
bots' reference game for the same start.

A start is kept (``ok`` in the manifest) only when the audit passes:

  one_continent  every player's turn-1 units stand on the same land component (ships
                 are banned, and startpos ALL silently falls back on some seeds)
  no_banned      no banned unit or building exists or is in production in any
                 audited save
  no_treaty      no cease-fire, armistice, peace or alliance between any two players
                 (diplomacy is disabled)

Also recorded: land tiles of the shared continent per player, the scores and cities at
turn 120. The server is CIVHARNESS_SERVER with the civm41 ruleset installed
(scripts/install_freeciv_server.sh).

    python scripts/build_starts_v41.py --seeds 9100-9103 --workers 4 --out data/starts_v41
"""

import argparse
import gzip
import json
import os
import re
import shutil
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RULESET = "civm41"
ENDTURN, TILES_PER_PLAYER = 120, 200
OCEANIC = {"Lake", "Ocean", "Deep Ocean"}
TREATIES = ("Cease-fire", "Armistice", "Peace", "Alliance")


def banned_sets():
    """(units with NoBuild, buildings with the never-true Nation requirement)."""
    from civharness.buildability import Ruleset, pinned_ruleset_dir

    rs = Ruleset.load(pinned_ruleset_dir(RULESET))
    units = {n for n, u in rs.units.items() if u.get("no_build")}
    buildings = {
        n
        for n, b in rs.buildings.items()
        if any(
            r.get("type") == "Nation" and r.get("name") == "Animal Kingdom"
            for r in b["reqs"]
        )
    }
    return units - {"Barbarian Leader", "Leader", "Storm"}, buildings


def parse_save(save):
    """(turn, {player: fields}, terrain) of a save, full information."""
    from civharness import EnvSpec
    from civharness.observe import observe_state

    env = EnvSpec.from_dict(
        {
            "name": "start-audit",
            "observations": {
                "techs": "known_set",
                "cities": True,
                "units": True,
                "terrain": True,
                "players": "all",
                "visibility": "full_state",
            },
        }
    )
    return observe_state(save, env)


def read_text(p):
    with open(p, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(p, "rt") as fh:
            return fh.read()
    return Path(p).read_text()


def land_components(terrain):
    """{land tile: component id} over the iso-hex map."""
    from civmarsh.env.v41.hexmap import HexMap

    xs, ys = terrain["xsize"], terrain["ysize"]
    hm = HexMap(xs, ys)
    legend = terrain["legend"]
    land = set()
    for y, row in enumerate(terrain["grid"]):
        for x, ident in enumerate(row):
            name = legend.get(ident) or str(ident)
            if name not in OCEANIC and name != "Inaccessible":
                land.add(y * xs + x)
    comp, cid = {}, 0
    for t in land:
        if t in comp:
            continue
        cid += 1
        comp[t] = cid
        q = deque([t])
        while q:
            a = q.popleft()
            for _d, b in hm.neighbors(a):
                if b in land and b not in comp:
                    comp[b] = cid
                    q.append(b)
    return comp


def audit(wd, banned_units, banned_buildings):
    saves = sorted(Path(wd, "saves").glob("freeciv-T*-auto.sav*"))
    _turn, players, terrain = parse_save(saves[0])
    comp = land_components(terrain)
    xs = terrain["xsize"]
    start_comp = {}
    for name, p in players.items():
        cs = {comp.get(u["y"] * xs + u["x"]) for u in p.get("units") or []}
        if cs:
            start_comp[name] = sorted(c for c in cs if c is not None)
    comps = {c for v in start_comp.values() for c in v}
    sizes = {}
    for c in comp.values():
        sizes[c] = sizes.get(c, 0) + 1
    one = len(comps) == 1 and all(len(v) == 1 for v in start_comp.values())
    shared = sizes.get(next(iter(comps)), 0) if one else None
    bad, treaty = [], []
    banned_items = banned_units | banned_buildings
    for sv in [*saves[::5], saves[-1]]:
        t, ps, _ = parse_save(sv)
        for name, p in ps.items():
            for u in p.get("units") or []:
                if u.get("type") in banned_units:
                    bad.append((t, name, "unit", u["type"]))
            for c in p.get("cities") or []:
                for b in c.get("buildings") or []:
                    if b in banned_buildings:
                        bad.append((t, name, "building", b))
                if c.get("building") in banned_items:
                    bad.append((t, name, "production", c["building"]))
    pattern = re.compile(r'^"({})","'.format("|".join(TREATIES)), re.MULTILINE)
    for sv in [*saves[::10], saves[-1]]:
        for m in pattern.finditer(read_text(sv)):
            treaty.append((sv.name, m.group(1)))
    final_turn, final_players, _ = parse_save(saves[-1])
    return {
        "one_continent": one,
        "start_components": start_comp,
        "shared_land_tiles": shared,
        "land_per_player": round(shared / max(1, len(start_comp)), 1)
        if shared
        else None,
        "no_banned": not bad,
        "banned_seen": bad[:20],
        "no_treaty": not treaty,
        "treaties_seen": treaty[:10],
        "final_turn": final_turn,
        "scores_end": {n: p.get("score") for n, p in final_players.items()},
        "cities_end": {n: len(p.get("cities") or []) for n, p in final_players.items()},
    }


def one(seed, out, prefix):
    from civharness import GameConfig, position, snapshot_series

    gid = f"{prefix}{seed}"
    wd = (Path(out) / gid).resolve()
    if wd.exists():
        shutil.rmtree(wd)
    cfg = GameConfig(
        aifill=4,
        skill="hard",
        endturn=ENDTURN,
        mapseed=seed,
        gameseed=seed,
        size=1,
        ruleset=RULESET,
        extra=(f"set tilesperplayer {TILES_PER_PLAYER}",),
    )
    refs = sorted(snapshot_series(cfg, wd, every=1), key=lambda r: r.turn)
    turns = [r.turn for r in refs]
    if not refs or turns != list(range(1, turns[-1] + 1)):
        raise AssertionError(
            f"non-contiguous snapshots: {gid}: {turns[:3]}..{turns[-3:]}"
        )
    a = audit(wd, *banned_sets())
    end = position(refs[-1].path)
    focal = next(n for n, p in end.players.items() if p["player_id"] == 0)
    ok = (
        a["one_continent"]
        and a["no_banned"]
        and a["no_treaty"]
        and a["final_turn"] == ENDTURN
    )
    return {
        "game_id": gid,
        "seed": seed,
        "ruleset": RULESET,
        "save_path": str(refs[0].path),
        "turn": refs[0].turn,
        "focal_player": focal,
        "ok": ok,
        **a,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", required=True, help="a-b inclusive")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--out", required=True, help="output directory (manifest starts.jsonl)"
    )
    ap.add_argument("--prefix", default="v41g", help="game id prefix")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "starts.jsonl"
    lo, hi = map(int, a.seeds.split("-"))
    done = set()
    if manifest.exists():
        done = {json.loads(line)["seed"] for line in manifest.open() if line.strip()}
    todo = [s for s in range(lo, hi + 1) if s not in done]
    print(f"{len(todo)} games queued ({len(done)} in manifest)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, manifest.open("a") as fh:
        futs = {ex.submit(one, s, str(out), a.prefix): s for s in todo}
        for f in as_completed(futs):
            s = futs[f]
            try:
                row = f.result()
            except Exception:  # noqa: BLE001
                print(s, "FAILED\n" + traceback.format_exc(limit=3), flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            print(
                s,
                "ok" if row["ok"] else "REJECT",
                "continent",
                row["one_continent"],
                "land/player",
                row["land_per_player"],
                "banned",
                row["no_banned"],
                "treaty-free",
                row["no_treaty"],
                "focal score",
                row["scores_end"].get(row["focal_player"]),
                flush=True,
            )


if __name__ == "__main__":
    main()
