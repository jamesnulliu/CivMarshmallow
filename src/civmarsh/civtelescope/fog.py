"""Player-visible (fog-of-war) renderings of bank positions.

CivTelescope reads exactly what the policy's player would see, so each bank
position is rendered from the focal player's fog view, in two stages.

Recapture (needs the Freeciv server): load the position's autosave, take over
the focal player with a client, and record the view the server sends at login.
The server rebuilds that view from the knowledge the save persists per player
(terrain and extras memory, remembered cities) plus current unit vision, so no
replay is needed; observing a replay would perturb the game's random stream.
Each capture is cross-checked against the save's own knowledge layers.

Render (CPU only): rebuild the observation the live fog-of-war path gives an
agent (own-player fields from the save, own cities and units from the view,
no other players' dicts, no raw save) and render it with
``SpatialSnapshot.render_value_input()``.

Input positions are bank records with position_id, game_id, turn,
focal_player and save_path (absolute, or relative to the bank's pool
directory).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

CAPTURE_ATTEMPTS = 3

# Own-player observation fields of the player-visible environment; world
# entities come from the recaptured view, not from the save.
_ENV_DICT = {
    "name": "fog_render",
    "observations": {
        "techs": "known_set",
        "cities": True,
        "units": True,
        "government": True,
        "rates": True,
        "diplomacy": True,
        "players": "self",
    },
}


def save_layers(sections: dict, pid: int):
    """(known tiles, extras_of(tile)) from a save's persisted knowledge of
    player `pid`: the ``map_t*`` rows ('u' = unknown) and the ``map_eNN_*``
    hex-nibble extras bit layers."""
    psec = sections.get(f"player{pid}", {})
    known, xsize = set(), None
    for k in sorted(k for k in psec if re.fullmatch(r"map_t\d+", k)):
        y = int(k[5:])
        line = psec[k].strip('"')
        if xsize is None:
            xsize = len(line)
        for x, ch in enumerate(line):
            if ch != "u":
                known.add(y * xsize + x)
    layers = {}
    for k, v in psec.items():
        m = re.fullmatch(r"map_e(\d\d)_(\d+)", k)
        if m:
            layers[(int(m.group(1)), int(m.group(2)))] = v.strip('"')

    def extras_of(tile: int) -> set[int]:
        y, x = divmod(tile, xsize)
        out = set()
        for (layer, yy), row in layers.items():
            if yy == y:
                bits = int(row[x], 16)
                out.update(layer * 4 + b for b in range(4) if bits >> b & 1)
        return out

    return known, extras_of


def capture_position(save: Path, focal: str, turn: int, *, endturn: int) -> dict:
    """Load `save`, take `focal`, and capture the login-burst fog view.

    `endturn` is the server's end turn for the loaded game; it must exceed
    `turn`. Returns {"pid", "player_view", "crosscheck"}; the crosscheck
    compares the captured known tiles and extras with the save's layers."""
    from civharness import server
    from civharness.branch import players_of, wait_for_log_marker
    from civharness.client import RULE_DUMP_LUA, FreecivClient, RuleIds
    from civharness.config import client_resume_script
    from civharness.observe_view import player_view
    from civharness.parse import parse_save_sections

    pid = next(i for i, n in players_of(save).items() if n == focal)
    with tempfile.TemporaryDirectory(prefix="fogrec-") as td:
        wd = Path(td)
        dump = wd / "dump.lua"
        dump.write_text(RULE_DUMP_LUA)
        h = server.launch(
            client_resume_script(endturn, dump_lua_path=dump, scorelog=False),
            wd,
            load=save,
        )
        try:
            wait_for_log_marker(h, "CIVHARNESS dump_done", timeout=60)
            h.assert_loaded()
            cli = FreecivClient("127.0.0.1", h.port, username="civ0")
            cli.rules = RuleIds.from_log(h.log)
            cli.turn = turn
            cli.connect()
            cli.pump(0.5)
            cli.take(focal)
            cli.pump(0.5)
            cli.start()
            t = cli.next_phase(max_seconds=60)
            if t != turn:
                raise RuntimeError(f"expected START_PHASE turn {turn}, got {t}")
            cli.pump(1.0)
            snap = cli.view.snapshot()
            pv = player_view(snap, cli.rules, my_player_id=pid, catalog=cli.rulesets)
            cli.close()
        finally:
            h.kill()

    known_save, extras_of = save_layers(parse_save_sections(save), pid)
    tiles = pv["map"]["tiles"]
    known_cache = set(tiles)
    extras_mismatch = sum(
        1 for idx, rec in tiles.items() if set(rec["extra_ids"]) != extras_of(idx)
    )
    return {
        "pid": pid,
        "player_view": pv,
        "crosscheck": {
            "known_cache": len(known_cache),
            "known_save": len(known_save),
            "known_equal": known_cache == known_save,
            "extras_mismatch": extras_mismatch,
        },
    }


def recapture_game(rows: list[dict], out_file: Path, *, endturn: int, pool_dir=None):
    """Recapture every position of one game into a gzipped JSONL file.

    A complete existing file is kept (the stage is resumable per game).
    Returns {"game_id", "cached", "n", "bad"}; `bad` counts positions whose
    crosscheck failed."""
    out_file = Path(out_file)
    gid = rows[0]["game_id"]
    if out_file.exists():
        with gzip.open(out_file, "rt") as f:
            n = sum(1 for _ in f)
        if n == len(rows):
            return {"game_id": gid, "cached": True, "n": n, "bad": 0}
    recs, bad = [], 0
    for r in sorted(rows, key=lambda r: r["turn"]):
        save = Path(pool_dir or ".") / r["save_path"]
        focal = r.get("focal_player") or r["focal"]
        last_err = None
        for attempt in range(CAPTURE_ATTEMPTS):
            try:
                cap = capture_position(save, focal, r["turn"], endturn=endturn)
                break
            except Exception as e:  # noqa: BLE001 -- port race or transient failure
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"{r['position_id']}: {last_err}")
        cc = cap["crosscheck"]
        if not cc["known_equal"] or cc["extras_mismatch"]:
            bad += 1
        rec = {
            "position_id": r["position_id"],
            "game_id": gid,
            "turn": r["turn"],
            "focal_player": focal,
            "pid": cap["pid"],
            "save_path": str(save),
            "crosscheck": cc,
            "player_view": cap["player_view"],
        }
        if "split" in r:
            rec["split"] = r["split"]
        recs.append(rec)
    tmp = out_file.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    tmp.rename(out_file)
    return {"game_id": gid, "cached": False, "n": len(recs), "bad": bad}


def recapture_positions(
    positions: list[dict],
    rec_dir,
    *,
    endturn: int,
    pool_dir=None,
    workers: int = 24,
    log=print,
) -> dict:
    """Recapture all `positions`, one ``<game_id>.jsonl.gz`` per game under
    `rec_dir`, games in parallel. Returns a summary with per-game errors."""
    t0 = time.time()
    rec_dir = Path(rec_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)
    by_game: dict[str, list] = {}
    for r in positions:
        by_game.setdefault(r["game_id"], []).append(r)
    stats, errs = [], []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                recapture_game,
                rows,
                rec_dir / f"{gid}.jsonl.gz",
                endturn=endturn,
                pool_dir=pool_dir,
            ): gid
            for gid, rows in sorted(by_game.items())
        }
        for i, f in enumerate(as_completed(futs), 1):
            gid = futs[f]
            try:
                s = f.result()
            except Exception as e:  # noqa: BLE001 -- one failed game, not a stop
                errs.append({"game_id": gid, "error": f"{type(e).__name__}: {e}"})
                log(f"[recapture] error {gid}: {e}")
                continue
            with lock:
                stats.append(s)
            if i % 15 == 0 or i == len(futs):
                bad = sum(s["bad"] for s in stats)
                log(
                    f"[recapture] {i}/{len(futs)} games, crosscheck bad={bad} "
                    f"({time.time() - t0:.0f}s)"
                )
    return {
        "games": len(stats),
        "errors": errs,
        "positions": sum(s["n"] for s in stats),
        "crosscheck_bad_positions": sum(s["bad"] for s in stats),
        "elapsed_s": round(time.time() - t0, 1),
    }


def render_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def render_recapture(rec: dict) -> dict:
    """The value-input rendering of one recaptured position."""
    from civharness import Observation
    from civharness.envspec import EnvSpec
    from civharness.observe import observe_state
    from civmarsh.env.snapshot import SpatialSnapshot

    env = EnvSpec.from_dict(_ENV_DICT)
    pv = rec["player_view"]
    focal = rec.get("focal_player") or rec["focal"]
    _turn, players, _terrain = observe_state(Path(rec["save_path"]), env)
    mine = dict(players[focal])
    mine["cities"] = [c for c in pv["cities"] if c["mine"]]
    mine["units"] = [u for u in pv["units"] if u["mine"]]
    obs = Observation(
        turn=rec["turn"],
        me=focal,
        my=mine,
        players={focal: mine},
        save_path=None,
        terrain=None,
        rates=mine.get("rates"),
        diplomacy=mine.get("diplomacy"),
        player_view=pv,
    )
    snap = SpatialSnapshot.from_observation(obs)
    text = snap.render_value_input()
    row = {
        "position_id": rec["position_id"],
        "game_id": rec["game_id"],
        "turn": rec["turn"],
        "focal_player": focal,
        "format": snap.format_name,
        "fact_sha": snap.fact_sha,
        "render_sha": render_sha(text),
        "rendering": text,
    }
    if "split" in rec:
        row["split"] = rec["split"]
    return row


def _render_file(rec_path) -> list[dict]:
    with gzip.open(rec_path, "rt") as f:
        return [render_recapture(json.loads(line)) for line in f]


def render_recaptures(rec_dir, out_path, *, workers: int = 16, log=print) -> dict:
    """Render every recaptured position under `rec_dir` into one JSONL file
    sorted by position id. Returns summary statistics."""
    t0 = time.time()
    rec_files = sorted(Path(rec_dir).glob("*.jsonl.gz"))
    if not rec_files:
        raise FileNotFoundError(f"no recapture files under {rec_dir}")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, chunk in enumerate(ex.map(_render_file, rec_files), 1):
            rows.extend(chunk)
            if i % 30 == 0:
                log(f"[render] {i}/{len(rec_files)} games ({time.time() - t0:.0f}s)")
    rows.sort(key=lambda r: r["position_id"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    fmts: dict[str, int] = {}
    for r in rows:
        fmts[r["format"]] = fmts.get(r["format"], 0) + 1
    lens = sorted(len(r["rendering"]) for r in rows)
    return {
        "positions": len(rows),
        "formats": fmts,
        "render_chars_p50": lens[len(lens) // 2],
        "render_chars_p95": lens[int(len(lens) * 0.95)],
        "render_chars_max": lens[-1],
        "distinct_render_sha": len({r["render_sha"] for r in rows}),
        "elapsed_s": round(time.time() - t0, 1),
    }
