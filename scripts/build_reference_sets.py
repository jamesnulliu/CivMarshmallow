#!/usr/bin/env python3
"""Build the CivTelescope reference sets that the potential compares against.

Usage:
  # 1. bot-played reference games, saved every 5 turns (needs CIVHARNESS_SERVER)
  python scripts/build_reference_sets.py games --data-root DATA \\
      --seeds 8000-8119 --tiles-per-player 200
  python scripts/build_reference_sets.py games --data-root DATA \\
      --seeds 9000-9031 --tiles-per-player 300
  # 2. bot reference positions: 8 games per turn bucket 10, 20, ..., 110
  python scripts/build_reference_sets.py bot --data-root DATA
  # 3. one reference set per phase, from policy episodes harvested by
  #    CivTelescope runs on that phase (value_harvest.jsonl)
  python scripts/build_reference_sets.py phase --data-root DATA --phase rem80 \\
      --harvest RUN_A/value_harvest.jsonl --harvest RUN_B/value_harvest.jsonl

Bot positions (``bot``): the held-out games of the pool (seed % 5 == 0; they are
kept out of CivTelescope training) are ranked by the focal player's score at the
horizon turn (120), and the same 8 games, evenly spaced over that ranking, fill
every turn bucket. Each position is recaptured from the focal player's fog view
and rendered as the value input (``civmarsh.civtelescope.fog``), so the
reference renderings match what the potential sees during training. Rows:
``{turn, focal, rendering, end_score, position_id}`` with ``end_score`` the
focal score at the horizon. Output: DATA/refsets/bot.jsonl.

Phase sets (``phase``): the bot positions of the buckets the phase reaches (turn
>= its start turn) seed ``civmarsh.rewards.reference_set.build_reference_set``:
per bucket, 6 harvested policy positions evenly spaced by final score plus the
2 strongest bot positions. Harvested episodes that ended in an elimination or
without an end turn are left out of the pool. Without --harvest the phase set is
the bot positions alone. Output: DATA/refsets/<phase>.jsonl plus a
``.stats.json`` summary, whose audit lists any position labelled 0 whose
rendering shows a score of 30 or more.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HORIZON = 120
BUCKETS = tuple(range(10, HORIZON, 10))
PER_BUCKET = 8
REFERENCE_ENDTURN = 200
SNAPSHOT_EVERY = 5
CAPTURE_ENDTURN = 130
CAPTURE_ATTEMPTS = 3
PHASE_START_TURN = {"rem20": 100, "rem40": 80, "rem80": 40, "rem120": 1}
AUDIT_SCORE = 30


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(tmp, path)


def _save_at(workdir: Path, turn: int) -> Path:
    path = next(iter(sorted((workdir / "saves").glob(f"freeciv-T{turn:04d}-*"))), None)
    if path is None:
        raise FileNotFoundError(f"no T{turn:04d} save under {workdir}")
    return path


# ---------------------------------------------------------------------- games


def play_game(seed: int, tiles_per_player: int, data_root: Path) -> dict:
    from civharness import GameConfig, position, snapshot_series
    from civmarsh.utils.freeciv import focal_player

    gid = f"tpp{tiles_per_player}s{seed}"
    workdir = (data_root / "refgames" / gid).resolve()
    config = GameConfig(
        aifill=4,
        skill="hard",
        endturn=REFERENCE_ENDTURN,
        mapseed=seed,
        gameseed=seed,
        size=1,
        extra=(f"set tilesperplayer {tiles_per_player}",),
    )
    refs = sorted(
        snapshot_series(config, workdir, every=SNAPSHOT_EVERY), key=lambda r: r.turn
    )
    final = position(refs[-1].path)
    return {
        "game_id": gid,
        "seed": seed,
        "tiles_per_player": tiles_per_player,
        "workdir": str(workdir.relative_to(data_root.resolve())),
        "focal_player": focal_player(final),
        "end_turn": final.turn,
    }


def cmd_games(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    manifest = data_root / "refsets" / "games.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = {row["game_id"] for row in _read_jsonl(manifest)}
    tpp = args.tiles_per_player
    seeds = [s for s in _seeds(args.seeds) if f"tpp{tpp}s{s}" not in done]
    print(f"{len(seeds)} games queued ({len(done)} already in {manifest})", flush=True)
    failed = []
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        manifest.open("a") as out,
    ):
        futures = {pool.submit(play_game, s, tpp, data_root): s for s in seeds}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception:  # noqa: BLE001 - a failed game is reported, the rest continue
                failed.append(futures[future])
                print(traceback.format_exc(limit=2), flush=True)
            if n % 10 == 0:
                print(f"{n}/{len(futures)}", flush=True)
    if failed:
        raise SystemExit(f"failed seeds: {sorted(failed)}")


# ------------------------------------------------------------------------ bot


def render_position(save: Path, focal: str, turn: int) -> str:
    """Fog-of-war recapture of one saved position, rendered as the value input."""
    from civmarsh.civtelescope.fog import capture_position, render_recapture

    for attempt in range(CAPTURE_ATTEMPTS):
        try:
            captured = capture_position(save, focal, turn, endturn=CAPTURE_ENDTURN)
            break
        except Exception:
            if attempt == CAPTURE_ATTEMPTS - 1:
                raise
            time.sleep(1 + attempt)
    record = {
        "player_view": captured["player_view"],
        "focal_player": focal,
        "save_path": str(save),
        "turn": turn,
        "position_id": "",
        "game_id": "",
    }
    return render_recapture(record)["rendering"]


def select_games(games: list[dict], scores: dict[str, int], m: int) -> list[dict]:
    """``m`` games evenly spaced over the ranking by horizon score (ties by id)."""
    ordered = sorted(games, key=lambda g: (scores[g["game_id"]], g["game_id"]))
    idx = [round(i * (len(ordered) - 1) / (m - 1)) for i in range(m)]
    return [ordered[i] for i in idx]


def cmd_bot(args: argparse.Namespace) -> None:
    from civharness import position

    data_root = Path(args.data_root)
    games = [
        g
        for g in _read_jsonl(data_root / "refsets" / "games.jsonl")
        if g["seed"] % args.heldout_mod == 0
    ]
    if len(games) < args.per_bucket:
        raise SystemExit(f"{len(games)} held-out reference games < {args.per_bucket}")
    scores = {}
    for g in games:
        players = position(_save_at(data_root / g["workdir"], args.horizon)).players
        focal = g["focal_player"]
        scores[g["game_id"]] = players[focal]["score"] if focal in players else 0
    chosen = select_games(games, scores, args.per_bucket)
    buckets = [t for t in BUCKETS if t < args.horizon]
    jobs = sorted({(g["game_id"], t) for g in chosen for t in buckets})
    by_id = {g["game_id"]: g for g in chosen}
    print(f"{len(games)} held-out games, {len(jobs)} positions to render", flush=True)

    renders, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_position,
                _save_at(data_root / by_id[gid]["workdir"], t),
                by_id[gid]["focal_player"],
                t,
            ): (gid, t)
            for gid, t in jobs
        }
        for n, future in enumerate(as_completed(futures), 1):
            try:
                renders[futures[future]] = future.result()
            except Exception:  # noqa: BLE001 - collected and reported below
                failed.append(futures[future])
                traceback.print_exc()
            if n % 20 == 0:
                print(f"[render] {n}/{len(jobs)}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} recaptures failed: {sorted(failed)}")

    rows = [
        {
            "turn": t,
            "focal": g["focal_player"],
            "rendering": renders[(g["game_id"], t)],
            "end_score": scores[g["game_id"]],
            "position_id": f"{g['game_id']}_T{t:03d}",
        }
        for t in buckets
        for g in chosen
    ]
    out = data_root / "refsets" / "bot.jsonl"
    _write_jsonl(out, rows)
    print(f"wrote {out}: {len(rows)} positions")


# ---------------------------------------------------------------------- phase


def load_pool(paths: list[str]) -> tuple[list[dict], int]:
    """Harvested policy episodes that finished alive with a final score."""
    pool, dropped = [], 0
    for path in paths:
        for ep in _read_jsonl(Path(path)):
            if ep.get("score_end") is None or not ep.get("decisions"):
                continue
            if ep.get("eliminated") or ep.get("end_turn") is None:
                dropped += 1
                continue
            pool.append(ep)
    return pool, dropped


def audit(rows: list[dict]) -> list[dict]:
    """Positions labelled 0 whose rendering shows a live civilization."""
    from civmarsh.utils.freeciv import score_from_rendering

    bad = []
    for row in rows:
        now = score_from_rendering(row["rendering"])
        if row["end_score"] == 0 and now is not None and now >= AUDIT_SCORE:
            bad.append({"turn": row["turn"], "end_score": 0, "score_now": now})
    return bad


def cmd_phase(args: argparse.Namespace) -> None:
    from civmarsh.rewards.reference_set import build_reference_set

    data_root = Path(args.data_root)
    bot = Path(args.bot or data_root / "refsets" / "bot.jsonl")
    start = PHASE_START_TURN[args.phase]
    bot_rows = [r for r in _read_jsonl(bot) if r["turn"] >= start]
    if not bot_rows:
        raise SystemExit(f"no bot positions at turn >= {start} in {bot}")
    pool, dropped = load_pool(args.harvest or [])
    rebuilt: list[int] = []
    if pool:
        with tempfile.TemporaryDirectory() as td:
            trimmed = Path(td) / "bot.jsonl"
            _write_jsonl(trimmed, bot_rows)
            rows, rebuilt = build_reference_set(pool, str(trimmed))
        if rows is None:
            raise SystemExit("no bucket had enough harvested policy positions")
    else:
        rows = bot_rows
    out = Path(args.out or data_root / "refsets" / f"{args.phase}.jsonl")
    _write_jsonl(out, rows)
    by_turn: dict[int, list] = {}
    for row in rows:
        by_turn.setdefault(row["turn"], []).append(row["end_score"])
    bad = audit(rows)
    stats = {
        "phase": args.phase,
        "bot_positions": str(bot),
        "harvest": args.harvest or [],
        "pool_episodes": len(pool),
        "pool_dropped_eliminated": dropped,
        "rebuilt_buckets": rebuilt,
        "per_bucket_end_scores": {
            str(t): sorted(v) for t, v in sorted(by_turn.items())
        },
        "zero_label_audit": bad,
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(
        f"{args.phase}: {len(rows)} positions, pool {len(pool)} "
        f"(dropped {dropped}), rebuilt buckets {rebuilt}, audit hits {len(bad)} -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    games = sub.add_parser("games", help="play and save the reference games")
    games.add_argument("--data-root", required=True)
    games.add_argument("--seeds", required=True, help="e.g. 8000-8119")
    games.add_argument("--tiles-per-player", type=int, default=200)
    games.add_argument("--workers", type=int, default=40)
    games.set_defaults(func=cmd_games)

    bot = sub.add_parser("bot", help="render the bot reference positions")
    bot.add_argument("--data-root", required=True)
    bot.add_argument("--horizon", type=int, default=HORIZON)
    bot.add_argument("--per-bucket", type=int, default=PER_BUCKET)
    bot.add_argument(
        "--heldout-mod", type=int, default=5, help="use games with seed %% N == 0"
    )
    bot.add_argument("--workers", type=int, default=16)
    bot.set_defaults(func=cmd_bot)

    phase = sub.add_parser("phase", help="build one phase's reference set")
    phase.add_argument("--data-root", required=True)
    phase.add_argument("--phase", choices=sorted(PHASE_START_TURN), required=True)
    phase.add_argument("--bot", help="bot positions (default DATA/refsets/bot.jsonl)")
    phase.add_argument(
        "--harvest", action="append", help="value_harvest.jsonl of a policy run"
    )
    phase.add_argument("--out", help="output (default DATA/refsets/<phase>.jsonl)")
    phase.set_defaults(func=cmd_phase)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
