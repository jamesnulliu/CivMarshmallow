#!/usr/bin/env python3
"""Generate the bot-played games and build the per-phase start banks.

Usage:
  # 1. whole games, one save per turn (needs CIVHARNESS_SERVER)
  python scripts/build_start_banks.py games --data-root DATA --split train --seeds 8500-8599
  python scripts/build_start_banks.py games --data-root DATA --split test  --seeds 8600-8699
  # 2. start banks for the four phases, from the saves
  python scripts/build_start_banks.py banks --data-root DATA

Games: 5 players (the focal seat and four fill players), hard AI in every seat,
map size 1 with 200 tiles per player, played to turn 120, saved every turn under
DATA/games/<game_id>/saves. ``games`` appends one row per game to
DATA/starts/games_<split>.jsonl (the turn-1 start, its focal player, the AI's own
score at turns 80 and 120) and skips games already listed.

Banks: a phase starts every game at a fixed turn and plays it to turn 120
(rem20: turn 100, rem40: 80, rem80: 40, rem120: 1). A game enters a phase when
its AI game reached turn 120, the snapshot at the start turn exists and the focal
player is alive in it. Written under DATA/starts:

  train_all_<phase>.jsonl  every valid training game; split "heldout" for
                           seed % 5 == 0, "train" otherwise
  train_<phase>.jsonl      the training bank: every second "train" game ranked by
                           the AI's final score (ties by game id), the same games
                           in every phase
  test_all_<phase>.jsonl   every valid test game, split "test", by seed
  test_<phase>.jsonl       the evaluation bank: the first --n-test of those, the
                           same games in every phase

``save_path`` in every row is relative to --data-root.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

ENDTURN = 120
TILES_PER_PLAYER = 200
PHASE_START_TURN = {"rem20": 100, "rem40": 80, "rem80": 40, "rem120": 1}
HELDOUT_MOD = 5


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(tmp, path)


def game_id(seed: int) -> str:
    return f"g{seed}"


# ---------------------------------------------------------------------- games


def play_game(seed: int, data_root: Path) -> dict:
    """One whole bot-played game, saved every turn; returns its manifest row."""
    from civharness import GameConfig, position, snapshot_series
    from civmarsh.utils.freeciv import focal_player

    gid = game_id(seed)
    workdir = (data_root / "games" / gid).resolve()
    config = GameConfig(
        aifill=4,
        skill="hard",
        endturn=ENDTURN,
        mapseed=seed,
        gameseed=seed,
        size=1,
        extra=(f"set tilesperplayer {TILES_PER_PLAYER}",),
    )
    refs = sorted(snapshot_series(config, workdir, every=1), key=lambda r: r.turn)
    if not refs:
        raise RuntimeError(f"{gid}: no snapshots")
    turns = [r.turn for r in refs]
    if turns != list(range(1, turns[-1] + 1)):
        raise RuntimeError(f"{gid}: snapshots are not one per turn")
    final = position(refs[-1].path)
    focal = focal_player(final)
    if focal not in position(refs[0].path).players:
        raise RuntimeError(f"{gid}: focal player absent at the first turn")
    at80 = next((r for r in refs if r.turn == 80), None)
    score80 = None
    if at80 is not None:
        players80 = position(at80.path).players
        score80 = players80[focal]["score"] if focal in players80 else None
    return {
        "position_id": f"ep_{gid}",
        "game_id": gid,
        "seed": seed,
        "save_path": str(refs[0].path.relative_to(data_root.resolve())),
        "turn": refs[0].turn,
        "focal_player": focal,
        "ai_end_score": final.players[focal]["score"],
        "ai_end_turn": final.turn,
        "ai_score_t80": score80,
    }


def cmd_games(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    (data_root / "starts").mkdir(parents=True, exist_ok=True)
    manifest = data_root / "starts" / f"games_{args.split}.jsonl"
    done = {row["game_id"] for row in _read_jsonl(manifest)}
    seeds = [s for s in _seeds(args.seeds) if game_id(s) not in done]
    print(f"{len(seeds)} games queued ({len(done)} already in {manifest})", flush=True)
    failed = []
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        manifest.open("a") as out,
    ):
        futures = {pool.submit(play_game, s, data_root): s for s in seeds}
        for n, future in enumerate(as_completed(futures), 1):
            seed = futures[future]
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception:  # noqa: BLE001 - a failed game is reported, the rest continue
                failed.append(seed)
                print(f"{game_id(seed)} failed\n{traceback.format_exc(limit=2)}")
            if n % 10 == 0:
                print(f"{n}/{len(futures)}", flush=True)
    if failed:
        raise SystemExit(f"failed seeds: {sorted(failed)}")


# ---------------------------------------------------------------------- banks


def _phase_rows(job: tuple[dict, str, str]) -> tuple[str, dict, dict]:
    """The start row of one game in every phase it qualifies for."""
    from civharness import position
    from civharness.branch import turn_of_save

    row, split, data_root = job
    root = Path(data_root)
    gid = row["game_id"]
    out, excluded = {}, {}
    if (row.get("ai_end_turn") or 0) < ENDTURN:
        return gid, out, dict.fromkeys(PHASE_START_TURN, "ai_game_short")
    saves = {}
    for path in (root / "games" / gid / "saves").glob("freeciv-T*"):
        saves.setdefault(turn_of_save(path), path)
    focal = row["focal_player"]
    for phase, turn in PHASE_START_TURN.items():
        path = saves.get(turn)
        if path is None:
            excluded[phase] = "missing_snapshot"
            continue
        start = position(path)
        if start.turn != turn:
            raise RuntimeError(f"{gid}: snapshot T{turn} holds turn {start.turn}")
        if focal not in start.players:
            excluded[phase] = "focal_absent"
            continue
        if split == "test":
            row_split = "test"
        else:
            row_split = "heldout" if row["seed"] % HELDOUT_MOD == 0 else "train"
        record = {
            "position_id": row["position_id"]
            if turn == 1
            else f"{row['position_id']}_T{turn:03d}",
            "game_id": gid,
            "save_path": str(path.resolve().relative_to(root.resolve())),
            "turn": turn,
            "start_turn": turn,
            "focal_player": focal,
            "score_at_start": start.players[focal]["score"],
            "ai_end_score": row["ai_end_score"],
            "ai_end_turn": row["ai_end_turn"],
            "split": row_split,
        }
        if turn == 1:
            record["ai_score_t80"] = row.get("ai_score_t80")
        out[phase] = record
    return gid, out, excluded


def _build_split(data_root: Path, split: str, workers: int) -> dict[str, list[dict]]:
    games = _read_jsonl(data_root / "starts" / f"games_{split}.jsonl")
    if not games:
        raise SystemExit(f"no games_{split}.jsonl under {data_root / 'starts'}")
    per_phase = {phase: [] for phase in PHASE_START_TURN}
    reasons: dict[str, dict[str, int]] = {phase: {} for phase in PHASE_START_TURN}
    jobs = [(row, split, str(data_root)) for row in games]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _gid, rows, excluded in pool.map(_phase_rows, jobs):
            for phase, record in rows.items():
                per_phase[phase].append(record)
            for phase, reason in excluded.items():
                reasons[phase][reason] = reasons[phase].get(reason, 0) + 1
    seed_of = {row["game_id"]: row["seed"] for row in games}
    for phase, rows in per_phase.items():
        rows.sort(key=lambda r: seed_of[r["game_id"]])
        print(f"{split} {phase}: {len(rows)} starts, excluded {reasons[phase]}")
    return per_phase


def _same_games(banks: dict[str, list[dict]], label: str) -> None:
    sets = {frozenset(r["game_id"] for r in rows) for rows in banks.values()}
    if len(sets) != 1:
        raise SystemExit(f"{label}: the phases hold different games")


def cmd_banks(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = data_root / "starts"
    written = {}

    if (out_dir / "games_train.jsonl").exists():
        train = _build_split(data_root, "train", args.workers)
        ranked = sorted(
            (r for r in train["rem20"] if r["split"] == "train"),
            key=lambda r: (r["ai_end_score"], r["game_id"]),
        )
        keep = {r["game_id"] for r in ranked[::2]}
        if args.n_train and len(keep) != args.n_train:
            raise SystemExit(
                f"training bank has {len(keep)} games, expected {args.n_train}"
            )
        banks = {}
        for phase, rows in train.items():
            written[f"train_all_{phase}"] = rows
            banks[phase] = [r for r in rows if r["game_id"] in keep]
            if len(banks[phase]) != len(keep):
                raise SystemExit(f"train {phase}: a training game is missing")
            written[f"train_{phase}"] = banks[phase]
        _same_games(banks, "training bank")

    if (out_dir / "games_test.jsonl").exists():
        test = _build_split(data_root, "test", args.workers)
        _same_games(test, "test games")
        banks = {}
        for phase, rows in test.items():
            if len(rows) < args.n_test:
                raise SystemExit(f"test {phase}: {len(rows)} starts < {args.n_test}")
            written[f"test_all_{phase}"] = rows
            banks[phase] = written[f"test_{phase}"] = rows[: args.n_test]
        _same_games(banks, "evaluation bank")
        train_ids = {
            r["game_id"] for k, v in written.items() if k.startswith("train") for r in v
        }
        if train_ids & {r["game_id"] for r in test["rem120"]}:
            raise SystemExit("test games overlap training games")

    if not written:
        raise SystemExit(f"no games manifests under {out_dir}; run `games` first")
    for name, rows in written.items():
        _write_jsonl(out_dir / f"{name}.jsonl", rows)
        print(f"wrote {out_dir / name}.jsonl: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    games = sub.add_parser("games", help="play and save the bot games")
    games.add_argument("--data-root", required=True)
    games.add_argument("--split", choices=("train", "test"), required=True)
    games.add_argument("--seeds", required=True, help="e.g. 8500-8599 or 1,2,5-9")
    games.add_argument("--workers", type=int, default=40)
    games.set_defaults(func=cmd_games)

    banks = sub.add_parser("banks", help="build the per-phase start banks")
    banks.add_argument("--data-root", required=True)
    banks.add_argument("--workers", type=int, default=16)
    banks.add_argument(
        "--n-train", type=int, default=40, help="expected training games (0: any)"
    )
    banks.add_argument("--n-test", type=int, default=48, help="evaluation games")
    banks.set_defaults(func=cmd_banks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
