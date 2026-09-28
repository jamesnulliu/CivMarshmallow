#!/usr/bin/env python3
"""Build a Freeciv replay-oracle bank: seeded AI games, positions in a turn
window, and K-branch replay-oracle labels (civmarsh.oracle.bank).

Usage:
  CIVHARNESS_SERVER=/path/to/freeciv-server \
  python scripts/build_freeciv_bank.py --preset short_game \
      --work-dir runs/banks/short_game --out-dir data/banks/short_game --workers 16

  # a custom bank: every preset field can be overridden
  python scripts/build_freeciv_bank.py --ruleset classic --endturn 120 \
      --min-turn 10 --max-turn 65 --seeds 4300-4339 \
      --work-dir runs/banks/custom --out-dir data/banks/custom

Presets: short_game, classic_transfer, civ2civ3_120, classic_120,
game_length_60, game_length_70, game_length_80 (see civmarsh.oracle.bank.PRESETS).

Outputs in --out-dir: games.json, positions.jsonl, labels.jsonl, stats.json.
Game saves live under <work-dir>/pool (record save paths are relative to it);
branch replays run under <work-dir>/branches and are deleted once labeled.
Every stage resumes: finished games are reloaded from their saves, positions
already listed are kept, and labeled positions are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from civmarsh.oracle.bank import (
    PRESETS,
    BankPreset,
    generate_games,
    label_positions,
    select_positions,
)
from civmarsh.utils.io import read_jsonl, write_jsonl


def parse_seeds(text: str) -> list[int]:
    """'4300-4339' or '4300,4301,4400-4409'."""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def resolve_preset(a) -> tuple[str, BankPreset]:
    if a.preset:
        name, preset = a.preset, PRESETS[a.preset]
    else:
        missing = [
            f
            for f in ("endturn", "min_turn", "max_turn", "seeds")
            if getattr(a, f) is None
        ]
        if missing:
            sys.exit(
                f"without --preset, set {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        name, preset = (
            "custom",
            BankPreset(
                ruleset=None,
                endturn=a.endturn,
                min_turn=a.min_turn,
                max_turn=a.max_turn,
                splits={},
            ),
        )
    over = {}
    if a.ruleset is not None:
        over["ruleset"] = None if a.ruleset == "default" else a.ruleset
    for key in ("endturn", "min_turn", "max_turn", "every", "k", "skill"):
        if getattr(a, key) is not None:
            over[key] = getattr(a, key)
    if a.seeds is not None:
        seeds = parse_seeds(a.seeds)
        over["splits"] = {a.split: seeds}
    return name, replace(preset, **over)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument(
        "--ruleset", help="ruleset directory name, or 'default' for the server default"
    )
    ap.add_argument("--endturn", type=int)
    ap.add_argument("--min-turn", type=int)
    ap.add_argument("--max-turn", type=int)
    ap.add_argument("--every", type=int, help="autosave / position interval in turns")
    ap.add_argument("--seeds", help="game seeds, e.g. 4300-4339 or 1,2,10-19")
    ap.add_argument("--split", default="eval", help="split name for --seeds")
    ap.add_argument("--k", type=int, help="oracle branches per position")
    ap.add_argument("--skill", help="AI skill of every player in the oracle branches")
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage", default="all", choices=("games", "labels", "all"))
    a = ap.parse_args()

    name, p = resolve_preset(a)
    work = a.work_dir.resolve()
    pool_dir, branch_dir = work / "pool", work / "branches"
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    games_f, pos_f, lab_f = (
        out / "games.json",
        out / "positions.jsonl",
        out / "labels.jsonl",
    )

    games = json.loads(games_f.read_text()) if games_f.exists() else []
    positions = read_jsonl(pos_f) if pos_f.exists() else []
    have = {g["seed"] for g in games}
    split_of = p.split_of()
    todo = [s for s in p.seeds if s not in have]
    failed_seeds = []
    if todo:
        new = generate_games(
            todo,
            ruleset=p.ruleset,
            endturn=p.endturn,
            every=p.every,
            pool_dir=pool_dir,
            workers=a.workers,
            skill=p.game_skill,
            aifill=p.aifill,
            size=p.size,
        )
        for g in new:
            g["split"] = split_of[g["seed"]]
        failed_seeds = sorted(set(todo) - {g["seed"] for g in new})
        positions += select_positions(
            new,
            min_turn=p.min_turn,
            max_turn=p.max_turn,
            every=p.every,
            pool_dir=pool_dir,
        )
        games = sorted(games + new, key=lambda g: g["seed"])
        positions.sort(key=lambda r: r["position_id"])
        games_f.write_text(json.dumps(games, indent=1) + "\n")
        write_jsonl(pos_f, positions)
    print(f"bank {name}: {len(games)} games, {len(positions)} positions", flush=True)

    stats = {
        "preset": name,
        "protocol": {
            "ruleset": p.ruleset,
            "endturn": p.endturn,
            "every": p.every,
            "turns": [p.min_turn, p.max_turn],
            "k": p.k,
            "oracle_skill": p.skill,
            "until": p.until,
            "game_skill": p.game_skill,
            "aifill": p.aifill,
            "size": p.size,
            "splits": {s: [min(r), max(r)] for s, r in p.splits.items() if len(r)},
        },
        "pool_dir": str(pool_dir),
        "n_games": len(games),
        "n_positions": len(positions),
        "failed_seeds": failed_seeds,
    }
    if a.stage != "games":
        res = label_positions(
            positions,
            pool_dir=pool_dir,
            until=p.until,
            k=p.k,
            skill=p.skill,
            work_dir=branch_dir,
            out_path=lab_f,
            workers=a.workers,
        )
        stats.update({"n_labeled": res["n_labeled"], "label_failures": res["failures"]})
        print(
            f"labels: {res['n_labeled']}/{len(positions)} labeled, "
            f"{len(res['failures'])} failed",
            flush=True,
        )
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
