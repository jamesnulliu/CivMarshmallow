#!/usr/bin/env python3
"""Build the start positions of the Othello or Catan policy runs.

othello  Each start is the standard opening after k uniformly random legal
         plies (k drawn in 4..9, then one more ply if White would be to move),
         so the policy meets many different early positions rather than one
         opening.  A start is kept only if Black is to move, both sides still
         have a legal move, and its board was not seen before.
catan    Each start is a fresh game (random board, random seating) before the
         first placement; the focal seat rotates 0-3, so the policy plays every
         position in the initial-placement order.  The row carries the full
         save: a Catan position is not reproducible from a seed across engine
         versions, and the sibling episodes of a start must load the same state.

Splits (starts, seed base): train (40), eval (45) and val (40, for choosing a
configuration without touching eval).  They use disjoint seed ranges and are
built in that order with a shared seen-set, so no split repeats a position of
an earlier one.  A row is a harness save (civharness.games) plus position_id,
split and seed.  Deterministic: rerunning reproduces the files byte for byte
(a Catan save embeds a pickle whose bytes follow the hash seed, so for Catan
the script re-executes itself with PYTHONHASHSEED=0).

Usage:
  python scripts/games/build_starts.py --game othello   # data/games/othello/starts/{train,eval,val}.jsonl
  python scripts/games/build_starts.py --game catan
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

SPLITS = {
    "othello": {
        "train": (40, 41_000_000),
        "eval": (45, 42_000_000),
        "val": (40, 43_000_000),
    },
    "catan": {
        "train": (40, 61_000_000),
        "eval": (45, 62_000_000),
        "val": (40, 63_000_000),
    },
}
ORDER = ("train", "eval", "val")


def othello_start(seed):
    """Board after k random plies, or None if the line ended up unusable."""
    from civharness.games import othello_engine as oth

    rng = random.Random(seed)
    black, white, black_to_move, move_no = oth.START_BLACK, oth.START_WHITE, True, 0
    target = rng.randint(4, 9)
    while move_no < target or not black_to_move:
        cur, other = (black, white) if black_to_move else (white, black)
        legal = list(oth.bits(oth.moves(cur, other)))
        if not legal:
            if not oth.moves(other, cur):
                return None
            black_to_move = not black_to_move
            continue
        cur, other = oth.apply_move(cur, other, rng.choice(legal))
        black, white = (cur, other) if black_to_move else (other, cur)
        black_to_move = not black_to_move
        move_no += 1
    if not oth.moves(black, white) or not oth.moves(white, black):
        return None
    return black, white, move_no


def build_othello(split, seen):
    from civharness.games import othello_engine as oth

    n, seed_base = SPLITS["othello"][split]
    rows, seed = [], seed_base
    while len(rows) < n:
        start = othello_start(seed)
        seed += 1
        if start is None or start[:2] in seen:
            continue
        seen.add(start[:2])
        black, white, move_no = start
        rows.append(
            {
                "position_id": f"oth_{split}_{len(rows):03d}",
                "game": "othello",
                "board_black": hex(black),
                "board_white": hex(white),
                "black_to_move": True,
                "move_no": move_no,
                "score_at_start": oth.popcount(black) - oth.popcount(white),
                "split": split,
                "seed": seed - 1,
            }
        )
    return rows


def build_catan(split, seen):
    from civharness.games import make_env

    n, seed_base = SPLITS["catan"][split]
    env = make_env("catan")
    rows = []
    for i in range(n):
        seed = seed_base + i
        env.new_game(random.Random(seed))
        env.focal_seat = i % 4
        save = env.save()
        if save["state"] in seen:
            raise SystemExit(f"catan {split} start {i} repeats an earlier position")
        seen.add(save["state"])
        rows.append(
            {
                "position_id": f"catan_{split}_{i:03d}",
                "game": "catan",
                "focal_seat": save["focal_seat"],
                "focal_player": save["focal_player"],
                "turn": 0,
                "score_at_start": 0,
                "split": split,
                "seed": seed,
                "save": save,
            }
        )
    return rows


BUILDERS = {"othello": build_othello, "catan": build_catan}


def build_splits(game, splits=ORDER):
    """{split: rows} for the requested splits; earlier splits are always built
    so a later split shares no position with them."""
    seen, out = set(), {}
    last = max(ORDER.index(s) for s in splits)
    for split in ORDER[: last + 1]:
        rows = BUILDERS[game](split, seen)
        if split in splits:
            out[split] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", choices=sorted(SPLITS), default="othello")
    ap.add_argument("--out-dir", help="default data/games/<game>/starts")
    ap.add_argument("--splits", default="train,eval,val", help="splits to write")
    args = ap.parse_args()
    if args.game == "catan" and os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(
            sys.executable,
            [sys.executable, *sys.argv],
            {**os.environ, "PYTHONHASHSEED": "0"},
        )
    splits = tuple(s for s in args.splits.split(",") if s)
    unknown = set(splits) - set(ORDER)
    if unknown:
        ap.error(f"unknown splits {sorted(unknown)}")
    out_dir = Path(args.out_dir or f"data/games/{args.game}/starts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in build_splits(args.game, splits).items():
        out = out_dir / f"{split}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        where = sorted({r.get("move_no", r.get("turn")) for r in rows})
        print(
            f"{len(rows)} {split} starts (progress {where}) -> {out} "
            f"({out.stat().st_size // 1024} KB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
