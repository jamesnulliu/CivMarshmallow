#!/usr/bin/env python3
"""Typical full-game length of each cross-game engine, the denominator of game
progress: complete self-play games per game with the bank's own engine, on a
seed range disjoint from the banks (civmarsh.traps.games.lengths).

Chess (median final ply of its bank) and Othello (60 moves) need no run.

Usage:
  python scripts/measure_game_lengths.py --out data/games/lengths.json
  python scripts/measure_game_lengths.py --games go9,catan --n-games 30 --out lengths.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civmarsh.traps.games.lengths import (
    LENGTH_SEED,
    MEASURED_GAMES,
    N_LENGTH_GAMES,
    measure_lengths,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", default=",".join(MEASURED_GAMES))
    ap.add_argument("--n-games", type=int, default=N_LENGTH_GAMES)
    ap.add_argument("--seed", type=int, default=LENGTH_SEED)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    games = [g for g in a.games.split(",") if g]
    out = measure_lengths(games, n_games=a.n_games, seed=a.seed, workers=a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
