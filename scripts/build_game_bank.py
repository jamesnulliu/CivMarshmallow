#!/usr/bin/env python3
"""Build a cross-game replay-oracle bank (civmarsh.traps.games): self-play
games, snapshots at fixed move numbers, a visible proxy and K-replay oracle.

Usage:
  STOCKFISH_BIN=/path/to/stockfish python scripts/build_game_bank.py \
      --game chess --out data/games/chess_bank.jsonl --workers 20
  python scripts/build_game_bank.py --game go9 --out data/games/go9_bank.jsonl

  # more snapshots of new games, e.g. opening and late-game moves; games whose
  # self-play does not reproduce from the seed (chess, catan) need their own
  # index range
  python scripts/build_game_bank.py --game chess --game-offset 1000 \
      --snaps 8,12,80,88,96,104,112,120,128 --out data/games/chess_bank_more.jsonl

Defaults per game (games x replays, seed): chess 80 x 20 (20260909), othello
70 x 24 (20260909), backgammon 40 x 100, go9 / hearts / 2048 60 x 100, catan
40 x 48 (20260911). Requires python-chess and STOCKFISH_BIN (chess),
open_spiel (backgammon, go9, hearts, 2048) or catanatron (catan).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from civmarsh.traps.games import GAMES, SPIEL_GAMES
from civmarsh.utils.io import write_jsonl

T0 = time.time()


def log(msg: str) -> None:
    print(f"[bank +{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", required=True, choices=sorted(GAMES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-games", type=int, help="default: the game's preset")
    ap.add_argument("--game-offset", type=int, default=0, help="first game index")
    ap.add_argument("--replays", type=int, help="oracle replays K per position")
    ap.add_argument("--snaps", help="comma-separated move numbers to snapshot")
    ap.add_argument("--seed", type=int, help="base seed; game i uses seed + i")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()

    kw = {"game_offset": a.game_offset, "workers": a.workers, "log": log}
    if a.n_games is not None:
        kw["n_games"] = a.n_games
    if a.replays is not None:
        kw["replays"] = a.replays
    if a.seed is not None:
        kw["seed"] = a.seed
    if a.snaps:
        kw["snaps"] = tuple(int(x) for x in a.snaps.split(","))

    if a.game == "chess":
        from civmarsh.traps.games import chess_bank

        bank = chess_bank.build_bank(**kw)
    elif a.game == "othello":
        from civmarsh.traps.games import othello_bank

        bank = othello_bank.build_bank(**kw)
    elif a.game == "catan":
        from civmarsh.traps.games import catan_bank

        bank = catan_bank.build_bank(**kw)
    elif a.game in SPIEL_GAMES:
        from civmarsh.traps.games import spiel_bank

        bank = spiel_bank.build_bank(a.game, **kw)
    else:
        raise SystemExit(f"no bank builder for {a.game}")

    write_jsonl(a.out, bank, sort_keys=True)
    log(
        f"wrote {len(bank)} positions ({len({r['game_id'] for r in bank})} games) -> {a.out}"
    )
    if bank:
        log(
            f"proxy {min(r['score_now'] for r in bank):+g} .. "
            f"{max(r['score_now'] for r in bank):+g}; "
            f"oracle {min(r['oracle_mean'] for r in bank):+.2f} .. "
            f"{max(r['oracle_mean'] for r in bank):+.2f}; "
            f"median se {statistics.median(r['oracle_se'] for r in bank):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
