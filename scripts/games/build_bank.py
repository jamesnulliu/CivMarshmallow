#!/usr/bin/env python3
"""Build a replay-oracle position bank for Othello or Catan through the harness.

``civharness.games.bank_rows``: one engine game per seed, a save at every
snapshot index, and K re-seeded engine continuations of each save to the end
of the game (``branch``) as the oracle.

  othello  the generator of the Othello trap bank, row for row
           (civmarsh.traps.games.othello_bank): K stochastic depth-2 replays,
           mean final disc differential for Black
  catan    the construction of the Catan trap bank: four VictoryPointPlayer
           bots, one row per seat per snapshot, the four seats sharing the K
           continuations; mean final victory points.  The row's rendering is
           the player-visible one (board included)

Three disjoint banks per game, by role (seed range / game-id range):

  role       othello seeds / ids      catan seeds / ids       use
  train      31000000 / 10000         51000000 / 10000        CivTelescope training pairs
  heldout    32000000 / 20000         52000000 / 20000        held-out trap / non-trap pairs
  reference  33000000 / 30000         53000000 / 30000        reference set of the potential

Default snapshots cover the whole range the policy decides in:
  othello  every 4th ply from 6 to 54
  catan    turn 10, then every 20 turns from 20 to 240 (a game that ends
           earlier has fewer snapshots)

Usage:
  python scripts/games/build_bank.py --game othello --role train \\
      --games 300 --replays 16 --workers 32 --out data/games/othello/bank_train.jsonl

CPU only.  Catan rows carry the position's save only with --keep-saves
(about 6 KB each; CivTelescope and the reference set need only the rendering).
"""

from __future__ import annotations

import argparse
import statistics
import time

from civharness.games import bank_rows
from civmarsh.utils.io import write_jsonl

ROLES = {
    "othello": {
        "train": (31_000_000, 10_000),
        "heldout": (32_000_000, 20_000),
        "reference": (33_000_000, 30_000),
    },
    "catan": {
        "train": (51_000_000, 10_000),
        "heldout": (52_000_000, 20_000),
        "reference": (53_000_000, 30_000),
    },
}
DEFAULT_SNAPS = {
    "othello": "6,10,14,18,22,26,30,34,38,42,46,50,54",
    "catan": "10,20,40,60,80,100,120,140,160,180,200,220,240",
}
DEFAULT_REPLAYS = {"othello": 24, "catan": 48}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[build_bank +{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", choices=sorted(ROLES), default="othello")
    ap.add_argument(
        "--role",
        choices=sorted(ROLES["othello"]),
        help="sets --game-seed-base and --id-base to the role's disjoint range",
    )
    ap.add_argument("--game-seed-base", type=int)
    ap.add_argument("--id-base", type=int)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument(
        "--replays",
        type=int,
        help="K continuations per snapshot (default 24 for othello, 48 for catan)",
    )
    ap.add_argument("--snaps", help="comma-separated progress indices to snapshot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-depth", type=int, default=2, help="othello")
    ap.add_argument("--gen-temp", type=float, default=40.0, help="othello")
    ap.add_argument("--replay-depth", type=int, default=2, help="othello")
    ap.add_argument("--replay-temp", type=float, default=20.0, help="othello")
    ap.add_argument(
        "--replay-exact-from",
        type=int,
        default=10,
        help="othello: the replay engine solves the endgame exactly from this many "
        "empty squares (0 = never); match the opponent the policy will meet",
    )
    ap.add_argument(
        "--keep-saves", action="store_true", help="catan: store each row's save"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_base, id_base = ROLES[args.game][args.role] if args.role else (None, None)
    seed_base = args.game_seed_base if args.game_seed_base is not None else seed_base
    id_base = args.id_base if args.id_base is not None else id_base
    if seed_base is None or id_base is None:
        ap.error("give --role, or both --game-seed-base and --id-base")
    if args.games > 9_000:
        ap.error("at most 9000 games per bank (the role ranges are 10000 ids apart)")
    replays = args.replays or DEFAULT_REPLAYS[args.game]
    snaps = [int(x) for x in (args.snaps or DEFAULT_SNAPS[args.game]).split(",")]
    if args.game == "othello":
        cfg = {
            "gen_depth": args.gen_depth,
            "gen_temp": args.gen_temp,
            "opp_depth": args.replay_depth,
            "opp_temp": args.replay_temp,
            "opp_exact_from": args.replay_exact_from,
        }
    else:
        cfg = {"keep_saves": args.keep_saves}

    log(
        f"{args.game}: {args.games} games (seeds {seed_base}.., ids {id_base}..) x "
        f"{len(snaps)} snapshots x {replays} replays, {args.workers} workers"
    )

    def progress(done, total, n_rows):
        if done % 20 == 0 or done == total:
            log(f"  {done}/{total} games, {n_rows} positions")

    bank = bank_rows(
        args.game,
        [seed_base + i for i in range(args.games)],
        snaps,
        replays,
        index_offset=id_base,
        workers=args.workers,
        on_game=progress,
        **cfg,
    )
    out = write_jsonl(args.out, bank, sort_keys=True)
    log(f"wrote {len(bank)} positions -> {out}")
    if bank:
        log(
            f"oracle mean spread {min(r['oracle_mean'] for r in bank):+.1f}"
            f" .. {max(r['oracle_mean'] for r in bank):+.1f}; "
            f"median se {statistics.median(r['oracle_se'] for r in bank):.2f}; "
            f"turns {sorted({r['turn'] for r in bank})}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
