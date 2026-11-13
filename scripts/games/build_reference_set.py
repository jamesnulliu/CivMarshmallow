#!/usr/bin/env python3
"""Build the reference set of the CivTelescope potential from a reference bank.

Same file shape as the Freeciv reference sets:
  {turn, focal, rendering, end_score, position_id}
with end_score = the replay oracle's mean final score (Othello: final disc
differential; Catan: final victory points), so the potential comes out in
final-score units.  Per snapshot index the bank's positions are sorted by
oracle mean and --per-bucket of them are taken at evenly spaced ranks (lowest
and highest included), so the winrate-to-score interpolation spans the
bucket's whole outcome range.  Deterministic: no sampling.

Othello: only Black-to-move positions are used by default; every position the
policy decides in is Black to move, and the rendering states the side to move.
Catan: every seat's row is a candidate (the rendering names the evaluated
player), and a bucket takes at most one seat per snapshot, so its references
are different game states.

Usage:
  python scripts/games/build_reference_set.py \\
      --bank data/games/othello/bank_reference.jsonl --out data/games/othello/refset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civmarsh.utils.io import read_jsonl


def spread_indices(n, k):
    """k distinct ranks evenly spaced over 0..n-1, ends included."""
    if n < k:
        raise ValueError(f"need at least {k} positions, have {n}")
    if k == 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def distinct_states(rows, per_bucket):
    """--per-bucket rows at evenly spaced outcome ranks, one row per game state:
    a multi-seat bank (Catan) has several rows per (game, turn), and a rank whose
    state is already taken moves to the nearest free rank."""
    picks, used = [], set()
    for rank in spread_indices(len(rows), per_bucket):
        for offset in sorted(range(-len(rows), len(rows)), key=abs):
            j = rank + offset
            if (
                0 <= j < len(rows)
                and (rows[j]["game_id"], rows[j]["turn"]) not in used
                and rows[j] not in picks
            ):
                picks.append(rows[j])
                used.add((rows[j]["game_id"], rows[j]["turn"]))
                break
    return sorted(picks, key=lambda r: (r["oracle_mean"], r["position_id"]))


def build_refset(bank, per_bucket=8, black_only=True, min_spread=0.0):
    by_turn = {}
    for row in bank:
        if black_only and not row.get("black_to_move", True):
            continue
        by_turn.setdefault(row["turn"], []).append(row)
    refset, skipped = [], {}
    for turn in sorted(by_turn):
        rows = sorted(by_turn[turn], key=lambda r: (r["oracle_mean"], r["position_id"]))
        if len({(r["game_id"], r["turn"]) for r in rows}) < per_bucket:
            skipped[turn] = f"{len(rows)} positions"
            continue
        picks = distinct_states(rows, per_bucket)
        if picks[-1]["oracle_mean"] - picks[0]["oracle_mean"] <= min_spread:
            skipped[turn] = "no outcome spread"
            continue
        refset.extend(
            {
                "turn": r["turn"],
                "focal": r["focal_player"],
                "rendering": r["rendering"],
                "end_score": r["oracle_mean"],
                "position_id": r["position_id"],
            }
            for r in picks
        )
    return refset, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument(
        "--all-sides",
        action="store_true",
        help="othello: also use White-to-move positions",
    )
    args = ap.parse_args()

    bank = read_jsonl(args.bank)
    refset, skipped = build_refset(bank, args.per_bucket, black_only=not args.all_sides)
    if not refset:
        raise SystemExit("no bucket has enough positions; grow the reference bank")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in refset:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    turns = sorted({r["turn"] for r in refset})
    print(
        f"{len(refset)} reference positions in {len(turns)} buckets (turns {turns}) -> {out}"
    )
    for turn in turns:
        scores = [r["end_score"] for r in refset if r["turn"] == turn]
        print(f"  turn {turn:3d}: end_score {min(scores):+6.1f} .. {max(scores):+6.1f}")
    for turn, why in skipped.items():
        print(f"  turn {turn:3d}: SKIPPED ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
