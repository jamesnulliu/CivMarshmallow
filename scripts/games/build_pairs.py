#!/usr/bin/env python3
"""Build CivTelescope's training and held-out pairs for Othello or Catan.

Pairs follow the cross-game pair rule (``civmarsh.oracle.pairs``): positions
from different games, a bounded turn gap, decidable when the oracle gap exceeds
twice the pooled standard error; a pair is a trap when the visible score ranks
it against the oracle.  The turn gap is the bank's own (``meta.max_turn_gap``,
20 for Catan) or 8 (Othello) unless --max-turn-gap is given.  Pairs with a tied
visible score are left out of every slice (the scoreboard cannot pick a side).

Every pair of positions is enumerated, so each bank is first cut to
--max-positions rows (seeded sample; 2000 rows = two million candidate pairs).

Writes a dataset directory in the layout of ``civmarsh.civtelescope.data``
(read by scripts/games/train_civtelescope.py):
  train_pairs.jsonl    --n-train pairs from the train bank, --trap-frac of them
                       trap pairs (seeded)
  eval_trap.jsonl      --n-eval trap pairs from the held-out bank
  eval_natural.jsonl   --n-eval non-trap pairs from the held-out bank
  positions.jsonl      position_id, focal_player, rendering of both banks
  stats.json           counts and trap rates

Usage:
  python scripts/games/build_pairs.py --train-bank data/games/othello/bank_train.jsonl \\
      --heldout-bank data/games/othello/bank_heldout.jsonl --out-dir data/games/othello/civtelescope
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from civmarsh.civtelescope.data import (
    EVAL_NATURAL,
    EVAL_TRAP,
    POSITIONS,
    STATS,
    TRAIN_PAIRS,
)
from civmarsh.oracle.pairs import decidable_pairs
from civmarsh.utils.io import read_jsonl, write_jsonl

DEFAULT_TURN_GAP = 8


def pairs_of(bank: list[dict], max_turn_gap: int) -> list[dict]:
    """Decidable pairs of one bank in the CivTelescope dataset schema."""
    out = []
    for p in decidable_pairs(
        bank,
        max_turn_gap=max_turn_gap,
        value_key="oracle_mean",
        se_key="oracle_se",
        drop_ties=False,
    ):
        out.append(
            {
                "a": p["a"],
                "b": p["b"],
                "pool": "replay",
                "game_a": p["game_a"],
                "game_b": p["game_b"],
                "turn_a": p["turn_a"],
                "turn_b": p["turn_b"],
                "oracle_winner": p["winner"],
                "margin": p["margin"],
                "pooled_se": p["pooled_se"],
                "margin_z": p["margin_z"],
                "score_a": p["score_a"],
                "score_b": p["score_b"],
                "score_tied": p["score_tied"],
                "trap": p["is_trap"],
            }
        )
    return out


def split_untied(pairs):
    """(trap pairs, non-trap pairs) among pairs whose visible score is not tied."""
    untied = [p for p in pairs if not p["score_tied"]]
    return [p for p in untied if p["trap"]], [p for p in untied if not p["trap"]]


def take(rows, n, rng):
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:n]


def bank_turn_gap(bank, default=DEFAULT_TURN_GAP):
    """The bank's own pairing gap (cross-game banks carry it in ``meta``)."""
    gaps = {(r.get("meta") or {}).get("max_turn_gap") for r in bank} - {None}
    if len(gaps) > 1:
        raise ValueError(f"bank mixes pairing gaps {sorted(gaps)}")
    return gaps.pop() if gaps else default


def cut(bank, max_positions, rng):
    if not max_positions or len(bank) <= max_positions:
        return bank
    keep = set(rng.sample(range(len(bank)), max_positions))
    return [row for i, row in enumerate(bank) if i in keep]


def build_all(
    train_bank,
    heldout_bank,
    *,
    max_turn_gap=None,
    n_train=7200,
    trap_frac=0.5,
    n_eval=600,
    seed=47,
    max_positions=2000,
):
    """(train pairs, held-out trap pairs, held-out non-trap pairs, stats)."""
    train_games = {r["game_id"] for r in train_bank}
    heldout_games = {r["game_id"] for r in heldout_bank}
    if train_games & heldout_games:
        raise ValueError(
            f"banks share games: {sorted(train_games & heldout_games)[:5]}"
        )
    if max_turn_gap is None:
        max_turn_gap = bank_turn_gap(train_bank + heldout_bank)
    rng = random.Random(seed)
    n_train_bank, n_heldout_bank = len(train_bank), len(heldout_bank)
    train_bank = cut(train_bank, max_positions, rng)
    heldout_bank = cut(heldout_bank, max_positions, rng)
    t_trap, t_nat = split_untied(pairs_of(train_bank, max_turn_gap))
    h_trap, h_nat = split_untied(pairs_of(heldout_bank, max_turn_gap))
    n_trap = round(n_train * trap_frac)
    train = take(t_trap, n_trap, rng) + take(t_nat, n_train - n_trap, rng)
    rng.shuffle(train)
    eval_trap, eval_nat = take(h_trap, n_eval, rng), take(h_nat, n_eval, rng)

    def bank_stats(n_rows, paired, games, trap, nat):
        return {
            "positions": n_rows,
            "positions_paired": len(paired),
            "games": len(games),
            "untied_decidable_pairs": len(trap) + len(nat),
            "trap_rate": round(len(trap) / max(len(trap) + len(nat), 1), 4),
        }

    stats = {
        "seed": seed,
        "max_turn_gap": max_turn_gap,
        "max_positions": max_positions,
        "train_bank": bank_stats(n_train_bank, train_bank, train_games, t_trap, t_nat),
        "heldout_bank": bank_stats(
            n_heldout_bank, heldout_bank, heldout_games, h_trap, h_nat
        ),
        "train_pairs": len(train),
        "train_trap": sum(p["trap"] for p in train),
        "eval_trap": len(eval_trap),
        "eval_natural": len(eval_nat),
    }
    return train, eval_trap, eval_nat, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-bank", required=True)
    ap.add_argument("--heldout-bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--max-turn-gap", type=int, help="default: the bank's meta.max_turn_gap, else 8"
    )
    ap.add_argument(
        "--max-positions",
        type=int,
        default=2000,
        help="seeded cut of each bank before pairing (0 = no cut)",
    )
    ap.add_argument("--n-train", type=int, default=7200)
    ap.add_argument("--trap-frac", type=float, default=0.5)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()

    train_bank, heldout_bank = (
        read_jsonl(args.train_bank),
        read_jsonl(args.heldout_bank),
    )
    train, eval_trap, eval_nat, stats = build_all(
        train_bank,
        heldout_bank,
        max_turn_gap=args.max_turn_gap,
        n_train=args.n_train,
        trap_frac=args.trap_frac,
        n_eval=args.n_eval,
        seed=args.seed,
        max_positions=args.max_positions,
    )
    out = Path(args.out_dir)
    write_jsonl(out / TRAIN_PAIRS, train, sort_keys=True)
    write_jsonl(out / EVAL_TRAP, eval_trap, sort_keys=True)
    write_jsonl(out / EVAL_NATURAL, eval_nat, sort_keys=True)
    write_jsonl(
        out / POSITIONS,
        (
            {
                "position_id": r["position_id"],
                "focal_player": r["focal_player"],
                "rendering": r["rendering"],
            }
            for r in train_bank + heldout_bank
        ),
    )
    (out / STATS).write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    for name, want, got in (
        ("train", args.n_train, len(train)),
        ("eval_trap", args.n_eval, len(eval_trap)),
        ("eval_natural", args.n_eval, len(eval_nat)),
    ):
        if got < want:
            print(
                f"WARNING: {name} has {got} pairs, fewer than the {want} asked for "
                "(bank too small); grow the bank before training",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
