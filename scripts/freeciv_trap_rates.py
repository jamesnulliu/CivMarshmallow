#!/usr/bin/env python3
"""Scoreboard-trap rate of labeled Freeciv banks, overall and by game progress.

Pairs: positions of different games at most ten turns apart whose oracle
values (mean terminal score) differ by more than twice the pooled SE; the
trap rate is taken over the pairs whose visible scores differ. Progress is the
mean turn of a pair over the game length (default: the bank's oracle end
turn); a progress bin with fewer than 30 pairs has no rate.

Usage:
  python scripts/freeciv_trap_rates.py \\
      --labels freeciv_civ2civ3=data/banks/civ2civ3_120/labels.jsonl \\
      --labels freeciv_classic=data/banks/classic_120/labels.jsonl \\
      --out results/freeciv_trap_rates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civmarsh.traps.rates import (
    FREECIV_MAX_TURN_GAP,
    MIN_BIN_PAIRS,
    N_BINS,
    bin_edges,
    freeciv_trap_rates,
)
from civmarsh.utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--labels",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a bank's labels.jsonl (repeatable)",
    )
    ap.add_argument("--length", type=float, help="game length (default: label 'until')")
    ap.add_argument("--max-turn-gap", type=int, default=FREECIV_MAX_TURN_GAP)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--min-bin-pairs", type=int, default=MIN_BIN_PAIRS)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows = []
    for spec in a.labels:
        name, _, path = spec.partition("=")
        labels = read_jsonl(path)
        length = a.length or max(r["until"] for r in labels)
        row = freeciv_trap_rates(
            labels,
            length=length,
            max_turn_gap=a.max_turn_gap,
            n_bins=a.n_bins,
            min_pairs=a.min_bin_pairs,
        )
        rows.append({"setting": name, "source": path, **row})
        cells = " ".join(
            f"{100 * b['trap_rate']:4.0f}" if b["trap_rate"] is not None else "   ."
            for b in row["bins"]
        )
        rate = row["trap_rate"]
        print(
            f"{name:20s} trap {100 * rate if rate is not None else float('nan'):5.1f}% "
            f"untied {row['untied']:6d} | {cells}"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps({"bin_edges": bin_edges(a.n_bins), "settings": rows}, indent=2)
        + "\n"
    )
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
