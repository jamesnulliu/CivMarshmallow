#!/usr/bin/env python3
"""Scoreboard-trap rate of every cross-game bank, overall and by game progress.

For each bank: decidable pairs under the game's move gap (oracle gap above
twice the pooled SE), trap rate over the pairs whose visible proxy is not
tied, and the same rate per progress bin (mean move of a pair over the game's
typical length; bins with fewer than 30 pairs have no rate).

Usage:
  python scripts/cross_game_trap_rates.py \\
      --bank chess=data/games/chess_bank.jsonl \\
      --bank go9=data/games/go9_bank.jsonl \\
      --progress-bank chess=data/games/chess_bank_more.jsonl \\
      --lengths data/games/lengths.json --out results/cross_game_trap_rates.json

--progress-bank adds snapshots that enter the progress bins only; the overall
rate stays on --bank. --lengths is the output of measure_game_lengths.py
(needed for every game except chess and othello).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civmarsh.traps.games import GAMES
from civmarsh.traps.games.lengths import typical_length
from civmarsh.traps.rates import MIN_BIN_PAIRS, N_BINS, bin_edges, game_trap_rates
from civmarsh.utils.io import read_jsonl


def parse_pairs(specs: list[str] | None) -> dict[str, str]:
    out = {}
    for spec in specs or []:
        name, _, path = spec.partition("=")
        if name not in GAMES:
            raise SystemExit(f"unknown game {name!r}; one of {sorted(GAMES)}")
        out[name] = path
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", action="append", required=True, metavar="GAME=PATH")
    ap.add_argument("--progress-bank", action="append", metavar="GAME=PATH")
    ap.add_argument("--lengths", type=Path, help="measure_game_lengths.py output")
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--min-bin-pairs", type=int, default=MIN_BIN_PAIRS)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    banks = parse_pairs(a.bank)
    extra = parse_pairs(a.progress_bank)
    lengths = json.loads(a.lengths.read_text()) if a.lengths else {}
    rows = []
    for game, path in banks.items():
        bank = read_jsonl(path)
        row = game_trap_rates(
            game,
            bank,
            length=typical_length(game, bank, lengths),
            progress_bank=read_jsonl(extra[game]) if game in extra else None,
            n_bins=a.n_bins,
            min_pairs=a.min_bin_pairs,
        )
        row["source"] = path
        if game in extra:
            row["source_bins"] = [path, extra[game]]
        rows.append(row)
    rows.sort(key=lambda r: (r["trap_rate"] is None, r["trap_rate"]))
    for r in rows:
        cells = " ".join(
            f"{100 * b['trap_rate']:4.0f}" if b["trap_rate"] is not None else "   ."
            for b in r["bins"]
        )
        rate = r["trap_rate"]
        print(
            f"{r['label']:12s} {r['positions']:4d} pos / {r['games']:3d} games, "
            f"untied {r['untied']:6d}, trap "
            f"{100 * rate if rate is not None else float('nan'):5.1f}% | {cells}"
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
