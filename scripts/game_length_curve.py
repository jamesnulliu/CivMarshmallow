#!/usr/bin/env python3
"""Game-length curve (civmarsh.traps.game_length): per game length, how well the
scoreboard and the replay oracle correlate with the real game's terminal score,
the turns the oracle wins, and the trap rate.

Inputs are the labels of the game-length banks built by build_freeciv_bank.py
(presets game_length_60, game_length_70, game_length_80).

Usage:
  python scripts/game_length_curve.py \\
      --labels 60=data/banks/game_length_60/labels.jsonl \\
      --labels 70=data/banks/game_length_70/labels.jsonl \\
      --labels 80=data/banks/game_length_80/labels.jsonl \\
      --out results/game_length_curve.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civmarsh.traps.game_length import (
    MAX_TURN_GAP,
    TURN_WIN_MARGIN,
    game_length_summary,
)
from civmarsh.utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--labels",
        action="append",
        required=True,
        metavar="LENGTH=PATH",
        help="game length (end turn) and its bank's labels.jsonl (repeatable)",
    )
    ap.add_argument("--margin", type=float, default=TURN_WIN_MARGIN)
    ap.add_argument("--max-turn-gap", type=int, default=MAX_TURN_GAP)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows = []
    for spec in a.labels:
        length, _, path = spec.partition("=")
        s = game_length_summary(
            read_jsonl(path), margin=a.margin, max_turn_gap=a.max_turn_gap
        )
        rows.append({"game_length": int(length), "source": path, **s})
        print(
            f"{length} turns: corr scoreboard {s['corr_scoreboard']:.2f}, "
            f"oracle {s['corr_oracle']:.2f}, turns won by oracle "
            f"{s['turns_won_by_oracle']}/{s['n_turns']}, trap rate {s['trap_rate']}"
        )
    rows.sort(key=lambda r: r["game_length"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"game_lengths": rows}, indent=2) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
