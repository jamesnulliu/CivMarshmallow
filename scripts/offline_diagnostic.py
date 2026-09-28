#!/usr/bin/env python3
"""Offline diagnostics of the value signals and training scores of RL runs.

Usage:
  # per-turn-bucket error and rank correlation of per-decision values against
  # the realized terminal score (value_log.jsonl of rem120 runs)
  python scripts/offline_diagnostic.py spectrum \
      --run civtelescope=runs/civtelescope/seed43/rem120 \
      --run scoreboard=runs/scoreboard/seed43/rem120

  # explained variance of the terminal score beyond the current score
  python scripts/offline_diagnostic.py delta-r2 --run civtelescope=RUN_DIR

  # training-score slope per iteration (averaged over starts); with two or more
  # runs, also the paired per-start slope difference of every pair
  python scripts/offline_diagnostic.py slope \
      --run terminal_civtelescope=RUN_A --run sparse=RUN_B

Each ``--run`` is ``NAME=RUN_DIR``; results are printed as JSON (or written to
``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from civmarsh.eval import diagnostics as dg


def _runs(values: list[str]) -> dict[str, Path]:
    runs = {}
    for item in values:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--run expects NAME=RUN_DIR, got {item!r}")
        runs[name] = Path(path)
    return runs


def cmd_spectrum(runs: dict[str, Path]) -> dict:
    return {
        name: dg.turn_spectrum(dg.read_value_log(run / "value_log.jsonl"))
        for name, run in runs.items()
    }


def cmd_delta_r2(runs: dict[str, Path]) -> dict:
    out = {}
    for name, run in runs.items():
        triplets, counts = dg.value_triplets(
            run / "value_log.jsonl", run / "value_harvest.jsonl"
        )
        out[name] = {**dg.delta_r2(triplets), "counts": counts}
    return out


def cmd_slope(runs: dict[str, Path], n_iterations: int, groups: int) -> dict:
    slopes = {
        name: dg.training_slope(
            run / "reward_log.jsonl",
            n_iterations=n_iterations,
            groups_per_iteration=groups,
        )
        for name, run in runs.items()
    }
    paired = {
        f"{a}_minus_{b}": dg.paired_slope_difference(
            slopes[a]["per_start"], slopes[b]["per_start"]
        )
        for a in slopes
        for b in slopes
        if a != b
    }
    return {"runs": slopes, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("spectrum", "delta-r2", "slope"))
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--out", help="output json (default: stdout)")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=dg.N_ITERATIONS,
        help="training iterations per run (slope)",
    )
    parser.add_argument(
        "--groups-per-iteration",
        type=int,
        default=dg.GROUPS_PER_ITERATION,
        help="start groups consumed per iteration, the rollout batch size (slope)",
    )
    args = parser.parse_args()
    runs = _runs(args.run)
    if args.command == "spectrum":
        result = cmd_spectrum(runs)
    elif args.command == "delta-r2":
        result = cmd_delta_r2(runs)
    else:
        result = cmd_slope(runs, args.n_iterations, args.groups_per_iteration)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
