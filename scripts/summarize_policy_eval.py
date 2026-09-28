#!/usr/bin/env python3
"""Validate policy evaluation runs and aggregate them into S / G / W.

Usage:
  # one evaluation run directory (scripts/eval_policy.sh) -> cell json
  python scripts/summarize_policy_eval.py cell --run-dir RUN --starts test_rem80.jsonl \
      --out RUN/cell.json [--require-split test]

  # many cells -> aggregate json on stdout (or --out)
  python scripts/summarize_policy_eval.py summary --cells cells.json \
      [--base-cell base_rem120.json] [--out summary.json]

``cells.json`` maps ``{reward: {seed: {phase: {"start": cell.json, "end":
cell.json}}}}``; relative paths resolve against the mapping file's directory.
The start cell of a phase is the checkpoint the phase began from (the base model
for rem20, the previous phase's final checkpoint otherwise), evaluated on the
same starts as the end cell. With ``--base-cell`` (the untrained base model on
the whole-game starts), games on which that policy is eliminated in every valid
episode are dropped from every phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from civmarsh.eval import policy_eval


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(obj: dict, out: str | None) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
    else:
        sys.stdout.write(text)


def cmd_cell(args: argparse.Namespace) -> None:
    cell = policy_eval.validate_run(
        args.run_dir,
        args.starts,
        n_samples=args.n_samples,
        min_valid_episodes=args.min_valid_episodes,
        require_split=args.require_split,
    )
    _write(cell, args.out)
    print(
        f"{args.run_dir}: {cell['n_positions']} positions, "
        f"{cell['n_valid_episodes']} valid episodes, "
        f"{cell['n_infra_failures']} infrastructure failures",
        file=sys.stderr,
    )


def cmd_summary(args: argparse.Namespace) -> None:
    root = Path(args.cells).resolve().parent
    mapping = _load(Path(args.cells))
    cells = {
        reward: {
            str(seed): {
                phase: {side: _load(root / path) for side, path in pair.items()}
                for phase, pair in by_phase.items()
            }
            for seed, by_phase in by_seed.items()
        }
        for reward, by_seed in mapping.items()
    }
    exclude = []
    if args.base_cell:
        exclude = policy_eval.excluded_games(_load(Path(args.base_cell)))
    _write(policy_eval.summarize(cells, exclude), args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    cell = sub.add_parser("cell", help="validate one evaluation run directory")
    cell.add_argument("--run-dir", required=True)
    cell.add_argument("--starts", required=True, help="frozen evaluation bank (jsonl)")
    cell.add_argument("--out", help="cell json path (default: stdout)")
    cell.add_argument("--n-samples", type=int, default=policy_eval.N_SAMPLES)
    cell.add_argument(
        "--min-valid-episodes",
        type=int,
        help="valid-episode floor for the cell "
        f"(default {policy_eval.MIN_VALID_PER_POSITION} per position)",
    )
    cell.add_argument("--require-split", help="every start row must carry this split")
    cell.set_defaults(func=cmd_cell)

    summary = sub.add_parser("summary", help="aggregate cells into S / G / W")
    summary.add_argument(
        "--cells", required=True, help="reward/seed/phase mapping json"
    )
    summary.add_argument("--base-cell", help="untrained base model on rem120 starts")
    summary.add_argument("--out", help="summary json path (default: stdout)")
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
