#!/usr/bin/env python3
"""Read out the prompting baselines from their episode logs.

Collects ``<arm>.jsonl`` from every run directory given and its immediate
subdirectories (the shards of scripts/run_prompting_baselines.sh are merged),
keeps one record per (position, sample), and reads each arm out over the valid
episodes: engine errors and INFRA/UNKNOWN seat stalls are excluded, a death in
the game counts with score 0.  Per arm: score (mean over positions of the
per-position mean), eliminated rate, invalid-action rate, calls per decision.

Games can be removed with the excluded-games rule: pass the evaluation cell of
the untrained policy from the same starts (``--base-cell``) and every game in
which it is eliminated in all of its valid episodes is dropped.

Usage:
  python scripts/summarize_prompting_baselines.py runs/prompting_baselines \\
      [--base-cell cells/base_rem120.json] [--out summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

from civmarsh.baselines import driver
from civmarsh.eval.policy_eval import excluded_games

ARMS = ["direct", "baselang", "mastaba", "saga", "reflexion"]


def excluded_positions(base_cell):
    """Start positions of the games removed by the excluded-games rule
    (``civmarsh.eval.policy_eval.excluded_games``) applied to ``base_cell``."""
    games = set(excluded_games(base_cell))
    return sorted(
        pid
        for pid, row in base_cell["per_position"].items()
        if row.get("game_id") in games
    )


def arm_records(run_dirs, arm):
    """Merged records of one arm from the run dirs and their shard subdirs."""
    recs = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        paths = [run_dir / f"{arm}.jsonl", *sorted(run_dir.glob(f"*/{arm}.jsonl"))]
        for path in (p for p in paths if p.is_file()):
            for line in path.read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:  # truncated tail of a killed run
                    continue
    return driver.final_records(recs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", help="output dirs of the baseline runs")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n-samples", type=int, default=8, help="games per position")
    ap.add_argument(
        "--base-cell",
        default=None,
        help="evaluation cell of the untrained policy; applies the excluded-games rule",
    )
    ap.add_argument(
        "--exclude", default="", help="comma-separated position ids to drop"
    )
    ap.add_argument("--out", default=None, help="write the JSON here as well")
    args = ap.parse_args()

    excluded = {p for p in args.exclude.split(",") if p}
    if args.base_cell:
        excluded |= set(
            excluded_positions(json.loads(Path(args.base_cell).read_text()))
        )

    per_arm = {
        arm: arm_records(args.run_dirs, arm) for arm in args.arms.split(",") if arm
    }
    all_positions = sorted(
        {r.get("position_id") for recs in per_arm.values() for r in recs} - {None}
    )
    positions = [p for p in all_positions if p not in excluded]

    arms = {}
    for arm, recs in per_arm.items():
        s = driver.summarize_arm(recs, positions=positions)
        s["n_expected"] = args.n_samples * len(positions)
        s["complete"] = (
            s["n_positions"] == len(positions) and s["n_episodes"] >= s["n_expected"]
        )
        arms[arm] = s

    out = {
        "positions": positions,
        "excluded": sorted(excluded),
        "n_samples": args.n_samples,
        "arms": arms,
    }
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
