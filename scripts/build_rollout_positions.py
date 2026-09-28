#!/usr/bin/env python3
"""Cut rollout position slices for CivTelescope's training data from RL runs.

Each run's per-episode decision log (one JSONL line per episode, in rollout
order: ``{"decisions": [{"turn", "focal_player", "rendering"}], "score_end"}``)
is split by training step into named windows; every 5th episode is held out.
Writes ``<slice>_train.jsonl`` and ``<slice>_eval.jsonl`` under --out-dir, runs
appended in the order given (the order fixes the pair enumeration downstream).
A step inside two windows goes to the one listed first.

Usage:
  python scripts/build_rollout_positions.py --out-dir data/rollout \\
      --run runA=runs/runA/value_log.jsonl --run runB=runs/runB/value_log.jsonl \\
      --window runA:rollout_early:8-20 --window runA:rollout_late:35-39 \\
      --window runB:rollout_early:12-24 --window runB:rollout_late:35-39
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope.data import rollout_positions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=PATH")
    ap.add_argument(
        "--window",
        action="append",
        required=True,
        help="RUN:SLICE:FIRST-LAST (training steps, inclusive)",
    )
    ap.add_argument("--episodes-per-step", type=int, default=64)
    ap.add_argument("--heldout-every", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    windows: dict[str, dict[str, range]] = {}
    for spec in a.window:
        run, slice_name, span = spec.split(":")
        first, last = map(int, span.split("-"))
        windows.setdefault(run, {})[slice_name] = range(first, last + 1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    files = {}
    try:
        for spec in a.run:
            run, path = spec.split("=", 1)
            slices = rollout_positions(
                path,
                run=run,
                windows=windows.get(run, {}),
                episodes_per_step=a.episodes_per_step,
                heldout_every=a.heldout_every,
            )
            for (slice_name, split), rows in slices.items():
                key = f"{slice_name}_{split}"
                if key not in files:
                    files[key] = open(out_dir / f"{key}.jsonl", "w")  # noqa: SIM115
                for r in rows:
                    files[key].write(json.dumps(r) + "\n")
                counts[key] = counts.get(key, 0) + len(rows)
    finally:
        for f in files.values():
            f.close()
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
