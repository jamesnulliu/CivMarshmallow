#!/usr/bin/env python3
"""Build CivTelescope's training set and held-out slices.

Pools: replay-bank positions (fog renderings joined with replay-oracle labels)
and two rollout slices (see build_rollout_positions.py). Per pool, training
keeps a fixed quota of decidable pairs with at most 40% traps; the held-out
natural slice (600 pairs) samples all decidable pairs of the held-out
positions, the held-out trap slice (600 pairs) samples trap pairs; both are
apportioned across pools like the training quotas. One seeded generator drives
every draw. See civmarsh.civtelescope.data.build_dataset.

Replay renderings may carry a ``split`` column ("train" / "eval"); the same
file can then be passed for both splits. Writes train_pairs.jsonl,
eval_natural.jsonl, eval_trap.jsonl, positions.jsonl and stats.json to --out.

Usage:
  python scripts/build_civtelescope_pairs.py --out data/civtelescope \\
      --replay-train bank/rendered_fog.jsonl --replay-eval bank/rendered_fog.jsonl \\
      --replay-labels bank/labels.jsonl --rollout-dir data/rollout
"""

import argparse
import json

from civmarsh.civtelescope.data import (
    PAIR_SEED,
    build_dataset,
    replay_pool,
    rollout_pool,
    write_dataset,
)
from civmarsh.utils.io import iter_jsonl


def read_labels(path, filters: list[str]) -> dict[str, dict]:
    """Labels by position id; with KEY=VALUE filters, only matching rows."""
    want = [f.split("=", 1) for f in filters]
    out = {}
    for r in iter_jsonl(path):
        if all(str(r.get(k)) == v for k, v in want):
            out[r["position_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay-train", required=True, help="fog renderings (train)")
    ap.add_argument("--replay-eval", required=True, help="fog renderings (held out)")
    ap.add_argument("--replay-labels", required=True, help="replay-oracle labels")
    ap.add_argument(
        "--label-filter",
        action="append",
        default=[],
        help="KEY=VALUE; keep only label rows with this field value",
    )
    ap.add_argument("--rollout-dir", help="<pool>_{train,eval}.jsonl rollout slices")
    for pool in ("rollout-early", "rollout-late"):
        for split in ("train", "eval"):
            ap.add_argument(f"--{pool}-{split}", help=f"{pool} {split} slice file")
    ap.add_argument("--seed", type=int, default=PAIR_SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = read_labels(a.replay_labels, a.label_filter)
    pools, drops = {}, {}
    for split, path in (("train", a.replay_train), ("eval", a.replay_eval)):
        pools[("replay", split)], drops[f"replay_{split}"] = replay_pool(
            iter_jsonl(path), labels, split=split
        )
    for pool in ("rollout_early", "rollout_late"):
        for split in ("train", "eval"):
            path = getattr(a, f"{pool}_{split}")
            if path is None:
                if not a.rollout_dir:
                    ap.error(f"no {pool} {split} slice: pass --rollout-dir or the file")
                path = f"{a.rollout_dir}/{pool}_{split}.jsonl"
            pools[(pool, split)], drops[f"{pool}_{split}"] = rollout_pool(
                iter_jsonl(path)
            )
    ds = build_dataset(pools, seed=a.seed)
    ds.stats["position_drop_counts"] = drops
    ds.stats["file_lines"] = {
        "train_pairs": len(ds.train),
        "eval_natural": len(ds.eval_natural),
        "eval_trap": len(ds.eval_trap),
        "positions": len(ds.positions),
    }
    write_dataset(a.out, ds)
    print(json.dumps(ds.stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
