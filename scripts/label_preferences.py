#!/usr/bin/env python3
"""Label CivTelescope's training pairs with a zero-shot model's preference.

The training pairs are labeled in a seeded permutation (Random(97)), one
presentation order per pair (Random(29)), with the pairwise prompt CivTelescope
is trained on. Successful labels are appended to --out; the run resumes from
it and stops at --max pairs or at --deadline. build_label_source_pairs.py
turns the labels into the two training sets of the label-source ablation.

Usage:
  python scripts/label_preferences.py --data-dir data/civtelescope \\
      --model MODEL --max 1000 --out results/preference_labels.jsonl \\
      [--deadline YYYY-MM-DDTHH:MMZ]
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from civmarsh.civtelescope.ablations import label_preferences, labeling_plan
from civmarsh.civtelescope.data import load_dataset, read_pairs
from civmarsh.utils.api import add_client_args, client_from_args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-pairs", help="pair file (default: the dataset's)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ruleset", default="civ2civ3")
    ap.add_argument("--max", type=int, default=1000, help="label the first N planned")
    ap.add_argument("--deadline", help="UTC time, YYYY-MM-DDTHH:MMZ")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True, help="labels JSONL (appended)")
    add_client_args(ap)
    a = ap.parse_args()

    ds = load_dataset(a.data_dir)
    train = read_pairs(a.train_pairs) if a.train_pairs else ds.train
    plan = labeling_plan(train)[: a.max]
    deadline = None
    if a.deadline:
        deadline = (
            datetime.strptime(a.deadline, "%Y-%m-%dT%H:%MZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    out = Path(a.out)
    client = client_from_args(a, a.model, out.with_suffix(".cache.jsonl"))
    res = label_preferences(
        client,
        plan,
        ds.positions,
        out,
        ruleset=a.ruleset,
        workers=a.workers,
        deadline=deadline,
    )
    print(f"{res}; usage {client.usage_totals()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
