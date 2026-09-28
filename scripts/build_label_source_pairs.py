#!/usr/bin/env python3
"""Build the two training sets of the label-source ablation.

From the preference labels of label_preferences.py: the first --n labeled
pairs in the labeling permutation, once with the replay-oracle winner
(replay_labeled.jsonl) and once with the model's preferred side as winner
(preference_labeled.jsonl). Train each with train_civtelescope.py
--train-pairs on the same dataset directory; --eval-interval equal to
ceil(2 * pairs / (batch * accum)) evaluates once per nominal epoch. Score the
snapshots with eval_civtelescope.py --data-dir; --cluster game-prefix clusters
the intervals by policy run for rollout positions.

Usage:
  python scripts/build_label_source_pairs.py --data-dir data/civtelescope \\
      --labels results/preference_labels.jsonl --out-dir data/label_source
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope.ablations import label_source_pairs
from civmarsh.civtelescope.data import load_dataset, read_pairs
from civmarsh.utils.io import read_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-pairs", help="pair file (default: the dataset's)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, help="use the first N labeled pairs (default all)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    train = (
        read_pairs(a.train_pairs) if a.train_pairs else load_dataset(a.data_dir).train
    )
    replay, preference, stats = label_source_pairs(train, read_jsonl(a.labels), n=a.n)
    out = Path(a.out_dir)
    write_jsonl(out / "replay_labeled.jsonl", replay, sort_keys=True)
    write_jsonl(out / "preference_labeled.jsonl", preference, sort_keys=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
