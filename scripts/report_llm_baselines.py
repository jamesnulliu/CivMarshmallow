#!/usr/bin/env python3
"""Compare zero-shot models with CivTelescope on the held-out set.

For every model (CivTelescope and its backbone from raw pick files, zero-shot
runs from their per-order pick files, warned runs under their own label):
trap and natural accuracy on the subsample and on the full held-out set, with
episode-clustered intervals; and on the subsample's trap pairs, CivTelescope
minus the model and CivTelescope / the model, both paired over the same pairs
(2,000 draws, seed 7).

Usage:
  python scripts/report_llm_baselines.py --data-dir data/civtelescope \\
      --subsample results/llm/subsample_pairs.jsonl --raw results/heldout.raw.jsonl \\
      --picks MODEL=results/llm/MODEL_picks.jsonl \\
      --picks MODEL-warned=results/llm/MODEL-warned_picks.jsonl \\
      --out results/llm/comparison.json
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope import evaluate, llm_baselines
from civmarsh.civtelescope.data import load_dataset, read_pairs
from civmarsh.utils.stats import CLUSTER_RULES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", help="full held-out pair file (default: --data-dir)")
    ap.add_argument("--data-dir")
    ap.add_argument("--subsample", required=True, help="frozen subsample file")
    ap.add_argument("--raw", nargs="*", default=[], help="raw pick files")
    ap.add_argument("--picks", action="append", default=[], help="LABEL=PATH")
    ap.add_argument("--reference", default=evaluate.MODEL_LABEL)
    ap.add_argument("--cluster", default="episode", choices=sorted(CLUSTER_RULES))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.pairs:
        full = read_pairs(a.pairs)
    elif a.data_dir:
        full = load_dataset(a.data_dir).heldout_pairs()
    else:
        ap.error("pass --pairs or --data-dir")
    picks = evaluate.read_raw(a.raw)
    for spec in a.picks:
        label, path = spec.split("=", 1)
        picks[label] = evaluate.read_order_picks(path)
    report = llm_baselines.heldout_comparison(
        read_pairs(a.subsample),
        full,
        picks,
        reference=a.reference,
        cluster_of=CLUSTER_RULES[a.cluster],
        n_boot=a.n_boot,
    )
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
