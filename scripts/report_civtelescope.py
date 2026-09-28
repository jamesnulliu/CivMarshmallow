#!/usr/bin/env python3
"""Readouts from CivTelescope's raw picks and zero-shot picks on the same pairs.

Merges raw pick files (shards of eval_civtelescope.py, or the raw picks of
eval_value_head.py) and per-order pick files of zero-shot models
(eval_llm_baselines.py), then reports:

- readout: per model, trap and natural accuracy with cluster-bootstrap
  intervals (2,000 draws);
- paired: with both CivTelescope and backbone rows, the accuracy table, the
  paired difference CivTelescope minus backbone per slice (10,000 draws) and
  trap accuracy by turn bucket;
- lineup (with --api-pairs): on the pairs every model scored, trap accuracy
  with intervals and CivTelescope minus the best other row, the best re-taken
  in every draw (1,000 draws). Zero-shot runs covering less than
  --min-coverage of the subset are left out.

Usage:
  python scripts/report_civtelescope.py --pairs data/transfer/pairs.jsonl \\
      --raw results/transfer.shard*.raw.jsonl \\
      --picks MODEL=results/llm/MODEL_picks.jsonl \\
      --api-pairs data/transfer/pairs_api.jsonl --out results/transfer_report.json
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope import evaluate, llm_baselines
from civmarsh.civtelescope.data import load_dataset, read_pairs
from civmarsh.utils.stats import CLUSTER_RULES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--pairs", help="pair file (default: held-out slices of --data-dir)"
    )
    ap.add_argument("--data-dir")
    ap.add_argument("--raw", nargs="*", default=[], help="raw pick files")
    ap.add_argument("--picks", action="append", default=[], help="LABEL=PATH")
    ap.add_argument("--api-pairs", help="subset for the zero-shot lineup")
    ap.add_argument("--reference", default=evaluate.MODEL_LABEL)
    ap.add_argument("--backbone", default=evaluate.BACKBONE_LABEL)
    ap.add_argument("--cluster", default="episode", choices=sorted(CLUSTER_RULES))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-paired", type=int, default=10000)
    ap.add_argument("--n-boot-lineup", type=int, default=1000)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.pairs:
        pairs = read_pairs(a.pairs)
    elif a.data_dir:
        pairs = load_dataset(a.data_dir).heldout_pairs()
    else:
        ap.error("pass --pairs or --data-dir")
    cluster_of = CLUSTER_RULES[a.cluster]
    raw = evaluate.read_raw(a.raw)
    zero_shot = {}
    for spec in a.picks:
        label, path = spec.split("=", 1)
        zero_shot[label] = evaluate.read_order_picks(path)
    everything = {**raw, **zero_shot}

    report = {
        "cluster": a.cluster,
        "readout": evaluate.readout(
            pairs, everything, cluster_of=cluster_of, n_boot=a.n_boot
        ),
    }
    if a.reference in raw and a.backbone in raw:
        report["paired"] = evaluate.transfer_report(
            pairs,
            raw,
            reference=a.reference,
            backbone=a.backbone,
            cluster_of=cluster_of,
            n_boot=a.n_boot_paired,
        )
    if a.api_pairs:
        api_pairs = read_pairs(a.api_pairs)
        models = {m: raw[m] for m in (a.reference, a.backbone) if m in raw}
        skipped = {}
        for label, picks in zero_shot.items():
            covered = len(
                {p for p in picks if picks[p]}
                & {f"{q['a']}|{q['b']}" for q in api_pairs}
            )
            if covered < a.min_coverage * len(api_pairs):
                skipped[label] = covered
                continue
            models[label] = picks
        report["lineup"] = llm_baselines.lineup(
            api_pairs,
            models,
            reference=a.reference,
            cluster_of=cluster_of,
            n_boot=a.n_boot_lineup,
        )
        report["lineup"]["skipped_incomplete"] = skipped
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
