#!/usr/bin/env python3
"""Score a trained value head on pairs, next to CivTelescope on the same pairs.

Held-out mode (--data-dir and --labels): the same split as training; reports
the Pearson correlation between predicted and true end score on the training
and the held-out positions, and picks on the held-out replay pairs. Pair mode
(--pairs and --rendered): picks on any pair file, e.g. the ruleset-transfer
pairs. A pair's pick is the position with the higher prediction.

Raw picks of other models (--reference-raw, e.g. eval_civtelescope.py output)
are restricted to the same pairs and read out alongside: trap and natural
accuracy with cluster-bootstrap intervals. The head reads the prompt it was
trained with (--ruleset, civ2civ3 by default).

Usage:
  python scripts/eval_value_head.py --base-model /models/Qwen2.5-7B-Instruct \\
      --checkpoint ckpts/value_head.pt --data-dir data/civtelescope \\
      --labels bank/labels.jsonl --reference-raw results/heldout.raw.jsonl \\
      --out results/value_head.raw.jsonl --readout results/value_head.json
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope import evaluate
from civmarsh.civtelescope.data import load_dataset, load_renderings, read_pairs
from civmarsh.civtelescope.value_head import (
    LABEL,
    build_examples,
    load_value_head,
    pair_picks,
    pearson,
    predict_positions,
    value_head_split,
)
from civmarsh.utils.io import iter_jsonl
from civmarsh.utils.stats import CLUSTER_RULES, pair_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", help="held-out mode: dataset directory")
    ap.add_argument("--labels", help="held-out mode: replay-oracle labels")
    ap.add_argument("--label-filter", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("--pairs", help="pair mode: pair file")
    ap.add_argument("--rendered", help="pair mode: fog renderings")
    ap.add_argument("--ruleset", default="civ2civ3", help="ruleset named in the prompt")
    ap.add_argument("--reference-raw", nargs="*", default=[], help="raw pick files")
    ap.add_argument("--cluster", default="episode", choices=sorted(CLUSTER_RULES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=20000)
    ap.add_argument("--out", required=True, help="raw picks JSONL")
    ap.add_argument("--readout", help="readout JSON")
    a = ap.parse_args()

    report = {"checkpoint": a.checkpoint}
    if a.data_dir and a.labels:
        ds = load_dataset(a.data_dir)
        want = [f.split("=", 1) for f in a.label_filter]
        targets = {
            r["position_id"]: float(r["mean_end"])
            for r in iter_jsonl(a.labels)
            if all(str(r.get(k)) == v for k, v in want)
        }
        train, ev, pairs, _ = value_head_split(
            ds.positions, targets, ds.heldout_pairs()
        )
        splits = {"train": train, "eval": ev}
    elif a.pairs and a.rendered:
        pairs = read_pairs(a.pairs)
        need = {p[k] for p in pairs for k in ("a", "b")}
        positions = load_renderings(a.rendered, ids=need)
        rows = [
            {**positions[pid], "mean_end": 0.0}
            for pid in sorted(need)
            if pid in positions
        ]
        pairs = [p for p in pairs if p["a"] in positions and p["b"] in positions]
        splits = {"eval": rows}
    else:
        ap.error("pass --data-dir and --labels, or --pairs and --rendered")

    model, head, tok, ck = load_value_head(a.base_model, a.checkpoint, device=a.device)
    report["steps"] = ck.get("steps")
    preds = {}
    for name, rows in splits.items():
        ex = build_examples(rows, tok, ruleset=a.ruleset)
        preds.update(
            predict_positions(
                model,
                head,
                ex,
                tok,
                a.device,
                batch=a.batch,
                token_budget=a.token_budget,
            )
        )
        if a.labels:
            xs = [preds[r["position_id"]] for r in rows]
            ys = [r["mean_end"] for r in rows]
            report[f"pearson_{name}"] = round(pearson(xs, ys), 4)
            report[f"n_{name}"] = len(rows)

    raw = {LABEL: pair_picks(pairs, preds)}
    keep = {pair_id(p) for p in pairs}
    for label, picks in evaluate.read_raw(a.reference_raw).items():
        raw[label] = {pid: pk for pid, pk in picks.items() if pid in keep}
    evaluate.write_raw(a.out, raw)
    report["readout"] = evaluate.readout(
        pairs, raw, cluster_of=CLUSTER_RULES[a.cluster]
    )
    text = json.dumps(report, indent=1)
    print(text)
    if a.readout:
        Path(a.readout).parent.mkdir(parents=True, exist_ok=True)
        Path(a.readout).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
