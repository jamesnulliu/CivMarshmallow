#!/usr/bin/env python3
"""Train a scalar regression value head on CivTelescope's backbone.

The head is trained on the replay positions of the dataset directory with
their replay-oracle mean end score as target; every game that appears in a
held-out replay pair is excluded from training. Held-out pair accuracy is
logged to --monitor during training; the saved model is the one after the
last epoch. Defaults are those of civmarsh.civtelescope.value_head.

Usage:
  python scripts/train_value_head.py --base-model /models/Qwen2.5-7B-Instruct \\
      --data-dir data/civtelescope --labels bank/labels.jsonl \\
      --out ckpts/value_head.pt --monitor ckpts/value_head_monitor.jsonl
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope.data import load_dataset
from civmarsh.civtelescope.value_head import (
    ValueHeadConfig,
    train_value_head,
    value_head_split,
)
from civmarsh.utils.io import iter_jsonl


def read_targets(path, filters: list[str]) -> dict[str, float]:
    want = [f.split("=", 1) for f in filters]
    return {
        r["position_id"]: float(r["mean_end"])
        for r in iter_jsonl(path)
        if all(str(r.get(k)) == v for k, v in want)
    }


def main() -> int:
    d = ValueHeadConfig(base_model="")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels", required=True, help="replay-oracle labels (mean_end)")
    ap.add_argument("--label-filter", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("--ruleset", default="civ2civ3", help="ruleset named in the prompt")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--monitor", help="JSONL of held-out evaluations")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--accum", type=int, default=d.accum)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--head-lr", type=float, default=d.head_lr)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--token-budget", type=int, default=d.token_budget)
    ap.add_argument("--eval-interval", type=int, default=d.eval_interval)
    a = ap.parse_args()

    ds = load_dataset(a.data_dir)
    targets = read_targets(a.labels, a.label_filter)
    train, ev, pairs, stats = value_head_split(
        ds.positions, targets, ds.heldout_pairs()
    )
    if stats["game_overlap"]:
        raise SystemExit(f"evaluation games in training: {stats['game_overlap'][:5]}")
    print(json.dumps(stats, indent=1), flush=True)
    cfg = ValueHeadConfig(
        base_model=a.base_model,
        epochs=a.epochs,
        batch=a.batch,
        accum=a.accum,
        lr=a.lr,
        head_lr=a.head_lr,
        lora_r=a.lora_r,
        seed=a.seed,
        token_budget=a.token_budget,
        eval_interval=a.eval_interval,
        device=a.device,
    )
    res = train_value_head(
        cfg, train, ev, pairs, a.out, ruleset=a.ruleset, monitor_path=a.monitor
    )
    out = Path(a.out)
    out.with_suffix(".split.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(res["final"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
