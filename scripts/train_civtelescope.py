#!/usr/bin/env python3
"""Train CivTelescope's LoRA adapters on a dataset directory.

The dataset directory is the output of build_civtelescope_pairs.py. Training
pairs default to its train_pairs.jsonl; --train-pairs trains on another pair
file over the same positions (e.g. the label-source ablation's pair sets). The
held-out natural and trap slices are scored every --eval-interval optimizer
steps; each evaluation appends a row to --monitor and saves
``lora_step<step>_e<epoch>.pt`` under --out. The final adapters are
``<out>/lora.pt``. Defaults are those of civmarsh.civtelescope.train.

Usage:
  python scripts/train_civtelescope.py --base-model /models/Qwen2.5-7B-Instruct \\
      --data-dir data/civtelescope --out ckpts/civtelescope \\
      --monitor ckpts/civtelescope/monitor.jsonl
"""

import argparse

from civmarsh.civtelescope.data import load_dataset, read_pairs
from civmarsh.civtelescope.train import TrainConfig, train_civtelescope


def main() -> int:
    d = TrainConfig(base_model="")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True, help="HF model directory or id")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-pairs", help="pair file to train on instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--monitor", help="append-only JSONL of interval evaluations")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--accum", type=int, default=d.accum)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--token-budget", type=int, default=d.token_budget)
    ap.add_argument("--eval-interval", type=int, default=d.eval_interval)
    ap.add_argument(
        "--no-snapshots", action="store_true", help="do not save per-eval adapters"
    )
    a = ap.parse_args()

    ds = load_dataset(a.data_dir)
    train_pairs = read_pairs(a.train_pairs) if a.train_pairs else ds.train
    cfg = TrainConfig(
        base_model=a.base_model,
        epochs=a.epochs,
        batch=a.batch,
        accum=a.accum,
        lr=a.lr,
        lora_r=a.lora_r,
        seed=a.seed,
        token_budget=a.token_budget,
        eval_interval=a.eval_interval,
        save_every_eval=not a.no_snapshots,
        device=a.device,
    )
    train_civtelescope(
        cfg,
        train_pairs,
        ds.positions,
        a.out,
        eval_pairs=ds.heldout_pairs(),
        monitor_path=a.monitor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
