#!/usr/bin/env python3
"""Train CivTelescope on an Othello or Catan pair dataset.

The Freeciv recipe unchanged (``civmarsh.civtelescope.train``: LoRA r16 on
q_proj / v_proj, SFT on the answer letter, both presentation orders); only the
pairwise prompt names the game instead of Freeciv.  It is the template the
reward uses at training time (``civmarsh.rewards.games.game_pairwise_prompt``),
so the training and reward inputs cannot drift apart.  The held-out trap and
non-trap slices are scored every --eval-interval optimizer steps; each
evaluation appends a row to --monitor (acc_trap, acc_natural, and the untrained
backbone on the same pairs) and saves ``lora_step<step>_e<epoch>.pt``.  Pick
the snapshot by held-out trap accuracy, then merge it with scripts/merge_lora.py.

Usage:
  python scripts/games/train_civtelescope.py --game othello \\
      --base-model /models/Qwen2.5-7B-Instruct \\
      --data-dir data/games/othello/civtelescope --out ckpts/games/othello/civtelescope \\
      --monitor ckpts/games/othello/civtelescope/monitor.jsonl

One 80 GB GPU.
"""

from __future__ import annotations

import argparse
import functools

from civmarsh.civtelescope import train as civtelescope_train
from civmarsh.civtelescope.data import load_dataset
from civmarsh.civtelescope.train import TrainConfig
from civmarsh.rewards.games import PROMPT_GAME_NAME, game_pairwise_prompt


def main() -> int:
    d = TrainConfig(base_model="")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", choices=sorted(PROMPT_GAME_NAME), required=True)
    ap.add_argument("--base-model", required=True, help="HF model directory or id")
    ap.add_argument(
        "--data-dir", required=True, help="scripts/games/build_pairs.py output"
    )
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
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument(
        "--no-snapshots", action="store_true", help="do not save per-eval adapters"
    )
    ap.add_argument(
        "--allow-cudnn-attention",
        action="store_true",
        help="keep cuDNN scaled-dot-product attention enabled.  By default it is "
        "disabled: on left-padded batches its backward has produced non-finite "
        "LoRA gradients with some torch versions",
    )
    a = ap.parse_args()

    if not a.allow_cudnn_attention:
        import torch

        torch.backends.cuda.enable_cudnn_sdp(False)

    # Training and the held-out evaluation build their examples through the
    # module-level ``build_examples``; bind the game's template there.
    template = game_pairwise_prompt(PROMPT_GAME_NAME[a.game])
    civtelescope_train.build_examples = functools.partial(
        civtelescope_train.build_examples, prompt_template=template
    )

    ds = load_dataset(a.data_dir)
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
    civtelescope_train.train_civtelescope(
        cfg,
        ds.train,
        ds.positions,
        a.out,
        eval_pairs=ds.heldout_pairs(),
        monitor_path=a.monitor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
