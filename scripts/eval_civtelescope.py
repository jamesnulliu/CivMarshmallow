#!/usr/bin/env python3
"""Score pairs with CivTelescope and its backbone.

Every pair is scored in both presentation orders; the backbone row is the same
model with the adapters switched off (--no-adapter scores the backbone alone).
Pairs come from --pairs (a pair file, e.g. build_transfer_pairs.py output) or
from the held-out slices of --data-dir; renderings from --rendered or from
--data-dir. --ruleset names the ruleset in the prompt. --input-variant scores
an input ablation of the renderings (see input_ablation.py).

Writes raw picks to --out. Unsharded runs also write the per-model readout
(trap / natural accuracy with game-clustered intervals) to --readout; shards
(--shard i/n, interleaved over pairs sorted by id) are merged by
report_civtelescope.py.

Usage:
  python scripts/eval_civtelescope.py --base-model /models/Qwen2.5-7B-Instruct \\
      --adapters ckpts/civtelescope/lora.pt --data-dir data/civtelescope \\
      --out results/heldout.raw.jsonl --readout results/heldout.json
  python scripts/eval_civtelescope.py --base-model ... --adapters ... \\
      --pairs data/transfer/pairs.jsonl --rendered bank/rendered_fog.jsonl \\
      --ruleset classic --shard 0/4 --out results/transfer.shard0.raw.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from civmarsh.civtelescope import evaluate
from civmarsh.civtelescope.ablations import VARIANTS, transform
from civmarsh.civtelescope.data import load_dataset, load_renderings, read_pairs
from civmarsh.utils.stats import CLUSTER_RULES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapters", help="CivTelescope adapter checkpoint (.pt)")
    g.add_argument("--no-adapter", action="store_true", help="score the backbone only")
    ap.add_argument(
        "--pairs", help="pair file (default: held-out slices of --data-dir)"
    )
    ap.add_argument("--rendered", help="fog renderings (default: --data-dir positions)")
    ap.add_argument("--data-dir", help="dataset directory")
    ap.add_argument("--ruleset", default="civ2civ3")
    ap.add_argument("--input-variant", default="full", choices=VARIANTS)
    ap.add_argument("--shard", default="0/1", help="i/n")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=evaluate.EVAL_BATCH)
    ap.add_argument("--token-budget", type=int, default=evaluate.EVAL_TOKEN_BUDGET)
    ap.add_argument("--out", required=True, help="raw picks JSONL")
    ap.add_argument("--readout", help="readout JSON (unsharded runs)")
    ap.add_argument("--cluster", default="episode", choices=sorted(CLUSTER_RULES))
    a = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[eval +{time.time() - t0:7.1f}s] {msg}", flush=True)

    ds = load_dataset(a.data_dir) if a.data_dir else None
    if a.pairs:
        pairs = read_pairs(a.pairs)
    elif ds:
        pairs = ds.heldout_pairs()
    else:
        ap.error("pass --pairs or --data-dir")
    if a.rendered:
        positions = load_renderings(a.rendered)
    elif ds:
        positions = ds.positions
    else:
        ap.error("pass --rendered or --data-dir")
    if a.input_variant != "full":
        positions = {
            pid: {**r, "rendering": transform(r["rendering"], a.input_variant)}
            for pid, r in positions.items()
        }
    pairs = [p for p in pairs if p["a"] in positions and p["b"] in positions]
    shard = evaluate.shard(pairs, a.shard)
    log(
        f"shard {a.shard}: {len(shard)} of {len(pairs)} pairs "
        f"({sum(p['trap'] for p in shard)} trap), input {a.input_variant}"
    )
    raw = evaluate.score_pairs(
        a.base_model,
        shard,
        positions,
        adapters_path=a.adapters,
        ruleset=a.ruleset,
        device=a.device,
        batch=a.batch,
        token_budget=a.token_budget,
        log=log,
    )
    evaluate.write_raw(a.out, raw)
    log(f"raw picks -> {a.out}")
    if a.shard.split("/")[1] == "1":
        rep = evaluate.readout(pairs, raw, cluster_of=CLUSTER_RULES[a.cluster])
        print(json.dumps(rep, indent=1))
        if a.readout:
            Path(a.readout).parent.mkdir(parents=True, exist_ok=True)
            Path(a.readout).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
