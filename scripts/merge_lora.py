#!/usr/bin/env python3
"""Merge a CivTelescope adapter checkpoint into its base model.

Writes a Hugging Face model directory (bf16 weights + tokenizer) that an
inference server such as sglang can load (see serve_civtelescope.sh). The LoRA
rank is read from the checkpoint.

Usage:
  python scripts/merge_lora.py --base-model /models/Qwen2.5-7B-Instruct \\
      --adapters ckpts/civtelescope/lora.pt --out models/civtelescope
"""

import argparse

from civmarsh.civtelescope.lora import DEFAULT_ALPHA, merge_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapters", required=True, help="adapter checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output model directory")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    a = ap.parse_args()
    n = merge_checkpoint(a.base_model, a.adapters, a.out, alpha=a.alpha)
    print(f"merged {n} adapted layers -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
