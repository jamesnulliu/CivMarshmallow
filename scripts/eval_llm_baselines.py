#!/usr/bin/env python3
"""Score pairs with a zero-shot language model through an OpenAI-compatible API.

Same prompt as CivTelescope (the ruleset named), same fog renderings, both
presentation orders. --warned adds the trap warning to the prompt. With
--n-trap / --n-natural the pairs are first subsampled (Random(--seed)); the
subsample is frozen in --subsample (read if it exists, written otherwise) so
every model is scored on the same pairs. Calls are cached under
<out-dir>/cache/, so a rerun only repeats failed calls.

Writes <out-dir>/<label>_picks.jsonl (one row per pair and order) and
<label>_summary.json; the label defaults to the model name, with "-warned"
appended for the warned prompt. The endpoint comes from --api-base or
CIVMARSH_API_BASE, the key from CIVMARSH_API_KEY.

Usage:
  python scripts/eval_llm_baselines.py --model MODEL --data-dir data/civtelescope \\
      --n-trap 300 --n-natural 300 --subsample results/llm/subsample_pairs.jsonl \\
      --out-dir results/llm [--warned] [--reasoning-effort none]
  python scripts/eval_llm_baselines.py --model MODEL --ruleset classic \\
      --pairs data/transfer/pairs_api.jsonl --rendered bank/rendered_fog.jsonl \\
      --out-dir results/transfer_llm
"""

import argparse
import json
import time
from pathlib import Path

from civmarsh.civtelescope import llm_baselines
from civmarsh.civtelescope.data import load_dataset, load_renderings, read_pairs
from civmarsh.utils.api import add_client_args, client_from_args
from civmarsh.utils.io import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", help="output label (default: model name)")
    ap.add_argument("--warned", action="store_true", help="add the trap warning")
    ap.add_argument(
        "--pairs", help="pair file (default: held-out slices of --data-dir)"
    )
    ap.add_argument("--rendered", help="fog renderings (default: --data-dir positions)")
    ap.add_argument("--data-dir")
    ap.add_argument("--ruleset", default="civ2civ3")
    ap.add_argument("--n-trap", type=int, help="subsample this many trap pairs")
    ap.add_argument("--n-natural", type=int, help="subsample this many natural pairs")
    ap.add_argument("--seed", type=int, default=llm_baselines.SUBSAMPLE_SEED)
    ap.add_argument("--subsample", help="frozen subsample file")
    ap.add_argument("--limit", type=int, help="first N pairs only (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", required=True)
    add_client_args(ap)
    a = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[llm +{time.time() - t0:7.1f}s] {msg}", flush=True)

    ds = load_dataset(a.data_dir) if a.data_dir else None
    if a.subsample and Path(a.subsample).exists():
        pairs = read_pairs(a.subsample)
    else:
        if a.pairs:
            pairs = read_pairs(a.pairs)
        elif ds:
            pairs = ds.heldout_pairs()
        else:
            ap.error("pass --pairs or --data-dir")
        if a.n_trap is not None or a.n_natural is not None:
            pairs = llm_baselines.heldout_subsample(
                pairs, n_trap=a.n_trap or 0, n_natural=a.n_natural or 0, seed=a.seed
            )
            if a.subsample:
                write_jsonl(a.subsample, pairs, sort_keys=True)
                log(f"froze subsample of {len(pairs)} pairs -> {a.subsample}")
    if a.limit:
        pairs = pairs[: a.limit]
    need = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    if a.rendered:
        positions = load_renderings(a.rendered, ids=need)
    elif ds:
        positions = ds.positions
    else:
        ap.error("pass --rendered or --data-dir")
    pairs = [p for p in pairs if p["a"] in positions and p["b"] in positions]

    label = a.label or (a.model + ("-warned" if a.warned else ""))
    out = Path(a.out_dir)
    client = client_from_args(a, a.model, out / "cache" / f"{label}.jsonl")
    log(f"{a.model} ({label}): {len(pairs)} pairs -> {2 * len(pairs)} calls")
    rows, counts = llm_baselines.score_pairs(
        client,
        pairs,
        positions,
        ruleset=a.ruleset,
        warned=a.warned,
        workers=a.workers,
        log=log,
    )
    rows.sort(key=lambda r: (r["pair_id"], r["order"]))
    write_jsonl(out / f"{label}_picks.jsonl", rows, sort_keys=True)
    summary = {
        "model": a.model,
        "label": label,
        "ruleset": a.ruleset,
        "warned": a.warned,
        "reasoning_effort": client.reasoning_effort,
        **llm_baselines.summarize(pairs, rows, counts),
        "usage": client.usage_totals(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / f"{label}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(json.dumps(summary))
    return 0 if counts["n_http_fail"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
