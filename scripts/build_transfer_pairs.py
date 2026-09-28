#!/usr/bin/env python3
"""Build the ruleset-transfer evaluation pairs from a labeled bank.

Every decidable cross-game pair of labeled, rendered positions at most 10
turns apart (outcome gap above 2 pooled standard errors), then a seeded
subsample: at most 1,800 trap and 600 non-trap pairs (Random(97)), and from
those the zero-shot API subset of 200 trap + 100 non-trap pairs
(Random(29)). The visible score used for the trap rule is the bank's.

Writes pairs.jsonl, pairs_api.jsonl and pairs_stats.json to --out-dir.

Usage:
  python scripts/build_transfer_pairs.py --labels bank/labels.jsonl \\
      --rendered bank/rendered_fog.jsonl --ruleset classic --out-dir data/transfer
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from civmarsh.civtelescope.data import (
    MAX_TURN_GAP,
    MIN_GAP_Z,
    PAIR_SEED,
    TRANSFER_API_NATURAL,
    TRANSFER_API_SEED,
    TRANSFER_API_TRAP,
    TRANSFER_NATURAL_KEEP,
    TRANSFER_TRAP_CAP,
    read_bank_records,
    subsample_transfer_pairs,
    transfer_pairs,
    turn_bucket,
)
from civmarsh.utils.io import iter_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True, help="bank labels (mean_end, se_end)")
    ap.add_argument("--positions", help="bank positions, for fields labels lack")
    ap.add_argument("--rendered", required=True, help="fog renderings of the bank")
    ap.add_argument("--ruleset", required=True)
    ap.add_argument("--trap-cap", type=int, default=TRANSFER_TRAP_CAP)
    ap.add_argument("--natural-keep", type=int, default=TRANSFER_NATURAL_KEEP)
    ap.add_argument("--api-trap", type=int, default=TRANSFER_API_TRAP)
    ap.add_argument("--api-natural", type=int, default=TRANSFER_API_NATURAL)
    ap.add_argument("--seed", type=int, default=PAIR_SEED)
    ap.add_argument("--api-seed", type=int, default=TRANSFER_API_SEED)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    rendered = {r["position_id"] for r in iter_jsonl(a.rendered)}
    records = [
        r
        for r in read_bank_records(a.labels, a.positions)
        if r["position_id"] in rendered
    ]
    pairs = transfer_pairs(records, ruleset=a.ruleset)
    keep, api = subsample_transfer_pairs(
        pairs,
        trap_cap=a.trap_cap,
        natural_keep=a.natural_keep,
        seed=a.seed,
        api_trap=a.api_trap,
        api_natural=a.api_natural,
        api_seed=a.api_seed,
    )
    out = Path(a.out_dir)
    write_jsonl(out / "pairs.jsonl", keep, sort_keys=True)
    write_jsonl(out / "pairs_api.jsonl", api, sort_keys=True)
    n_trap = sum(p["trap"] for p in pairs)
    stats = {
        "n_positions_paired": len(records),
        "n_games": len({r["game_id"] for r in records}),
        "n_decidable": len(pairs),
        "n_trap": n_trap,
        "trap_rate_decidable": round(n_trap / max(len(pairs), 1), 4),
        "kept": {
            "n": len(keep),
            "trap": sum(p["trap"] for p in keep),
            "natural": sum(not p["trap"] for p in keep),
            "trap_by_bucket": dict(Counter(turn_bucket(p) for p in keep if p["trap"])),
        },
        "api": {
            "n": len(api),
            "trap": sum(p["trap"] for p in api),
            "natural": sum(not p["trap"] for p in api),
        },
        "rule": {
            "max_turn_gap": MAX_TURN_GAP,
            "min_gap_z": MIN_GAP_Z,
            "trap_cap": a.trap_cap,
            "natural_keep": a.natural_keep,
            "api_trap": a.api_trap,
            "api_natural": a.api_natural,
            "seed": a.seed,
            "api_seed": a.api_seed,
        },
    }
    (out / "pairs_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
