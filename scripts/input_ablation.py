#!/usr/bin/env python3
"""Write an input ablation of a fog-rendering file.

Variants: ``full`` (unchanged), ``masked`` (the focal score removed from the
digest, every other line byte-identical), ``score_only`` (the header and a
digest holding only the score, the turn and the digest schema). --check
asserts these invariants on every rendering and writes nothing. The ablated
file is scored with eval_civtelescope.py --rendered (or directly with
eval_civtelescope.py --input-variant).

Usage:
  python scripts/input_ablation.py --rendered bank/rendered_fog.jsonl \\
      --variant masked --out work/rendered_masked.jsonl
  python scripts/input_ablation.py --data-dir data/civtelescope --check
"""

import argparse
import hashlib
import json

from civmarsh.civtelescope.ablations import VARIANTS, check_transform, transform
from civmarsh.civtelescope.data import load_positions
from civmarsh.utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rendered", help="fog renderings JSONL")
    ap.add_argument("--data-dir", help="dataset directory (its positions)")
    ap.add_argument("--variant", choices=VARIANTS, default="masked")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.rendered:
        rows = read_jsonl(a.rendered)
    elif a.data_dir:
        rows = list(load_positions(a.data_dir).values())
    else:
        ap.error("pass --rendered or --data-dir")
    if a.check:
        for r in rows:
            check_transform(r["rendering"])
        n_score = sum('"score"' in r["rendering"] for r in rows)
        print(f"{len(rows)} renderings checked, {n_score} contain a score key")
        return 0
    if not a.out:
        ap.error("--out is required unless --check")
    text = "".join(
        json.dumps({**r, "rendering": transform(r["rendering"], a.variant)}) + "\n"
        for r in rows
    )
    with open(a.out, "w") as f:
        f.write(text)
    print(
        f"{a.out}: {len(rows)} rows, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
