#!/usr/bin/env python3
"""Same-start trap rate from RL rollout value harvests (civmarsh.traps.same_start):
scoreboard traps between rollouts that share a start, compared at the same
turn, against cross-start pairs of the same episodes.

Each input is one arm's per-episode value harvest (value_harvest.jsonl). An input may be
NAME=PATH, a file (arm name = its parent directory's name), a directory (every
*.jsonl beneath it is an arm, named by its subdirectory, or by its file name
when it sits directly in the directory), or a glob pattern.

Usage:
  python scripts/same_start_traps.py --inputs runs/*/value_harvest.jsonl \\
      --out results/same_start_traps.json
  python scripts/same_start_traps.py --inputs scoreboard_rem40=runs/a/value_harvest.jsonl \\
      civtelescope_rem40=runs/b/value_harvest.jsonl --out same_start.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from civmarsh.traps.same_start import same_start_traps


def collect_arms(inputs: list[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}

    def add(name: str, path: Path) -> None:
        if name in arms:
            raise SystemExit(f"duplicate arm name {name!r} ({arms[name]} and {path})")
        arms[name] = path

    for spec in inputs:
        name, sep, rest = spec.partition("=")
        if sep and not Path(spec).exists():
            add(name, Path(rest))
            continue
        paths = [Path(p) for p in sorted(glob.glob(spec))] or [Path(spec)]
        for p in paths:
            if p.is_dir():
                for f in sorted(p.rglob("*.jsonl")):
                    rel = f.relative_to(p)
                    add(str(rel.parent) if rel.parent != Path(".") else f.stem, f)
            elif p.is_file():
                add(p.resolve().parent.name, p)
            else:
                raise SystemExit(f"no such input: {p}")
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    arms = collect_arms(a.inputs)
    out = same_start_traps(arms)
    for name, arm in out["arms"].items():
        print(f"{name}: {arm['n_episodes']} episodes ({arm['n_excluded']} excluded)")
    print("turn  same-start trap (n)        cross-start trap (n)")
    for t, d in out["pooled_per_turn"].items():
        s, c = d["same_start"], d["cross_start"]
        print(
            f"t{t:<4} {s['trap_strict']!s:>8} ({s['n_strict']:>7})  "
            f"{c['trap_strict']!s:>8} ({c['n_strict']:>7})"
        )
    for b, v in out["pooled_same_start_by_bucket"].items():
        if v:
            print(
                f"{b}: same-start trap {v['rate']} CI {v['ci95']} "
                f"(n={v['n_strict']}, {v['n_starts']} start clusters)"
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
