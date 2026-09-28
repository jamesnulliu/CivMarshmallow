#!/usr/bin/env python3
"""Zero-shot LLM accuracy on trap and non-trap pairs of a cross-game bank
(civmarsh.traps.games.zero_shot).

Builds the bank's decidable pairs, samples 300 trap and 300 non-trap pairs
(seed 97, visible-score ties dropped), asks every model each pair in both
orders, and writes per-model accuracy on trap and non-trap pairs.

Usage:
  CIVMARSH_API_BASE=https://... CIVMARSH_API_KEY=... \
  python scripts/eval_game_pairs.py --game chess --bank data/games/chess_bank.jsonl \
      --models MODEL_A,MODEL_B --out results/chess_zero_shot.json --workers 8
  # thinking off for a reasoning model
  python scripts/eval_game_pairs.py --game othello --bank data/games/othello_bank.jsonl \
      --models MODEL --reasoning-effort none --out results/othello_no_thinking.json

Calls are cached per model under --cache-dir (default: next to --out), so a
rerun only repeats failed calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from civmarsh.oracle.pairs import decidable_pairs
from civmarsh.traps.games import GAMES
from civmarsh.traps.games.zero_shot import (
    N_NON_TRAP,
    N_TRAP,
    SEED,
    cache_path_for,
    evaluate_model,
    select_pairs,
)
from civmarsh.utils.api import ApiClient
from civmarsh.utils.io import read_jsonl

T0 = time.time()


def log(msg: str) -> None:
    print(f"[zero-shot +{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", required=True, choices=sorted(GAMES))
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--reasoning-effort", help="e.g. 'none' to turn thinking off")
    ap.add_argument("--n-trap", type=int, default=N_TRAP)
    ap.add_argument("--n-non-trap", type=int, default=N_NON_TRAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-turn-gap", type=int, help="default: the game's pair gap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, help="per-request timeout in seconds")
    ap.add_argument("--max-concurrency", type=int)
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--cache-dir", type=Path)
    a = ap.parse_args()

    bank = {r["position_id"]: r for r in read_jsonl(a.bank)}
    gap = a.max_turn_gap or GAMES[a.game]["max_turn_gap"]
    all_pairs = decidable_pairs(
        bank.values(),
        max_turn_gap=gap,
        value_key="oracle_mean",
        se_key="oracle_se",
        drop_ties=False,
    )
    pairs = select_pairs(
        all_pairs, n_trap=a.n_trap, n_non_trap=a.n_non_trap, seed=a.seed
    )
    n_trap = sum(p["is_trap"] for p in pairs)
    log(
        f"{a.game}: {len(all_pairs)} decidable pairs; {n_trap} trap + "
        f"{len(pairs) - n_trap} non-trap selected"
    )

    cache_dir = a.cache_dir or a.out.parent
    models = [m for m in a.models.split(",") if m]
    client_kw = {
        "reasoning_effort": a.reasoning_effort,
        "max_concurrency": a.max_concurrency,
        "min_interval": a.min_interval,
    }
    if a.timeout is not None:
        client_kw["timeout"] = a.timeout
    res = {"game": a.game, "models": {}}
    for model in models:
        client = ApiClient(
            model, cache_path=cache_path_for(cache_dir, a.game, model), **client_kw
        )
        log(f"{model}: {2 * len(pairs)} calls (cached calls are free)")
        res["models"][model] = evaluate_model(
            client, a.game, bank, pairs, workers=a.workers, log=log
        )
        log(f"{model}: {res['models'][model]}")
    res["protocol"] = {
        "bank_file": str(a.bank),
        "max_turn_gap": gap,
        "n_trap": a.n_trap,
        "n_non_trap": a.n_non_trap,
        "seed": a.seed,
        "both_orders": True,
        "reasoning_effort": a.reasoning_effort,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
