"""Zero-shot LLM evaluation on cross-game position pairs.

The protocol of the Freeciv zero-shot baselines, with only the game named in
the prompt changed: a seeded sample of trap and non-trap pairs (visible-score
ties dropped), each pair asked in both orders, and a model's accuracy on a pair
averaged over the orders it answered. Accuracy is reported separately on trap
pairs (where following the visible score is wrong) and non-trap pairs.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from civmarsh.traps.games.prompt import cross_game_prompt

N_TRAP = 300
N_NON_TRAP = 300
SEED = 97


def select_pairs(
    pairs: list[dict],
    *,
    n_trap: int = N_TRAP,
    n_non_trap: int = N_NON_TRAP,
    seed: int = SEED,
) -> list[dict]:
    """Shuffle all decidable pairs (ties included) with `seed`, drop the pairs
    with a tied visible score, and take the first `n_trap` trap and first
    `n_non_trap` non-trap pairs."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    pairs = [p for p in pairs if not p["score_tied"]]
    trap = [p for p in pairs if p["is_trap"]][:n_trap]
    non_trap = [p for p in pairs if not p["is_trap"]][:n_non_trap]
    return trap + non_trap


def wilson(k: float, n: int, z: float = 1.96):
    """Wilson score interval for k successes out of n."""
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def evaluate_model(
    client,
    game: str,
    bank: dict[str, dict],
    pairs: list[dict],
    *,
    workers: int = 8,
    log=print,
) -> dict:
    """Ask one model every pair in both orders.

    client  an object with ``chat(prompt, tag=...) -> str`` (a
            `civmarsh.utils.api.ApiClient`); errors count as unanswered.
    bank    position_id -> bank record (focal_player, rendering)."""
    from civmarsh.civtelescope.prompt import parse_pairwise

    jobs = [(p, order) for p in pairs for order in ("ab", "ba")]
    t0 = time.time()

    def one(job):
        p, order = job
        a, b = bank[p["a"]], bank[p["b"]]
        if order == "ba":
            a, b = b, a
        prompt = cross_game_prompt(game, a, b)
        try:
            text = client.chat(
                prompt, tag=f"cross_game|{game}|{p['a']}|{p['b']}|{order}"
            )
        except Exception as exc:  # noqa: BLE001 -- counted as unanswered
            return job, exc
        return job, parse_pairwise(text)

    per_pair: dict[tuple, list] = {}
    n_errors = n_unparsed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (job, val) in enumerate(ex.map(one, jobs), 1):
            p, order = job
            if isinstance(val, Exception):
                n_errors += 1
            elif val is None:
                n_unparsed += 1
            else:
                shown = val if order == "ab" else ("A" if val == "B" else "B")
                per_pair.setdefault((p["a"], p["b"]), []).append(
                    (shown == p["winner"], p["is_trap"])
                )
            if i % 500 == 0:
                log(f"  {i}/{len(jobs)} calls, {time.time() - t0:.0f}s")

    def acc(trap: bool) -> list[float]:
        return [
            sum(c for c, _ in v) / len(v) for v in per_pair.values() if v[0][1] == trap
        ]

    non_trap, trap = acc(False), acc(True)
    if not non_trap and not trap:
        return {"error": "no successful calls", "n_errors": n_errors}
    return {
        "acc_non_trap": round(sum(non_trap) / len(non_trap), 4) if non_trap else None,
        "acc_trap": round(sum(trap) / len(trap), 4) if trap else None,
        "ci95_trap": wilson(sum(trap), len(trap)) if trap else None,
        "n_non_trap": len(non_trap),
        "n_trap": len(trap),
        "n_errors": n_errors,
        "n_unparsed": n_unparsed,
    }


def cache_path_for(cache_dir: Path, game: str, model: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model)
    return Path(cache_dir) / f"{game}_{safe}_calls.jsonl"
