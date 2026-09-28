"""Same-start trap rate: scoreboard traps between policy rollouts that share a
start (same map, seed, opponents and start save), compared at the same turn.

Input is the per-episode value harvest written during RL training
(``value_harvest.jsonl``, one JSON line per episode): ``position_id`` (the start, optionally prefixed ``ep_``),
``score_end`` (realized terminal score), ``eliminated`` / ``control_lost``, and
``decisions`` of ``{turn, rendering}`` whose rendering carries the visible
score. Episodes that end in elimination or control loss are excluded.

Pair rule: two episodes at the same turn whose realized terminal scores differ
by at least `GAP_MIN`. A pair is a trap when the episode ahead on the visible
score now finishes behind. The primary rate drops pairs with equal visible
scores ("strict"); the secondary rate keeps them and resolves both orders
with ">=" ("ge"). Cross-start pairs are drawn from the same episodes at the
same turn but from different starts, so the two rates share everything except
the start.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from civmarsh.utils.freeciv import score_from_rendering

GAP_MIN = 6.0
BUCKETS = [
    ("t10-29", 10, 29),
    ("t30-49", 30, 49),
    ("t50-69", 50, 69),
    ("t70-89", 70, 89),
    ("t90-119", 90, 119),
]
SEED = 7
CROSS_CAP = 200_000
N_BOOT = 1000

# (start, realized terminal score, {turn: visible score})
Episode = tuple[str, float, dict[int, int]]


def load_harvest(path: Path) -> tuple[list[Episode], int, int]:
    """Episodes of one value harvest, plus the counts of excluded episodes
    (eliminated, control lost or unscored) and of episodes without any
    visible score."""
    eps, n_excluded, n_no_score = [], 0, 0
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (
                r.get("eliminated")
                or r.get("control_lost")
                or r.get("score_end") is None
            ):
                n_excluded += 1
                continue
            pid = r["position_id"]
            start = pid.removeprefix("ep_")
            now = {}
            for d in r["decisions"]:
                s = score_from_rendering(d["rendering"])
                if s is not None:
                    now[d["turn"]] = s
            if not now:
                n_no_score += 1
                continue
            eps.append((start, r["score_end"], now))
    return eps, n_excluded, n_no_score


def pair_counts(items: list[tuple], pairs_idx) -> tuple[int, int, int, int]:
    """items: (start, visible score, terminal score). Returns (decidable,
    untied, traps among untied, traps among all decidable with ">=")."""
    n_dec = n_strict = n_trap_strict = n_trap_ge = 0
    for i, j in pairs_idx:
        _, na, ea = items[i]
        _, nb, eb = items[j]
        if abs(ea - eb) < GAP_MIN:
            continue
        n_dec += 1
        trap = (ea >= eb) != (na >= nb)
        n_trap_ge += trap
        if na != nb:
            n_strict += 1
            n_trap_strict += trap
    return n_dec, n_strict, n_trap_strict, n_trap_ge


def _rates(counts, n_pairs: int) -> dict:
    n_dec, n_strict, n_trap_strict, n_trap_ge = counts
    return {
        "n_pairs": n_pairs,
        "n_decidable": n_dec,
        "n_strict": n_strict,
        "trap_strict": round(n_trap_strict / n_strict, 4) if n_strict else None,
        "trap_ge": round(n_trap_ge / n_dec, 4) if n_dec else None,
    }


def per_turn_rates(
    eps: list[Episode], *, seed: int = SEED, cross_cap: int = CROSS_CAP
) -> dict[int, dict]:
    """Same-start and cross-start trap counts and rates at every turn.

    Same-start pairs are all pairs of episodes of one start. Cross-start
    pairs are a seeded sample of episode pairs from two different starts,
    max(#same-start pairs, 1000) of them, capped at `cross_cap`."""
    rng = random.Random(seed)
    by_turn = defaultdict(list)
    for start, end, now in eps:
        for t, s in now.items():
            by_turn[t].append((start, s, end))
    out = {}
    for t, items in sorted(by_turn.items()):
        by_start = defaultdict(list)
        for k, (st, _, _) in enumerate(items):
            by_start[st].append(k)
        same = [
            (i, j)
            for idx in by_start.values()
            for i, j in itertools.combinations(idx, 2)
        ]
        starts = list(by_start)
        cross = []
        target = min(cross_cap, max(len(same), 1000))
        tries = 0
        while len(cross) < target and tries < target * 5 and len(starts) > 1:
            tries += 1
            a, b = rng.sample(starts, 2)
            cross.append((rng.choice(by_start[a]), rng.choice(by_start[b])))
        s = pair_counts(items, same)
        c = pair_counts(items, cross)
        out[t] = {
            "n_episodes": len(items),
            "n_starts": len(by_start),
            "same_start": _rates(s, len(same)),
            "cross_start": _rates(c, len(cross)),
            "same_start_counts": s,
            "cross_start_counts": c,
        }
    return out


def same_start_bootstrap(
    eps: list[Episode],
    lo: int,
    hi: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict | None:
    """Strict same-start trap rate over turns lo..hi with a cluster bootstrap
    over starts (95% percentile interval). None when no pair qualifies."""
    by_start = defaultdict(list)
    for start, end, now in eps:
        by_start[start].append((end, now))
    starts = sorted(by_start)
    counts = {}
    for st in starts:
        tr = ns = 0
        for (ea, na), (eb, nb) in itertools.combinations(by_start[st], 2):
            for t in na:
                if (
                    t in nb
                    and lo <= t <= hi
                    and abs(ea - eb) >= GAP_MIN
                    and na[t] != nb[t]
                ):
                    ns += 1
                    tr += (ea >= eb) != (na[t] >= nb[t])
        counts[st] = (tr, ns)
    tot_tr = sum(v[0] for v in counts.values())
    tot_ns = sum(v[1] for v in counts.values())
    if not tot_ns:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array([counts[s] for s in starts], dtype=float)
    draws = []
    for _ in range(n_boot):
        c = np.bincount(
            rng.integers(0, len(starts), len(starts)), minlength=len(starts)
        )
        num = (c * arr[:, 0]).sum()
        den = (c * arr[:, 1]).sum()
        if den:
            draws.append(num / den)
    draws.sort()
    return {
        "rate": round(tot_tr / tot_ns, 4),
        "n_strict": int(tot_ns),
        "ci95": [
            round(float(draws[int(0.025 * len(draws))]), 4),
            round(float(draws[int(0.975 * len(draws))]), 4),
        ],
        "n_starts": len(starts),
    }


def _public(per_turn: dict[int, dict]) -> dict[str, dict]:
    return {
        str(t): {k: v for k, v in d.items() if not k.endswith("_counts")}
        for t, d in per_turn.items()
    }


def arm_summary(eps: list[Episode]) -> dict:
    """Per-turn rates and per-bucket rates of one arm (pooled over the turns
    of each bucket), with the bucket's same-start bootstrap."""
    per_turn = per_turn_rates(eps)
    buckets = {}
    for name, lo, hi in BUCKETS:
        s = [per_turn[t]["same_start_counts"] for t in per_turn if lo <= t <= hi]
        c = [per_turn[t]["cross_start_counts"] for t in per_turn if lo <= t <= hi]
        if not s:
            continue
        S = [sum(x[k] for x in s) for k in range(4)]
        C = [sum(x[k] for x in c) for k in range(4)]
        buckets[name] = {
            "same_start_trap_strict": round(S[2] / S[1], 4) if S[1] else None,
            "same_start_n_strict": S[1],
            "same_start_trap_ge": round(S[3] / S[0], 4) if S[0] else None,
            "cross_start_trap_strict": round(C[2] / C[1], 4) if C[1] else None,
            "cross_start_n_strict": C[1],
            "cross_start_trap_ge": round(C[3] / C[0], 4) if C[0] else None,
            "same_start_bootstrap": same_start_bootstrap(eps, lo, hi),
        }
    return {"per_turn": _public(per_turn), "buckets": buckets}


def same_start_traps(arms: dict[str, Path]) -> dict:
    """Same-start trap rates of every arm and pooled over arms.

    arms: arm name -> value-log path. Pooled, a start cluster is one arm's
    start (``<arm>|<start>``), so the bootstrap resamples arm-by-start
    clusters."""
    per_arm, pooled_eps = {}, []
    for name, path in arms.items():
        eps, n_excluded, n_no_score = load_harvest(path)
        per_arm[name] = {
            "n_episodes": len(eps),
            "n_excluded": n_excluded,
            "n_no_score": n_no_score,
            **arm_summary(eps),
        }
        pooled_eps.extend((f"{name}|{s}", e, n) for s, e, n in eps)
    return {
        "rule": {
            "gap_min": GAP_MIN,
            "score_source": "visible score in the rendered value input",
            "exclude": "eliminated or control lost",
        },
        "arms": per_arm,
        "pooled_same_start_by_bucket": {
            name: same_start_bootstrap(pooled_eps, lo, hi) for name, lo, hi in BUCKETS
        },
        "pooled_per_turn": _public(per_turn_rates(pooled_eps)),
    }
