"""Accuracy and cluster-bootstrap statistics for pairwise evaluations.

A pair is a dict with position ids ``a`` and ``b``, ``oracle_winner`` ("A" or
"B") and a boolean ``trap`` flag. Per-pair picks are lists of
``{"order": "ab"|"ba", "pick": <position id>}``; the content-level accuracy of a
pair is the fraction of its picks that name the oracle winner, so a model scored
in both presentation orders contributes 0, 0.5 or 1.

Pairs are not independent: positions from the same game (or rollout episode)
share outcomes. Every interval here is a pigeonhole bootstrap over clusters:
each draw resamples the clusters with replacement, and a pair is weighted by
count(cluster of a) * count(cluster of b). The cluster of a position is given
by an explicit `cluster_of` function of its id.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable

_TURN_SUFFIX = re.compile(r"_[tT]\d+$")


def pair_id(pair: dict) -> str:
    return f"{pair['a']}|{pair['b']}"


def winner_id(pair: dict) -> str:
    """Position id of the pair's oracle winner."""
    return pair["a"] if pair["oracle_winner"] == "A" else pair["b"]


def content_accuracy(picks: list[dict], winner: str) -> float:
    """Fraction of a pair's picks that name `winner`."""
    return sum(p["pick"] == winner for p in picks) / len(picks)


def episode_cluster(position_id: str) -> str:
    """The game or rollout episode of a position: its id without the trailing
    ``_T<turn>`` / ``_t<turn>`` (``g1_T10`` -> ``g1``,
    ``run_e3_t30`` -> ``run_e3``)."""
    return _TURN_SUFFIX.sub("", position_id)


def game_prefix_cluster(position_id: str) -> str:
    """The id up to its first underscore (``g1_T10`` -> ``g1``). For a
    rollout position (``<run>_e<episode>_t<turn>``) this is the whole policy
    run, a coarser cluster than `episode_cluster`; the two agree on replay-bank
    positions."""
    return position_id.split("_", 1)[0]


CLUSTER_RULES: dict[str, Callable[[str], str]] = {
    "episode": episode_cluster,
    "game-prefix": game_prefix_cluster,
}


def _clusters(pairs: list[dict], cluster_of) -> tuple[list[str], list[tuple]]:
    ends = [(cluster_of(p["a"]), cluster_of(p["b"])) for p in pairs]
    names = sorted({c for e in ends for c in e})
    return names, ends


def accuracy_ci(
    pairs: list[dict],
    acc: dict[str, float],
    *,
    cluster_of: Callable[[str], str] = episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[list[float], int]:
    """95% interval of the mean of `acc` (keyed by pair id) over `pairs`.

    Draws use ``random.Random(seed)``; the interval is the 2.5th percentile and
    the element just below the 97.5th percentile index of the sorted draws.
    Returns ([lo, hi], number of clusters)."""
    names, ends = _clusters(pairs, cluster_of)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        counts: dict[str, int] = {}
        for _ in range(len(names)):
            g = names[rng.randrange(len(names))]
            counts[g] = counts.get(g, 0) + 1
        ws = wa = 0.0
        for p, (ca, cb) in zip(pairs, ends):
            w = counts.get(ca, 0) * counts.get(cb, 0)
            if not w:
                continue
            wa += w * acc[pair_id(p)]
            ws += w
        if ws:
            draws.append(wa / ws)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return [round(lo, 4), round(hi, 4)], len(names)


def slice_readout(
    pairs: Iterable[dict],
    acc: dict[str, float],
    *,
    cluster_of: Callable[[str], str] = episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """Trap and natural accuracy of one model with `accuracy_ci` intervals.

    `pairs` may list a pair id more than once (a pair can sit in both held-out
    slices); the record listed last decides its slice, and only pairs with an
    entry in `acc` count. Keys: acc_{trap,natural}, acc_*_ci95, acc_*_n,
    acc_*_clusters."""
    by_pid = {pair_id(p): p for p in pairs}
    got = [by_pid[pid] for pid in acc if pid in by_pid]
    row = {}
    for tag, sl in (
        ("trap", [p for p in got if p["trap"]]),
        ("natural", [p for p in got if not p["trap"]]),
    ):
        if not sl:
            continue
        point = sum(acc[pair_id(p)] for p in sl) / len(sl)
        ci, n_clusters = accuracy_ci(
            sl, acc, cluster_of=cluster_of, n_boot=n_boot, seed=seed
        )
        row[f"acc_{tag}"] = round(point, 4)
        row[f"acc_{tag}_ci95"] = ci
        row[f"acc_{tag}_n"] = len(sl)
        row[f"acc_{tag}_clusters"] = n_clusters
    return row


def best_baseline_bootstrap(
    pairs: list[dict],
    acc_by_model: dict[str, dict[str, float]],
    baselines: set[str],
    models: list[str],
    *,
    cluster_of: Callable[[str], str] = episode_cluster,
    n_boot: int = 1000,
    seed: int = 7,
) -> tuple[dict, dict]:
    """Trap-slice accuracy intervals, and each non-baseline model's margin
    over the best baseline with the best re-taken inside every draw.

    Only the trap pairs of `pairs` enter. Draws use ``random.Random(seed)``; an
    interval is None when fewer than 40 draws are defined. Returns
    ({model: [lo, hi]}, {non-baseline model: [lo, hi]})."""
    traps = [p for p in pairs if p["trap"]]
    names, ends = _clusters(traps, cluster_of)
    rng = random.Random(seed)
    acc_draws = {m: [] for m in models}
    diff_draws = {m: [] for m in models if m not in baselines}
    for _ in range(n_boot):
        counts: dict[str, int] = {}
        for _ in range(len(names)):
            g = names[rng.randrange(len(names))]
            counts[g] = counts.get(g, 0) + 1
        accs = {}
        for m in models:
            ws = wa = 0.0
            for p, (ca, cb) in zip(traps, ends):
                w = counts.get(ca, 0) * counts.get(cb, 0)
                if not w:
                    continue
                wa += w * acc_by_model[m][pair_id(p)]
                ws += w
            accs[m] = wa / ws if ws else None
        best = max((accs[b] for b in baselines if accs[b] is not None), default=None)
        for m in models:
            acc_draws[m].append(accs[m])
            if m not in baselines and accs[m] is not None and best is not None:
                diff_draws[m].append(accs[m] - best)

    def ci(xs):
        xs = sorted(x for x in xs if x is not None)
        if len(xs) < 40:
            return None
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs))], 4)]

    return (
        {m: ci(v) for m, v in acc_draws.items()},
        {m: ci(v) for m, v in diff_draws.items()},
    )


def _weights_setup(pairs, cluster_of):
    import numpy as np

    names, ends = _clusters(pairs, cluster_of)
    index = {c: k for k, c in enumerate(names)}
    ga = np.array([index[a] for a, _ in ends])
    gb = np.array([index[b] for _, b in ends])
    return len(names), ga, gb


def paired_bootstrap(
    pairs: list[dict],
    values: dict[str, float],
    *,
    cluster_of: Callable[[str], str] = episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """Mean of per-pair `values` (e.g. an accuracy, or a paired accuracy
    difference between two models) with a 95% interval.

    Draws use ``numpy.random.default_rng(seed)``; the interval is the 2.5th and
    97.5th percentile of the sorted draws. Returns point, ci95, excludes_zero,
    n_pairs, n_clusters, n_boot (the number of defined draws)."""
    import numpy as np

    n_clusters, ga, gb = _weights_setup(pairs, cluster_of)
    v = np.array([values[pair_id(p)] for p in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        w = c[ga] * c[gb]
        if w.sum():
            draws.append(float((w * v).sum() / w.sum()))
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    return {
        "point": round(float(v.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_pairs": len(pairs),
        "n_clusters": n_clusters,
        "n_boot": len(draws),
    }


def ratio_bootstrap(
    pairs: list[dict],
    numerator: dict[str, float],
    denominator: dict[str, float],
    *,
    cluster_of: Callable[[str], str] = episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """Ratio of two models' mean accuracies over the same pairs, with a paired
    95% interval (ratio of weighted means per draw, same draws as
    `paired_bootstrap` for the same seed). Rounded to 3 decimals."""
    import numpy as np

    n_clusters, ga, gb = _weights_setup(pairs, cluster_of)
    a1 = np.array([numerator[pair_id(p)] for p in pairs], dtype=float)
    a2 = np.array([denominator[pair_id(p)] for p in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        w = c[ga] * c[gb]
        if w.sum() and (w * a2).sum() > 0:
            ratios.append(float((w * a1).sum() / (w * a2).sum()))
    ratios.sort()
    return {
        "point": round(float(a1.mean() / a2.mean()), 3) if a2.mean() else None,
        "ci95": (
            [
                round(ratios[int(0.025 * len(ratios))], 3),
                round(ratios[int(0.975 * len(ratios))], 3),
            ]
            if ratios
            else None
        ),
    }
