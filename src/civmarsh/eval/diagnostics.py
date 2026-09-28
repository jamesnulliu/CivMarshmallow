"""Offline diagnostics over the logs a training run writes.

* :func:`turn_spectrum` — per turn bucket, how well a per-decision value predicts
  the episode's realized terminal score: median and 90th-percentile absolute
  error and Spearman correlation. Applied to the CivTelescope potential and to
  the scoreboard value (both logged per decision in ``value_log.jsonl``).
* :func:`delta_r2` — explained variance of the terminal score that the potential
  adds beyond the current scoreboard score, from (potential, score now, score
  end) triplets aligned by turn across ``value_log.jsonl`` and
  ``value_harvest.jsonl``.
* :func:`training_slope` — the per-iteration slope of the training score:
  per start, an OLS fit of the group-mean score on the training iteration over
  the groups training actually consumed, averaged over starts.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy import stats

BUCKETS = ((10, 29), (30, 49), (50, 69), (70, 89), (90, 109), (110, 129))
N_ITERATIONS = 12
GROUPS_PER_ITERATION = 8

_DIGEST = "DIGEST "
_DECODER = json.JSONDecoder()


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_now_from_rendering(rendering: str) -> float | None:
    """The focal player's current score from a rendering's DIGEST object.

    The DIGEST line holds one JSON object; decoding it (rather than matching a
    pattern) avoids other ``score`` fields in the rendering.
    """
    k = rendering.find(_DIGEST)
    if k < 0:
        return None
    try:
        obj, _ = _DECODER.raw_decode(rendering[k + len(_DIGEST) :])
    except ValueError:
        return None
    metrics = obj.get("metrics") if isinstance(obj, dict) else None
    value = metrics.get("score") if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def read_value_log(path: str | Path) -> list[tuple[int, float, float]]:
    """(turn, value, score_end) for every logged decision of every episode."""
    rows = []
    for record in _read_jsonl(path):
        score_end = record.get("score_end")
        if score_end is None:
            continue
        for turn, value in zip(record.get("turns") or [], record.get("values") or []):
            if turn is None or value is None:
                continue
            rows.append((int(turn), float(value), float(score_end)))
    return rows


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties; None when undefined."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(stats.spearmanr(xs, ys).statistic)


def _bucket(turn: int, buckets: Sequence[tuple[int, int]]) -> str | None:
    for lo, hi in buckets:
        if lo <= turn <= hi:
            return f"t{lo}-{hi}"
    return None


def turn_spectrum(
    rows: Iterable[tuple[int, float, float]],
    buckets: Sequence[tuple[int, int]] = BUCKETS,
) -> dict[str, dict]:
    """Per turn bucket: n, median / p90 absolute error and Spearman vs score end."""
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for turn, value, score_end in rows:
        key = _bucket(turn, buckets)
        if key is not None:
            grouped[key].append((value, score_end))
    out = {}
    for lo, hi in buckets:
        key = f"t{lo}-{hi}"
        items = grouped.get(key, [])
        errors = [abs(v - s) for v, s in items]
        out[key] = {
            "n": len(items),
            "median_abs_err": float(np.percentile(errors, 50)) if errors else None,
            "p90_abs_err": float(np.percentile(errors, 90)) if errors else None,
            "spearman": spearman([v for v, _ in items], [s for _, s in items]),
        }
    return out


def value_triplets(
    value_log: str | Path, value_harvest: str | Path
) -> tuple[list[tuple[float, float, float]], dict]:
    """(potential, score now, score end) triplets aligned by turn.

    The two logs have one line per episode in the same order; within a line the
    value log covers every decision while the harvest keeps a subset, so
    triplets are matched on the decision turn. Also returns match counts.
    """
    values = _read_jsonl(value_log)
    harvest = _read_jsonl(value_harvest)
    out = []
    seen = no_value = no_score = no_end = 0
    for v, h in zip(values, harvest):
        score_end = v.get("score_end")
        by_turn = dict(zip(v.get("turns") or [], v.get("values") or []))
        for decision in h.get("decisions") or []:
            seen += 1
            if score_end is None:
                no_end += 1
                continue
            phi = by_turn.get(decision.get("turn"))
            if phi is None:
                no_value += 1
                continue
            now = score_now_from_rendering(decision.get("rendering") or "")
            if now is None:
                no_score += 1
                continue
            out.append((float(phi), now, float(score_end)))
    counts = {
        "value_log_rows": len(values),
        "value_harvest_rows": len(harvest),
        "decisions_seen": seen,
        "matched": len(out),
        "dropped_no_value": no_value,
        "dropped_no_score_now": no_score,
        "dropped_no_score_end": no_end,
    }
    return out, counts


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum()) / den if den > 0 else float("nan")


def delta_r2(triplets: Sequence[tuple[float, float, float]]) -> dict:
    """Variance of score end explained by the potential beyond score now.

    Closed form of the two-regressor linear fit from the three pairwise
    correlations (p = potential, n = score now, e = score end):

        R2(e ~ n, p) = (r_pe^2 + r_ne^2 - 2 r_pe r_ne r_pn) / (1 - r_pn^2)
        delta_R2     = R2(e ~ n, p) - r_ne^2
        partial r    = (r_pe - r_ne r_pn) / sqrt((1 - r_pn^2)(1 - r_ne^2))
    """
    phi = [t[0] for t in triplets]
    now = [t[1] for t in triplets]
    end = [t[2] for t in triplets]
    r_pe, r_ne, r_pn = _corr(phi, end), _corr(now, end), _corr(phi, now)
    denom = 1 - r_pn**2
    r2_both = (r_pe**2 + r_ne**2 - 2 * r_pe * r_ne * r_pn) / denom
    return {
        "n": len(triplets),
        "r_potential_score_end": r_pe,
        "r_score_now_score_end": r_ne,
        "r_potential_score_now": r_pn,
        "partial_potential_given_score_now": (
            (r_pe - r_ne * r_pn) / math.sqrt(denom * (1 - r_ne**2))
        ),
        "r2_score_now": r_ne**2,
        "r2_score_now_and_potential": r2_both,
        "delta_r2": r2_both - r_ne**2,
    }


def consumed_groups(
    reward_rows: Iterable[dict],
    *,
    n_iterations: int = N_ITERATIONS,
    groups_per_iteration: int = GROUPS_PER_ITERATION,
) -> list[tuple[int, str, float]]:
    """(iteration, position_id, group mean score) of the groups training used.

    A reward-log row is one group of sibling episodes from one start. Training
    consumes the groups whose episodes all finished; in time order, the first
    ``n_iterations * groups_per_iteration`` of them fill the iterations in turn
    (later ones were generated but not trained on).
    """
    kept = []
    for row in reward_rows:
        episodes = row.get("episodes") or []
        if not episodes or not all(ep.get("ok") for ep in episodes):
            continue
        pids = {ep.get("position_id") for ep in episodes}
        if len(pids) != 1 or None in pids:
            raise ValueError("a reward-log group mixes positions")
        mean = statistics.mean(float(ep["score_end"]) for ep in episodes)
        kept.append((float(row["ts"]), pids.pop(), mean))
    kept.sort(key=lambda item: item[0])
    kept = kept[: n_iterations * groups_per_iteration]
    return [
        (i // groups_per_iteration, pid, mean) for i, (_, pid, mean) in enumerate(kept)
    ]


def _ols_slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    x_mean = statistics.mean(x for x, _ in points)
    y_mean = statistics.mean(y for _, y in points)
    den = sum((x - x_mean) ** 2 for x, _ in points)
    if den == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / den


def per_start_slopes(groups: Iterable[tuple[int, str, float]]) -> dict[str, float]:
    """Per start, the OLS slope of its per-iteration mean score on the iteration.

    A start needs at least two distinct iterations spanning at least two steps.
    """
    by_start: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for iteration, pid, mean in groups:
        by_start[pid][iteration].append(mean)
    out = {}
    for pid, iterations in by_start.items():
        points = sorted((it, statistics.mean(v)) for it, v in iterations.items())
        if len(points) < 2 or points[-1][0] - points[0][0] < 2:
            continue
        slope = _ols_slope(points)
        if slope is not None:
            out[pid] = slope
    return out


def mean_se(values: Sequence[float]) -> dict:
    """Mean, standard error, and two-sided one-sample t test against zero."""
    values = list(values)
    n = len(values)
    mean = statistics.mean(values) if values else None
    if n < 2:
        return {"n": n, "mean": mean, "se": None, "t": None, "p": None}
    se = statistics.stdev(values) / math.sqrt(n)
    if se == 0:
        return {"n": n, "mean": mean, "se": 0.0, "t": None, "p": None}
    t = mean / se
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "t": t,
        "p": float(2 * stats.t.sf(abs(t), n - 1)),
    }


def training_slope(reward_log: str | Path, **kwargs) -> dict:
    """Training-score slope (points per iteration) of one run, averaged over starts."""
    slopes = per_start_slopes(consumed_groups(_read_jsonl(reward_log), **kwargs))
    return {**mean_se(list(slopes.values())), "per_start": slopes}


def paired_slope_difference(
    slopes_a: dict[str, float], slopes_b: dict[str, float]
) -> dict:
    """Mean per-start slope difference (a minus b) over the starts both runs fit."""
    common = sorted(set(slopes_a) & set(slopes_b))
    return mean_se([slopes_a[p] - slopes_b[p] for p in common])
