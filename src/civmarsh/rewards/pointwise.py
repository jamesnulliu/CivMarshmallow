"""Pointwise potentials on CivTelescope's reference-set ruler.

CivTelescope turns a decision into a potential by comparing it pairwise with
the reference positions of the nearest turn bucket and reading the mean
winrate off their sorted final scores (``potential.telescope_value``).  A
pointwise scorer f(position) -> number is placed on the same ruler:

    w   = fractional rank of f(decision) among f(reference positions)
    Phi = winrate_to_score(w, reference positions)

so a pointwise signal and CivTelescope share the reference positions, the map
into final-score units and the GAE; only the source of the ordering differs.

  value_head_value   f = the regression value head of
                     ``civmarsh.civtelescope.value_head`` (same backbone and
                     LoRA recipe as CivTelescope, trained to predict the
                     replayed final score of one position), served over HTTP
                     by ``scripts/serve_value_head.py``

Same signature and return contract as ``potential.telescope_value``; the
value-log line adds the raw scorer outputs (``raw_scores``).  Reference
positions are scored once per reference-set file version (its mtime), so a
refreshed reference set is rescored on first use.

Config keys (in addition to civ_value_refset, civ_value_refset_idx,
civ_value_log and civ_value_harvest of ``potential``):
  civ_value_head_urls   list of value-head endpoints (``http://host:port/value``);
                        requests are spread over them round-robin
  civ_value_head_batch  positions per request (default 16)
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import threading
import urllib.request

from .potential import (
    append_harvest,
    append_value_log,
    bucket_and_last,
    load_reference_set,
    select_references,
    winrate_to_score,
)

DEFAULT_HEAD_BATCH = 16
REQUEST_TIMEOUT = 600.0

_CYCLE_LOCK = threading.Lock()
_REF_LOCK = threading.Lock()
_REF_CACHE: dict = {}  # (refset path, mtime, scorer id) -> {reference key: score}
_URL_CYCLE: dict = {}


def fractional_rank(x, ref_scores) -> float:
    """Position of ``x`` among the reference scores, in [0, 1]: 0 at or below
    the lowest, 1 at or above the highest, linear between neighbours.  With
    all reference scores equal the scorer cannot place anything: 0.5."""
    ref = sorted(ref_scores)
    if len(ref) < 2 or ref[-1] == ref[0]:
        return 0.5
    if x <= ref[0]:
        return 0.0
    if x >= ref[-1]:
        return 1.0
    for i in range(len(ref) - 1):
        lo, hi = ref[i], ref[i + 1]
        if lo <= x <= hi:
            frac = 0.5 if hi == lo else (x - lo) / (hi - lo)
            return (i + frac) / (len(ref) - 1)
    return 1.0


def _reference_key(row) -> tuple:
    return (
        row["turn"],
        row.get("position_id"),
        row["focal"],
        row["end_score"],
        hash(row["rendering"]),
    )


def _reference_scores(path, scorer_id, score_fn) -> dict:
    """f(reference position) for every position of the reference-set file,
    computed once per (file version, scorer) and cached for the process."""
    key = (path, os.stat(path).st_mtime_ns, scorer_id)
    with _REF_LOCK:
        if key not in _REF_CACHE:
            rows = [r for bucket in load_reference_set(path).values() for r in bucket]
            scores = score_fn(
                [{"focal": r["focal"], "rendering": r["rendering"]} for r in rows]
            )
            if any(s is None for s in scores):
                raise RuntimeError(f"could not score the reference positions of {path}")
            _REF_CACHE[key] = {
                _reference_key(r): s for r, s in zip(rows, scores, strict=True)
            }
        return _REF_CACHE[key]


def values_on_ruler(decisions, args, scorer_id, score_fn):
    """Phi per decision for a pointwise ``score_fn(items) -> [score or None]``.

    Returns (values, raw scores, number of failed decisions, reference set).
    A decision whose score failed gets the winrate 0.5, as a failed
    CivTelescope call does."""
    path = args.civ_value_refset
    idx = getattr(args, "civ_value_refset_idx", None)
    refset = load_reference_set(path)
    ref_scores = _reference_scores(path, scorer_id, score_fn)
    references = [select_references(refset, d["turn"], idx) for d in decisions]
    raw = score_fn(
        [{"focal": d["focal"], "rendering": d["value_input"]} for d in decisions]
    )
    values, fails = [], 0
    for score, refs in zip(raw, references, strict=True):
        if score is None:
            fails += 1
            w = 0.5
        else:
            w = fractional_rank(score, [ref_scores[_reference_key(r)] for r in refs])
        values.append(winrate_to_score(w, refs))
    return values, raw, fails, refset


def _log(args, decisions, result, values, raw, fails, refset) -> None:
    append_value_log(
        args,
        {
            "turns": [d["turn"] for d in decisions],
            "values": [round(v, 2) for v in values],
            "raw_scores": [None if s is None else round(float(s), 4) for s in raw],
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": fails,
        },
    )
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))


# ----------------------------------------------------------------- value head


def _post_value_head(url, items, timeout=REQUEST_TIMEOUT):
    """Scores for ``items`` from one value-head replica; all None on failure."""
    req = urllib.request.Request(
        url,
        data=json.dumps({"items": items}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            scores = json.load(response)["scores"]
    except Exception:  # noqa: BLE001 - a failed request scores nothing
        return [None] * len(items)
    return scores if len(scores) == len(items) else [None] * len(items)


def value_head_scorer(args):
    """(scorer id, ``score(items)``) over the replicas in civ_value_head_urls."""
    urls = args.civ_value_head_urls
    if isinstance(urls, str):
        urls = [u for u in urls.split(",") if u]
    urls = list(urls)
    if not urls:
        raise ValueError("civ_value_head_urls is empty")
    batch = int(getattr(args, "civ_value_head_batch", DEFAULT_HEAD_BATCH))
    with _CYCLE_LOCK:
        cycle = _URL_CYCLE.setdefault(tuple(urls), itertools.cycle(urls))

    def score(items):
        out = []
        for i in range(0, len(items), batch):
            with _CYCLE_LOCK:
                url = next(cycle)
            out.extend(_post_value_head(url, items[i : i + batch]))
        return out

    return tuple(urls), score


def value_head_value(decisions, result, args) -> list[float]:
    """Phi_t = the value head's fractional rank among the reference positions,
    in final-score units."""
    scorer_id, score = value_head_scorer(args)
    values, raw, fails, refset = values_on_ruler(decisions, args, scorer_id, score)
    _log(args, decisions, result, values, raw, fails, refset)
    return values


# ----------------------------------------------------------------- diagnostic


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def reference_rank_correlation(refset_path, score_fn) -> dict:
    """Per turn bucket: Spearman correlation between a scorer and the final
    scores of the reference positions (None below three scored positions or
    with a constant side).  A log-only check of a scorer before training."""
    out = {}
    for turn, bucket in sorted(load_reference_set(refset_path).items()):
        scores = score_fn(
            [{"focal": r["focal"], "rendering": r["rendering"]} for r in bucket]
        )
        pairs = [(s, r["end_score"]) for s, r in zip(scores, bucket) if s is not None]
        if len(pairs) < 3:
            out[turn] = None
            continue
        rx = _ranks([p[0] for p in pairs])
        ry = _ranks([p[1] for p in pairs])
        if statistics.pstdev(rx) == 0 or statistics.pstdev(ry) == 0:
            out[turn] = None
        else:
            out[turn] = round(statistics.correlation(rx, ry), 3)
    return out
