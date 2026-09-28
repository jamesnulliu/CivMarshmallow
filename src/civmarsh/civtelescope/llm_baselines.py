"""Zero-shot language models on CivTelescope's pairs.

Each model answers the same pairwise prompt CivTelescope is trained on (same
fog renderings, same ruleset named), in both presentation orders, through an
OpenAI-compatible endpoint (`civmarsh.utils.api.ApiClient`). A completion
without a parsable letter is asked once more under a fresh cache tag; a pair
order that still fails is left out, and a rerun resumes from the cache. The
*warned* variant adds `TRAP_WARNING` to the prompt and changes nothing else.

Picks are written one row per (pair, order): ``{"pair_id", "order", "pick"}``.

Comparisons:

- `heldout_comparison`: every model against CivTelescope on the held-out
  subsample, with episode-clustered intervals, the paired accuracy
  difference and the paired accuracy ratio on trap pairs (2,000 draws).
- `lineup`: on the pairs every model has scored, trap accuracy with
  intervals and CivTelescope's margin over the best zero-shot row, the best
  re-taken inside every draw (1,000 draws).
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

from civmarsh.utils.stats import (
    best_baseline_bootstrap,
    content_accuracy,
    episode_cluster,
    pair_id,
    paired_bootstrap,
    ratio_bootstrap,
    winner_id,
)

from .prompt import pairwise_prompt, parse_pairwise

SUBSAMPLE_SEED = 97
RETRY_SUFFIX = "|r1"


def heldout_subsample(
    pairs: list[dict],
    *,
    n_trap: int = 300,
    n_natural: int = 300,
    seed: int = SUBSAMPLE_SEED,
) -> list[dict]:
    """A seeded subsample of `n_trap` trap and `n_natural` natural pairs."""
    trap = [p for p in pairs if p["trap"]]
    nat = [p for p in pairs if not p["trap"]]
    rng = random.Random(seed)
    rng.shuffle(trap)
    rng.shuffle(nat)
    keep = trap[:n_trap] + nat[:n_natural]
    rng.shuffle(keep)
    return keep


def ask_pairwise(client, prompt: str, tag: str) -> tuple[str | None, str]:
    """(letter, status) for one prompt: status is "ok", "parse_fail" (answered
    but no letter, twice) or "http_fail" (no answer)."""
    http_ok = False
    for t in (tag, tag + RETRY_SUFFIX):
        try:
            content = client.chat(prompt, tag=t)
        except Exception:  # noqa: BLE001, S112 -- an unanswered call
            continue
        http_ok = True
        shown = parse_pairwise(content)
        if shown is not None:
            return shown, "ok"
    return None, ("parse_fail" if http_ok else "http_fail")


def score_pairs(
    client,
    pairs: list[dict],
    positions: dict[str, dict],
    *,
    ruleset: str = "civ2civ3",
    warned: bool = False,
    workers: int = 2,
    log=None,
) -> tuple[list[dict], dict]:
    """Ask `client` about every pair in both orders. Returns (pick rows,
    counts of http and parse failures)."""
    t0 = time.time()
    jobs = [(p, o) for p in pairs for o in ("ab", "ba")]

    def one(job):
        p, o = job
        first, second = (p["a"], p["b"]) if o == "ab" else (p["b"], p["a"])
        prompt = pairwise_prompt(
            positions[first], positions[second], ruleset=ruleset, warned=warned
        )
        shown, status = ask_pairwise(client, prompt, f"{p['a']}|{p['b']}|{o}")
        if shown is None:
            return job, None, status
        return job, first if shown == "A" else second, status

    rows, n_http, n_parse = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (job, pick, status) in enumerate(ex.map(one, jobs), 1):
            p, o = job
            if status == "http_fail":
                n_http += 1
            elif status == "parse_fail":
                n_parse += 1
            else:
                rows.append({"pair_id": pair_id(p), "order": o, "pick": pick})
            if log and (i % 100 == 0 or i == len(jobs)):
                log(
                    f"{i}/{len(jobs)} (http_fail {n_http}, parse_fail {n_parse}, "
                    f"{(time.time() - t0) / i:.1f} s/call)"
                )
    return rows, {"n_calls": len(jobs), "n_http_fail": n_http, "n_parse_fail": n_parse}


def picks_by_pair(rows: list[dict]) -> dict[str, list]:
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["pair_id"], []).append(
            {"order": r["order"], "pick": r["pick"]}
        )
    return out


def summarize(pairs: list[dict], rows: list[dict], counts: dict) -> dict:
    """Trap and natural accuracy of one run, and its failure counts."""
    picks = picks_by_pair(rows)
    acc = {"natural": [0.0, 0], "trap": [0.0, 0]}
    for p in pairs:
        pk = picks.get(pair_id(p))
        if not pk:
            continue
        k = "trap" if p["trap"] else "natural"
        acc[k][0] += content_accuracy(pk, winner_id(p))
        acc[k][1] += 1
    n_ok = counts["n_calls"] - counts["n_http_fail"]
    return {
        "n_pairs": len(pairs),
        **counts,
        "parse_rate_of_answered": (
            round((n_ok - counts["n_parse_fail"]) / n_ok, 4) if n_ok else None
        ),
        "n_pairs_both_orders": sum(1 for v in picks.values() if len(v) == 2),
        **{f"acc_{k}": round(s / n, 4) if n else None for k, (s, n) in acc.items()},
        **{f"n_{k}_scored": n for k, (_, n) in acc.items()},
    }


def _acc_table(pairs, picks_by_model):
    return {
        m: {
            pair_id(p): content_accuracy(pk[pair_id(p)], winner_id(p))
            for p in pairs
            if pk.get(pair_id(p))
        }
        for m, pk in picks_by_model.items()
    }


def heldout_comparison(
    subsample: list[dict],
    full: list[dict],
    picks_by_model: dict[str, dict],
    *,
    reference: str = "civtelescope",
    cluster_of=episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
    min_pairs: int = 10,
) -> dict:
    """Every model on the subsample and on the full held-out set, and each
    model against `reference` on the subsample's trap pairs.

    Per model and slice: accuracy, interval, pair and cluster counts (slices
    with fewer than `min_pairs` scored pairs are skipped). Per contrast:
    reference minus model and reference / model, both paired."""
    acc = _acc_table(full, picks_by_model)
    table = {}
    for m in acc:
        row = {}
        for name, rows in (("sub", subsample), ("full", full)):
            for sl, keep in (("trap", True), ("natural", False)):
                rs = [p for p in rows if p["trap"] == keep and pair_id(p) in acc[m]]
                if len(rs) < min_pairs:
                    continue
                b = paired_bootstrap(
                    rs, acc[m], cluster_of=cluster_of, n_boot=n_boot, seed=seed
                )
                row[f"{name}_{sl}"] = {
                    "acc": b["point"],
                    "ci95": b["ci95"],
                    "n": len(rs),
                    "n_clusters": b["n_clusters"],
                }
        table[m] = row
    contrasts = {}
    ref = acc.get(reference, {})
    for m in acc:
        if m == reference:
            continue
        rs = [
            p
            for p in subsample
            if p["trap"] and pair_id(p) in acc[m] and pair_id(p) in ref
        ]
        if len(rs) < min_pairs:
            continue
        diff = {pair_id(p): ref[pair_id(p)] - acc[m][pair_id(p)] for p in rs}
        d = paired_bootstrap(rs, diff, cluster_of=cluster_of, n_boot=n_boot, seed=seed)
        r = ratio_bootstrap(
            rs, ref, acc[m], cluster_of=cluster_of, n_boot=n_boot, seed=seed
        )
        contrasts[m] = {
            "n_trap": len(rs),
            "diff": d["point"],
            "diff_ci95": d["ci95"],
            "ratio": r["point"],
            "ratio_ci95": r["ci95"],
        }
    return {
        "subsample": {"n": len(subsample), "trap": sum(p["trap"] for p in subsample)},
        "table": table,
        f"{reference}_vs_model_on_subsample_trap": contrasts,
    }


def lineup(
    pairs: list[dict],
    picks_by_model: dict[str, dict],
    *,
    reference: str = "civtelescope",
    cluster_of=episode_cluster,
    n_boot: int = 1000,
    seed: int = 7,
) -> dict:
    """Every model on the pairs all of them have scored: trap accuracy with
    an interval, natural accuracy, and `reference` minus the best other row
    on trap pairs with the best re-taken inside every draw."""
    models = list(picks_by_model)
    acc = _acc_table(pairs, picks_by_model)
    common = [p for p in pairs if all(pair_id(p) in acc[m] for m in models)]
    traps = [p for p in common if p["trap"]]
    natural = [p for p in common if not p["trap"]]
    baselines = set(models) - {reference}
    acc_ci, diff_ci = best_baseline_bootstrap(
        common, acc, baselines, models, cluster_of=cluster_of, n_boot=n_boot, seed=seed
    )

    def mean(m, rows):
        return (
            round(sum(acc[m][pair_id(p)] for p in rows) / len(rows), 4)
            if rows
            else None
        )

    table = {
        m: {
            "acc_trap": mean(m, traps),
            "acc_trap_ci95": acc_ci[m],
            "acc_natural": mean(m, natural),
            "n_trap_scored": sum(pair_id(p) in acc[m] for p in pairs if p["trap"]),
        }
        for m in models
    }
    out = {
        "n_pairs_common": len(common),
        "n_trap_common": len(traps),
        "n_natural_common": len(natural),
        "table": table,
    }
    if reference in models and baselines and traps:
        best = max(table[b]["acc_trap"] for b in baselines)
        out[f"{reference}_minus_best_baseline"] = {
            "point": round(table[reference]["acc_trap"] - best, 4),
            "ci95": diff_ci[reference],
            "baselines": sorted(baselines),
            "n_boot": n_boot,
        }
    return out
