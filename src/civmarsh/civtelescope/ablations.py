"""Ablations of CivTelescope: what it reads, and where its labels come from.

Input ablation. A value-input rendering is a header line followed by one line
per block (``DIGEST``, ``CITY_METRICS``, ``ARMY_ROSTER``, ``TECH_STATUS``,
``GRAPH_SUMMARY``), each a JSON object; the focal player's score is
``DIGEST.metrics.score``. Variants:

- ``full``: the rendering unchanged;
- ``masked``: ``DIGEST.metrics.score`` removed, every other line byte-identical;
- ``score_only``: the header and a ``DIGEST`` holding only ``metrics.score``,
  ``metrics.turn`` and the digest ``schema``.

Label source. The same training pairs labeled two ways: by the replay oracle,
and by a zero-shot model's preference. A seeded permutation of the training
pairs is labeled in order, one presentation order per pair (drawn per pair), so
any prefix of the permutation is a seeded subset; the two training sets share
the pairs and differ only in ``oracle_winner``.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .llm_baselines import ask_pairwise
from .prompt import pairwise_prompt

VARIANTS = ("full", "masked", "score_only")
_DIGEST = "DIGEST "

PERMUTATION_SEED = 97
ORDER_SEED = 29


# ------------------------------------------------------------ input ablation


def _split_lines(rendering: str):
    lines = rendering.split("\n")
    header, rest = lines[0], lines[1:]
    digest_idx = next(i for i, line in enumerate(rest) if line.startswith(_DIGEST))
    return header, rest, digest_idx


def transform(rendering: str, variant: str) -> str:
    """The `variant` of a value-input rendering (see the module docstring)."""
    if variant == "full":
        return rendering
    header, rest, di = _split_lines(rendering)
    digest = json.loads(rest[di][len(_DIGEST) :])
    if variant == "masked":
        metrics = {k: v for k, v in digest["metrics"].items() if k != "score"}
        rest = list(rest)
        rest[di] = _DIGEST + json.dumps(
            dict(digest, metrics=metrics), separators=(",", ":"), sort_keys=True
        )
        return "\n".join([header] + rest)
    if variant == "score_only":
        metrics = {
            k: digest["metrics"][k] for k in ("score", "turn") if k in digest["metrics"]
        }
        out = {"metrics": metrics, "schema": digest.get("schema")}
        return "\n".join(
            [header, _DIGEST + json.dumps(out, separators=(",", ":"), sort_keys=True)]
        )
    raise ValueError(f"unknown variant {variant!r}")


def check_transform(rendering: str) -> None:
    """Assert the invariants of the variants on one rendering: ``masked``
    differs from ``full`` only by the score key; ``score_only`` keeps two lines
    and no metric beyond score and turn."""
    header, rest, di = _split_lines(rendering)
    masked = transform(rendering, "masked")
    _, mrest, mdi = _split_lines(masked)
    d_full = json.loads(rest[di][len(_DIGEST) :])
    d_mask = json.loads(mrest[mdi][len(_DIGEST) :])
    assert set(d_full["metrics"]) - set(d_mask["metrics"]) == {"score"}
    assert {k: v for k, v in d_full["metrics"].items() if k != "score"} == d_mask[
        "metrics"
    ]
    assert rest[:di] == mrest[:mdi] and rest[di + 1 :] == mrest[mdi + 1 :]
    only = transform(rendering, "score_only")
    lines = only.split("\n")
    assert len(lines) == 2 and lines[0] == header
    assert set(json.loads(lines[1][len(_DIGEST) :])["metrics"]) <= {"score", "turn"}


# ------------------------------------------------------------- label source


def labeling_plan(
    train_pairs: list[dict],
    *,
    permutation_seed: int = PERMUTATION_SEED,
    order_seed: int = ORDER_SEED,
) -> list[dict]:
    """The training pairs in labeling order, each with its index ``idx`` in
    `train_pairs` and the presentation ``order`` the model sees."""
    perm = list(range(len(train_pairs)))
    random.Random(permutation_seed).shuffle(perm)
    rng = random.Random(order_seed)
    return [dict(train_pairs[i], idx=i, order=rng.choice(("ab", "ba"))) for i in perm]


def label_preferences(
    client,
    plan: list[dict],
    positions: dict[str, dict],
    out_path,
    *,
    ruleset: str = "civ2civ3",
    workers: int = 2,
    deadline: float | None = None,
    log=print,
) -> dict:
    """Ask `client` which position of each planned pair is better (one order
    per pair) and append successful labels to `out_path` (JSONL). Pairs already
    in `out_path` are skipped; no new pair is submitted after `deadline`
    (a Unix time). A label row: idx, a, b, order, status, pick, winner (the
    model's "A"/"B" in stored pair orientation), oracle_winner, agree, trap,
    pool."""
    from pathlib import Path

    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}
    todo = [
        p
        for p in plan
        if p["idx"] not in done and p["a"] in positions and p["b"] in positions
    ]
    log(f"{len(plan)} planned, {len(done)} labeled, {len(todo)} to label")

    def one(p):
        first, second = (p["a"], p["b"]) if p["order"] == "ab" else (p["b"], p["a"])
        prompt = pairwise_prompt(positions[first], positions[second], ruleset=ruleset)
        shown, status = ask_pairwise(client, prompt, f"{p['a']}|{p['b']}|{p['order']}")
        pick = None if shown is None else (first if shown == "A" else second)
        winner = None if pick is None else ("A" if pick == p["a"] else "B")
        return {
            "idx": p["idx"],
            "a": p["a"],
            "b": p["b"],
            "order": p["order"],
            "status": status,
            "pick": pick,
            "winner": winner,
            "oracle_winner": p["oracle_winner"],
            "agree": (winner == p["oracle_winner"]) if winner else None,
            "trap": p["trap"],
            "pool": p.get("pool"),
        }

    t0 = time.time()
    n_ok = n_fail = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex, out_path.open("a") as f:
        inflight = []
        it = iter(todo)
        stopped = False
        while inflight or not stopped:
            while not stopped and len(inflight) < workers * 2:
                if deadline and time.time() > deadline:
                    log("deadline reached; no new submissions")
                    stopped = True
                    break
                p = next(it, None)
                if p is None:
                    stopped = True
                    break
                inflight.append(ex.submit(one, p))
            if not inflight:
                break
            row = inflight.pop(0).result()
            if row["status"] == "ok":
                n_ok += 1
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
            else:
                n_fail += 1
            if (n_ok + n_fail) % 50 == 0:
                rate = (time.time() - t0) / max(n_ok + n_fail, 1)
                log(f"{n_ok} ok / {n_fail} failed of {len(todo)} ({rate:.1f} s/pair)")
    return {"new_ok": n_ok, "failed": n_fail}


def label_source_pairs(
    train_pairs: list[dict],
    labels: list[dict],
    *,
    n: int | None = None,
    permutation_seed: int = PERMUTATION_SEED,
) -> tuple[list[dict], list[dict], dict]:
    """(replay-labeled pairs, preference-labeled pairs, stats) over the first
    `n` successfully labeled pairs in permutation order (all when `n` is
    None). Both lists hold the same pairs in the same order."""
    perm = list(range(len(train_pairs)))
    random.Random(permutation_seed).shuffle(perm)
    by_idx = {r["idx"]: r for r in labels if r["status"] == "ok"}
    chosen = [i for i in perm if i in by_idx]
    if n:
        chosen = chosen[:n]
    replay, preference = [], []
    for i in chosen:
        base = dict(train_pairs[i])
        replay.append(base)
        preference.append({**base, "oracle_winner": by_idx[i]["winner"]})
    agree = [by_idx[i]["agree"] for i in chosen]
    agree_trap = [by_idx[i]["agree"] for i in chosen if train_pairs[i]["trap"]]
    stats = {
        "n_pairs": len(chosen),
        "n_trap": len(agree_trap),
        "label_agreement_all": round(sum(agree) / len(agree), 4) if agree else None,
        "label_agreement_trap": (
            round(sum(agree_trap) / len(agree_trap), 4) if agree_trap else None
        ),
        "by_pool": dict(Counter(train_pairs[i].get("pool") for i in chosen)),
        "orders": dict(Counter(by_idx[i]["order"] for i in chosen)),
    }
    return replay, preference, stats
