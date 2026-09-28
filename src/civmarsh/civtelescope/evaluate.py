"""Scoring pairs with CivTelescope (and its backbone) and the readouts.

A pair is scored in both presentation orders; the pick of one order is the
shown position whose letter has the larger logit after ``"ANSWER:"``. The
backbone row is the same model with the adapters switched off, so both rows see
identical inputs. Picks are stored as raw rows
``{"model", "pair_id", "picks": [{"order", "d", "pick"}]}`` (``d`` is
logit(A) - logit(B) as shown), which every readout below consumes.

Readouts:

- `readout`: per-model trap and natural accuracy with cluster-bootstrap
  intervals (2,000 draws).
- `transfer_report`: accuracy table, the paired trap-accuracy difference
  CivTelescope minus backbone (10,000 draws), and trap accuracy by turn bucket.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from civmarsh.utils.io import iter_jsonl
from civmarsh.utils.stats import (
    content_accuracy,
    episode_cluster,
    pair_id,
    paired_bootstrap,
    slice_readout,
    winner_id,
)

from .data import turn_bucket
from .lora import adapters_disabled, inject_lora, load_adapter_state
from .prompt import PAIRWISE_PROMPT_TMPL
from .train import answer_logit_diff, batches, build_examples

MODEL_LABEL = "civtelescope"
BACKBONE_LABEL = "backbone"
EVAL_BATCH = 8
EVAL_TOKEN_BUDGET = 60000


def load_model(base_model: str, adapters_path=None, *, device="cuda:0"):
    """(model, tokenizer, adapters): the bf16 base with injected adapters,
    loaded from `adapters_path` when given (else they stay untrained and only
    the backbone row is meaningful)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    model.config.use_cache = False
    loras = inject_lora(model)
    if adapters_path:
        load_adapter_state(loras, torch.load(adapters_path, map_location="cpu"))
    return model, tok, loras


def score_examples(
    model,
    examples,
    tok,
    device,
    *,
    batch=EVAL_BATCH,
    token_budget=EVAL_TOKEN_BUDGET,
    disable=None,
    log=None,
):
    """{pair_id: [{"order", "d", "pick"}]}; with `disable` (the adapters),
    the backbone is scored."""
    import torch

    picks = {}
    ctx = adapters_disabled(disable) if disable else nullcontext()
    done = 0
    with torch.no_grad(), ctx:
        for ids, mask, chunk in batches(
            examples, batch, tok, device, token_budget=token_budget
        ):
            d, _ = answer_logit_diff(model, ids, mask, chunk)
            for e, dv in zip(chunk, d.tolist()):
                a_id, b_id = e["pair_id"].split("|")
                shown_a = a_id if e["order"] == "ab" else b_id
                pick = shown_a if dv > 0 else (b_id if shown_a == a_id else a_id)
                picks.setdefault(e["pair_id"], []).append(
                    {"order": e["order"], "d": dv, "pick": pick}
                )
            done += len(chunk)
            if log and done % 500 < len(chunk):
                log(f"scored {done}/{len(examples)} examples")
    return picks


def score_pairs(
    base_model: str,
    pairs: list[dict],
    positions: dict[str, dict],
    *,
    adapters_path=None,
    ruleset: str = "civ2civ3",
    device="cuda:0",
    batch=EVAL_BATCH,
    token_budget=EVAL_TOKEN_BUDGET,
    log=None,
) -> dict[str, dict]:
    """Picks of CivTelescope (when `adapters_path` is given) and of its
    backbone for every pair, keyed by model label."""
    model, tok, loras = load_model(base_model, adapters_path, device=device)
    examples = build_examples(
        pairs,
        positions,
        tok,
        prompt_template=PAIRWISE_PROMPT_TMPL.format(ruleset=ruleset),
    )
    if log:
        log(f"{len(examples)} examples (both orders)")
    raw = {}
    if adapters_path:
        raw[MODEL_LABEL] = score_examples(
            model,
            examples,
            tok,
            device,
            batch=batch,
            token_budget=token_budget,
            log=log,
        )
    raw[BACKBONE_LABEL] = score_examples(
        model,
        examples,
        tok,
        device,
        batch=batch,
        token_budget=token_budget,
        disable=loras,
        log=log,
    )
    return raw


def shard(pairs: list[dict], spec: str) -> list[dict]:
    """The i-th of n interleaved shards (``spec`` = "i/n") of the pairs in
    (a, b) order."""
    i, n = map(int, spec.split("/"))
    return sorted(pairs, key=lambda p: (p["a"], p["b"]))[i::n]


# ------------------------------------------------------------------ raw picks


def raw_rows(raw: dict[str, dict]):
    for label, picks in raw.items():
        for pid, pk in picks.items():
            yield {"model": label, "pair_id": pid, "picks": pk}


def read_raw(paths) -> dict[str, dict[str, list]]:
    """Merge raw pick files (e.g. shards) into {model: {pair_id: picks}}."""
    raw: dict[str, dict[str, list]] = {}
    for path in paths:
        for r in iter_jsonl(path):
            raw.setdefault(r["model"], {}).setdefault(r["pair_id"], []).extend(
                r["picks"]
            )
    return raw


def read_order_picks(path) -> dict[str, list]:
    """Picks written one row per (pair, order), as the zero-shot runs write
    them: {pair_id: [{"order", "pick"}]}."""
    out: dict[str, list] = {}
    for r in iter_jsonl(path):
        out.setdefault(r["pair_id"], []).append(
            {"order": r["order"], "pick": r["pick"]}
        )
    return out


def accuracies(pairs: list[dict], picks: dict[str, list]) -> dict[str, float]:
    """Content-level accuracy per pair id for the pairs that have picks."""
    return {
        pair_id(p): content_accuracy(picks[pair_id(p)], winner_id(p))
        for p in pairs
        if picks.get(pair_id(p))
    }


# ------------------------------------------------------------------ readouts


def readout(
    pairs: list[dict],
    raw: dict[str, dict],
    *,
    cluster_of=episode_cluster,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict[str, dict]:
    """`slice_readout` for every model in `raw` over `pairs`."""
    by_pid = {pair_id(p): p for p in pairs}
    out = {}
    for label, picks in sorted(raw.items()):
        acc = {
            pid: content_accuracy(v, winner_id(by_pid[pid]))
            for pid, v in picks.items()
            if pid in by_pid and v
        }
        out[label] = slice_readout(
            pairs, acc, cluster_of=cluster_of, n_boot=n_boot, seed=seed
        )
    return out


def _mean(acc, rows):
    xs = [acc[pair_id(p)] for p in rows if pair_id(p) in acc]
    return round(sum(xs) / len(xs), 4) if xs else None


def transfer_report(
    pairs: list[dict],
    raw: dict[str, dict],
    *,
    reference: str = MODEL_LABEL,
    backbone: str = BACKBONE_LABEL,
    cluster_of=episode_cluster,
    n_boot: int = 10000,
    seed: int = 7,
) -> dict:
    """Accuracy per model on all / trap / non-trap pairs, the paired
    difference `reference` - `backbone` per slice, and trap accuracy by turn
    bucket. Both rows must have picks for every pair."""
    acc = {label: accuracies(pairs, picks) for label, picks in raw.items()}
    for label in (reference, backbone):
        missing = [pair_id(p) for p in pairs if pair_id(p) not in acc.get(label, {})]
        if missing:
            raise ValueError(f"{label}: {len(missing)} pairs without picks")
    traps = [p for p in pairs if p["trap"]]
    natural = [p for p in pairs if not p["trap"]]
    table = {
        label: {
            "acc_all": _mean(acc[label], pairs),
            "acc_trap": _mean(acc[label], traps),
            "acc_natural": _mean(acc[label], natural),
            "n_trap_scored": sum(pair_id(p) in acc[label] for p in traps),
        }
        for label in sorted(acc)
    }
    diff = {pid: acc[reference][pid] - acc[backbone][pid] for pid in acc[reference]}
    paired = {
        name: paired_bootstrap(
            rows, diff, cluster_of=cluster_of, n_boot=n_boot, seed=seed
        )
        for name, rows in (("trap", traps), ("natural", natural), ("all", pairs))
        if rows
    }
    buckets = {}
    for b in sorted({turn_bucket(p) for p in traps}):
        rows = [p for p in traps if turn_bucket(p) == b]
        buckets[b] = {
            "n": len(rows),
            **{lab: _mean(acc[lab], rows) for lab in sorted(acc)},
        }
    return {
        "n_pairs": len(pairs),
        "n_trap": len(traps),
        "n_natural": len(natural),
        "table": table,
        f"{reference}_minus_{backbone}": paired,
        "trap_acc_by_turn_bucket": buckets,
    }


def write_raw(path, raw: dict[str, dict]) -> Path:
    """Write raw pick rows atomically (a partial file never has the final
    name)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in raw_rows(raw):
            f.write(json.dumps(row) + "\n")
    tmp.rename(path)
    return path
