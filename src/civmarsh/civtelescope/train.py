"""Supervised training of CivTelescope on replay-oracle pairs.

Every pair becomes two examples, one per presentation order. An example is the
chat-formatted pairwise prompt followed by ``"ANSWER:"``; the loss is the
cross-entropy of the next token toward the letter of the oracle winner as
shown (" A" or " B"). Only the answer position is projected through the LM
head (``logits_to_keep=1``).

Recipe: LoRA r16 / alpha 32 on q_proj and v_proj over a bf16 base, AdamW
(lr 1e-4, default weight decay), gradient clipping at 1.0, micro-batches of up
to 4 examples capped at 3,500 padded tokens, 4 micro-batches per optimizer
step, a cosine schedule over the nominal step count
ceil(examples / batch / accum) * epochs, 2 epochs, seed 47, gradient
checkpointing. The held-out slices are scored every `eval_interval` optimizer
steps, with an adapter snapshot per evaluation.

torch and transformers are imported inside the functions that need them.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from .lora import adapter_state, adapters_disabled, inject_lora
from .prompt import ANSWER_PREFIX, PAIRWISE_PROMPT, fill_pairwise


@dataclass
class TrainConfig:
    base_model: str
    epochs: int = 2
    batch: int = 4
    accum: int = 4
    lr: float = 1e-4
    lora_r: int = 16
    seed: int = 47
    token_budget: int = 3500
    eval_interval: int | None = 800
    save_every_eval: bool = True
    device: str = "cuda:0"


def _logger(prefix: str):
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{prefix} +{time.time() - t0:8.1f}s] {msg}", flush=True)

    return log


# ------------------------------------------------------------------ examples


def build_examples(pairs, positions, tok, prompt_template: str = PAIRWISE_PROMPT):
    """One example per (pair, presentation order).

    `positions` maps position id to a record with focal_player and rendering;
    `prompt_template` is the pairwise template with its ruleset filled in. An
    example holds the prompt token ids ending right before the answer letter,
    the letter token ids, and ``sign`` = +1 when the shown winner is A."""
    base = tok(ANSWER_PREFIX, add_special_tokens=False).input_ids
    ida = tok(f"{ANSWER_PREFIX} A", add_special_tokens=False).input_ids
    idb = tok(f"{ANSWER_PREFIX} B", add_special_tokens=False).input_ids
    if ida[: len(base)] != base or idb[: len(base)] != base:
        raise ValueError("tokenizer splits the answer prefix differently with a letter")
    if len(ida) != len(base) + 1 or len(idb) != len(base) + 1:
        raise ValueError("answer letter must be a single token")
    tok_a, tok_b = ida[-1], idb[-1]
    out = []
    for p in pairs:
        for order in ("ab", "ba"):
            a, b = p["a"], p["b"]
            if order == "ba":
                a, b = b, a
            prompt = fill_pairwise(prompt_template, positions[a], positions[b])
            text = (
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + ANSWER_PREFIX
            )
            ids = tok(text, add_special_tokens=False).input_ids
            winner = p["oracle_winner"]
            shown = winner if order == "ab" else ("A" if winner == "B" else "B")
            out.append(
                {
                    "ids": ids,
                    "sign": 1.0 if shown == "A" else -1.0,
                    "pair_id": f"{p['a']}|{p['b']}",
                    "order": order,
                    "trap": bool(p.get("trap", False)),
                    "tokA": tok_a,
                    "tokB": tok_b,
                }
            )
    return out


def batches(examples, bs, tok, device, shuffle_rng=None, token_budget=20000):
    """Left-padded micro-batches of at most `bs` examples whose padded size
    (longest example x count) stays within `token_budget`; a single example
    longer than the budget forms its own batch. Yields (ids, mask, chunk)."""
    import torch

    exs = examples[:]
    if shuffle_rng:
        shuffle_rng.shuffle(exs)
    pad = tok.pad_token_id or tok.eos_token_id
    i = 0
    while i < len(exs):
        chunk, maxlen = [], 0
        while i < len(exs) and len(chunk) < bs:
            n = len(exs[i]["ids"])
            if chunk and max(maxlen, n) * (len(chunk) + 1) > token_budget:
                break
            chunk.append(exs[i])
            maxlen = max(maxlen, n)
            i += 1
        ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, e in enumerate(chunk):
            n = len(e["ids"])
            # left padding keeps every example's last token in the last column
            ids[j, maxlen - n :] = torch.tensor(e["ids"])
            mask[j, maxlen - n :] = 1
        yield ids.to(device), mask.to(device), chunk


def answer_logit_diff(model, ids, mask, chunk):
    """(logit(A) - logit(B) at the final position, final-position logits)."""
    import torch

    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits
    final = out[:, -1, :]
    rows = torch.arange(ids.shape[0], device=ids.device)
    tok_a = torch.tensor([e["tokA"] for e in chunk], device=ids.device)
    tok_b = torch.tensor([e["tokB"] for e in chunk], device=ids.device)
    return final[rows, tok_a] - final[rows, tok_b], final


# ---------------------------------------------------------------- evaluation


def run_eval(model, loras, examples, tok, device, bs, token_budget):
    """Held-out metrics with the adapters on ("trained") and off ("backbone").

    Accuracy is content-level (both orders averaged per pair); a pair counts
    toward the slice of its first example. Also reports the fraction of picks
    of the shown A (a position-bias alarm), the fraction of pairs answered the
    same way in both orders, and the mean logistic loss of the letter margin."""
    import torch
    import torch.nn.functional as F

    def pass_once(disabled):
        per_pair = {}
        shown_a = tot = 0
        loss_sum = 0.0
        ctx = adapters_disabled(loras) if disabled else nullcontext()
        with torch.no_grad(), ctx:
            for ids, mask, chunk in batches(
                examples, bs, tok, device, token_budget=token_budget
            ):
                d, _ = answer_logit_diff(model, ids, mask, chunk)
                for e, dv in zip(chunk, d.tolist()):
                    correct = (dv > 0) == (e["sign"] > 0)
                    shown_a += dv > 0
                    tot += 1
                    loss_sum += float(F.softplus(torch.tensor(-e["sign"] * dv)))
                    per_pair.setdefault(e["pair_id"], []).append((correct, e["trap"]))
        nat = [sum(c for c, _ in v) / len(v) for v in per_pair.values() if not v[0][1]]
        trap = [sum(c for c, _ in v) / len(v) for v in per_pair.values() if v[0][1]]
        both = [v for v in per_pair.values() if len(v) == 2]
        agree = [v[0][0] == v[1][0] for v in both]
        return {
            "acc_natural": round(sum(nat) / len(nat), 4) if nat else None,
            "acc_trap": round(sum(trap) / len(trap), 4) if trap else None,
            "n_natural": len(nat),
            "n_trap": len(trap),
            "pick_shown_a_frac": round(shown_a / tot, 4) if tot else None,
            "order_agreement": round(sum(agree) / len(agree), 4) if agree else None,
            "eval_loss": round(loss_sum / tot, 4) if tot else None,
        }

    return {"trained": pass_once(False), "backbone": pass_once(True) if loras else None}


# ------------------------------------------------------------------ training


def train_civtelescope(
    cfg: TrainConfig,
    train_pairs: list[dict],
    positions: dict[str, dict],
    out_dir,
    *,
    eval_pairs: list[dict] | None = None,
    monitor_path=None,
    log=None,
) -> Path:
    """Train adapters on `train_pairs`; returns the final checkpoint path.

    `eval_pairs` (the held-out slices, each pair flagged ``trap``) are scored
    every `cfg.eval_interval` optimizer steps and after every epoch; interval
    evaluations are appended to `monitor_path` and, with
    `cfg.save_every_eval`, saved as ``lora_step<step>_e<epoch>.pt``. The final
    adapters go to ``lora.pt`` and the loss/eval history to ``history.json``."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log = log or _logger("train")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    log(f"loading {cfg.base_model} on {cfg.device}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map=cfg.device
    )
    model.gradient_checkpointing_enable()
    # With every base parameter frozen, checkpointed segments see inputs that
    # need no grad and keep full activations; making the embedding output
    # require grad re-enables recomputation.
    model.enable_input_require_grads()
    model.config.use_cache = False
    loras = inject_lora(model, r=cfg.lora_r)
    params = [p for m in loras for p in (m.A, m.B)]
    log(f"LoRA on {len(loras)} linears (r={cfg.lora_r})")

    train_ex = build_examples(train_pairs, positions, tok)
    log(f"train: {len(train_pairs)} pairs -> {len(train_ex)} examples")
    eval_ex = build_examples(eval_pairs, positions, tok) if eval_pairs else None
    if eval_ex:
        n_trap = sum(bool(p.get("trap")) for p in eval_pairs)
        log(
            f"eval: {len(eval_pairs) - n_trap} natural + {n_trap} trap pairs "
            f"-> {len(eval_ex)} examples"
        )

    opt = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_ex) / cfg.batch / cfg.accum)
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    log(f"epochs={cfg.epochs} nominal steps={total_steps}")

    def evaluate():
        model.eval()
        ev = run_eval(
            model, loras, eval_ex, tok, cfg.device, cfg.batch, cfg.token_budget
        )
        model.train()
        return ev

    hist = []
    micro = 0
    for epoch in range(cfg.epochs):
        for ids, mask, chunk in batches(
            train_ex, cfg.batch, tok, cfg.device, rng, token_budget=cfg.token_budget
        ):
            sign = torch.tensor([e["sign"] for e in chunk], device=cfg.device)
            d, final = answer_logit_diff(model, ids, mask, chunk)
            tgt = torch.tensor(
                [e["tokA"] if e["sign"] > 0 else e["tokB"] for e in chunk],
                device=cfg.device,
            )
            loss = F.cross_entropy(final.float(), tgt)
            (loss / cfg.accum).backward()
            micro += 1
            if micro % cfg.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = micro // cfg.accum
            if step % 20 == 0:
                acc = ((d > 0) == (sign > 0)).float().mean().item()
                log(
                    f"e{epoch} step {step}/{total_steps} "
                    f"loss={loss.item():.4f} batch_acc={acc:.3f}"
                )
                hist.append({"step": step, "loss": loss.item(), "batch_acc": acc})
            if cfg.eval_interval and eval_ex and step % cfg.eval_interval == 0:
                ev = evaluate()
                log(f"e{epoch} step {step} eval: {ev}")
                hist.append({"epoch": epoch, "step": step, "eval": ev})
                if monitor_path:
                    with open(monitor_path, "a") as mf:
                        row = {"epoch": epoch, "step": step, "train_loss": loss.item()}
                        mf.write(json.dumps({**row, **ev["trained"]}) + "\n")
                if cfg.save_every_eval:
                    torch.save(
                        adapter_state(loras), out / f"lora_step{step}_e{epoch}.pt"
                    )
        if eval_ex:
            ev = evaluate()
            log(f"epoch {epoch} eval: {ev}")
            hist.append({"epoch": epoch, "eval": ev})

    final_path = out / "lora.pt"
    torch.save(adapter_state(loras), final_path)
    (out / "history.json").write_text(
        json.dumps({"config": asdict(cfg), "history": hist}, indent=1)
    )
    log(f"saved -> {out}")
    return final_path
