"""A scalar regression value head on CivTelescope's backbone, scored on the
same pairs.

The head reads one position at a time, through the single-position half of the
pairwise prompt, and regresses the replay-oracle mean end score: the
last-token hidden state goes through ``Linear(hidden, 1)``. Targets are
standardised by the training mean and standard deviation; the loss is Huber
(delta 1). The backbone gets LoRA r16 / alpha 32 on q_proj and v_proj as in
CivTelescope's recipe; AdamW with a constant learning rate (1e-4 for both the
adapters and the head), gradient clipping at 1.0, micro-batches of up to 4
positions within 3,500 padded tokens, 4 micro-batches per optimizer step,
2 epochs, seed 47. The saved model is the one after the last epoch.

For a pair, the pick is the position with the higher prediction; a head has no
presentation order, so a pair gets one pick. Picks are written in the raw
format of `civmarsh.civtelescope.evaluate`.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from civmarsh.utils.stats import episode_cluster, pair_id

from .lora import adapter_state, inject_lora
from .train import batches

# The single-position counterpart of the pairwise prompt: same framing, one
# position, a score question. Model-visible.
VALUE_PROMPT_TMPL = (
    "You will see one position from a game of Freeciv ({ruleset} ruleset). Evaluate the "
    "named player's LONG-RUN prospects over the next several dozen turns — not just the "
    "current score.\n\n"
    "=== Position (evaluate player '{{focal}}') ===\n{{rendering}}\n\n"
    "Question: what will player '{{focal}}' final score be at the end of the game?\n\n"
    'Reply with exactly one line: "SCORE: <number>".\n'
)
SCORE_PREFIX = "SCORE:"
LABEL = "value_head"


@dataclass
class ValueHeadConfig:
    base_model: str
    epochs: int = 2
    batch: int = 4
    accum: int = 4
    lr: float = 1e-4
    head_lr: float = 1e-4
    lora_r: int = 16
    seed: int = 47
    token_budget: int = 3500
    eval_interval: int = 160  # micro-batches between held-out evaluations
    device: str = "cuda:0"


def value_prompt(rec: dict, *, ruleset: str = "civ2civ3") -> str:
    return VALUE_PROMPT_TMPL.format(ruleset=ruleset).format(
        focal=rec["focal_player"], rendering=rec["rendering"]
    )


# ---------------------------------------------------------------------- data


def value_head_split(
    positions: dict[str, dict],
    labels: dict[str, float],
    heldout_pairs: list[dict],
    *,
    pool: str = "replay",
    game_of=episode_cluster,
):
    """Training positions, evaluation positions and evaluation pairs.

    Evaluation pairs are the `pool` pairs of the held-out set (their slice is
    the held-out file they come from). Training positions are every labeled
    position whose game appears in no evaluation pair, so no evaluation game
    leaks. Returns (train rows, eval rows, eval pairs, stats); a row is
    {position_id, focal_player, rendering, mean_end}."""
    eval_pairs = [p for p in heldout_pairs if p.get("pool") == pool]
    eval_games = {game_of(p[k]) for p in eval_pairs for k in ("a", "b")}
    labeled = sorted(pid for pid in positions if pid in labels)
    train = [
        {
            "position_id": pid,
            "focal_player": positions[pid]["focal_player"],
            "rendering": positions[pid]["rendering"],
            "mean_end": labels[pid],
        }
        for pid in labeled
        if game_of(pid) not in eval_games
    ]
    need = {p[k] for p in eval_pairs for k in ("a", "b")}
    missing = sorted(need - set(labeled))
    if missing:
        raise ValueError(
            f"{len(missing)} evaluation positions lack a label: {missing[:5]}"
        )
    ev = [
        {
            "position_id": pid,
            "focal_player": positions[pid]["focal_player"],
            "rendering": positions[pid]["rendering"],
            "mean_end": labels[pid],
        }
        for pid in sorted(need)
    ]
    train_games = {game_of(r["position_id"]) for r in train}
    mu, sd = target_stats(train)
    stats = {
        "n_train_positions": len(train),
        "n_train_games": len(train_games),
        "n_eval_games": len(eval_games),
        "game_overlap": sorted(train_games & eval_games),
        "n_eval_pairs": len(eval_pairs),
        "n_eval_trap": sum(1 for p in eval_pairs if p["trap"]),
        "n_eval_natural": sum(1 for p in eval_pairs if not p["trap"]),
        "n_eval_positions": len(ev),
        "target_mean": mu,
        "target_sd": sd,
    }
    return train, ev, eval_pairs, stats


def target_stats(rows: list[dict]) -> tuple[float, float]:
    """Mean and sample standard deviation of the training targets."""
    ys = [r["mean_end"] for r in rows]
    mu = sum(ys) / len(ys)
    sd = (sum((y - mu) ** 2 for y in ys) / (len(ys) - 1)) ** 0.5
    return mu, sd


def build_examples(rows, tok, *, ruleset: str = "civ2civ3"):
    out = []
    for r in rows:
        text = (
            tok.apply_chat_template(
                [{"role": "user", "content": value_prompt(r, ruleset=ruleset)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            + SCORE_PREFIX
        )
        out.append(
            {
                "ids": tok(text, add_special_tokens=False).input_ids,
                "position_id": r["position_id"],
                "y": float(r["mean_end"]),
            }
        )
    return out


# --------------------------------------------------------------------- model


def predict(model, head, ids, mask):
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1, :]
    return head(h.float()).squeeze(-1)


def predict_positions(
    model, head, examples, tok, device, *, batch=8, token_budget=20000
):
    """{position_id: prediction in standardised target units}."""
    import torch

    preds = {}
    with torch.no_grad():
        for ids, mask, chunk in batches(
            examples, batch, tok, device, token_budget=token_budget
        ):
            for e, v in zip(chunk, predict(model, head, ids, mask).tolist()):
                preds[e["position_id"]] = v
    return preds


def pair_accuracy(pairs: list[dict], preds: dict[str, float]) -> dict:
    """Trap and natural accuracy of higher-prediction picks."""
    res = {}
    for tag, sub in (
        ("trap", [p for p in pairs if p["trap"]]),
        ("natural", [p for p in pairs if not p["trap"]]),
    ):
        ok = 0
        for p in sub:
            win = p["a"] if p["oracle_winner"] == "A" else p["b"]
            lose = p["b"] if p["oracle_winner"] == "A" else p["a"]
            ok += preds[win] > preds[lose]
        res[f"acc_{tag}"] = round(ok / len(sub), 4) if sub else None
        res[f"n_{tag}"] = len(sub)
    return res


def pair_picks(pairs: list[dict], preds: dict[str, float]) -> dict[str, list]:
    """Raw picks: one per pair, the position with the higher prediction."""
    return {
        pair_id(p): [
            {
                "order": "ab",
                "d": preds[p["a"]] - preds[p["b"]],
                "pick": p["a"] if preds[p["a"]] > preds[p["b"]] else p["b"],
            }
        ]
        for p in pairs
    }


def pearson(xs, ys) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return (
        sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den if den else float("nan")
    )


def _new_head(model, device):
    from torch import nn

    head = nn.Linear(model.config.hidden_size, 1).to(device).float()
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, std=0.01)
    return head


def train_value_head(
    cfg: ValueHeadConfig,
    train_rows: list[dict],
    eval_rows: list[dict],
    eval_pairs: list[dict],
    out_path,
    *,
    ruleset: str = "civ2civ3",
    monitor_path=None,
    log=print,
) -> dict:
    """Train the adapters and the head; save ``{"lora", "head", "target_mu",
    "target_sd", "seed", "steps", "config"}`` to `out_path`. Held-out pair
    accuracy is logged every `cfg.eval_interval` micro-batches (monitoring
    only). Returns the final evaluation row and predictions."""
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    mu, sd = target_stats(train_rows)

    log(f"loading {cfg.base_model} on {cfg.device}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map=cfg.device
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    loras = inject_lora(model, r=cfg.lora_r)
    head = _new_head(model, cfg.device)
    lora_params = [p for m in loras for p in (m.A, m.B)]
    params = lora_params + list(head.parameters())

    train_ex = build_examples(train_rows, tok, ruleset=ruleset)
    eval_ex = build_examples(eval_rows, tok, ruleset=ruleset)
    log(
        f"train {len(train_ex)} positions, eval {len(eval_ex)} positions / {len(eval_pairs)} pairs"
    )

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr},
            {"params": list(head.parameters()), "lr": cfg.head_lr},
        ]
    )
    loss_fn = nn.HuberLoss(delta=1.0)
    monitor = open(monitor_path, "w") if monitor_path else None  # noqa: SIM115

    def evaluate():
        model.eval()
        preds = predict_positions(
            model,
            head,
            eval_ex,
            tok,
            cfg.device,
            batch=cfg.batch,
            token_budget=cfg.token_budget,
        )
        model.train()
        return pair_accuracy(eval_pairs, preds), preds

    t0 = time.time()
    step = 0
    for ep in range(cfg.epochs):
        for ids, mask, chunk in batches(
            train_ex,
            cfg.batch,
            tok,
            cfg.device,
            shuffle_rng=rng,
            token_budget=cfg.token_budget,
        ):
            y = torch.tensor(
                [(e["y"] - mu) / sd for e in chunk],
                device=cfg.device,
                dtype=torch.float32,
            )
            loss = loss_fn(predict(model, head, ids, mask), y) / cfg.accum
            loss.backward()
            step += 1
            if step % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 40 == 0:
                log(f"e{ep} step {step} loss={loss.item() * cfg.accum:.4f}")
            if cfg.eval_interval and step % cfg.eval_interval == 0:
                res, _ = evaluate()
                row = {
                    "epoch": ep,
                    "step": step,
                    **res,
                    "elapsed_s": round(time.time() - t0),
                }
                log(json.dumps(row))
                if monitor:
                    monitor.write(json.dumps(row) + "\n")
                    monitor.flush()
    res, preds = evaluate()
    row = {
        "epoch": cfg.epochs - 1,
        "step": step,
        "final": True,
        **res,
        "elapsed_s": round(time.time() - t0),
    }
    log(json.dumps(row))
    if monitor:
        monitor.write(json.dumps(row) + "\n")
        monitor.close()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora": adapter_state(loras),
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "target_mu": mu,
            "target_sd": sd,
            "seed": cfg.seed,
            "steps": step,
            "config": asdict(cfg),
        },
        out_path,
    )
    return {"final": row, "preds": preds}


def load_value_head(base_model: str, ckpt_path, *, device="cuda:0", lora_r: int = 16):
    """(model, head, tokenizer, checkpoint) with the trained weights loaded."""
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device
    )
    model.config.use_cache = False
    loras = inject_lora(model, r=lora_r)
    ck = torch.load(ckpt_path, map_location="cpu")
    for i, m in enumerate(loras):
        m.A.data.copy_(ck["lora"][str(i)]["A"])
        m.B.data.copy_(ck["lora"][str(i)]["B"])
    head = nn.Linear(model.config.hidden_size, 1).to(device).float()
    head.load_state_dict(ck["head"])
    model.eval()
    return model, head, tok, ck
