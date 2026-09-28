"""Pure-inference episode driver and readout for the prompting baselines.

One call to ``run_episode`` is one whole game played by a scaffold instead of
the trained policy.  The engine side is the RL rollout's, unchanged: the same
``civmarsh.train.episode.Episode`` / ``drive_episode`` thread, the same
``obs_ready`` / ``reply_ready`` handshake, the same action menu and the same
terminal ``score_end`` read out of the finished save.  Only the source of the
reply text differs: the single forward pass of the RL rollout becomes one call
into a scaffold adapter (``civmarsh.baselines.scaffolds``) that may issue
several LLM calls per decision.

Nothing here trains, so there is no slime ``Sample`` assembly, no loss mask or
log probs, no value capture and no reward hook, and no tokenizer: the cost
axis is measured from the server's own usage counters (``TokenLedger``).

Two injection seams keep the loop testable without a freeciv server or a GPU:
  * ``llm(messages) -> (text, prompt_tokens, completion_tokens)``: the real one
    posts to a pure-inference sglang server (``make_sglang_llm``).
  * ``engine``: the module supplying ``Episode`` / ``drive_episode``; defaults
    to ``civmarsh.train.episode``.
"""

import asyncio
import json
import re
import statistics
import threading
import urllib.request
from collections import Counter, defaultdict

from civmarsh.env.menu import BundleValidationError
from civmarsh.env.prompts import event_digest
from civmarsh.train.episode import parse_reply

from .prompts import TASK_CONTRACT

# Per-call cap (chars) on the persisted response text: it bounds a runaway
# repetition loop without cutting ordinary scaffold reasoning.
RESPONSE_TEXT_CAP = 6000

# ControlLost classes that mark an engine seat stall rather than a death in the
# game; such episodes are excluded from the readout, like engine errors.
STALL_CLASSES = ("INFRA", "UNKNOWN")


def _default_engine():
    """The RL rollout's engine thread (``Episode`` / ``drive_episode``)."""
    from civmarsh.train import episode

    return episode


# ---------------------------------------------------------------------------
# token accounting
# ---------------------------------------------------------------------------


class TokenLedger:
    """Per-LLM-call log for one episode: prompt/completion counts plus the raw
    response text.

    Every scaffold call goes through ``ScaffoldContext.ask``, so the ledger is
    the single funnel: a scaffold cannot spend tokens off the books.  The raw
    text rides on the same rows because the adapters hand the engine only the
    extracted JSON object (``scaffolds._last_json_object``); the rows keep the
    scaffold's reasoning.  Rows carry ``turn``, so a decision's reasoning is
    the rows with that turn; end-of-game calls (Reflexion's lesson) carry
    ``turn: None``.
    """

    def __init__(self):
        self.calls = []  # turn / tag / prompt_tokens / completion_tokens / text

    def record(self, turn, tag, prompt_tokens, completion_tokens, text=None):
        text = text or ""
        if len(text) > RESPONSE_TEXT_CAP:
            text = text[:RESPONSE_TEXT_CAP] + f"...[truncated, {len(text)} chars total]"
        self.calls.append(
            {
                "turn": turn,
                "tag": tag,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "text": text,
            }
        )

    def totals(self, turn=None):
        rows = (
            self.calls if turn is None else [c for c in self.calls if c["turn"] == turn]
        )
        prompt = sum(c["prompt_tokens"] for c in rows)
        completion = sum(c["completion_tokens"] for c in rows)
        return {
            "n_calls": len(rows),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


# ---------------------------------------------------------------------------
# scaffold-facing context
# ---------------------------------------------------------------------------


class ScaffoldContext:
    """What a scaffold sees.  One instance per episode, mutated in place per
    decision (``turn`` / ``obs_text`` / ``menu`` / ``digest``)."""

    def __init__(self, *, meta, cfg, llm, ledger, memory):
        self.meta = meta
        self.cfg = cfg
        self.ledger = ledger
        self.memory = memory  # cross-episode store (Reflexion)
        self.state = {}  # per-episode scratch (SAGA goal cache)
        self.records = []  # past decisions, digest source
        self.result = None  # engine terminal record, set at the end
        self.turn = None
        self.obs_text = ""
        self.menu = None
        self.digest = ""
        self.startturn = meta.get("start_turn", 1)
        self.endturn = getattr(cfg, "civ_endturn", 70)
        self.system = TASK_CONTRACT.format(
            focal=meta["focal_player"], startturn=self.startturn, endturn=self.endturn
        )
        self._llm = llm

    def decision_prompt(self, instruction=""):
        """The RL policy's per-decision prompt body (contract + digest +
        bounded state + menu), plus a scaffold-specific instruction."""
        parts = [self.system, self.digest, self.obs_text]
        if instruction:
            parts.append(instruction)
        return "\n".join(parts)

    async def ask(self, user_text, *, tag):
        messages = [{"role": "user", "content": user_text}]
        text, prompt_tokens, completion_tokens = await self._llm(messages)
        self.ledger.record(self.turn, tag, prompt_tokens, completion_tokens, text=text)
        return text


# ---------------------------------------------------------------------------
# real LLM: pure-inference sglang server
# ---------------------------------------------------------------------------


def make_sglang_llm(
    base_url,
    *,
    model="policy",
    temperature=1.0,
    max_tokens=1024,
    timeout=600,
    extra_body=None,
):
    """``llm(messages)`` against a pure-inference sglang server.

    Uses the OpenAI-compatible ``/v1/chat/completions`` route: it applies the
    model's chat template server-side and returns ``usage`` counts, so the
    baseline needs neither a local tokenizer nor a second source of truth for
    the cost axis.  stdlib urllib in a thread, so no aiohttp dependency.

    ``extra_body`` is merged into every request; Qwen3 runs with thinking off,
    as the RL policy does, via
    ``{"chat_template_kwargs": {"enable_thinking": False}}``.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    extra_body = dict(extra_body or {})

    def _post(messages):
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **extra_body,
            }
        ).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    async def llm(messages):
        out = await asyncio.to_thread(_post, messages)
        usage = out.get("usage") or {}
        return (
            out["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    return llm


# ---------------------------------------------------------------------------
# decision audit
# ---------------------------------------------------------------------------


def _reply_audit(reply):
    if isinstance(reply, BundleValidationError):
        return [], str(reply)
    if reply is None:
        return [], "no reply"
    return list(reply.keys), None


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------


async def run_episode(scaffold, meta, cfg, *, llm=None, engine=None, memory=None):
    """Play one whole game with ``scaffold`` and return its record.

    ``meta`` is a starts-bank row (position_id, save_path, focal_player,
    optionally start_turn); ``cfg`` carries the civ_* keys the engine thread
    reads (civ_workdir / civ_endturn / civ_engine_seed) plus
    ``civ_digest_window`` and ``civ_baseline_llm_url``.
    ``memory`` is a caller-owned dict that survives across episodes, the only
    channel a cross-episode scaffold (Reflexion) has.

    Returns score_end / end_turn / eliminated / control_lost, the
    per-decision records (turn, keys, error, per-decision token totals), the
    per-call log rows (each with the raw response text, keyed by turn) and
    the episode token account.  ``error`` is set instead of a score when the
    engine session itself failed.
    """
    engine = engine or _default_engine()
    if llm is None:
        llm = make_sglang_llm(cfg.civ_baseline_llm_url)
    digest_window = getattr(cfg, "civ_digest_window", 5)

    ledger = TokenLedger()
    ctx = ScaffoldContext(
        meta=meta,
        cfg=cfg,
        llm=llm,
        ledger=ledger,
        memory=memory if memory is not None else {},
    )

    ep = engine.Episode()
    ep.loop = asyncio.get_running_loop()
    ep.obs_ready = asyncio.Event()
    th = threading.Thread(
        target=engine.drive_episode, args=(ep, meta, cfg), daemon=True
    )
    th.start()
    episode_id = f"{meta.get('position_id', 'pos')}-{ep.uid}"

    decisions = []
    while True:
        await ep.obs_ready.wait()
        ep.obs_ready.clear()
        if ep.done:
            break
        obs_text, menu, turn = ep.obs

        ctx.turn = turn
        ctx.obs_text = obs_text
        ctx.menu = menu
        ctx.digest = event_digest(ctx.records, digest_window)

        try:
            reply_text = await scaffold.reply(ctx)
        except BaseException:
            # release the engine thread before unwinding, or it blocks on
            # reply_ready forever and the live session leaks
            ep.raw = "(scaffold failed)"
            ep.reply = None
            ep.reply_ready.set()
            await _drain(ep)
            raise

        reply = parse_reply(reply_text, menu)
        ep.raw = reply_text
        ep.reply = reply
        ep.reply_ready.set()

        keys, error = _reply_audit(reply)
        record = {"turn": turn, "keys": keys}
        if error:
            record["error"] = error
        ctx.records.append(record)
        decisions.append(
            {
                **record,
                "reply_head": (reply_text or "")[:200],
                "tokens": ledger.totals(turn),
            }
        )

    for _ in range(600):
        if not th.is_alive():
            break
        await asyncio.sleep(0.1)

    out = {
        "episode_id": episode_id,
        "position_id": meta.get("position_id"),
        "game_id": meta.get("game_id"),
        "scaffold": getattr(scaffold, "name", type(scaffold).__name__),
        "n_decisions": len(decisions),
        "decisions": decisions,
    }
    if ep.error or ep.result is None:
        out["error"] = ep.error or "no result"
        out["llm_calls"] = ledger.calls
        out["tokens"] = ledger.totals()
        return out

    ctx.result = dict(ep.result)
    end_hook = getattr(scaffold, "episode_end", None)
    if end_hook is not None:
        ctx.turn = None  # end-of-game calls are not attributed to a decision
        await end_hook(ctx)

    out.update(
        {
            "score_end": ep.result["score_end"],
            "end_turn": ep.result["end_turn"],
            "eliminated": ep.result["eliminated"],
            # the ControlLost reason separates a death in the game from an
            # engine seat stall (its class prefix drives the readout)
            "control_lost": ep.result.get("control_lost"),
            "llm_calls": ledger.calls,
            "tokens": ledger.totals(),
        }
    )
    return out


async def _drain(ep):
    """Let the engine thread finish its session after we stop answering."""
    while not ep.done:
        await ep.obs_ready.wait()
        ep.obs_ready.clear()
        if not ep.done:
            ep.reply = None
            ep.reply_ready.set()


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


def final_records(records):
    """One record per (position_id, sample_idx).

    A later record replaces the kept one only when the kept one is an error,
    so a retried episode supersedes its failed attempt and a finished episode
    is never replaced.
    """
    keep = {}
    for r in records:
        key = (r.get("position_id"), r.get("sample_idx"))
        if key not in keep or keep[key].get("error"):
            keep[key] = r
    return list(keep.values())


def is_stall(record):
    """True when the game ended on an engine seat stall (ControlLost class
    INFRA or UNKNOWN) rather than a death in the game."""
    lost = record.get("control_lost") or ""
    return any(lost.startswith(cls + ":") for cls in STALL_CLASSES)


def is_valid(record):
    """A valid episode finished with a terminal score: no engine error and
    no seat stall.  A death in the game is valid and scores 0."""
    return (
        not record.get("error")
        and record.get("score_end") is not None
        and not is_stall(record)
    )


def _lost_reason(record):
    """The control-lost reason with player names normalized away, so the same
    failure mode from different starts lands in one bucket."""
    return re.sub(r"'[^']*'", "'X'", record["control_lost"])


def summarize_arm(records, *, positions=None):
    """Readout of one arm from its episode records (one per position and
    sample, see ``final_records``).

    Only valid episodes count (``is_valid``); engine errors and seat stalls
    are reported as ``n_excluded``.  ``score`` is the mean over positions of
    the per-position mean terminal score; ``eliminated_rate`` is the share of
    valid games that end with score 0.  ``positions`` restricts the readout to
    those position ids.
    """
    if positions is not None:
        positions = set(positions)
        records = [r for r in records if r.get("position_id") in positions]
    ok = [r for r in records if is_valid(r)]
    by_position = defaultdict(list)
    for r in ok:
        by_position[r["position_id"]].append(r["score_end"])
    position_means = {k: statistics.mean(v) for k, v in sorted(by_position.items())}
    survivors = [r["score_end"] for r in ok if r["score_end"] > 0]
    lost = [r for r in ok if r.get("control_lost")]
    n_decisions = sum(r.get("n_decisions") or len(r.get("decisions", [])) for r in ok)
    n_invalid = sum(
        sum(1 for d in r.get("decisions", []) if d.get("error")) for r in ok
    )
    tok = {
        k: sum((r.get("tokens") or {}).get(k, 0) for r in ok)
        for k in ("n_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "n_episodes": len(records),
        "n_valid": len(ok),
        "n_excluded": len(records) - len(ok),
        "n_positions": len(position_means),
        "score": (statistics.mean(position_means.values()) if position_means else None),
        "position_means": position_means,
        "eliminated_rate": (
            sum(1 for r in ok if r["score_end"] == 0) / len(ok) if ok else None
        ),
        "survivor_score": statistics.mean(survivors) if survivors else None,
        "control_lost_counts": dict(Counter(_lost_reason(r) for r in lost)),
        "n_decisions": n_decisions,
        "invalid_decisions": n_invalid,
        "invalid_rate": n_invalid / n_decisions if n_decisions else None,
        "decisions_per_game": n_decisions / len(ok) if ok else None,
        "calls_per_decision": tok["n_calls"] / n_decisions if n_decisions else None,
        "tokens_total": tok,
        "tokens_per_decision": (
            tok["total_tokens"] / n_decisions if n_decisions else None
        ),
    }
