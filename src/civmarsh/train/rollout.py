"""slime rollout hooks: one whole game per source sample, one sample per decision.

One source sample (a start position) runs as one live engine episode
(``train.episode``), and each decision becomes an INDEPENDENT slime ``Sample``:

    prompt   = system contract + recent-actions digest (last K decisions)
             + current bounded state block + grouped action menu
    response = this decision's strict-JSON action bundle only

No earlier observation is carried forward, so rollout and training condition on
the exact same bounded prompt (PPO ratios are well defined).  All decision
samples of one episode share one ``rollout_id`` (the source sample's index), so
slime's rollout-level loss denominator counts the episode once.

Per decision, ``train_metadata`` records the episode id, turn, decision index,
prompt and menu hashes, reward and done flag and, when a value function is
configured, the potential and the cross-decision GAE advantage computed at
episode end (before any data-parallel partitioning; see ``rewards.gae``).

Reward paths (all rewards share this generate path):
  * sparse / terminal_civtelescope: the episode reward is set on EVERY sibling
    by ``group_reward`` and whitened per start position across episodes by
    ``train.reward_post.post_process``.
  * dense (scoreboard, civtelescope and the hybrids): rewards stay
    [0, ..., 0, score_end] in train_metadata; ``civ_value_fn_path`` supplies
    Phi per decision and ``train.advantage`` broadcasts the precomputed A_t.

slime wiring:
  --custom-generate-function-path    civmarsh.train.rollout.generate
  --custom-rm-path                   civmarsh.train.rollout.group_reward  (with --group-rm)
  --dynamic-sampling-filter-path     civmarsh.train.rollout.drop_failed_episode_groups
  --custom-reward-post-process-path  civmarsh.train.reward_post.post_process
  --custom-advantage-function-path   civmarsh.train.advantage.broadcast_decision_advantages
                                     (dense rewards only)

Config keys (``--custom-config-path`` yaml, set onto the args), besides the
``train.episode`` keys and the value-function keys of ``civmarsh.rewards``:
  civ_choice_tokens           max tokens per action bundle (default 192)
  civ_max_prompt_tokens       per-decision prompt cap (default 8192); a
                              decision over the cap becomes a logged no-op
  civ_digest_window           decisions kept in the recent-actions digest
                              (default 5)
  civ_enable_thinking         chat-template ``enable_thinking`` (default false)
  civ_err_penalty             fixed penalty per invalid decision (default 1.0)
  civ_value_fn_path           "module.fn": fn(decisions, result, args) ->
                              Phi per decision; enables per-decision GAE
  civ_terminal_value_fn_path  "module.fn": fn(decisions, result, args) ->
                              Phi(s_T); the episode reward becomes this value
                              instead of the final score (exclusive with
                              civ_value_fn_path)
  civ_reward_log              per-group jsonl
  civ_decision_log            optional per-episode decision audit jsonl
  civ_drop_log                optional jsonl, one line per dropped episode
  civ_stall_max_frac          per-episode seat-stall ceiling over the last
                              16 groups (default 0.30)
  civ_min_group               keep a group with its failed episodes removed
                              when at least this many survived (default: every
                              episode must survive).  Requires slime's
                              ``--log-passrate`` and ``--partial-rollout`` off
                              and ``reward_post.post_process`` as the reward
                              post-process (it groups by position, not by
                              fixed width).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time

from civmarsh.env.prompts import SYSTEM_PROMPT, event_digest
from civmarsh.env.snapshot import SpatialSnapshot
from civmarsh.rewards.gae import decision_gae, decision_rewards

from .episode import DEFAULT_ENDTURN, Episode, drive_episode, parse_reply, resolve_reply

DEFAULT_CHOICE_TOKENS = 192
DEFAULT_MAX_PROMPT_TOKENS = 8192
DEFAULT_DIGEST_WINDOW = 5
DEFAULT_ERR_PENALTY = 1.0
THREAD_JOIN_POLLS = 600  # x 0.1 s after the last decision


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _append_jsonl(path, row) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _load_function(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def assemble_decision_samples(
    decisions,
    meta,
    result,
    *,
    episode_id,
    source_index,
    source_group_index,
    over_cap_turns,
    gamma,
    lam,
    value_fn=None,
    value_fn_args=None,
    err_penalty=0.0,
    terminal_value_fn=None,
):
    """Episode-end assembly: rewards/dones, optional GAE, one Sample per decision.

    ``decisions`` entries carry turn, prompt_text, prompt_ids, prompt_sha,
    response_ids, response_lps, response_text, candidate_sha, keys, error (and
    focal / value_input for the value functions).  ``result`` is the engine
    thread's terminal record.  The last decision is terminal (done=1) and
    carries the final score; a ``value_fn`` enables cross-decision GAE here.
    A ``terminal_value_fn`` instead attaches Phi(s_T) as ``phi_terminal`` in
    the metadata, for ``group_reward``; it does not enter rewards or GAE.
    Loss is on the response (action) tokens only.
    """
    from slime.utils.types import Sample

    # value functions tag their log lines with the start position
    result = {**result, "position_id": meta.get("position_id")}

    score_end = float(result["score_end"])
    n = len(decisions)
    rewards = decision_rewards(decisions, score_end, err_penalty=err_penalty)
    dones = [0] * (n - 1) + [1]

    values = advantages = returns = None
    if value_fn is not None:
        values = [float(v) for v in value_fn(decisions, result, value_fn_args)]
        assert len(values) == n, (
            f"value fn returned {len(values)} values for {n} decisions"
        )
        advantages, returns = decision_gae(rewards, values, dones, gamma=gamma, lam=lam)

    phi_terminal = None
    if terminal_value_fn is not None:
        assert value_fn is None, (
            "civ_value_fn_path and civ_terminal_value_fn_path are exclusive: the "
            "first is a dense per-decision potential, the second a terminal reward"
        )
        phi_terminal = float(terminal_value_fn(decisions, result, value_fn_args))

    out_samples = []
    for i, d in enumerate(decisions):
        s = Sample()
        s.group_index = source_group_index
        s.index = source_index
        s.rollout_id = source_index  # shared by all siblings of this episode
        s.prompt = d["prompt_text"]
        s.tokens = d["prompt_ids"] + d["response_ids"]
        s.response = d["response_text"]
        s.response_length = len(d["response_ids"])
        s.loss_mask = [1] * len(d["response_ids"])
        s.rollout_log_probs = d["response_lps"]
        s.status = Sample.Status.COMPLETED
        s.metadata = {
            **meta,
            "episode_id": episode_id,
            "turn": d["turn"],
            "decision_index": i,
            "n_decisions": n,
            "keys": d["keys"],
            "parse_error": d["error"],
            "score_end": score_end,
            "end_turn": result["end_turn"],
            "eliminated": result["eliminated"],
            # ControlLost reason and class; None on a clean finish
            "control_lost": result.get("control_lost"),
            "elim_class": result.get("elim_class"),
            "over_cap_turns": over_cap_turns,
        }
        if phi_terminal is not None:
            s.metadata["phi_terminal"] = phi_terminal
        s.train_metadata = {
            "episode_id": episode_id,
            "turn": d["turn"],
            "decision_index": i,
            "n_decisions": n,
            "prompt_sha": d["prompt_sha"],
            "candidate_sha": d["candidate_sha"],
            "reward": rewards[i],
            "done": dones[i],
            "value": None if values is None else values[i],
            "advantage": None if advantages is None else advantages[i],
            "return": None if returns is None else returns[i],
        }
        out_samples.append(s)
    return out_samples


async def _drain(ep):
    """Answer every remaining turn with no reply until the engine thread ends."""
    while not ep.done:
        await ep.obs_ready.wait()
        ep.obs_ready.clear()
        if not ep.done:
            ep.reply = None
            ep.reply_ready.set()


async def generate(args, sample, sampling_params):
    """Custom generate: one source sample (a start position) -> list[Sample],
    one independent bounded sample per decision."""
    import asyncio

    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    tok = state.tokenizer
    meta = sample.metadata or {}
    endturn = getattr(args, "civ_endturn", DEFAULT_ENDTURN)
    choice_budget = getattr(args, "civ_choice_tokens", DEFAULT_CHOICE_TOKENS)
    prompt_cap = getattr(args, "civ_max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
    digest_window = getattr(args, "civ_digest_window", DEFAULT_DIGEST_WINDOW)
    # A thinking trace would consume the whole per-decision token budget.
    enable_thinking = bool(getattr(args, "civ_enable_thinking", False))

    system_text = SYSTEM_PROMPT.format(
        focal=meta["focal_player"], endturn=endturn, startturn=meta.get("start_turn", 1)
    )

    # Any value function needs the per-decision value input, rendered from the
    # live fog-of-war observation (the format CivTelescope is trained on).
    value_fn_path = getattr(args, "civ_value_fn_path", None)
    terminal_value_fn_path = getattr(args, "civ_terminal_value_fn_path", None)
    capture_value = bool(value_fn_path or terminal_value_fn_path)

    ep = Episode()
    ep.loop = asyncio.get_running_loop()
    ep.obs_ready = asyncio.Event()
    th = threading.Thread(target=drive_episode, args=(ep, meta, args), daemon=True)
    th.start()

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    episode_id = f"{meta.get('position_id', 'pos')}-{ep.uid}"

    decisions = []  # one dict per decision: sample fields + audit record
    records = []  # digest source: turn / keys / error per past decision
    over_cap_turns = []
    aborted = False

    while True:
        await ep.obs_ready.wait()
        ep.obs_ready.clear()
        if ep.done:
            break
        obs_text, menu, turn = ep.obs
        value_input = None
        if capture_value:
            value_input = SpatialSnapshot.from_observation(
                ep.obs_raw
            ).render_value_input()

        digest = event_digest(records, digest_window)
        user_text = system_text + "\n" + digest + "\n" + obs_text
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + choice_budget > prompt_cap:
            over_cap_turns.append(turn)
            records.append({"turn": turn, "error": "prompt over frozen cap"})
            ep.raw = "(prompt over cap)"
            ep.reply = None
            ep.reply_ready.set()
            continue

        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                **sampling_params,
                "max_new_tokens": choice_budget,
                "stop": None,
            },
            "return_logprob": True,
        }
        out = await post(url, payload)
        if out["meta_info"]["finish_reason"]["type"] == "abort":
            aborted = True
            ep.reply = None
            ep.reply_ready.set()
            break
        ids = [t[1] for t in out["meta_info"]["output_token_logprobs"]]
        lps = [t[0] for t in out["meta_info"]["output_token_logprobs"]]
        # Decode locally with skip_special_tokens: through the sglang router
        # out["text"] can carry the literal end-of-turn token, which breaks the
        # strict JSON parse.  Training tokens keep the eos id.
        resp_text = tok.decode(ids, skip_special_tokens=True)
        reply = parse_reply(resp_text, menu)
        _orders, pick, error = resolve_reply(reply, menu)

        ep.raw = resp_text
        ep.reply = reply
        ep.reply_ready.set()

        record = {"turn": turn, "keys": (pick or {}).get("keys", [])}
        if error:
            record["error"] = error
        records.append(record)

        decisions.append(
            {
                "turn": turn,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_sha": _sha(prompt_text),
                "response_ids": ids,
                "response_lps": lps,
                "response_text": out["text"],
                "candidate_sha": menu.sha,
                "keys": record["keys"],
                "error": error,
                "value_input": value_input,
                "focal": meta["focal_player"],
            }
        )

    if aborted:
        await _drain(ep)
        sample.status = Sample.Status.ABORTED
        return [sample]
    for _ in range(THREAD_JOIN_POLLS):
        if not th.is_alive():
            break
        await asyncio.sleep(0.1)

    if ep.error or ep.result is None or not decisions:
        sample.status = Sample.Status.ABORTED
        sample.metadata = {
            **meta,
            "episode_error": ep.error or "no result or no decisions",
        }
        return [sample]

    decision_log = getattr(args, "civ_decision_log", None)
    if decision_log:
        _append_jsonl(
            decision_log,
            {
                "episode_id": episode_id,
                "position_id": meta.get("position_id"),
                "score_end": ep.result.get("score_end"),
                "eliminated": ep.result.get("eliminated"),
                "control_lost": ep.result.get("control_lost"),
                "elim_class": ep.result.get("elim_class"),
                "last_decision_turn": decisions[-1]["turn"],
                "decisions": [
                    {
                        "turn": d["turn"],
                        "err": d["error"],
                        "keys": d["keys"],
                        "resp": d["response_text"][:120],
                    }
                    for d in decisions
                ],
            },
        )

    # An INFRA/UNKNOWN seat stall is not a terminal outcome: abort before
    # assembly so the group is dropped and resampled, and the value functions
    # never see (or harvest) its placeholder score.  A GAME death keeps 0.
    if ep.result.get("infra_stall"):
        sample.status = Sample.Status.ABORTED
        sample.metadata = {
            **meta,
            "episode_error": (
                f"infra_stall ({ep.result.get('elim_class')}): "
                f"{ep.result.get('control_lost')}"
            ),
            "elim_class": ep.result.get("elim_class"),
            "infra_stall": True,
            "episode_id": episode_id,
            "turn": decisions[-1]["turn"],
            "n_decisions": len(decisions),
        }
        return [sample]

    value_fn = _load_function(value_fn_path) if value_fn_path else None
    terminal_value_fn = (
        _load_function(terminal_value_fn_path) if terminal_value_fn_path else None
    )
    # Off the event loop: the value functions do blocking HTTP fan-out, and
    # inline they would serialize every episode behind each finisher.
    return await asyncio.to_thread(
        assemble_decision_samples,
        decisions,
        meta,
        ep.result,
        episode_id=episode_id,
        source_index=sample.index,
        source_group_index=sample.group_index,
        over_cap_turns=over_cap_turns,
        gamma=args.gamma,
        lam=args.lambd,
        value_fn=value_fn,
        value_fn_args=args,
        terminal_value_fn=terminal_value_fn,
        err_penalty=float(getattr(args, "civ_err_penalty", DEFAULT_ERR_PENALTY)),
    )


# ---------------------------------------------------------------------------
# Failed-episode filter
#
# Two failure kinds are counted per EPISODE over a sliding window of the most
# recent groups (the filter gets no step index, so a per-step reset is not
# possible):
#   infra_stall   an INFRA/UNKNOWN ControlLost (engine seat stall); a
#                 sustained rate above civ_stall_max_frac raises StallStorm
#   engine_abort  any other non-COMPLETED episode (port race, no result,
#                 generation abort); counted and logged only
# ---------------------------------------------------------------------------

_DROP_LOCK = threading.Lock()
_DROP_WINDOW = 16  # groups in the sliding window
_DROP_MIN_GROUPS = 20  # groups seen before the ceiling can fire (capped by the window)
_STALL_MAX_FRAC = 0.30
_DROP_WINDOW_ROWS = []  # (n_episodes, n_stalled, n_aborted) per filter call
_DROP_STATS = {"total": 0, "dropped": 0}  # cumulative group counters, for logs


class StallStorm(RuntimeError):
    """Per-episode seat-stall rate above the ceiling.  Raised from the filter
    so a failing environment stops the run loudly instead of being resampled
    away (the surviving minority would be a biased sample)."""


def _log_dropped(args, siblings, detail, stall_frac, abort_frac, n_window):
    """Write the dropped episode's identity and length to civ_drop_log, so the
    resampling bias against long episodes is measurable.  Never raises."""
    path = getattr(args, "civ_drop_log", None)
    if not path:
        return
    md = siblings[0].metadata or {}
    line = {
        "ts": round(time.time(), 1),
        "detail": detail,
        "position_id": md.get("position_id"),
        "episode_id": md.get("episode_id"),
        "turn": md.get("turn"),
        "n_decisions": md.get("n_decisions"),
        "elim_class": md.get("elim_class"),
        "infra_stall": bool(md.get("infra_stall")),
        "episode_error": md.get("episode_error"),
        "dropped": _DROP_STATS["dropped"],
        "total": _DROP_STATS["total"],
        "stall_frac": round(stall_frac, 4),
        "abort_frac": round(abort_frac, 4),
        "window_episodes": n_window,
    }
    try:
        _append_jsonl(path, line)
    except Exception as exc:  # noqa: BLE001 - logging must not stop the run
        print(
            f"[drop-filter] drop log write failed: {exc}", file=sys.stderr, flush=True
        )


def drop_failed_episode_groups(args, group):
    """Dynamic-sampling filter for groups containing non-COMPLETED episodes.

    A failed episode is a bare ABORTED sample without tokens or log-probs and
    must never reach training.  By default the whole group is dropped (and
    resampled); with ``civ_min_group`` set and at least that many survivors,
    the failed episodes are removed from ``group`` in place and the smaller
    group is kept.  A per-episode seat-stall rate above ``civ_stall_max_frac``
    over the sliding window raises ``StallStorm``.
    """
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput
    from slime.utils.types import Sample

    survivors, failures = [], []
    for entry in group:
        siblings = entry if isinstance(entry, list) else [entry]
        if all(s.status == Sample.Status.COMPLETED for s in siblings):
            survivors.append(entry)
        else:
            failures.append(siblings)

    n_stalled = sum(1 for f in failures if (f[0].metadata or {}).get("infra_stall"))
    n_aborted = len(failures) - n_stalled

    with _DROP_LOCK:
        _DROP_WINDOW_ROWS.append((len(group), n_stalled, n_aborted))
        del _DROP_WINDOW_ROWS[: max(0, len(_DROP_WINDOW_ROWS) - _DROP_WINDOW)]
        n_groups = len(_DROP_WINDOW_ROWS)
        n_window = sum(r[0] for r in _DROP_WINDOW_ROWS)
        w_stalled = sum(r[1] for r in _DROP_WINDOW_ROWS)
        w_aborted = sum(r[2] for r in _DROP_WINDOW_ROWS)
        _DROP_STATS["total"] += 1
        if failures:
            _DROP_STATS["dropped"] += 1
        dropped, total = _DROP_STATS["dropped"], _DROP_STATS["total"]

    stall_frac = w_stalled / n_window if n_window else 0.0
    abort_frac = w_aborted / n_window if n_window else 0.0

    min_group = getattr(args, "civ_min_group", None)
    min_group = len(group) if min_group is None else int(min_group)
    keep_partial = bool(failures) and len(survivors) >= min_group

    for siblings in failures:
        md = siblings[0].metadata or {}
        detail = (
            f"infra_stall:{md.get('elim_class')}"
            if md.get("infra_stall")
            else "engine_abort"
        )
        print(
            f"[drop-filter] {'shrank' if keep_partial else 'dropped'} group "
            f"{len(survivors)}/{len(group)} kept "
            f"(window {n_groups}g/{n_window}ep: stall {w_stalled} "
            f"{100.0 * stall_frac:.1f}%, abort {w_aborted} "
            f"{100.0 * abort_frac:.1f}%; cumulative groups touched "
            f"{dropped}/{total}) detail={detail} "
            f"position={md.get('position_id')} turn={md.get('turn')} "
            f"n_decisions={md.get('n_decisions')}",
            file=sys.stderr,
            flush=True,
        )
        _log_dropped(args, siblings, detail, stall_frac, abort_frac, n_window)

    if n_groups >= min(_DROP_MIN_GROUPS, _DROP_WINDOW):
        stall_max = getattr(args, "civ_stall_max_frac", _STALL_MAX_FRAC)
        if stall_max is not None and stall_frac > float(stall_max):
            raise StallStorm(
                f"{w_stalled}/{n_window} episodes ({100.0 * stall_frac:.1f}%) "
                f"infra_stall over the last {n_groups} groups, above the "
                f"{100.0 * float(stall_max):.1f}% per-EPISODE ceiling "
                f"(engine_abort over the same window: {w_aborted}/{n_window} = "
                f"{100.0 * abort_frac:.1f}%, counted separately). The environment "
                f"is stalling seats; resampling cannot fix it and the surviving "
                f"episodes are a biased (shorter) sample. Fix the engine host, "
                f"then restart."
            )

    if not failures:
        return DynamicFilterOutput(keep=True)
    if keep_partial:
        # slime keeps a reference to this exact list, so shrinking it in place
        # is how the reduced group reaches training.
        group[:] = survivors
        return DynamicFilterOutput(keep=True)
    return DynamicFilterOutput(keep=False, reason="aborted_episode")


async def group_reward(args, samples, **kwargs):
    """Group reward for fan-out samples.

    Each element of ``samples`` is one episode's list[Sample] (or a bare
    aborted Sample).  The episode reward is set on EVERY sibling, and [] is
    returned so the caller's zip-assignment never touches the nested lists;
    cross-episode whitening happens in ``reward_post.post_process``.

    The episode reward is the final score, or Phi(s_T) when
    ``civ_terminal_value_fn_path`` is set.  With ``civ_err_penalty`` > 0 it is
    ``sum(decision_rewards(...))`` = base - penalty * n_invalid, the same
    quantity the dense rewards distribute per decision; the error count varies
    across siblings of a position, so group whitening preserves it.  The log
    keeps the raw final score in ``score_end`` either way.
    """
    from slime.utils.types import Sample

    err_penalty = float(getattr(args, "civ_err_penalty", DEFAULT_ERR_PENALTY))
    terminal_reward = bool(getattr(args, "civ_terminal_value_fn_path", None))
    per_episode = []
    for entry in samples:
        siblings = entry if isinstance(entry, list) else [entry]
        md0 = siblings[0].metadata or {}
        ok = all(s.status == Sample.Status.COMPLETED for s in siblings)
        score = float(md0.get("score_end", 0.0)) if ok else 0.0
        n_err = sum(1 for s in siblings if (s.metadata or {}).get("parse_error"))
        phi_terminal = md0.get("phi_terminal")
        base = score
        if terminal_reward and ok:
            if phi_terminal is None:
                raise RuntimeError(
                    "civ_terminal_value_fn_path is set but the episode has no "
                    "phi_terminal; refusing to fall back to the final score"
                )
            base = float(phi_terminal)
        reward = base
        if ok and err_penalty:
            reward = sum(
                decision_rewards(
                    [
                        {"error": (s.metadata or {}).get("parse_error")}
                        for s in siblings
                    ],
                    base,
                    err_penalty=err_penalty,
                )
            )
        for s in siblings:
            s.reward = reward
        per_episode.append(
            {
                "episode_id": md0.get("episode_id"),
                "position_id": md0.get("position_id"),
                "reward": reward,
                "score_end": score,
                "phi_terminal": phi_terminal,
                "n_err": n_err,
                "n_decisions": md0.get("n_decisions"),
                "end_turn": md0.get("end_turn"),
                "eliminated": md0.get("eliminated"),
                "control_lost": md0.get("control_lost"),
                "over_cap_turns": md0.get("over_cap_turns"),
                "ok": ok,
            }
        )

    log_path = getattr(args, "civ_reward_log", None)
    if log_path:
        _append_jsonl(log_path, {"ts": round(time.time(), 1), "episodes": per_episode})
    return []
