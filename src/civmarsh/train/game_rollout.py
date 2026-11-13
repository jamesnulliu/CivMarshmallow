"""slime rollout hooks for the policy runs on other games (Othello, Catan).

The sample contract of ``train.rollout``: one WHOLE game is one episode, every
decision becomes an independent slime ``Sample``, and all decisions of an
episode share one ``rollout_id``.  The environment is an in-process
``civharness.games`` game instead of a Freeciv server, so there is no engine
thread, no work directory and no seat to lose:

    prompt   = the game's system contract + the current position + legal actions
    response = this decision's action only (an Othello square, a Catan menu key)

Every sibling episode of a start loads the same saved position and gets its own
engine seed (``game_opp_seed`` in the sample metadata: the opponents' choices,
and in Catan also the dice), so siblings differ only by the policy's sampling
and the engine's randomness.

Episode-end assembly (rewards, optional GAE, Sample construction) is
``train.rollout.assemble_decision_samples``, unchanged, so the rewards follow
the Freeciv contract:

  * sparse        final score on every sibling, whitened per start position by
                  ``train.reward_post.post_process``;
  * scoreboard    Phi = visible proxy now (``rewards.games.score_now_value``);
  * civtelescope  Phi = CivTelescope vs a reference set
                  (``rewards.games.telescope_value``); ``zero_shot`` serves the
                  untrained backbone through the same function;
  * mix           ``rewards.games.mix_value``;
  the potential-based ones through cross-decision GAE and ``train.advantage``.

An invalid reply (anything but exactly one legal action) is recorded as a parse
error, priced by ``civ_err_penalty`` like an invalid Freeciv bundle, and the
environment plays a uniformly random legal action so the game can continue.
A position with a single legal action (an Othello pass, a lone Catan ROLL /
DISCARD / END_TURN) is played by the harness and is not a decision.

slime wiring:
  --custom-generate-function-path    civmarsh.train.game_rollout.generate
  --custom-rm-path                   civmarsh.train.game_rollout.group_reward  (with --group-rm)
  --dynamic-sampling-filter-path     civmarsh.train.game_rollout.drop_failed_episode_groups
  --custom-reward-post-process-path  civmarsh.train.reward_post.post_process
  --custom-advantage-function-path   civmarsh.train.advantage.broadcast_decision_advantages
                                     (potential-based rewards only)

Config keys (``--custom-config-path`` yaml, set onto the args):
  civ_game                 "othello" (default) or "catan"
  civ_game_opp_depth       Othello: opponent search depth (default 2, the
                           replay oracle's)
  civ_game_opp_temp        Othello: opponent softmax temperature (default 20.0,
                           the replay oracle's)
  civ_game_opp_exact_from  Othello: the opponent solves the endgame exactly
                           from this many empty squares (default 10; 0 = never)
  civ_game_max_turns       Catan: turn cap, a normal end scored by victory
                           points (default: the engine's 1000)
  civ_game_choice_tokens   max new tokens per decision (default 8; a reply is
                           one square or one menu key)
  civ_enable_thinking      chat-template ``enable_thinking`` (default false)
  civ_value_fn_path / civ_err_penalty / civ_decision_log / civ_reward_log /
  civ_drop_log / civ_stall_max_frac / civ_min_group
                           as in ``train.rollout``

slime and torch are imported inside the functions that need them, so this
module (and ``run_episode``) imports and unit-tests on a CPU machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random

from civharness.games import (
    decision_snapshot,
    episode_result,
    finish_decision,
    make_env,
)

from . import rollout as _rollout

DEFAULT_CHOICE_TOKENS = 8

# Start-row fields that are only needed to load the position stay out of the
# per-decision Sample metadata (a Catan save is about 6 KB, and every decision
# sample of an episode would carry a copy).
_HEAVY_START_FIELDS = ("save", "state")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sample_metadata(start: dict, opp_seed: int) -> dict:
    """The start row as it travels in every decision Sample: without the save,
    with the episode's engine seed."""
    return {
        **{k: v for k, v in start.items() if k not in _HEAVY_START_FIELDS},
        "game_opp_seed": opp_seed,
    }


def env_from_args(args):
    """The game environment the config keys describe."""
    game = getattr(args, "civ_game", "othello")
    if game == "othello":
        return make_env(
            game,
            opp_depth=int(getattr(args, "civ_game_opp_depth", 2)),
            opp_temp=float(getattr(args, "civ_game_opp_temp", 20.0)),
            opp_exact_from=int(getattr(args, "civ_game_opp_exact_from", 10)),
        )
    if game == "catan":
        return make_env(game, max_turns=getattr(args, "civ_game_max_turns", None))
    return make_env(game)


async def run_episode(
    env,
    start,
    rng,
    *,
    tok,
    post_fn,
    sampling_params,
    choice_tokens=DEFAULT_CHOICE_TOKENS,
    tpl_kwargs=None,
):
    """Drive one episode; ``post_fn(payload)`` is the awaitable sglang
    /generate call.  Returns (decisions, result, aborted); on abort the
    decisions collected so far are returned and ``result`` is None.  Free of
    slime, so it runs in tests with a fake tokenizer and post_fn."""
    tpl_kwargs = tpl_kwargs or {}
    env.reset(start, rng)
    decisions = []
    while not env.done():
        snap = decision_snapshot(env)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": snap["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **tpl_kwargs,
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                **sampling_params,
                "max_new_tokens": choice_tokens,
                "stop": None,
            },
            "return_logprob": True,
        }
        out = await post_fn(payload)
        if out["meta_info"]["finish_reason"]["type"] == "abort":
            return decisions, None, True
        ids = [t[1] for t in out["meta_info"]["output_token_logprobs"]]
        lps = [t[0] for t in out["meta_info"]["output_token_logprobs"]]
        # Detokenize here: through the sglang router ``out["text"]`` keeps the
        # literal end-of-turn token, which would turn every clean move into a
        # parse error.
        resp_text = tok.decode(ids, skip_special_tokens=True)
        # the opponents' search is CPU work: keep it off the event loop
        record = await asyncio.to_thread(finish_decision, env, snap, resp_text, rng)
        decisions.append(
            {
                **record,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_sha": _sha(prompt_text),
                "response_ids": ids,
                "response_lps": lps,
                "response_text": out["text"],
            }
        )
    return decisions, episode_result(env), False


async def generate(args, sample, sampling_params):
    """Custom generate: one source sample (a start position) -> list[Sample],
    one independent sample per decision."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    tok = state.tokenizer
    tpl_kwargs = {"enable_thinking": bool(getattr(args, "civ_enable_thinking", False))}

    # Engine randomness is fresh per episode (siblings of one start must
    # differ) and logged, so an episode can be replayed.
    opp_seed = int.from_bytes(os.urandom(8), "big")
    start = sample.metadata or {}
    meta = sample_metadata(start, opp_seed)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    episode_id = f"{meta.get('position_id', 'pos')}-{opp_seed:016x}"

    # No try/except around the episode: an in-process game has no
    # infrastructure to fail, so an exception here is a bug (a bad start row,
    # an adapter error) and must stop the run; as an ABORTED sample it would be
    # resampled forever.
    decisions, result, aborted = await run_episode(
        env_from_args(args),
        start,
        random.Random(opp_seed),
        tok=tok,
        post_fn=lambda p: post(url, p),
        sampling_params=sampling_params,
        choice_tokens=int(
            getattr(args, "civ_game_choice_tokens", DEFAULT_CHOICE_TOKENS)
        ),
        tpl_kwargs=tpl_kwargs,
    )

    if aborted:
        sample.status = Sample.Status.ABORTED
        return [sample]
    if result is None or not decisions:
        raise RuntimeError(
            f"{meta.get('position_id')}: episode produced no decisions "
            "(start position already finished?)"
        )

    decision_log = getattr(args, "civ_decision_log", None)
    if decision_log:
        _rollout._append_jsonl(
            decision_log,
            {
                "episode_id": episode_id,
                "position_id": meta.get("position_id"),
                "score_end": result.get("score_end"),
                "eliminated": result.get("eliminated"),
                "control_lost": result.get("control_lost"),
                "elim_class": result.get("elim_class"),
                "last_decision_turn": decisions[-1]["turn"],
                "decisions": [
                    {
                        "turn": d["turn"],
                        "err": d["error"],
                        "keys": d["keys"],
                        "act": d.get("action"),
                        "resp": d["response_text"][:120],
                    }
                    for d in decisions
                ],
            },
        )

    value_fn_path = getattr(args, "civ_value_fn_path", None)
    value_fn = _rollout._load_function(value_fn_path) if value_fn_path else None
    # off the event loop: the CivTelescope value function does blocking HTTP fan-out
    return await asyncio.to_thread(
        _rollout.assemble_decision_samples,
        decisions,
        meta,
        result,
        episode_id=episode_id,
        source_index=sample.index,
        source_group_index=sample.group_index,
        over_cap_turns=[],
        gamma=args.gamma,
        lam=args.lambd,
        value_fn=value_fn,
        value_fn_args=args,
        err_penalty=float(
            getattr(args, "civ_err_penalty", _rollout.DEFAULT_ERR_PENALTY)
        ),
    )


# The group reward and the failed-group filter are game-independent: they read
# only Sample status and metadata.
group_reward = _rollout.group_reward
drop_failed_episode_groups = _rollout.drop_failed_episode_groups
