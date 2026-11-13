"""Focal-mode episode driver.

The per-decision helpers (``decision_snapshot`` / ``finish_decision`` /
``episode_result``) are what an asynchronous policy bridge calls around its own
generate request; ``play_episode`` strings the same helpers together around a
synchronous ``act_fn``, so the parse / fallback / bookkeeping logic exists once.

Decision records carry: turn, keys (the policy's own action; empty when the
reply was invalid), action (its human-readable form), error, fallback (what the
environment played instead of an invalid reply), value_input, focal, score_now,
candidate_sha, n_legal.
"""

from __future__ import annotations

import hashlib


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def decision_snapshot(env) -> dict:
    """Everything a decision needs, captured BEFORE the action is played."""
    legal = env.legal()
    prompt = env.policy_prompt()
    return {
        "turn": env.turn(),
        "legal": legal,
        # the offered menu, including what each key stands for
        "candidate_sha": _sha("\n".join(f"{k}={env.describe(k)}" for k in legal)),
        "prompt": prompt,
        "value_input": env.value_input(),
        "focal": env.focal,
        "score_now": env.score_now(),
        "describe": {k: env.describe(k) for k in legal},
    }


def finish_decision(env, snap: dict, raw_text, rng) -> dict:
    """Parse the reply, play it (or the random-legal fallback), and return the
    decision record."""
    key, error = env.parse(raw_text, snap["legal"])
    executed = env.step(key, rng)
    return {
        "turn": snap["turn"],
        "keys": [key] if key else [],
        "action": snap["describe"][key] if key else None,
        "error": error,
        "fallback": None if key else executed,
        "value_input": snap["value_input"],
        "focal": snap["focal"],
        "score_now": snap["score_now"],
        "candidate_sha": snap["candidate_sha"],
        "n_legal": len(snap["legal"]),
    }


def episode_result(env) -> dict:
    """Terminal record in the shape of the Freeciv agent driver's result.  An
    in-process game has no seat to lose, so the elimination fields are constant."""
    return {
        "score_end": env.outcome(),
        "end_turn": env.turn(),
        "eliminated": False,
        "control_lost": None,
        "elim_class": None,
        **env.result_extra(),
    }


def play_episode(env, start: dict, act_fn, rng):
    """Play one episode with ``act_fn(prompt_text) -> raw reply text``.
    Returns (decisions, result)."""
    env.reset(start, rng)
    decisions = []
    while not env.done():
        snap = decision_snapshot(env)
        raw = act_fn(snap["prompt"])
        decisions.append(finish_decision(env, snap, raw, rng))
    return decisions, episode_result(env)
