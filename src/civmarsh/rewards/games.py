"""Potentials for the policy runs on other games (Othello, Catan).

Same contract as the Freeciv value functions (``potential``, ``scoreboard``):
``fn(decisions, result, args) -> list[float]``, one Phi per decision in
final-score units, so the terminal reward and Phi stay commensurate inside
GAE.  Decision records come from ``civharness.games`` through
``train.game_rollout`` and carry ``turn``, ``focal``, ``value_input`` and
``score_now``.

  score_now_value  Phi = the visible proxy now (``decision["score_now"]``,
                   recorded by the environment).  Othello: disc differential.
                   Catan: public victory points, an integer on a 2..10 scale,
                   so a coarse potential that is constant over most decisions.
  telescope_value  Phi = the reference-set winrate of a pairwise value model
                   (CivTelescope trained on the game's replay bank, or the
                   untrained backbone for the zero-shot comparison), mapped to
                   final-score units by interpolating the references' sorted
                   final scores: the Freeciv potential with the cross-game
                   pairwise prompt.
  mix_value        Phi = w_t(t) * telescope + w_s(t) * scoreboard; whatever
                   weight is left over belongs to the sparse reward (a zero
                   baseline).  w_t(t) = j0 + (j1 - j0) * sigmoid((turn - c) / s),
                   w_s(t) = sb * (1 - w_t(t)).  The end weights need not be 0
                   or 1, so no signal has to take over alone.

Config keys, besides the reference-set keys of ``potential`` (civ_value_url,
civ_value_refset, civ_value_refset_idx, civ_value_single_order,
civ_value_concurrency, civ_value_log, civ_value_harvest):
  civ_game                 "othello" (default) or "catan"
  civ_game_prompt_name     game name filled into the pairwise prompt (default:
                           ``PROMPT_GAME_NAME`` of civ_game)
  civ_game_value_chatml    default false: the raw prompt + "\\nANSWER:" goes to
                           /generate, as on Freeciv.  true wraps the prompt in
                           the ChatML turn that training applied through the
                           chat template, for checking which serving format
                           tracks the outcome better on a new game.
  civ_game_value_system    system text of that wrapper (default: Qwen2.5's)
  civ_game_mix_j0, civ_game_mix_j1
                           mix_value: CivTelescope weight before / after the
                           hand-off (default 0.5 / 0.5); 0 / 0 makes no
                           CivTelescope call at all
  civ_game_mix_c, civ_game_mix_s
                           mix_value: hand-off center and width in turns
                           (default 0 / 1)
  civ_game_mix_sb          mix_value: share of the non-CivTelescope weight on
                           the scoreboard, the rest on the sparse baseline
                           (default 1)

The reference set is static for these runs (no refresh).

Wired via civ_value_fn_path =
  "civmarsh.rewards.games.score_now_value"   (scoreboard)
  "civmarsh.rewards.games.telescope_value"   (civtelescope, zero_shot)
  "civmarsh.rewards.games.mix_value"         (mix)
"""

from __future__ import annotations

import math

from civmarsh.traps.games.prompt import CROSS_GAME_PAIRWISE_PROMPT

from .potential import (
    append_harvest,
    append_value_log,
    bucket_and_last,
    load_reference_set,
    select_references,
    telescope_winrates,
    value_config,
    winrate_to_score,
)

# The game name filled into the pairwise prompt.  "othello" is the name the
# cross-game zero-shot evaluation uses; a Catan rendering opens with
# "Settlers of Catan", and the prompt names the game the same way.
PROMPT_GAME_NAME = {"othello": "othello", "catan": "Settlers of Catan"}
QWEN25_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def game_pairwise_prompt(game: str) -> str:
    """The cross-game pairwise template with the game name filled in; the
    position fields (focal_a, rendering_a, focal_b, rendering_b) stay open."""
    return CROSS_GAME_PAIRWISE_PROMPT.replace("{game}", game)


def chatml_wrap(template: str, system: str = QWEN25_SYSTEM) -> str:
    """``template`` inside a ChatML user turn, ending at the assistant header.
    The client appends "\\nANSWER:" after it, which is the text the training
    examples end with (chat template + "ANSWER:")."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{template}<|im_end|>\n"
        f"<|im_start|>assistant"
    )


def value_prompt(args) -> str:
    """The pairwise template the value model is served with, from the args."""
    game = getattr(args, "civ_game", "othello")
    name = getattr(args, "civ_game_prompt_name", None) or PROMPT_GAME_NAME.get(
        game, game
    )
    template = game_pairwise_prompt(name)
    if getattr(args, "civ_game_value_chatml", False):
        template = chatml_wrap(
            template, getattr(args, "civ_game_value_system", None) or QWEN25_SYSTEM
        )
    return template


def _log(args, decisions, result, values, fails):
    append_value_log(
        args,
        {
            "turns": [d["turn"] for d in decisions],
            "values": [round(v, 2) for v in values],
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": fails,
        },
    )


def score_now_value(decisions, result, args) -> list[float]:
    """Phi_t = the visible proxy at decision t."""
    values = [float(d["score_now"]) for d in decisions]
    _log(args, decisions, result, values, 0)
    # a game is at most a few dozen decisions: harvest all of them
    append_harvest(args, decisions, result, range(len(decisions)))
    return values


def telescope_value(decisions, result, args) -> list[float]:
    """Phi_t = reference-set winrate of decision t, in final-score units."""
    url, refset_path, idx, single_order, workers = value_config(args)
    refset = load_reference_set(refset_path)
    references = [select_references(refset, d["turn"], idx) for d in decisions]
    winrates, failed = telescope_winrates(
        url,
        decisions,
        references,
        range(len(decisions)),
        single_order=single_order,
        workers=workers,
        prompt=value_prompt(args),
    )
    values = [
        winrate_to_score(winrates[i], references[i]) for i in range(len(decisions))
    ]
    _log(args, decisions, result, values, failed)
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))
    return values


def mix_weights(turns, args) -> list[tuple[float, float]]:
    """(w_t, w_s) per decision turn; see ``mix_value``."""
    j0 = float(getattr(args, "civ_game_mix_j0", 0.5))
    j1 = float(getattr(args, "civ_game_mix_j1", 0.5))
    c = float(getattr(args, "civ_game_mix_c", 0.0))
    s = float(getattr(args, "civ_game_mix_s", 1.0))
    sb = float(getattr(args, "civ_game_mix_sb", 1.0))
    if not (0 <= j0 <= 1 and 0 <= j1 <= 1 and 0 <= sb <= 1 and s > 0):
        raise ValueError("invalid mix weights")
    out = []
    for t in turns:
        z = max(-40.0, min(40.0, (float(t) - c) / s))
        wt = j0 + (j1 - j0) / (1.0 + math.exp(-z))
        out.append((wt, sb * (1.0 - wt)))
    return out


class _TelescopeComponent:
    """Args view for the CivTelescope part of ``mix_value``: its value log goes
    to ``<log>_civtelescope.jsonl`` next to the mixed log, and it harvests
    nothing (the mixed potential harvests once)."""

    def __init__(self, args):
        self._args = args

    def __getattr__(self, name):
        if name == "civ_value_harvest":
            return None
        if name == "civ_value_log":
            path = getattr(self._args, "civ_value_log", None)
            return path.replace(".jsonl", "_civtelescope.jsonl") if path else None
        return getattr(self._args, name)


def mix_value(decisions, result, args) -> list[float]:
    """Phi_t = w_t * telescope + w_s * scoreboard (weights from ``mix_weights``)."""
    weights = mix_weights([d["turn"] for d in decisions], args)
    scoreboard = [float(d["score_now"]) for d in decisions]
    telescope = [0.0] * len(decisions)
    if any(wt > 0 for wt, _ in weights):
        telescope = telescope_value(decisions, result, _TelescopeComponent(args))
    values = [
        wt * tv + ws * sv for (wt, ws), tv, sv in zip(weights, telescope, scoreboard)
    ]
    _log(args, decisions, result, values, 0)
    append_harvest(args, decisions, result, range(len(decisions)))
    return values
