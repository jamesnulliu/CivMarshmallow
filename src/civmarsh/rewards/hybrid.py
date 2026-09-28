"""Hybrid potentials: sigmoid hand-offs between scoreboard, CivTelescope and sparse credit.

Every hand-off uses the same weight on the decision turn t,

    w(t) = sigmoid((t - c) / s),

and every component is expressed in the reference set's final-score units, so
blends are dimensionally consistent.

* ``hybrid_sigmoid_value`` (reward ``scoreboard_to_civtelescope``):
  Phi(t) = (1 - w) * V_scoreboard(t) + w * Phi_telescope(t).
* ``sparse_hybrid_sigmoid_value`` (rewards ``sparse_to_civtelescope`` and, with
  ``civ_sparse_hybrid_telescope_first``, ``civtelescope_to_sparse``): returns
  the CivTelescope potential only.  Sparse credit is the position-group
  advantage of the final score; it depends on sibling episodes and has no
  per-state value, so the blend happens after group reward, in advantage space
  (``train.reward_post``):  A(t) = (1 - w) * A_sparse + w * A_telescope,
  where w is always the CivTelescope weight (falling instead of rising when
  CivTelescope comes first).
* ``sparse_hybrid3_value`` (reward ``scoreboard_to_civtelescope_to_sparse``):
  the first hand-off is blended here in value space,
  V_seam(t) = (1 - w1) * V_scoreboard(t) + w1 * Phi_telescope(t), and the
  second in advantage space in ``train.reward_post``,
  A(t) = (1 - w2) * A_seam(t) + w2 * A_sparse, with w2 = sigmoid((t - c2) / s2).
  The three layer weights 1 - w1, w1 * (1 - w2), w1 * w2 sum to 1 at every turn.

CivTelescope calls are skipped for decisions whose CivTelescope weight is below
``civ_hybrid_sigmoid_skip_threshold``.  In the three-signal potential the late
calls are never skipped: the seam value feeds the backward GAE recursion.

Config keys (in addition to those of ``potential``):
  civ_hybrid_sigmoid_c, civ_hybrid_sigmoid_s   first hand-off (default 35, 5)
  civ_hybrid_sigmoid_skip_threshold           default 0.01
  civ_sparse_hybrid_sigmoid                   enable the sparse blend in reward_post
  civ_sparse_hybrid_telescope_first           CivTelescope -> sparse ordering
  civ_hybrid3_sigmoid                         enable the second hand-off in reward_post
  civ_hybrid3_c2, civ_hybrid3_s2              second hand-off (default 80, 5)
"""

from __future__ import annotations

import math

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
from .scoreboard import score_nows, scoreboard_component

DEFAULT_C1 = 35.0
DEFAULT_S1 = 5.0
DEFAULT_C2 = 80.0
DEFAULT_S2 = 5.0
DEFAULT_SKIP = 0.01


# --------------------------------------------------------------------- weights


def sigmoid(turn, center, scale) -> float:
    z = (float(turn) - float(center)) / float(scale)
    if z >= 40:
        return 1.0
    if z <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def validate_sigmoid(center, scale) -> None:
    if (
        not math.isfinite(float(center))
        or not math.isfinite(float(scale))
        or scale <= 0
    ):
        raise ValueError("invalid sigmoid center/scale")


def _validate_skip(skip_threshold) -> None:
    if not (0 <= float(skip_threshold) < 1):
        raise ValueError("invalid sigmoid skip threshold")


def validate_hybrid3(c1, s1, c2, s2, skip_threshold=DEFAULT_SKIP) -> None:
    """Fail loudly on parameters that would silently break the hand-offs."""
    for name, value in (("c1", c1), ("s1", s1), ("c2", c2), ("s2", s2)):
        if not math.isfinite(float(value)):
            raise ValueError(f"hybrid3: non-finite {name}={value!r}")
    if float(s1) <= 0 or float(s2) <= 0:
        raise ValueError(f"hybrid3: scales must be positive, got s1={s1!r} s2={s2!r}")
    if float(c2) <= float(c1):
        raise ValueError(
            f"hybrid3: the CivTelescope window would be empty or inverted "
            f"(c1={c1!r} >= c2={c2!r})"
        )
    _validate_skip(skip_threshold)


def seam_weight(turn, c1=DEFAULT_C1, s1=DEFAULT_S1) -> float:
    """First hand-off: 0 = pure scoreboard, 1 = pure CivTelescope."""
    return sigmoid(turn, c1, s1)


def sparse_weight(turn, c2=DEFAULT_C2, s2=DEFAULT_S2) -> float:
    """Second hand-off: 0 = pure seam, 1 = pure sparse outcome credit."""
    return sigmoid(turn, c2, s2)


def layer_weights(turn, c1=DEFAULT_C1, s1=DEFAULT_S1, c2=DEFAULT_C2, s2=DEFAULT_S2):
    """(lam_scoreboard, lam_telescope, lam_sparse) for one decision turn."""
    w1 = sigmoid(turn, c1, s1)
    w2 = sigmoid(turn, c2, s2)
    return 1.0 - w1, w1 * (1.0 - w2), w1 * w2


def _first_handoff(args):
    center = float(getattr(args, "civ_hybrid_sigmoid_c", DEFAULT_C1))
    scale = float(getattr(args, "civ_hybrid_sigmoid_s", DEFAULT_S1))
    skip = float(getattr(args, "civ_hybrid_sigmoid_skip_threshold", DEFAULT_SKIP))
    return center, scale, skip


def _second_handoff(args):
    c2 = float(getattr(args, "civ_hybrid3_c2", DEFAULT_C2))
    s2 = float(getattr(args, "civ_hybrid3_s2", DEFAULT_S2))
    return c2, s2


# -------------------------------------------------------------- value functions


def _telescope_parts(decisions, args, weights, skip):
    """Shared setup: references, current scores and CivTelescope winrates for
    the decisions whose weight is at or above ``skip``."""
    url, refset_path, idx, single_order, workers = value_config(args)
    refset = load_reference_set(refset_path)
    turns = [int(d["turn"]) for d in decisions]
    references = [select_references(refset, t, idx) for t in turns]
    scores, score_fails = score_nows(decisions)
    active = [i for i, w in enumerate(weights) if w >= skip]
    winrates, telescope_fails = telescope_winrates(
        url, decisions, references, active, single_order=single_order, workers=workers
    )
    return refset, turns, references, scores, winrates, score_fails + telescope_fails


def hybrid_sigmoid_value(decisions, result, args) -> list[float]:
    """scoreboard_to_civtelescope: (1 - w) * V_scoreboard + w * Phi_telescope."""
    center, scale, skip = _first_handoff(args)
    validate_sigmoid(center, scale)
    _validate_skip(skip)
    weights = [sigmoid(int(d["turn"]), center, scale) for d in decisions]
    refset, turns, references, scores, winrates, fails = _telescope_parts(
        decisions, args, weights, skip
    )

    values, components = [], []
    for i, d in enumerate(decisions):
        sb_rank, sb_value = scoreboard_component(scores[i], references[i])
        w = weights[i]
        skipped = w < skip
        if skipped:
            tel_rank = tel_value = None
            mixed = sb_value
        else:
            tel_rank = winrates[i]
            tel_value = winrate_to_score(tel_rank, references[i])
            mixed = (1.0 - w) * sb_value + w * tel_value
        values.append(mixed)
        components.append(
            {
                "turn": d["turn"],
                "score_now": scores[i],
                "sb_rank": sb_rank,
                "sb_value": round(sb_value, 6),
                "telescope_rank": tel_rank,
                "telescope_value": None if tel_value is None else round(tel_value, 6),
                "mix_weight": round(w, 8),
                "mix_value": round(mixed, 6),
                "telescope_skipped": skipped,
            }
        )

    append_value_log(
        args,
        {
            "position_id": result.get("position_id"),
            "turns": turns,
            "values": [round(v, 2) for v in values],
            "components": components,
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": fails,
            "telescope_skipped_n": sum(c["telescope_skipped"] for c in components),
            "reward": "scoreboard_to_civtelescope",
            "hybrid_sigmoid_c": center,
            "hybrid_sigmoid_s": scale,
            "hybrid_sigmoid_skip_threshold": skip,
        },
    )
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))
    return values


def sparse_hybrid_sigmoid_value(decisions, result, args) -> list[float]:
    """sparse_to_civtelescope / civtelescope_to_sparse: the CivTelescope potential.

    The sparse blend itself happens in ``train.reward_post``; the scoreboard
    component is computed for the value log only.
    """
    center, scale, skip = _first_handoff(args)
    telescope_first = bool(getattr(args, "civ_sparse_hybrid_telescope_first", False))
    validate_sigmoid(center, scale)
    _validate_skip(skip)
    # w is always the CivTelescope weight; reward_post uses the same flag.
    weights = [
        1.0 - sigmoid(int(d["turn"]), center, scale)
        if telescope_first
        else sigmoid(int(d["turn"]), center, scale)
        for d in decisions
    ]
    refset, turns, references, scores, winrates, fails = _telescope_parts(
        decisions, args, weights, skip
    )

    values, components = [], []
    for i, d in enumerate(decisions):
        sb_rank, sb_value = scoreboard_component(scores[i], references[i])
        w = weights[i]
        skipped = w < skip
        tel_rank = 0.5 if skipped else winrates[i]
        tel_value = winrate_to_score(tel_rank, references[i])
        values.append(tel_value)
        components.append(
            {
                "turn": d["turn"],
                "score_now": scores[i],
                "sb_rank": sb_rank,
                "sb_value": round(sb_value, 6),
                "telescope_rank": tel_rank,
                "telescope_value": round(tel_value, 6),
                "mix_weight": round(w, 8),
                "telescope_skipped": skipped,
            }
        )

    append_value_log(
        args,
        {
            "position_id": result.get("position_id"),
            "turns": turns,
            "values": [round(v, 2) for v in values],
            "components": components,
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": fails,
            "telescope_skipped_n": sum(c["telescope_skipped"] for c in components),
            "reward": "civtelescope_to_sparse"
            if telescope_first
            else "sparse_to_civtelescope",
            "hybrid_sigmoid_c": center,
            "hybrid_sigmoid_s": scale,
            "hybrid_sigmoid_skip_threshold": skip,
        },
    )
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))
    return values


def sparse_hybrid3_value(decisions, result, args) -> list[float]:
    """scoreboard_to_civtelescope_to_sparse, first hand-off: the seam value."""
    c1, s1, skip = _first_handoff(args)
    c2, s2 = _second_handoff(args)
    validate_hybrid3(c1, s1, c2, s2, skip)
    if not getattr(args, "civ_hybrid3_sigmoid", False):
        # Without the second hand-off in reward_post this would silently be a
        # two-signal potential; fail on the first episode instead.
        raise RuntimeError(
            "sparse_hybrid3_value requires civ_hybrid3_sigmoid: true so that "
            "reward_post applies the CivTelescope -> sparse hand-off"
        )
    weights = [seam_weight(int(d["turn"]), c1, s1) for d in decisions]
    refset, turns, references, scores, winrates, fails = _telescope_parts(
        decisions, args, weights, skip
    )

    values, components = [], []
    for i, d in enumerate(decisions):
        sb_rank, sb_value = scoreboard_component(scores[i], references[i])
        w1 = weights[i]
        w2 = sparse_weight(turns[i], c2, s2)
        skipped = w1 < skip
        if skipped:
            tel_rank = tel_value = None
            seam = sb_value
        else:
            tel_rank = winrates[i]
            tel_value = winrate_to_score(tel_rank, references[i])
            seam = (1.0 - w1) * sb_value + w1 * tel_value
        values.append(seam)
        components.append(
            {
                "turn": d["turn"],
                "score_now": scores[i],
                "sb_rank": sb_rank,
                "sb_value": round(sb_value, 6),
                "telescope_rank": tel_rank,
                "telescope_value": None if tel_value is None else round(tel_value, 6),
                "lam_sb": round(1.0 - w1, 8),
                "lam_telescope": round(w1 * (1.0 - w2), 8),
                "lam_sparse": round(w1 * w2, 8),
                "mix_weight": round(w1, 8),
                "hybrid3_w2": round(w2, 8),
                "mix_value": round(seam, 6),
                "telescope_skipped": skipped,
            }
        )

    append_value_log(
        args,
        {
            "position_id": result.get("position_id"),
            "turns": turns,
            "values": [round(v, 2) for v in values],
            "components": components,
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": fails,
            "telescope_skipped_n": sum(c["telescope_skipped"] for c in components),
            "reward": "scoreboard_to_civtelescope_to_sparse",
            "hybrid_sigmoid_c": c1,
            "hybrid_sigmoid_s": s1,
            "hybrid3_c2": c2,
            "hybrid3_s2": s2,
            "hybrid_sigmoid_skip_threshold": skip,
        },
    )
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))
    return values
