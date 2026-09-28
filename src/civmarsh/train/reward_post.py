"""Reward post-processing for per-decision fan-out samples (slime hook).

slime's default post-processing whitens by reshaping the flat reward list to
``(-1, n_samples_per_prompt)``.  With per-decision fan-out that list holds one
entry per DECISION and episodes have different lengths, so the reshape either
fails or whitens across the whole batch.  ``post_process``
(``--custom-reward-post-process-path``) whitens at the EPISODE level instead:

  * group = the episodes sharing one ``position_id`` (the GRPO group);
  * each episode contributes its reward once, however many decisions it has;
  * the whitened episode advantage is broadcast to every sibling.

For dense rewards the advantage function reads the precomputed GAE from
``train_metadata`` instead; this hook still runs and additionally, once per
training step and before the data-parallel partition (so every rank sees the
same statistics):

  * blends the sparse position-group advantage into the CivTelescope GAE
    (``civ_sparse_hybrid_sigmoid``) or into the scoreboard/CivTelescope seam
    GAE (``civ_hybrid3_sigmoid``); see ``civmarsh.rewards.hybrid``;
  * mean-centres the precomputed advantages over the whole step batch and
    divides by the batch standard deviation (``civ_adv_norm``, default true);
    ``return`` stays in raw score units (logged only);
  * triggers the reference-set refresh (``rewards.reference_set.maybe_refresh``),
    so the reference set changes only between steps.
"""

from __future__ import annotations

from collections import defaultdict

from civmarsh.rewards.hybrid import (
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_S1,
    DEFAULT_S2,
    sigmoid,
    sparse_weight,
    validate_sigmoid,
)
from civmarsh.rewards.reference_set import maybe_refresh


def _episode_id(sample) -> str:
    return (sample.metadata or {}).get("episode_id") or f"sample-{sample.index}"


def post_process(args, samples):
    """Return (raw_rewards, rewards), both per sample; rewards are GRPO-style
    whitened within each start position's episode group."""
    raw_rewards = [float(s.reward if s.reward is not None else 0.0) for s in samples]

    # one reward per EPISODE (all siblings carry the same episode reward)
    episode_reward: dict[str, float] = {}
    episode_position: dict[str, str] = {}
    for s, r in zip(samples, raw_rewards, strict=True):
        eid = _episode_id(s)
        episode_reward[eid] = r
        episode_position[eid] = str((s.metadata or {}).get("position_id", "pos"))

    by_position: dict[str, list[str]] = defaultdict(list)
    for eid, pos in episode_position.items():
        by_position[pos].append(eid)

    episode_adv: dict[str, float] = {}
    for eids in by_position.values():
        vals = [episode_reward[e] for e in eids]
        mean = sum(vals) / len(vals)
        centered = [v - mean for v in vals]
        if getattr(args, "grpo_std_normalization", True) and len(vals) > 1:
            var = sum(c * c for c in centered) / len(centered)
            std = var**0.5
            centered = [c / (std + 1e-6) for c in centered]
        for e, c in zip(eids, centered, strict=True):
            episode_adv[e] = c

    rewards = [episode_adv[_episode_id(s)] for s in samples]

    sparse_hybrid = getattr(args, "civ_sparse_hybrid_sigmoid", False)
    hybrid3 = getattr(args, "civ_hybrid3_sigmoid", False)
    if sparse_hybrid and hybrid3:
        raise RuntimeError(
            "civ_sparse_hybrid_sigmoid and civ_hybrid3_sigmoid select different "
            "rewards and must not both be set"
        )
    if sparse_hybrid:
        _mix_sparse_hybrid_advantages(args, samples, episode_adv)
    if hybrid3:
        _mix_hybrid3_advantages(args, samples, episode_adv)
    if getattr(args, "civ_adv_norm", True):
        _normalize_advantages(samples)
    maybe_refresh(args)
    return raw_rewards, rewards


def _decision_turn_and_advantage(sample, flag):
    train_metadata = sample.train_metadata
    if not train_metadata or train_metadata.get("advantage") is None:
        raise RuntimeError(f"{flag} requires a precomputed train_metadata advantage")
    turn = train_metadata.get("turn")
    if not isinstance(turn, int) or isinstance(turn, bool):
        raise RuntimeError(  # noqa: TRY004 - a misconfigured run, not a caller bug
            f"{flag} requires an integer decision turn, got {turn!r}"
        )
    return train_metadata, turn


def _mix_sparse_hybrid_advantages(args, samples, episode_adv):
    """A(t) = (1 - w) * A_sparse_group + w * A_telescope(t).

    The value hook runs once per episode, before sibling outcomes exist; here
    ``episode_adv`` is exactly the sparse GRPO advantage and train_metadata
    still holds the CivTelescope-only GAE.  ``w`` is the CivTelescope weight;
    ``civ_sparse_hybrid_telescope_first`` flips it to fall with the turn and
    must match the flag seen by ``sparse_hybrid_sigmoid_value``.
    """
    center = float(getattr(args, "civ_hybrid_sigmoid_c", DEFAULT_C1))
    scale = float(getattr(args, "civ_hybrid_sigmoid_s", DEFAULT_S1))
    validate_sigmoid(center, scale)
    telescope_first = bool(getattr(args, "civ_sparse_hybrid_telescope_first", False))
    for sample in samples:
        sparse_advantage = episode_adv[_episode_id(sample)]
        train_metadata, turn = _decision_turn_and_advantage(
            sample, "civ_sparse_hybrid_sigmoid"
        )
        weight = sigmoid(turn, center, scale)
        if telescope_first:
            weight = 1.0 - weight
        telescope_advantage = float(train_metadata["advantage"])
        train_metadata["sparse_group_advantage"] = sparse_advantage
        train_metadata["telescope_advantage"] = telescope_advantage
        train_metadata["hybrid_mix_weight"] = weight
        train_metadata["advantage"] = (
            1.0 - weight
        ) * sparse_advantage + weight * telescope_advantage


def _mix_hybrid3_advantages(args, samples, episode_adv):
    """Second hand-off of the three-signal reward:

        A(t) = (1 - w2(t)) * A_seam(t) + w2(t) * A_sparse_group

    The value hook already blended scoreboard and CivTelescope into the seam
    value, and the cross-decision GAE turned it into A_seam.  The three layer
    weights are recorded per decision for auditing.
    """
    center1 = float(getattr(args, "civ_hybrid_sigmoid_c", DEFAULT_C1))
    scale1 = float(getattr(args, "civ_hybrid_sigmoid_s", DEFAULT_S1))
    center2 = float(getattr(args, "civ_hybrid3_c2", DEFAULT_C2))
    scale2 = float(getattr(args, "civ_hybrid3_s2", DEFAULT_S2))
    validate_sigmoid(center1, scale1)
    validate_sigmoid(center2, scale2)
    for sample in samples:
        sparse_advantage = episode_adv[_episode_id(sample)]
        train_metadata, turn = _decision_turn_and_advantage(
            sample, "civ_hybrid3_sigmoid"
        )
        w1 = sigmoid(turn, center1, scale1)
        w2 = sparse_weight(turn, center2, scale2)
        seam_advantage = float(train_metadata["advantage"])
        train_metadata["seam_advantage"] = seam_advantage
        train_metadata["sparse_group_advantage"] = sparse_advantage
        train_metadata["hybrid3_w2"] = w2
        train_metadata["hybrid3_lam_sb"] = 1.0 - w1
        train_metadata["hybrid3_lam_telescope"] = w1 * (1.0 - w2)
        train_metadata["hybrid3_lam_sparse"] = w1 * w2
        train_metadata["advantage"] = (
            1.0 - w2
        ) * seam_advantage + w2 * sparse_advantage


def _normalize_advantages(samples):
    """Mean-centre every precomputed decision advantage across the step batch,
    then divide by the batch (population) std.  Mutates train_metadata in
    place, before the data-parallel partition carries it to the ranks."""
    advs = [
        s.train_metadata["advantage"]
        for s in samples
        if s.train_metadata and s.train_metadata.get("advantage") is not None
    ]
    if len(advs) < 2:
        return
    mean = sum(advs) / len(advs)
    std = (sum((a - mean) ** 2 for a in advs) / len(advs)) ** 0.5
    scale = std + 1e-6
    for s in samples:
        tm = s.train_metadata
        if tm and tm.get("advantage") is not None:
            tm["advantage"] = (tm["advantage"] - mean) / scale
