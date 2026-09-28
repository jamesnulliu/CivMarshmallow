"""Custom advantage function for dense rewards (slime hook).

slime wiring: ``--custom-advantage-function-path
civmarsh.train.advantage.broadcast_decision_advantages``.

The cross-decision GAE is computed at episode end inside
``train.rollout.generate`` (before the data-parallel partition) and stored per
decision in ``train_metadata["advantage"]`` / ``["return"]``.  This hook only
broadcasts those scalars over each decision's response tokens; the loss mask
restricts the loss to the action tokens, as with GRPO's whole-sequence
broadcast.

Requires the patched slime (``scripts/slime/slime.patch``) that carries the
per-sample ``metadata`` through the data-parallel split; fails loudly if it is
missing rather than training on garbage.
"""

from __future__ import annotations


def broadcast_decision_advantages(args, rollout_data) -> None:
    import torch

    metadata = rollout_data.get("metadata")
    assert metadata is not None, (
        "rollout_data has no 'metadata': train_metadata was not carried through "
        "the data-parallel split, so per-decision advantages cannot be recovered"
    )
    kl = rollout_data["kl"]
    assert len(metadata) == len(kl), (
        f"metadata/kl length mismatch: {len(metadata)} vs {len(kl)}"
    )

    advantages, returns = [], []
    for md, k in zip(metadata, kl, strict=True):
        adv = md.get("advantage") if md else None
        ret = md.get("return") if md else None
        assert adv is not None and ret is not None, (
            f"decision sample missing precomputed advantage/return "
            f"(train_metadata={md!r}); run the rollout with civ_value_fn_path set"
        )
        advantages.append(torch.full_like(k, float(adv), dtype=torch.float32))
        returns.append(torch.full_like(k, float(ret), dtype=torch.float32))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
