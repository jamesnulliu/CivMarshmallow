"""Per-decision rewards and cross-decision GAE for fan-out decision samples.

Every decision of an episode becomes its own training sample.  The potential
Phi_t (CivTelescope, scoreboard, or a hybrid of them) is fixed, not a learned
critic, so advantages are computed once per episode, after it finishes and
before the sibling samples are distributed across data-parallel ranks.  slime's
built-in GAE runs within one sample's token sequence and never connects
siblings; this module is the explicit cross-decision path.

Semantics (one entry per decision, in turn order):

    delta_t = r_t + gamma * (1 - done_t) * Phi_{t+1} - Phi_t
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

* ``done_t = 1`` marks a real episode end (end turn reached, or elimination):
  nothing after t is bootstrapped.
* If the last decision has ``done = 0`` the trajectory was truncated;
  ``bootstrap_value`` stands in for Phi_{T+1}.  For a terminal last step the
  bootstrap is multiplied by 0.

Pure stdlib, no torch.
"""

from __future__ import annotations


def decision_gae(
    rewards: list[float],
    values: list[float],
    dones: list[int],
    *,
    gamma: float,
    lam: float,
    bootstrap_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Return (advantages, returns) per decision.

    ``rewards[t]``/``values[t]``/``dones[t]`` describe decision t of ONE
    episode, in increasing turn order.  ``returns[t] = A_t + Phi_t`` is logged
    only (there is no learned critic to regress on it).
    """
    n = len(rewards)
    if not (n == len(values) == len(dones)):
        raise ValueError(
            f"length mismatch: rewards={n} values={len(values)} dones={len(dones)}"
        )
    if n == 0:
        return [], []
    for d in dones:
        if d not in (0, 1):
            raise ValueError(f"done flags must be 0/1, got {d!r}")
    if any(d == 1 for d in dones[:-1]):
        raise ValueError("done=1 before the last decision: episode must end once")

    advantages = [0.0] * n
    gae = 0.0
    for t in range(n - 1, -1, -1):
        not_done = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < n else bootstrap_value
        delta = rewards[t] + gamma * not_done * next_value - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def decision_rewards(
    decisions: list[dict],
    score_end: float,
    err_penalty: float = 0.0,
) -> list[float]:
    """Per-decision reward vector.

    0 everywhere except the terminal decision, which carries the episode's
    final score.  ``err_penalty`` > 0 subtracts a fixed amount from every
    INVALID decision (``error`` set), so invalid output is priced with a
    stable negative sign instead of riding on potential drift.  Legal passes
    are not penalized: choosing not to act is a valid decision.
    """
    n = len(decisions)
    rewards = [0.0] * (n - 1) + [float(score_end)] if n else []
    if err_penalty:
        for i, d in enumerate(decisions):
            if d.get("error"):
                rewards[i] -= err_penalty
    return rewards
