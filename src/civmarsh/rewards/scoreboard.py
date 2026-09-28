"""Scoreboard potential: the focal player's current in-game score.

``score_now_value`` (reward ``scoreboard``) sets Phi_t to the score read from
decision t's own value-input DIGEST, with the same contract and the same
value-log / harvest lines as ``potential.telescope_value``; no CivTelescope
server is involved.  A decision whose rendering has no parseable score carries
the last seen score forward (0.0 before the first) and is counted in
``value_fails``.

The helpers below also express the scoreboard in reference-set units, which is
how the hybrid potentials blend it with CivTelescope: the decision's current
score is ranked among the reference positions' CURRENT scores, and that rank is
read off the references' FINAL scores (``scoreboard_component``).
"""

from __future__ import annotations

from civmarsh.utils.freeciv import SCORE_RE

from .potential import append_harvest, append_value_log, winrate_to_score


def score_nows(decisions) -> tuple[list[float], int]:
    """Current score per decision (carried forward when unparseable), and the
    number of decisions that had no parseable score."""
    values, fails, last = [], 0, 0.0
    for d in decisions:
        m = SCORE_RE.search(d.get("value_input") or "")
        if m:
            last = float(m.group(1))
        else:
            fails += 1
        values.append(last)
    return values, fails


def reference_score_now(reference: dict) -> float | None:
    """A reference position's current score, from its rendering; None if absent."""
    m = SCORE_RE.search(reference.get("rendering") or "")
    return float(m.group(1)) if m else None


def score_rank(score_now: float, reference_score_nows: list[float]) -> float:
    """Quantile w in [0, 1] of ``score_now`` among the references' current scores.

    ``reference_score_nows`` is ascending and may contain ties; ties take the
    average rank, values in between are interpolated linearly, and values
    outside the range clamp to the ends.
    """
    s = reference_score_nows
    n = len(s)
    if n < 2:
        return 0.5
    if score_now <= s[0]:
        return 0.0
    if score_now >= s[-1]:
        return 1.0
    n_less = sum(1 for x in s if x < score_now)
    n_eq = sum(1 for x in s if x == score_now)
    if n_eq:
        pos = n_less + (n_eq - 1) / 2.0
    else:
        i = n_less - 1  # s[i] < score_now < s[i + 1]
        pos = i + (score_now - s[i]) / (s[i + 1] - s[i])
    return pos / (n - 1)


def scoreboard_component(
    score_now: float, references: list[dict]
) -> tuple[float, float]:
    """(rank, value) of the scoreboard in reference-set final-score units.

    References without a parseable current score are dropped from both the
    ranking basis and the value scale; with fewer than two left the rank is
    0.5 on the full set.
    """
    kept = [(r, reference_score_now(r)) for r in references]
    kept = [(r, s) for r, s in kept if s is not None]
    if len(kept) < 2:
        return 0.5, winrate_to_score(0.5, references)
    basis = sorted(s for _, s in kept)
    scale = sorted((r for r, _ in kept), key=lambda r: r["end_score"])
    rank = score_rank(score_now, basis)
    return rank, winrate_to_score(rank, scale)


def score_now_value(decisions, result, args) -> list[float]:
    """Phi_t = the focal player's current score at decision t."""
    values, fails = score_nows(decisions)
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
    # No reference set here: harvest every tenth turn, which coincides with
    # the reference-set bucket turns, plus the last decision.
    last = len(decisions) - 1
    keep = [i for i, d in enumerate(decisions) if d["turn"] % 10 == 0 or i == last]
    append_harvest(args, decisions, result, keep)
    return values
