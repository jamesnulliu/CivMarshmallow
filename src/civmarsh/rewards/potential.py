"""CivTelescope potential: a decision's value measured against a reference set.

Each decision's value input (``SpatialSnapshot.render_value_input``, captured
live from the policy's fog-of-war observation) is compared pairwise with the M
reference positions of the nearest turn bucket by a served CivTelescope model.
The mean win probability w is mapped into final-score units by interpolating
the reference positions' sorted final scores at quantile w, so the potential
Phi and the terminal reward (the final score) are commensurate inside GAE.

Reference-set file (jsonl): one row per reference position,
``{turn, focal, rendering, end_score}``.  The file is live state:
``reference_set.refresh`` replaces it atomically between training steps, and
the loader keys its cache on the file's mtime, so every process picks up the
new set on its next episode without any signalling.

Value functions here and in ``scoreboard`` / ``hybrid`` share one contract:
``fn(decisions, result, args) -> list[float]``, one value per decision in
final-score units.  ``decisions`` are the rollout's per-decision records
(``turn``, ``focal``, ``value_input``); ``result`` is the episode outcome.

Config keys (read from the slime args):
  civ_value_url           CivTelescope server ``/generate`` endpoint
  civ_value_refset        reference-set jsonl (live; rewritten by the refresh)
  civ_value_refset_idx    indices into each bucket's final-score-sorted set,
                          i.e. the M reference positions used (default: all)
  civ_value_single_order  compare with the decision as position A only
                          (default true)
  civ_value_concurrency   global in-flight request cap (default 64)
  civ_value_log           optional jsonl, one line per episode
  civ_value_harvest       optional jsonl: each episode's renderings at the
                          bucket turns and at its last decision, with the
                          final score; the reference-set refresh draws from it
"""

from __future__ import annotations

import json
import os
import statistics
import threading
from functools import lru_cache

from civmarsh.civtelescope import client
from civmarsh.civtelescope.prompt import PAIRWISE_PROMPT

DEFAULT_CONCURRENCY = client.DEFAULT_CONCURRENCY
LOG_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def _load_at(path, mtime_ns):
    by_turn = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_turn.setdefault(r["turn"], []).append(r)
    return {
        t: sorted(by_turn[t], key=lambda r: r["end_score"]) for t in sorted(by_turn)
    }


def load_reference_set(path) -> dict[int, list[dict]]:
    """{bucket turn: reference rows sorted by final score}, cached per mtime."""
    return _load_at(path, os.stat(path).st_mtime_ns)


def select_references(refset: dict, turn: int, idx=None) -> list[dict]:
    """The reference positions of the bucket nearest to ``turn`` (ties go to
    the earlier bucket), optionally subset by ``idx``."""
    bucket = refset[min(sorted(refset), key=lambda t: abs(t - turn))]
    if idx:
        bucket = [bucket[j] for j in idx]
    return bucket


def winrate_to_score(w: float, references: list[dict]) -> float:
    """Monotone map from winrate to final-score units: interpolate the
    references' ascending final scores at quantile w."""
    scores = [r["end_score"] for r in references]
    x = w * (len(scores) - 1)
    i = min(int(x), len(scores) - 2)
    frac = x - i
    return scores[i] * (1 - frac) + scores[i + 1] * frac


def value_config(args):
    """(url, refset path, refset idx, single_order, concurrency) from args."""
    return (
        args.civ_value_url,
        args.civ_value_refset,
        getattr(args, "civ_value_refset_idx", None),
        bool(getattr(args, "civ_value_single_order", True)),
        int(getattr(args, "civ_value_concurrency", DEFAULT_CONCURRENCY)),
    )


def telescope_winrates(url, decisions, references, indices, *, single_order, workers):
    """Mean P(decision better than reference) for each decision in ``indices``.

    ``references[i]`` is decision i's reference list.  With ``single_order``
    the decision is always position A; otherwise the swapped order is also
    scored and inverted.  A decision whose calls all failed gets 0.5.
    Returns ({index: winrate}, number of decisions that fell back to 0.5).
    """
    jobs = []  # (decision index, prompt, inverted)
    for i in indices:
        d = decisions[i]
        for ref in references[i]:
            jobs.append(
                (
                    i,
                    PAIRWISE_PROMPT.format(
                        focal_a=d["focal"],
                        rendering_a=d["value_input"],
                        focal_b=ref["focal"],
                        rendering_b=ref["rendering"],
                    ),
                    False,
                )
            )
            if not single_order:
                jobs.append(
                    (
                        i,
                        PAIRWISE_PROMPT.format(
                            focal_a=ref["focal"],
                            rendering_a=ref["rendering"],
                            focal_b=d["focal"],
                            rendering_b=d["value_input"],
                        ),
                        True,
                    )
                )
    probs = client.preferences(url, [job[1] for job in jobs], workers=workers)
    per_decision = {i: [] for i in indices}
    for (i, _prompt, inverted), p in zip(jobs, probs):
        if p is not None:
            per_decision[i].append(1.0 - p if inverted else p)
    winrates, failed = {}, 0
    for i in indices:
        if per_decision[i]:
            winrates[i] = statistics.mean(per_decision[i])
        else:
            winrates[i] = 0.5
            failed += 1
    return winrates, failed


def append_value_log(args, row: dict) -> None:
    path = getattr(args, "civ_value_log", None)
    if path:
        with LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def append_harvest(args, decisions, result, keep) -> None:
    """Append this episode's renderings at the ``keep`` indices to the harvest.

    An engine seat stall (``infra_stall``) never reaches the value functions,
    but the gate is explicit: its placeholder final score of 0 must not enter
    the reference-set candidate pool.
    """
    path = getattr(args, "civ_value_harvest", None)
    if not path or result.get("infra_stall"):
        return
    row = {
        "position_id": result.get("position_id"),
        "score_end": result.get("score_end"),
        "end_turn": result.get("end_turn"),
        "eliminated": result.get("eliminated"),
        "control_lost": result.get("control_lost"),
        "elim_class": result.get("elim_class"),
        "last_decision_turn": decisions[-1]["turn"],
        "decisions": [
            {
                "turn": decisions[i]["turn"],
                "focal": decisions[i]["focal"],
                "rendering": decisions[i]["value_input"],
            }
            for i in keep
        ],
    }
    with LOG_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def bucket_and_last(decisions, bucket_turns) -> list[int]:
    """Harvest indices: decisions at a reference bucket turn, plus the last."""
    bucket_turns = set(bucket_turns)
    last = len(decisions) - 1
    return [
        i for i, d in enumerate(decisions) if d["turn"] in bucket_turns or i == last
    ]


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
    )
    values = [
        winrate_to_score(winrates[i], references[i]) for i in range(len(decisions))
    ]

    append_value_log(
        args,
        {
            "turns": [d["turn"] for d in decisions],
            "values": [round(v, 2) for v in values],
            "score_end": result.get("score_end"),
            "elim_class": result.get("elim_class"),
            "value_fails": failed,
        },
    )
    append_harvest(args, decisions, result, bucket_and_last(decisions, refset))
    return values


class _WithoutHarvest:
    """Read-only view of the args with ``civ_value_harvest`` masked out."""

    def __init__(self, args):
        self._args = args

    def __getattr__(self, name):
        if name == "civ_value_harvest":
            return None
        return getattr(self._args, name)


def terminal_telescope_value(decisions, result, args) -> float:
    """Phi(s_T) of the final decision only, as one scalar.

    Used by ``terminal_civtelescope``, where the episode reward is this value
    instead of the final score.  It runs ``telescope_value`` on the last
    decision alone, so it is the same potential as the dense arm (same
    reference selection, prompt and winrate-to-score map).  Harvesting is
    masked: a one-decision row would pollute the refresh candidate pool.
    """
    if not decisions:
        raise ValueError("terminal_telescope_value: no decisions")
    return float(telescope_value([decisions[-1]], result, _WithoutHarvest(args))[0])
