"""Evaluation cells for trained policies and their aggregation.

An evaluation cell is one checkpoint played from every start of a frozen
evaluation bank, ``n_samples`` stochastic episodes per start, with training
switched off (``scripts/eval_policy.sh``). :func:`validate_run` checks the run
directory against that protocol and reduces its per-episode reward log to a cell:

    {"per_position": {position_id: {"scores", "mean", "n_valid", ...}}, ...}

Only episodes that finished (``ok``) are scored. A focal player eliminated by the
game finishes with score 0 and is kept; an episode whose engine seat stalled
(an infrastructure failure) is not a game outcome and is excluded.

Aggregates, all paired per start position:

* terminal score ``S``: mean over positions of the per-position mean score;
* phase gain ``G``: per position, end-checkpoint mean minus start-checkpoint mean
  on the same starts, averaged over positions, then over training seeds;
* paired win rate ``W``: the fraction of (seed, position) pairs on which one
  reward's phase gain is strictly larger than another's, pooled over seeds;
* invalid-action rate: invalid decisions over all decisions of valid episodes.

Games whose start is hopeless for the untrained policy are removed from every
phase by :func:`excluded_games`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

N_SAMPLES = 8
MIN_VALID_PER_POSITION = 5  # cell-level floor: n_positions * this many valid episodes
TRAIN_STEP_MARKER = "step 0: {"


class CellError(AssertionError):
    """An evaluation run directory does not satisfy the evaluation protocol."""


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CellError(message)


def _check_launch(run_dir: Path, n_positions: int, n_samples: int) -> None:
    """The run was rollout-only on one GPU with one group per start."""
    args = (run_dir / "slime_args.txt").read_text().splitlines()
    _require("--debug-rollout-only" in args, "launch is not --debug-rollout-only")
    expected = {
        "--rollout-num-gpus": "1",
        "--rollout-batch-size": str(n_positions),
        "--global-batch-size": str(n_positions * n_samples),
        "--n-samples-per-prompt": str(n_samples),
    }
    for flag, value in expected.items():
        # argparse keeps the last occurrence of a repeated flag
        where = [i for i, arg in enumerate(args[:-1]) if arg == flag]
        _require(bool(where), f"launch is missing {flag}")
        got = args[where[-1] + 1]
        _require(got == value, f"launch has {flag} {got}, expected {value}")
    _require(not list(run_dir.glob("hf_iter*")), "evaluation run saved a checkpoint")
    log = run_dir / "train.log"
    if log.exists():
        text = log.read_text(errors="replace")
        _require(TRAIN_STEP_MARKER not in text, "evaluation run took a training step")


def _check_config(run_dir: Path) -> None:
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    _require(config.get("civ_min_group") == 1, "config does not set civ_min_group: 1")
    _require(
        config.get("civ_enable_thinking") is False, "config does not disable thinking"
    )


def validate_run(
    run_dir: str | Path,
    starts: str | Path | list[dict],
    *,
    n_samples: int = N_SAMPLES,
    min_valid_episodes: int | None = None,
    require_split: str | None = None,
    check_launch: bool = True,
) -> dict:
    """Validate an evaluation run directory and return its cell.

    ``starts`` is the frozen evaluation bank (path or rows). The run must hold
    exactly one reward-log group per start, each with ``n_samples`` raw episodes
    of a single position, at least one valid episode per position, and at least
    ``min_valid_episodes`` valid episodes in total (default
    ``MIN_VALID_PER_POSITION`` per position). Raises :class:`CellError`.
    """
    run_dir = Path(run_dir)
    rows = read_jsonl(starts) if not isinstance(starts, list) else starts
    by_id = {row["position_id"]: row for row in rows}
    expected = sorted(by_id)
    n = len(expected)
    _require(n == len(rows) and n > 0, "starts are empty or repeat a position_id")
    if require_split is not None:
        _require(
            all(row.get("split") == require_split for row in rows),
            f"a start row is not split={require_split}",
        )
    for name in ("reward_log.jsonl", "config.yaml", "rl_starts.jsonl"):
        _require((run_dir / name).exists(), f"missing {run_dir / name}")
    _check_config(run_dir)
    if check_launch:
        _check_launch(run_dir, n, n_samples)

    rl_ids = sorted(
        r["metadata"]["position_id"] for r in read_jsonl(run_dir / "rl_starts.jsonl")
    )
    _require(rl_ids == expected, "rl_starts.jsonl differs from the frozen starts")

    groups = sorted(read_jsonl(run_dir / "reward_log.jsonl"), key=lambda r: r["ts"])
    _require(len(groups) == n, f"expected {n} reward groups, found {len(groups)}")

    per_position: dict[str, dict] = {}
    for group in groups:
        episodes = group.get("episodes") or []
        _require(len(episodes) == n_samples, f"a group has {len(episodes)} episodes")
        pids = {ep.get("position_id") for ep in episodes}
        _require(len(pids) == 1 and None not in pids, "a group mixes positions")
        pid = pids.pop()
        _require(pid not in per_position, f"duplicate group for {pid}")
        valid = [ep for ep in episodes if ep.get("ok")]
        _require(bool(valid), f"position {pid} has no valid episode")
        scores = [float(ep["score_end"]) for ep in valid]
        per_position[pid] = {
            "game_id": by_id.get(pid, {}).get("game_id"),
            "scores": scores,
            "mean": statistics.mean(scores),
            "n_valid": len(valid),
            "n_infra_failures": len(episodes) - len(valid),
            "n_err": sum(int(ep.get("n_err") or 0) for ep in valid),
            "n_decisions": sum(int(ep.get("n_decisions") or 0) for ep in valid),
        }
    _require(
        sorted(per_position) == expected, "reward-log positions differ from starts"
    )

    n_valid = sum(p["n_valid"] for p in per_position.values())
    floor = (
        MIN_VALID_PER_POSITION * n if min_valid_episodes is None else min_valid_episodes
    )
    _require(n_valid >= floor, f"only {n_valid} valid episodes, need {floor}")
    return {
        "run_dir": str(run_dir),
        "n_positions": n,
        "n_samples": n_samples,
        "n_valid_episodes": n_valid,
        "n_infra_failures": sum(p["n_infra_failures"] for p in per_position.values()),
        "position_ids": expected,
        "per_position": {pid: per_position[pid] for pid in expected},
        "invalid_action_rate": invalid_action_rate(per_position.values()),
        "valid": True,
    }


def invalid_action_rate(positions: Iterable[Mapping]) -> float | None:
    """Invalid decisions over all decisions, summed over valid episodes."""
    positions = list(positions)
    decisions = sum(p["n_decisions"] for p in positions)
    if not decisions:
        return None
    return sum(p["n_err"] for p in positions) / decisions


def excluded_games(base_cell: Mapping) -> list[str]:
    """Games on which the untrained policy is eliminated in every valid episode.

    ``base_cell`` is the untrained base model evaluated from the whole-game
    (rem120) starts. A game qualifies when it has at least one valid episode and
    every valid episode ends with score 0. The rule reads only the untrained
    policy, never the trained rewards being compared; the returned game ids are
    removed from every phase.
    """
    out = set()
    for row in base_cell["per_position"].values():
        scores = row["scores"]
        if scores and all(score == 0 for score in scores):
            out.add(row["game_id"])
    return sorted(out)


def position_means(
    cell: Mapping, exclude_games: Iterable[str] = ()
) -> dict[str, float]:
    """Per-position mean score, without the positions of ``exclude_games``."""
    drop = set(exclude_games)
    return {
        pid: row["mean"]
        for pid, row in cell["per_position"].items()
        if row.get("game_id") not in drop
    }


def terminal_score(cell: Mapping, exclude_games: Iterable[str] = ()) -> float:
    """``S``: mean over positions of the per-position mean score."""
    return statistics.mean(position_means(cell, exclude_games).values())


def phase_gains(
    start_cell: Mapping, end_cell: Mapping, exclude_games: Iterable[str] = ()
) -> dict[str, float]:
    """Per-position gain, end-checkpoint mean minus start-checkpoint mean.

    Both cells must cover the same starts.
    """
    start = position_means(start_cell, exclude_games)
    end = position_means(end_cell, exclude_games)
    if set(start) != set(end):
        raise CellError("start and end cells cover different positions")
    return {pid: end[pid] - start[pid] for pid in sorted(end)}


def paired_win_rate(
    gains_a: Mapping[str, Mapping[str, float]],
    gains_b: Mapping[str, Mapping[str, float]],
) -> dict:
    """``W`` of reward A over reward B from per-seed, per-position gains.

    Counts the (seed, position) pairs on which A's gain is strictly larger,
    pooled over the seeds both rewards have.
    """
    seeds = sorted(set(gains_a) & set(gains_b))
    wins = ties = n = 0
    for seed in seeds:
        a, b = gains_a[seed], gains_b[seed]
        if set(a) != set(b):
            raise CellError(f"seed {seed}: the two rewards cover different positions")
        for pid in a:
            wins += a[pid] > b[pid]
            ties += a[pid] == b[pid]
            n += 1
    return {
        "wins": wins,
        "ties": ties,
        "n": n,
        "rate": wins / n if n else None,
        "seeds": seeds,
    }


def _per_seed_summary(values: Mapping[str, float | None]) -> dict:
    present = [v for v in values.values() if v is not None]
    return {
        "per_seed": dict(values),
        "mean": statistics.mean(present) if present else None,
    }


def summarize(
    cells: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping]]]],
    exclude_games: Iterable[str] = (),
) -> dict:
    """Aggregate ``{reward: {seed: {phase: {"start": cell, "end": cell}}}}``.

    Returns, per reward and phase, ``G``, ``S`` (of the end checkpoint) and the
    end checkpoint's invalid-action rate, each per seed and averaged over seeds;
    and per phase, ``W`` for every ordered pair of rewards.
    """
    exclude = set(exclude_games)
    gains: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    rewards: dict[str, dict[str, dict]] = {}
    n_positions: dict[str, int] = {}
    for reward, by_seed in cells.items():
        per_phase: dict[str, dict[str, dict]] = {}
        for seed, by_phase in by_seed.items():
            for phase, pair in by_phase.items():
                g = phase_gains(pair["start"], pair["end"], exclude)
                gains.setdefault(phase, {}).setdefault(reward, {})[seed] = g
                n_positions.setdefault(phase, len(g))
                if n_positions[phase] != len(g):
                    raise CellError(f"{phase}: cells cover different numbers of starts")
                slot = per_phase.setdefault(phase, {"G": {}, "S": {}, "invalid": {}})
                slot["G"][seed] = statistics.mean(g.values())
                slot["S"][seed] = terminal_score(pair["end"], exclude)
                kept = [
                    row
                    for row in pair["end"]["per_position"].values()
                    if row.get("game_id") not in exclude
                ]
                slot["invalid"][seed] = invalid_action_rate(kept)
        rewards[reward] = {
            phase: {
                "G": _per_seed_summary(slot["G"]),
                "S": _per_seed_summary(slot["S"]),
                "invalid_action_rate": _per_seed_summary(slot["invalid"]),
            }
            for phase, slot in per_phase.items()
        }

    win_rates: dict[str, dict[str, dict]] = {}
    for phase, by_reward in gains.items():
        for a in by_reward:
            for b in by_reward:
                if a != b:
                    win_rates.setdefault(phase, {})[f"{a}_over_{b}"] = paired_win_rate(
                        by_reward[a], by_reward[b]
                    )
    return {
        "excluded_games": sorted(exclude),
        "n_positions": n_positions,
        "rewards": rewards,
        "paired_win_rate": win_rates,
    }
