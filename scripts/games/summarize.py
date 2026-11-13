#!/usr/bin/env python3
"""Readouts of the Othello and Catan policy runs, from their run directories.

Usage:
  python scripts/games/summarize.py train runs/games/othello/civtelescope/seed43 [more run dirs ...]
  python scripts/games/summarize.py eval  runs/games/othello/eval/civtelescope_seed43 [more ...]
  python scripts/games/summarize.py gain  --entry <eval dir> --grad <eval dir> \\
      [--entry-b <eval dir> --grad-b <eval dir>]
  (every mode: --out result.json)

train  per iteration: mean final score of the kept episodes, win rate and the
       share of decisions with an invalid reply.  An iteration is
       --groups-per-iter (8) consecutive KEPT groups of reward_log.jsonl in
       time order; a group with a failed episode is dropped and resampled by
       the rollout, and is skipped here.
eval   terminal score S = mean over starts of the start's mean final score
       (Othello: final disc differential, Catan: final victory points),
       per-start means, win rate over valid episodes, and the valid-episode
       count.  A win is: Othello, final differential > 0; Catan, the focal
       seat reached the winning victory points (10; the game stops at the
       first seat to get there).  The game is read from the run's config.yaml
       (``civ_game``), or given with --game.
gain   phase gain G = S_grad - S_entry, per start and its mean; with a second
       (entry-b, grad-b) pair also the paired win rate W_{a,b} = share of
       starts on which run a gains more than run b.

Episode validity is the rollout's own flag (``ok``): a game the generation
server aborted is not a result and is excluded, as in the Freeciv evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def reward_rows(run_dir):
    rows = read_jsonl(Path(run_dir) / "reward_log.jsonl")
    if not rows:
        raise SystemExit(f"{run_dir}: no reward_log.jsonl rows")
    return sorted(rows, key=lambda row: row["ts"])


WIN = {"othello": lambda score: score > 0, "catan": lambda score: score >= 10}


def game_of(run_dir, game=None):
    """The run's game: --game, else ``civ_game`` in its config.yaml, else othello."""
    if game:
        return game
    config = Path(run_dir) / "config.yaml"
    if config.is_file():
        for line in config.read_text().splitlines():
            if line.startswith("civ_game:"):
                return line.split(":", 1)[1].strip().strip('"')
    return "othello"


def summarize_train(run_dir, groups_per_iter=8, game=None):
    won = WIN[game_of(run_dir, game)]
    kept = [
        row
        for row in reward_rows(run_dir)
        if row["episodes"] and all(e.get("ok") for e in row["episodes"])
    ]
    errors = {}  # episode_id -> (n_invalid, n_decisions)
    for row in read_jsonl(Path(run_dir) / "decision_log.jsonl"):
        decisions = row.get("decisions") or []
        errors[row["episode_id"]] = (
            sum(1 for d in decisions if d.get("err")),
            len(decisions),
        )
    iterations = []
    for start in range(0, len(kept), groups_per_iter):
        chunk = kept[start : start + groups_per_iter]
        episodes = [e for row in chunk for e in row["episodes"]]
        scores = [float(e["score_end"]) for e in episodes]
        n_err = sum(errors.get(e["episode_id"], (0, 0))[0] for e in episodes)
        n_dec = sum(errors.get(e["episode_id"], (0, 0))[1] for e in episodes)
        iterations.append(
            {
                "iteration": len(iterations),
                "n_groups": len(chunk),
                "complete": len(chunk) == groups_per_iter,
                "n_episodes": len(episodes),
                "mean_score_end": round(statistics.mean(scores), 4),
                "sd_score_end": round(statistics.pstdev(scores), 4),
                "win_rate": round(sum(won(s) for s in scores) / len(scores), 4),
                "invalid_decision_rate": round(n_err / n_dec, 4) if n_dec else None,
                "decisions_per_episode": round(n_dec / len(episodes), 1)
                if n_dec
                else None,
            }
        )
    return {
        "run_dir": str(run_dir),
        "game": game_of(run_dir, game),
        "groups_per_iter": groups_per_iter,
        "n_kept_groups": len(kept),
        "iterations": iterations,
    }


def summarize_eval(run_dir, game=None):
    won = WIN[game_of(run_dir, game)]
    per_start, n_failed = {}, 0
    for row in reward_rows(run_dir):
        for episode in row["episodes"]:
            if not episode.get("ok"):
                n_failed += 1
                continue
            per_start.setdefault(episode["position_id"], []).append(
                float(episode["score_end"])
            )
    if not per_start:
        raise SystemExit(f"{run_dir}: no valid episode")
    scores = [s for values in per_start.values() for s in values]
    means = {pid: statistics.mean(values) for pid, values in sorted(per_start.items())}
    return {
        "run_dir": str(run_dir),
        "game": game_of(run_dir, game),
        "n_starts": len(per_start),
        "n_valid_episodes": len(scores),
        "n_failed_episodes": n_failed,
        "S": round(statistics.mean(means.values()), 4),
        "win_rate": round(sum(won(s) for s in scores) / len(scores), 4),
        "per_start": {
            pid: {"n_valid": len(per_start[pid]), "mean": round(mean, 4)}
            for pid, mean in means.items()
        },
    }


def start_means(run_dir):
    return {
        pid: row["mean"] for pid, row in summarize_eval(run_dir)["per_start"].items()
    }


def gains(entry_dir, grad_dir):
    """Per-start gain g_p = grad - entry over the starts both cells evaluated."""
    entry, grad = start_means(entry_dir), start_means(grad_dir)
    common = sorted(set(entry) & set(grad))
    if not common:
        raise SystemExit(f"{entry_dir} and {grad_dir} share no start")
    return {pid: grad[pid] - entry[pid] for pid in common}, len(entry), len(grad)


def summarize_gain(entry, grad, entry_b=None, grad_b=None):
    g_a, n_entry, n_grad = gains(entry, grad)
    out = {
        "entry": str(entry),
        "grad": str(grad),
        "n_common_starts": len(g_a),
        "n_entry_starts": n_entry,
        "n_grad_starts": n_grad,
        "G": round(statistics.mean(g_a.values()), 4),
        "share_of_starts_improved": round(
            sum(g > 0 for g in g_a.values()) / len(g_a), 4
        ),
        "per_start_gain": {pid: round(g, 4) for pid, g in g_a.items()},
    }
    if entry_b and grad_b:
        g_b, _, _ = gains(entry_b, grad_b)
        common = sorted(set(g_a) & set(g_b))
        if not common:
            raise SystemExit("the two runs share no start")
        out["b"] = {
            "entry": str(entry_b),
            "grad": str(grad_b),
            "G": round(statistics.mean(g_b.values()), 4),
        }
        out["paired_win_rate_a_over_b"] = round(
            sum(g_a[pid] > g_b[pid] for pid in common) / len(common), 4
        )
        out["mean_gain_difference_a_minus_b"] = round(
            statistics.mean(g_a[pid] - g_b[pid] for pid in common), 4
        )
        out["n_paired_starts"] = len(common)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("run_dirs", nargs="+")
    p_train.add_argument("--groups-per-iter", type=int, default=8)
    p_train.add_argument("--game", choices=sorted(WIN))
    p_train.add_argument("--out")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("run_dirs", nargs="+")
    p_eval.add_argument("--game", choices=sorted(WIN))
    p_eval.add_argument("--out")
    p_gain = sub.add_parser("gain")
    p_gain.add_argument("--entry", required=True)
    p_gain.add_argument("--grad", required=True)
    p_gain.add_argument("--entry-b")
    p_gain.add_argument("--grad-b")
    p_gain.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "train":
        result = [
            summarize_train(d, args.groups_per_iter, args.game) for d in args.run_dirs
        ]
        for run in result:
            print(run["run_dir"])
            for it in run["iterations"]:
                tail = "" if it["complete"] else "  (partial)"
                print(
                    f"  iter {it['iteration']:2d}: score {it['mean_score_end']:+7.2f}  "
                    f"win {it['win_rate']:.2f}  invalid {it['invalid_decision_rate']}{tail}"
                )
    elif args.mode == "eval":
        result = [summarize_eval(d, args.game) for d in args.run_dirs]
        for cell in result:
            print(
                f"{cell['run_dir']}: S {cell['S']:+.2f}  win {cell['win_rate']:.3f}  "
                f"{cell['n_starts']} starts, {cell['n_valid_episodes']} valid episodes, "
                f"{cell['n_failed_episodes']} failed"
            )
    else:
        if bool(args.entry_b) != bool(args.grad_b):
            ap.error("--entry-b and --grad-b go together")
        result = summarize_gain(args.entry, args.grad, args.entry_b, args.grad_b)
        print(
            f"G = {result['G']:+.2f} over {result['n_common_starts']} starts "
            f"({result['share_of_starts_improved']:.0%} improved)"
        )
        if "b" in result:
            print(
                f"G_b = {result['b']['G']:+.2f}; W(a over b) = "
                f"{result['paired_win_rate_a_over_b']:.0%} on {result['n_paired_starts']} starts"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
