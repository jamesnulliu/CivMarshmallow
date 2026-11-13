#!/usr/bin/env python3
"""Calibrate the Othello opponent: pick the strongest engine the untrained
policy can still beat now and then.

Against an opponent the untrained policy never beats, every reward starts from
a floor of lost games, and runs that differ only in the reward cannot be told
apart.  The fix is the environment, applied to every reward alike, and chosen
from the untrained policy's results only (no trained run is looked at):

  1. ``ladder`` prints the rungs, strongest first, as launcher arguments;
  2. evaluate the untrained policy once per rung on the validation starts
     (scripts/games/train_policy.sh EVAL=1 with the rung's arguments);
  3. ``select`` checks every probe cell and picks the first rung (strongest
     first) on which the untrained policy wins at least --threshold of its
     games.  Train and evaluate every reward against that opponent, and build
     the CivTelescope banks with the same replay engine
     (scripts/games/build_bank.py --replay-depth / --replay-temp /
     --replay-exact-from), so the oracle labels come from the opponent the
     policy meets.

The default engine (depth 2, temperature 20, exact endgame from 10 empty
squares) is the first rung.  The exact endgame is what makes an opponent hard
to beat for a weak player, so every other rung drops it; depth 1 is a one-ply
positional evaluation, softened by its temperature (move values span roughly
10-150).

Usage:
  python scripts/games/calibrate_opponent.py ladder
  python scripts/games/calibrate_opponent.py select \\
      --probe d2t20x10=runs/games/othello/eval/probe_d2t20x10 \\
      --probe d2t20x0=runs/games/othello/eval/probe_d2t20x0 ... \\
      --starts data/games/othello/starts/val.jsonl --out data/games/othello/opponent.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize

# strongest first: launcher arguments of each rung
LADDER = {
    "d2t20x10": {"GAME_OPP_DEPTH": 2, "GAME_OPP_TEMP": 20.0, "GAME_OPP_EXACT_FROM": 10},
    "d2t20x0": {"GAME_OPP_DEPTH": 2, "GAME_OPP_TEMP": 20.0, "GAME_OPP_EXACT_FROM": 0},
    "d1t20x0": {"GAME_OPP_DEPTH": 1, "GAME_OPP_TEMP": 20.0, "GAME_OPP_EXACT_FROM": 0},
    "d1t60x0": {"GAME_OPP_DEPTH": 1, "GAME_OPP_TEMP": 60.0, "GAME_OPP_EXACT_FROM": 0},
    "d1t200x0": {"GAME_OPP_DEPTH": 1, "GAME_OPP_TEMP": 200.0, "GAME_OPP_EXACT_FROM": 0},
    "d1t1000x0": {
        "GAME_OPP_DEPTH": 1,
        "GAME_OPP_TEMP": 1000.0,
        "GAME_OPP_EXACT_FROM": 0,
    },
}
# launcher argument -> config.yaml key the launcher writes
CONFIG_KEYS = {
    "GAME_OPP_DEPTH": "civ_game_opp_depth",
    "GAME_OPP_TEMP": "civ_game_opp_temp",
    "GAME_OPP_EXACT_FROM": "civ_game_opp_exact_from",
}
DEFAULT_THRESHOLD = 0.15
MIN_VALID_PER_START = 5


def read_config(run_dir: Path) -> dict:
    config = {}
    path = run_dir / "config.yaml"
    if path.is_file():
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip().strip('"')
    return config


def check_cell(run_dir, starts, opponent, games_per_start=8) -> list[str]:
    """Problems of one probe cell (empty list = usable): one group of
    ``games_per_start`` games per start of ``starts``, exactly those starts,
    at least one valid game per start and ``MIN_VALID_PER_START`` on average,
    and the rung's opponent in config.yaml."""
    run_dir = Path(run_dir)
    log = run_dir / "reward_log.jsonl"
    if not log.is_file():
        return [f"{run_dir}: no reward_log.jsonl"]
    problems = []
    want = [row["position_id"] for row in summarize.read_jsonl(starts)]
    seen, valid = {}, 0
    for row in summarize.read_jsonl(log):
        episodes = row.get("episodes") or []
        ids = {e.get("position_id") for e in episodes}
        if len(episodes) != games_per_start or len(ids) != 1:
            problems.append(
                f"{run_dir}: malformed group ({len(episodes)} games, "
                f"ids {sorted(map(str, ids))})"
            )
            continue
        pid = ids.pop()
        if pid in seen:
            problems.append(f"{run_dir}: start {pid} twice")
        seen[pid] = sum(1 for e in episodes if e.get("ok"))
        valid += seen[pid]
    if set(seen) != set(want):
        problems.append(
            f"{run_dir}: starts differ from {starts} (missing "
            f"{len(set(want) - set(seen))}, extra {len(set(seen) - set(want))})"
        )
    if any(n == 0 for n in seen.values()):
        problems.append(f"{run_dir}: a start has no valid game")
    if valid < MIN_VALID_PER_START * len(want):
        problems.append(f"{run_dir}: {valid} valid games for {len(want)} starts")
    config = read_config(run_dir)
    if config.get("civ_game") != "othello":
        problems.append(f"{run_dir}: config.yaml is not an Othello cell")
    for arg, key in CONFIG_KEYS.items():
        got = config.get(key, "10" if key == "civ_game_opp_exact_from" else None)
        if got is None or float(got) != float(opponent[arg]):
            problems.append(f"{run_dir}: {key} is {got}, expected {opponent[arg]}")
    return problems


def cmd_ladder(_args) -> int:
    for rung, env in LADDER.items():
        print(rung, " ".join(f"{k}={v}" for k, v in env.items()))
    return 0


def cmd_select(args) -> int:
    probes = dict(item.split("=", 1) for item in args.probe)
    unknown = set(probes) - set(LADDER)
    if unknown:
        raise SystemExit(f"--probe: unknown rungs {sorted(unknown)}")
    rows, chosen, problems = {}, None, []
    for rung, env in LADDER.items():
        if rung not in probes:
            continue
        cell = Path(probes[rung])
        bad = check_cell(cell, args.starts, env, args.games_per_start)
        if bad:
            problems += bad
            continue
        rec = summarize.summarize_eval(cell, game="othello")
        rows[rung] = {
            k: rec[k]
            for k in ("S", "win_rate", "n_valid_episodes", "n_failed_episodes")
        }
        if chosen is None and rec["win_rate"] >= args.threshold:
            chosen = rung
    if problems:
        raise SystemExit("unusable probe cells:\n  " + "\n  ".join(problems))
    skipped = [r for r in LADDER if r not in probes]
    for rung, rec in rows.items():
        print(
            f"  {rung:10s} S {rec['S']:+7.2f}  win {rec['win_rate']:.3f}  "
            f"({rec['n_valid_episodes']} games)"
        )
    out = {
        "rule": "first rung of the ladder (strongest first) on which the untrained "
        f"policy wins >= {args.threshold:.0%} of the validation games",
        "ladder": list(LADDER),
        "not_probed": skipped,
        "probe": rows,
        "rung": chosen,
        "launcher_args": LADDER[chosen] if chosen else None,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if chosen is None:
        print("no rung reaches the threshold")
        return 1
    print(
        f"chosen: {chosen}  " + " ".join(f"{k}={v}" for k, v in LADDER[chosen].items())
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ladder", help="print every rung as launcher arguments")
    p = sub.add_parser("select", help="pick the opponent from the probe cells")
    p.add_argument(
        "--probe",
        action="append",
        required=True,
        metavar="RUNG=EVAL_DIR",
        help="evaluation cell of the untrained policy against a rung (repeatable)",
    )
    p.add_argument("--starts", required=True, help="the start file the probes played")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--games-per-start", type=int, default=8)
    p.add_argument("--out", help="write the selection as JSON")
    args = ap.parse_args()
    return {"ladder": cmd_ladder, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
