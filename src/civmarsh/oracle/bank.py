"""Replay-oracle position banks for Freeciv.

A bank is built in three stages, each resumable:

1. `generate_games`: seeded AI-only games (every seat an engine AI), each
   autosaving every `every` turns, so one pass yields a series of positions.
2. `select_positions`: the snapshots inside a turn window, with the focal
   player's current score and the real game's final score.
3. `label_positions`: the replay oracle. From every position, K reseeded
   branches play on to turn `until` with every player at one AI skill level;
   the label is the mean of the focal player's K terminal scores, with its
   standard error sd / sqrt(K). A position is labeled only when all K branches
   report a focal score.

The focal player is player_id 0. Save paths in game, position and label records
are relative to the pool directory the games were generated in.
"""

from __future__ import annotations

import json
import shutil
import statistics
import threading
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from civmarsh.utils.freeciv import focal_player
from civmarsh.utils.io import iter_jsonl

# Server launches that lose the free-port race to a concurrent worker are
# retried this many times in all.
LAUNCH_ATTEMPTS = 5


@dataclass(frozen=True)
class BankPreset:
    """A bank's settings: games, position window and oracle."""

    ruleset: str | None  # None = the server default ruleset (civ2civ3)
    endturn: int
    min_turn: int
    max_turn: int
    splits: dict[str, Sequence[int]]  # split name -> game seeds
    every: int = 5
    k: int = 8
    skill: str = "normal"  # oracle branch skill, for every player
    game_skill: str = "hard"  # AI skill of the generated games
    aifill: int = 4
    size: int = 1

    @property
    def until(self) -> int:
        return self.endturn

    @property
    def seeds(self) -> list[int]:
        return [s for r in self.splits.values() for s in r]

    def split_of(self) -> dict[int, str]:
        return {s: name for name, r in self.splits.items() for s in r}


PRESETS: dict[str, BankPreset] = {
    # 70-turn civ2civ3 games: CivTelescope training and evaluation bank.
    "short_game": BankPreset(
        ruleset=None,
        endturn=70,
        min_turn=10,
        max_turn=65,
        splits={"train": range(4000, 4240), "eval": range(5000, 5030)},
    ),
    # 120-turn classic games, early and mid game: ruleset-transfer evaluation.
    "classic_transfer": BankPreset(
        ruleset="classic",
        endturn=120,
        min_turn=10,
        max_turn=65,
        splits={"eval": range(4300, 4340)},
    ),
    # 120-turn games labeled over the whole game: trap rate by game progress.
    # classic_120 plays the same games as classic_transfer (a shared --work-dir
    # reuses them).
    "civ2civ3_120": BankPreset(
        ruleset="civ2civ3",
        endturn=120,
        min_turn=10,
        max_turn=115,
        splits={"eval": range(12000, 12040)},
    ),
    "classic_120": BankPreset(
        ruleset="classic",
        endturn=120,
        min_turn=10,
        max_turn=115,
        splits={"eval": range(4300, 4340)},
    ),
    # Game-length curve: ten games per length, disjoint seed pools.
    "game_length_60": BankPreset(
        ruleset=None,
        endturn=60,
        min_turn=10,
        max_turn=55,
        splits={"eval": range(7000, 7010)},
    ),
    "game_length_70": BankPreset(
        ruleset=None,
        endturn=70,
        min_turn=10,
        max_turn=65,
        splits={"eval": range(3000, 3010)},
    ),
    "game_length_80": BankPreset(
        ruleset=None,
        endturn=80,
        min_turn=10,
        max_turn=75,
        splits={"eval": range(7100, 7110)},
    ),
}

_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[bank +{time.time() - _T0:7.1f}s] {msg}", flush=True)


def game_id(seed: int) -> str:
    return f"g{seed}"


def position_id(gid: str, turn: int, endturn: int) -> str:
    """`<game>_T<turn>`, the turn zero-padded to the width of the end turn."""
    return f"{gid}_T{turn:0{len(str(endturn))}d}"


def _rel(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def _with_retry(fn, label: str):
    from civharness.server import with_port_retry

    def on_retry(attempt: int, exc: Exception) -> None:
        _log(f"retry {label} (attempt {attempt}): {type(exc).__name__}")

    return with_port_retry(fn, LAUNCH_ATTEMPTS, on_retry=on_retry)


# --------------------------------------------------------------- stage 1


def generate_game(
    seed: int,
    *,
    ruleset: str | None,
    endturn: int,
    every: int,
    pool_dir: Path,
    skill: str = "hard",
    aifill: int = 4,
    size: int = 1,
) -> dict:
    """Play (or reload) one seeded game and return its game record.

    A game whose work directory already holds a finished run is reloaded
    from its saves instead of being replayed."""
    from civharness import (
        GameConfig,
        load_snapshot_series,
        position,
        snapshot_series,
    )

    pool_dir = Path(pool_dir).resolve()
    gid = game_id(seed)
    wd = pool_dir / gid
    cfg = GameConfig(
        aifill=aifill,
        skill=skill,
        endturn=endturn,
        mapseed=seed,
        gameseed=seed,
        size=size,
        ruleset=ruleset,
    )
    if (wd / "result.json").exists():
        refs = load_snapshot_series(wd)
    else:
        refs = _with_retry(lambda: snapshot_series(cfg, wd, every=every), gid)
    refs = sorted(refs, key=lambda r: r.turn)
    if not refs:
        raise RuntimeError(f"{gid}: no saves in {wd}")
    final = position(refs[-1].path)
    focal = focal_player(final)
    return {
        "game_id": gid,
        "seed": seed,
        "ruleset": ruleset,
        "endturn": endturn,
        "config_hash": cfg.hash(),
        "focal_player": focal,
        "end_turn": final.turn,
        "end_score": final.players[focal]["score"],
        "saves": [{"turn": r.turn, "path": _rel(r.path, pool_dir)} for r in refs],
    }


def generate_games(
    seeds,
    *,
    ruleset: str | None,
    endturn: int,
    every: int,
    pool_dir: Path,
    workers: int,
    skill: str = "hard",
    aifill: int = 4,
    size: int = 1,
) -> list[dict]:
    """Generate (or reload) one game per seed in parallel.

    Returns the game records sorted by seed. A game that fails after the
    launch retries is logged and left out."""
    seeds = list(seeds)
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    games, failed = [], []
    _log(f"games: {len(seeds)} seeds, endturn {endturn}, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                generate_game,
                s,
                ruleset=ruleset,
                endturn=endturn,
                every=every,
                pool_dir=pool_dir,
                skill=skill,
                aifill=aifill,
                size=size,
            ): s
            for s in seeds
        }
        for i, fut in enumerate(as_completed(futs), 1):
            seed = futs[fut]
            try:
                games.append(fut.result())
            except Exception:  # noqa: BLE001 -- one failed game, not a stop
                failed.append(seed)
                _log(f"games: {game_id(seed)} failed\n{traceback.format_exc(limit=2)}")
            if i % 10 == 0 or i == len(seeds):
                _log(f"games: {i}/{len(seeds)} done, {len(failed)} failed")
    return sorted(games, key=lambda g: g["seed"])


# --------------------------------------------------------------- stage 2


def select_positions(
    games: list[dict],
    *,
    min_turn: int,
    max_turn: int,
    every: int,
    pool_dir: Path,
) -> list[dict]:
    """The positions of `games` at turns min_turn..max_turn that are multiples
    of `every`, one per turn, skipping any snapshot without the focal player.

    Each position carries the real game's final focal score (game_end_score),
    the ground truth of a single realization."""
    from civharness import position

    pool_dir = Path(pool_dir)
    rows = []
    for g in games:
        focal = g["focal_player"]
        seen = set()
        for s in sorted(g["saves"], key=lambda s: (s["turn"], s["path"])):
            turn = s["turn"]
            if not (min_turn <= turn <= max_turn) or turn % every or turn in seen:
                continue
            state = position(pool_dir / s["path"])
            if focal not in state.players:
                continue
            seen.add(turn)
            row = {
                "position_id": position_id(g["game_id"], turn, g["endturn"]),
                "game_id": g["game_id"],
                "turn": turn,
                "focal_player": focal,
                "score_now": state.players[focal]["score"],
                "game_end_turn": g["end_turn"],
                "game_end_score": g["end_score"],
                "save_path": s["path"],
            }
            if "split" in g:
                row["split"] = g["split"]
            rows.append(row)
    return sorted(rows, key=lambda r: r["position_id"])


# --------------------------------------------------------------- stage 3


def label_position(
    pos: dict,
    *,
    pool_dir: Path,
    until: int,
    k: int = 8,
    skill: str = "normal",
    work_dir: Path,
) -> dict:
    """Replay-oracle label of one position: K reseeded branches to `until`.

    Raises when fewer than K branches report a focal score."""
    from civharness import branch

    wd = Path(work_dir) / pos["position_id"]
    if wd.exists():  # partial branches of an interrupted run
        shutil.rmtree(wd, ignore_errors=True)
    save = Path(pool_dir) / pos["save_path"]
    trajs = _with_retry(
        lambda: branch(save, wd, until=until, k=k, skill=skill), pos["position_id"]
    )
    focal = pos["focal_player"]
    ends = [t.scores[focal]["total"] for t in trajs if focal in t.scores]
    if len(ends) < k:
        raise RuntimeError(f"only {len(ends)}/{k} branches scored")
    sd = statistics.stdev(ends) if len(ends) > 1 else 0.0
    return {
        **{key: pos[key] for key in pos if key != "save_path"},
        "until": until,
        "skill": skill,
        "end_scores": ends,
        "mean_end": statistics.mean(ends),
        "sd_end": sd,
        "se_end": sd / len(ends) ** 0.5,
        "n": len(ends),
        "save_path": pos["save_path"],
    }


def labeled_ids(out_path: Path) -> set[str]:
    """Position ids already present in a labels file."""
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    return {r["position_id"] for r in iter_jsonl(out_path)}


def label_positions(
    positions: list[dict],
    *,
    pool_dir: Path,
    until: int,
    k: int = 8,
    skill: str = "normal",
    work_dir: Path,
    out_path: Path,
    workers: int,
) -> dict:
    """Label every position not yet in `out_path`, appending one JSON line per
    labeled position as it completes (so an interrupted run resumes).

    Branch directories are deleted once a position is labeled. Positions are
    labeled earliest turn first, so the longest replays start first.
    Returns {"n_labeled", "n_new", "failures"}."""
    work_dir, out_path = Path(work_dir), Path(out_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = labeled_ids(out_path)
    todo = sorted(
        (p for p in positions if p["position_id"] not in done),
        key=lambda p: (p["turn"], p["position_id"]),
    )
    _log(
        f"labels: {len(todo)} to label ({len(done)} done), "
        f"K={k} skill={skill} until={until}, {workers} workers"
    )
    lock = threading.Lock()
    n_new, failures = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, out_path.open("a") as f:
        futs = {
            ex.submit(
                label_position,
                p,
                pool_dir=pool_dir,
                until=until,
                k=k,
                skill=skill,
                work_dir=work_dir,
            ): p["position_id"]
            for p in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            pid = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 -- one failed label, not a stop
                failures.append(
                    {"position_id": pid, "error": f"{type(e).__name__}: {e}"}
                )
                _log(f"labels: {pid} failed: {type(e).__name__}: {e}")
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
            n_new += 1
            shutil.rmtree(work_dir / pid, ignore_errors=True)
            if i % 25 == 0 or i == len(todo):
                _log(
                    f"labels: {i}/{len(todo)} ({n_new} ok, {len(failures)} failed), "
                    f"{(time.time() - t0) / 60:.0f} min"
                )
    return {
        "n_labeled": len(labeled_ids(out_path)),
        "n_new": n_new,
        "failures": failures,
    }
