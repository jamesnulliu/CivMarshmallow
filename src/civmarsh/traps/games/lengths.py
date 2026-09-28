"""Typical full-game length per game, the denominator of game progress.

OpenSpiel games and Catan: the median length of complete self-play games
played by the bank's own engine, on a seed range disjoint from the banks.
Chess: the median final ply of the bank's own games. Othello: 60 moves.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ProcessPoolExecutor

from civmarsh.traps.games import GAMES, OTHELLO_LENGTH, SPIEL_GAMES

LENGTH_SEED = 90000
N_LENGTH_GAMES = 30
MEASURED_GAMES = (*SPIEL_GAMES, "catan")


def _summary(lens: list[int], unit: str) -> dict:
    return {
        "n_games": len(lens),
        "median": statistics.median(lens),
        "mean": round(statistics.mean(lens), 1),
        "min": min(lens),
        "max": max(lens),
        "unit": unit,
    }


def measure_lengths(
    games=MEASURED_GAMES,
    *,
    n_games: int = N_LENGTH_GAMES,
    seed: int = LENGTH_SEED,
    workers: int = 20,
    log=print,
) -> dict:
    """{game: {n_games, median, mean, min, max, unit}}; game i uses seed + i."""
    from civmarsh.traps.games import catan_bank, spiel_bank

    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for name in games:
            if name == "catan":
                lens = list(
                    ex.map(
                        catan_bank.self_play_length, [seed + i for i in range(n_games)]
                    )
                )
                out[name] = _summary(lens, "turns (num_turns)")
            elif name in SPIEL_GAMES:
                jobs = [(name, seed + i) for i in range(n_games)]
                lens = list(ex.map(spiel_bank.self_play_length, jobs))
                out[name] = _summary(lens, "player moves")
            else:
                raise ValueError(f"no self-play length measurement for {name!r}")
            log(f"{name}: {out[name]}")
    return out


def typical_length(game: str, bank: list[dict], lengths: dict) -> float:
    """The progress denominator of `game`: chess from its bank's final plies,
    Othello fixed, every other game from `measure_lengths` output."""
    if game == "chess":
        return statistics.median(
            {r["game_id"]: r["game_end_ply"] for r in bank}.values()
        )
    if game == "othello":
        return OTHELLO_LENGTH
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r}")
    return lengths[game]["median"]
