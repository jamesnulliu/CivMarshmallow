"""Position banks: engine games, snapshots, replay-oracle labels.

``bank_rows`` produces rows in the cross-game bank schema — ``position_id``,
``game_id``, ``turn``, ``focal_player``, ``score_now`` (the visible proxy),
``oracle_mean / oracle_sd / oracle_se / oracle_n`` (K engine continuations to
the end, i.e. ``branch``), ``rendering`` (the value-model input) — plus what is
needed to load the position again (the save fields for Othello, a ``save``
field for Catan).  Each adapter's ``bank_game`` defines its own row details
and continuation seeds; this module only fans the games out.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from civharness.games.base import make_env


def _bank_game(job):
    game, idx, seed, k, snaps, cfg = job
    return make_env(game, **cfg).bank_game(idx, seed, k, snaps)


def bank_rows(
    game: str,
    seeds,
    snaps,
    k: int,
    *,
    index_offset: int = 0,
    workers: int = 1,
    on_game=None,
    **cfg,
) -> list[dict]:
    """Bank rows for one generated game per seed.

    ``seeds[i]`` generates game ``index_offset + i``; every snapshot in
    ``snaps`` (progress indices) is labelled with ``k`` continuations.
    Deterministic: the same arguments give the same rows, with any ``workers``.
    ``on_game(done, total, n_rows)`` is called after each game (progress)."""
    snaps = tuple(sorted(int(s) for s in snaps))
    jobs = [
        (game, index_offset + i, int(seed), int(k), snaps, cfg)
        for i, seed in enumerate(seeds)
    ]
    rows: list[dict] = []
    if workers <= 1:
        results = map(_bank_game, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_bank_game, jobs)
    for done, game_rows in enumerate(results, 1):
        rows.extend(game_rows)
        if on_game is not None:
            on_game(done, len(jobs), len(rows))
    if workers > 1:
        pool.shutdown()
    return rows
