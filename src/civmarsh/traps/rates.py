"""Scoreboard-trap rates: overall and by game progress.

The trap rate of a bank is the share of its decidable pairs, excluding pairs
with a tied visible score, on which the visible score orders the two positions
against the replay oracle (see `civmarsh.oracle.pairs`). By game progress,
each pair sits at the mean turn of its two positions divided by the typical
length of a full game; a progress bin with fewer than `min_pairs` pairs has no
rate.
"""

from __future__ import annotations

from civmarsh.oracle.pairs import decidable_pairs

N_BINS = 10
MIN_BIN_PAIRS = 30
# Freeciv pairs: turns at most ten apart.
FREECIV_MAX_TURN_GAP = 10


def bin_edges(n_bins: int = N_BINS) -> list[float]:
    return [round(i / n_bins, 6) for i in range(n_bins + 1)]


def progress_bins(
    items: list[tuple[float, bool]],
    length: float,
    *,
    n_bins: int = N_BINS,
    min_pairs: int = MIN_BIN_PAIRS,
) -> dict:
    """Trap rate overall and per progress bin.

    items   (mean turn of the pair, is_trap) for every untied decidable pair.
    length  typical full-game length in the same unit as the turns; progress
            at or beyond 1 falls in the last bin.
    """
    edges = bin_edges(n_bins)
    bins = [[0, 0] for _ in range(n_bins)]
    for turn, trap in items:
        p = min(turn / length, 0.999999)
        k = next(i for i in range(n_bins) if edges[i] <= p < edges[i + 1])
        bins[k][0] += 1
        bins[k][1] += trap
    n = len(items)
    return {
        "overall": round(sum(t for _, t in items) / n, 4) if n else None,
        "n_untied": n,
        "bins": [
            {
                "lo": edges[i],
                "hi": edges[i + 1],
                "n": b[0],
                "trap_rate": round(b[1] / b[0], 4) if b[0] >= min_pairs else None,
            }
            for i, b in enumerate(bins)
        ],
        "progress_range": (
            [
                round(min(t for t, _ in items) / length, 3),
                round(max(t for t, _ in items) / length, 3),
            ]
            if n
            else None
        ),
    }


def pair_items(pairs: list[dict]) -> list[tuple[float, bool]]:
    """(mean turn, is_trap) of the untied pairs in `pairs`."""
    return [
        ((p["turn_a"] + p["turn_b"]) / 2, p["is_trap"])
        for p in pairs
        if not p["score_tied"]
    ]


def untied_summary(pairs: list[dict]) -> dict:
    """decidable (ties included), untied, trap and trap_rate over untied pairs."""
    untied = [p for p in pairs if not p["score_tied"]]
    trap = sum(p["is_trap"] for p in untied)
    return {
        "decidable": len(pairs),
        "untied": len(untied),
        "trap": trap,
        "trap_rate": round(trap / len(untied), 4) if untied else None,
        "scoreboard_accuracy": round(1 - trap / len(untied), 4) if untied else None,
    }


def game_trap_rates(
    game: str,
    bank: list[dict],
    *,
    length: float,
    progress_bank: list[dict] | None = None,
    n_bins: int = N_BINS,
    min_pairs: int = MIN_BIN_PAIRS,
) -> dict:
    """Trap rate of a cross-game bank (`civmarsh.traps.games`).

    The overall rate uses `bank` alone. `progress_bank` holds further
    snapshots (earlier or later moves) that enter the progress bins only,
    pooled with `bank`."""
    from civmarsh.traps.games import GAMES

    info = GAMES[game]
    gap = info["max_turn_gap"]

    def pairs_of(records):
        return decidable_pairs(
            records,
            max_turn_gap=gap,
            value_key="oracle_mean",
            se_key="oracle_se",
            drop_ties=False,
        )

    pairs = pairs_of(bank)
    out = {
        "setting": game,
        "label": info["label"],
        "genre": info["genre"],
        "proxy": info["proxy"],
        "oracle": info["oracle"],
        "positions": len(bank),
        "games": len({r["game_id"] for r in bank}),
        "replays": bank[0]["oracle_n"] if bank else None,
        "max_turn_gap": gap,
        **untied_summary(pairs),
        "length": length,
        "length_unit": info["length_unit"],
    }
    binned = pairs_of(bank + progress_bank) if progress_bank else pairs
    prog = progress_bins(pair_items(binned), length, n_bins=n_bins, min_pairs=min_pairs)
    out.update(bins=prog["bins"], progress_range=prog["progress_range"])
    if progress_bank:
        out["n_untied_bins"] = prog["n_untied"]
    return out


def freeciv_trap_rates(
    labels: list[dict],
    *,
    length: float,
    max_turn_gap: int = FREECIV_MAX_TURN_GAP,
    n_bins: int = N_BINS,
    min_pairs: int = MIN_BIN_PAIRS,
) -> dict:
    """Trap rate of a labeled Freeciv bank (`civmarsh.oracle.bank` labels):
    oracle = mean terminal score, visible score = score now."""
    pairs = decidable_pairs(labels, max_turn_gap=max_turn_gap, drop_ties=False)
    prog = progress_bins(pair_items(pairs), length, n_bins=n_bins, min_pairs=min_pairs)
    return {
        "positions": len(labels),
        "games": len({r["game_id"] for r in labels}),
        "max_turn_gap": max_turn_gap,
        "length": length,
        **untied_summary(pairs),
        "bins": prog["bins"],
        "progress_range": prog["progress_range"],
    }
