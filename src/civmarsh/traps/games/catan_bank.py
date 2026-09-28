"""Settlers of Catan bank (Catanatron): public victory points as the visible
proxy, final victory points under K bot continuations as the oracle.

Four players, hidden information (resource hands and development cards). Every
snapshot yields four positions, one per seat, sharing the same K
continuations. All seats are Catanatron's VictoryPointPlayer (greedy on
immediate victory points), for generation and for the continuations; the
module-level RNG is reseeded per continuation. Requires catanatron.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
DEFAULTS = {
    "n_games": 40,
    "replays": 48,
    "seed": 20260911,
    "snaps": tuple(range(20, 121, 20)),
}


def _colors():
    from catanatron import Color

    return [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def _player(color):
    from catanatron.players.search import VictoryPointPlayer

    return VictoryPointPlayer(color)


def ps(state, i, key):
    return state.player_state[f"P{i}_{key}"]


def continue_game(game, seed):
    random.seed(seed)
    g = copy.deepcopy(game)
    g.players = [_player(c) for c in g.state.colors]
    g.play()
    return g


def render(game, focal_i):
    """Structured text in the shape of the Freeciv rendering (model-visible)."""
    st = game.state
    lines = [
        (
            f"Settlers of Catan, turn {st.num_turns}, four players, "
            "first to 10 victory points wins."
        ),
        f"You are evaluating the long-run prospects of {st.colors[focal_i].name}.",
        "",
    ]
    for i, color in enumerate(st.colors):
        tag = "the evaluated player" if i == focal_i else "opponent"
        lines += [
            f"== {color.name} ({tag}) ==",
            f"public victory points: {ps(st, i, 'VICTORY_POINTS')}",
            (
                f"settlements left: {ps(st, i, 'SETTLEMENTS_AVAILABLE')}, "
                f"cities left: {ps(st, i, 'CITIES_AVAILABLE')}, "
                f"roads left: {ps(st, i, 'ROADS_AVAILABLE')}, "
                f"longest road: {ps(st, i, 'LONGEST_ROAD_LENGTH')}"
            ),
            (
                f"holds longest road: {bool(ps(st, i, 'HAS_ROAD'))}, "
                f"holds largest army: {bool(ps(st, i, 'HAS_ARMY'))}, "
                f"knights played: {ps(st, i, 'PLAYED_KNIGHT')}"
            ),
            "resources in hand: "
            + ", ".join(f"{r.lower()} {ps(st, i, r + '_IN_HAND')}" for r in RESOURCES)
            + " (development cards in hand: "
            + f"{sum(ps(st, i, c + '_IN_HAND') for c in DEV_CARDS)})",
            "",
        ]
    lines += [
        "== visible score ==",
        (
            f"public victory points ({st.colors[focal_i].name}): "
            f"{ps(st, focal_i, 'VICTORY_POINTS')}"
        ),
    ]
    return "\n".join(lines)


def generate_game(job) -> list[dict]:
    """One bot game; four positions (one per seat) per snapshot turn."""
    from catanatron import Game

    idx, seed, k_replays, snaps = job
    random.seed(seed)
    g = Game([_player(c) for c in _colors()])
    snapshots = {}
    while g.winning_color() is None and g.state.num_turns <= max(snaps):
        g.play_tick()
        t = g.state.num_turns
        if t in snaps and t not in snapshots:
            snapshots[t] = copy.deepcopy(g)
    rows = []
    for t, snap in snapshots.items():
        finals = [
            continue_game(snap, (seed * 7919 + t * 31 + k) % (2**31 - 1))
            for k in range(k_replays)
        ]
        for i, color in enumerate(snap.state.colors):
            vals = [float(ps(f.state, i, "ACTUAL_VICTORY_POINTS")) for f in finals]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if k_replays > 1 else 0.0
            rows.append(
                {
                    "position_id": f"catan-g{idx:03d}-t{t:03d}-{color.name}",
                    "game_id": f"catan-g{idx:03d}",
                    "turn": t,
                    "focal_player": color.name,
                    "score_now": float(ps(snap.state, i, "VICTORY_POINTS")),
                    "hidden_vp": ps(snap.state, i, "ACTUAL_VICTORY_POINTS")
                    - ps(snap.state, i, "VICTORY_POINTS"),
                    "oracle_mean": round(mean, 4),
                    "oracle_sd": round(sd, 4),
                    "oracle_se": round(sd / math.sqrt(k_replays), 4),
                    "oracle_n": k_replays,
                    "win_rate": statistics.mean(
                        1.0 if f.winning_color() == color else 0.0 for f in finals
                    ),
                    "rendering": render(snap, i),
                    "meta": {
                        "game": "catan",
                        "label": "Catan",
                        "proxy": "public victory points",
                        "oracle": "mean final victory points",
                        "policy": "catanatron:vp",
                        "max_turn_gap": 20,
                    },
                }
            )
    return rows


def build_bank(
    *,
    n_games: int = DEFAULTS["n_games"],
    game_offset: int = 0,
    replays: int = DEFAULTS["replays"],
    snaps=DEFAULTS["snaps"],
    seed: int = DEFAULTS["seed"],
    workers: int = 12,
    log=print,
) -> list[dict]:
    """Games game_offset .. game_offset + n_games - 1 (game i uses seed + i).

    Catanatron self-play does not reproduce from the seed, so further games
    for an existing bank need their own index range (`game_offset`)."""
    snap_turns = frozenset(snaps)
    jobs = [
        (game_offset + i, seed + game_offset + i, replays, snap_turns)
        for i in range(n_games)
    ]
    log(
        f"catan: {len(jobs)} games x {len(snap_turns)} snapshots x 4 seats x "
        f"{replays} continuations, {workers} workers"
    )
    bank = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, rows in enumerate(ex.map(generate_game, jobs), 1):
            bank.extend(rows)
            if done % 5 == 0:
                log(f"  {done}/{len(jobs)} games, {len(bank)} positions")
    return bank


def self_play_length(seed: int) -> int:
    """Turns (``num_turns``) of one complete bot game."""
    from catanatron import Game

    random.seed(seed)
    g = Game([_player(c) for c in _colors()])
    g.play()
    return g.state.num_turns
