"""Cross-game replay-oracle banks: the Freeciv bank construction in other games.

Every bank has the same record schema, so `civmarsh.oracle.pairs.decidable_pairs`
applies unchanged (``value_key="oracle_mean", se_key="oracle_se"``):

    position_id, game_id, turn, focal_player, score_now (the visible proxy),
    oracle_mean, oracle_sd, oracle_se, oracle_n (K replays), rendering, meta

A game is self-played by a deliberately weak, stochastic engine; snapshots at
fixed move numbers are the positions; the oracle is the mean outcome of K
replays to the end with the same kind of engine.

The bank modules (`chess_bank`, `othello_bank`, `spiel_bank`, `catan_bank`)
import their game libraries lazily, so every module imports without them.
"""

from __future__ import annotations

# setting -> label, genre, visible proxy, oracle, pair turn gap, length unit.
GAMES: dict[str, dict] = {
    "chess": {
        "label": "Chess",
        "genre": "board",
        "proxy": "material balance",
        "oracle": "mean result under stochastic Stockfish replays",
        "max_turn_gap": 8,
        "length_unit": "plies",
    },
    "othello": {
        "label": "Othello",
        "genre": "board",
        "proxy": "disc differential",
        "oracle": "mean final disc differential",
        "max_turn_gap": 8,
        "length_unit": "moves",
    },
    "backgammon": {
        "label": "Backgammon",
        "genre": "board, dice",
        "proxy": "pip-count lead",
        "oracle": "mean result for X (+1 win, -1 loss)",
        "max_turn_gap": 20,
        "length_unit": "player moves",
    },
    "go9": {
        "label": "Go (9x9)",
        "genre": "board",
        "proxy": "area count with komi",
        "oracle": "mean final area margin for Black",
        "max_turn_gap": 8,
        "length_unit": "player moves",
    },
    "hearts": {
        "label": "Hearts",
        "genre": "card",
        "proxy": "penalty points so far, relative to the table",
        "oracle": "mean final relative penalty",
        "max_turn_gap": 8,
        "length_unit": "player moves",
    },
    "2048": {
        "label": "2048",
        "genre": "single-player puzzle",
        "proxy": "score",
        "oracle": "mean final score",
        "max_turn_gap": 20,
        "length_unit": "moves",
    },
    "catan": {
        "label": "Catan",
        "genre": "4-player economic",
        "proxy": "public victory points",
        "oracle": "mean final victory points",
        "max_turn_gap": 20,
        "length_unit": "turns",
    },
}

SPIEL_GAMES = ("backgammon", "go9", "hearts", "2048")
# Othello progress uses a fixed full-game length: 60 moves, one per square.
OTHELLO_LENGTH = 60.0
