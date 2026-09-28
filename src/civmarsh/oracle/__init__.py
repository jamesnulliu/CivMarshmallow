"""Replay-oracle position banks for Freeciv and the decidable-pair rule."""

from civmarsh.oracle.bank import (
    PRESETS,
    BankPreset,
    generate_game,
    generate_games,
    label_position,
    label_positions,
    select_positions,
)
from civmarsh.oracle.pairs import decidable_pairs, trap_summary

__all__ = [
    "PRESETS",
    "BankPreset",
    "decidable_pairs",
    "generate_game",
    "generate_games",
    "label_position",
    "label_positions",
    "select_positions",
    "trap_summary",
]
