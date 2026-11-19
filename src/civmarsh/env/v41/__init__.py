"""Environment v4.1: the Freeciv environment of the step-level judge and RL experiments.

- Rules: the ``civm41`` ruleset (``scripts/make_civm41.py``): civ2civ3 without great
  wonders, Coinage, ships, diplomats, spies, caravans, migrants and pillage; no huts,
  animals, disasters or barbarians; diplomacy disabled (every civilization met is at
  war); a constant two-turn anarchy.
- Server: freeciv 3.2.5 built from the ``civm-3.2.5`` branch of
  github.com/jamesnulliu/freeciv (``scripts/install_freeciv_server.sh``): Hard bots
  see only what their player sees, and a conquest victory does not end the game.
- ``observation``: ``StateV4``, the player-visible state with reference tables
  (cities, sites, threats, enemy cities, explore points).
- ``menu``: the action menu, the merged v4.1 turn prompt (``render_turn_v41``), the
  constrained-decoding JSON schema and the strict entity-bundle parser.
- ``play``: whole games with a language-model policy (``LLMPlayer``, ``play_game``).
- ``hexmap``: iso-hex map geometry (distances, paths, compass directions).
"""

from civmarsh.env.v41.menu import (
    TURN_SCHEMA,
    BundleValidationError,
    obs_block_v4,
    parse_entity_bundle,
    render_turn_v41,
)
from civmarsh.env.v41.observation import RULESET, StateV4
from civmarsh.env.v41.play import SYSTEM_V41, LLMPlayer, play_game

__all__ = [
    "RULESET",
    "SYSTEM_V41",
    "TURN_SCHEMA",
    "BundleValidationError",
    "LLMPlayer",
    "StateV4",
    "obs_block_v4",
    "parse_entity_bundle",
    "play_game",
    "render_turn_v41",
]
