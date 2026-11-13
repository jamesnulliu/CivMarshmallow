"""Other games under the harness contract (save / branch / replay).

Freeciv is driven through a server process; the games here run in-process, but
they answer to the same three verbs:

  save      ``env.save()`` is a JSON-serialisable snapshot of the full state;
            ``snapshot_series`` plays an engine game and returns a position
            library, like the Freeciv function of the same name.
  branch    ``branch(save, k, seed)`` plays K re-seeded engine continuations to
            the end and returns the outcomes — the replay oracle.
  replay    the same save and the same seed give the same trajectory and the
            same outcome, in one process and across processes
            (tests/civharness/test_games_determinism.py).

External control goes through ``play(save, act_fn, seed)``: the focal player's
moves come from ``act_fn(prompt) -> reply text``, every other seat stays the
engine — the analogue of ``client_branch`` with a policy.

Games: ``othello`` (stdlib only) and ``catan`` (needs the optional ``games``
extra, i.e. catanatron).  Importing this package imports neither engine;
adapters register on first use.
"""

from civharness.games.base import (
    GameEnv,
    branch,
    make_env,
    noise_floor,
    play,
    register,
    registered_games,
    snapshot_series,
    summary,
)
from civharness.games.episode import (
    decision_snapshot,
    episode_result,
    finish_decision,
    play_episode,
)
from civharness.games.oracle import bank_rows

__all__ = [
    "GameEnv",
    "bank_rows",
    "branch",
    "decision_snapshot",
    "episode_result",
    "finish_decision",
    "make_env",
    "noise_floor",
    "play",
    "play_episode",
    "register",
    "registered_games",
    "snapshot_series",
    "summary",
]
