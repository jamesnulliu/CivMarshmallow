"""Game interface and the save / branch / replay functions built on it.

A ``GameEnv`` holds ONE game.  It is used in two modes:

* engine mode — every seat is played by the game's own engine
  (``selfplay`` / ``rollout``); this is what ``snapshot_series`` and ``branch``
  (the replay oracle) run;
* focal mode — one seat, the FOCAL player, is driven from outside: the caller
  only ever sees positions in which the focal player has a real choice, and
  ``step`` plays the focal action and then every engine move (and every forced
  focal move) up to the next such position or the end of the game.

Per decision the focal-mode contract is what a policy bridge needs:

  ``turn()``          progress index (the reference-set bucket key)
  ``legal()``         offered action keys for the focal player
  ``policy_prompt()`` user text for the policy (system contract + position)
  ``value_input()``   the position in the bank's rendering format — the exact
                      text a pairwise value model (CivTelescope) is trained on
  ``score_now()``     the visible proxy (the game's "scoreboard")
  ``outcome()``       terminal score, in the same units as ``score_now``

All randomness comes from the ``random.Random`` handed in by the caller, so a
save plus a seed determines the continuation.
"""

from __future__ import annotations

import math
import random
import statistics


class GameEnv:
    """Base class; adapters override everything that raises."""

    name = "game"
    focal = "player"

    # -- save / load ---------------------------------------------------------

    def new_game(self, rng: random.Random) -> None:
        """Set up a fresh game (engine mode: nobody has moved)."""
        raise NotImplementedError

    def save(self) -> dict:
        """JSON-serialisable snapshot of the full state, including whose move
        it is and the progress index (``turn``)."""
        raise NotImplementedError

    def load(self, save: dict) -> None:
        """Restore exactly the state ``save()`` captured; plays nothing."""
        raise NotImplementedError

    def reset(self, start: dict, rng: random.Random) -> None:
        """Focal mode: ``load`` the start, then advance engine seats and forced
        focal moves until the focal player has a real choice (or the game is
        over).  ``start`` is a save or a starts-bank row that embeds one."""
        raise NotImplementedError

    # -- engine mode ---------------------------------------------------------

    def selfplay(self, rng: random.Random, snaps) -> list[dict]:
        """Play an engine game from the current state to the end and return a
        save at each progress index in ``snaps`` (the first position reached
        with that index)."""
        raise NotImplementedError

    def rollout(self, rng: random.Random) -> None:
        """Every seat is the engine: play from the current state to the end."""
        raise NotImplementedError

    def outcomes(self) -> dict[str, float]:
        """Terminal score of every player, by player name."""
        raise NotImplementedError

    # -- focal mode ----------------------------------------------------------

    def done(self) -> bool:
        raise NotImplementedError

    def turn(self) -> int:
        raise NotImplementedError

    def legal(self) -> list[str]:
        raise NotImplementedError

    def describe(self, key: str) -> str:
        """Human-readable form of an action key (default: the key itself)."""
        return key

    def policy_prompt(self) -> str:
        raise NotImplementedError

    def value_input(self) -> str:
        raise NotImplementedError

    def score_now(self) -> float:
        raise NotImplementedError

    def parse(self, text: str, legal: list[str]) -> tuple[str | None, str | None]:
        """Raw policy reply -> (action key, None) or (None, error string)."""
        raise NotImplementedError

    def step(self, action: str | None, rng: random.Random) -> str:
        """Play ``action`` for the focal player (``None`` = the reply was
        invalid: a uniformly random legal action is played instead), then
        advance until the focal player has a real choice again or the game
        ends.  Returns the action key that was actually executed."""
        raise NotImplementedError

    def outcome(self) -> float:
        raise NotImplementedError

    def result_extra(self) -> dict:
        """Game-specific fields added to the episode's terminal record."""
        return {}


_REGISTRY: dict[str, type] = {}
_BUILTIN = {"othello": "civharness.games.othello", "catan": "civharness.games.catan"}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_games() -> list[str]:
    return sorted(set(_REGISTRY) | set(_BUILTIN))


def make_env(name: str, **cfg) -> GameEnv:
    if name not in _REGISTRY and name in _BUILTIN:
        # adapters register on import; pulled in lazily so that importing the
        # package never imports an engine (catan needs an optional dependency)
        import importlib

        importlib.import_module(_BUILTIN[name])
    if name not in _REGISTRY:
        raise KeyError(f"unknown game {name!r}; known: {registered_games()}")
    return _REGISTRY[name](**cfg)


# ---------------------------------------------------------------- the three verbs
def replica_seed(seed: int, k: int) -> int:
    """Seed of the k-th continuation of ``branch(save, K, seed)``."""
    return (int(seed) * 1_000_003 + k) % (2**63 - 1)


def snapshot_series(game: str, seed: int, snaps, **cfg) -> list[dict]:
    """One engine game from a fresh start, saved at the progress indices in
    ``snaps`` — the position library (Freeciv: ``snapshot_series``)."""
    env = make_env(game, **cfg)
    rng = random.Random(seed)
    env.new_game(rng)
    return env.selfplay(rng, frozenset(int(s) for s in snaps))


def branch(
    save: dict,
    k: int | None = None,
    seed: int | None = None,
    *,
    seeds=None,
    all_players: bool = False,
    game: str | None = None,
    **cfg,
) -> list:
    """K re-seeded engine continuations of ``save`` to the end of the game.

    Returns the focal player's outcome per continuation (``all_players=True``:
    one ``{player: outcome}`` dict per continuation).  Continuation i is seeded
    with ``replica_seed(seed, i)``; pass ``seeds`` to name the seeds yourself.
    The same save and seeds always give the same list.  ``game`` names the
    game when the save does not carry it (e.g. an Othello bank row)."""
    if seeds is None:
        if k is None or seed is None:
            raise ValueError("branch needs (k, seed) or seeds")
        seeds = [replica_seed(seed, i) for i in range(k)]
    env = make_env(game or save["game"], **cfg)
    out = []
    for s in seeds:
        env.load(save)
        env.rollout(random.Random(s))
        finals = env.outcomes()
        out.append(finals if all_players else finals[save["focal_player"]])
    return out


def summary(values) -> dict:
    """Mean / sd / se / n of a list of outcomes (sd = sample standard deviation)."""
    values = [float(v) for v in values]
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "se": sd / math.sqrt(n), "n": n}


def noise_floor(save: dict, k: int, seed: int, **cfg) -> dict:
    """The spread of a position's own continuations: ``summary`` of
    ``branch(save, k, seed)``.  A difference between two positions is only
    readable against this."""
    return summary(branch(save, k, seed, **cfg))


def play(start: dict, act_fn, seed: int, **cfg):
    """External control: ``act_fn(prompt) -> reply text`` plays the focal
    player from ``start`` (a save or a starts-bank row), the engine plays the
    rest.  Returns (decisions, result); see ``civharness.games.episode``."""
    from civharness.games.episode import play_episode

    env = make_env(start.get("game", "othello"), **cfg)
    return play_episode(env, start, act_fn, random.Random(seed))
