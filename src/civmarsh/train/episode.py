"""Engine side of one policy episode: a live CivHarness session on its own thread.

``drive_episode`` resumes the start save and lets the focal player be driven
by the policy until the end turn; every other seat is a hard AI.  Each turn the
engine thread hands the observation text and action menu to the async side
(``Episode.obs`` + ``obs_ready``), blocks until a reply arrives
(``reply_ready``), and issues the resolved orders.  The async side is the slime
rollout (``train.rollout.generate``) or any other driver that speaks the same
handshake.  This module has no slime dependency.

Terminal outcome (``Episode.result``): the focal score read from the finished
game's save.  A ``ControlLost`` from the harness is classified by its message
prefix (``parse_elim_class``): only ``GAME`` (the player is out of the game) is
a real terminal outcome, priced at score 0; ``INFRA`` / ``UNKNOWN`` is an
engine seat stall, flagged ``infra_stall`` so the rollout drops and resamples
the episode instead of charging the policy for the infrastructure.

Config keys (read from the slime args):
  civ_workdir      scratch directory for the episode sessions
  civ_endturn      absolute episode end turn (default 120)
  civ_engine_seed  fixed engine seed for the resumed game (default 1)
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from civharness import EnvSpec, FunctionAgent, position, run_agents
from civharness.client import ControlLost
from civmarsh.env.menu import (
    BundleValidationError,
    build_menu,
    obs_block,
    parse_action_bundle,
)

DEFAULT_ENDTURN = 120
DEFAULT_ENGINE_SEED = 1
OPPONENT_SKILL = "hard"
# Every turn waits on the policy and concurrent episodes queue on the rollout
# engines, so the drive deadline is 4x the harness's bot-speed deadline (the
# per-phase wait stays at the harness default of 240 s).
TIMEOUT_SCALE = 4.0

# Genuine fog of war: the policy sees only its own player view.
ENV = EnvSpec.from_dict(
    {
        "name": "civmarsh-policy",
        "observations": {
            "techs": "known_set",
            "cities": True,
            "units": True,
            "government": True,
            "rates": True,
            "diplomacy": True,
            "players": "self",
            "terrain": True,
            "legal_actions": True,
            "visibility": "player_visible",
        },
    }
)

ELIM_CLASSES = ("GAME", "INFRA", "UNKNOWN")


def parse_elim_class(msg) -> str:
    """Class of a ``ControlLost``, from the harness's message prefix.

    The harness reads the newest save's player table before raising and
    prefixes the message with ``"GAME: "``, ``"INFRA: "`` or ``"UNKNOWN: "``:

      GAME     the focal player is out of the game; score 0 is the true
               terminal reward and the episode trains
      INFRA    the seat stalled while the civilization was alive; not a
               terminal outcome, the episode is dropped and resampled
      UNKNOWN  unreadable save or disagreeing seats; treated like INFRA
               (never invent a death that cannot be proven)

    A message without a prefix is UNKNOWN.
    """
    text = str(msg or "")
    for cls in ELIM_CLASSES:
        if text.startswith(cls + ":"):
            return cls
    return "UNKNOWN"


def parse_reply(text, menu):
    """ActionBundle for a valid reply, else the BundleValidationError."""
    try:
        return parse_action_bundle(text, menu)
    except BundleValidationError as exc:
        return exc


def resolve_reply(reply, menu):
    """(orders, audit pick, error) for a parsed reply; invalid -> no orders."""
    if isinstance(reply, BundleValidationError):
        return [], None, str(reply)
    if reply is None:
        return [], None, "no reply"
    pick = {
        "keys": list(reply.keys),
        "candidate_sha": reply.candidate_sha,
        "n_orders": len(reply.orders),
    }
    return list(reply.orders), pick, None


class Episode:
    """State shared between the engine thread and the async side for one
    trajectory.  The async side sets ``loop`` and ``obs_ready`` (an
    ``asyncio.Event``) before starting the thread."""

    def __init__(self):
        self.uid = uuid.uuid4().hex[:8]
        self.loop = None
        self.obs_ready = None
        self.reply_ready = threading.Event()
        self.obs = None  # (observation text, menu, turn)
        self.obs_raw = None  # the raw Observation, for the value input
        self.reply = None
        self.raw = None
        self.done = False
        self.error = None
        self.result = None
        self.choices = []
        self.handed = False


def drive_episode(ep: Episode, meta, args) -> None:
    """Engine thread: live session from ``meta["save_path"]`` to the end turn."""
    save = Path(meta["save_path"])
    focal = meta["focal_player"]
    endturn = getattr(args, "civ_endturn", DEFAULT_ENDTURN)
    seed = getattr(args, "civ_engine_seed", DEFAULT_ENGINE_SEED)
    wd = Path(args.civ_workdir) / meta["position_id"] / f"s{ep.uid}"

    def act(obs):
        if obs.turn >= endturn:
            return []
        try:
            menu = build_menu(obs)
        except Exception as e:  # noqa: BLE001 - a menu failure is a logged no-op
            ep.choices.append({"turn": obs.turn, "error": f"menu: {e}"})
            return []
        text = obs_block(obs, menu)
        ep.reply_ready.clear()
        ep.obs = (text, menu, obs.turn)
        ep.obs_raw = obs
        ep.handed = True
        ep.loop.call_soon_threadsafe(ep.obs_ready.set)
        ep.reply_ready.wait()
        orders, pick, error = resolve_reply(ep.reply, menu)
        record = {"turn": obs.turn, "pick": pick}
        if error:
            record.update({"error": error, "raw": (ep.raw or "")[:160]})
        ep.choices.append(record)
        return orders  # an invalid reply is a logged no-op

    try:
        try:
            for attempt in (1, 2):  # port-race retry, fresh subdir
                try:
                    trajs = run_agents(
                        save,
                        wd / f"a{attempt}",
                        agents={focal: FunctionAgent(act)},
                        # one turn past the end so the final save carries the
                        # terminal score
                        until=endturn + 1,
                        seeds=[seed],
                        env=ENV,
                        skill=OPPONENT_SKILL,
                        timeout_scale=TIMEOUT_SCALE,
                    )
                    break
                except ControlLost:
                    raise
                except Exception:
                    # retry only before any observation reached the policy;
                    # afterwards the decision sequence would mix two attempts
                    if attempt == 2 or ep.handed:
                        raise
            final = Path(trajs[0].save_path)
            st = position(final)
            alive = focal in st.players
            ep.result = {
                "endpoint_save": str(final),
                "end_turn": st.turn,
                "score_end": st.players[focal]["score"] if alive else 0,
                "eliminated": not alive,
                "elim_class": None if alive else "GAME",
                "infra_stall": False,
            }
        except ControlLost as e:
            # ControlLost is the single exit both for a real in-game death and
            # for an engine seat that stopped receiving phases; only GAME is a
            # terminal outcome the policy pays for.
            elim_class = parse_elim_class(e)
            ep.result = {
                "endpoint_save": None,
                "end_turn": None,
                "score_end": 0,
                "eliminated": True,
                "control_lost": str(e)[:400],
                "elim_class": elim_class,
                "infra_stall": elim_class != "GAME",
            }
    except Exception as e:  # noqa: BLE001 - reported to the async side
        ep.error = f"{type(e).__name__}: {e}"
    finally:
        ep.done = True
        ep.loop.call_soon_threadsafe(ep.obs_ready.set)
