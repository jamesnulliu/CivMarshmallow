"""Agent interface and multi-agent driver over the faithful network client.

`client_branch` runs a *static* ClientPolicy. This module adds what agents
need: a per-turn **observation** and a driver that lets **one or more** agents
each control a player in the same game.

- `Observation` — a turn's structured state, gated by the run's EnvSpec: the
  save-derived world under ``visibility: full_state``, or the acting client's
  own fog-of-war view under ``visibility: player_visible``. An agent's
  `act(obs) -> [Order]` returns orders built from the same order types as
  ClientPolicy (`civharness.policy.client`).
- `run_agents(save, agents={player_name: Agent}, ...)` — N clients, one per
  controlled player, driven through a shared game; uncontrolled seats stay AI.
  Deterministic: same seeds + deterministic agents -> byte-identical.
- `run_agent(...)` — the single-seat convenience.
- `agent_noise_floor(...)` — the replica-identity + seed-spread check for this
  path, so an effect is never quoted without its floor.

Why not a Gym `Env`: a Freeciv turn is multi-player and turn-based; the RL
`step()` framing fights that and a hard Gymnasium dependency is baggage. This
agent protocol is dependency-free; a Gym adapter can wrap it if needed.

Multi-seat facts the driver depends on: each connected player must ready
itself (a `/start` only readies the caller's own player; the game starts when
the last one readies), and the socket must NOT be drained between `/start`
and the phase loop, or the first START_PHASE is swallowed.
"""

import copy
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from civharness import candidates as _cand
from civharness import observe_view as _pv
from civharness import parse, server
from civharness import spatial as _spatial
from civharness.branch import (
    ELIM_GAME,
    ELIM_INFRA,
    ELIM_UNKNOWN,
    FloorReport,
    SaveRef,
    Trajectory,
    _alive_in_save,
    _ensure_clean_saves,
    _final_save,
    classify_control_loss,
    control_lost,
    reseed_save,
    turn_of_save,
    wait_for_log_marker,
)
from civharness.client import RULE_DUMP_LUA, FreecivClient, RuleIds
from civharness.config import DEFAULT_BINARY, client_resume_script
from civharness.envspec import EnvSpec
from civharness.episode import EpisodeWriter
from civharness.observe import observe_state
from civharness.runner import default_timeout

# Ceiling (seconds) on ONE phase wait in the multi-agent driver. With a policy
# in the loop every turn waits on the policy's own inference, so the per-phase
# budget is well above bot speed.
DEFAULT_PHASE_BUDGET = 240.0

# `default_timeout` is sized for bot-speed turns whose phase waits stay under
# this many seconds. The drive deadline scales with phase_budget / this, so a
# larger per-phase ceiling is not cancelled by fewer turns fitting into the
# overall deadline.
_BOT_SPEED_PHASE_BUDGET = 60.0

# Schema identifier of `Observation`, recorded in every episode's provenance.
OBSERVATION_SCHEMA = "civharness-observation"


def drive_timeout(
    until: int,
    *,
    timeout_scale: float = 1.0,
    phase_budget: float = DEFAULT_PHASE_BUDGET,
) -> float:
    """Total wall-clock deadline (seconds) for driving a game to `until`:
    `default_timeout(until)` scaled by `timeout_scale` and by
    `phase_budget / 60`."""
    return (
        timeout_scale
        * default_timeout(until)
        * (phase_budget / _BOT_SPEED_PHASE_BUDGET)
    )


def _player_id_of(all_players: dict, pname: str, saves_dir=None) -> int:
    """Numeric player id for a name, from the save-derived player table.

    Absence from the table is not proof of death: the table was parsed from
    the turn's autosave, which `_obs_save` returns as soon as it appears — it
    may still be being written, and a truncated player table reads as
    "absent" while the seat is alive. So absence is confirmed against the
    second-newest save, which is complete by construction: only an absence
    (or is_alive FALSE) reproduced there is a game outcome (GAME). Anything
    short of that — alive in the previous save, no previous save, unreadable
    previous save — raises as UNKNOWN.
    """
    p = all_players.get(pname) or {}
    pid = p.get("player_id")
    if pid is None:
        cls, note = ELIM_UNKNOWN, "no complete earlier save to confirm against"
        if saves_dir is not None:
            try:
                saves = sorted(
                    Path(saves_dir).glob("*.sav*"), key=lambda q: q.stat().st_mtime
                )
                if len(saves) >= 2:
                    if _alive_in_save(saves[-2], [pname]).get(pname):
                        note = (
                            "alive in the previous save "
                            f"({saves[-2].name}) — mid-write race suspected"
                        )
                    else:
                        cls = ELIM_GAME
                        note = f"absence confirmed in {saves[-2].name}"
            except Exception as e:  # noqa: BLE001 -- any failure means UNKNOWN
                note = f"previous-save recheck failed ({e!r})"
        raise control_lost(
            cls,
            f"player {pname!r} absent from the observation save ({note})",
        )
    return pid


@dataclass(frozen=True)
class Observation:
    """The state an agent sees at the start of a turn's phase.

    Under the default ``visibility: full_state`` this is save-derived full
    information; an agent is free to ignore what a fog-limited player couldn't
    see. Under ``visibility: player_visible`` the map/entity world comes
    from the acting client's own PlayerViewCache (what the server actually
    sent that connection), `players` is restricted to the acting player, and
    the raw save is withheld.  `my` is the acting player's own slice of
    `players`; own-player fields (gold, techs, government, rates, diplomacy)
    are never fogged for oneself, so they are safe in both modes.
    """

    turn: int
    me: str  # the acting player's name
    my: dict  # this player's dict: gold, cities, units, techs_known, government, ...
    players: dict  # visible players (all, or just `me` when env players: self)
    # The raw save, for a full-information agent that wants to dig deeper. It is
    # None under a restricted EnvSpec, so a myopic/self-only agent cannot bypass
    # its observation restriction by re-parsing the god-view save.
    save_path: str | None
    terrain: dict | None = None  # {xsize, ysize, grid, legend} if env wants it
    # If the EnvSpec enables legal_actions, a bound client.get_unit_actions:
    # legal_actions(actor_unit_id, target_tile=..., target_unit=...) ->
    # {action_rule_name: (min, max)}. None otherwise. This is a live query
    # (a protocol round-trip), so it is not part of the recorded episode.
    legal_actions: object = None
    rates: dict | None = None  # acting player's tax/luxury/science split
    diplomacy: dict | None = None  # acting player's save-persisted relations
    # Under visibility: player_visible, the typed fog view built from the
    # acting client's own frozen PlayerViewCache snapshot (observe_view
    # .player_view): {"visibility", "map": {xsize,ysize,tiles}, "cities",
    # "units"}. None under full_state.
    player_view: dict | None = None
    # When legal_actions is enabled, the acting client's decoded ruleset
    # catalog (proto.rulesets.RulesetCatalog: extras with causes/buildable,
    # actions with sub-target kinds) and its RuleIds — the typed inputs for
    # resolving action sub-targets. None otherwise.
    rulesets: object = None
    rule_ids: object = None
    # Audit callback — the agent MAY call obs.audit(offered_sha=...,
    # offered=[...], **extra) once per decision to register the exact
    # candidate set it was offered; the driver writes it into this step's
    # episode record. None when no episode is being recorded.
    audit: object = None

    @property
    def my_cities(self) -> list[dict]:
        return self.my.get("cities", [])

    @property
    def my_units(self) -> list[dict]:
        return self.my.get("units", [])

    @property
    def my_techs(self) -> set:
        """The set of tech rule-names this player knows (empty if the EnvSpec
        did not request the known-tech set)."""
        return self.my.get("techs_known", set())


class Agent(Protocol):
    """Return the orders to issue for `obs.me` this turn (possibly empty)."""

    def act(self, obs: Observation) -> list: ...


@dataclass
class FunctionAgent:
    """Wrap a plain `fn(obs) -> [Order]` as an Agent."""

    fn: object

    def act(self, obs: Observation) -> list:
        return list(self.fn(obs) or [])


@dataclass
class PolicyAgent:
    """Adapt a static ClientPolicy (turn -> [Order]) to the Agent interface,
    so scripted and reactive control share one driver."""

    policy: object  # civharness.policy.client.ClientPolicy

    def act(self, obs: Observation) -> list:
        return list(self.policy.orders.get(obs.turn, ()))


class MissingObservation(RuntimeError):
    """The per-turn autosave an observation must be read from never appeared.
    Raised instead of silently falling back to the loaded (wrong-turn) save,
    which would feed the agent a stale position."""


@dataclass
class AgentFactory:
    """A zero-arg builder called once per seed/branch so a STATEFUL agent gets a
    fresh instance each time — otherwise one instance, reused across seeds,
    leaks state and breaks branch independence. Wrap with `fresh(...)`."""

    make: object  # () -> Agent | ClientPolicy | fn(obs)->[Order]

    def build(self) -> "Agent":
        return _as_agent(self.make())


def fresh(make) -> AgentFactory:
    """Mark a zero-arg callable as a per-branch agent factory:
    `run_agents(agents={name: fresh(lambda: MyLLMAgent(...))})`. Use this for
    any agent that carries state between turns, so no state crosses branches."""
    return AgentFactory(make)


def _as_agent(a) -> Agent:
    if isinstance(a, AgentFactory):  # build a fresh instance per seed
        return a.build()
    if hasattr(a, "act"):
        return a
    if hasattr(a, "orders"):  # a ClientPolicy
        return PolicyAgent(a)
    if callable(a):
        return FunctionAgent(a)
    raise TypeError(f"not an agent: {a!r}")


def _seat_agent(spec, isolate: bool = True) -> Agent:
    """Build one branch's agent from a spec, keeping branches independent without
    the caller having to remember to:

    - a `fresh(...)` factory yields a brand-new instance — full isolation, the
      reliable choice for any stateful agent;
    - a bare instance that defines `reset()` is reused and reset, trusting it to
      clear its own per-branch state (so a heavy model is not copied);
    - otherwise, when `isolate` (the default — every branch, so the caller's own
      object is never mutated across seeds *or* across separate run calls), the
      instance is deep-copied, and if it cannot be copied we refuse loudly.

    **Limit:** `deepcopy` copies attribute state, but it does NOT
    copy a Python function/closure, so a `FunctionAgent` wrapping a *stateful
    closure* (captured `nonlocal`/list) still shares that state, as does any
    externally-shared object. Such agents must use `fresh(...)` or `reset()`;
    deepcopy is a best-effort convenience for the simple attribute-state case,
    not a guarantee for every agent."""
    if isinstance(spec, AgentFactory):
        return spec.build()
    agent = _as_agent(spec)
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()
        return agent
    if isolate:
        try:
            return copy.deepcopy(agent)
        except Exception as e:
            raise TypeError(
                f"agent {type(agent).__name__} carries state that cannot be "
                f"deep-copied for per-seed isolation ({e!r}); wrap it in "
                f"fresh(lambda: ...) so each seed builds its own instance, or give "
                f"it a reset() method that clears its per-branch state"
            ) from e
    return agent


def _obs_save(
    saves_dir: Path, load_save: Path, turn: int, from_turn: int, wait: float = 5.0
) -> Path:
    """The savegame that represents `turn`'s starting state. The resumed turn
    itself emits no autosave, so it's the loaded save; later turns are the
    per-turn autosaves (saveturns=1). Polls briefly for the autosave (it is
    written at turn-begin, just before START_PHASE, and a slow filesystem can
    make it visible after the packet) and raises `MissingObservation` if it
    never lands, rather than silently returning the wrong-turn loaded save."""
    if turn <= from_turn:
        return load_save
    pattern = f"*-T{turn:04d}-*.sav*"
    deadline = time.time() + wait
    while True:
        matches = sorted(saves_dir.glob(pattern))
        if matches:
            return matches[0]
        if time.time() >= deadline:
            raise MissingObservation(
                f"no autosave matching {pattern!r} in {saves_dir} after {wait}s"
            )
        time.sleep(0.05)


def _bind_legal_actions(cli):
    """Expose the client's legal-action query to an agent as a bare closure, not
    the bound method `cli.get_unit_actions` — handing over the bound method would
    let an agent reach the raw client via `.__self__` and drive it around the
    EnvSpec. The closure removes that trivial handle; it is
    not a hard sandbox (a determined agent can still reach `cli` via the
    closure's cell — see the scope note in civharness.envspec), just no free
    client on a plate."""

    def legal_actions(
        actor_unit_id,
        target_tile=-1,
        target_unit=0,
        target_extra=-1,
        request_kind=1,
        timeout=5.0,
    ):
        return cli.get_unit_actions(
            actor_unit_id,
            target_tile=target_tile,
            target_unit=target_unit,
            target_extra=target_extra,
            request_kind=request_kind,
            timeout=timeout,
        )

    return legal_actions


def _drive(
    clients,
    saves_dir,
    load_save,
    from_turn,
    max_seconds,
    env,
    episode=None,
    *,
    phase_budget: float = DEFAULT_PHASE_BUDGET,
):
    """Round-robin the controlled players through the shared game. Each round:
    bring every still-active client to this turn's phase, observe once, then let
    each agent act for its player and end its phase — in a fixed order, so the
    server sees the same packet sequence every replay (determinism). Orders are
    gated by `env` (a forbidden action raises in strict mode, else is dropped).
    If `episode` is given, each player's observation source and the orders
    actually applied are recorded so the run can be replayed. No single phase
    wait may exceed `phase_budget` seconds."""
    active = list(clients)
    deadline = time.time() + max_seconds
    last_players = None  # newest observed player table (classification fallback)
    while active and time.time() < deadline:
        phase = []
        for cli, agent, pname in active:
            budget = min(phase_budget, deadline - time.time())
            if budget <= 0:
                break
            t = cli.next_phase(max_seconds=budget)
            if t is None:
                if cli.over:
                    continue  # this seat's game ended cleanly
                if time.time() >= deadline:
                    break  # overall drive deadline (not a control loss)
                # Dropped, or no phase arrived: with autotoggle the AI would
                # retake this seat and finish the game, so a trajectory
                # continued now would be a lie. Fail loud. A missing phase
                # cannot separate a stalled seat from a dead player, so the
                # class comes from the save's player table. A dropped socket
                # is an infrastructure failure whatever the player's state.
                if cli.dropped:
                    reason = "connection dropped"
                    cls = ELIM_INFRA
                else:
                    reason = f"no phase within {budget:.0f}s (seat stalled/eliminated)"
                    cls = classify_control_loss(
                        saves_dir, [pname], fallback_players=last_players
                    )
                raise control_lost(cls, f"lost control of {pname!r}: {reason}")
            phase.append((cli, agent, pname, t))
        if not phase:
            break
        turn = phase[0][3]
        obs_path = _obs_save(Path(saves_dir), load_save, turn, from_turn)
        _t, all_players, terrain = observe_state(obs_path, env)
        last_players = all_players
        self_only = env.obs("players", "all") == "self"
        want_legal = env.wants("legal_actions")
        fogged = env.obs("visibility", "full_state") == "player_visible"
        # Only a full-information env gets the raw save path; a restricted one
        # withholds it so the agent cannot re-parse the god-view save and
        # recover what the observation is meant to hide.
        expose_save = env.full_information()
        # Multi-agent order independence: freeze EVERY player's view
        # cache at this decision boundary BEFORE any player acts, so the
        # packets triggered by an earlier seat's orders cannot leak into a
        # later seat's observation of the same turn.
        view_snaps = (
            {pname: cli.view.snapshot() for cli, _a, pname, _t2 in phase}
            if fogged
            else {}
        )
        for cli, agent, pname, _t in phase:
            mine = all_players.get(pname, {})
            if fogged:
                # own-player facts (gold/techs/government/rates/diplomacy) are
                # never fogged for oneself; the WORLD comes from the client's
                # own frozen cache, and other players' entity lists are absent
                pview = _pv.player_view(
                    view_snaps[pname],
                    cli.rules,
                    my_player_id=_player_id_of(all_players, pname, saves_dir=saves_dir),
                    catalog=cli.rulesets,
                )
                mine = dict(mine)
                mine["cities"] = [c for c in pview["cities"] if c["mine"]]
                mine["units"] = [u for u in pview["units"] if u["mine"]]
                visible = {pname: mine}
            else:
                pview = None
                visible = {pname: mine} if self_only else all_players
            audit_box: dict = {}
            obs = Observation(
                turn=turn,
                me=pname,
                my=mine,
                players=visible,
                save_path=str(obs_path) if (expose_save and not fogged) else None,
                terrain=None if fogged else terrain,
                rates=mine.get("rates"),
                diplomacy=mine.get("diplomacy"),
                legal_actions=(_bind_legal_actions(cli) if want_legal else None),
                player_view=pview,
                rulesets=(cli.rulesets if want_legal else None),
                rule_ids=(cli.rules if want_legal else None),
                audit=(audit_box.update if episode is not None else None),
            )
            applied = []
            for order in agent.act(obs):
                if env.check_order(order):  # strict: raises; lenient: skip
                    order.apply(cli)
                    applied.append(order)
            if episode is not None:
                episode.step(turn, pname, obs_path, applied, audit=audit_box or None)
            cli.phase_done()
        active = [(c, a, p) for (c, a, p, _t) in phase]
    # The loop ended. If it stopped on the wall-clock deadline rather than every
    # seat reaching game-over, the drive is incomplete: with autotoggle the AI
    # takes the vacated seats and finishes the game, so returning now would
    # report AI play as the agent's. Fail loud — the timeout twin of the dropped
    # /stalled cases handled above, and the mirror of client_branch's guard.
    if not all(cli.over for cli, _a, _p in clients):
        live = [p for cli, _a, p in clients if not cli.over]
        # "Still live" here means the client never saw game-over, which is not
        # the same as the player being alive. Ask the save's player table.
        cls = classify_control_loss(saves_dir, live, fallback_players=last_players)
        raise control_lost(
            cls,
            f"agent drive did not reach game-over within {max_seconds:.0f}s "
            f"(seats still live: {live}); refusing a possibly AI-finished "
            f"trajectory",
        )


def run_agents(
    save: Path | SaveRef,
    workdir: Path,
    *,
    agents: dict,
    until: int,
    env: EnvSpec | None = None,
    seeds: list[int] | None = None,
    k: int | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    scorelog: bool = True,
    username_prefix: str = "civ",
    binary: Path = DEFAULT_BINARY,
    timeout_scale: float = 1.0,
    phase_budget: float = DEFAULT_PHASE_BUDGET,
) -> list[Trajectory]:
    """Fork continuations of `save` in which each player named in `agents` is
    driven by its own Agent, all in one shared game (uncontrolled seats stay
    AI). Returns one Trajectory per seed (the whole game's outcome).

    agents        : {player_name: Agent | AgentFactory | ClientPolicy |
                    fn(obs)->[Order]}; names come from players_of(save), never
                    guessed. Wrap a stateful agent in fresh(...) so it is
                    rebuilt per seed.
    env           : an EnvSpec restricting the allowed actions (and, via
                    observe_state, the observation fields). Defaults to
                    EnvSpec.full() — unrestricted.
    seeds/k       : as in branch()/client_branch(); one game per seed.
    timeout_scale : multiplies the bot-speed `default_timeout(until)` used for
                    the drive deadline and the server's exit wait; raise it when
                    each turn also waits on slow agents (e.g. a policy model
                    shared by many concurrent games).
    phase_budget  : ceiling (seconds) on one wait for a seat's next phase; the
                    drive deadline also scales with phase_budget / 60 (see
                    `drive_timeout`).

    Determinism: with the server waiting on us (timeout 0) and each round's
    orders issued in a fixed player order, same seed + deterministic agents ->
    byte-identical continuation.
    """
    env = env or EnvSpec.full()
    save = save.path if isinstance(save, SaveRef) else Path(save)
    from_turn = turn_of_save(save)
    from_save_sha = parse.normalized_save_sha(save)  # provenance for replay
    if seeds is None:
        seeds = list(range(1, k + 1)) if k else [None]
    workdir = Path(workdir)
    out = []
    for i, seed in enumerate(seeds):
        bdir = workdir / f"c{i:03d}-s{seed}"
        bdir.mkdir(parents=True, exist_ok=True)
        _ensure_clean_saves(bdir)
        # Isolate every branch's agent — not just across seeds within this call,
        # but across separate run_agents calls sharing the same agent objects
        # (e.g. agent_noise_floor's replica + spread runs).
        # fresh(...) rebuilds, reset() clears, else the bare instance is
        # deep-copied so the caller's object is never mutated.
        seat_agents = {name: _seat_agent(a, isolate=True) for name, a in agents.items()}
        load = save if seed is None else reseed_save(save, bdir / "reseeded.sav", seed)
        (bdir / "dump.lua").write_text(RULE_DUMP_LUA)
        serv = client_resume_script(
            until,
            dump_lua_path=bdir / "dump.lua",
            skill=skill,
            skill_by_player=skill_by_player,
            scorelog=scorelog,
            saveturns=1,  # per-turn autosaves are the agents' observations
        )
        episode = EpisodeWriter(
            bdir / "episode.jsonl",
            {
                "seed": seed,
                "from_turn": from_turn,
                "from_save_sha": from_save_sha,  # replay must match this position
                "until": until,
                "players": list(seat_agents),
                "agents": {p: type(a).__name__ for p, a in seat_agents.items()},
                "skill": skill,
                # Record the per-player AI levels too, so a replay restores the
                # exact opponent behaviour and cannot silently diverge.
                "skill_by_player": dict(skill_by_player) if skill_by_player else None,
                "env": {
                    "name": env.name,
                    "actions": sorted(env.actions),
                    "strict": env.strict,
                    "observations": env.observations,
                },
                # Provenance: schema identifiers of the typed layers this run's
                # observations/candidates conform to, plus the visibility mode,
                # so downstream data can be attributed without inspecting code.
                "provenance": {
                    "visibility": env.obs("visibility", "full_state"),
                    "observation_schema": OBSERVATION_SCHEMA,
                    "scene_schema": _spatial.SCENE_SCHEMA,
                    "digest_schema": _spatial.DIGEST_SCHEMA,
                    "candidate_schema": _cand.ACTION_CANDIDATE_SCHEMA,
                    "graph_config_sha": _spatial.SceneGraphConfig.saga().sha,
                },
            },
        )
        h = server.launch(serv, bdir, binary=binary, load=load)
        try:
            wait_for_log_marker(h, "CIVHARNESS dump_done", timeout=60)
            h.assert_loaded()
            rules = RuleIds.from_log(h.log)
            clients = []
            for idx, (pname, agent) in enumerate(seat_agents.items()):
                cli = FreecivClient(
                    "127.0.0.1", h.port, username=f"{username_prefix}{idx}"
                )
                cli.rules = rules
                cli.turn = from_turn
                cli.connect()
                cli.pump(0.5)
                cli.take(pname)
                clients.append((cli, agent, pname))
            for cli, _a, _p in clients:
                cli.pump(0.3)  # let the takes + autotoggle settle (still pregame)
            for cli, _a, _p in clients:
                cli.start()  # ready each seat; game starts when the last readies
            # NB: no pump() here — it would swallow the first START_PHASE.
            _drive(
                clients,
                bdir / "saves",
                load,
                from_turn,
                drive_timeout(
                    until, timeout_scale=timeout_scale, phase_budget=phase_budget
                ),
                env,
                episode=episode,
                phase_budget=phase_budget,
            )
            for cli, _a, _p in clients:
                cli.close()
            h.wait(timeout_scale * default_timeout(until))
            h.assert_loaded()
            final = _final_save(h)
            sections = parse.parse_save_sections(final)
            final_sha = parse.normalized_save_sha(final)
            final_turn = parse.save_turn(sections)
            # Close the episode with a completion footer carrying the final save
            # digest, BEFORE closing the file — an exact replay verifies against
            # this to reject a truncated or last-turn-divergent log.
            episode.footer(final_save_sha=final_sha, final_turn=final_turn)
        except Exception:
            h.kill()
            raise
        finally:
            episode.close()
        out.append(
            Trajectory(
                seed=seed,
                skill=skill,
                skill_by_player=dict(skill_by_player) if skill_by_player else None,
                from_turn=from_turn,
                turns=final_turn,
                save_path=str(final),
                save_sha256_normalized=final_sha,
                scores=parse.save_scores(sections),
                scorelog_path=str(h.scorelog) if h.scorelog.exists() else None,
                workdir=str(bdir),
                episode_log_path=str(bdir / "episode.jsonl"),
            )
        )
    return out


def run_agent(
    save: Path | SaveRef,
    workdir: Path,
    *,
    focal_player: str,
    agent,
    until: int,
    **kwargs,
) -> list[Trajectory]:
    """Single-seat convenience: one agent controls `focal_player`, the rest AI."""
    return run_agents(
        save, workdir, agents={focal_player: agent}, until=until, **kwargs
    )


def agent_noise_floor(
    save: Path | SaveRef,
    workdir: Path,
    *,
    agents: dict,
    until: int,
    env: EnvSpec | None = None,
    seeds: list[int] | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    binary: Path = DEFAULT_BINARY,
    timeout_scale: float = 1.0,
    phase_budget: float = DEFAULT_PHASE_BUDGET,
) -> FloorReport:
    """The mandatory companion to any agent-driven effect estimate: a same-seed
    replica identity check plus the different-seed spread of final total score.
    Mirrors branch.noise_floor for the agent path."""
    save = save.path if isinstance(save, SaveRef) else Path(save)
    workdir = Path(workdir)
    timeouts = {"timeout_scale": timeout_scale, "phase_budget": phase_budget}
    replays = [
        run_agents(
            save,
            workdir / f"replay{i}",
            agents=agents,
            until=until,
            env=env,
            skill=skill,
            skill_by_player=skill_by_player,
            scorelog=False,
            binary=binary,
            **timeouts,
        )[0]
        for i in (0, 1)
    ]
    identical = replays[0].save_sha256_normalized == replays[1].save_sha256_normalized
    seeds = seeds or [101, 102, 103, 104]
    trajs = run_agents(
        save,
        workdir / "spread",
        agents=agents,
        until=until,
        env=env,
        seeds=seeds,
        skill=skill,
        skill_by_player=skill_by_player,
        scorelog=False,
        binary=binary,
        **timeouts,
    )
    spread = {}
    for p in trajs[0].scores:
        vals = [t.scores[p]["total"] for t in trajs if p in t.scores]
        spread[p] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return FloorReport(
        replicas_identical=identical, n_seeds=len(seeds), score_spread=spread
    )
