"""Snapshot / branch / readout: forking continuations of a saved position.

Branch semantics
----------------
A savegame stores the complete RNG generator state (`[random] saved=TRUE`),
so loading and continuing replays the original stream: bit-identical, the
noise-free replica. To get a *different* continuation, `reseed_save()`
invalidates the stored state and rewrites the save's `gameseed` setting; the
server then derives a fresh deterministic stream from that seed. Same seed
-> same continuation; different seed -> independent continuation.

Noise discipline
----------------
`branch()` returns trajectories; `noise_floor()` returns the same-seed
replica check plus the different-seed spread. An effect measured between
branches is only meaningful relative to that floor, so the floor is a
first-class return value.

Control loss
------------
The client-driven paths (`client_branch`, and `civharness.agent.run_agents`)
raise `ControlLost` whenever a controlled seat stops receiving phases before
game over. `classify_control_loss` reads the save's player table to tag the
exception as a real elimination (GAME), a stalled but still-alive seat
(INFRA), or undecidable (UNKNOWN).
"""

import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from civharness import parse, server
from civharness.config import (
    DEFAULT_BINARY,
    GameConfig,
    client_resume_script,
    resume_script,
)
from civharness.envspec import EnvSpec
from civharness.runner import default_timeout, run_game


@dataclass(frozen=True)
class SaveRef:
    """A savegame and the turn it was written at."""

    path: Path
    turn: int


@dataclass(frozen=True)
class Trajectory:
    """The endpoint of one branch: final save, its digest and the scores."""

    seed: int | None  # None = replay of the stored RNG stream
    skill: str | None
    skill_by_player: dict | None  # per-player AI overrides ({} recorded as None)
    from_turn: int
    turns: int
    save_path: str
    save_sha256_normalized: str
    scores: dict  # {player: {metric: int}} at the endpoint
    scorelog_path: str | None
    workdir: str
    episode_log_path: str | None = None  # agent episode log (None for AI-only)

    def series(self, tag: str, player: int | str):
        if not self.scorelog_path:
            raise ValueError("trajectory ran without scorelog")
        return parse.parse_scorelog(Path(self.scorelog_path)).series(tag, player)


def turn_of_save(path: Path) -> int:
    """The turn a savegame was written at: from the `-T<turn>-` part of the
    server's autosave name, else from the save's `[game] turn`."""
    m = re.search(r"-T(\d+)-", Path(path).name)
    if m:
        return int(m.group(1))
    return parse.save_turn(parse.parse_save_sections(path))


def _ensure_clean_saves(bdir: Path) -> None:
    """Delete any stale savegames in a branch's (harness-owned) saves dir before
    the run. Otherwise a reused workdir still holds the saves of a *longer*
    previous run, and `_final_save` would return that stale trajectory instead
    of this one's."""
    sd = Path(bdir) / "saves"
    if sd.exists():
        for p in sd.glob("*.sav*"):
            p.unlink()


def _final_save(h) -> Path:
    """The savegame this run actually ended on: the highest-turn save in the
    (freshly cleaned) saves dir. Robust to lexicographic sort quirks by keying
    on the parsed turn, and loud if the run produced nothing."""
    saves = h.saves()
    if not saves:
        raise server.ServerHang(f"no savegame produced in {h.savedir}")
    return max(saves, key=lambda p: (turn_of_save(p), p.name))


def players_of(save: Path | SaveRef) -> dict[int, str]:
    """{player index: name} read from the save — the input to a
    skill_by_player profile."""
    save = save.path if isinstance(save, SaveRef) else Path(save)
    return parse.save_players(parse.parse_save_sections(save))


# --- ControlLost classification --------------------------------------------
#
# `ControlLost` is the single exit for two facts that consumers must treat
# differently: a player that is genuinely out of the game, and a still-live
# seat that merely stopped receiving phases. The driver-level symptom cannot
# separate them (no phase arrived either way), so the save's own player table
# is the discriminator. Drivers tag the exception message with one of these
# prefixes; consumers read the prefix.

ELIM_GAME = "GAME"
ELIM_INFRA = "INFRA"
ELIM_UNKNOWN = "UNKNOWN"


def _alive_in_save(save: Path, names) -> dict:
    """{name: still in the game} for `names`, from a save's player table.
    A name with no [playerN] section is gone; so is one with is_alive FALSE."""
    sections = parse.parse_save_sections(Path(save))
    alive = {n: False for n in names}
    for sec, kv in sections.items():
        if not re.fullmatch(r"player(\d+)", sec):
            continue
        name = kv.get("name", "").strip('"')
        if name in alive:
            alive[name] = kv.get("is_alive", "").upper().endswith("TRUE")
    return alive


def _newest_save(saves_dir: Path) -> Path | None:
    """The most recently written savegame in a run's saves dir, or None.
    Keyed on mtime rather than the parsed turn: this runs on a failure path
    where a partially written save must not cost a whole save parse."""
    saves = list(Path(saves_dir).glob("*.sav*"))
    if not saves:
        return None
    return max(saves, key=lambda p: p.stat().st_mtime)


def classify_control_loss(saves_dir, names, fallback_players=None) -> str:
    """Is a lost seat's player still IN the game? -> ELIM_GAME / ELIM_INFRA /
    ELIM_UNKNOWN.

      INFRA    every named seat is still alive in the newest save: the seat
               stalled, the civilisation did not die — NOT a terminal outcome
      GAME     every named seat is absent from the save (or is_alive FALSE):
               a real elimination, a genuine terminal outcome
      UNKNOWN  no save could be read, or the named seats disagree — fail-safe,
               callers must treat it as "not known to be a game outcome"

    `fallback_players` is the last save-derived player table the driver
    observed (`observe_state`'s second return). It is up to one turn stale, so
    it is consulted only when the saves dir itself cannot be read.
    """
    names = list(names)
    if not names:
        return ELIM_UNKNOWN
    alive = None
    try:
        newest = _newest_save(saves_dir)
        if newest is not None:
            alive = _alive_in_save(newest, names)
    except Exception:  # noqa: BLE001 -- unparseable / half-written save
        alive = None  # fall through to the fallback table, then UNKNOWN
    if alive is None and fallback_players:
        alive = {
            n: bool(fallback_players.get(n, {}).get("is_alive", False)) for n in names
        }
    if alive is None:
        return ELIM_UNKNOWN
    seen = {bool(alive.get(n, False)) for n in names}
    if seen == {True}:
        return ELIM_INFRA
    if seen == {False}:
        return ELIM_GAME
    return ELIM_UNKNOWN


def control_lost(cls: str, reason: str):
    """Build a classified ControlLost: the class (ELIM_GAME / ELIM_INFRA /
    ELIM_UNKNOWN) is the message prefix (`"INFRA: ..."`), which is what
    consumers parse, and is also attached as `exc.kind`."""
    from civharness.client import ControlLost

    exc = ControlLost(f"{cls}: {reason}")
    exc.kind = cls
    return exc


def snapshot_series(
    config: GameConfig, workdir: Path, *, every: int, binary: Path = DEFAULT_BINARY
) -> list[SaveRef]:
    """One seeded game, autosaving every N turns: a position library from a
    single pass. The saves stay in `workdir/saves`; `load_snapshot_series`
    reads them back without replaying the game."""
    cfg = GameConfig(**{**config.__dict__, "saveturns": every})
    run_game(cfg, workdir, binary=binary)
    return load_snapshot_series(workdir)


def load_snapshot_series(workdir: Path) -> list[SaveRef]:
    """The SaveRefs of a finished `snapshot_series(config, workdir, ...)` run,
    in the same (file-name) order `snapshot_series` returns them."""
    saves = sorted(Path(workdir, "saves").glob("*.sav*"))
    return [SaveRef(path=s, turn=turn_of_save(s)) for s in saves]


_RANDOM_SECTION_RE = re.compile(r"^\[random\]\n.*?(?=^\[)", re.MULTILINE | re.DOTALL)


def reseed_save(save: Path, out: Path, gameseed: int) -> Path:
    """Copy `save` with its `[random]` section replaced by the exact
    generator state fc_srand(gameseed) would produce.

    This is the only deterministic way to re-seed a branch: the engine loads
    [random] before [settings], so an invalidated stream falls back to
    wall-clock entropy and `set gameseed` post-load is silently too late
    (verified empirically and in savegame3.c). See civharness/rand.py.
    """
    from civharness.rand import random_section

    txt = Path(save).read_text()
    new_txt, n = _RANDOM_SECTION_RE.subn(random_section(gameseed) + "\n", txt, count=1)
    if n != 1:
        raise ValueError(f"no [random] section found in {save}")
    # Keep the recorded setting consistent with the injected state (cosmetic
    # — the state above is what actually drives the stream).
    new_txt = re.sub(
        r'"gameseed",\d+,\d+', f'"gameseed",{gameseed},{gameseed}', new_txt, count=1
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new_txt)
    return out


def branch(
    save: Path | SaveRef,
    workdir: Path,
    *,
    until: int,
    seeds: list[int] | None = None,
    k: int | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    scorelog: bool = True,
    policy=None,
    binary: Path = DEFAULT_BINARY,
) -> list[Trajectory]:
    """Fork continuations of `save` to turn `until`.

    seeds=None, k=None      -> one replay of the stored stream.
    seeds=[s1, s2, ...]     -> one branch per seed (k inferred).
    k=N                     -> seeds 1..N.
    skill                   -> base AI level for every player in the branch.
    skill_by_player         -> per-player overrides {name: level}, applied
                               AFTER the base level, so an opponent profile
                               ("focal normal, rivals hard") is expressible.
                               Get names via players_of(save); never guess.
    policy                  -> a ScriptedPolicy (server-side Lua orders)
                               executed inside each branch.
    """
    save = save.path if isinstance(save, SaveRef) else Path(save)
    if seeds is None:
        seeds = list(range(1, k + 1)) if k else [None]
    workdir = Path(workdir)
    out = []
    for i, seed in enumerate(seeds):
        bdir = workdir / f"b{i:03d}-s{seed}"
        bdir.mkdir(parents=True, exist_ok=True)
        _ensure_clean_saves(bdir)
        load = save if seed is None else reseed_save(save, bdir / "reseeded.sav", seed)
        extra = ()
        if policy is not None:
            lua_path = policy.write(bdir / "policy.lua")
            extra = (f"lua file {lua_path}",)
        h = server.launch(
            resume_script(
                until,
                skill=skill,
                skill_by_player=skill_by_player,
                scorelog=scorelog,
                extra=extra,
            ),
            bdir,
            binary=binary,
            load=load,
        )
        h.wait(default_timeout(until))
        h.assert_loaded()
        final = _final_save(h)
        sections = parse.parse_save_sections(final)
        out.append(
            Trajectory(
                seed=seed,
                skill=skill,
                skill_by_player=dict(skill_by_player) if skill_by_player else None,
                from_turn=turn_of_save(save),
                turns=parse.save_turn(sections),
                save_path=str(final),
                save_sha256_normalized=parse.normalized_save_sha(final),
                scores=parse.save_scores(sections),
                scorelog_path=str(h.scorelog) if h.scorelog.exists() else None,
                workdir=str(bdir),
            )
        )
    return out


def wait_for_log_marker(h, marker: str, timeout: float) -> None:
    """Poll a launched server's log until `marker` appears (the Lua id dump
    signals completion this way), so the client only reads a complete table."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if h.proc.poll() is not None:
            break  # server exited; assert_loaded/wait will surface why
        if h.log.exists() and marker in h.log.read_text(errors="replace"):
            return
        time.sleep(0.1)
    raise server.ServerHang(f"marker {marker!r} not seen in {h.log} within {timeout}s")


def _gated_on_phase(policy, env: EnvSpec):
    """Adapt a policy to `on_phase(client, turn)`, gating each order through the
    EnvSpec. A ClientPolicy (exposes `.orders`) is filtered order-by-order (a
    forbidden order raises in strict mode). A custom policy with its own
    `on_phase` drives the client directly, so the harness cannot inspect its
    packets: rather than run it silently ungated under a restricted action space
    (a gating bypass), it is refused loudly — an unrestricted env (`full()`)
    still runs it as-is."""
    if policy is None:
        return None
    orders_by_turn = getattr(policy, "orders", None)
    if orders_by_turn is None:
        if env.restricts_actions():
            raise ValueError(
                "EnvSpec action gating cannot be enforced on a custom on_phase "
                "policy (the harness cannot see which packets it sends). Build the "
                "policy from order objects (a ClientPolicy), or pass "
                "env=EnvSpec.full() to run it ungated on purpose."
            )
        return policy.on_phase

    def on_phase(client, turn):
        for order in orders_by_turn.get(turn, ()):
            if env.check_order(order):
                order.apply(client)

    return on_phase


def client_branch(
    save: Path | SaveRef,
    workdir: Path,
    *,
    until: int,
    focal_player: str,
    policy=None,
    env: EnvSpec | None = None,
    seeds: list[int] | None = None,
    k: int | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    scorelog: bool = True,
    saveturns: int | None = None,
    username: str = "civharness",
    binary: Path = DEFAULT_BINARY,
) -> list[Trajectory]:
    """Fork continuations of `save` in which an external client makes the
    genuine decisions for `focal_player` — production, research, unit orders —
    the ones server Lua cannot express.

    Mechanism: the server loads the save in pregame (client_resume_script,
    which never `start`s), a network client takes the focal player and issues
    `policy`'s orders each turn, ending each phase so the game advances only on
    our cue. Same seed -> bit-identical continuation (determinism holds with a
    client in the loop because the server waits for us — no wall clock — and
    our packet sequence is fixed by the policy).

    seeds/k mirror branch(); `policy` is a civharness.policy.client.ClientPolicy
    (or any object with `on_phase(client, turn)`). `env` (an EnvSpec) gates a
    ClientPolicy's orders by action_key — a forbidden order raises in strict
    mode. `focal_player` is a player name from the save (players_of(save));
    never guessed.
    """
    from civharness.client import RULE_DUMP_LUA, FreecivClient, RuleIds

    env = env or EnvSpec.full()
    save = save.path if isinstance(save, SaveRef) else Path(save)
    from_turn = turn_of_save(save)
    if seeds is None:
        seeds = list(range(1, k + 1)) if k else [None]
    on_phase = _gated_on_phase(policy, env)
    workdir = Path(workdir)
    out = []
    for i, seed in enumerate(seeds):
        bdir = workdir / f"c{i:03d}-s{seed}"
        bdir.mkdir(parents=True, exist_ok=True)
        _ensure_clean_saves(bdir)
        load = save if seed is None else reseed_save(save, bdir / "reseeded.sav", seed)
        dump_path = bdir / "dump.lua"
        dump_path.write_text(RULE_DUMP_LUA)
        serv = client_resume_script(
            until,
            dump_lua_path=dump_path,
            skill=skill,
            skill_by_player=skill_by_player,
            scorelog=scorelog,
            saveturns=saveturns,
        )
        h = server.launch(serv, bdir, binary=binary, load=load)
        try:
            wait_for_log_marker(h, "CIVHARNESS dump_done", timeout=60)
            h.assert_loaded()
            cli = FreecivClient("127.0.0.1", h.port, username=username)
            cli.rules = RuleIds.from_log(h.log)
            cli.turn = from_turn  # the resumed turn emits no NEW_YEAR
            cli.connect()
            cli.pump(1.0)  # absorb the login state dump
            cli.take(focal_player)
            cli.pump(0.5)  # let the take + autotoggle settle
            cli.start()
            cli.run(on_phase=on_phase, max_seconds=default_timeout(until))
            over = cli.over
            cli.close()
            if not over:
                # The client stopped driving before game-over (drop or timeout);
                # do NOT accept a trajectory the AI may have ghost-finished.
                # A dropped socket is never a game outcome; a silent timeout
                # may be, so the save's player table decides.
                cls = (
                    ELIM_INFRA
                    if cli.dropped
                    else classify_control_loss(bdir / "saves", [focal_player])
                )
                raise control_lost(
                    cls,
                    f"lost control of {focal_player!r} before game over "
                    f"(dropped={cli.dropped}); refusing a possibly AI-finished "
                    f"trajectory. Log: {h.log}",
                )
            h.wait(default_timeout(until))  # server exits at game over (-e)
        except Exception:
            h.kill()
            raise
        h.assert_loaded()
        final = _final_save(h)
        sections = parse.parse_save_sections(final)
        out.append(
            Trajectory(
                seed=seed,
                skill=skill,
                skill_by_player=dict(skill_by_player) if skill_by_player else None,
                from_turn=from_turn,
                turns=parse.save_turn(sections),
                save_path=str(final),
                save_sha256_normalized=parse.normalized_save_sha(final),
                scores=parse.save_scores(sections),
                scorelog_path=str(h.scorelog) if h.scorelog.exists() else None,
                workdir=str(bdir),
            )
        )
    return out


@dataclass(frozen=True)
class FloorReport:
    """Same-seed replica identity plus the different-seed score spread."""

    replicas_identical: bool  # same stream replayed twice, bit-level
    n_seeds: int
    score_spread: dict  # {player: {"mean": .., "stdev": .., "values": [..]}}

    def stdev(self, player: str) -> float:
        return self.score_spread[player]["stdev"]


def noise_floor(
    save: Path | SaveRef,
    workdir: Path,
    *,
    until: int,
    seeds: list[int] | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    binary: Path = DEFAULT_BINARY,
) -> FloorReport:
    """The mandatory companion to any effect estimate at this position:
    replay determinism check + different-seed spread of final total score."""
    workdir = Path(workdir)
    replays = [
        branch(
            save,
            workdir / f"replay{i}",
            until=until,
            skill=skill,
            skill_by_player=skill_by_player,
            scorelog=False,
            binary=binary,
        )[0]
        for i in (0, 1)
    ]
    identical = replays[0].save_sha256_normalized == replays[1].save_sha256_normalized
    seeds = seeds or [101, 102, 103, 104]
    trajs = branch(
        save,
        workdir / "spread",
        until=until,
        seeds=seeds,
        skill=skill,
        skill_by_player=skill_by_player,
        scorelog=False,
        binary=binary,
    )
    spread = {}
    players = trajs[0].scores.keys()
    for p in players:
        vals = [t.scores[p]["total"] for t in trajs if p in t.scores]
        spread[p] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return FloorReport(
        replicas_identical=identical, n_seeds=len(seeds), score_spread=spread
    )


@dataclass(frozen=True)
class PositionState:
    """A save's turn and per-player summary (see `position`)."""

    turn: int
    # players: name -> {gold, nation, is_alive, ncities, nunits,
    #                   techs, score, cities: [{name, size, building}]}
    players: dict


def position(save: Path | SaveRef) -> PositionState:
    """Structured summary of a position for downstream rendering/features."""
    save = save.path if isinstance(save, SaveRef) else Path(save)
    sections = parse.parse_save_sections(save)
    scores = parse.save_scores(sections)
    players = {}
    for sec, kv in sections.items():
        m = re.fullmatch(r"player(\d+)", sec)
        if not m:
            continue
        name = kv.get("name", "").strip('"')
        if not name:
            continue
        cities = []
        if "c" in kv:
            for row in parse.parse_table(kv["c"]):
                cities.append(
                    {
                        "id": int(row["id"]) if row.get("id", "").isdigit() else None,
                        "name": row.get("name"),
                        "size": row.get("size"),
                        "building": row.get(
                            "currently_building_name",
                            row.get("currently_building_rule_name"),
                        ),
                    }
                )
        units = []
        if "u" in kv:
            for row in parse.parse_table(kv["u"]):
                units.append(
                    {
                        "id": int(row["id"]) if row.get("id", "").isdigit() else None,
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "type": row.get("type_by_name"),
                        "activity": row.get("activity"),
                    }
                )
        score = scores.get(name, {})
        players[name] = {
            "player_id": int(m.group(1)),
            "gold": int(kv.get("gold", "0") or 0),
            "nation": kv.get("nation", "").strip('"'),
            "is_alive": kv.get("is_alive", "").upper().endswith("TRUE"),
            "ncities": int(kv.get("ncities", len(cities) or 0) or 0),
            "nunits": int(kv.get("nunits", "0") or 0),
            "techs": score.get("techs"),
            "score": score.get("total"),
            "cities": cities,
            "units": units,
        }
    return PositionState(turn=parse.save_turn(sections), players=players)
