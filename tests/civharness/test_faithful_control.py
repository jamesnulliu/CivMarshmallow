"""Correctness guards for the client-controlled path:

- a reused branch workdir must NOT return a previous, longer run's stale
  trajectory;
- losing control of a seat before game-over must FAIL LOUD, not silently let
  the autotoggle AI ghost-finish a trajectory reported as agent-controlled —
  whether the loss is a dropped client, a stalled seat, OR the overall drive
  deadline being reached;
- per-seed agent isolation is automatic for multi-seed runs
  (fresh()/reset()/deep-copy), so a bare stateful instance cannot leak state
  across branches.

The stale-dir / state-isolation cases need the pinned binary; the control-loss
cases drive `_drive` with a fake client (one real save for the observation
parse), and the `_seat_agent` cases need no server at all.
"""

import threading

import pytest

from civharness import (
    ControlLost,
    EnvSpec,
    FunctionAgent,
    fresh,
    position,
    run_agents,
)
from civharness.agent import _bind_legal_actions, _drive, _seat_agent
from civharness.branch import ELIM_GAME, ELIM_INFRA, ELIM_UNKNOWN, branch
from civharness.policy.client import BuildUnit


def test_stale_dir_not_reused(save, root):
    """Reusing a workdir for a SHORTER run must return the short trajectory,
    not the stale longer one left in the saves dir. Proven by comparing the
    reused-dir run against a fresh clean run of the same short branch: they must
    be byte-identical (they would differ if the stale long save leaked)."""
    fresh_short = branch(save, root / "fresh", until=save.turn + 3, seeds=[1])
    wd = root / "reused"
    long = branch(save, wd, until=save.turn + 8, seeds=[1])
    reused_short = branch(save, wd, until=save.turn + 3, seeds=[1])  # SAME wd + seed
    assert reused_short[0].turns < long[0].turns, (
        f"stale save leaked: reused dir returned T{reused_short[0].turns}, "
        f"the stale long run was T{long[0].turns}"
    )
    assert reused_short[0].turns == fresh_short[0].turns
    assert (
        reused_short[0].save_sha256_normalized == fresh_short[0].save_sha256_normalized
    ), "reused dir did not reproduce the clean short run (stale save leaked)"


class _OnceBuilder:
    """STATEFUL agent: on its first act() ever it sets its first city to build
    Warriors, then does nothing. If one instance is reused across seeds, seed 2
    sees `done=True` and skips the build — a state leak across branches."""

    def __init__(self):
        self.done = False

    def act(self, obs):
        if self.done or not obs.my_cities:
            return []
        self.done = True
        return [BuildUnit(obs.my_cities[0]["id"], "Warriors")]


def test_agent_state_isolation(save, root):
    """Multi-seed runs isolate agent state automatically: a fresh(...) factory
    rebuilds per seed, and even a bare stateful instance is deep-copied per
    seed, so a same-seed-twice run is byte-identical either way and the two
    paths agree."""
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    until = save.turn + 4

    # (a) fresh factory: both seed-1 runs get a new agent -> identical.
    iso = run_agents(
        save,
        root / "fresh_iso",
        agents={focal: fresh(_OnceBuilder)},
        until=until,
        seeds=[1, 1],
    )
    assert iso[0].save_sha256_normalized == iso[1].save_sha256_normalized, (
        "fresh() factory did not isolate agent state across same-seed runs"
    )

    # (b) a bare stateful instance is auto-isolated (deep-copied per seed),
    # so both runs still build -> identical, matching the fresh() result, and
    # the original instance is left untouched (it was copied, not mutated).
    shared = _OnceBuilder()
    auto = run_agents(
        save,
        root / "auto_iso",
        agents={focal: shared},
        until=until,
        seeds=[1, 1],
    )
    assert auto[0].save_sha256_normalized == auto[1].save_sha256_normalized, (
        "a bare stateful instance leaked across seeds — auto per-seed isolation "
        "(deepcopy) did not take effect"
    )
    assert auto[0].save_sha256_normalized == iso[0].save_sha256_normalized, (
        "bare-instance auto-isolation diverged from the fresh() factory result"
    )
    assert shared.done is False, "the original shared instance was mutated (not copied)"


class _Stateful:
    """A bare stateful agent (no reset, cheaply deep-copyable)."""

    def __init__(self):
        self.n = 0

    def act(self, obs):
        self.n += 1
        return []


class _WithReset(_Stateful):
    """A stateful agent that cooperates via reset() (so it is reused + cleared,
    not deep-copied)."""

    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.n = 0


def test_seat_agent_isolation():
    """Unit-level proof of _seat_agent's per-seed isolation policy, with no
    server needed."""
    # fresh() factory -> a brand-new instance every seed.
    built = []
    fac = fresh(lambda: built.append(1) or _Stateful())
    a1, a2 = _seat_agent(fac, isolate=True), _seat_agent(fac, isolate=True)
    assert a1 is not a2 and len(built) == 2

    # bare stateful, multi-seed -> deep-copied to independent instances.
    bare = _Stateful()
    bare.n = 7
    c1, c2 = _seat_agent(bare, isolate=True), _seat_agent(bare, isolate=True)
    assert c1 is not c2 and c1 is not bare
    c1.act(None)
    assert c2.n == 7 and bare.n == 7  # mutating one copy touches no other

    # bare stateful, single-seed -> reused as-is (no sibling branch to leak into).
    assert _seat_agent(bare, isolate=False) is bare

    # reset()-defining agent -> reused and cleared, not copied (keeps heavy state).
    wr = _WithReset()
    wr.n = 5
    got = _seat_agent(wr, isolate=True)
    assert got is wr and wr.reset_calls == 1 and wr.n == 0

    # un-deep-copyable stateful agent in a multi-seed run -> refused loudly.
    class _Uncopyable:
        def __init__(self):
            self._lock = threading.Lock()  # locks cannot be deep-copied

        def act(self, obs):
            return []

    with pytest.raises(TypeError):
        _seat_agent(_Uncopyable(), isolate=True)


def test_legal_actions_is_not_a_client_handle():
    """obs.legal_actions must be a closure, not the client's bound method — a
    bound method would leak the raw FreecivClient via `.__self__` and let an
    agent drive it around the EnvSpec."""

    class _Sentinel:
        secret = "raw client internals"

        def get_unit_actions(self, *a, **k):
            return {"ok": (1, 1)}

    cli = _Sentinel()
    la = _bind_legal_actions(cli)
    assert la(5, target_tile=10) == {"ok": (1, 1)}  # still works as a query
    assert getattr(la, "__self__", None) is not cli  # not a bound method of cli
    assert not hasattr(la, "secret")  # no attribute path to the client


class _FakeClient:
    """Yields a scripted sequence of next_phase() results; a None means the
    seat stopped (over or dropped, per the flags set at construction)."""

    def __init__(self, seq, *, over_at_end=False, dropped_at_end=False):
        self._seq = list(seq)
        self._i = 0
        self.over = False
        self._over_at_end = over_at_end
        self._dropped_at_end = dropped_at_end
        self.dropped = False

    def next_phase(self, max_seconds):
        v = self._seq[self._i] if self._i < len(self._seq) else None
        self._i += 1
        if v is None:
            self.over = self._over_at_end
            self.dropped = self._dropped_at_end
        return v

    def phase_done(self):
        pass


def test_control_lost_raises(save):
    """A seat that drops mid-game (not game-over) must raise ControlLost, and a
    dropped socket is always classified INFRA."""
    ft = save.turn
    agent = FunctionAgent(lambda obs: [])  # issue nothing
    cli = _FakeClient([ft], dropped_at_end=True)  # one phase, then a drop
    with pytest.raises(ControlLost) as info:
        _drive(
            [(cli, agent, "Oscar II")],
            save.path.parent,
            save.path,
            ft,
            30.0,
            EnvSpec.full(),
        )
    assert str(info.value).startswith(f"{ELIM_INFRA}: "), str(info.value)
    assert info.value.kind == ELIM_INFRA


def test_control_over_is_clean(save):
    """A seat that ends because the GAME IS OVER must NOT raise."""
    ft = save.turn
    agent = FunctionAgent(lambda obs: [])
    cli = _FakeClient([ft], over_at_end=True)  # one phase, then clean game-over
    _drive(
        [(cli, agent, "Oscar II")],
        save.path.parent,
        save.path,
        ft,
        30.0,
        EnvSpec.full(),
    )


def test_control_lost_on_timeout(save):
    """Reaching the drive deadline while a seat is still LIVE (not game-over)
    must raise ControlLost — otherwise run_agents would return a trajectory the
    autotoggle AI finishes after the clients close. A zero budget makes the
    deadline already-spent, so the loop exits with the seat not over. The
    class comes from the save's player table."""
    ft = save.turn
    agent = FunctionAgent(lambda obs: [])
    cli = _FakeClient([ft, ft, ft])  # still live: never over, never dropped
    with pytest.raises(ControlLost) as info:
        _drive(
            [(cli, agent, "Oscar II")],
            save.path.parent,
            save.path,
            ft,
            0.0,  # zero budget -> deadline already reached, seat still live
            EnvSpec.full(),
        )
    assert info.value.kind in (ELIM_GAME, ELIM_INFRA, ELIM_UNKNOWN)
    assert str(info.value).startswith(f"{info.value.kind}: "), str(info.value)
