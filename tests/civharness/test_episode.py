"""An agent-driven run is replayable from its episode log, even when the
agent is nondeterministic.

The engine is deterministic given the orders; the episode log captures the exact
orders an agent issued each turn, so replaying them reproduces the trajectory
byte-for-byte without needing the (possibly stochastic) agent again.

The round-trip and validation tests need no server; the replay test needs the
pinned binary.
"""

import json
import random
from pathlib import Path

import pytest

from civharness import (
    GameConfig,
    ReplayMismatch,
    position,
    read_episode,
    replay_episode,
    reseed_save,
    run_agents,
    snapshot_series,
)
from civharness.episode import (
    EpisodeWriter,
    episode_policies,
    order_as_record,
    order_from_record,
)
from civharness.policy.client import BuildUnit, SetWorklist, UnitOrders


def test_order_record_round_trip():
    for o in [
        BuildUnit(120, "Warriors"),
        SetWorklist(120, (("unit", "Warriors"), ("building", "Barracks"))),
        UnitOrders(
            5,
            100,
            (
                {
                    "order": 1,
                    "activity": 0,
                    "target": -1,
                    "sub_target": -1,
                    "action": 255,
                    "dir": 2,
                },
            ),
            101,
        ),
    ]:
        assert order_from_record(order_as_record(o)) == o, o


def test_episode_policies_groups_by_player_and_turn():
    steps = [
        {
            "turn": 10,
            "player": "A",
            "orders": [order_as_record(BuildUnit(1, "Warriors"))],
        },
        {
            "turn": 11,
            "player": "A",
            "orders": [order_as_record(BuildUnit(1, "Workers"))],
        },
        {
            "turn": 10,
            "player": "B",
            "orders": [order_as_record(BuildUnit(2, "Explorer"))],
        },
    ]
    pols = episode_policies(steps)
    assert set(pols) == {"A", "B"}
    assert pols["A"].orders[10] == [BuildUnit(1, "Warriors")]
    assert pols["A"].orders[11] == [BuildUnit(1, "Workers")]
    assert pols["B"].orders[10] == [BuildUnit(2, "Explorer")]


def test_episode_header_and_obs_sha(root):
    """The log is self-contained: the header carries the AI
    config (skill + skill_by_player) a replay must restore, and each step records
    a normalized obs digest. A missing save hashes to None without crashing."""
    log = root / "hdr.jsonl"
    w = EpisodeWriter(
        log,
        {
            "seed": 1,
            "until": 5,
            "skill": "hard",
            "skill_by_player": {"A": "hard", "B": "easy"},
        },
    )
    w.step(3, "A", "/no/such/save.sav", [BuildUnit(1, "Warriors")])
    w.close()
    header, steps = read_episode(log)
    assert header["skill"] == "hard"
    assert header["skill_by_player"] == {"A": "hard", "B": "easy"}
    assert "obs_sha" in steps[0] and steps[0]["obs_sha"] is None  # bad path -> None
    assert steps[0]["orders"][0]["type"] == "BuildUnit"


def test_incomplete_log_rejected(root):
    """An episode with no completion footer (a truncated or crashed run) is
    refused for exact replay — and before any server launch."""
    log = root / "incomplete.jsonl"
    w = EpisodeWriter(log, {"seed": 1, "until": 5, "from_save_sha": "abc"})
    w.step(3, "A", "/no/such.sav", [BuildUnit(1, "Warriors")])
    w.close()  # NB: no footer() -> the log is incomplete
    raised = False
    try:
        replay_episode(str(log), "/nonexistent/save.sav", root / "x")
    except ReplayMismatch:
        raised = True
    assert raised, "an incomplete (footer-less) log was accepted for exact replay"


def test_malformed_verification_records_rejected(root):
    """Exact replay validates the footer SCHEMA and requires non-null obs digests
    before launching — a footer that isn't complete or lacks a final digest, and
    a step with a null observation hash, must fail closed rather than skip the
    check on absent metadata."""
    # (a) a footer that exists but is not complete and has no final digest.
    log1 = root / "badfooter.jsonl"
    w = EpisodeWriter(log1, {"seed": 1, "until": 5})
    w.step(3, "A", "/no/such.sav", [BuildUnit(1, "Warriors")])  # obs_sha -> None
    w.footer(complete=False)  # malformed: not complete, no final_save_sha
    w.close()
    rejected_a = False
    try:
        replay_episode(str(log1), "/nonexistent.sav", root / "r1")
    except ReplayMismatch:
        rejected_a = True
    assert rejected_a, "a malformed footer (not complete / no final hash) was accepted"

    # (b) a well-formed footer, but a step whose observation hash is null.
    log2 = root / "nullhash.jsonl"
    w = EpisodeWriter(log2, {"seed": 1, "until": 5})
    w.step(3, "A", "/no/such.sav", [BuildUnit(1, "Warriors")])  # obs_sha -> None
    w.footer(final_save_sha="a" * 64)  # complete=True default; well-formed digest
    w.close()
    rejected_b = False
    try:
        replay_episode(str(log2), "/nonexistent.sav", root / "r2")
    except ReplayMismatch:
        rejected_b = True
    assert rejected_b, "a step with a null observation digest was accepted"


class _RandomBuilder:
    """Each turn, build a unit chosen by an internal RNG. Two instances seeded
    differently produce different (but each fully deterministic) trajectories —
    a stand-in for a nondeterministic agent whose choices we must capture."""

    UNITS = ("Warriors", "Workers", "Explorer")

    def __init__(self, agent_seed):
        self._rng = random.Random(agent_seed)

    def act(self, obs):
        if not obs.my_cities:
            return []
        return [BuildUnit(obs.my_cities[0]["id"], self._rng.choice(self.UNITS))]


@pytest.mark.server
def test_replay_reproduces_run(root):
    refs = snapshot_series(
        GameConfig(aifill=3, endturn=20, mapseed=9, gameseed=9), root / "seed", every=10
    )
    save = next(r for r in refs if r.turn >= 10)
    focal = next(n for n, p in position(save).players.items() if p["cities"])
    until = save.turn + 5

    # Two agents with different internal RNGs -> different trajectories, same
    # engine seed. This proves the trajectory rides on the agent's choices.
    t1 = run_agents(
        save, root / "run1", agents={focal: _RandomBuilder(1)}, until=until, seeds=[3]
    )
    t2 = run_agents(
        save, root / "run2", agents={focal: _RandomBuilder(2)}, until=until, seeds=[3]
    )
    assert t1[0].save_sha256_normalized != t2[0].save_sha256_normalized, (
        "different agent decisions produced the same trajectory — test can't "
        "show that replay (not engine determinism alone) carries the choices"
    )

    header, steps = read_episode(t1[0].episode_log_path)
    assert header["seed"] == 3 and header["until"] == until and steps, header

    # Replay run1's episode: reissue its captured orders, no agent involved.
    # verify=True also checks the source-save digest and every per-turn obs
    # digest, so a silent divergence would raise here.
    replay = replay_episode(t1[0].episode_log_path, save, root / "replay")
    assert replay[0].save_sha256_normalized == t1[0].save_sha256_normalized, (
        "replaying the episode did not reproduce the recorded trajectory"
    )

    # A DIFFERENT same-turn save is now rejected, not silently replayed: reseed
    # the position (same turn, different RNG -> different digest) and confirm
    # replay refuses it.
    wrong = reseed_save(save.path, root / "wrong.sav", 987654)
    mismatched = False
    try:
        replay_episode(t1[0].episode_log_path, wrong, root / "badreplay")
    except ReplayMismatch:
        mismatched = True
    assert mismatched, (
        "replay accepted a different same-turn save (provenance not verified)"
    )

    # The recording is COMPLETE — a footer carries the final digest.
    hdr, _st = read_episode(t1[0].episode_log_path)
    assert hdr.get("footer") and hdr["footer"].get("final_save_sha"), (
        "recorded episode has no completion footer with the final save digest"
    )
    recs = [
        json.loads(x)
        for x in Path(t1[0].episode_log_path).read_text().splitlines()
        if x.strip()
    ]

    # A truncated recording (footer dropped) is rejected as incomplete.
    trunc = root / "trunc.jsonl"
    trunc.write_text(
        "\n".join(json.dumps(r) for r in recs if r.get("kind") != "footer") + "\n"
    )
    truncated_rejected = False
    try:
        replay_episode(str(trunc), save, root / "truncreplay")
    except ReplayMismatch:
        truncated_rejected = True
    assert truncated_rejected, "a truncated (footer-less) log was accepted as exact"

    # A tampered final digest is caught end-to-end (last-turn divergence proxy).
    def _tamper(r):
        return {**r, "final_save_sha": "0" * 64} if r.get("kind") == "footer" else r

    tampered = root / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(_tamper(r)) for r in recs) + "\n")
    tamper_caught = False
    try:
        replay_episode(str(tampered), save, root / "tampreplay")
    except ReplayMismatch:
        tamper_caught = True
    assert tamper_caught, "a tampered final-save digest was not caught end-to-end"
