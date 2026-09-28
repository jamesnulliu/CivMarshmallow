"""Server/GPU-free tests for the per-decision sample path (needs slime).

Covers: terminal / elimination semantics, siblings sharing one rollout_id so
an episode's loss weight is counted once, loss on action tokens only,
per-decision train_metadata, GAE attachment, the failed-episode filter, and
the group reward (error penalty and terminal CivTelescope value).

slime is an external dependency: these tests are skipped unless ``import
slime`` works (optionally from a checkout named by ``SLIME_DIR``).
"""

import asyncio
import math
import os
import sys
import types

import pytest

_slime_dir = os.environ.get("SLIME_DIR")
if _slime_dir and _slime_dir not in sys.path:
    sys.path.insert(0, _slime_dir)
pytest.importorskip("slime.utils.types")

from slime.utils.types import Sample

from civmarsh.rewards.gae import decision_gae
from civmarsh.train import rollout
from civmarsh.train.reward_post import post_process


def _decision(turn, i):
    return {
        "turn": turn,
        "prompt_text": f"prompt-{i}",
        "prompt_ids": [100 + i, 101 + i],
        "prompt_sha": f"sha-{i}",
        "response_ids": [7, 8, 9],
        "response_lps": [-0.1, -0.2, -0.3],
        "response_text": "{}",
        "candidate_sha": f"cand-{i}",
        "keys": [f"k{i}"],
        "error": None,
    }


def _assemble(result=None, value_fn=None, n=3, err_penalty=0.0, terminal_value_fn=None):
    decisions = [_decision(t, i) for i, t in enumerate(range(1, n + 1))]
    result = result or {"score_end": 12.0, "end_turn": 71, "eliminated": False}
    return rollout.assemble_decision_samples(
        decisions,
        {"position_id": "p0", "focal_player": "Alice"},
        result,
        episode_id="ep-1",
        source_index=42,
        source_group_index=0,
        over_cap_turns=[],
        gamma=0.9,
        lam=0.8,
        value_fn=value_fn,
        value_fn_args=None,
        err_penalty=err_penalty,
        terminal_value_fn=terminal_value_fn,
    )


# ------------------------------------------------------------------ assembly


def test_siblings_share_rollout_id_and_terminal_semantics():
    samples = _assemble()
    assert [s.rollout_id for s in samples] == [42, 42, 42]
    tm = [s.train_metadata for s in samples]
    assert [m["done"] for m in tm] == [0, 0, 1]
    assert [m["reward"] for m in tm] == [0.0, 0.0, 12.0]
    assert [m["decision_index"] for m in tm] == [0, 1, 2]
    assert tm[1]["prompt_sha"] == "sha-1" and tm[1]["candidate_sha"] == "cand-1"


def test_elimination_is_terminal_and_priced_zero():
    samples = _assemble(
        result={
            "score_end": 0,
            "end_turn": None,
            "eliminated": True,
            "control_lost": "GAME: out",
        }
    )
    tm = samples[-1].train_metadata
    assert tm["done"] == 1 and tm["reward"] == 0.0
    assert samples[-1].metadata["eliminated"] is True
    assert samples[-1].metadata["control_lost"] == "GAME: out"


def test_loss_mask_covers_action_tokens_only():
    for s in _assemble():
        assert s.loss_mask == [1, 1, 1]
        assert s.response_length == 3
        assert len(s.tokens) == 2 + 3
        assert s.rollout_log_probs == [-0.1, -0.2, -0.3]
        assert s.prompt.startswith("prompt-")


def test_gae_attached_when_value_fn_given():
    samples = _assemble(value_fn=lambda ds, res, a: [1.0, 2.0, 3.0])
    adv = [s.train_metadata["advantage"] for s in samples]
    exp, _ = decision_gae(
        [0.0, 0.0, 12.0], [1.0, 2.0, 3.0], [0, 0, 1], gamma=0.9, lam=0.8
    )
    assert all(math.isclose(a, e, abs_tol=1e-9) for a, e in zip(adv, exp))


def test_err_penalty_enters_the_dense_rewards():
    decisions = [_decision(t, i) for i, t in enumerate(range(1, 4))]
    decisions[0]["error"] = "invalid JSON: Extra data"
    samples = rollout.assemble_decision_samples(
        decisions,
        {"position_id": "p0"},
        {"score_end": 12.0, "end_turn": 71, "eliminated": False},
        episode_id="ep-1",
        source_index=0,
        source_group_index=0,
        over_cap_turns=[],
        gamma=1.0,
        lam=0.9,
        err_penalty=1.0,
    )
    assert [s.train_metadata["reward"] for s in samples] == [-1.0, 0.0, 12.0]
    # the invalid decision stays in the loss
    assert samples[0].loss_mask == [1, 1, 1]
    assert samples[0].metadata["parse_error"].startswith("invalid JSON")


def test_sparse_and_dense_share_the_same_prompt_path():
    sparse = _assemble()
    dense = _assemble(value_fn=lambda ds, res, a: [1.0, 2.0, 3.0])
    assert [s.prompt for s in sparse] == [s.prompt for s in dense]
    assert [s.tokens for s in sparse] == [s.tokens for s in dense]
    assert [s.loss_mask for s in sparse] == [s.loss_mask for s in dense]


def test_terminal_value_is_attached_not_trained_on():
    samples = _assemble(terminal_value_fn=lambda ds, res, a: 33.0)
    assert all(s.metadata["phi_terminal"] == 33.0 for s in samples)
    assert [s.train_metadata["reward"] for s in samples] == [0.0, 0.0, 12.0]
    assert all(s.train_metadata["advantage"] is None for s in samples)
    with pytest.raises(AssertionError):
        _assemble(
            value_fn=lambda ds, res, a: [1.0] * 3,
            terminal_value_fn=lambda ds, r, a: 1.0,
        )


def test_reward_post_whitens_assembled_episodes_once_each():
    ep_a = _assemble(
        result={"score_end": 10.0, "end_turn": 71, "eliminated": False}, n=3
    )
    ep_b = _assemble(
        result={"score_end": 20.0, "end_turn": 71, "eliminated": False}, n=2
    )
    for s in ep_a:
        s.metadata["episode_id"] = "ep-a"
        s.reward = 10.0
    for s in ep_b:
        s.metadata["episode_id"] = "ep-b"
        s.reward = 20.0
    raw, rewards = post_process(
        types.SimpleNamespace(grpo_std_normalization=False), ep_a + ep_b
    )
    assert raw == [10.0] * 3 + [20.0] * 2
    assert rewards == [-5.0] * 3 + [5.0] * 2


# ------------------------------------------------------ slime data contract


def test_sibling_samples_do_not_multiply_episode_weight():
    ray_rollout = pytest.importorskip("slime.ray.rollout")
    cls = getattr(
        ray_rollout.RolloutManager, "__ray_actor_class__", ray_rollout.RolloutManager
    )
    samples = _assemble()
    for s in samples:
        s.reward = 12.0
    mgr = object.__new__(cls)  # the converter needs args only
    mgr.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        eval_reward_key=None,
    )
    mgr.custom_convert_samples_to_train_data_func = None
    mgr.custom_reward_post_process_func = None
    train_data = cls._convert_samples_to_train_data(mgr, samples)
    # the reducer divides each sample's mask by the EPISODE total, so an
    # episode counts once however many decisions it has
    assert train_data["rollout_ids"] == [42, 42, 42]
    per_sample_mask = [sum(m) for m in train_data["loss_masks"]]
    assert train_data["rollout_mask_sums"] == [sum(per_sample_mask)] * 3
    assert train_data["metadata"] == [s.train_metadata for s in samples]


def test_dp_partition_carries_metadata():
    import inspect

    ray_rollout = pytest.importorskip("slime.ray.rollout")
    cls = getattr(
        ray_rollout.RolloutManager, "__ray_actor_class__", ray_rollout.RolloutManager
    )
    src = inspect.getsource(cls._split_train_data_by_dp)
    assert '"metadata"' in src, (
        "metadata missing from the per-rank keys of _split_train_data_by_dp; "
        "apply scripts/slime/slime.patch"
    )


# ------------------------------------------------------ failed-episode filter


@pytest.fixture
def civ_filter():
    """The filter with its process-global window and counters reset."""
    rollout._DROP_WINDOW_ROWS.clear()
    rollout._DROP_STATS.update(total=0, dropped=0)
    yield rollout.drop_failed_episode_groups
    rollout._DROP_WINDOW_ROWS.clear()
    rollout._DROP_STATS.update(total=0, dropped=0)


def _stalled_episode():
    s = Sample()
    s.status = Sample.Status.ABORTED
    s.metadata = {"infra_stall": True, "elim_class": "INFRA", "n_decisions": 30}
    return [s]


def _aborted_episode():
    s = Sample()
    s.status = Sample.Status.ABORTED
    s.metadata = {"episode_error": "no result or no decisions"}
    return [s]


def _group(n_ok, stalled=0, aborted=0):
    return (
        [_assemble() for _ in range(n_ok)]
        + [_stalled_episode() for _ in range(stalled)]
        + [_aborted_episode() for _ in range(aborted)]
    )


def test_filter_drops_group_with_aborted_episode(civ_filter):
    dead = Sample()
    dead.status = Sample.Status.ABORTED  # bare sample, no tokens/log_probs
    out = civ_filter(None, [_assemble(), dead])
    assert out.keep is False and out.reason == "aborted_episode"
    assert civ_filter(None, [_assemble(), _assemble()]).keep is True


def test_min_group_unset_is_all_or_nothing(civ_filter):
    group = _group(7, stalled=1)
    out = civ_filter(types.SimpleNamespace(), group)
    assert out.keep is False and out.reason == "aborted_episode"
    assert len(group) == 8


def test_min_group_set_shrinks_group_in_place(civ_filter, tmp_path):
    log = tmp_path / "drop_log.jsonl"
    args = types.SimpleNamespace(civ_min_group=4, civ_drop_log=str(log))
    group = _group(5, stalled=2, aborted=1)
    assert civ_filter(args, group).keep is True
    assert len(group) == 5
    assert all(s.status == Sample.Status.COMPLETED for ep in group for s in ep)
    assert len(log.read_text().splitlines()) == 3


def test_min_group_set_still_drops_below_the_floor(civ_filter):
    args = types.SimpleNamespace(civ_min_group=4)
    group = _group(3, stalled=5)
    out = civ_filter(args, group)
    assert out.keep is False and len(group) == 8


def test_stall_ceiling_counts_episodes_not_groups(civ_filter):
    """One stall in a group of 8 is a 12.5% per-episode rate, tolerated
    indefinitely under a 30% ceiling; 3 of 8 (37.5%) fires once it dominates
    the window."""
    args = types.SimpleNamespace(civ_stall_max_frac=0.30)
    for _ in range(40):
        civ_filter(args, _group(7, stalled=1))
    with pytest.raises(rollout.StallStorm) as exc:
        for _ in range(20):
            civ_filter(args, _group(5, stalled=3))
    assert "infra_stall" in str(exc.value) and "per-EPISODE" in str(exc.value)


def test_engine_abort_does_not_count_toward_the_stall_ceiling(civ_filter):
    args = types.SimpleNamespace(civ_stall_max_frac=0.30)
    for _ in range(40):
        civ_filter(args, _group(1, aborted=7))  # 87.5% aborts, 0% stalls


def test_window_recovers_after_a_storm_subsides(civ_filter):
    args = types.SimpleNamespace(civ_stall_max_frac=0.30)
    for _ in range(10):
        civ_filter(args, _group(5, stalled=3))  # 37.5%, not yet armed (< 16 groups)
    for _ in range(40):
        civ_filter(args, _group(8))


def test_stall_ceiling_not_armed_before_the_window_fills(civ_filter):
    args = types.SimpleNamespace(civ_stall_max_frac=0.30)
    for _ in range(15):
        civ_filter(args, _group(0, stalled=8))


# -------------------------------------------------------------- group reward


def _reward_episode(score, errors, episode_id, phi_terminal=None):
    eps = _assemble(
        result={"score_end": score, "end_turn": 71, "eliminated": False}, n=len(errors)
    )
    for s, err in zip(eps, errors):
        s.metadata["episode_id"] = episode_id
        s.metadata["parse_error"] = err
        if phi_terminal is not None:
            s.metadata["phi_terminal"] = phi_terminal
    return eps


def _run_group_reward(samples, err_penalty, **kw):
    args = types.SimpleNamespace(civ_err_penalty=err_penalty, civ_reward_log=None, **kw)
    return asyncio.run(rollout.group_reward(args, samples))


def test_group_reward_penalty_off_is_raw_score():
    ep = _reward_episode(30.0, [None, None, None], "ep-a")
    assert _run_group_reward([ep], err_penalty=0.0) == []
    assert all(s.reward == 30.0 for s in ep)


def test_group_reward_penalty_subtracts_per_invalid():
    ep = _reward_episode(30.0, ["cap overflow", None, "bad JSON"], "ep-b")
    _run_group_reward([ep], err_penalty=1.0)
    assert all(s.reward == 28.0 for s in ep)


def test_group_reward_penalty_defaults_to_one():
    ep = _reward_episode(30.0, ["bad JSON", None], "ep-b")
    asyncio.run(rollout.group_reward(types.SimpleNamespace(), [ep]))
    assert all(s.reward == 29.0 for s in ep)


def test_group_reward_penalty_varies_across_siblings_of_a_position():
    clean = _reward_episode(30.0, [None, None, None], "ep-c")
    dirty = _reward_episode(30.0, ["x", "y", None], "ep-d")
    _run_group_reward([clean, dirty], err_penalty=1.0)
    assert clean[0].reward == 30.0 and dirty[0].reward == 28.0


def test_group_reward_terminal_value_replaces_the_score(tmp_path):
    import json

    log = tmp_path / "reward_log.jsonl"
    ep = _reward_episode(30.0, [None, "bad JSON"], "ep-t", phi_terminal=41.5)
    path = "civmarsh.rewards.potential.terminal_telescope_value"
    asyncio.run(
        rollout.group_reward(
            types.SimpleNamespace(
                civ_err_penalty=1.0,
                civ_reward_log=str(log),
                civ_terminal_value_fn_path=path,
            ),
            [ep],
        )
    )
    assert all(s.reward == 40.5 for s in ep)
    row = json.loads(log.read_text())["episodes"][0]
    assert row["score_end"] == 30.0 and row["phi_terminal"] == 41.5 and row["ok"]

    missing = _reward_episode(30.0, [None], "ep-m")
    with pytest.raises(RuntimeError):
        _run_group_reward([missing], err_penalty=0.0, civ_terminal_value_fn_path=path)


def test_group_reward_failed_episode_gets_zero():
    dead = Sample()
    dead.status = Sample.Status.ABORTED
    dead.metadata = {"episode_error": "no result"}
    _run_group_reward([[dead]], err_penalty=1.0)
    assert dead.reward == 0.0
