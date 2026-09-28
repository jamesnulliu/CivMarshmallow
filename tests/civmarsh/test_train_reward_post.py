"""Reward post-processing on fan-out decision samples (no slime needed).

Pins episode-level whitening (not per decision), the step-batch advantage
normalization, and both advantage-space hand-offs of the hybrid rewards.
"""

import math
from types import SimpleNamespace

import pytest

import civmarsh.train.reward_post as rp
from civmarsh.civtelescope import client
from civmarsh.rewards import hybrid
from civmarsh.rewards.gae import decision_gae
from civmarsh.rewards.hybrid import sigmoid, sparse_weight

C1, S1, C2, S2 = 35.0, 5.0, 80.0, 5.0


class _Sample:
    """Stand-in for slime's Sample with the fields post_process reads."""

    def __init__(
        self, episode_id, turn, advantage=None, reward=None, index=0, position="p0"
    ):
        self.index = index
        self.reward = reward
        self.metadata = {"episode_id": episode_id, "position_id": position}
        self.train_metadata = {"turn": turn, "advantage": advantage, "return": None}


def _episode(episode_id, reward, n, values=None, position="p0"):
    """One episode's decision samples; with ``values``, GAE as the rollout
    attaches it (gamma 0.9, lam 0.8)."""
    rewards = [0.0] * (n - 1) + [reward]
    dones = [0] * (n - 1) + [1]
    advs = rets = [None] * n
    if values is not None:
        advs, rets = decision_gae(rewards, values, dones, gamma=0.9, lam=0.8)
    samples = []
    for i in range(n):
        s = _Sample(episode_id, i + 1, advs[i], reward, position=position)
        s.train_metadata["return"] = rets[i]
        samples.append(s)
    return samples


def test_whitens_per_episode_not_per_decision():
    ep_a = _episode("ep-a", 10.0, n=3)
    ep_b = _episode("ep-b", 20.0, n=2)
    args = SimpleNamespace(grpo_std_normalization=False)
    raw, rewards = rp.post_process(args, ep_a + ep_b)
    assert raw == [10.0] * 3 + [20.0] * 2
    # the episode mean is (10 + 20) / 2 = 15, NOT length-weighted
    assert rewards == [-5.0] * 3 + [5.0] * 2


def test_whitening_is_per_start_position():
    samples = (
        _episode("x1", 10.0, 2, position="pos0")
        + _episode("x2", 30.0, 2, position="pos0")
        + _episode("y1", 100.0, 1, position="pos1")
    )
    _raw, rewards = rp.post_process(
        SimpleNamespace(grpo_std_normalization=True), samples
    )
    assert rewards[:2] == pytest.approx([-1.0, -1.0], abs=1e-6)
    assert rewards[2:4] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert rewards[4] == 0.0  # a single-episode group has no variance


def test_adv_norm_centres_then_scales_by_batch_std():
    samples = _episode("ep-a", 12.0, 3, values=[1.0, 2.0, 3.0]) + _episode(
        "ep-b", 12.0, 2, values=[5.0, 6.0]
    )
    raw_adv = [s.train_metadata["advantage"] for s in samples]
    raw_ret = [s.train_metadata["return"] for s in samples]
    mean = sum(raw_adv) / len(raw_adv)
    std = (sum((a - mean) ** 2 for a in raw_adv) / len(raw_adv)) ** 0.5

    rp.post_process(
        SimpleNamespace(grpo_std_normalization=False), samples
    )  # on by default

    got = [s.train_metadata["advantage"] for s in samples]
    assert all(
        math.isclose(g, (r - mean) / (std + 1e-6), rel_tol=1e-12)
        for g, r in zip(got, raw_adv)
    )
    assert math.isclose(sum(got), 0.0, abs_tol=1e-9)
    assert [s.train_metadata["return"] for s in samples] == raw_ret  # untouched


def test_adv_norm_off_leaves_advantages_alone():
    ep = _episode("ep-a", 12.0, 3, values=[1.0, 2.0, 3.0])
    raw_adv = [s.train_metadata["advantage"] for s in ep]
    rp.post_process(
        SimpleNamespace(grpo_std_normalization=False, civ_adv_norm=False), ep
    )
    assert [s.train_metadata["advantage"] for s in ep] == raw_adv


def test_adv_norm_tolerates_sparse_samples():
    ep = _episode("ep-a", 12.0, 3)  # sparse: no value fn, advantage is None
    rp.post_process(
        SimpleNamespace(grpo_std_normalization=False, civ_adv_norm=True), ep
    )
    assert all(s.train_metadata["advantage"] is None for s in ep)


def test_sparse_hybrid_blends_group_advantage_with_telescope_gae():
    ep_a = _episode("ep-a", 10.0, 2)
    ep_b = _episode("ep-b", 20.0, 2)
    for samples, advs in ((ep_a, [100.0, 200.0]), (ep_b, [300.0, 400.0])):
        for s, a in zip(samples, advs, strict=True):
            s.train_metadata["advantage"] = a

    args = SimpleNamespace(
        grpo_std_normalization=False,
        civ_adv_norm=False,
        civ_sparse_hybrid_sigmoid=True,
        civ_hybrid_sigmoid_c=C1,
        civ_hybrid_sigmoid_s=S1,
    )
    rp.post_process(args, ep_a + ep_b)

    for samples, sparse_adv, advs in (
        (ep_a, -5.0, [100.0, 200.0]),
        (ep_b, 5.0, [300.0, 400.0]),
    ):
        for s, tel_adv in zip(samples, advs, strict=True):
            w = 1.0 / (1.0 + math.exp(-(s.train_metadata["turn"] - C1) / S1))
            assert math.isclose(
                s.train_metadata["advantage"], (1 - w) * sparse_adv + w * tel_adv
            )
            assert s.train_metadata["sparse_group_advantage"] == sparse_adv
            assert s.train_metadata["telescope_advantage"] == tel_adv


def test_sparse_hybrid_telescope_first_flips_the_weight():
    samples = [_Sample("e0", 20, 1.0)]
    args = SimpleNamespace(
        civ_hybrid_sigmoid_c=C1,
        civ_hybrid_sigmoid_s=S1,
        civ_sparse_hybrid_telescope_first=True,
    )
    rp._mix_sparse_hybrid_advantages(args, samples, {"e0": -2.0})
    w = 1.0 - sigmoid(20, C1, S1)
    assert samples[0].train_metadata["advantage"] == pytest.approx(
        (1 - w) * -2.0 + w * 1.0
    )
    assert "hybrid3_w2" not in samples[0].train_metadata


def test_second_handoff_blends_seam_and_sparse():
    samples = [_Sample("e0", 20, 1.0), _Sample("e0", 80, 1.0), _Sample("e0", 118, 1.0)]
    args = SimpleNamespace(
        civ_hybrid_sigmoid_c=C1,
        civ_hybrid_sigmoid_s=S1,
        civ_hybrid3_c2=C2,
        civ_hybrid3_s2=S2,
    )
    rp._mix_hybrid3_advantages(args, samples, {"e0": -2.0})
    for s in samples:
        w2 = sparse_weight(s.train_metadata["turn"], C2, S2)
        assert s.train_metadata["advantage"] == pytest.approx(
            (1 - w2) * 1.0 + w2 * -2.0
        )
        assert s.train_metadata["seam_advantage"] == 1.0
        assert s.train_metadata["sparse_group_advantage"] == -2.0
        lam = (
            s.train_metadata["hybrid3_lam_sb"],
            s.train_metadata["hybrid3_lam_telescope"],
            s.train_metadata["hybrid3_lam_sparse"],
        )
        assert sum(lam) == pytest.approx(1.0)
    assert samples[0].train_metadata["advantage"] == pytest.approx(1.0, abs=1e-4)
    assert samples[2].train_metadata["advantage"] == pytest.approx(-2.0, abs=1e-2)


def test_second_handoff_requires_integer_turn_and_advantage():
    args = SimpleNamespace(civ_hybrid3_c2=C2, civ_hybrid3_s2=S2)
    bad = _Sample("e0", 20, 1.0)
    bad.train_metadata["turn"] = "20"
    with pytest.raises(RuntimeError):
        rp._mix_hybrid3_advantages(args, [bad], {"e0": 0.0})
    missing = _Sample("e0", 20, None)
    with pytest.raises(RuntimeError):
        rp._mix_hybrid3_advantages(args, [missing], {"e0": 0.0})


def test_the_two_hybrid_blends_are_mutually_exclusive():
    sample = _Sample("e0", 20, 1.0, reward=3.0)
    args = SimpleNamespace(
        civ_sparse_hybrid_sigmoid=True,
        civ_hybrid3_sigmoid=True,
        grpo_std_normalization=False,
    )
    with pytest.raises(RuntimeError):
        rp.post_process(args, [sample])


def test_seam_gae_then_sparse_handoff_end_to_end(monkeypatch, tmp_path):
    """Value hook -> decision_gae -> reward_post, against hand-computed targets."""
    import json

    from civmarsh.rewards import potential

    path = tmp_path / "refset.jsonl"
    with open(path, "w") as f:
        f.writelines(
            json.dumps(
                {
                    "turn": 40,
                    "focal": f"ref{i}",
                    "rendering": f'{{"score":{score * 2},}}',
                    "end_score": score,
                }
            )
            + "\n"
            for i, score in enumerate([0, 10, 20, 30])
        )
    potential._load_at.cache_clear()
    monkeypatch.setattr(client, "prompt_preference", lambda url, p: 0.5)
    turns = [10, 50, 90, 120]
    decisions = [
        {"turn": t, "focal": "me", "value_input": '{"score":15,}'} for t in turns
    ]
    args = SimpleNamespace(
        civ_value_url="http://mock/generate",
        civ_value_refset=str(path),
        civ_hybrid_sigmoid_c=C1,
        civ_hybrid_sigmoid_s=S1,
        civ_hybrid3_c2=C2,
        civ_hybrid3_s2=S2,
        civ_hybrid3_sigmoid=True,
    )
    values = hybrid.sparse_hybrid3_value(decisions, {"score_end": 25.0}, args)
    seam_adv, _ = decision_gae(
        [0.0, 0.0, 0.0, 25.0], values, [0, 0, 0, 1], gamma=1.0, lam=0.9
    )

    samples = [
        _Sample("e0", t, a, index=i) for i, (t, a) in enumerate(zip(turns, seam_adv))
    ]
    rp._mix_hybrid3_advantages(args, samples, {"e0": 4.0})
    for turn, seam, s in zip(turns, seam_adv, samples):
        w2 = sparse_weight(turn, C2, S2)
        assert s.train_metadata["advantage"] == pytest.approx(
            (1 - w2) * seam + w2 * 4.0
        )
    # early decisions keep seam credit, the last is essentially group-outcome credit
    assert abs(samples[0].train_metadata["advantage"] - seam_adv[0]) < 1e-3
    assert abs(samples[-1].train_metadata["advantage"] - 4.0) < 1e-2


def test_post_process_triggers_the_refresh_hook(monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "maybe_refresh", lambda args: calls.append(args))
    args = SimpleNamespace(grpo_std_normalization=False)
    rp.post_process(args, _episode("e", 1.0, 1))
    assert calls == [args]
