"""Cross-decision GAE and per-decision rewards (hand-computed fixtures)."""

import math

import pytest

from civmarsh.rewards.gae import decision_gae, decision_rewards


def test_gae_three_step_hand_computed():
    # gamma=0.9, lam=0.8; Phi = [1, 2, 3]; terminal reward 10 at t2
    # delta_2 = 10 - 3.0            = 7.0
    # delta_1 = 0 + 0.9*3.0 - 2.0   = 0.7
    # delta_0 = 0 + 0.9*2.0 - 1.0   = 0.8
    # A_2 = 7.0; A_1 = 0.7 + 0.72*7.0 = 5.74; A_0 = 0.8 + 0.72*5.74 = 4.9328
    adv, ret = decision_gae(
        [0.0, 0.0, 10.0], [1.0, 2.0, 3.0], [0, 0, 1], gamma=0.9, lam=0.8
    )
    assert math.isclose(adv[2], 7.0, abs_tol=1e-9)
    assert math.isclose(adv[1], 5.74, abs_tol=1e-9)
    assert math.isclose(adv[0], 4.9328, abs_tol=1e-9)
    # returns = A + Phi
    assert math.isclose(ret[0], 5.9328, abs_tol=1e-9)
    assert math.isclose(ret[1], 7.74, abs_tol=1e-9)
    assert math.isclose(ret[2], 10.0, abs_tol=1e-9)


def test_gae_terminal_blocks_bootstrap():
    a1, _ = decision_gae(
        [0.0, 5.0], [1.0, 1.0], [0, 1], gamma=1.0, lam=1.0, bootstrap_value=0.0
    )
    a2, _ = decision_gae(
        [0.0, 5.0], [1.0, 1.0], [0, 1], gamma=1.0, lam=1.0, bootstrap_value=999.0
    )
    assert a1 == a2


def test_gae_truncation_bootstraps_tail():
    # done=0 on the last step: delta_1 = 0 + 1.0 * 4.0 - 1.0 = 3.0
    adv, _ = decision_gae(
        [0.0, 0.0], [1.0, 1.0], [0, 0], gamma=1.0, lam=1.0, bootstrap_value=4.0
    )
    assert math.isclose(adv[1], 3.0, abs_tol=1e-9)


def test_gae_rejects_mid_episode_done():
    with pytest.raises(ValueError):
        decision_gae([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1, 0, 1], gamma=0.9, lam=0.8)


def test_gae_rejects_length_mismatch():
    with pytest.raises(ValueError):
        decision_gae([0.0, 1.0], [0.0], [0, 1], gamma=1.0, lam=1.0)


def test_decision_rewards_terminal_only_and_error_penalty():
    decisions = [{"error": None}, {"error": "invalid JSON"}, {"error": None}]
    assert decision_rewards(decisions, 30.0) == [0.0, 0.0, 30.0]
    assert decision_rewards(decisions, 30.0, err_penalty=1.0) == [0.0, -1.0, 30.0]
    # an invalid terminal decision pays the penalty on top of the final score
    decisions[-1]["error"] = "not offered"
    assert decision_rewards(decisions, 30.0, err_penalty=1.0) == [0.0, -1.0, 29.0]
    assert decision_rewards([], 5.0) == []
