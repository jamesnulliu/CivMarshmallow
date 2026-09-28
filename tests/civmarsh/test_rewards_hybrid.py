"""Hybrid potentials: hand-off weights and the per-episode value hooks
(served model mocked)."""

import json
from types import SimpleNamespace

import pytest

from civmarsh.civtelescope import client
from civmarsh.rewards import hybrid, potential
from civmarsh.rewards.hybrid import (
    layer_weights,
    seam_weight,
    sigmoid,
    validate_hybrid3,
)

C1, S1, C2, S2 = 35.0, 5.0, 80.0, 5.0


# ----------------------------------------------------------------- weights


def test_weights_partition_unity_and_shape():
    for turn in range(141):
        lam_sb, lam_tel, lam_sparse = layer_weights(turn, C1, S1, C2, S2)
        assert abs(lam_sb + lam_tel + lam_sparse - 1.0) < 1e-12
        assert lam_sb >= 0 and lam_tel >= 0 and lam_sparse >= 0
    assert layer_weights(0, C1, S1, C2, S2)[0] > 0.99
    assert layer_weights(120, C1, S1, C2, S2)[2] > 0.95
    telescope = [layer_weights(t, C1, S1, C2, S2)[1] for t in range(141)]
    peak = telescope.index(max(telescope))
    assert C1 < peak < C2
    assert telescope[: peak + 1] == sorted(telescope[: peak + 1])
    assert telescope[peak:] == sorted(telescope[peak:], reverse=True)


def test_defaults():
    assert (hybrid.DEFAULT_C1, hybrid.DEFAULT_S1) == (35.0, 5.0)
    assert (hybrid.DEFAULT_C2, hybrid.DEFAULT_S2) == (80.0, 5.0)
    assert hybrid.DEFAULT_SKIP == 0.01


def test_validate_rejects_inverted_window():
    with pytest.raises(ValueError):
        validate_hybrid3(80.0, 5.0, 35.0, 5.0)
    with pytest.raises(ValueError):
        validate_hybrid3(35.0, 0.0, 80.0, 5.0)


# ----------------------------------------------------------- value hooks


@pytest.fixture()
def refset_file(tmp_path):
    path = tmp_path / "refset.jsonl"
    with open(path, "w") as handle:
        handle.writelines(
            json.dumps(
                {
                    "turn": 40,
                    "focal": f"ref{index}",
                    "rendering": f'{{"score":{score * 2},}}',
                    "end_score": score,
                }
            )
            + "\n"
            for index, score in enumerate([0, 10, 20, 30])
        )
    potential._load_at.cache_clear()
    return str(path)


def _args(refset_file, **kw):
    base = {
        "civ_value_url": "http://mock/generate",
        "civ_value_refset": refset_file,
        "civ_value_concurrency": 2,
        "civ_value_refset_idx": None,
        "civ_value_single_order": True,
        "civ_hybrid_sigmoid_c": C1,
        "civ_hybrid_sigmoid_s": S1,
        "civ_hybrid3_c2": C2,
        "civ_hybrid3_s2": S2,
        "civ_hybrid_sigmoid_skip_threshold": 0.01,
        "civ_hybrid3_sigmoid": True,
        "civ_value_log": None,
        "civ_value_harvest": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _references(refset_file):
    return potential.load_reference_set(refset_file)[40]


def test_scoreboard_to_civtelescope_blends_in_value_space(monkeypatch, refset_file):
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 0.75
    )
    decisions = [
        {"turn": t, "focal": "me", "value_input": '{"score":15,}'} for t in (5, 40, 100)
    ]
    values = hybrid.hybrid_sigmoid_value(
        decisions, {"score_end": 22.0}, _args(refset_file)
    )

    refs = _references(refset_file)
    _rank, sb_value = hybrid.scoreboard_component(15.0, refs)
    phi = potential.winrate_to_score(0.75, refs)
    assert values[0] == pytest.approx(sb_value)  # below the skip threshold
    for index, turn in ((1, 40), (2, 100)):
        w = sigmoid(turn, C1, S1)
        assert values[index] == pytest.approx((1 - w) * sb_value + w * phi)
    assert len(calls) == 2 * 4  # turn 5 skipped


def test_sparse_hybrid_returns_telescope_value_and_skips_early(monkeypatch, tmp_path):
    path = tmp_path / "refset.jsonl"
    with open(path, "w") as f:
        f.writelines(
            json.dumps(
                {
                    "turn": 10,
                    "focal": f"ref{i}",
                    "rendering": f"render{i}",
                    "end_score": score,
                }
            )
            + "\n"
            for i, score in enumerate([0, 10, 20, 30])
        )
    potential._load_at.cache_clear()
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 0.5
    )
    decisions = [
        {"turn": 0, "focal": "me", "value_input": 'x {"score":0,}'},
        {"turn": 10, "focal": "me", "value_input": 'x {"score":1,}'},
        {"turn": 50, "focal": "me", "value_input": 'x {"score":2,}'},
    ]
    args = SimpleNamespace(
        civ_value_url="http://mock/generate",
        civ_value_refset=str(path),
        civ_value_concurrency=4,
        civ_value_refset_idx=[0, 3],
        civ_value_single_order=True,
    )
    values = hybrid.sparse_hybrid_sigmoid_value(decisions, {"score_end": 5}, args)
    assert len(calls) == 1 * 2 * 1  # t=0 and t=10 are below the 0.01 skip threshold
    assert values == [15.0, 15.0, 15.0]

    # CivTelescope first: the weight falls with the turn, so t=70 is skipped
    calls.clear()
    args.civ_sparse_hybrid_telescope_first = True
    decisions[2]["turn"] = 70
    hybrid.sparse_hybrid_sigmoid_value(decisions, {"score_end": 5}, args)
    assert len(calls) == 2 * 2 * 1


def test_seam_value_is_the_scoreboard_telescope_blend(monkeypatch, refset_file):
    monkeypatch.setattr(client, "prompt_preference", lambda url, p: 0.75)
    decisions = [
        {"turn": t, "focal": "me", "value_input": '{"score":15,}'} for t in (5, 40, 100)
    ]
    values = hybrid.sparse_hybrid3_value(
        decisions, {"score_end": 22.0, "position_id": "p0"}, _args(refset_file)
    )

    refs = _references(refset_file)
    _rank, sb_value = hybrid.scoreboard_component(15.0, refs)
    phi = potential.winrate_to_score(0.75, refs)
    assert values[0] == pytest.approx(sb_value)
    for index, turn in ((1, 40), (2, 100)):
        w1 = seam_weight(turn, C1, S1)
        assert values[index] == pytest.approx((1 - w1) * sb_value + w1 * phi)
    assert values[1] != pytest.approx(values[2])


def test_late_telescope_calls_are_not_skipped(monkeypatch, refset_file):
    """The tail seam value still feeds the backward GAE, so it must be real."""
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 0.6
    )
    decisions = [
        {"turn": t, "focal": "me", "value_input": '{"score":15,}'} for t in (5, 60, 118)
    ]
    hybrid.sparse_hybrid3_value(
        decisions, {"score_end": 9.0, "position_id": "p"}, _args(refset_file)
    )
    # 4 references, single order, scored at turns 60 and 118 but not at 5
    assert len(calls) == 8


def test_three_signal_value_refuses_to_run_without_the_second_handoff(refset_file):
    args = _args(refset_file, civ_hybrid3_sigmoid=False)
    with pytest.raises(RuntimeError):
        hybrid.sparse_hybrid3_value(
            [{"turn": 40, "focal": "me", "value_input": '{"score":15,}'}],
            {"score_end": 1.0, "position_id": "p"},
            args,
        )
