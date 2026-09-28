"""Tests for civmarsh.eval.policy_eval: cell validation and S / G / W aggregation."""

import json

import pytest

from civmarsh.eval import policy_eval as pe

N_SAMPLES = 8


def _starts(n=3):
    return [
        {"position_id": f"ep_g{i}", "game_id": f"g{i}", "split": "test", "turn": 1}
        for i in range(n)
    ]


def _episode(pid, score, ok=True, n_err=1, n_decisions=10):
    return {
        "position_id": pid,
        "score_end": score if ok else 0.0,
        "ok": ok,
        "n_err": n_err,
        "n_decisions": n_decisions,
    }


def _write_run(tmp_path, starts, groups, *, extra_args=(), config=None):
    run = tmp_path / "run"
    run.mkdir()
    n = len(starts)
    with open(run / "rl_starts.jsonl", "w") as f:
        f.writelines(
            json.dumps({"prompt": row["position_id"], "metadata": row}) + "\n"
            for row in starts
        )
    with open(run / "reward_log.jsonl", "w") as f:
        f.writelines(
            json.dumps({"ts": float(ts), "episodes": episodes}) + "\n"
            for ts, episodes in enumerate(groups)
        )
    cfg = config or "civ_min_group: 1\nciv_enable_thinking: false\n"
    (run / "config.yaml").write_text(cfg)
    args = [
        "--rollout-batch-size",
        "8",
        "--global-batch-size",
        "64",
        "--n-samples-per-prompt",
        str(N_SAMPLES),
        "--debug-rollout-only",
        "--rollout-num-gpus",
        "1",
        "--rollout-batch-size",
        str(n),
        "--global-batch-size",
        str(n * N_SAMPLES),
        *extra_args,
    ]
    (run / "slime_args.txt").write_text("\n".join(args) + "\n")
    return run


def _groups(starts, scores_by_pid):
    return [
        [
            _episode(row["position_id"], s, ok=s is not None)
            for s in scores_by_pid[row["position_id"]]
        ]
        for row in starts
    ]


def _valid_run(tmp_path):
    starts = _starts()
    scores = {
        "ep_g0": [10, 12, 14, 16, 18, 20, None, None],  # two infrastructure failures
        "ep_g1": [0, 0, 5, 5, 5, 5, 5, 5],  # two eliminations, both scored 0
        "ep_g2": [30] * 8,
    }
    return starts, _write_run(tmp_path, starts, _groups(starts, scores))


def test_validate_run_scores_valid_episodes_only(tmp_path):
    starts, run = _valid_run(tmp_path)
    cell = pe.validate_run(run, starts, require_split="test")
    g0 = cell["per_position"]["ep_g0"]
    assert g0["n_valid"] == 6 and g0["n_infra_failures"] == 2
    assert g0["mean"] == pytest.approx(15.0)
    g1 = cell["per_position"]["ep_g1"]
    assert g1["n_valid"] == 8 and g1["scores"][:2] == [0.0, 0.0]
    assert g1["mean"] == pytest.approx(30 / 8)
    assert cell["n_valid_episodes"] == 22
    assert cell["per_position"]["ep_g2"]["game_id"] == "g2"
    # invalid-action rate: 1 error per 10 decisions in every valid episode
    assert cell["invalid_action_rate"] == pytest.approx(0.1)


def test_validate_run_rejects_protocol_violations(tmp_path):
    starts, run = _valid_run(tmp_path)
    (run / "hf_iter0").mkdir()
    with pytest.raises(pe.CellError, match="saved a checkpoint"):
        pe.validate_run(run, starts)
    (run / "hf_iter0").rmdir()
    (run / "train.log").write_text("... step 0: {'loss': 1.0}\n")
    with pytest.raises(pe.CellError, match="training step"):
        pe.validate_run(run, starts)
    (run / "train.log").unlink()
    (run / "config.yaml").write_text("civ_enable_thinking: false\n")
    with pytest.raises(pe.CellError, match="civ_min_group"):
        pe.validate_run(run, starts)


def test_validate_run_needs_every_start_and_enough_valid(tmp_path):
    starts = _starts()
    scores = {row["position_id"]: [1.0] * 8 for row in starts}
    run = _write_run(tmp_path, starts, _groups(starts, scores)[:2])
    with pytest.raises(pe.CellError, match="reward groups"):
        pe.validate_run(run, starts)


def test_validate_run_valid_floor(tmp_path):
    starts = _starts()
    scores = {row["position_id"]: [1.0] * 4 + [None] * 4 for row in starts}
    run = _write_run(tmp_path, starts, _groups(starts, scores))
    with pytest.raises(pe.CellError, match="valid episodes"):
        pe.validate_run(run, starts)
    assert pe.validate_run(run, starts, min_valid_episodes=12)["n_valid_episodes"] == 12


def test_validate_run_checks_last_launch_value(tmp_path):
    starts, run = _valid_run(tmp_path)
    args = (run / "slime_args.txt").read_text() + "--rollout-batch-size\n8\n"
    (run / "slime_args.txt").write_text(args)
    with pytest.raises(pe.CellError, match="--rollout-batch-size"):
        pe.validate_run(run, starts)


def _cell(means, games=None, scores=None):
    per = {}
    for i, (pid, mean) in enumerate(means.items()):
        s = scores[pid] if scores else [mean]
        per[pid] = {
            "game_id": (games or {}).get(pid, f"g{i}"),
            "scores": s,
            "mean": mean,
            "n_valid": len(s),
            "n_err": 1,
            "n_decisions": 4,
        }
    return {"per_position": per, "position_ids": sorted(per)}


def test_excluded_games_rule():
    base = _cell(
        {"a": 0.0, "b": 2.0, "c": 0.0},
        scores={"a": [0.0, 0.0, 0.0], "b": [0.0, 4.0], "c": [0.0]},
    )
    assert pe.excluded_games(base) == ["g0", "g2"]


def test_terminal_score_and_gains_with_exclusion():
    start = _cell({"a": 1.0, "b": 2.0, "c": 3.0})
    end = _cell({"a": 4.0, "b": 2.0, "c": 13.0})
    assert pe.terminal_score(end) == pytest.approx(19 / 3)
    assert pe.terminal_score(end, ["g2"]) == pytest.approx(3.0)
    assert pe.phase_gains(start, end) == {"a": 3.0, "b": 0.0, "c": 10.0}
    assert pe.phase_gains(start, end, ["g2"]) == {"a": 3.0, "b": 0.0}
    with pytest.raises(pe.CellError):
        pe.phase_gains(start, _cell({"a": 1.0, "b": 1.0}))


def test_paired_win_rate_is_strict_and_pooled():
    a = {"1": {"p": 2.0, "q": 1.0}, "2": {"p": 0.0, "q": 5.0}}
    b = {"1": {"p": 1.0, "q": 1.0}, "2": {"p": 1.0, "q": 4.0}, "3": {"p": 9.0}}
    w = pe.paired_win_rate(a, b)
    assert (w["wins"], w["ties"], w["n"]) == (2, 1, 4)
    assert w["rate"] == pytest.approx(0.5)
    assert w["seeds"] == ["1", "2"]


def test_summarize_means_over_positions_then_seeds():
    base_rem = _cell({"a": 0.0, "b": 0.0})
    cells = {
        "civtelescope": {
            "1": {"rem80": {"start": base_rem, "end": _cell({"a": 4.0, "b": 2.0})}},
            "2": {"rem80": {"start": base_rem, "end": _cell({"a": 1.0, "b": 1.0})}},
        },
        "sparse": {
            "1": {"rem80": {"start": base_rem, "end": _cell({"a": 1.0, "b": 3.0})}},
            "2": {"rem80": {"start": base_rem, "end": _cell({"a": 0.0, "b": 0.0})}},
        },
    }
    out = pe.summarize(cells)
    g = out["rewards"]["civtelescope"]["rem80"]["G"]
    assert g["per_seed"] == {"1": 3.0, "2": 1.0}
    assert g["mean"] == pytest.approx(2.0)
    assert out["rewards"]["sparse"]["rem80"]["S"]["mean"] == pytest.approx(1.0)
    assert out["rewards"]["sparse"]["rem80"]["invalid_action_rate"]["mean"] == 0.25
    w = out["paired_win_rate"]["rem80"]["civtelescope_over_sparse"]
    assert (w["wins"], w["n"]) == (3, 4)
    assert out["n_positions"] == {"rem80": 2}
    out = pe.summarize(cells, exclude_games=["g0"])
    assert out["rewards"]["civtelescope"]["rem80"]["G"]["per_seed"] == {
        "1": 2.0,
        "2": 1.0,
    }
    assert out["excluded_games"] == ["g0"]
