"""Scoreboard potential (Phi = current score) and its reference-set helpers."""

import json
from types import SimpleNamespace

import pytest

from civmarsh.rewards.scoreboard import (
    score_now_value,
    score_rank,
    scoreboard_component,
)


def _decisions(scores):
    return [
        {
            "turn": 10 + i,
            "focal": "Alice",
            "value_input": "x"
            if s is None
            else f'DIGEST {{"score":{s},"turn":{10 + i}}}',
        }
        for i, s in enumerate(scores)
    ]


def test_scoreboard_value_reads_score_now():
    vals = score_now_value(
        _decisions([5, 8, 12]), {"score_end": 20.0}, SimpleNamespace()
    )
    assert vals == [5.0, 8.0, 12.0]


def test_scoreboard_value_missing_score_falls_back_and_logs(tmp_path):
    log = tmp_path / "vlog.jsonl"
    vals = score_now_value(
        _decisions([5, None, 12]),
        {"score_end": 20.0},
        SimpleNamespace(civ_value_log=str(log)),
    )
    assert vals == [5.0, 5.0, 12.0]  # carry the last seen score
    row = json.loads(log.read_text())
    assert row["value_fails"] == 1 and row["values"] == [5.0, 5.0, 12.0]
    assert row["score_end"] == 20.0 and row["turns"] == [10, 11, 12]


def test_scoreboard_harvest_every_tenth_turn_and_last(tmp_path):
    harvest = tmp_path / "harvest.jsonl"
    decisions = [
        {"turn": t, "focal": "A", "value_input": '{"score":1,}'}
        for t in (9, 10, 19, 20, 23)
    ]
    score_now_value(
        decisions, {"score_end": 3.0}, SimpleNamespace(civ_value_harvest=str(harvest))
    )
    row = json.loads(harvest.read_text())
    assert [d["turn"] for d in row["decisions"]] == [10, 20, 23]


def test_score_rank_interpolates_ties_and_clamps():
    basis = [2.0, 4.0, 4.0, 10.0]
    assert score_rank(1.0, basis) == 0.0
    assert score_rank(12.0, basis) == 1.0
    assert score_rank(4.0, basis) == pytest.approx(1.5 / 3)  # average rank of the tie
    assert score_rank(7.0, basis) == pytest.approx((2 + 0.5) / 3)
    assert score_rank(5.0, [5.0]) == 0.5


def test_scoreboard_component_ranks_current_scores_on_final_scores():
    # current scores 0/2/4/6 against final scores 0/10/20/30
    refs = [
        {"rendering": f'{{"score":{2 * i},}}', "end_score": 10 * i} for i in range(4)
    ]
    rank, value = scoreboard_component(3.0, refs)
    assert rank == pytest.approx(1.5 / 3)
    assert value == pytest.approx(15.0)
    # references without a parseable score drop out; < 2 left -> rank 0.5
    rank, value = scoreboard_component(
        3.0, [{"rendering": "x", "end_score": 0}, refs[1]]
    )
    assert rank == 0.5
