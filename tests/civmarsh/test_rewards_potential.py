"""CivTelescope potential against a reference set (served model mocked).

Pins the call fan-out, reference subsetting (``civ_value_refset_idx``), the
single-order mode, the winrate-to-score map over the SUBSET, the failed-call
fallback, the terminal variant, and the client's answer-letter arithmetic.
"""

import io
import json
import math
from types import SimpleNamespace

import pytest

from civmarsh.civtelescope import client
from civmarsh.rewards import potential


@pytest.fixture()
def refset_file(tmp_path):
    # one bucket (turn 10) with 4 reference positions, final scores 0/10/20/30
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
    return str(path)


def _args(refset_file, **kw):
    return SimpleNamespace(
        civ_value_url="http://mock/generate",
        civ_value_refset=refset_file,
        civ_value_concurrency=4,
        **kw,
    )


DECISIONS = [
    {"turn": 8, "focal": "me", "value_input": "mine"},
    {"turn": 12, "focal": "me", "value_input": "mine2"},
]


def test_full_set_both_orders(refset_file, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 0.5
    )
    values = potential.telescope_value(
        DECISIONS, {"score_end": 5}, _args(refset_file, civ_value_single_order=False)
    )
    assert len(calls) == 2 * 4 * 2  # 2 decisions x 4 references x both orders
    # winrate 0.5 -> midpoint of the [0, 10, 20, 30] quantile map
    assert values == [15.0, 15.0]


def test_single_order_is_the_default(refset_file, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 0.5
    )
    potential.telescope_value(DECISIONS, {"score_end": 5}, _args(refset_file))
    assert len(calls) == 2 * 4


def test_subset_single_order(refset_file, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 1.0
    )
    values = potential.telescope_value(
        DECISIONS,
        {"score_end": 5},
        _args(refset_file, civ_value_refset_idx=[0, 3], civ_value_single_order=True),
    )
    assert len(calls) == 2 * 2 * 1  # 2 decisions x 2 references x one order
    # single order: the decision is always position A, so its rendering leads
    assert all(c.index("mine") < c.index("render") for c in calls)
    # all wins -> top of the SUBSET span [0, 30]
    assert values == [30.0, 30.0]


def test_all_losses_hit_subset_floor(refset_file, monkeypatch):
    monkeypatch.setattr(client, "prompt_preference", lambda url, p: 0.0)
    values = potential.telescope_value(
        DECISIONS,
        {"score_end": 5},
        _args(refset_file, civ_value_refset_idx=[1, 2], civ_value_single_order=True),
    )
    assert values == [10.0, 10.0]  # subset span [10, 20], winrate 0 -> 10


def test_failed_calls_fall_back_to_half(refset_file, monkeypatch, tmp_path):
    monkeypatch.setattr(client, "prompt_preference", lambda url, p: None)
    log = tmp_path / "value_log.jsonl"
    values = potential.telescope_value(
        DECISIONS, {"score_end": 5}, _args(refset_file, civ_value_log=str(log))
    )
    assert values == [15.0, 15.0]  # winrate defaults to 0.5 -> midpoint
    assert json.loads(log.read_text())["value_fails"] == 2


def test_harvest_keeps_bucket_turns_and_last(refset_file, monkeypatch, tmp_path):
    monkeypatch.setattr(client, "prompt_preference", lambda url, p: 0.5)
    harvest = tmp_path / "harvest.jsonl"
    decisions = [
        {"turn": t, "focal": "me", "value_input": f"r{t}"} for t in (9, 10, 11, 12)
    ]
    potential.telescope_value(
        decisions,
        {"score_end": 7, "position_id": "p0"},
        _args(refset_file, civ_value_harvest=str(harvest)),
    )
    row = json.loads(harvest.read_text())
    assert [d["turn"] for d in row["decisions"]] == [10, 12]
    assert row["score_end"] == 7 and row["last_decision_turn"] == 12
    # a seat stall never enters the reference-set candidate pool
    potential.telescope_value(
        decisions,
        {"score_end": 0, "infra_stall": True},
        _args(refset_file, civ_value_harvest=str(harvest)),
    )
    assert len(harvest.read_text().splitlines()) == 1


def test_terminal_value_scores_the_last_decision_only(
    refset_file, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        client, "prompt_preference", lambda url, p: calls.append(p) or 1.0
    )
    harvest = tmp_path / "harvest.jsonl"
    value = potential.terminal_telescope_value(
        DECISIONS,
        {"score_end": 5},
        _args(
            refset_file,
            civ_value_refset_idx=[0, 1, 2, 3],
            civ_value_harvest=str(harvest),
        ),
    )
    assert value == 30.0
    assert len(calls) == 4 and all("mine2" in c for c in calls)
    assert not harvest.exists()  # harvesting is masked for the terminal value
    with pytest.raises(ValueError):
        potential.terminal_telescope_value([], {"score_end": 5}, _args(refset_file))


def test_reference_selection_nearest_bucket_ties_to_earlier():
    refset = {10: ["a"], 20: ["b"], 30: ["c"]}
    assert potential.select_references(refset, 14) == ["a"]
    assert potential.select_references(refset, 15) == ["a"]
    assert potential.select_references(refset, 16) == ["b"]
    assert potential.select_references(refset, 99) == ["c"]


def test_client_preference_from_top_logprobs(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        top = [
            [math.log(0.6), 1, " A"],
            [math.log(0.2), 2, "B"],
            [math.log(0.1), 3, "A"],
        ]
        return io.BytesIO(
            json.dumps({"meta_info": {"output_top_logprobs": [top]}}).encode()
        )

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    p = client.prompt_preference("http://mock/generate", "PROMPT")
    assert p == pytest.approx(0.6 / (0.6 + 0.2))  # first occurrence of each letter
    assert sent["text"] == "PROMPT\nANSWER:"
    assert sent["sampling_params"] == {"max_new_tokens": 1, "temperature": 0}
    assert sent["top_logprobs_num"] == 20


def test_client_failure_returns_none(monkeypatch):
    def broken(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(client.urllib.request, "urlopen", broken)
    assert client.prompt_preference("http://mock/generate", "PROMPT") is None
