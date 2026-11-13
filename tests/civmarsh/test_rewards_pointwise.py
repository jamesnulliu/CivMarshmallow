"""Pointwise potentials on the CivTelescope reference set (no GPU, no HTTP)."""

import json
from types import SimpleNamespace

import pytest

from civmarsh.rewards import pointwise, potential


def render(score, turn):
    return "header\nDIGEST " + json.dumps({"metrics": {"score": score, "turn": turn}})


def score_of(item):
    return json.loads(item["rendering"].split("DIGEST ", 1)[1])["metrics"]["score"]


@pytest.fixture()
def refset_file(tmp_path):
    """Two turn buckets of four reference positions: final scores 10..40,
    visible scores 5..20."""
    rows = [
        {
            "turn": turn,
            "focal": "P",
            "end_score": end,
            "position_id": f"t{turn}-{i}",
            "rendering": render(end // 2, turn),
        }
        for turn in (40, 50)
        for i, end in enumerate((10, 20, 30, 40))
    ]
    path = tmp_path / "refset.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    potential._load_at.cache_clear()
    return str(path)


def decisions(scores, turn=41):
    return [
        {"turn": turn, "focal": "P", "value_input": render(s, turn)} for s in scores
    ]


def test_fractional_rank():
    ref = [5, 10, 15, 20]
    assert pointwise.fractional_rank(4, ref) == 0.0
    assert pointwise.fractional_rank(25, ref) == 1.0
    assert pointwise.fractional_rank(10, ref) == pytest.approx(1 / 3)
    assert pointwise.fractional_rank(12.5, ref) == pytest.approx(0.5)
    assert pointwise.fractional_rank(3, [7, 7, 7]) == 0.5


def test_value_head_is_mapped_through_the_reference_set(
    refset_file, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pointwise,
        "_post_value_head",
        lambda url, items, timeout=600: [score_of(it) for it in items],
    )
    args = SimpleNamespace(
        civ_value_refset=refset_file,
        civ_value_head_urls=["http://a/value"],
        civ_value_log=str(tmp_path / "value_log.jsonl"),
        civ_value_harvest=str(tmp_path / "harvest.jsonl"),
    )
    values = pointwise.value_head_value(
        decisions([5, 12.5, 20, 99]), {"score_end": 33.0}, args
    )
    assert values == pytest.approx([10.0, 25.0, 40.0, 40.0])
    log = json.loads((tmp_path / "value_log.jsonl").read_text())
    assert log["value_fails"] == 0 and log["raw_scores"][1] == 12.5
    assert json.loads((tmp_path / "harvest.jsonl").read_text())["score_end"] == 33.0


def test_value_head_uses_the_selected_reference_positions(refset_file, monkeypatch):
    monkeypatch.setattr(
        pointwise,
        "_post_value_head",
        lambda url, items, timeout=600: [score_of(it) * 2.0 for it in items],
    )
    args = SimpleNamespace(
        civ_value_refset=refset_file,
        civ_value_head_urls=["http://a/value", "http://b/value"],
        civ_value_refset_idx=[0, 3],
        civ_value_head_batch=2,
    )
    # idx [0, 3]: the final scores 10 and 40, whose head scores are 10 and 40
    values = pointwise.value_head_value(
        decisions([5, 12.5, 20]), {"score_end": 0.0}, args
    )
    assert values == pytest.approx([10.0, 25.0, 40.0])


def test_failed_decision_falls_back_to_half(refset_file, tmp_path, monkeypatch):
    def post(url, items, timeout=600):
        return [None if score_of(it) == 99 else score_of(it) for it in items]

    monkeypatch.setattr(pointwise, "_post_value_head", post)
    args = SimpleNamespace(
        civ_value_refset=refset_file,
        civ_value_head_urls=["http://c/value"],
        civ_value_head_batch=1,
        civ_value_log=str(tmp_path / "value_log.jsonl"),
    )
    values = pointwise.value_head_value(decisions([99]), {"score_end": 1.0}, args)
    assert values == pytest.approx([25.0])
    assert json.loads((tmp_path / "value_log.jsonl").read_text())["value_fails"] == 1


def test_unreachable_value_head_fails_loudly_on_the_reference_set(
    refset_file, monkeypatch
):
    monkeypatch.setattr(
        pointwise,
        "_post_value_head",
        lambda url, items, timeout=600: [None] * len(items),
    )
    args = SimpleNamespace(
        civ_value_refset=refset_file,
        civ_value_head_urls=["http://unreachable/value"],
    )
    with pytest.raises(RuntimeError):
        pointwise.value_head_value(decisions([5]), {"score_end": 0.0}, args)


def test_reference_rank_correlation(refset_file):
    corr = pointwise.reference_rank_correlation(
        refset_file, lambda items: [score_of(it) for it in items]
    )
    assert corr == {40: 1.0, 50: 1.0}
