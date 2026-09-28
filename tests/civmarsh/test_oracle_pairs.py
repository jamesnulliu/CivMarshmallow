"""Decidable-pair rule: game and turn gates, the 2-SE gap rule, trap and tie
flags, and the tie / zero-SE policies."""

from civmarsh.oracle.pairs import decidable_pairs, trap_summary


def rec(pid, game, turn, value, se, now):
    return {
        "position_id": pid,
        "game_id": game,
        "turn": turn,
        "mean_end": value,
        "se_end": se,
        "score_now": now,
    }


def ids(pairs):
    return [(p["a"], p["b"]) for p in pairs]


def test_same_game_and_turn_gap_are_excluded():
    recs = [
        rec("a", "g1", 10, 100, 1, 10),
        rec("b", "g1", 10, 50, 1, 5),  # same game as a
        rec("c", "g2", 21, 50, 1, 5),  # 11 turns from a
        rec("d", "g3", 20, 50, 1, 5),  # 10 turns from a
    ]
    assert ids(decidable_pairs(recs, max_turn_gap=10)) == [("a", "d")]


def test_gap_must_exceed_twice_pooled_se():
    se = 3.0  # pooled SE = hypot(3, 4) = 5 -> decidable iff gap > 10
    b_se = 4.0
    at_rule = [rec("a", "g1", 10, 110, se, 1), rec("b", "g2", 10, 100, b_se, 2)]
    above = [rec("a", "g1", 10, 110.01, se, 1), rec("b", "g2", 10, 100, b_se, 2)]
    assert decidable_pairs(at_rule, max_turn_gap=10) == []
    (p,) = decidable_pairs(above, max_turn_gap=10)
    assert p["pooled_se"] == 5.0
    assert p["margin"] == 10.01
    assert p["margin_z"] == round(10.01 / 5.0, 2)


def test_winner_and_trap_flag():
    recs = [
        rec("a", "g1", 10, 200, 1, 10),  # better oracle, lower score now
        rec("b", "g2", 12, 100, 1, 30),
        rec("c", "g3", 14, 50, 1, 5),  # worse oracle, lower score now
    ]
    by_id = {(p["a"], p["b"]): p for p in decidable_pairs(recs, max_turn_gap=10)}
    ab, bc = by_id[("a", "b")], by_id[("b", "c")]
    assert ab["winner"] == "A" and ab["is_trap"]
    assert bc["winner"] == "A" and not bc["is_trap"]
    assert ab["turn_a"] == 10 and ab["turn_b"] == 12
    assert ab["score_a"] == 10 and ab["score_b"] == 30
    assert trap_summary(list(by_id.values()))["trap"] == 1


def test_tie_policy():
    recs = [rec("a", "g1", 10, 90, 1, 20), rec("b", "g2", 10, 100, 1, 20)]
    assert decidable_pairs(recs, max_turn_gap=10) == []
    (p,) = decidable_pairs(recs, max_turn_gap=10, drop_ties=False)
    assert p["score_tied"]
    # ">=" on both sides: the scoreboard picks A on a tie, the oracle picks B
    assert p["winner"] == "B" and p["is_trap"]


def test_zero_se_policy():
    zero = [rec("a", "g1", 10, 41, 0, 1), rec("b", "g2", 10, 40, 0, 2)]
    tie = [rec("a", "g1", 10, 40, 0, 1), rec("b", "g2", 10, 40, 0, 2)]
    assert len(decidable_pairs(zero, max_turn_gap=10)) == 1
    assert decidable_pairs(zero, max_turn_gap=10, skip_zero_se=True) == []
    assert decidable_pairs(tie, max_turn_gap=10) == []


def test_custom_keys_and_combination_order():
    recs = [
        {
            "position_id": f"q{i}",
            "game_id": f"g{i}",
            "turn": 8,
            "oracle_mean": v,
            "oracle_se": 0.01,
            "score_now": -v,
        }
        for i, v in enumerate([0.1, 0.5, 0.9])
    ]
    pairs = decidable_pairs(
        recs, max_turn_gap=8, value_key="oracle_mean", se_key="oracle_se"
    )
    assert ids(pairs) == [("q0", "q1"), ("q0", "q2"), ("q1", "q2")]
    assert all(p["is_trap"] for p in pairs)
