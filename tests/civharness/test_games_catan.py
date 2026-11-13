"""Catan (Catanatron) under the harness contract.  Skipped without the
optional ``games`` extra (catanatron)."""

import json
import random

import pytest

pytest.importorskip("catanatron")

from civharness.games import bank_rows, branch, make_env, play


def fresh(seed=5, seat=0):
    env = make_env("catan")
    env.new_game(random.Random(seed))
    env.focal_seat = seat
    return env, env.save()


def test_save_round_trips_through_json():
    env, save = fresh()
    assert save["game"] == "catan" and save["engine"].startswith("catanatron==")
    again = json.loads(json.dumps(save))
    other = make_env("catan")
    other.load(again)
    assert other.focal == env.focal and other.turn() == 0
    assert other.value_input() == env.value_input()
    with pytest.raises(ValueError):
        other.load({**again, "engine": "catanatron==0.0.0"})


def test_menu_keys_are_stable_and_describe_every_action():
    env, save = fresh(seat=2)
    env.reset(save, random.Random(1))
    legal = env.legal()
    assert legal == [f"a{i}" for i in range(1, len(legal) + 1)]
    prompt = env.policy_prompt()
    for key in legal:
        assert f"{key}: {env.describe(key)}" in prompt
    assert f"as {env.focal} in a four-player game" in prompt
    # the same position offers the same menu again
    other = make_env("catan")
    other.reset(save, random.Random(1))
    assert [other.describe(k) for k in other.legal()] == [
        env.describe(k) for k in legal
    ]


def test_forced_actions_are_played_for_the_focal_seat():
    """Every decision offered has at least two actions; a lone ROLL / DISCARD /
    END_TURN never reaches the policy."""
    _, save = fresh(seed=8, seat=1)
    sizes = []

    def act(prompt):
        n = prompt.count("\n", prompt.index("== your legal actions ==")) - 1
        sizes.append(n)
        return f"a{n}"  # the last key: END_TURN whenever it is offered

    decisions, result = play(save, act, 3)
    assert decisions and min(sizes) >= 2
    assert all(d["n_legal"] >= 2 and not d["error"] for d in decisions)
    assert result["score_end"] >= 2.0 and result["end_turn"] > 0


def test_policy_prompt_hides_opponent_hands():
    env, save = fresh(seed=4)
    rng = random.Random(2)
    env.reset(save, rng)
    for _ in range(20):
        if env.done():
            break
        env.step(None, rng)
    prompt = env.policy_prompt()
    you, rest = prompt.split("== opponents ==")
    assert "resources in hand:" in you
    opponents = rest.split("== board ==")[0]
    assert "resources in hand" not in opponents and "resource cards" in opponents
    # the value model sees what the player sees: its own hand by type, the others'
    # by size, and the board (every settlement with the tiles it collects from, the robber)
    seen = env.value_input()
    assert seen.count("resources in hand:") == 1
    assert seen.count("resource cards in hand:") == 3
    own = seen.split("(the evaluated player) ==")[1].split("== ")[0]
    assert "resources in hand:" in own and "settlements: node " in own
    assert "== board ==" in seen and "robber: tile " in seen
    assert seen.count("settlements: ") == 4 and seen.count("cities: ") == 4


def test_parse():
    env, save = fresh()
    env.reset(save, random.Random(1))
    legal = env.legal()
    assert env.parse(" A2 ", legal) == ("a2", None)
    assert env.parse("'a1'.", legal) == ("a1", None)
    assert env.parse("a2 because it is good", legal)[0] is None
    assert env.parse("a999", legal)[0] is None
    assert env.parse(None, legal) == (None, "no reply")


def test_siblings_share_the_start_and_differ_by_seed():
    # seat 0 decides first, so every sibling's first decision is the saved
    # position itself; from then on the engine seed (dice, bots) separates them
    _, save = fresh(seed=6, seat=0)
    first_prompts = []

    def act(prompt):
        first_prompts.append(prompt)
        return "a1"

    a, ra = play(save, act, 1)
    n = len(first_prompts)
    b, rb = play(save, act, 2)
    assert first_prompts[0] == first_prompts[n]  # identical first decision
    assert [d["action"] for d in a] != [d["action"] for d in b] or ra != rb
    c, rc = play(save, lambda p: "a1", 1)
    assert [d["action"] for d in a] == [d["action"] for d in c] and ra == rc


def test_scores_and_turn_cap():
    env = make_env("catan", max_turns=30)
    env.new_game(random.Random(3))
    env.rollout(random.Random(4))
    assert env.done() and env.turn() == 30
    finals = env.outcomes()
    assert set(finals) == {"RED", "BLUE", "WHITE", "ORANGE"}
    assert env.outcome() == finals[env.focal] and env.result_extra()["truncated"]
    assert env.score_now() <= env.outcome()


def test_bank_rows_schema_and_determinism():
    rows = bank_rows("catan", [31], [20, 40], 3)
    assert rows == bank_rows("catan", [31], [20, 40], 3)
    assert len(rows) == 8 and {r["turn"] for r in rows} == {20, 40}
    for row in rows:
        for field in (
            "position_id",
            "game_id",
            "focal_player",
            "score_now",
            "oracle_mean",
            "oracle_sd",
            "oracle_se",
            "oracle_n",
            "rendering",
            "win_rate",
            "save",
        ):
            assert field in row
        assert row["save"]["focal_player"] == row["focal_player"]
        assert f"prospects of {row['focal_player']}." in row["rendering"]
    # the four seats of a snapshot share the continuations: the save is one state
    assert len({r["save"]["state"] for r in rows if r["turn"] == 20}) == 1
    # a row's save branches to that seat's outcomes
    outs = branch(rows[0]["save"], 3, 5)
    assert len(outs) == 3 and all(2.0 <= v <= 12.0 for v in outs)
    assert "save" not in bank_rows("catan", [31], [20], 2, keep_saves=False)[0]
