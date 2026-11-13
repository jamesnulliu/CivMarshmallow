"""Othello under the harness contract (stdlib only, no engine binary)."""

import json
import random

import pytest

from civharness.games import (
    bank_rows,
    branch,
    make_env,
    noise_floor,
    play,
    registered_games,
    snapshot_series,
    summary,
)
from civharness.games import othello_engine as oth
from civharness.games.othello import parse_move, square_index, square_name


def test_registry_lists_both_games_without_importing_an_engine():
    assert registered_games() == ["catan", "othello"]
    with pytest.raises(KeyError):
        make_env("chess")


def test_snapshot_series_saves_are_json_and_reload_exactly():
    saves = snapshot_series("othello", 7, [8, 20, 40])
    assert [s["turn"] for s in saves] == [8, 20, 40]
    env = make_env("othello")
    for save in saves:
        again = json.loads(json.dumps(save))
        env.load(again)
        assert env.save() == save
        assert env.turn() == save["turn"] and not env.done()


def test_branch_is_the_engine_playout():
    """branch(save, seeds) == othello_engine.play_out with the same seeds."""
    save = snapshot_series("othello", 3, [41])[0]
    black, white = int(save["board_black"], 0), int(save["board_white"], 0)
    tb = save["black_to_move"]
    me, opp = (black, white) if tb else (white, black)
    seeds = [11, 12]
    expect = [oth.play_out(me, opp, tb, 2, random.Random(s), 20.0) for s in seeds]
    assert branch(save, seeds=seeds) == expect
    values = branch(save, 3, 5)
    assert noise_floor(save, 3, 5) == summary(values) and summary(values)["n"] == 3


def test_focal_mode_matches_the_engine_rules():
    """A seeded random game through the adapter: every offered move is legal
    for Black, discs are conserved, and the terminal score is the board's."""
    env = make_env("othello")
    rng = random.Random(4)
    env.new_game(rng)
    env.reset(env.save(), rng)
    n = 0
    while not env.done():
        legal = env.legal()
        assert legal and env.black_to_move
        assert set(legal) == {
            square_name(sq) for sq in oth.bits(oth.moves(env.black, env.white))
        }
        before = oth.popcount(env.black | env.white)
        env.step(rng.choice(legal), rng)
        assert oth.popcount(env.black | env.white) > before
        n += 1
    assert n > 10
    assert not oth.moves(env.black, env.white) and not oth.moves(env.white, env.black)
    assert env.outcome() == oth.popcount(env.black) - oth.popcount(env.white)
    assert env.outcomes() == {"Black": env.outcome(), "White": -env.outcome()}


def test_reset_hands_over_when_white_is_to_move():
    save = next(
        s for s in snapshot_series("othello", 1, range(5, 30)) if not s["black_to_move"]
    )
    env = make_env("othello")
    env.reset(save, random.Random(1))
    assert env.done() or (env.black_to_move and env.legal())
    assert env.turn() > save["turn"]


def test_parse_move():
    legal = ["d3", "c4"]
    assert parse_move(" D3 ", legal) == ("d3", None)
    assert parse_move("`c4`.", legal) == ("c4", None)
    assert parse_move("d3 because", legal)[0] is None
    assert parse_move("e6", legal) == (None, "illegal move e6")
    assert parse_move(None, legal) == (None, "no reply")
    assert square_name(square_index("h8")) == "h8"


def test_play_is_reproducible_and_falls_back_on_invalid_replies():
    start = snapshot_series("othello", 2, [6])[0]
    decisions, result = play(start, lambda prompt: "not a move", 5)
    again, result2 = play(start, lambda prompt: "not a move", 5)
    assert result == result2 and decisions == again
    assert all(d["error"] and not d["keys"] and d["fallback"] for d in decisions)
    assert result["end_turn"] >= start["turn"] and result["eliminated"] is False
    other, _ = play(start, lambda prompt: "not a move", 6)
    assert [d["fallback"] for d in other] != [d["fallback"] for d in decisions]


def test_bank_rows_deterministic_and_independent_of_workers():
    snaps = [38, 46]
    a = bank_rows("othello", [101, 102], snaps, 2)
    b = bank_rows("othello", [101, 102], snaps, 2, workers=2)
    assert a == b and len(a) == 4
    assert {r["game_id"] for r in a} == {"o0000", "o0001"}
    row = a[-1]
    assert row["oracle_n"] == 2 and row["focal_player"] == "Black"
    # a bank row is itself a save
    assert branch(row, 2, 1, game="othello") == branch(row, 2, 1, game="othello")


def test_bank_rows_identical_to_the_trap_bank_generator():
    """The harness bank is the Othello trap bank's generator: same seeds, same
    rows (the trap bank adds a ``meta`` field)."""
    from civmarsh.traps.games import othello_bank as ref

    snaps = frozenset(range(12, 41, 5))
    for idx, seed in ((5, 20260914),):
        want = ref.generate_game((idx, seed, 2, snaps, 2, 40.0, 2, 20.0))
        for row in want:
            row.pop("meta")
        got = bank_rows("othello", [seed], snaps, 2, index_offset=idx)
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
