"""Cross-game pieces that need no game library: the Othello engine and bank,
the game registry, the zero-shot prompt, pair sampling and scoring."""

import random
import string

from civmarsh.traps.games import GAMES
from civmarsh.traps.games import othello_bank as oth
from civmarsh.traps.games.prompt import CROSS_GAME_PAIRWISE_PROMPT, cross_game_prompt
from civmarsh.traps.games.zero_shot import evaluate_model, select_pairs, wilson


def test_registry_pair_gaps():
    gaps = {g: info["max_turn_gap"] for g, info in GAMES.items()}
    assert gaps == {
        "chess": 8,
        "othello": 8,
        "backgammon": 20,
        "go9": 8,
        "hearts": 8,
        "2048": 20,
        "catan": 20,
    }


def test_othello_rules():
    black, white = oth.START_BLACK, oth.START_WHITE
    legal = oth.moves(black, white)
    assert oth.popcount(legal) == 4
    sq = next(oth.bits(legal))
    nb, nw = oth.apply_move(black, white, sq)
    assert oth.popcount(nb) == 4 and oth.popcount(nw) == 1
    assert oth.choose(black, white, 1, random.Random(0), 0.0) in set(oth.bits(legal))


def test_othello_bank_game():
    job = (3, 20260912, 2, frozenset({12, 17}), 1, 40.0, 1, 20.0)
    rows = oth.generate_game(job)
    assert [r["position_id"] for r in rows] == ["o0003-m012", "o0003-m017"]
    r = rows[0]
    assert r["score_now"] == r["discs_black"] - r["discs_white"]
    assert r["oracle_n"] == 2
    assert r["rendering"].startswith("Othello (reversi) position after 12 moves, ")
    assert "disc differential (Black minus White): " in r["rendering"]
    assert oth.generate_game(job) == rows  # reproducible from the seed


def test_prompt_fill():
    a = {"focal_player": "White", "rendering": "R-A"}
    b = {"focal_player": "Black", "rendering": "R-B"}
    text = cross_game_prompt("chess", a, b)
    assert text.startswith(
        "You will see two positions from two different games of chess, "
    )
    assert "=== Position A (evaluate player 'White') ===\nR-A\n" in text
    assert "=== Position B (evaluate player 'Black') ===\nR-B\n" in text
    assert text.endswith('Reply with exactly one line: "ANSWER: A" or "ANSWER: B".')
    fields = {f for _, f, _, _ in string.Formatter().parse(CROSS_GAME_PAIRWISE_PROMPT)}
    assert fields - {None} == {
        "game",
        "focal_a",
        "rendering_a",
        "focal_b",
        "rendering_b",
    }


def mk_pair(i, trap, tied=False):
    return {
        "a": f"a{i}",
        "b": f"b{i}",
        "winner": "A",
        "is_trap": trap,
        "score_tied": tied,
    }


def test_select_pairs_drops_ties_after_shuffle():
    pairs = [mk_pair(i, i % 3 == 0, tied=i % 5 == 0) for i in range(60)]
    sel = select_pairs(pairs, n_trap=5, n_non_trap=7, seed=97)
    assert sum(p["is_trap"] for p in sel) == 5 and len(sel) == 12
    assert not any(p["score_tied"] for p in sel)
    assert sel == select_pairs(pairs, n_trap=5, n_non_trap=7, seed=97)


class FakeClient:
    """Always answers A: right on every pair shown in order A-B (the oracle
    winner is A), wrong on every reversed one."""

    def __init__(self, fail_tag=None):
        self.fail_tag = fail_tag

    def chat(self, prompt, tag=""):
        if self.fail_tag and tag.endswith(self.fail_tag):
            raise RuntimeError("down")
        return "thinking...\nANSWER: A"


def test_evaluate_model_per_pair_accuracy():
    pairs = [mk_pair(0, True), mk_pair(1, False)]
    bank = {
        pid: {"focal_player": "X", "rendering": pid}
        for p in pairs
        for pid in (p["a"], p["b"])
    }
    res = evaluate_model(FakeClient(), "chess", bank, pairs, workers=2)
    assert res["acc_trap"] == 0.5 and res["acc_non_trap"] == 0.5
    assert res["n_trap"] == 1 and res["n_errors"] == 0
    res = evaluate_model(FakeClient(fail_tag="|ba"), "chess", bank, pairs, workers=2)
    assert res["acc_trap"] == 1.0 and res["n_errors"] == 2


def test_wilson():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == (None, None)
