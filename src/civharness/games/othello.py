"""Othello under the harness contract.  Stdlib only.

Focal player = Black.  Every other move is the bundled engine
(``othello_engine.choose``): a depth-``opp_depth`` search with a softmax over
root values at temperature ``opp_temp`` and an exact endgame from
``opp_exact_from`` empties.  The defaults (2 / 20.0 / 10) are the replay
engine of the Othello trap bank (``civmarsh.traps.games.othello_bank``), so
``branch`` is that bank's replay oracle and the opponent a policy meets in
``play`` is the continuation the oracle labels were computed under.
``snapshot_series`` generates games with ``gen_depth`` / ``gen_temp``
(2 / 40.0, the bank's generation engine).

``turn()`` is the number of discs placed since the standard start (passes do
not count).  A save is ``{game, board_black, board_white, black_to_move,
move_no, turn, focal_player}``; boards are "0x…" strings (64-bit values do not
survive every JSON round trip) and ints are accepted on load, so a row of the
position bank is itself a valid save.
"""

from __future__ import annotations

import random
import re
import statistics

from civharness.games import othello_engine as oth
from civharness.games.base import GameEnv, register

SYSTEM_PROMPT = """You are playing Othello (reversi) as Black against White. \
Your goal is the highest final disc differential (Black discs minus White discs) \
when the game ends. Each turn you receive the board and the list of your legal \
moves. Reply with exactly one legal square, for example d3, and nothing else.
"""

_SQUARE_RE = re.compile(r"^[a-h][1-8]$")
_WRAP = "\"'`"


def square_name(sq: int) -> str:
    return f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}"


def square_index(name: str) -> int:
    return (int(name[1]) - 1) * 8 + (ord(name[0]) - ord("a"))


def unwrap_reply(text):
    """Strip what a one-token answer may be wrapped in: whitespace, ONE trailing
    period, one layer of quotes/backticks.  Shared by the games' reply parsers."""
    s = text.strip().lower()
    if len(s) >= 2 and s[0] in _WRAP and s[-1] == s[0]:
        s = s[1:-1].strip()
    if s.endswith("."):
        s = s[:-1].strip()
    if len(s) >= 2 and s[0] in _WRAP and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s


def parse_move(text, legal):
    """Raw reply -> (square, None) or (None, error).

    Valid iff the whole stripped reply is exactly one square that is in
    ``legal`` — "Move: d3" or "d3 because ..." is an error."""
    if text is None:
        return None, "no reply"
    s = unwrap_reply(text)
    if not _SQUARE_RE.match(s):
        return None, f"not a single square: {text.strip()[:40]!r}"
    if s not in legal:
        return None, f"illegal move {s}"
    return s, None


def _board(value) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


def _grid(black, white):
    rows = []
    for row in range(7, -1, -1):
        line = []
        for col in range(8):
            sq = row * 8 + col
            line.append("B" if black >> sq & 1 else "W" if white >> sq & 1 else ".")
        rows.append(f"{row + 1} " + " ".join(line))
    return rows


@register("othello")
class OthelloEnv(GameEnv):
    name = "othello"
    focal = "Black"

    def __init__(
        self, opp_depth=2, opp_temp=20.0, opp_exact_from=10, gen_depth=2, gen_temp=40.0
    ):
        self.opp_depth = int(opp_depth)
        self.opp_temp = float(opp_temp)
        self.opp_exact_from = int(opp_exact_from)
        self.gen_depth = int(gen_depth)
        self.gen_temp = float(gen_temp)
        self.black = self.white = 0
        self.black_to_move = True
        self.move_no = 0
        self._over = True

    # -- save / load ---------------------------------------------------------

    def new_game(self, rng):
        self.black, self.white = oth.START_BLACK, oth.START_WHITE
        self.black_to_move, self.move_no, self._over = True, 0, False

    def save(self):
        return {
            "game": self.name,
            "board_black": hex(self.black),
            "board_white": hex(self.white),
            "black_to_move": self.black_to_move,
            "move_no": self.move_no,
            "turn": self.move_no,
            "focal_player": self.focal,
        }

    def load(self, save):
        self.black = _board(save["board_black"])
        self.white = _board(save["board_white"])
        if self.black & self.white:
            raise ValueError(f"{save.get('position_id')}: overlapping discs")
        self.black_to_move = bool(save.get("black_to_move", True))
        self.move_no = int(save["move_no"] if "move_no" in save else save["turn"])
        self._over = not (
            oth.moves(self.black, self.white) or oth.moves(self.white, self.black)
        )

    def reset(self, start, rng):
        self.load(start)
        if not self._over and not (
            self.black_to_move and oth.moves(self.black, self.white)
        ):
            # White to move, or Black must pass: hand the turn over like any
            # other forced pass, so the first decision is a real one
            self._advance(rng)

    # -- engine mode ---------------------------------------------------------

    def _engine_game(self, rng, depth, temp, snaps=(), exact_from=10):
        """Both sides are the engine, to the end.  The loop (pass handling and
        the single ``choose`` per move) consumes ``rng`` exactly like
        ``othello_engine.play_out`` and the bank's game generator.
        ``exact_from`` defaults to ``othello_engine.choose``'s own 10, so the
        generator and every default replay are unchanged."""
        saves = []
        while True:
            cur, other = (
                (self.black, self.white)
                if self.black_to_move
                else (self.white, self.black)
            )
            if not oth.moves(cur, other):
                if not oth.moves(other, cur):
                    break
                self.black_to_move = not self.black_to_move
                continue
            if self.move_no in snaps:
                saves.append(self.save())
            sq = oth.choose(cur, other, depth, rng, temp, exact_from=exact_from)
            cur, other = oth.apply_move(cur, other, sq)
            self.black, self.white = (
                (cur, other) if self.black_to_move else (other, cur)
            )
            self.black_to_move = not self.black_to_move
            self.move_no += 1
        self._over = True
        return saves

    def selfplay(self, rng, snaps):
        return self._engine_game(rng, self.gen_depth, self.gen_temp, snaps)

    def rollout(self, rng):
        # the replay oracle plays both sides as the focal-mode opponent does,
        # exact endgame included (opp_exact_from, default 10), so a bank's
        # labels match the opponent a policy is trained against
        self._engine_game(
            rng, self.opp_depth, self.opp_temp, exact_from=self.opp_exact_from
        )

    def outcomes(self):
        diff = oth.popcount(self.black) - oth.popcount(self.white)
        return {"Black": diff, "White": -diff}

    # -- focal mode ----------------------------------------------------------

    def done(self):
        return self._over

    def turn(self):
        return self.move_no

    def legal(self):
        return [square_name(sq) for sq in oth.bits(oth.moves(self.black, self.white))]

    def score_now(self):
        return float(oth.popcount(self.black) - oth.popcount(self.white))

    def outcome(self):
        return float(oth.popcount(self.black) - oth.popcount(self.white))

    def value_input(self):
        return oth.render(self.black, self.white, self.black_to_move, self.move_no)

    def policy_prompt(self):
        nb, nw = oth.popcount(self.black), oth.popcount(self.white)
        return "\n".join(
            [
                SYSTEM_PROMPT,
                f"Position after {self.move_no} moves, Black (you) to move.",
                "",
                "== board (B = Black, W = White, . = empty) ==",
                "  a b c d e f g h",
                *_grid(self.black, self.white),
                "",
                f"Black discs: {nb}",
                f"White discs: {nw}",
                f"disc differential (Black minus White): {nb - nw:+d}",
                f"empty squares remaining: {64 - nb - nw}",
                "",
                f"Your legal moves: {', '.join(self.legal())}",
                "Reply with exactly one of them.",
            ]
        )

    def parse(self, text, legal):
        return parse_move(text, legal)

    def step(self, action, rng):
        if self._over:
            raise RuntimeError("step() on a finished game")
        if not self.black_to_move:
            raise RuntimeError("step() while White is to move (use reset, not load)")
        legal = self.legal()
        if action is None:
            action = rng.choice(legal)
        elif action not in legal:
            raise ValueError(f"illegal move {action!r} (legal: {legal})")
        self.black, self.white = oth.apply_move(
            self.black, self.white, square_index(action)
        )
        self.move_no += 1
        self.black_to_move = False
        self._advance(rng)
        return action

    def _advance(self, rng):
        """White moves until Black has a legal move or neither side has one.
        A side with no legal move passes; passes are not counted in move_no."""
        while True:
            if oth.moves(self.white, self.black):
                sq = oth.choose(
                    self.white,
                    self.black,
                    self.opp_depth,
                    rng,
                    self.opp_temp,
                    exact_from=self.opp_exact_from,
                )
                self.white, self.black = oth.apply_move(self.white, self.black, sq)
                self.move_no += 1
                if oth.moves(self.black, self.white):
                    self.black_to_move = True
                    return
                continue  # Black must pass: White moves again (or the game ends)
            self.black_to_move = True
            if not oth.moves(self.black, self.white):
                self._over = True
            return

    # -- position bank -------------------------------------------------------

    def bank_game(self, idx, seed, k, snaps):
        """One generated game -> bank rows, one per snapshot.  Same seeds,
        arithmetic and rows (without its ``meta`` field) as
        ``civmarsh.traps.games.othello_bank.generate_game``, so a bank built
        here continues the trap bank's distribution exactly."""
        from civharness.games.base import branch

        rng = random.Random(seed)
        self.new_game(rng)
        saves = self.selfplay(rng, frozenset(snaps))
        final_diff = self.outcomes()["Black"]
        cfg = {
            "opp_depth": self.opp_depth,
            "opp_temp": self.opp_temp,
            "opp_exact_from": self.opp_exact_from,
        }
        rows = []
        for save in saves:
            move = save["move_no"]
            b, w = _board(save["board_black"]), _board(save["board_white"])
            results = branch(
                save, seeds=[seed * 1000 + move * 31 + i for i in range(k)], **cfg
            )
            mean = statistics.mean(results)
            sd = statistics.stdev(results) if len(results) > 1 else 0.0
            nb, nw = oth.popcount(b), oth.popcount(w)
            rows.append(
                {
                    "position_id": f"o{idx:04d}-m{move:03d}",
                    "game_id": f"o{idx:04d}",
                    "turn": move,
                    "focal_player": "Black",
                    "score_now": nb - nw,
                    "discs_black": nb,
                    "discs_white": nw,
                    "oracle_mean": round(mean, 4),
                    "oracle_sd": round(sd, 4),
                    "oracle_se": round(sd / (len(results) ** 0.5), 4),
                    "oracle_n": len(results),
                    "game_final_diff": final_diff,
                    "black_to_move": save["black_to_move"],
                    "board_black": b,
                    "board_white": w,
                    "rendering": oth.render(b, w, save["black_to_move"], move),
                }
            )
        return rows
