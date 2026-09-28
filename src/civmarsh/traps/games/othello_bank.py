"""Othello bank: disc differential as the visible proxy, the final disc
differential under K stochastic replays as the oracle.

The proxy and the oracle are the same quantity at two different times (disc
differential now and at the end of the game), as with Freeciv's score now and
terminal score. The engine is a dependency-free bitboard Othello with a small
alpha-beta search, softmax move sampling and an exact endgame.

Bit i is square (row i // 8, col i % 8); bit 0 = a1.
"""

from __future__ import annotations

import random
import statistics
from concurrent.futures import ProcessPoolExecutor

FULL = (1 << 64) - 1
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
# (shift, mask applied after shifting) for the eight ray directions
DIRS = (
    (1, ~FILE_A & FULL),
    (-1, ~FILE_H & FULL),
    (8, FULL),
    (-8, FULL),
    (9, ~FILE_A & FULL),
    (-9, ~FILE_H & FULL),
    (7, ~FILE_H & FULL),
    (-7, ~FILE_A & FULL),
)

# classic Othello square weights (corners good, X/C squares bad)
# fmt: off
WEIGHTS = [
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20, 5, 5, 20, -20, 120,
]
# fmt: on
START_BLACK = (1 << 28) | (1 << 35)  # d5, e4
START_WHITE = (1 << 27) | (1 << 36)  # d4, e5
CORNERS = 0x8100000000000081

DEFAULTS = {
    "n_games": 70,
    "replays": 24,
    "seed": 20260909,
    "snaps": tuple(range(12, 41, 5)),
    "gen_depth": 2,
    "gen_temp": 40.0,
    "replay_depth": 2,
    "replay_temp": 20.0,
}


# ------------------------------------------------------------------ rules


def sh(x, d, mask):
    return ((x << d) if d > 0 else (x >> -d)) & mask & FULL


def moves(me, opp):
    empty = ~(me | opp) & FULL
    out = 0
    for d, mask in DIRS:
        x = sh(me, d, mask) & opp
        for _ in range(5):
            x |= sh(x, d, mask) & opp
        out |= sh(x, d, mask) & empty
    return out


def apply_move(me, opp, mv):
    bit = 1 << mv
    flips = 0
    for d, mask in DIRS:
        run = 0
        x = sh(bit, d, mask) & opp
        while x:
            run |= x
            nxt = sh(x, d, mask)
            if nxt & me:
                flips |= run
                break
            x = nxt & opp
    me = me | bit | flips
    opp = opp & ~flips
    return me, opp


def popcount(x):
    return x.bit_count()


def bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


# ------------------------------------------------------------------ engine


def evaluate(me, opp):
    """Positive = good for `me`. Weights + mobility; used only in the midgame."""
    score = 0
    for sq in bits(me):
        score += WEIGHTS[sq]
    for sq in bits(opp):
        score -= WEIGHTS[sq]
    return score + 8 * (popcount(moves(me, opp)) - popcount(moves(opp, me)))


def negamax(me, opp, depth, alpha, beta, exact):
    mv = moves(me, opp)
    if not mv:
        if not moves(opp, me):
            return (popcount(me) - popcount(opp)) * 1000
        return -negamax(opp, me, depth, -beta, -alpha, exact)
    if depth == 0 and not exact:
        return evaluate(me, opp)
    best = -(10**9)
    for sq in bits(mv):
        nm, no = apply_move(me, opp, sq)
        val = -negamax(no, nm, depth - 1, -beta, -alpha, exact)
        best = max(best, val)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def choose(me, opp, depth, rng, temperature, exact_from=10):
    """Softmax over root move values; temperature=0 is greedy. With at most
    `exact_from` empty squares the search is exact and the move greedy."""
    mv = moves(me, opp)
    if not mv:
        return None
    empties = 64 - popcount(me | opp)
    exact = empties <= exact_from
    scored = []
    for sq in bits(mv):
        nm, no = apply_move(me, opp, sq)
        scored.append((-negamax(no, nm, depth - 1, -(10**9), 10**9, exact), sq))
    if temperature <= 0 or exact:
        return max(scored)[1]
    top = max(v for v, _ in scored)
    weights = [pow(2.718281828, (v - top) / temperature) for v, _ in scored]
    return rng.choices([sq for _, sq in scored], weights=weights)[0]


def play_out(me, opp, black_to_move, depth, rng, temperature):
    """Play to the end; returns the final disc differential for Black."""
    black, white = (me, opp) if black_to_move else (opp, me)
    turn_black = black_to_move
    while True:
        cur, other = (black, white) if turn_black else (white, black)
        if not moves(cur, other):
            if not moves(other, cur):
                break
            turn_black = not turn_black
            continue
        sq = choose(cur, other, depth, rng, temperature)
        cur, other = apply_move(cur, other, sq)
        black, white = (cur, other) if turn_black else (other, cur)
        turn_black = not turn_black
    return popcount(black) - popcount(white)


# --------------------------------------------------------------- rendering


def render(black, white, black_to_move, move_no):
    """Structured text in the shape of the Freeciv rendering (model-visible)."""
    grid = []
    for row in range(7, -1, -1):
        line = []
        for col in range(8):
            sq = row * 8 + col
            line.append("B" if black >> sq & 1 else "W" if white >> sq & 1 else ".")
        grid.append(f"{row + 1} " + " ".join(line))
    nb, nw = popcount(black), popcount(white)
    legal = moves(black, white) if black_to_move else moves(white, black)
    names = [f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}" for sq in bits(legal)]
    return "\n".join(
        [
            (
                f"Othello (reversi) position after {move_no} moves, "
                f"{'Black' if black_to_move else 'White'} to move."
            ),
            "You are evaluating the long-run prospects of Black.",
            "",
            "== Black (the evaluated player) ==",
            f"discs on board: {nb}",
            f"legal moves available now: {len(list(bits(moves(black, white))))}",
            f"corners held: {popcount(black & CORNERS)}",
            "",
            "== opponents ==",
            "White",
            f"discs on board: {nw}",
            f"legal moves available now: {len(list(bits(moves(white, black))))}",
            f"corners held: {popcount(white & CORNERS)}",
            "",
            "== board (B = Black, W = White, . = empty) ==",
            "  a b c d e f g h",
            *grid,
            "",
            "== visible score ==",
            f"disc differential (Black minus White): {nb - nw:+d}",
            f"empty squares remaining: {64 - nb - nw}",
            (
                "legal moves for the side to move: "
                f"{', '.join(names) if names else 'none (must pass)'}"
            ),
        ]
    )


# -------------------------------------------------------------------- bank


def generate_game(job) -> list[dict]:
    """One self-play game; its snapshots labeled by K replays each."""
    idx, seed, n_replays, snap_moves, gen_depth, gen_temp, rep_depth, rep_temp = job
    rng = random.Random(seed)
    black, white = START_BLACK, START_WHITE
    turn_black, move_no = True, 0
    snapshots = []
    while True:
        cur, other = (black, white) if turn_black else (white, black)
        if not moves(cur, other):
            if not moves(other, cur):
                break
            turn_black = not turn_black
            continue
        if move_no in snap_moves:
            snapshots.append((move_no, black, white, turn_black))
        sq = choose(cur, other, gen_depth, rng, gen_temp)
        cur, other = apply_move(cur, other, sq)
        black, white = (cur, other) if turn_black else (other, cur)
        turn_black = not turn_black
        move_no += 1
    final_diff = popcount(black) - popcount(white)

    rows = []
    for move, b, w, tb in snapshots:
        nb, nw = popcount(b), popcount(w)
        me, opp = (b, w) if tb else (w, b)
        results = [
            play_out(
                me,
                opp,
                tb,
                rep_depth,
                random.Random(seed * 1000 + move * 31 + k),
                rep_temp,
            )
            for k in range(n_replays)
        ]
        mean = statistics.mean(results)
        sd = statistics.stdev(results) if len(results) > 1 else 0.0
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
                "black_to_move": tb,
                "board_black": b,
                "board_white": w,
                "rendering": render(b, w, tb, move),
                "meta": {"game": "othello", "policy": "alphabeta:depth2+temperature"},
            }
        )
    return rows


def build_bank(
    *,
    n_games: int = DEFAULTS["n_games"],
    game_offset: int = 0,
    replays: int = DEFAULTS["replays"],
    snaps=DEFAULTS["snaps"],
    seed: int = DEFAULTS["seed"],
    workers: int = 6,
    gen_depth: int = DEFAULTS["gen_depth"],
    gen_temp: float = DEFAULTS["gen_temp"],
    replay_depth: int = DEFAULTS["replay_depth"],
    replay_temp: float = DEFAULTS["replay_temp"],
    log=print,
) -> list[dict]:
    """Games game_offset .. game_offset + n_games - 1 (game i uses seed + i)."""
    snap_moves = frozenset(snaps)
    jobs = [
        (
            game_offset + i,
            seed + game_offset + i,
            replays,
            snap_moves,
            gen_depth,
            gen_temp,
            replay_depth,
            replay_temp,
        )
        for i in range(n_games)
    ]
    log(
        f"othello: {len(jobs)} games x {len(snap_moves)} snapshots x "
        f"{replays} replays, {workers} workers"
    )
    bank = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, rows in enumerate(ex.map(generate_game, jobs), 1):
            bank.extend(rows)
            if done % 5 == 0:
                log(f"  {done}/{len(jobs)} games, {len(bank)} positions")
    return bank
