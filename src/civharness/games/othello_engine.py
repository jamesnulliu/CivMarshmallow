"""Minimal bitboard Othello: rules, a small alpha-beta engine, exact endgame.

Bit i is square (row i // 8, col i % 8); bit 0 = a1.  Kept dependency-free so
the bank is reproducible from the repo alone.
"""

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
WEIGHTS = [
    120,
    -20,
    20,
    5,
    5,
    20,
    -20,
    120,
    -20,
    -40,
    -5,
    -5,
    -5,
    -5,
    -40,
    -20,
    20,
    -5,
    15,
    3,
    3,
    15,
    -5,
    20,
    5,
    -5,
    3,
    3,
    3,
    3,
    -5,
    5,
    5,
    -5,
    3,
    3,
    3,
    3,
    -5,
    5,
    20,
    -5,
    15,
    3,
    3,
    15,
    -5,
    20,
    -20,
    -40,
    -5,
    -5,
    -5,
    -5,
    -40,
    -20,
    120,
    -20,
    20,
    5,
    5,
    20,
    -20,
    120,
]
START_BLACK = (1 << 28) | (1 << 35)  # d5, e4
START_WHITE = (1 << 27) | (1 << 36)  # d4, e5


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
        else:
            continue
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


def evaluate(me, opp):
    """Positive = good for `me`.  Weights + mobility; used only in the midgame."""
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
    """Softmax over root move values; temperature=0 is greedy."""
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


def render(black, white, black_to_move, move_no):
    """Structured text in the shape of the Freeciv rendering."""
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
            f"corners held: {popcount(black & 0x8100000000000081)}",
            "",
            "== opponents ==",
            "White",
            f"discs on board: {nw}",
            f"legal moves available now: {len(list(bits(moves(white, black))))}",
            f"corners held: {popcount(white & 0x8100000000000081)}",
            "",
            "== board (B = Black, W = White, . = empty) ==",
            "  a b c d e f g h",
            *grid,
            "",
            "== visible score ==",
            f"disc differential (Black minus White): {nb - nw:+d}",
            f"empty squares remaining: {64 - nb - nw}",
            f"legal moves for the side to move: {', '.join(names) if names else 'none (must pass)'}",
        ]
    )
