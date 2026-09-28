"""Chess bank: material balance as the visible proxy, the mean result under K
stochastic Stockfish replays as the oracle.

The oracle is a branch replay, not an engine evaluation: from the sampled
position the game is played out K times by Stockfish at a reduced "Skill
Level" (which adds move noise), and the label is the mean result for White
(1 win, 0.5 draw, 0 loss). Requires python-chess and a Stockfish binary named
by the STOCKFISH_BIN environment variable.
"""

from __future__ import annotations

import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

FOCAL_NAME = "White"
MAX_PLY = 300
ADJUDICATE_DEPTH = 12
ADJUDICATE_CP = 150

DEFAULTS = {
    "n_games": 80,
    "replays": 20,
    "seed": 20260909,
    "snaps": tuple(range(24, 73, 8)),
    "gen_skill": 8,
    "replay_skill": 10,
    "nodes": 20000,
}


def stockfish_bin() -> str:
    path = os.environ.get("STOCKFISH_BIN")
    if not path:
        raise RuntimeError("set STOCKFISH_BIN to the path of a Stockfish binary")
    return path


def _tables():
    import chess

    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    piece_name = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    return piece_value, piece_name


def new_engine(skill, threads=1, hash_mb=32):
    import chess.engine

    eng = chess.engine.SimpleEngine.popen_uci(stockfish_bin())
    eng.configure({"Skill Level": skill, "Threads": threads, "Hash": hash_mb})
    return eng


def material(board, color):
    piece_value, _ = _tables()
    return sum(
        piece_value[p.piece_type]
        for p in board.piece_map().values()
        if p.color == color
    )


def result_for(board, color):
    """1.0 win / 0.5 draw / 0.0 loss, from a finished board."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == color else 0.0


def play_out(board, eng, limit, rng, max_ply=MAX_PLY):
    """Play the position to a natural end (or the ply cap) and score it for
    White. At the cap a deeper deterministic search adjudicates: more than
    +150 centipawns is a win, below -150 a loss, a draw otherwise."""
    import chess
    import chess.engine

    b = board.copy()
    while not b.is_game_over(claim_draw=True) and b.ply() < max_ply:
        # a fresh random game key per move keeps replays from the same
        # position independent; Stockfish's Skill Level noise supplies the spread
        res = eng.play(b, limit, game=rng.random())
        if res.move is None:
            break
        b.push(res.move)
    if b.is_game_over(claim_draw=True):
        return result_for(b, chess.WHITE), True
    info = eng.analyse(b, chess.engine.Limit(depth=ADJUDICATE_DEPTH))
    cp = info["score"].pov(chess.WHITE).score(mate_score=10000)
    if cp > ADJUDICATE_CP:
        return 1.0, False
    return (0.0 if cp < -ADJUDICATE_CP else 0.5), False


def render(board, ply, material_w, material_b):
    """Structured text in the shape of the Freeciv rendering: the evaluated
    player's assets first, then the opponent's, then the visible score
    (model-visible)."""
    import chess

    _, piece_name = _tables()
    lines = [
        (
            f"Chess position after {ply} plies (move {ply // 2 + 1}, "
            f"{'White' if board.turn == chess.WHITE else 'Black'} to move)."
        ),
        f"You are evaluating the long-run prospects of {FOCAL_NAME}.",
        "",
        f"== {FOCAL_NAME} (the evaluated player) ==",
        f"material: {material_w} pawn-equivalents",
    ]
    for color in (chess.WHITE, chess.BLACK):
        if color == chess.BLACK:
            lines += [
                "",
                "== opponents ==",
                "Black",
                f"material: {material_b} pawn-equivalents",
            ]
        counts = {}
        for sq, piece in board.piece_map().items():
            if piece.color == color:
                counts.setdefault(piece_name[piece.piece_type], []).append(
                    chess.square_name(sq)
                )
        for name in ("queen", "rook", "bishop", "knight", "pawn", "king"):
            if name in counts:
                lines.append(f"  {name}s: {', '.join(sorted(counts[name]))}")
        rights = [
            name
            for name, has in (
                ("kingside", board.has_kingside_castling_rights(color)),
                ("queenside", board.has_queenside_castling_rights(color)),
            )
            if has
        ]
        lines.append(f"  castling rights: {' + '.join(rights) or 'none'}")
    lines += [
        "",
        "== board (uppercase = White, lowercase = Black, . = empty) ==",
        str(board),
        "",
        "== visible score ==",
        f"material balance ({FOCAL_NAME} minus Black): {material_w - material_b:+d}",
        f"halfmove clock: {board.halfmove_clock}",
        f"FEN: {board.fen()}",
    ]
    return "\n".join(lines)


def generate_game(job) -> list[dict]:
    """One self-play game from a random 4-10 move opening; its snapshots
    labeled by K replays each."""
    import chess
    import chess.engine

    game_index, seed, n_replays, sample_plies, gen_skill, replay_skill, nodes = job
    rng = random.Random(seed)
    eng = new_engine(gen_skill)
    try:
        board = chess.Board()
        gen_limit = chess.engine.Limit(nodes=nodes)
        # random legal opening so the games are not all the same line
        for _ in range(rng.randint(4, 10)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        snapshots = []
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLY:
            if board.ply() in sample_plies:
                snapshots.append((board.ply(), board.fen()))
            res = eng.play(board, gen_limit, game=rng.random())
            if res.move is None:
                break
            board.push(res.move)
        game_result = (
            result_for(board, chess.WHITE)
            if board.is_game_over(claim_draw=True)
            else None
        )
        game_end_ply = board.ply()
    finally:
        eng.quit()

    out = []
    eng = new_engine(replay_skill)
    try:
        limit = chess.engine.Limit(nodes=nodes)
        for ply, fen in snapshots:
            b = chess.Board(fen)
            mw, mb = material(b, chess.WHITE), material(b, chess.BLACK)
            results, natural = [], 0
            for k in range(n_replays):
                r, nat = play_out(
                    b, eng, limit, random.Random(seed * 1000 + ply * 31 + k)
                )
                results.append(r)
                natural += int(nat)
            mean = statistics.mean(results)
            sd = statistics.stdev(results) if len(results) > 1 else 0.0
            out.append(
                {
                    "position_id": f"c{game_index:04d}-p{ply:03d}",
                    "game_id": f"c{game_index:04d}",
                    "turn": ply,
                    "focal_player": FOCAL_NAME,
                    "score_now": mw - mb,
                    "material_white": mw,
                    "material_black": mb,
                    "oracle_mean": round(mean, 4),
                    "oracle_sd": round(sd, 4),
                    "oracle_se": round(sd / (len(results) ** 0.5), 4),
                    "oracle_n": len(results),
                    "oracle_natural_end_frac": round(natural / len(results), 3),
                    "game_result_focal": game_result,
                    "game_end_ply": game_end_ply,
                    "fen": fen,
                    "rendering": render(b, ply, mw, mb),
                    "meta": {"game": "chess", "policy": "stockfish:skill"},
                }
            )
    finally:
        eng.quit()
    return out


def build_bank(
    *,
    n_games: int = DEFAULTS["n_games"],
    game_offset: int = 0,
    replays: int = DEFAULTS["replays"],
    snaps=DEFAULTS["snaps"],
    seed: int = DEFAULTS["seed"],
    workers: int = 6,
    gen_skill: int = DEFAULTS["gen_skill"],
    replay_skill: int = DEFAULTS["replay_skill"],
    nodes: int = DEFAULTS["nodes"],
    log=print,
) -> list[dict]:
    """Games game_offset .. game_offset + n_games - 1 (game i uses seed + i).

    Stockfish self-play does not reproduce from the seed, so further games
    for an existing bank need their own index range (`game_offset`)."""
    stockfish_bin()
    sample_plies = frozenset(snaps)
    jobs = [
        (
            game_offset + i,
            seed + game_offset + i,
            replays,
            sample_plies,
            gen_skill,
            replay_skill,
            nodes,
        )
        for i in range(n_games)
    ]
    log(
        f"chess: {len(jobs)} games x {len(sample_plies)} snapshots x "
        f"{replays} replays, {workers} workers"
    )
    bank = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, rows in enumerate(ex.map(generate_game, jobs), 1):
            bank.extend(rows)
            if done % 5 == 0:
                log(f"  {done}/{len(jobs)} games, {len(bank)} positions")
    return bank
