"""OpenSpiel banks (backgammon, 9x9 Go, Hearts, 2048): self-play, the game's
own visible running score as the proxy, K replays to the end as the oracle.

The engine is OpenSpiel's C++ MCTS (one random rollout per simulation, few
simulations) with 10% uniformly random moves. Two games cannot use it: 2048
has no terminal-only reward (one-ply greedy on merge reward and empty cells,
10% random) and Hearts is imperfect-information (uniform random play for
every seat). Requires open_spiel (``pyspiel``).
"""

from __future__ import annotations

import math
import random
import re
import statistics
from concurrent.futures import ProcessPoolExecutor

EPSILON = 0.10
DEFAULT_MAX_REPLAY_MOVES = 3000
DEFAULTS = {"replays": 100, "seed": 20260911}


# ----------------------------------------------------------- visible proxies


def bg_pips(s, p):
    """Pip count for player p (0 = X moves toward the high indices, 1 = O
    toward the low)."""
    pips = 0
    for k in range(24):
        pips += s.board(p, k) * ((24 - k) if p == 0 else (k + 1))
    bar = next(line for line in s.to_string().splitlines() if line.startswith("Bar:"))
    return pips + 25 * bar.count("x" if p == 0 else "o")


def bg_adjudicate(s):
    """A replay that hits the move cap is scored as a pure race: the side with
    fewer pips wins."""
    d = bg_pips(s, 1) - bg_pips(s, 0)
    return 1.0 if d > 0 else -1.0 if d < 0 else 0.0


def go_area(s, size=9, komi=7.5):
    """Naive area count for Black (stones plus empty points reachable from one
    colour only), minus komi."""
    rows = [
        line
        for line in s.to_string().splitlines()
        if re.match(r"^\s*\d+ [XO+]+$", line)
    ]
    grid = [list(r.split()[1]) for r in rows]
    n = len(grid)
    assert n == size, (n, rows[:2])
    black = white = 0
    seen = set()
    for r in range(n):
        for c in range(n):
            ch = grid[r][c]
            if ch == "X":
                black += 1
            elif ch == "O":
                white += 1
            elif (r, c) not in seen:
                stack, region, touch = [(r, c)], 0, set()
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    region += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < n and 0 <= xx < n:
                            g = grid[yy][xx]
                            if g == "+" and (yy, xx) not in seen:
                                seen.add((yy, xx))
                                stack.append((yy, xx))
                            elif g != "+":
                                touch.add(g)
                if touch == {"X"}:
                    black += region
                elif touch == {"O"}:
                    white += region
    return black - white - komi


def hearts_points(s):
    """Penalty points of seat 0 relative to the other seats, sign-flipped so
    that larger is better."""
    return -(s.points(0) - statistics.mean(s.points(p) for p in range(1, 4)))


SPECS = {
    "backgammon": {
        "game": "backgammon",
        "label": "Backgammon",
        "focal": "X",
        "proxy_name": "pip-count lead",
        "proxy": lambda s: bg_pips(s, 1) - bg_pips(s, 0),
        "outcome": lambda s: s.returns()[0],
        "oracle_name": "mean result for X (+1 win, -1 loss)",
        "snaps": range(20, 121, 20),
        "gap": 20,
        "policy": "mcts:30",
        "n_games": 40,
        "max_replay_moves": 400,
        "adjudicate": bg_adjudicate,
    },
    "go9": {
        "game": "go(board_size=9)",
        "label": "Go (9x9)",
        "focal": "Black",
        "proxy_name": "area count with komi",
        "proxy": go_area,
        "outcome": go_area,
        "oracle_name": "mean final area margin for Black",
        "snaps": range(16, 65, 8),
        "gap": 8,
        "policy": "mcts:30",
        "n_games": 60,
    },
    "hearts": {
        "game": "hearts",
        "label": "Hearts",
        "focal": "North",
        "proxy_name": "penalty points so far, relative to the table",
        "proxy": hearts_points,
        "outcome": hearts_points,
        "oracle_name": "mean final relative penalty",
        "snaps": range(12, 45, 8),
        "gap": 8,
        "policy": "random",
        "n_games": 60,
    },
    "2048": {
        "game": "2048",
        "label": "2048",
        "focal": "the player",
        "proxy_name": "score",
        "proxy": lambda s: s.returns()[0],
        "outcome": lambda s: s.returns()[0],
        "oracle_name": "mean final score",
        "snaps": range(30, 151, 20),
        "gap": 20,
        "policy": "greedy",
        "n_games": 60,
    },
}


# ------------------------------------------------------------------ policies


def greedy2048(rng):
    def act(s):
        if rng.random() < EPSILON:
            return rng.choice(s.legal_actions())
        best, best_v = None, -1e9
        for a in s.legal_actions():
            t = s.clone()
            t.apply_action(a)
            v = t.rewards()[0] + 2.0 * t.to_string().count(" 0")
            if v > best_v:
                best, best_v = a, v
        return best

    return act


def make_policy(kind, game, seed):
    """random | greedy (2048) | mcts:N (N simulations, EPSILON random moves)."""
    rng = random.Random(seed)
    if kind == "random":
        return lambda s: rng.choice(s.legal_actions())
    if kind == "greedy":
        return greedy2048(rng)
    if kind.startswith("mcts"):
        import pyspiel

        sims = int(kind.split(":")[1])
        sd = seed % (2**31 - 1)
        bot = pyspiel.MCTSBot(
            game,
            pyspiel.RandomRolloutEvaluator(1, sd),
            1.4,
            sims,
            100,
            False,
            sd,
            False,
        )

        def act(s):
            if rng.random() < EPSILON:
                return rng.choice(s.legal_actions())
            return bot.step(s)

        return act
    raise ValueError(kind)


def step(s, policy, rng) -> bool:
    """Advance one node; True when a player moved (not a chance node)."""
    if s.is_chance_node():
        acts, probs = zip(*s.chance_outcomes())
        s.apply_action(rng.choices(acts, probs)[0])
        return False
    s.apply_action(policy(s))
    return True


def playout(s, policy, rng, max_moves=DEFAULT_MAX_REPLAY_MOVES):
    n = 0
    while not s.is_terminal() and n < max_moves:
        n += step(s, policy, rng)
    return s


def render(spec, s, move):
    """Structured text in the shape of the Freeciv rendering (model-visible)."""
    who = spec["focal"]
    return "\n".join(
        [
            f"{spec['label']} position after {move} moves.",
            f"You are evaluating the long-run prospects of {who}.",
            "",
            "== board / state (full information) ==",
            s.to_string().rstrip(),
            "",
            "== visible score ==",
            f"{spec['proxy_name']} ({who}): {spec['proxy'](s):+g}",
        ]
    )


def gen_position(job) -> dict | None:
    """One (game, snapshot) unit: regenerate the self-play game
    deterministically up to the snapshot, then replay it K times. Per-snapshot
    jobs keep one long game from holding a worker. None when the game ends
    before the snapshot."""
    import pyspiel

    name, idx, seed, move_target, k_replays, policy_kind = job
    spec = SPECS[name]
    game = pyspiel.load_game(spec["game"])
    rng = random.Random(seed)
    policy = make_policy(policy_kind, game, seed)
    s = game.new_initial_state()
    move = 0
    while not s.is_terminal() and move < move_target:
        if s.is_chance_node():
            step(s, policy, rng)
            continue
        step(s, policy, rng)
        move += 1
    while not s.is_terminal() and s.is_chance_node():
        step(s, policy, rng)
    if s.is_terminal() or move != move_target:
        return None
    cap = spec.get("max_replay_moves", DEFAULT_MAX_REPLAY_MOVES)
    vals, capped = [], 0
    for k in range(k_replays):
        rseed = (seed * 7919 + move * 31 + k) % (2**31 - 1)
        t = playout(
            s.clone(),
            make_policy(policy_kind, game, rseed),
            random.Random(rseed),
            max_moves=cap,
        )
        if not t.is_terminal() and "adjudicate" in spec:
            vals.append(float(spec["adjudicate"](t)))
            capped += 1
        else:
            vals.append(float(spec["outcome"](t)))
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if k_replays > 1 else 0.0
    return {
        "position_id": f"{name}-g{idx:03d}-m{move:03d}",
        "game_id": f"{name}-g{idx:03d}",
        "turn": move,
        "focal_player": spec["focal"],
        "score_now": float(spec["proxy"](s)),
        "oracle_mean": round(mean, 4),
        "oracle_sd": round(sd, 4),
        "oracle_se": round(sd / math.sqrt(k_replays), 4),
        "oracle_n": k_replays,
        "oracle_capped": capped,
        "rendering": render(spec, s, move),
        "meta": {
            "game": name,
            "label": spec["label"],
            "proxy": spec["proxy_name"],
            "oracle": spec["oracle_name"],
            "policy": policy_kind,
            "epsilon": EPSILON if policy_kind != "random" else 1.0,
            "max_turn_gap": spec["gap"],
            "max_replay_moves": cap,
        },
    }


def build_bank(
    game: str,
    *,
    n_games: int | None = None,
    game_offset: int = 0,
    replays: int = DEFAULTS["replays"],
    snaps=None,
    seed: int = DEFAULTS["seed"],
    workers: int = 20,
    log=print,
) -> list[dict]:
    """Games game_offset .. game_offset + n_games - 1 (game i uses seed + i;
    OpenSpiel self-play reproduces from the seed, so the same games can be
    snapshotted again at other moves). Defaults come from the game's spec."""
    spec = SPECS[game]
    n_games = spec["n_games"] if n_games is None else n_games
    snaps = list(spec["snaps"]) if snaps is None else list(snaps)
    policy = spec["policy"]
    jobs = [
        (game, game_offset + i, seed + game_offset + i, m, replays, policy)
        for i in range(n_games)
        for m in snaps
    ]
    log(
        f"{game}: {n_games} games x {len(snaps)} snapshots x {replays} replays, "
        f"policy {policy}, {workers} workers"
    )
    bank = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, row in enumerate(ex.map(gen_position, jobs, chunksize=1), 1):
            if row is not None:
                bank.append(row)
            if done % 50 == 0:
                log(f"  {done}/{len(jobs)} snapshots, {len(bank)} positions")
    bank.sort(key=lambda r: r["position_id"])
    capped = sum(r["oracle_capped"] for r in bank)
    if capped:
        log(f"{capped} of {len(bank) * replays} replays hit the move cap (adjudicated)")
    return bank


def self_play_length(job) -> int:
    """Player moves in one complete self-play game, (game, seed) -> length."""
    import pyspiel

    name, seed = job
    spec = SPECS[name]
    game = pyspiel.load_game(spec["game"])
    rng = random.Random(seed)
    pol = make_policy(spec["policy"], game, seed)
    s = game.new_initial_state()
    n = 0
    while not s.is_terminal() and n < spec.get(
        "max_replay_moves", DEFAULT_MAX_REPLAY_MOVES
    ):
        n += step(s, pol, rng)
    return n
