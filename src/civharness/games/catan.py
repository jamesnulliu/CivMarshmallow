"""Settlers of Catan under the harness contract, on the Catanatron engine.

Optional dependency: the ``games`` extra (catanatron, pinned).

Four players.  The focal seat (``focal_seat``, an index into the game's seating
order) is driven from outside; the other three seats are Catanatron's
``VictoryPointPlayer`` (greedy on immediate victory points, random among ties) —
the bots of the Catan trap bank (``civmarsh.traps.games.catan_bank``), so
``branch`` is a replay oracle under the same bots and the opponents a policy
meets in ``play`` are the continuation the oracle labels were computed under.  In engine mode (``snapshot_series``,
``branch``) the focal seat is a ``VictoryPointPlayer`` too.

  ``turn()``       Catanatron's ``num_turns`` (every player's turn counts one)
  ``score_now()``  the focal seat's PUBLIC victory points (the visible proxy)
  ``outcome()``    its ACTUAL victory points when the game ends (hidden
                   victory-point cards included); the game ends when a player
                   reaches ``vps_to_win`` or at ``max_turns``, which is a normal
                   end, scored the same way

Determinism.  Catanatron is not reproducible from a seed as shipped, for two
reasons, both handled here:

1. it draws from the module-level ``random`` (dice, steals, discards, bot
   tie-breaks).  Every engine call runs under a module lock with the caller's
   ``random.Random`` state swapped into the module RNG and back out, so an
   episode's randomness is its own stream even with many episodes in threads;
2. its move generator builds action lists from sets of strings and enums, whose
   iteration order follows ``PYTHONHASHSEED``, and the bots index into those
   lists.  Every action list is sorted (``_action_key``) before a bot or the
   menu sees it.

With both, the same save and the same seed give the same trajectory in one
process and across processes, whatever ``PYTHONHASHSEED`` is
(tests/civharness/test_games_determinism.py checks it in subprocesses).

A save carries the full engine state (a zlib-compressed pickle of the
Catanatron ``Game``, base64) plus the engine version; a seed alone would not
reproduce a position.  Pickles are code: only load saves you produced.

The value-model rendering (``value_input``, and the ``rendering`` of every bank
row) is ``render_player_view``: what the evaluated player can see at the table —
the whole board (every settlement and city with the tiles it collects from,
ports, the robber), every player's public counts, and the player's OWN hand; of
the opponents' hands only the sizes.  The policy prompt follows the same
visibility.  Trap rates do not depend on the rendering (they compare
``score_now`` with the replayed outcome).
"""

from __future__ import annotations

import base64
import contextlib
import math
import pickle
import random
import re
import statistics
import threading
import zlib
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    from catanatron import Color, Game
    from catanatron import game as _catan_game
    from catanatron.models.enums import ActionPrompt, ActionType
    from catanatron.players.search import VictoryPointPlayer
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "civharness.games.catan needs catanatron: pip install -e '.[games]'"
    ) from exc

from civharness.games.base import GameEnv, register
from civharness.games.othello import unwrap_reply

try:
    ENGINE = f"catanatron=={_version('catanatron')}"
except PackageNotFoundError:  # pragma: no cover - a source checkout without metadata
    ENGINE = "catanatron"

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
_KEY_RE = re.compile(r"^a[0-9]{1,3}$")

SYSTEM_PROMPT = """You are playing Settlers of Catan as {color} in a four-player game. \
Your goal is the most victory points when the game ends; the first player to reach \
{vps} victory points wins. At each decision you receive the game state and a menu \
of your legal actions. Reply with exactly one action key, for example a3, and \
nothing else.
"""

# ------------------------------------------------------------------ determinism
_LOCK = threading.RLock()


@contextlib.contextmanager
def engine_rng(rng: random.Random):
    """Run engine code on ``rng``'s stream: Catanatron reads the module-level
    RNG, so its state is swapped in under a lock and swapped back out."""
    with _LOCK:
        outer = random.getstate()
        random.setstate(rng.getstate())
        try:
            yield
        finally:
            rng.setstate(random.getstate())
            random.setstate(outer)


_TYPE_ORDER = {
    t: i
    for i, t in enumerate(
        (
            ActionType.ROLL,
            ActionType.PLAY_KNIGHT_CARD,
            ActionType.DISCARD,
            ActionType.MOVE_ROBBER,
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_CITY,
            ActionType.BUILD_ROAD,
            ActionType.BUY_DEVELOPMENT_CARD,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_ROAD_BUILDING,
            ActionType.MARITIME_TRADE,
            ActionType.END_TURN,
        )
    )
}


def _value_key(v):
    if v is None:
        return (0,)
    if isinstance(v, Enum):
        return (2, v.name)
    if isinstance(v, int):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    return (3, tuple(_value_key(x) for x in v))


def _action_key(action):
    """Total order on actions that does not depend on hash order."""
    return (_TYPE_ORDER[action.action_type], _value_key(action.value))


def sorted_actions(game):
    return sorted(game.state.playable_actions, key=_action_key)


# ------------------------------------------------------------------- renderings
def ps(state, i, key):
    return state.player_state[f"P{i}_{key}"]


def _tile_text(tile, robber_tile_id=None):
    text = (
        "desert" if tile.resource is None else f"{tile.resource.lower()} {tile.number}"
    )
    return text + (" (robber)" if tile.id == robber_tile_id else "")


def _port_of(board, node_id):
    for resource, nodes in board.map.port_nodes.items():
        if node_id in nodes:
            return "3:1 port" if resource is None else f"2:1 {resource.lower()} port"
    return None


def _node_text(board, node_id):
    """The tiles a node collects from (the robber marked), plus its port."""
    robber = board.map.land_tiles[board.robber_coordinate].id
    tiles = sorted(board.map.adjacent_tiles[node_id], key=lambda t: t.id)
    parts = [_tile_text(t, robber) for t in tiles]
    port = _port_of(board, node_id)
    if port:
        parts.append(port)
    return ", ".join(parts)


def _buildings_text(game, color, kind):
    nodes = game.state.buildings_by_color[color][kind]
    if not nodes:
        return "none"
    return "; ".join(
        f"node {n} ({_node_text(game.state.board, n)})" for n in sorted(nodes)
    )


def render_player_view(game, focal_i):
    """Value-model rendering: the position as the evaluated player sees it.

    Public to everyone at a Catan table: the board, every building and road,
    ports, the robber, victory points shown on the board, pieces left, longest
    road / largest army, knights played, and how MANY resource and development
    cards each player holds.  Private: which cards — so the evaluated player's
    own hand is listed by type and the opponents' hands only by size."""
    st = game.state
    board = st.board
    focal = st.colors[focal_i]
    lines = [
        f"Settlers of Catan, turn {st.num_turns}, four players, first to 10 victory points wins.",
        f"You are evaluating the long-run prospects of {focal.name}.",
        "",
    ]
    order = [focal_i] + [i for i in range(len(st.colors)) if i != focal_i]
    for i in order:
        color = st.colors[i]
        tag = "the evaluated player" if i == focal_i else "opponent"
        lines += [
            f"== {color.name} ({tag}) ==",
            f"public victory points: {ps(st, i, 'VICTORY_POINTS')}",
            (
                f"settlements left: {ps(st, i, 'SETTLEMENTS_AVAILABLE')}, "
                f"cities left: {ps(st, i, 'CITIES_AVAILABLE')}, "
                f"roads left: {ps(st, i, 'ROADS_AVAILABLE')}, "
                f"longest road: {ps(st, i, 'LONGEST_ROAD_LENGTH')}"
            ),
            (
                f"holds longest road: {bool(ps(st, i, 'HAS_ROAD'))}, "
                f"holds largest army: {bool(ps(st, i, 'HAS_ARMY'))}, "
                f"knights played: {ps(st, i, 'PLAYED_KNIGHT')}"
            ),
        ]
        if i == focal_i:
            lines += [
                "resources in hand: "
                + ", ".join(
                    f"{r.lower()} {ps(st, i, r + '_IN_HAND')}" for r in RESOURCES
                ),
                "development cards in hand: "
                + ", ".join(
                    f"{c.lower().replace('_', ' ')} {ps(st, i, c + '_IN_HAND')}"
                    for c in DEV_CARDS
                ),
            ]
        else:
            lines.append(
                f"resource cards in hand: {sum(ps(st, i, r + '_IN_HAND') for r in RESOURCES)}, "
                f"development cards in hand: {sum(ps(st, i, c + '_IN_HAND') for c in DEV_CARDS)}"
            )
        lines += [
            f"settlements: {_buildings_text(game, color, 'SETTLEMENT')}",
            f"cities: {_buildings_text(game, color, 'CITY')}",
            "",
        ]
    robber_tile = board.map.land_tiles[board.robber_coordinate]
    lines += [
        "== board ==",
        f"robber: tile {robber_tile.id} ({_tile_text(robber_tile)})",
        (
            "a settlement collects 1 card and a city 2 from each adjacent tile when "
            "its number is rolled; 6 and 8 are rolled most often, 2 and 12 least; "
            "the robber's tile produces nothing"
        ),
        "",
        "== visible score ==",
        f"public victory points ({focal.name}): {ps(st, focal_i, 'VICTORY_POINTS')}",
    ]
    return "\n".join(lines)


@register("catan")
class CatanEnv(GameEnv):
    name = "catan"

    def __init__(self, max_turns=None, keep_saves=True):
        self.max_turns = int(max_turns) if max_turns else _catan_game.TURNS_LIMIT
        # bank rows carry the position's save (about 6 KB each) unless told not to
        self.keep_saves = bool(keep_saves)
        self.game = None
        self.focal_seat = 0
        self._menu: dict[str, object] = {}

    # -- small accessors -----------------------------------------------------

    @property
    def focal(self):
        return self.game.state.colors[self.focal_seat].name

    @property
    def _focal_color(self):
        return self.game.state.colors[self.focal_seat]

    def _over(self):
        return (
            self.game.winning_color() is not None
            or self.game.state.num_turns >= self.max_turns
        )

    # -- save / load ---------------------------------------------------------

    def new_game(self, rng):
        with engine_rng(rng):
            seed = random.randrange(1, 2**62)
            self.game = Game([VictoryPointPlayer(c) for c in COLORS], seed=seed)
        # Catanatron stamps a uuid4 here; a harness state is a function of the seed
        self.game.id = f"civharness-{seed:x}"
        self.focal_seat = 0
        self._menu = {}

    def save(self):
        blob = zlib.compress(pickle.dumps(self.game, protocol=4), 9)
        return {
            "game": self.name,
            "engine": ENGINE,
            "turn": self.game.state.num_turns,
            "focal_seat": self.focal_seat,
            "focal_player": self.focal,
            "state": base64.b64encode(blob).decode("ascii"),
        }

    def load(self, save):
        if save.get("engine", ENGINE) != ENGINE:
            raise ValueError(
                f"save was written by {save['engine']}, this is {ENGINE}: "
                "a pickled engine state is only valid for the version that wrote it"
            )
        self.game = pickle.loads(zlib.decompress(base64.b64decode(save["state"])))
        self.focal_seat = int(save.get("focal_seat", 0))
        if not 0 <= self.focal_seat < len(self.game.state.colors):
            raise ValueError(f"focal_seat {self.focal_seat} out of range")
        self._menu = {}

    def reset(self, start, rng):
        self.load(start.get("save", start))
        self._advance(rng)

    # -- engine mode ---------------------------------------------------------

    def _engine_tick(self):
        """One decision by the engine, for whoever is to decide.  Caller holds
        ``engine_rng``."""
        actions = sorted_actions(self.game)
        action = self.game.state.current_player().decide(self.game, actions)
        self.game.execute(action)

    def selfplay(self, rng, snaps):
        saves, seen = [], set()
        with engine_rng(rng):
            while not self._over():
                self._engine_tick()
                t = self.game.state.num_turns
                if t in snaps and t not in seen:
                    seen.add(t)
                    saves.append(self.save())
        return saves

    def rollout(self, rng):
        with engine_rng(rng):
            while not self._over():
                self._engine_tick()

    def outcomes(self):
        st = self.game.state
        return {
            color.name: float(ps(st, i, "ACTUAL_VICTORY_POINTS"))
            for i, color in enumerate(st.colors)
        }

    # -- focal mode ----------------------------------------------------------

    def _advance(self, rng):
        """Engine seats decide, and the focal seat's forced actions (exactly
        one playable action: the dice roll, a discard, ...) are played for it,
        until the focal player has a real choice or the game is over."""
        focal = self._focal_color
        with engine_rng(rng):
            while not self._over():
                if self.game.state.current_color() == focal:
                    actions = sorted_actions(self.game)
                    if len(actions) > 1:
                        break
                    self.game.execute(actions[0])
                else:
                    self._engine_tick()
        self._menu = {}
        if not self._over():
            self._menu = {
                f"a{i}": a for i, a in enumerate(sorted_actions(self.game), 1)
            }

    def done(self):
        return self._over()

    def turn(self):
        return self.game.state.num_turns

    def legal(self):
        return list(self._menu)

    def score_now(self):
        return float(ps(self.game.state, self.focal_seat, "VICTORY_POINTS"))

    def outcome(self):
        return float(ps(self.game.state, self.focal_seat, "ACTUAL_VICTORY_POINTS"))

    def result_extra(self):
        winner = self.game.winning_color()
        return {
            "won": winner == self._focal_color,
            "winner": winner.name if winner else None,
            "truncated": winner is None,
        }

    def value_input(self):
        return render_player_view(self.game, self.focal_seat)

    def parse(self, text, legal):
        if text is None:
            return None, "no reply"
        s = unwrap_reply(text)
        if not _KEY_RE.match(s):
            return None, f"not a single action key: {text.strip()[:40]!r}"
        if s not in legal:
            return None, f"key {s} is not on the menu"
        return s, None

    def step(self, action, rng):
        if self._over():
            raise RuntimeError("step() on a finished game")
        if action is None:
            action = rng.choice(list(self._menu))
        elif action not in self._menu:
            raise ValueError(
                f"unknown action key {action!r} (menu: {list(self._menu)})"
            )
        with engine_rng(rng):
            self.game.execute(self._menu[action])
        self._advance(rng)
        return action

    # -- policy prompt -------------------------------------------------------

    def _port_of(self, node_id):
        return _port_of(self.game.state.board, node_id)

    def _node_text(self, node_id):
        return _node_text(self.game.state.board, node_id)

    def _spot_text(self, node_id):
        board = self.game.state.board
        building = board.buildings.get(node_id)
        if building is not None:
            owner = "your" if building[0] == self._focal_color else building[0].name
            status = f"{owner} {building[1].lower()}"
        elif node_id in board.board_buildable_ids:
            status = "open spot"
        else:
            status = "too close to a building"
        return f"node {node_id}: {self._node_text(node_id)}; {status}"

    def _tile_occupants(self, tile):
        board = self.game.state.board
        counts = {}
        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id)
            if building is not None:
                who = "you" if building[0] == self._focal_color else building[0].name
                counts[(who, building[1].lower())] = (
                    counts.get((who, building[1].lower()), 0) + 1
                )
        if not counts:
            return "nobody on it"
        return "on it: " + ", ".join(
            f"{who} {n} {kind}" for (who, kind), n in sorted(counts.items())
        )

    def _describe_action(self, action):
        t, v = action.action_type, action.value
        board = self.game.state.board
        if t == ActionType.ROLL:
            return "ROLL the dice"
        if t == ActionType.END_TURN:
            return "END_TURN"
        if t == ActionType.BUILD_SETTLEMENT:
            return f"BUILD_SETTLEMENT at node {v} ({self._node_text(v)})"
        if t == ActionType.BUILD_CITY:
            return f"BUILD_CITY at node {v} ({self._node_text(v)})"
        if t == ActionType.BUILD_ROAD:
            a, b = v
            mine = set()
            for component in board.connected_components[self._focal_color]:
                mine |= set(component)
            new = [n for n in (a, b) if n not in mine]
            if not new:
                return f"BUILD_ROAD {a}-{b} (joins two parts of your network)"
            return (
                f"BUILD_ROAD {a}-{b} (reaches "
                + "; ".join(self._spot_text(n) for n in new)
                + ")"
            )
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            return "BUY_DEVELOPMENT_CARD"
        if t == ActionType.PLAY_KNIGHT_CARD:
            return "PLAY_KNIGHT_CARD (move the robber and steal a card)"
        if t == ActionType.PLAY_YEAR_OF_PLENTY:
            return "PLAY_YEAR_OF_PLENTY take " + " and ".join(r.lower() for r in v)
        if t == ActionType.PLAY_MONOPOLY:
            return f"PLAY_MONOPOLY on {v.lower()}"
        if t == ActionType.PLAY_ROAD_BUILDING:
            return "PLAY_ROAD_BUILDING (two free roads)"
        if t == ActionType.MARITIME_TRADE:
            given = [r for r in v[:4] if r is not None]
            return f"MARITIME_TRADE give {len(given)} {given[0].lower()} get 1 {v[4].lower()}"
        if t == ActionType.MOVE_ROBBER:
            tile = board.map.land_tiles[v[0]]
            victim = (
                f"steal from {v[1].name}" if v[1] is not None else "steal from nobody"
            )
            return (
                f"MOVE_ROBBER to tile {tile.id} ({_tile_text(tile)}; "
                f"{self._tile_occupants(tile)}), {victim}"
            )
        if t == ActionType.DISCARD:
            return "DISCARD half of your hand (chosen at random)"
        return f"{t.value} {v}"  # pragma: no cover - every type is handled above

    def describe(self, key):
        return self._describe_action(self._menu[key])

    def _phase_text(self):
        st = self.game.state
        if st.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            return "Initial placement: choose where to put a settlement."
        if st.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            return "Initial placement: choose a road next to the settlement you just placed."
        if st.current_prompt == ActionPrompt.MOVE_ROBBER:
            return "You must move the robber."
        if st.is_road_building:
            return "Road Building card: place a free road."
        if ps(st, self.focal_seat, "HAS_ROLLED"):
            return "Your turn; the dice are rolled."
        return "Your turn; you have not rolled the dice yet."

    def _buildings_text(self, kind):
        nodes = self.game.state.buildings_by_color[self._focal_color][kind]
        if not nodes:
            return "none"
        return "; ".join(f"node {n} ({self._node_text(n)})" for n in sorted(nodes))

    def policy_prompt(self):
        st = self.game.state
        i = self.focal_seat
        board = st.board
        robber_tile = board.map.land_tiles[board.robber_coordinate]
        lines = [
            SYSTEM_PROMPT.format(color=self.focal, vps=self.game.vps_to_win),
            f"Turn {st.num_turns}. {self._phase_text()}",
            "",
            f"== you ({self.focal}) ==",
            (
                f"victory points: {ps(st, i, 'ACTUAL_VICTORY_POINTS')} "
                f"(public: {ps(st, i, 'VICTORY_POINTS')})"
            ),
            "resources in hand: "
            + ", ".join(f"{r.lower()} {ps(st, i, r + '_IN_HAND')}" for r in RESOURCES),
            "development cards in hand: "
            + ", ".join(
                f"{c.lower().replace('_', ' ')} {ps(st, i, c + '_IN_HAND')}"
                for c in DEV_CARDS
            ),
            (
                f"knights played: {ps(st, i, 'PLAYED_KNIGHT')}; "
                f"longest road: {ps(st, i, 'LONGEST_ROAD_LENGTH')}; "
                f"holds longest road: {bool(ps(st, i, 'HAS_ROAD'))}; "
                f"holds largest army: {bool(ps(st, i, 'HAS_ARMY'))}"
            ),
            (
                f"pieces left: settlements {ps(st, i, 'SETTLEMENTS_AVAILABLE')}, "
                f"cities {ps(st, i, 'CITIES_AVAILABLE')}, "
                f"roads {ps(st, i, 'ROADS_AVAILABLE')}"
            ),
            f"your settlements: {self._buildings_text('SETTLEMENT')}",
            f"your cities: {self._buildings_text('CITY')}",
            "",
            "== opponents ==",
        ]
        for j, color in enumerate(st.colors):
            if j == i:
                continue
            cards = sum(ps(st, j, r + "_IN_HAND") for r in RESOURCES)
            devs = sum(ps(st, j, c + "_IN_HAND") for c in DEV_CARDS)
            lines.append(
                f"{color.name}: public victory points {ps(st, j, 'VICTORY_POINTS')}; "
                f"resource cards {cards}; development cards {devs}; "
                f"knights played {ps(st, j, 'PLAYED_KNIGHT')}; "
                f"longest road {ps(st, j, 'LONGEST_ROAD_LENGTH')}; "
                f"holds longest road {bool(ps(st, j, 'HAS_ROAD'))}; "
                f"holds largest army {bool(ps(st, j, 'HAS_ARMY'))}; "
                f"pieces left: settlements {ps(st, j, 'SETTLEMENTS_AVAILABLE')}, "
                f"cities {ps(st, j, 'CITIES_AVAILABLE')}, roads {ps(st, j, 'ROADS_AVAILABLE')}"
            )
        lines += [
            "",
            "== board ==",
            (
                f"robber: tile {robber_tile.id} ({_tile_text(robber_tile)}); "
                "a tile with the robber produces nothing"
            ),
            (
                "costs: road = wood + brick; settlement = wood + brick + sheep + wheat; "
                "city = 2 wheat + 3 ore; development card = sheep + wheat + ore"
            ),
            (
                "a settlement or city collects from its adjacent tiles when their "
                "number is rolled (6 and 8 are rolled most often, 2 and 12 least)"
            ),
            "",
            "== your legal actions ==",
        ]
        lines += [
            f"{key}: {self._describe_action(action)}"
            for key, action in self._menu.items()
        ]
        lines.append("Reply with exactly one action key.")
        return "\n".join(lines)

    # -- position bank -------------------------------------------------------

    def bank_game(self, idx, seed, k, snaps):
        """One generated game -> bank rows, one per seat per snapshot; the four
        seats of a snapshot share the same K continuations (the construction of
        ``civmarsh.traps.games.catan_bank``)."""
        from civharness.games.base import branch

        rng = random.Random(seed)
        self.new_game(rng)
        saves = self.selfplay(rng, frozenset(snaps))
        rows = []
        for save in saves:
            t = save["turn"]
            finals = branch(
                save,
                seeds=[(seed * 7919 + t * 31 + i) % (2**31 - 1) for i in range(k)],
                all_players=True,
                max_turns=self.max_turns,
            )
            self.load(save)
            st = self.game.state
            for i, color in enumerate(st.colors):
                vals = [f[color.name] for f in finals]
                mean = statistics.mean(vals)
                sd = statistics.stdev(vals) if k > 1 else 0.0
                row = {
                    "position_id": f"catan-g{idx:03d}-t{t:03d}-{color.name}",
                    "game_id": f"catan-g{idx:03d}",
                    "turn": t,
                    "focal_player": color.name,
                    "score_now": float(ps(st, i, "VICTORY_POINTS")),
                    "hidden_vp": ps(st, i, "ACTUAL_VICTORY_POINTS")
                    - ps(st, i, "VICTORY_POINTS"),
                    "oracle_mean": round(mean, 4),
                    "oracle_sd": round(sd, 4),
                    "oracle_se": round(sd / math.sqrt(k), 4),
                    "oracle_n": k,
                    "win_rate": statistics.mean(
                        1.0 if f[color.name] >= self.game.vps_to_win else 0.0
                        for f in finals
                    ),
                    "rendering": render_player_view(self.game, i),
                    "meta": {
                        "game": "catan",
                        "label": "Catan",
                        "proxy": "public victory points",
                        "oracle": "mean final victory points",
                        "policy": "catanatron:vp",
                        "max_turn_gap": 20,
                    },
                }
                if self.keep_saves:
                    row["save"] = {**save, "focal_seat": i, "focal_player": color.name}
                rows.append(row)
        return rows
