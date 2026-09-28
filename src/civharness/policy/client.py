"""Client-driven policy: real construction / research / unit decisions issued
over the network protocol, the faithful counterpart to the Lua ScriptedPolicy.

Where the Lua policy (policy/__init__.py) can only reach what server Lua
exposes — rule-checked unit actions and god-mode grants — these orders are the
genuine player decisions the engine accepts only as client packets: choosing a
city's production and build queue, rush-buying, selling, and setting research.
They go through the normal economy/RNG pipeline, so a branch driven by them
realises a position's value faithfully rather than through cheats.

A ClientPolicy maps turn -> [Order]; civharness.branch.client_branch() plays
each turn's orders at that turn's phase, then ends the phase. Resuming a save
made at turn T, the first controllable phase is turn T, so orders keyed at T
do execute — unlike the Lua policy, whose first turn_begin is T+1.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class BuildUnit:
    """Set a city to build a unit (PACKET_CITY_CHANGE, VUT_UTYPE)."""

    action_key: ClassVar[str] = "build_unit"
    city_id: int
    unit: str  # rule name, e.g. "Warriors"

    def apply(self, client):
        client.build_unit(self.city_id, self.unit)


@dataclass(frozen=True)
class BuildImprovement:
    """Set a city to build a building/wonder (PACKET_CITY_CHANGE, VUT_IMPROVEMENT)."""

    action_key: ClassVar[str] = "build_improvement"
    city_id: int
    improvement: str  # rule name, e.g. "Barracks"

    def apply(self, client):
        client.build_improvement(self.city_id, self.improvement)


@dataclass(frozen=True)
class SetWorklist:
    """Set a city's build queue (PACKET_CITY_WORKLIST)."""

    action_key: ClassVar[str] = "set_worklist"
    city_id: int
    entries: tuple  # (("unit"|"building", rule_name), ...)

    def apply(self, client):
        client.set_worklist(self.city_id, list(self.entries))


@dataclass(frozen=True)
class Buy:
    """Rush-buy a city's current production (PACKET_CITY_BUY)."""

    action_key: ClassVar[str] = "buy"
    city_id: int

    def apply(self, client):
        client.buy(self.city_id)


@dataclass(frozen=True)
class Sell:
    """Sell one building from a city (PACKET_CITY_SELL)."""

    action_key: ClassVar[str] = "sell"
    city_id: int
    improvement: str

    def apply(self, client):
        client.sell(self.city_id, self.improvement)


@dataclass(frozen=True)
class ChangeSpecialist:
    """Reassign a citizen specialist (PACKET_CITY_CHANGE_SPECIALIST)."""

    action_key: ClassVar[str] = "change_specialist"
    city_id: int
    from_specialist: int
    to_specialist: int

    def apply(self, client):
        client.change_specialist(self.city_id, self.from_specialist, self.to_specialist)


@dataclass(frozen=True)
class SetResearch:
    """Set the player's current research (PACKET_PLAYER_RESEARCH)."""

    action_key: ClassVar[str] = "set_research"
    tech: str  # rule name, e.g. "Bronze Working"

    def apply(self, client):
        client.set_research(self.tech)


@dataclass(frozen=True)
class SetTechGoal:
    """Set the player's long-term research goal (PACKET_PLAYER_TECH_GOAL)."""

    action_key: ClassVar[str] = "set_tech_goal"
    tech: str

    def apply(self, client):
        client.set_tech_goal(self.tech)


@dataclass(frozen=True)
class UnitOrders:
    """Low-level unit order sequence (PACKET_UNIT_ORDERS): goto / activities.
    `orders` is a list of dicts (order, activity, target, sub_target, action,
    dir); tiles are packed indices y*xsize + x."""

    action_key: ClassVar[str] = "unit_orders"
    unit_id: int
    src_tile: int
    orders: tuple
    dest_tile: int
    repeat: bool = False
    vigilant: bool = False

    def apply(self, client):
        client.unit_orders(
            self.unit_id,
            self.src_tile,
            [dict(o) for o in self.orders],
            self.dest_tile,
            self.repeat,
            self.vigilant,
        )


@dataclass(frozen=True)
class SetRates:
    """Set tax/luxury/science percents (PACKET_PLAYER_RATES)."""

    action_key: ClassVar[str] = "set_rates"
    tax: int
    luxury: int
    science: int

    def apply(self, client):
        client.set_rates(self.tax, self.luxury, self.science)


@dataclass(frozen=True)
class ChangeGovernment:
    """Switch government (PACKET_PLAYER_CHANGE_GOVERNMENT)."""

    action_key: ClassVar[str] = "change_government"
    government: str  # rule name, e.g. "Monarchy"

    def apply(self, client):
        client.change_government(self.government)


@dataclass(frozen=True)
class MakeWorker:
    """Put a citizen to work a tile (PACKET_CITY_MAKE_WORKER). `tile` is the
    packed index y*xsize + x (xsize from the observation's terrain grid)."""

    action_key: ClassVar[str] = "make_worker"
    city_id: int
    tile: int

    def apply(self, client):
        client.make_worker(self.city_id, self.tile)


@dataclass(frozen=True)
class MakeSpecialist:
    """Pull the worker off a tile into a specialist (PACKET_CITY_MAKE_SPECIALIST)."""

    action_key: ClassVar[str] = "make_specialist"
    city_id: int
    tile: int

    def apply(self, client):
        client.make_specialist(self.city_id, self.tile)


@dataclass(frozen=True)
class DeclareWar:
    """Break the pact with another player toward war (PACKET_DIPLOMACY_CANCEL_PACT,
    sent `steps` times — peace -> war usually needs two)."""

    action_key: ClassVar[str] = "declare_war"
    player_id: int
    steps: int = 2

    def apply(self, client):
        client.declare_war(self.player_id, self.steps)


@dataclass(frozen=True)
class DoAction:
    """Perform one unit action by rule name (PACKET_UNIT_DO_ACTION) — the
    executable half of legal-action discovery (obs.legal_actions / the action
    names from get_unit_actions). `action` is a rule name, e.g. "Fortify",
    "Disband Unit", "Build Irrigation". `target_id` defaults to the unit's own id
    (self/unit actions); pass a packed tile index y*xsize+x for tile actions or a
    city id for city actions. `sub_target` is the extra/building id an action may
    need (0 = none)."""

    action_key: ClassVar[str] = "do_action"
    unit_id: int
    action: str
    target_id: int | None = None
    sub_target: int = 0
    name: str = ""

    def apply(self, client):
        client.do_action(
            self.unit_id, self.action, self.target_id, self.sub_target, self.name
        )


@dataclass(frozen=True)
class ClientPolicy:
    """orders: {turn: [Order, ...]} — issued at that turn's phase over the
    controlling connection."""

    orders: dict

    def on_phase(self, client, turn):
        for order in self.orders.get(turn, ()):
            order.apply(client)
