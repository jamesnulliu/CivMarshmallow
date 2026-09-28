"""CivHarness: a deterministic save/branch/replay harness for batch Freeciv.

Wraps the pinned `freeciv-server` 3.2.5 binary (installed by
scripts/install_freeciv_server.sh, located via CIVHARNESS_SERVER) with typed
game configs, process lifecycle, batch execution, a branch API whose readouts
carry their noise floor, a faithful network client that lets external agents
control players, and the observation/action layers those agents consume.
"""

from civharness.agent import (
    Agent,
    AgentFactory,
    FunctionAgent,
    MissingObservation,
    Observation,
    PolicyAgent,
    agent_noise_floor,
    fresh,
    run_agent,
    run_agents,
)
from civharness.batch import BatchRunner
from civharness.branch import (
    branch,
    client_branch,
    load_snapshot_series,
    noise_floor,
    players_of,
    position,
    reseed_save,
    snapshot_series,
)
from civharness.candidates import (
    ActionCandidate,
    CandidateDiscovery,
    SkippedAction,
    candidate_from_order,
    discover_unit_action_candidates,
)
from civharness.client import (
    ControlLost,
    FreecivClient,
    LegalActionsUnavailable,
    RuleIds,
)
from civharness.config import DEFAULT_BINARY, GameConfig
from civharness.envspec import EnvSpec, ForbiddenAction
from civharness.episode import (
    ReplayMismatch,
    order_from_record,
    read_episode,
    replay_episode,
)
from civharness.policy import CreateBuilding, MoveUnit, ScriptedPolicy, UnitAction
from civharness.policy.client import (
    BuildImprovement,
    BuildUnit,
    Buy,
    ChangeGovernment,
    ChangeSpecialist,
    ClientPolicy,
    DeclareWar,
    DoAction,
    MakeSpecialist,
    MakeWorker,
    Sell,
    SetRates,
    SetResearch,
    SetTechGoal,
    SetWorklist,
    UnitOrders,
)
from civharness.runner import GameResult, run_game
from civharness.spatial import (
    BoundedDigest,
    GoalProgress,
    ObservationTools,
    SceneEdge,
    SceneGraph,
    SceneGraphConfig,
    SceneNode,
    build_bounded_digest,
    build_scene_graph,
)

__all__ = [  # noqa: RUF022 -- grouped by topic
    "GameConfig",
    "DEFAULT_BINARY",
    "GameResult",
    "run_game",
    "BatchRunner",
    "branch",
    "client_branch",
    "noise_floor",
    "players_of",
    "snapshot_series",
    "load_snapshot_series",
    "position",
    "reseed_save",
    "ScriptedPolicy",
    "UnitAction",
    "MoveUnit",
    "CreateBuilding",
    # Client-driven construction, research and unit control:
    "FreecivClient",
    "RuleIds",
    "ClientPolicy",
    "BuildUnit",
    "BuildImprovement",
    "SetWorklist",
    "Buy",
    "Sell",
    "ChangeSpecialist",
    "SetResearch",
    "SetTechGoal",
    "UnitOrders",
    "SetRates",
    "ChangeGovernment",
    "MakeWorker",
    "MakeSpecialist",
    "DeclareWar",
    "DoAction",
    # Agent interface and multi-agent driver:
    "Observation",
    "Agent",
    "AgentFactory",
    "fresh",
    "FunctionAgent",
    "PolicyAgent",
    "run_agent",
    "run_agents",
    "agent_noise_floor",
    # Control-loss error signalling:
    "ControlLost",
    "LegalActionsUnavailable",
    "MissingObservation",
    # Episode logging and replay of agent runs:
    "replay_episode",
    "read_episode",
    "order_from_record",
    "ReplayMismatch",
    # Configurable observation/action space:
    "EnvSpec",
    "ForbiddenAction",
    # SAGA-aligned deterministic observation substrate:
    "SceneGraphConfig",
    "SceneNode",
    "SceneEdge",
    "SceneGraph",
    "GoalProgress",
    "BoundedDigest",
    "ObservationTools",
    "build_scene_graph",
    "build_bounded_digest",
    # Structured, serializable legal-action candidates:
    "ActionCandidate",
    "SkippedAction",
    "CandidateDiscovery",
    "candidate_from_order",
    "discover_unit_action_candidates",
]
