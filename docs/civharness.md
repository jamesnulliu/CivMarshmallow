# CivHarness

CivHarness wraps a headless freeciv-server 3.2.5 and turns it into a research environment built on three
operations:

1. **Save** a game position to a file (the server writes the save itself).
2. **Branch**: load the save into fresh server processes and play it forward many times under controlled
   conditions (different random seeds, AI levels, or an external policy).
3. **Replay**: the save stores the full random-number-generator state, so the same save with the same seed
   reproduces the game byte for byte.

Determinism is what makes replayed outcomes usable as labels: every effect measured between two branches
comes with a noise floor from same-position, different-seed replicas.

The package is stdlib-only (PyYAML for YAML environment specs). Python code below runs from the repository
root with `CIVHARNESS_SERVER` set (see [installation](installation.md)).

## Run a game

```python
from civharness import GameConfig, run_game

result = run_game(
    GameConfig(aifill=4, skill="hard", endturn=100, mapseed=42, gameseed=42),
    "work/game1",
)
print(result.turns, result.winners, result.scores)
```

`run_game` returns a `GameResult` with provenance (binary digest, config hash, seeds) and a normalized save
digest; the same config gives the same `result.identity()`.

## Snapshot and branch

```python
from civharness import branch, snapshot_series

# one seeded game, autosaved every 10 turns: a library of positions
refs = snapshot_series(
    GameConfig(aifill=3, endturn=100, mapseed=3, gameseed=3), "work/lib", every=10
)
mid = next(r for r in refs if r.turn >= 50)

# 16 futures of that position with distinct seeds, 30 turns forward
trajs = branch(mid, "work/br", until=mid.turn + 30, k=16)
print(trajs[0].scores)  # scores at the endpoint
print(trajs[0].series("settlers", 0))  # per-turn settler curve of player 0
```

- `k=16` uses seeds `1..16`; `seeds=[...]` names them explicitly.
- `seeds=None, k=None` replays the stored random stream: the noise-free reference.
- `skill="hard"` sets the AI level of every player; `skill_by_player={"Frederick": "hard"}` overrides single
  players. Read the names with `players_of(save)`.

`noise_floor(mid, "work/floor", until=mid.turn + 30)` checks that same-seed replicas are identical and reports
the score spread over seeds. `position(save)` returns a structured summary (players, cities, units, techs) for
rendering and features.

## External control

There are two ways for an outside agent to act inside a branch.

**Lua policy** (`branch(..., policy=ScriptedPolicy(...))`): rule-checked unit actions and god-mode grants,
compiled to a Lua script the server runs. Orders keyed at turn T of a save made at turn T never run; key them
at T+1 or later.

**Client policy** (`client_branch`): a real network client takes the focal player and sends the packets a
human client would send, so construction, research, rates, government, diplomacy and unit actions are genuine
decisions the engine checks. The game advances only when the client ends its phase, so the branch stays
deterministic.

```python
from civharness import (
    BuildImprovement,
    ClientPolicy,
    SetResearch,
    client_branch,
    players_of,
    position,
)

focal = players_of(mid)[0]
city = position(mid).players[focal]["cities"][0]["id"]
policy = ClientPolicy(
    orders={
        mid.turn: [BuildImprovement(city, "Barracks"), SetResearch("Bronze Working")]
    }
)
trajs = client_branch(
    mid, "work/ctrl", until=mid.turn + 20, focal_player=focal, policy=policy
)
```

Orders take rule names and resolve them to ids: `BuildUnit`, `BuildImprovement`, `SetWorklist`, `Buy`, `Sell`,
`ChangeSpecialist`, `SetResearch`, `SetTechGoal`, `SetRates`, `ChangeGovernment`, `MakeWorker`,
`MakeSpecialist`, `DeclareWar`, `DoAction`, `UnitOrders`, `SetServerSideAgent` (2 = the engine's
auto-explore; in 3.2 the only way to start it), `ChangeActivity`.

## Agents

An agent is anything with `act(obs) -> [Order]`. `run_agent` drives one player; `run_agents` gives each named
player its own agent while every other seat stays AI, and same seeds with deterministic agents reproduce the
game byte for byte.

```python
from civharness import FunctionAgent, run_agent


def my_agent(obs):
    return [
        BuildImprovement(c["id"], "Barracks")
        for c in obs.my_cities
        if c["building"] != "Barracks"
    ]


trajs = run_agent(
    mid,
    "work/agent",
    focal_player=focal,
    agent=FunctionAgent(my_agent),
    until=mid.turn + 20,
)
```

- Multi-seed runs isolate agent state: a `fresh(...)` factory is rebuilt per seed, an instance with `reset()`
  is reset, and any other instance is deep-copied.
- Agents must issue legal orders: the engine rejects what the player cannot do, as it would for a human.
- Control loss fails loudly. A dropped client, a stalled seat or a missed deadline raises `ControlLost`
  instead of letting the server AI finish a game reported as agent-controlled.

## Environment specs

An `EnvSpec` declares the allowed actions and the visible observation fields in one YAML file, and the
drivers enforce it: a forbidden order raises `ForbiddenAction`, and disabled fields are absent from the
observation. Keys and values are validated, so a malformed spec never widens what the agent sees.

```python
from civharness import EnvSpec, run_agents

env = EnvSpec.from_yaml("src/civharness/configs/economy.yaml")
trajs = run_agents(
    mid, "work/econ", agents={focal: agent}, until=mid.turn + 20, env=env
)
```

| Config | Content |
|---|---|
| `full.yaml` | every action and a rich observation; the terrain grid and legal-action queries stay off (the default) |
| `economy.yaml`, `research_only.yaml`, `military.yaml` | action and observation subsets |
| `saga_visible.yaml` | player-visible fog of war: the map, cities and units come from what the server sent this player's connection |
| `saga_fullinfo.yaml` | the same layout with full information |

With `legal_actions: true`, `obs.legal_actions(unit_id, target_tile=...)` asks the engine what a unit may do,
and `DoAction` executes one action by rule name. An EnvSpec keeps an honest agent inside its subset; it is not
a security boundary against adversarial in-process code.

## Observation substrate

`build_scene_graph`, `build_bounded_digest`, `ObservationTools` and `discover_unit_action_candidates` turn an
observation into a scene graph (own and visible opponent cities and units with proximity, attack and threat
edges), a bounded text digest, city metrics and serializable legal-action candidates. CivMarshmallow renders
the policy's observation and CivTelescope's input from these (`civmarsh.env`).

## Episode logs and replay

Every agent run writes an episode log (`<branch>/episode.jsonl`): a header with seeds, horizon, agents, EnvSpec
and AI levels, then per turn and player the observation digest and the orders issued. Because the engine is
deterministic given the orders, `replay_episode` reproduces the run byte for byte, even for a sampling
language-model agent:

```python
from civharness import replay_episode, run_agents

trajs = run_agents(
    mid, "work/run", agents={focal: llm_agent}, until=mid.turn + 20, seeds=[7]
)
again = replay_episode(trajs[0].episode_log_path, mid, "work/replay")
assert again[0].save_sha256_normalized == trajs[0].save_sha256_normalized
```

Replay restores the recorded AI levels and EnvSpec, refuses a different source save or a log without its
completion footer, and checks every per-turn observation digest. Pass `until=` / `seeds=` to branch from a
recorded run instead.

## Other games

`civharness.games` runs Othello and Catan in-process under the same contract: `snapshot_series` builds a
position library, `branch` plays re-seeded engine continuations to the end (the replay oracle), and `play`
lets an external policy control the focal seat. See [other games](other-games.md).

## Tests

```bash
uv run pytest tests/civharness/test_determinism.py   # byte-identical replay from a mid-game save
uv run pytest tests/civharness                       # the full acceptance suite
```
