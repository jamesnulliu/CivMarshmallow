# Documentation

| Page | Content |
|---|---|
| [Installation](installation.md) | server, Python package (uv or pip), slime, environment variables, hardware |
| [CivHarness](civharness.md) | save / branch / replay, external control, agents, environment specs, episode replay |
| [Scoreboard traps](scoreboard-traps.md) | position banks and trap rates in Freeciv and other games |
| [CivTelescope](civtelescope.md) | training data, training, evaluation, baselines, ruleset transfer, ablations |
| [Training](training.md) | rewards, phases, start banks and reference sets, RL training, evaluation, baselines |
| [Other games](other-games.md) | the full pipeline on Othello and Catan |

Every script prints its options with `python scripts/<name>.py --help`; the shell launchers document their
variables in their headers. The commands in these pages write to `data/`, `results/`, `runs/`, `ckpts/` and
`models/`, which are git-ignored.

## Repository layout

```
src/civharness/          save / branch / replay harness: server lifecycle, batch runs, branching, network
                         client, agents, environment specs, player-visible observations, episode replay
  configs/               environment specs (EnvSpec YAML)
  games/                 Othello and Catan under the same contract
src/civmarsh/
  oracle/                replay-oracle banks and decidable position pairs
  traps/                 scoreboard-trap rates in Freeciv and other games, same-start traps, game-length curve
  civtelescope/          fog rendering, pair building, LoRA training, evaluation, zero-shot baselines,
                         value head, ablations, serving client
  env/                   observation rendering and action menu shown to the policy
  rewards/               sparse, scoreboard, CivTelescope, hybrid and value-head potentials, reference sets, GAE;
                         potentials for the other games
  train/                 slime hooks: Freeciv and in-process game rollouts, group reward, advantages
  eval/                  policy evaluation cells and aggregation, offline diagnostics
  baselines/             prompting-scaffold agents
  utils/                 I/O, statistics, API client
scripts/                 entry points for Freeciv; scripts/slime/ pins and patches slime
scripts/games/           entry points for the Othello and Catan pipeline
tests/                   harness acceptance suite and unit tests
docs/                    these pages
```
