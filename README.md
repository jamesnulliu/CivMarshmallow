# Rome Wasn’t Built in One Turn: Replay-Grounded Reward Signals for Long-Horizon Freeciv Agents

Code for **CivHarness**, a save/branch/replay harness for Freeciv; **CivTelescope**, a pairwise LLM judge
trained on replayed future outcomes; and **CivMarshmallow**, a phased RL recipe that trains a Freeciv agent
with CivTelescope as a dense reward.

![Overview of CivHarness, CivTelescope, and CivMarshmallow](docs/figures/overview.png)

**Overview of CivHarness, CivTelescope, and CivMarshmallow.**
CivHarness branches saved Freeciv states and exchanges player-visible observations and actions with the agent.
CivTelescope, a pairwise judge trained on replay-labeled trap pairs, gives a potential Φ(s) through a reference set.
CivMarshmallow trains the policy in four phases with CivTelescope reward, and with a hybrid reward for full games.

## Abstract

Training language-model agents to play long-horizon strategy games is hard because deterministic feedback is
sparse and often arrives only at the end of a game.
In-game scores can offer dense signals, but they do not reliably predict future outcomes: in 8–68% of position
pairs across eight strategy games, the higher-scoring position leads to the worse outcome.
We term this phenomenon *scoreboard traps*.
In Freeciv, an open-source turn-based empire-building game where several players manage units, cities, and
diplomacy over hundreds of turns, we look into better reward signals for gameplay agents.
To facilitate this, we build CivHarness, an open system that saves current game states, branches on different
decisions, and replays the game to future turns to provide measurable long-horizon feedback from saved positions.
We use CivHarness to train CivTelescope, an LLM Judge that predicts which one of two game positions will yield
the better outcome.
On scoreboard-trap pairs, CivTelescope is 1.3–2.0 times more accurate than off-the-shelf language models, and it
generalizes to new game starts, later turns, and unseen rules.
We then train a Freeciv gameplay agent with CivTelescope as a dense reward, and call this recipe CivMarshmallow.
When used in the middle stage of the game where scoreboard traps are common, CivMarshmallow reaches an average
training phase gain of 8.94 points, 77% more than the best baseline, the end-of-game reward.
Over the full 120-turn game, CivMarshmallow uses a hybrid reward that starts from the scoreboard or the
end-of-game reward and shifts to CivTelescope as the game advances; this hybrid reward scores highest among the
rewards we compare.

## Quick start

### 1. Install

CivHarness drives a headless freeciv-server 3.2.5. The script downloads the pinned release, checks its SHA-256
and builds it into `$HOME/opt/freeciv-3.2.5` (override with `PREFIX=...`):

```bash
git clone https://github.com/jamesnulliu/CivMarshmallow.git
cd CivMarshmallow
bash scripts/install_freeciv_server.sh
export CIVHARNESS_SERVER=$HOME/opt/freeciv-3.2.5/bin/freeciv-server
```

With [uv](https://docs.astral.sh/uv/). `uv.toml` restricts uv to its own managed Python builds, so the system
Python is never used:

```bash
uv sync --extra civtelescope --extra games
uv run python -c "import civharness, civmarsh"
```

With pip (Python 3.10 or newer):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[civtelescope,games]"
pip install --group dev        # pytest and ruff (pip >= 25.1)
```

Extras: `civtelescope` (PyTorch, Transformers for training and scoring CivTelescope and the value head),
`games` (python-chess, OpenSpiel, Catanatron for the cross-game trap banks). `uv sync` also installs the `dev`
group (pytest, ruff). If the default PyTorch wheel does not match your accelerator, install the matching PyTorch
build first.

RL training additionally needs [slime](https://github.com/THUDM/slime) at a pinned commit with a small patch.
Run the setup inside the slime Docker image, which ships Megatron-LM, sglang and the CUDA stack:

```bash
bash scripts/slime/setup_slime.sh third_party/slime
```

Format and lint with ruff: `ruff format .` and `ruff check .`

### 2. Configure

| Variable | Used by | Content |
|---|---|---|
| `CIVHARNESS_SERVER` | everything that plays Freeciv | freeciv-server binary (default `$HOME/opt/freeciv-3.2.5/bin/freeciv-server`) |
| `CIVMARSH_API_BASE`, `CIVMARSH_API_KEY` | zero-shot baselines, preference labels | OpenAI-compatible endpoint (e.g. `https://api.openai.com/v1`) and key |
| `STOCKFISH_BIN` | chess trap bank | Stockfish binary |
| `CIVMARSH_RULESET_DIR` | action menu | Freeciv ruleset directory (default: `civ2civ3` of the server install) |
| `SLIME_DIR`, `MEGATRON_DIR`, `BASE_MODEL`, `DATA_ROOT`, `CIVTELESCOPE_HF` | RL training and evaluation | patched slime, Megatron-LM, Qwen3-8B weights, start banks and reference sets, merged CivTelescope model; see the header of `scripts/train_policy.sh` |
| `WANDB_PROJECT`, `WANDB_API_KEY` | RL training | optional; logging is skipped without them |

Scripts write their data and results under the paths you pass (`data/`, `results/` and `runs/` in the examples
below, all git-ignored).

### 3. Check the harness

The acceptance suite checks replica identity, deterministic re-seeded branching, faithful client actions,
restricted observation and action spaces, and complete game replay. Tests that need the server are skipped
when `CIVHARNESS_SERVER` does not point to a binary:

```bash
uv run pytest tests/civharness/test_determinism.py    # byte-identical replay from a mid-game save
uv run pytest                                         # everything
```

### 4. Run a first experiment (CPU, no server)

A small Othello bank (self-play games, snapshots, replay-oracle labels) and its scoreboard-trap rate, overall
and by game progress, in under a minute:

```bash
uv run python scripts/build_game_bank.py --game othello --n-games 6 --replays 4 --out results/othello_bank.jsonl
uv run python scripts/cross_game_trap_rates.py --bank othello=results/othello_bank.jsonl \
    --min-bin-pairs 1 --out results/othello_trap_rates.json
```

### 5. Run the experiments

Every step is a script in `scripts/` (`python scripts/<name>.py --help` lists its options). Run the stages in
this order:

| Stage | Entry points | Paper |
|---|---|---|
| Scoreboard traps in Freeciv | `build_freeciv_bank.py`, `freeciv_trap_rates.py`, `game_length_curve.py` | Section 3, Appendices A.2–A.3 |
| Scoreboard traps in other games | `build_game_bank.py`, `measure_game_lengths.py`, `cross_game_trap_rates.py`, `eval_game_pairs.py` | Section 3, Appendix A.2 |
| CivTelescope | `render_fog_positions.py`, `build_rollout_positions.py`, `build_civtelescope_pairs.py`, `train_civtelescope.py`, `merge_lora.py`, `eval_civtelescope.py` | Sections 5.1–5.2, 6.2, Appendix C.1 |
| Zero-shot LLM judges | `eval_llm_baselines.py`, `report_llm_baselines.py` | Sections 3, 6.2, Appendix D.2 |
| Ruleset transfer | `build_transfer_pairs.py`, `eval_civtelescope.py`, `report_civtelescope.py` | Section 6.2, Appendix D.4 |
| Judge comparisons and ablations | `train_value_head.py`, `eval_value_head.py`, `label_preferences.py`, `build_label_source_pairs.py`, `input_ablation.py` | Sections 6.2, 6.5, Appendices D.3, D.7 |
| CivMarshmallow training | `build_start_banks.py`, `build_reference_sets.py`, `serve_civtelescope.sh`, `train_policy.sh`, `run_curriculum.sh` | Sections 5.3, 6.3–6.5, Appendix C |
| Policy evaluation | `eval_policy.sh`, `summarize_policy_eval.py` | Sections 6.1, 6.3–6.5, Appendix D.1 |
| Prompting baselines | `run_prompting_baselines.sh`, `summarize_prompting_baselines.py` | Section 6.4, Appendix D.6 |
| Diagnostics | `same_start_traps.py`, `offline_diagnostic.py` | Appendices A.4, D.5, D.8 |

The full command sequence is below. Freeciv banks run many game servers in parallel on CPU cores.
CivTelescope training and scoring need one GPU with 80 GB of memory; each RL phase needs one node with eight
80 GB GPUs (the CivTelescope server shares four of them); policy evaluation needs one GPU per checkpoint.

## Reproducing the paper

### Scoreboard traps in Freeciv (Section 3, Appendices A.2–A.3)

```bash
# 120-turn games under both rulesets, positions every 5 turns from turn 10 to 115, K = 8 replays each
for p in civ2civ3_120 classic_120; do
  python scripts/build_freeciv_bank.py --preset $p --work-dir runs/banks/$p --out-dir data/banks/$p --workers 32
done
python scripts/freeciv_trap_rates.py \
    --labels freeciv_civ2civ3=data/banks/civ2civ3_120/labels.jsonl \
    --labels freeciv_classic=data/banks/classic_120/labels.jsonl \
    --out results/freeciv_trap_rates.json

# game-length curve: 60-, 70- and 80-turn games
for n in 60 70 80; do
  python scripts/build_freeciv_bank.py --preset game_length_$n \
      --work-dir runs/banks/game_length_$n --out-dir data/banks/game_length_$n
done
python scripts/game_length_curve.py \
    --labels 60=data/banks/game_length_60/labels.jsonl \
    --labels 70=data/banks/game_length_70/labels.jsonl \
    --labels 80=data/banks/game_length_80/labels.jsonl \
    --out results/game_length_curve.json
```

### Scoreboard traps in other games (Section 3, Appendix A.2)

```bash
GAMES="chess othello backgammon go9 hearts 2048 catan"
for g in $GAMES; do                                  # chess needs STOCKFISH_BIN
  python scripts/build_game_bank.py --game $g --out data/games/${g}_bank.jsonl
done
python scripts/measure_game_lengths.py --out data/games/lengths.json
python scripts/cross_game_trap_rates.py $(for g in $GAMES; do echo --bank $g=data/games/${g}_bank.jsonl; done) \
    --lengths data/games/lengths.json --out results/cross_game_trap_rates.json

# zero-shot judges on chess and Othello pairs
for g in chess othello; do
  python scripts/eval_game_pairs.py --game $g --bank data/games/${g}_bank.jsonl \
      --models MODEL_A,MODEL_B --reasoning-effort none --out results/${g}_zero_shot.json
done
```

More snapshots of new games (`build_game_bank.py --game-offset ... --snaps ...`) can be added to the progress
bins with `cross_game_trap_rates.py --progress-bank`.

### CivTelescope (Sections 5.1–5.2, 6.2, Appendix C.1)

```bash
BASE=/models/Qwen2.5-7B-Instruct

# replay bank: 70-turn games, positions at turns 10-65, rendered from the focal player's fog-of-war view
python scripts/build_freeciv_bank.py --preset short_game \
    --work-dir runs/banks/short_game --out-dir data/banks/short_game
python scripts/render_fog_positions.py --positions data/banks/short_game/labels.jsonl \
    --pool-dir runs/banks/short_game/pool --rec-dir runs/fog/short_game \
    --out data/banks/short_game/rendered_fog.jsonl --endturn 70

# policy-rollout positions from the value harvests of RL runs; scoreboard-reward runs
# (see CivMarshmallow training) write one without a CivTelescope server
python scripts/build_rollout_positions.py --out-dir data/rollout \
    --run RUN=runs/policy/RUN/value_harvest.jsonl \
    --window RUN:rollout_early:FIRST-LAST --window RUN:rollout_late:FIRST-LAST

# training pairs and held-out slices (600 natural, 600 trap)
python scripts/build_civtelescope_pairs.py --out data/civtelescope \
    --replay-train data/banks/short_game/rendered_fog.jsonl \
    --replay-eval data/banks/short_game/rendered_fog.jsonl \
    --replay-labels data/banks/short_game/labels.jsonl --rollout-dir data/rollout

# LoRA fine-tuning, merge for serving, held-out evaluation (CivTelescope and its backbone)
python scripts/train_civtelescope.py --base-model $BASE --data-dir data/civtelescope \
    --out ckpts/civtelescope --monitor ckpts/civtelescope/monitor.jsonl
python scripts/merge_lora.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt --out models/civtelescope
python scripts/eval_civtelescope.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt \
    --data-dir data/civtelescope --out results/heldout.raw.jsonl --readout results/heldout.json
```

### Zero-shot LLM judges (Sections 3, 6.2, Appendix D.2)

```bash
for m in MODEL_A MODEL_B; do
  for w in "" --warned; do                           # neutral and trap-warned prompt
    python scripts/eval_llm_baselines.py --model $m --data-dir data/civtelescope \
        --n-trap 300 --n-natural 300 --subsample results/llm/subsample_pairs.jsonl \
        --reasoning-effort none --out-dir results/llm $w
  done
done
python scripts/report_llm_baselines.py --data-dir data/civtelescope \
    --subsample results/llm/subsample_pairs.jsonl --raw results/heldout.raw.jsonl \
    --picks MODEL_A=results/llm/MODEL_A_picks.jsonl \
    --picks MODEL_A-warned=results/llm/MODEL_A-warned_picks.jsonl \
    --out results/llm/comparison.json
```

### Ruleset transfer (Section 6.2, Appendix D.4)

```bash
python scripts/build_freeciv_bank.py --preset classic_transfer \
    --work-dir runs/banks/classic_transfer --out-dir data/banks/classic_transfer
python scripts/render_fog_positions.py --positions data/banks/classic_transfer/labels.jsonl \
    --pool-dir runs/banks/classic_transfer/pool --rec-dir runs/fog/classic_transfer \
    --out data/banks/classic_transfer/rendered_fog.jsonl --endturn 120
python scripts/build_transfer_pairs.py --labels data/banks/classic_transfer/labels.jsonl \
    --rendered data/banks/classic_transfer/rendered_fog.jsonl --ruleset classic --out-dir data/transfer
for i in 0 1 2 3; do                                 # one shard per GPU
  CUDA_VISIBLE_DEVICES=$i python scripts/eval_civtelescope.py --base-model $BASE \
      --adapters ckpts/civtelescope/lora.pt --pairs data/transfer/pairs.jsonl \
      --rendered data/banks/classic_transfer/rendered_fog.jsonl --ruleset classic \
      --shard $i/4 --out results/transfer.shard$i.raw.jsonl &
done; wait
python scripts/eval_llm_baselines.py --model MODEL_A --ruleset classic \
    --pairs data/transfer/pairs_api.jsonl --rendered data/banks/classic_transfer/rendered_fog.jsonl \
    --reasoning-effort none --out-dir results/transfer_llm
python scripts/report_civtelescope.py --pairs data/transfer/pairs.jsonl \
    --raw results/transfer.shard*.raw.jsonl --picks MODEL_A=results/transfer_llm/MODEL_A_picks.jsonl \
    --api-pairs data/transfer/pairs_api.jsonl --out results/transfer_report.json
```

### Judge comparisons and ablations (Sections 6.2, 6.5, Appendices D.3, D.7)

```bash
# regression value head on the same backbone
python scripts/train_value_head.py --base-model $BASE --data-dir data/civtelescope \
    --labels data/banks/short_game/labels.jsonl --out ckpts/value_head.pt --monitor ckpts/value_head_monitor.jsonl
python scripts/eval_value_head.py --base-model $BASE --checkpoint ckpts/value_head.pt \
    --data-dir data/civtelescope --labels data/banks/short_game/labels.jsonl \
    --reference-raw results/heldout.raw.jsonl --out results/value_head.raw.jsonl --readout results/value_head.json

# label source: replay labels vs a zero-shot model's preferences on the same pairs
python scripts/label_preferences.py --data-dir data/civtelescope --model MODEL_A --max N \
    --out results/preference_labels.jsonl
python scripts/build_label_source_pairs.py --data-dir data/civtelescope \
    --labels results/preference_labels.jsonl --out-dir data/label_source
for arm in replay preference; do
  python scripts/train_civtelescope.py --base-model $BASE --data-dir data/civtelescope \
      --train-pairs data/label_source/${arm}_labeled.jsonl --eval-interval EVAL_INTERVAL \
      --out ckpts/label_source_$arm --monitor ckpts/label_source_$arm/monitor.jsonl
done

# CivTelescope inputs: score masked, score only
for v in masked score_only; do
  python scripts/eval_civtelescope.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt \
      --data-dir data/civtelescope --input-variant $v \
      --out results/input_$v.raw.jsonl --readout results/input_$v.json
done
```

`EVAL_INTERVAL = ceil(2 * N / 16)` evaluates the label-source arms once per epoch at the default batch size.

### CivMarshmallow training (Sections 5.3, 6.3–6.5, Appendix C)

```bash
export DATA_ROOT=data/rl SLIME_DIR=third_party/slime MEGATRON_DIR=/path/to/Megatron-LM
export BASE_MODEL=/models/Qwen3-8B CIVTELESCOPE_HF=models/civtelescope

# start banks: training and evaluation games played to turn 120, starts at turns 100, 80, 40 and 1
python scripts/build_start_banks.py games --data-root $DATA_ROOT --split train --seeds 8500-8599
python scripts/build_start_banks.py games --data-root $DATA_ROOT --split test --seeds 8600-8699
python scripts/build_start_banks.py banks --data-root $DATA_ROOT

# reference sets: bot-played positions, then one set per phase; add policy positions with
# --harvest runs/policy/<reward>/seed<seed>/<phase>/value_harvest.jsonl (repeatable) once runs exist
python scripts/build_reference_sets.py games --data-root $DATA_ROOT --seeds 8000-8119 --tiles-per-player 200
python scripts/build_reference_sets.py games --data-root $DATA_ROOT --seeds 9000-9031 --tiles-per-player 300
python scripts/build_reference_sets.py bot --data-root $DATA_ROOT
for p in rem20 rem40 rem80 rem120; do
  python scripts/build_reference_sets.py phase --data-root $DATA_ROOT --phase $p
done

# four phases (rem20 -> rem40 -> rem80 -> rem120) per reward and seed
for r in sparse scoreboard civtelescope; do
  for s in 43 44 45; do bash scripts/run_curriculum.sh REWARD=$r SEED=$s; done
done
# hybrid rewards: rem120 only, from the civtelescope rem80 checkpoint of the same seed
for r in scoreboard_to_civtelescope sparse_to_civtelescope civtelescope_to_sparse scoreboard_to_civtelescope_to_sparse; do
  for s in 43 44 45; do bash scripts/run_curriculum.sh REWARD=$r SEED=$s; done
done
# terminal-only CivTelescope control (rem40, rem80)
for s in 43 44; do bash scripts/run_curriculum.sh REWARD=terminal_civtelescope SEED=$s; done
```

`scripts/train_policy.sh` starts a CivTelescope server for the rewards that need one; to share a running
server, start it with `scripts/serve_civtelescope.sh` and pass `CIVTELESCOPE_URL`. Run directories are
`runs/policy/<reward>/seed<seed>/<phase>`; the final checkpoint of a phase is `hf_iter11`.

### Policy evaluation (Sections 6.1, 6.3–6.5, Appendix D.1)

Each checkpoint plays eight games from every start of the phase's evaluation bank; a phase gain compares the
checkpoint a phase ends with against the one it starts from on the same starts:

```bash
bash scripts/eval_policy.sh CHECKPOINT=$BASE_MODEL PHASE=rem120 OUT_DIR=runs/eval/base/rem120
bash scripts/eval_policy.sh CHECKPOINT=runs/policy/civtelescope/seed43/rem80/hf_iter11 \
    PHASE=rem80 OUT_DIR=runs/eval/civtelescope/seed43/rem80/end
# ... one cell per (reward, seed, phase) and start/end checkpoint
python scripts/summarize_policy_eval.py summary --cells cells.json \
    --base-cell runs/eval/base/rem120/cell.json --out results/phase_gains.json
```

`cells.json` maps `{reward: {seed: {phase: {"start": cell.json, "end": cell.json}}}}`; the summary reports the
terminal score, the phase gain, paired win rates and invalid-action rates.

### Prompting baselines (Section 6.4, Appendix D.6)

```bash
MODEL=$BASE_MODEL STARTS=$DATA_ROOT/starts/test_rem120.jsonl DATA_ROOT=$DATA_ROOT \
OUT_DIR=runs/prompting_baselines GPUS=0,1,2,3 bash scripts/run_prompting_baselines.sh
python scripts/summarize_prompting_baselines.py runs/prompting_baselines \
    --base-cell runs/eval/base/rem120/cell.json --out results/prompting_baselines.json
```

### Diagnostics (Appendices A.4, D.5, D.8)

```bash
# same-start trap rate over the policy's own training games
python scripts/same_start_traps.py --out results/same_start_traps.json --inputs \
    $(for f in runs/policy/*/seed*/*/value_harvest.jsonl; do echo "$(dirname ${f#runs/policy/} | tr / _)=$f"; done)

# offline window diagnostic on rem120 runs
python scripts/offline_diagnostic.py spectrum \
    --run civtelescope=runs/policy/civtelescope/seed43/rem120 --run scoreboard=runs/policy/scoreboard/seed43/rem120
python scripts/offline_diagnostic.py delta-r2 --run civtelescope=runs/policy/civtelescope/seed43/rem120

# training-score slope of the terminal-only control against the per-decision rewards
python scripts/offline_diagnostic.py slope \
    --run terminal_civtelescope=runs/policy/terminal_civtelescope/seed43/rem40 \
    --run sparse=runs/policy/sparse/seed43/rem40
```

## Repository layout

```
src/civharness/          save/branch/replay harness: server lifecycle, branching, batch execution, network
                         client, player-visible observations, action spaces, episode logs and replay
src/civmarsh/
  oracle/                replay-oracle banks and decidable position pairs
  traps/                 scoreboard-trap rates in Freeciv and other games, same-start traps, game-length curve
  civtelescope/          fog rendering, pair building, LoRA training, evaluation, zero-shot baselines,
                         value head, ablations, serving client
  env/                   observation rendering and action menu shown to the policy
  rewards/               sparse, scoreboard, CivTelescope and hybrid rewards, reference sets, GAE
  train/                 slime rollout, reward and advantage hooks
  eval/                  policy evaluation cells and aggregation, offline diagnostics
  baselines/             prompting-scaffold agents
  utils/                 I/O, statistics, API client
scripts/                 entry points; scripts/slime/ pins and patches slime
tests/                   harness acceptance suite and unit tests
```

## License

Apache-2.0.
