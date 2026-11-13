# Training a Freeciv agent

CivMarshmallow trains Qwen3-8B to play Freeciv with [slime](https://github.com/THUDM/slime). One episode is one
game from a start position to turn 120. Every decision (the policy sees its player-visible observation and an
action menu, and replies with its choice) is its own training sample, and the sibling episodes played from one
start form a group.

## Rewards

| `REWARD` | Signal |
|---|---|
| `sparse` | final score, whitened over the sibling episodes of each start (GRPO) |
| `scoreboard` | potential Φ = current in-game score; per-decision GAE |
| `civtelescope` | potential Φ = CivTelescope against a reference set; per-decision GAE |
| `scoreboard_to_civtelescope` | sigmoid hand-off from the scoreboard to CivTelescope as the game advances |
| `sparse_to_civtelescope` | sigmoid hand-off from the sparse reward to CivTelescope |
| `civtelescope_to_sparse` | sigmoid hand-off from CivTelescope to the sparse reward |
| `scoreboard_to_civtelescope_to_sparse` | two hand-offs, three signals |
| `terminal_civtelescope` | episode reward = CivTelescope value of the final position (control) |
| `value_head` | potential Φ = a regression value head placed on the reference set (see [other dense rewards](#other-dense-rewards)) |

**Reference-set potential.** CivTelescope compares a decision's position with M reference positions from the
nearest turn bucket. The mean win probability w is mapped to final-score units by interpolating the reference
positions' sorted final scores at quantile w, so Φ and the final score are commensurate inside GAE. Every
`REFRESH_EVERY` steps the reference set is rebuilt from the policy's own recent episodes and kept only if its
values still rank final scores well on held-out episodes.

**Hybrid rewards** blend signals with w(t) = sigmoid((t − c) / s) on the decision turn t (c = 35, s = 5 for the
first hand-off, c2 = 80 for the second). The scoreboard-to-CivTelescope blend happens in value space; blends
with the sparse reward happen in advantage space, since the sparse reward has no per-state value.

## Phases

| Phase | Start turn | Turns played |
|---|---|---|
| `rem20` | 100 | 20 |
| `rem40` | 80 | 40 |
| `rem80` | 40 | 80 |
| `rem120` | 1 | 120 |

`scripts/run_curriculum.sh` runs the phases in this order, each starting from the previous phase's final
checkpoint. `sparse`, `scoreboard` and `civtelescope` run all four phases from the base model; the hybrid
rewards run `rem120` only, from the `civtelescope` `rem80` checkpoint of the same seed.

Trained checkpoints are released as [`CivMarshmallow/CivMarshmallow-Qwen3-8B`](https://huggingface.co/CivMarshmallow/CivMarshmallow-Qwen3-8B)
(CivTelescope and hybrid rewards) and [`CivMarshmallow/Baselines-Qwen3-8B`](https://huggingface.co/CivMarshmallow/Baselines-Qwen3-8B)
(sparse and scoreboard rewards), organized as `<reward>/<phase>/seed<seed>/`.

## Data: start banks and reference sets

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
```

## Train

```bash
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

A single phase is `bash scripts/train_policy.sh REWARD=<reward> PHASE=<phase> SEED=<seed> [KEY=VALUE ...]`;
its header lists every path, knob and default. `train_policy.sh` starts a CivTelescope server for the rewards
that need one; to share a running server, start it with `scripts/serve_civtelescope.sh` and pass
`CIVTELESCOPE_URL`. Run directories are `runs/policy/<reward>/seed<seed>/<phase>`, and the final checkpoint
of a phase is `hf_iter11`. Each phase needs one node with eight 80 GB GPUs; the CivTelescope server shares four
of them.

Each run directory also keeps the logs the analyses read: `reward_log.jsonl`, `decision_log.jsonl`,
`value_log.jsonl` (per-decision Φ), `value_harvest.jsonl` (positions with their final scores) and
`refresh_log.jsonl`.

## Other dense rewards

The same reference-set ruler can host other dense signals, so they compete with CivTelescope on equal terms:
same reference positions, same map into final-score units, same GAE.

**Regression value head** (`REWARD=value_head`). The value head from
[CivTelescope comparisons](civtelescope.md#comparisons-and-ablations) scores each position on its own; a
decision's value is the fractional rank of its score among the reference positions' scores, read off their
sorted final scores. Serve one replica per GPU, then pass the endpoints:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4 + i)) python scripts/serve_value_head.py \
      --base-model /models/Qwen2.5-7B-Instruct --checkpoint ckpts/value_head.pt --port 3088$i &
done
bash scripts/train_policy.sh REWARD=value_head PHASE=rem80 SEED=43 INIT_CHECKPOINT=<rem40 checkpoint> \
    VALUE_HEAD_URLS=http://127.0.0.1:30880/value,http://127.0.0.1:30881/value,http://127.0.0.1:30882/value,http://127.0.0.1:30883/value
```

The launcher checks each replica's `/health` before it starts. The value-head reward always uses a static
reference set, because the refresh validates candidates with CivTelescope.

**Other pairwise models.** A model trained on a zero-shot model's preference labels instead of replay labels
(`train_civtelescope.py --train-pairs data/label_source/preference_labeled.jsonl`, then `merge_lora.py`), or
the untrained backbone as a zero-shot scorer, plugs into the `civtelescope` reward. Use a static reference set
so only the scorer differs:

```bash
bash scripts/train_policy.sh REWARD=civtelescope PHASE=rem80 SEED=43 REFRESH_EVERY=0 \
    CIVTELESCOPE_HF=models/preference_labeled OUT_DIR=runs/policy/preference/seed43/rem80
bash scripts/train_policy.sh REWARD=civtelescope PHASE=rem80 SEED=43 REFRESH_EVERY=0 \
    CIVTELESCOPE_HF=/models/Qwen2.5-7B-Instruct OUT_DIR=runs/policy/zero_shot/seed43/rem80
```

## Launcher knobs

Besides paths and cluster settings, `scripts/train_policy.sh` exposes the knobs below; its header lists the
rest.

| Knob | Default | Effect |
|---|---|---|
| `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES` | 12, 8, 8 | iterations per phase, starts per iteration, episodes per start |
| `LR` | 1e-6 | learning rate |
| `GAE_LAMBDA` | 0.9 | GAE λ for the potential-based rewards |
| `ERR_PENALTY` | 1.0 | reward penalty of an invalid reply |
| `REFRESH_EVERY` | 4 | reference-set refresh interval in steps; `0` keeps the reference set static |
| `REFSET_IDX` | per phase | which reference positions of each turn bucket are compared |
| `HYBRID_C2`, `HYBRID_S2` | unset, 5 | `scoreboard_to_civtelescope` only: CivTelescope's weight falls back to the scoreboard after turn c2, w(t) = sigmoid((t − c) / s) · (1 − sigmoid((t − c2) / s2)) |
| `VALUE_HEAD_URLS`, `VALUE_HEAD_BATCH` | unset, 16 | `value_head` only: replica endpoints and positions per request |
| `EVAL` | 0 | `1`: rollout-only evaluation on one GPU (used by `scripts/eval_policy.sh`) |

## Evaluate

Each checkpoint plays eight games from every start of the phase's evaluation bank. A phase gain compares the
checkpoint a phase ends with against the one it starts from, on the same starts:

```bash
bash scripts/eval_policy.sh CHECKPOINT=$BASE_MODEL PHASE=rem120 OUT_DIR=runs/eval/base/rem120
bash scripts/eval_policy.sh CHECKPOINT=runs/policy/civtelescope/seed43/rem80/hf_iter11 \
    PHASE=rem80 OUT_DIR=runs/eval/civtelescope/seed43/rem80/end
# ... one cell per (reward, seed, phase) and start/end checkpoint
python scripts/summarize_policy_eval.py summary --cells cells.json \
    --base-cell runs/eval/base/rem120/cell.json --out results/phase_gains.json
```

`cells.json` maps `{reward: {seed: {phase: {"start": cell.json, "end": cell.json}}}}`; the summary reports the
terminal score, the phase gain, paired win rates and invalid-action rates. Evaluation needs one GPU per
checkpoint.

## Prompting baselines

Prompting scaffolds around the untrained model, evaluated on the same starts:

```bash
MODEL=$BASE_MODEL STARTS=$DATA_ROOT/starts/test_rem120.jsonl DATA_ROOT=$DATA_ROOT \
OUT_DIR=runs/prompting_baselines GPUS=0,1,2,3 bash scripts/run_prompting_baselines.sh
python scripts/summarize_prompting_baselines.py runs/prompting_baselines \
    --base-cell runs/eval/base/rem120/cell.json --out results/prompting_baselines.json
```

## Offline diagnostics

```bash
# per turn bucket: error and rank correlation of the per-decision values against the final score (rem120 runs)
python scripts/offline_diagnostic.py spectrum \
    --run civtelescope=runs/policy/civtelescope/seed43/rem120 --run scoreboard=runs/policy/scoreboard/seed43/rem120
# variance of the final score that the potential explains beyond the current score
python scripts/offline_diagnostic.py delta-r2 --run civtelescope=runs/policy/civtelescope/seed43/rem120

# training-score slope of the terminal-only control against the per-decision rewards
python scripts/offline_diagnostic.py slope \
    --run terminal_civtelescope=runs/policy/terminal_civtelescope/seed43/rem40 \
    --run sparse=runs/policy/sparse/seed43/rem40
```

The same-start trap rate over the policy's own games is in [scoreboard traps](scoreboard-traps.md#traps-in-the-policys-own-games).

## Code

| Module | Content |
|---|---|
| `civmarsh.env` | the observation rendering and action menu shown to the policy |
| `civmarsh.rewards` | sparse, scoreboard, CivTelescope and hybrid potentials, reference sets and their refresh, GAE |
| `civmarsh.train` | slime hooks: episode rollout, group reward, reward post-processing, advantages |
| `civmarsh.eval` | evaluation cells and their aggregation, offline diagnostics |
| `civmarsh.baselines` | prompting-scaffold agents |
