#!/usr/bin/env bash
# Train (or evaluate) the policy on Othello or Catan with one reward and one
# seed, through the slime recipe of the Freeciv phases.  The games run
# in-process through civharness.games; no game server is started.
#
# Usage:
#   bash scripts/games/train_policy.sh GAME=<othello|catan> REWARD=<reward> SEED=<seed> [KEY=VALUE ...]
#   bash scripts/games/train_policy.sh GAME=<game> EVAL=1 LABEL=<cell name> INIT_CHECKPOINT=<hf dir> [KEY=VALUE ...]
#
# Every KEY=VALUE argument may equally be set in the environment; arguments win.
#
# Rewards (one training recipe; only the advantage differs):
#   sparse        final score (Othello: final disc differential; Catan: final
#                 victory points), GRPO-whitened per start position
#   scoreboard    potential = the visible proxy now (disc differential / public
#                 victory points), per-decision GAE
#   civtelescope  potential = the game's CivTelescope vs the reference set, same GAE
#   zero_shot     the civtelescope reward served by the UNTRAINED backbone: same
#                 pairwise prompt, same reference set, same GAE; only the
#                 weights differ (CIVTELESCOPE_BASE_MODEL)
#   mix           potential = w_t(t) * CivTelescope + w_s(t) * scoreboard, the
#                 leftover weight on the sparse baseline (civmarsh.rewards.games):
#                 MIX_J0 / MIX_J1 CivTelescope weight before / after the hand-off
#                 (0.5 / 0.5), MIX_C / MIX_S hand-off center / width in turns
#                 (0 / 1), MIX_SB share of the non-CivTelescope weight on the
#                 scoreboard (1); needs what civtelescope needs
#
# Required:
#   SLIME_DIR          patched slime checkout (scripts/slime/setup_slime.sh)
#   MEGATRON_DIR       Megatron-LM checkout, put on PYTHONPATH
#   BASE_MODEL         Qwen3-8B Hugging Face weights
#   civtelescope, mix: CIVTELESCOPE_HF (the game's merged CivTelescope,
#                      scripts/merge_lora.py) or CIVTELESCOPE_URL (/generate
#                      endpoint of a running server), and a reference set
#   zero_shot:         CIVTELESCOPE_BASE_MODEL (the untrained backbone, e.g.
#                      Qwen2.5-7B-Instruct) or CIVTELESCOPE_URL, and a reference set
#   GAME=catan:        catanatron in the rollout Python (the `games` extra; the
#                      version is pinned because Catan saves embed a pickled
#                      engine state)
# Optional:
#   GAME_DATA          per-game data directory (default data/games/<GAME>)
#   STARTS             start file (default GAME_DATA/starts/train.jsonl, or
#                      GAME_DATA/starts/eval.jsonl with EVAL=1;
#                      scripts/games/build_starts.py)
#   REFSET             reference set (default GAME_DATA/refset.jsonl;
#                      scripts/games/build_reference_set.py)
#   REFSET_IDX         reference positions used per turn bucket, yaml list
#                      (default [0, 2, 5, 7] of the 8 sorted by outcome)
#   INIT_CHECKPOINT    HF checkpoint the run starts from (default BASE_MODEL)
#   OUT_DIR            run directory (default runs/games/<GAME>/<REWARD>/seed<SEED>,
#                      or runs/games/<GAME>/eval/<LABEL> with EVAL=1)
#   CKPT_DIR           Megatron checkpoints (default OUT_DIR/ckpts)
#   NUM_ROLLOUT (12)  ROLLOUT_BATCH_SIZE (8)  GLOBAL_BATCH_SIZE (64)  N_SAMPLES (8)
#   LR (1e-6)  GAE_LAMBDA (0.9)  ERR_PENALTY (1.0)  SAVE_INTERVAL (4)
#   STALL_MAX_FRAC (0.30)  MIN_GROUP  MAX_TOKENS_PER_GPU (6144)
#   GAME_CHOICE_TOKENS (8)  max new tokens per decision
#   GAME_OPP_DEPTH (2)  GAME_OPP_TEMP (20.0)  GAME_OPP_EXACT_FROM (10; 0 = never)
#                      Othello: the opponent engine (scripts/games/calibrate_opponent.py
#                      picks one the untrained policy can beat)
#   GAME_MAX_TURNS     Catan: turn cap, a normal end (default: the engine's 1000)
#   VALUE_CHATML=1     serve the CivTelescope prompt inside the ChatML turn it was
#                      trained in (civmarsh.rewards.games); default: the raw prompt,
#                      as on Freeciv
#   CIVTELESCOPE_GPUS (4,5,6,7)  one data-parallel CivTelescope replica per GPU,
#                      sharing them with the trainer at CIVTELESCOPE_MEM_FRACTION (0.20)
#   CIVTELESCOPE_PORT (30870)
#   GPUS               GPUs given to Ray and the colocated actor (default 0,...,7;
#                      0 with EVAL=1)
#   TP_SIZE (4)  CP_SIZE (2)  tensor / context parallel sizes of the trainer
#   SGLANG_ATTENTION_BACKEND (fa3; flashinfer on GPUs without FlashAttention 3)
#   SGLANG_MEM_FRACTION (0.45 for the potential-based rewards, else 0.5)
#   MODEL_CONFIG       slime model-args script under scripts/models (default qwen3-8B)
#   RAY_ADDRESS        submit to this running Ray dashboard instead of starting a head
#   RAY_PORT (6379)  RAY_DASHBOARD_PORT (8265)  RAY_TEMP_DIR  MASTER_ADDR (127.0.0.1)
#   SLIME_ROLLOUT_BASE_PORT  first port of the rollout engines (default 15000)
#   WANDB_PROJECT      enables wandb logging (WANDB_GROUP, WANDB_API_KEY optional)
#   EVAL=1             rollout-only evaluation: every start once, N_SAMPLES games
#                      each, one GPU, no training step, no checkpoint, rollout seed
#                      42, invalid replies unpriced, a start kept if >= 1 game finished
#   DRY_RUN=1          print config.yaml and the slime command line; start and
#                      write nothing
#   EXTRA_ARGS         appended to the slime command line (last one wins)
#
# The final checkpoint of a training run is OUT_DIR/hf_iter<NUM_ROLLOUT - 1>.
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "unexpected argument '$arg' (expected KEY=VALUE)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
EVAL=${EVAL:-0}
DRY_RUN=${DRY_RUN:-0}
GAME=${GAME:-othello}
case "$GAME" in
  othello | catan) ;;
  *) echo "unknown GAME '$GAME' (othello | catan)" >&2; exit 2 ;;
esac
if [ "$EVAL" = 1 ]; then
  REWARD=sparse # rollout-only: no value function, no CivTelescope server
  SEED=${SEED:-42}
  LABEL=${LABEL:?EVAL=1 needs LABEL (the name of the evaluated cell)}
else
  REWARD=${REWARD:?set REWARD (sparse | scoreboard | civtelescope | zero_shot | mix)}
  SEED=${SEED:?set SEED}
fi
SLIME_DIR=${SLIME_DIR:?set SLIME_DIR (scripts/slime/setup_slime.sh)}
MEGATRON_DIR=${MEGATRON_DIR:?set MEGATRON_DIR}
BASE_MODEL=${BASE_MODEL:?set BASE_MODEL}

VALUE_FN=""
USES_TELESCOPE=0
case "$REWARD" in
  sparse) ;;
  scoreboard) VALUE_FN=civmarsh.rewards.games.score_now_value ;;
  civtelescope) VALUE_FN=civmarsh.rewards.games.telescope_value; USES_TELESCOPE=1 ;;
  mix) VALUE_FN=civmarsh.rewards.games.mix_value; USES_TELESCOPE=1 ;;
  zero_shot)
    VALUE_FN=civmarsh.rewards.games.telescope_value; USES_TELESCOPE=1
    # the only difference from civtelescope: the served weights.  A
    # CIVTELESCOPE_HF left in the environment is ignored on purpose.
    if [ -z "${CIVTELESCOPE_URL:-}" ]; then
      CIVTELESCOPE_HF=${CIVTELESCOPE_BASE_MODEL:?REWARD=zero_shot needs CIVTELESCOPE_BASE_MODEL (the untrained backbone)}
    fi
    ;;
  *) echo "unknown REWARD '$REWARD'" >&2; exit 2 ;;
esac

GAME_DATA=${GAME_DATA:-$REPO_ROOT/data/games/$GAME}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$BASE_MODEL}
if [ "$EVAL" = 1 ]; then
  STARTS=${STARTS:-$GAME_DATA/starts/eval.jsonl}
  OUT_DIR=${OUT_DIR:-$REPO_ROOT/runs/games/$GAME/eval/$LABEL}
  GPUS=${GPUS:-0}
  ERR_PENALTY=0
  MIN_GROUP=1
else
  STARTS=${STARTS:-$GAME_DATA/starts/train.jsonl}
  OUT_DIR=${OUT_DIR:-$REPO_ROOT/runs/games/$GAME/$REWARD/seed$SEED}
  GPUS=${GPUS:-0,1,2,3,4,5,6,7}
  ERR_PENALTY=${ERR_PENALTY:-1.0}
fi
NUM_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
CKPT_DIR=${CKPT_DIR:-$OUT_DIR/ckpts}
N_SAMPLES=${N_SAMPLES:-8}
LR=${LR:-1e-6}
GAE_LAMBDA=${GAE_LAMBDA:-0.9}
SAVE_INTERVAL=${SAVE_INTERVAL:-4}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144}
STALL_MAX_FRAC=${STALL_MAX_FRAC:-0.30}
GAME_CHOICE_TOKENS=${GAME_CHOICE_TOKENS:-8}
GAME_OPP_DEPTH=${GAME_OPP_DEPTH:-2}
GAME_OPP_TEMP=${GAME_OPP_TEMP:-20.0}
GAME_OPP_EXACT_FROM=${GAME_OPP_EXACT_FROM:-10}
TP_SIZE=${TP_SIZE:-4}
CP_SIZE=${CP_SIZE:-2}
MODEL_CONFIG=${MODEL_CONFIG:-qwen3-8B}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
CIVTELESCOPE_GPUS=${CIVTELESCOPE_GPUS:-4,5,6,7}
CIVTELESCOPE_PORT=${CIVTELESCOPE_PORT:-30870}
CIVTELESCOPE_MEM_FRACTION=${CIVTELESCOPE_MEM_FRACTION:-0.20}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}
if [ -n "$VALUE_FN" ]; then MEM_DEFAULT=0.45; else MEM_DEFAULT=0.5; fi
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-$MEM_DEFAULT}

# In a dry run a missing path is reported, not fatal: the point is to read the
# config and the command line on a machine without weights or GPUs.
need() {
  if [ ! -e "$2" ]; then
    echo "missing $1: $2" >&2
    if [ "$DRY_RUN" != 1 ]; then exit 1; fi
  fi
}
need BASE_MODEL "$BASE_MODEL"
need INIT_CHECKPOINT "$INIT_CHECKPOINT"
need STARTS "$STARTS"
need "slime model config" "$SLIME_DIR/scripts/models/$MODEL_CONFIG.sh"
if [ "$USES_TELESCOPE" = 1 ]; then
  REFSET=${REFSET:-$GAME_DATA/refset.jsonl}
  REFSET_IDX=${REFSET_IDX:-[0, 2, 5, 7]}
  need REFSET "$REFSET"
  if [ -z "${CIVTELESCOPE_URL:-}" ]; then
    CIVTELESCOPE_HF=${CIVTELESCOPE_HF:?REWARD=$REWARD needs CIVTELESCOPE_HF or CIVTELESCOPE_URL}
    need CIVTELESCOPE_HF "$CIVTELESCOPE_HF"
  fi
fi

if [ -s "$STARTS" ]; then N_STARTS=$(grep -c . "$STARTS"); else N_STARTS=0; fi
if [ "$EVAL" = 1 ]; then
  NUM_ROLLOUT=1
  ROLLOUT_BATCH_SIZE=$N_STARTS
  GLOBAL_BATCH_SIZE=$((N_STARTS * N_SAMPLES))
else
  NUM_ROLLOUT=${NUM_ROLLOUT:-12}
  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
fi

if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$OUT_DIR"
  if compgen -G "$OUT_DIR/hf_iter*" >/dev/null || [ -s "$OUT_DIR/reward_log.jsonl" ]; then
    echo "$OUT_DIR already holds a run; choose a fresh OUT_DIR" >&2
    exit 1
  fi
  # append-mode logs start empty: every launch is a fresh run
  for log in reward_log decision_log drop_log value_log value_harvest; do
    : >"$OUT_DIR/$log.jsonl"
  done
fi

# ---------------------------------------------------------------- cleanup
STARTED_RAY=0
TELESCOPE_PID=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  if [ -n "$TELESCOPE_PID" ]; then
    kill -- "-$TELESCOPE_PID" 2>/dev/null || true
  fi
  if [ "$STARTED_RAY" = 1 ]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------ start bank
# slime reads one prompt per start; the start row travels as its metadata.
if [ "$DRY_RUN" != 1 ]; then
  python3 - "$STARTS" "$OUT_DIR/rl_starts.jsonl" <<'PY'
import json
import sys

starts, out = sys.argv[1:3]
rows = [json.loads(line) for line in open(starts) if line.strip()]
with open(out, "w") as f:
    for row in rows:
        f.write(json.dumps({"prompt": row["position_id"], "metadata": row}) + "\n")
print(f"rl_starts.jsonl: {len(rows)} starts from {starts}")
PY
fi

# ---------------------------------------------------------- CivTelescope
if [ "$USES_TELESCOPE" = 1 ] && [ -z "${CIVTELESCOPE_URL:-}" ]; then
  CIVTELESCOPE_URL="http://127.0.0.1:$CIVTELESCOPE_PORT/generate"
  if [ "$DRY_RUN" != 1 ]; then
    TELESCOPE_DP=$(awk -F, '{print NF}' <<<"$CIVTELESCOPE_GPUS")
    served=$(CUDA_VISIBLE_DEVICES=$CIVTELESCOPE_GPUS bash "$REPO_ROOT/scripts/serve_civtelescope.sh" \
      "$CIVTELESCOPE_HF" "$CIVTELESCOPE_PORT" "$TELESCOPE_DP" "$CIVTELESCOPE_MEM_FRACTION" \
      "$OUT_DIR/civtelescope_server.log")
    echo "$served"
    # the server runs in its own session; its pid is the process-group id
    TELESCOPE_PID=$(sed -n 's/.*(pid \([0-9][0-9]*\),.*/\1/p' <<<"$served")
  fi
fi

# -------------------------------------------------------------- config
write_config() {
  cat <<YAML
civ_game: "$GAME"
civ_game_choice_tokens: $GAME_CHOICE_TOKENS
civ_enable_thinking: false
civ_err_penalty: $ERR_PENALTY
civ_stall_max_frac: $STALL_MAX_FRAC
civ_reward_log: "$OUT_DIR/reward_log.jsonl"
civ_decision_log: "$OUT_DIR/decision_log.jsonl"
civ_drop_log: "$OUT_DIR/drop_log.jsonl"
YAML
  if [ "$GAME" = othello ]; then
    echo "civ_game_opp_depth: $GAME_OPP_DEPTH"
    echo "civ_game_opp_temp: $GAME_OPP_TEMP"
    echo "civ_game_opp_exact_from: $GAME_OPP_EXACT_FROM"
  elif [ -n "${GAME_MAX_TURNS:-}" ]; then
    echo "civ_game_max_turns: $GAME_MAX_TURNS"
  fi
  if [ -n "${MIN_GROUP:-}" ]; then
    echo "civ_min_group: $MIN_GROUP"
  fi
  if [ -n "$VALUE_FN" ]; then
    cat <<YAML
civ_adv_norm: true
civ_value_fn_path: "$VALUE_FN"
civ_value_log: "$OUT_DIR/value_log.jsonl"
civ_value_harvest: "$OUT_DIR/value_harvest.jsonl"
YAML
  else
    echo "civ_adv_norm: false"
  fi
  if [ "$USES_TELESCOPE" = 1 ]; then
    cat <<YAML
civ_value_url: "$CIVTELESCOPE_URL"
civ_value_refset: "$OUT_DIR/refset.jsonl"
civ_value_refset_idx: $REFSET_IDX
civ_value_single_order: true
civ_value_concurrency: 64
YAML
    if [ "${VALUE_CHATML:-0}" = 1 ]; then
      echo "civ_game_value_chatml: true"
    fi
    if [ "$REWARD" = mix ]; then
      cat <<YAML
civ_game_mix_j0: ${MIX_J0:-0.5}
civ_game_mix_j1: ${MIX_J1:-0.5}
civ_game_mix_c: ${MIX_C:-0}
civ_game_mix_s: ${MIX_S:-1}
civ_game_mix_sb: ${MIX_SB:-1}
YAML
    fi
  fi
}
CONFIG="$OUT_DIR/config.yaml"
if [ "$DRY_RUN" != 1 ]; then
  write_config >"$CONFIG"
  # the run's own copy of the reference set (static: no refresh)
  if [ "$USES_TELESCOPE" = 1 ]; then cp "$REFSET" "$OUT_DIR/refset.jsonl"; fi
fi

# ------------------------------------------------------------ slime args
MODEL_ARGS=()
if [ -f "$SLIME_DIR/scripts/models/$MODEL_CONFIG.sh" ]; then
  # shellcheck source=/dev/null
  source "$SLIME_DIR/scripts/models/$MODEL_CONFIG.sh" # defines MODEL_ARGS
fi

CKPT_ARGS=(
  --hf-checkpoint "$INIT_CHECKPOINT"
  --load "$INIT_CHECKPOINT"
  --megatron-to-hf-mode bridge
)
if [ "$EVAL" != 1 ]; then
  CKPT_ARGS+=(
    --save "$CKPT_DIR"
    --save-interval "$SAVE_INTERVAL"
    --save-hf "$OUT_DIR/hf_iter{rollout_id}"
  )
fi

ROLLOUT_ARGS=(
  --prompt-data "$OUT_DIR/rl_starts.jsonl"
  --input-key prompt
  --metadata-key metadata
  --rollout-shuffle
  --custom-generate-function-path civmarsh.train.game_rollout.generate
  --group-rm
  --custom-rm-path civmarsh.train.game_rollout.group_reward
  --dynamic-sampling-filter-path civmarsh.train.game_rollout.drop_failed_episode_groups
  --custom-reward-post-process-path civmarsh.train.reward_post.post_process
  --custom-config-path "$CONFIG"
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$GAME_CHOICE_TOKENS"
  --rollout-temperature 1.0
  --rollout-seed "$SEED"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)
if [ -n "$VALUE_FN" ]; then
  ROLLOUT_ARGS+=(
    --custom-advantage-function-path civmarsh.train.advantage.broadcast_decision_advantages
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "$CP_SIZE"
  --seq-length 16384
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  --log-probs-chunk-size 2048
)

# The estimator stays grpo: for the potential-based rewards the custom advantage
# function supplies the precomputed per-decision GAE (gamma 1); "ppo" would
# enable a learned critic.
GRPO_ARGS=(
  --advantage-estimator grpo
  --lambd "$GAE_LAMBDA"
  --kl-loss-coef 0.00
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-distributed-optimizer
  --offload-optimizer-states
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION"
  --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --seed "$SEED"
)

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
  WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$GAME-$REWARD-seed$SEED")
  if [ -n "${WANDB_GROUP:-}" ]; then WANDB_ARGS+=(--wandb-group "$WANDB_GROUP"); fi
  if [ -n "${WANDB_API_KEY:-}" ]; then WANDB_ARGS+=(--wandb-key "$WANDB_API_KEY"); fi
fi

EVAL_ARGS=()
if [ "$EVAL" = 1 ]; then
  EVAL_ARGS=(--debug-rollout-only --rollout-num-gpus 1)
fi

read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
SLIME_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "$NUM_GPUS"
  --colocate
  ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
  ${EXTRA[@]+"${EXTRA[@]}"}
)

if [ "$DRY_RUN" = 1 ]; then
  echo "=== DRY RUN: game=$GAME reward=$REWARD seed=$SEED eval=$EVAL starts=$N_STARTS ($STARTS)"
  echo "=== $CONFIG"
  write_config
  echo "=== ray: CUDA_VISIBLE_DEVICES=$GPUS --num-gpus $NUM_GPUS"
  if [ "$USES_TELESCOPE" = 1 ] && [ -n "${CIVTELESCOPE_HF:-}" ]; then
    echo "=== CivTelescope: $CIVTELESCOPE_HF -> $CIVTELESCOPE_URL on GPUs $CIVTELESCOPE_GPUS"
  elif [ "$USES_TELESCOPE" = 1 ]; then
    echo "=== CivTelescope: running server at $CIVTELESCOPE_URL"
  fi
  echo "=== python3 $SLIME_DIR/train.py"
  printf '  %s\n' "${SLIME_ARGS[@]}"
  exit 0
fi
printf '%s\n' "${SLIME_ARGS[@]}" >"$OUT_DIR/slime_args.txt"

# ----------------------------------------------------------------- launch
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

RUNTIME_ENV_JSON=$(
  python3 - "$MEGATRON_DIR:$REPO_ROOT/src" "$HAS_NVLINK" "${SLIME_ROLLOUT_BASE_PORT:-15000}" <<'PY'
import json
import sys

pythonpath, nvls, base_port = sys.argv[1:4]
print(json.dumps({"env_vars": {
    "PYTHONPATH": pythonpath,
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.85",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": nvls,
    "SLIME_ROLLOUT_BASE_PORT": base_port,
}}))
PY
)

if [ -z "${RAY_ADDRESS:-}" ]; then
  # shellcheck disable=SC2086  # the optional flag must word-split
  CUDA_VISIBLE_DEVICES=$GPUS ray start --head --node-ip-address "$MASTER_ADDR" \
    --port "$RAY_PORT" --dashboard-port "$RAY_DASHBOARD_PORT" \
    --num-gpus "$NUM_GPUS" --disable-usage-stats \
    ${RAY_TEMP_DIR:+--temp-dir $RAY_TEMP_DIR}
  STARTED_RAY=1
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT"
fi

export PYTHONUNBUFFERED=1
set +e
ray job submit --address="$RAY_ADDRESS" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 "$SLIME_DIR/train.py" "${SLIME_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if [ "$EVAL" != 1 ] && [ "$status" = 0 ]; then
  FINAL="$OUT_DIR/hf_iter$((NUM_ROLLOUT - 1))"
  [ -d "$FINAL" ] || { echo "run finished without $FINAL" >&2; exit 1; }
  echo "final checkpoint: $FINAL"
fi
exit "$status"
