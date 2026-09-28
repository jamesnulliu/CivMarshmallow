#!/usr/bin/env bash
# Train the policy with slime for one reward, one curriculum phase and one seed.
#
# Usage:
#   bash scripts/train_policy.sh REWARD=<reward> PHASE=<phase> SEED=<seed> [KEY=VALUE ...]
#
# Every KEY=VALUE argument may equally be set in the environment; arguments win.
#
# Rewards:
#   sparse                                 final score, GRPO-whitened per start
#   scoreboard                             potential = current score, per-decision GAE
#   civtelescope                           potential = CivTelescope vs a reference set
#   scoreboard_to_civtelescope             sigmoid hand-off scoreboard -> CivTelescope
#   sparse_to_civtelescope                 sigmoid hand-off sparse -> CivTelescope
#   civtelescope_to_sparse                 sigmoid hand-off CivTelescope -> sparse
#   scoreboard_to_civtelescope_to_sparse   two hand-offs, three signals
#   terminal_civtelescope                  episode reward = CivTelescope value of the
#                                          final position (static reference set)
# Phases (start turn; every episode ends at turn 120):
#   rem20 (100)  rem40 (80)  rem80 (40)  rem120 (1)
#
# Required (paths):
#   SLIME_DIR          patched slime checkout (scripts/slime/setup_slime.sh)
#   MEGATRON_DIR       Megatron-LM checkout, put on PYTHONPATH
#   BASE_MODEL         Qwen3-8B Hugging Face weights
#   CIVHARNESS_SERVER  pinned freeciv-server binary
#   DATA_ROOT          root of the start banks and reference sets
#                      (scripts/build_start_banks.py, scripts/build_reference_sets.py);
#                      relative save paths in a start bank resolve against it
#   CIVTELESCOPE_HF    merged CivTelescope checkpoint, for the rewards that use
#                      CivTelescope, unless CIVTELESCOPE_URL names a running server
# Optional:
#   INIT_CHECKPOINT    HF checkpoint the phase starts from (default BASE_MODEL)
#   STARTS             start bank (default DATA_ROOT/starts/train_<PHASE>.jsonl)
#   REFSET             reference set (default DATA_ROOT/refsets/<PHASE>.jsonl)
#   REFSET_IDX         reference positions used per turn bucket, as a yaml list
#                      (default [0, 2, 4, 5] at rem80, else [0, 1, 3, 5];
#                      [0, 1, 3, 5] for terminal_civtelescope)
#   OUT_DIR            run directory (default runs/policy/<REWARD>/seed<SEED>/<PHASE>)
#   WORK_DIR           episode scratch with the game saves (default OUT_DIR/episodes);
#                      a fast local disk keeps the engine off the critical path
#   CKPT_DIR           Megatron checkpoints (default OUT_DIR/ckpts)
#   CIVTELESCOPE_URL   /generate endpoint of an already running CivTelescope server
#   CIVTELESCOPE_GPUS  GPUs of the CivTelescope server started here through
#                      scripts/serve_civtelescope.sh, one data-parallel replica per
#                      GPU (default 4,5,6,7)
#   CIVTELESCOPE_PORT  its port (default 30870)
#   GPUS               GPUs given to Ray and the colocated actor (default 0,...,7)
#   NUM_ROLLOUT (12)  ROLLOUT_BATCH_SIZE (8)  GLOBAL_BATCH_SIZE (64)  N_SAMPLES (8)
#   ERR_PENALTY (1.0)  STALL_MAX_FRAC (0.30; 0.40 at rem120)  REFRESH_EVERY (4)
#   MIN_GROUP          smallest surviving group kept (default: the whole group)
#   MAX_TOKENS_PER_GPU (6144)  SAVE_INTERVAL (4)  SGLANG_ATTENTION_BACKEND (fa3)
#   SGLANG_MEM_FRACTION (0.45 for the potential-based rewards, else 0.5)
#   MODEL_CONFIG       slime model-args script under scripts/models (default qwen3-8B)
#   RAY_ADDRESS        submit to this running Ray dashboard instead of starting a head
#   RAY_PORT (6379)  RAY_DASHBOARD_PORT (8265)  MASTER_ADDR (127.0.0.1)
#   SLIME_ROLLOUT_BASE_PORT  first port of the rollout engines (default 15000)
#   WANDB_PROJECT      enables wandb logging (WANDB_GROUP, WANDB_API_KEY optional)
#   EVAL=1             rollout-only evaluation (used by scripts/eval_policy.sh):
#                      one GPU, no training step, no checkpoint
#   EXTRA_ARGS         appended to the slime command line (last one wins)
#
# The final checkpoint of a phase is OUT_DIR/hf_iter<NUM_ROLLOUT - 1>.
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "unexpected argument '$arg' (expected KEY=VALUE)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
REWARD=${REWARD:?set REWARD}
PHASE=${PHASE:?set PHASE}
SEED=${SEED:?set SEED}
SLIME_DIR=${SLIME_DIR:?set SLIME_DIR (scripts/slime/setup_slime.sh)}
MEGATRON_DIR=${MEGATRON_DIR:?set MEGATRON_DIR}
BASE_MODEL=${BASE_MODEL:?set BASE_MODEL}
CIVHARNESS_SERVER=${CIVHARNESS_SERVER:?set CIVHARNESS_SERVER}
EVAL=${EVAL:-0}

case "$PHASE" in
  rem20 | rem40 | rem80 | rem120) ;;
  *) echo "unknown PHASE '$PHASE'" >&2; exit 2 ;;
esac

# Which pieces each reward needs: a potential (value function + GAE), the
# CivTelescope server, the sigmoid hand-off keys.
VALUE_FN=""
USES_TELESCOPE=0
case "$REWARD" in
  sparse) ;;
  scoreboard) VALUE_FN=civmarsh.rewards.scoreboard.score_now_value ;;
  civtelescope) VALUE_FN=civmarsh.rewards.potential.telescope_value; USES_TELESCOPE=1 ;;
  scoreboard_to_civtelescope) VALUE_FN=civmarsh.rewards.hybrid.hybrid_sigmoid_value; USES_TELESCOPE=1 ;;
  sparse_to_civtelescope | civtelescope_to_sparse)
    VALUE_FN=civmarsh.rewards.hybrid.sparse_hybrid_sigmoid_value; USES_TELESCOPE=1 ;;
  scoreboard_to_civtelescope_to_sparse)
    VALUE_FN=civmarsh.rewards.hybrid.sparse_hybrid3_value; USES_TELESCOPE=1 ;;
  terminal_civtelescope) USES_TELESCOPE=1 ;;
  *) echo "unknown REWARD '$REWARD'" >&2; exit 2 ;;
esac
if [ "$EVAL" = 1 ] && [ "$REWARD" != sparse ]; then
  echo "EVAL=1 plays episodes only; run it with REWARD=sparse" >&2
  exit 2
fi

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$BASE_MODEL}
STARTS=${STARTS:-$DATA_ROOT/starts/train_$PHASE.jsonl}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/runs/policy/$REWARD/seed$SEED/$PHASE}
WORK_DIR=${WORK_DIR:-$OUT_DIR/episodes}
CKPT_DIR=${CKPT_DIR:-$OUT_DIR/ckpts}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
NUM_ROLLOUT=${NUM_ROLLOUT:-12}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
N_SAMPLES=${N_SAMPLES:-8}
ERR_PENALTY=${ERR_PENALTY:-1.0}
if [ "$PHASE" = rem120 ]; then STALL_DEFAULT=0.40; else STALL_DEFAULT=0.30; fi
STALL_MAX_FRAC=${STALL_MAX_FRAC:-$STALL_DEFAULT}
REFRESH_EVERY=${REFRESH_EVERY:-4}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144}
SAVE_INTERVAL=${SAVE_INTERVAL:-4}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}
if [ -n "$VALUE_FN" ]; then MEM_DEFAULT=0.45; else MEM_DEFAULT=0.5; fi
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-$MEM_DEFAULT}
MODEL_CONFIG=${MODEL_CONFIG:-qwen3-8B}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
CIVTELESCOPE_GPUS=${CIVTELESCOPE_GPUS:-4,5,6,7}
CIVTELESCOPE_PORT=${CIVTELESCOPE_PORT:-30870}

if [ "$USES_TELESCOPE" = 1 ]; then
  REFSET=${REFSET:-$DATA_ROOT/refsets/$PHASE.jsonl}
  if [ "$PHASE" = rem80 ] && [ "$REWARD" != terminal_civtelescope ]; then
    REFSET_IDX=${REFSET_IDX:-[0, 2, 4, 5]}
  else
    REFSET_IDX=${REFSET_IDX:-[0, 1, 3, 5]}
  fi
  [ -s "$REFSET" ] || { echo "missing reference set $REFSET" >&2; exit 1; }
fi
[ -s "$STARTS" ] || { echo "missing start bank $STARTS" >&2; exit 1; }
[ -e "$INIT_CHECKPOINT" ] || { echo "missing INIT_CHECKPOINT $INIT_CHECKPOINT" >&2; exit 1; }

mkdir -p "$OUT_DIR" "$WORK_DIR"
if [ "$EVAL" != 1 ] && compgen -G "$OUT_DIR/hf_iter*" >/dev/null; then
  echo "$OUT_DIR already holds checkpoints; choose a fresh OUT_DIR" >&2
  exit 1
fi
# append-mode logs start empty: every launch is a fresh run
for log in reward_log decision_log drop_log value_log value_harvest refresh_log; do
  : >"$OUT_DIR/$log.jsonl"
done

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
python3 - "$STARTS" "$DATA_ROOT" "$OUT_DIR/rl_starts.jsonl" <<'PY'
import json
import os
import sys

starts, root, out = sys.argv[1:4]
rows = [json.loads(line) for line in open(starts) if line.strip()]
with open(out, "w") as f:
    for row in rows:
        row = dict(row)
        row["save_path"] = os.path.join(root, row["save_path"])
        f.write(json.dumps({"prompt": row["position_id"], "metadata": row}) + "\n")
print(f"rl_starts.jsonl: {len(rows)} starts from {starts}")
PY

# ---------------------------------------------------------- CivTelescope
if [ "$USES_TELESCOPE" = 1 ] && [ -z "${CIVTELESCOPE_URL:-}" ]; then
  CIVTELESCOPE_HF=${CIVTELESCOPE_HF:?set CIVTELESCOPE_HF or CIVTELESCOPE_URL}
  TELESCOPE_DP=$(awk -F, '{print NF}' <<<"$CIVTELESCOPE_GPUS")
  served=$(CUDA_VISIBLE_DEVICES=$CIVTELESCOPE_GPUS bash "$REPO_ROOT/scripts/serve_civtelescope.sh" \
    "$CIVTELESCOPE_HF" "$CIVTELESCOPE_PORT" "$TELESCOPE_DP" 0.20 \
    "$OUT_DIR/civtelescope_server.log")
  echo "$served"
  # the server runs in its own session; its pid is the process-group id
  TELESCOPE_PID=$(sed -n 's/.*(pid \([0-9][0-9]*\),.*/\1/p' <<<"$served")
  CIVTELESCOPE_URL="http://127.0.0.1:$CIVTELESCOPE_PORT/generate"
fi

# -------------------------------------------------------------- config
CONFIG="$OUT_DIR/config.yaml"
cat >"$CONFIG" <<YAML
civ_workdir: "$WORK_DIR"
civ_endturn: 120
civ_engine_seed: 1
civ_choice_tokens: 192
civ_max_prompt_tokens: 8192
civ_digest_window: 5
civ_enable_thinking: false
civ_err_penalty: $ERR_PENALTY
civ_stall_max_frac: $STALL_MAX_FRAC
civ_reward_log: "$OUT_DIR/reward_log.jsonl"
civ_decision_log: "$OUT_DIR/decision_log.jsonl"
civ_drop_log: "$OUT_DIR/drop_log.jsonl"
YAML
if [ -n "${MIN_GROUP:-}" ]; then
  echo "civ_min_group: $MIN_GROUP" >>"$CONFIG"
fi
if [ -n "$VALUE_FN" ]; then
  cat >>"$CONFIG" <<YAML
civ_adv_norm: true
civ_value_fn_path: "$VALUE_FN"
civ_value_log: "$OUT_DIR/value_log.jsonl"
civ_value_harvest: "$OUT_DIR/value_harvest.jsonl"
YAML
else
  echo "civ_adv_norm: false" >>"$CONFIG"
fi
if [ "$USES_TELESCOPE" = 1 ]; then
  # the reference set is live state for a refreshed run: work on a copy
  cp "$REFSET" "$OUT_DIR/refset_live.jsonl"
  cat >>"$CONFIG" <<YAML
civ_value_url: "$CIVTELESCOPE_URL"
civ_value_refset: "$OUT_DIR/refset_live.jsonl"
civ_value_refset_idx: $REFSET_IDX
civ_value_single_order: true
civ_value_concurrency: 64
YAML
  if [ "$REWARD" = terminal_civtelescope ]; then
    cat >>"$CONFIG" <<YAML
civ_value_log: "$OUT_DIR/value_log.jsonl"
civ_terminal_value_fn_path: "civmarsh.rewards.potential.terminal_telescope_value"
YAML
  else
    cp "$REFSET" "$OUT_DIR/refset_pool.jsonl"
    cat >>"$CONFIG" <<YAML
civ_value_ai_refset: "$OUT_DIR/refset_pool.jsonl"
civ_value_refresh_every: $REFRESH_EVERY
civ_value_refresh_log: "$OUT_DIR/refresh_log.jsonl"
YAML
  fi
fi
case "$REWARD" in
  scoreboard_to_civtelescope | sparse_to_civtelescope | civtelescope_to_sparse | scoreboard_to_civtelescope_to_sparse)
    cat >>"$CONFIG" <<YAML
civ_hybrid_sigmoid_c: 35
civ_hybrid_sigmoid_s: 5
civ_hybrid_sigmoid_skip_threshold: 0.01
YAML
    ;;
esac
case "$REWARD" in
  sparse_to_civtelescope) echo "civ_sparse_hybrid_sigmoid: true" >>"$CONFIG" ;;
  civtelescope_to_sparse)
    printf '%s\n' "civ_sparse_hybrid_sigmoid: true" \
      "civ_sparse_hybrid_telescope_first: true" >>"$CONFIG"
    ;;
  scoreboard_to_civtelescope_to_sparse)
    printf '%s\n' "civ_hybrid3_sigmoid: true" "civ_hybrid3_c2: 80" \
      "civ_hybrid3_s2: 5" >>"$CONFIG"
    ;;
esac

# ------------------------------------------------------------ slime args
# shellcheck source=/dev/null
source "$SLIME_DIR/scripts/models/$MODEL_CONFIG.sh" # defines MODEL_ARGS

CKPT_ARGS=(
  --hf-checkpoint "$INIT_CHECKPOINT"
  --load "$INIT_CHECKPOINT"
  --megatron-to-hf-mode bridge
)
if [ "$EVAL" != 1 ]; then
  mkdir -p "$CKPT_DIR"
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
  --custom-generate-function-path civmarsh.train.rollout.generate
  --group-rm
  --custom-rm-path civmarsh.train.rollout.group_reward
  --dynamic-sampling-filter-path civmarsh.train.rollout.drop_failed_episode_groups
  --custom-reward-post-process-path civmarsh.train.reward_post.post_process
  --custom-config-path "$CONFIG"
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len 192
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
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --seq-length 16384
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  --log-probs-chunk-size 2048
)

# The estimator stays grpo: for the potential-based rewards the custom advantage
# function supplies the precomputed per-decision GAE (lambda 0.9, gamma 1).
GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)
if [ -n "$VALUE_FN" ]; then
  GRPO_ARGS+=(--lambd 0.9)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
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
    --wandb-run-name "$REWARD-$PHASE-seed$SEED")
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
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${EXTRA[@]}"
)
printf '%s\n' "${SLIME_ARGS[@]}" >"$OUT_DIR/slime_args.txt"

# ----------------------------------------------------------------- launch
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

RUNTIME_ENV_JSON=$(
  python3 - "$MEGATRON_DIR:$REPO_ROOT/src" "$HAS_NVLINK" "$CIVHARNESS_SERVER" \
    "${SLIME_ROLLOUT_BASE_PORT:-15000}" "${CIVMARSH_RULESET_DIR:-}" <<'PY'
import json
import sys

pythonpath, nvls, server, base_port, ruleset_dir = sys.argv[1:6]
env = {
    "PYTHONPATH": pythonpath,
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.85",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": nvls,
    "CIVHARNESS_SERVER": server,
    "SLIME_ROLLOUT_BASE_PORT": base_port,
}
if ruleset_dir:
    env["CIVMARSH_RULESET_DIR"] = ruleset_dir
print(json.dumps({"env_vars": env}))
PY
)

if [ -z "${RAY_ADDRESS:-}" ]; then
  CUDA_VISIBLE_DEVICES=$GPUS ray start --head --node-ip-address "$MASTER_ADDR" \
    --port "$RAY_PORT" --dashboard-port "$RAY_DASHBOARD_PORT" \
    --num-gpus "$NUM_GPUS" --disable-usage-stats
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
