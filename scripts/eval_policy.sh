#!/usr/bin/env bash
# Evaluate one policy checkpoint on a frozen evaluation bank.
#
# Usage:
#   bash scripts/eval_policy.sh CHECKPOINT=<hf dir> PHASE=<phase> OUT_DIR=<dir> [KEY=VALUE ...]
#
# Plays every start of STARTS (default DATA_ROOT/starts/test_<PHASE>.jsonl) with
# 8 episodes per start to turn 120, in one rollout-only slime call on one GPU:
# rollout batch = the number of starts, global batch = 8 x that, rollout seed 42,
# temperature 1.0, thinking off, invalid actions unpriced, groups kept with as
# few as one surviving episode (MIN_GROUP 1), no training step, no checkpoint,
# no CivTelescope server. The run directory is then validated and reduced to
# OUT_DIR/cell.json (scripts/summarize_policy_eval.py cell).
#
# The start cell of a phase gain is the checkpoint the phase began from on the
# same bank: the base model for rem20, the previous phase's final checkpoint
# otherwise (for the hybrid rewards, the civtelescope rem80 final checkpoint).
#
# Paths and cluster settings are those of scripts/train_policy.sh (SLIME_DIR,
# MEGATRON_DIR, BASE_MODEL, CIVHARNESS_SERVER, DATA_ROOT, GPUS, RAY_*, ...).
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "unexpected argument '$arg' (expected KEY=VALUE)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
PHASE=${PHASE:?set PHASE}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
STARTS=${STARTS:-$DATA_ROOT/starts/test_$PHASE.jsonl}
N_SAMPLES=${N_SAMPLES:-8}
REQUIRE_SPLIT=${REQUIRE_SPLIT-test} # every start row must carry this split ("" to skip)

if [ -e "$OUT_DIR/reward_log.jsonl" ] && [ -s "$OUT_DIR/reward_log.jsonl" ]; then
  echo "$OUT_DIR already holds an evaluation; choose a fresh OUT_DIR" >&2
  exit 1
fi
N_STARTS=$(grep -c . "$STARTS")

bash "$REPO_ROOT/scripts/train_policy.sh" \
  REWARD=sparse EVAL=1 PHASE="$PHASE" SEED=42 OUT_DIR="$OUT_DIR" \
  STARTS="$STARTS" INIT_CHECKPOINT="$CHECKPOINT" \
  NUM_ROLLOUT=1 N_SAMPLES="$N_SAMPLES" ROLLOUT_BATCH_SIZE="$N_STARTS" \
  GLOBAL_BATCH_SIZE=$((N_STARTS * N_SAMPLES)) \
  ERR_PENALTY=0 STALL_MAX_FRAC=0.40 MIN_GROUP=1

SPLIT_ARGS=()
if [ -n "$REQUIRE_SPLIT" ]; then SPLIT_ARGS=(--require-split "$REQUIRE_SPLIT"); fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$REPO_ROOT/scripts/summarize_policy_eval.py" cell \
  --run-dir "$OUT_DIR" --starts "$STARTS" --n-samples "$N_SAMPLES" \
  "${SPLIT_ARGS[@]}" --out "$OUT_DIR/cell.json"
echo "cell: $OUT_DIR/cell.json"
