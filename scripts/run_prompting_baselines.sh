#!/usr/bin/env bash
# Run the prompting-scaffold baselines on one or more GPUs.
#
# One pure-inference sglang server per GPU serves the base model as "policy";
# the starts are sharded round-robin over the GPUs (shard i gets rows i, i+N,
# i+2N, ...) and each shard runs scripts/run_prompting_baselines.py against its
# own server.  Requests turn Qwen3 thinking off (chat_template_kwargs), as the
# RL policy runs.  Only the servers this script started are stopped at the end.
#
# Usage:
#   MODEL=/path/to/Qwen3-8B STARTS=starts_test_rem120.jsonl DATA_ROOT=/path/to/data \
#   OUT_DIR=runs/prompting_baselines GPUS=0,1,2,3 \
#   bash scripts/run_prompting_baselines.sh [--resume]
#
# Environment:
#   MODEL (required)      base model directory (Qwen3-8B)
#   STARTS (required)     starts-bank jsonl (save_path relative to DATA_ROOT)
#   DATA_ROOT (required)  directory the starts' save paths resolve against
#   OUT_DIR (required)    output directory; shard k writes OUT_DIR/sh<k>/
#   CIVHARNESS_SERVER     freeciv-server binary used by the harness
#   GPUS                  comma-separated GPU ids, one shard each (default 0)
#   BASE_PORT             server port of the first shard (default 30900)
#   ARMS                  default direct,baselang,mastaba,saga,reflexion
#   N_SAMPLES             games per start and arm (default 8)
#   CONCURRENCY           concurrent games per shard (default 28)
#   MEM_FRACTION          sglang --mem-fraction-static (default 0.80)
#   ATTENTION_BACKEND     sglang --attention-backend (default flashinfer)
#   SERVER_TIMEOUT        seconds to wait for a server to come up (default 900)
#   PYTHON                python interpreter (default python3)
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
: "${MODEL:?set MODEL to the base model directory}"
: "${STARTS:?set STARTS to the starts-bank jsonl}"
: "${DATA_ROOT:?set DATA_ROOT to the directory holding the start saves}"
: "${OUT_DIR:?set OUT_DIR}"
GPUS=${GPUS:-0}
BASE_PORT=${BASE_PORT:-30900}
ARMS=${ARMS:-direct,baselang,mastaba,saga,reflexion}
N_SAMPLES=${N_SAMPLES:-8}
CONCURRENCY=${CONCURRENCY:-28}
MEM_FRACTION=${MEM_FRACTION:-0.80}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}
PYTHON=${PYTHON:-python3}
RESUME=${1:-}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_SHARDS=${#GPU_LIST[@]}
mkdir -p "$OUT_DIR"

SERVER_PIDS=()
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  # each server runs in its own session (setsid), so its process group holds
  # the server and every worker it spawned, and nothing else
  for pid in "${SERVER_PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${SERVER_PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_ready() {
  local log=$1 pid=$2 port=$3 waited=0
  until grep -q "The server is fired up and ready to roll" "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null \
      || grep -q "Initialization failed\|address already in use" "$log" 2>/dev/null; then
      echo "server on port $port failed to start:" >&2
      tail -30 "$log" >&2
      return 1
    fi
    if ((waited >= SERVER_TIMEOUT)); then
      echo "server on port $port not ready after ${SERVER_TIMEOUT}s" >&2
      tail -30 "$log" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "server up on port $port"
}

# shard the starts round-robin over the non-empty lines
for ((i = 0; i < N_SHARDS; i++)); do
  mkdir -p "$OUT_DIR/sh$i"
  awk -v n="$N_SHARDS" -v i="$i" 'NF && (k++ % n) == i' "$STARTS" \
    >"$OUT_DIR/sh$i/starts.jsonl"
done

for ((i = 0; i < N_SHARDS; i++)); do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name policy \
    --tp 1 --port "$port" --host 127.0.0.1 \
    --mem-fraction-static "$MEM_FRACTION" --attention-backend "$ATTENTION_BACKEND" \
    >"$OUT_DIR/sh$i/server.log" 2>&1 &
  SERVER_PIDS+=($!)
done
for ((i = 0; i < N_SHARDS; i++)); do
  wait_ready "$OUT_DIR/sh$i/server.log" "${SERVER_PIDS[$i]}" $((BASE_PORT + i))
done

RUN_PIDS=()
for ((i = 0; i < N_SHARDS; i++)); do
  "$PYTHON" "$REPO/scripts/run_prompting_baselines.py" \
    --url "http://127.0.0.1:$((BASE_PORT + i))" \
    --starts "$OUT_DIR/sh$i/starts.jsonl" \
    --data-root "$DATA_ROOT" \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --arms "$ARMS" --n-samples "$N_SAMPLES" --endturn 120 \
    --concurrency "$CONCURRENCY" \
    --out-dir "$OUT_DIR/sh$i" \
    --workdir "$OUT_DIR/sh$i/work" \
    ${RESUME:+"$RESUME"} \
    >"$OUT_DIR/sh$i/run.log" 2>&1 &
  RUN_PIDS+=($!)
done

RC=0
for pid in "${RUN_PIDS[@]}"; do
  wait "$pid" || RC=$?
done

"$PYTHON" "$REPO/scripts/summarize_prompting_baselines.py" "$OUT_DIR" \
  --arms "$ARMS" --n-samples "$N_SAMPLES" --out "$OUT_DIR/summary.json" >/dev/null \
  || RC=$?
echo "prompting baselines finished (exit $RC); readout in $OUT_DIR/summary.json"
exit "$RC"
