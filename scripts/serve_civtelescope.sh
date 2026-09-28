#!/usr/bin/env bash
# Serve a merged CivTelescope model with sglang for RL reward computation.
#
# The RL client (civmarsh.civtelescope.client) posts to <url>/generate and
# reads next-token log-probabilities; pass the printed URL as CIVTELESCOPE_URL to
# scripts/train_policy.sh. The server keeps running after this script returns;
# readiness is the server's own "ready to roll" log line.
#
# Usage:
#   scripts/serve_civtelescope.sh MODEL_DIR [PORT] [DP] [MEM_FRACTION] [LOG]
#   e.g. CUDA_VISIBLE_DEVICES=4,5,6,7 scripts/serve_civtelescope.sh models/civtelescope 30870 4 0.20
set -euo pipefail

MODEL_DIR=${1:?usage: serve_civtelescope.sh MODEL_DIR [PORT] [DP] [MEM_FRACTION] [LOG]}
PORT=${2:-30870}
DP=${3:-1}
MEM_FRACTION=${4:-0.20}
LOG=${5:-civtelescope_server.log}

setsid python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name civtelescope \
  --dp "$DP" --tp 1 --port "$PORT" --host 127.0.0.1 \
  --mem-fraction-static "$MEM_FRACTION" --attention-backend fa3 \
  > "$LOG" 2>&1 &
PID=$!

for _ in $(seq 1 120); do
  if grep -q "The server is fired up and ready to roll" "$LOG" 2>/dev/null; then
    echo "CivTelescope server ready: http://127.0.0.1:$PORT/generate (pid $PID, log $LOG)"
    exit 0
  fi
  if grep -q "Initialization failed\|address already in use" "$LOG" 2>/dev/null \
      || ! kill -0 "$PID" 2>/dev/null; then
    echo "CivTelescope server failed to start; last log lines:" >&2
    tail -5 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "CivTelescope server not ready after 600 s; see $LOG" >&2
exit 1
