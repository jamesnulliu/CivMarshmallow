#!/usr/bin/env bash
# Run the phase curriculum for one reward and one seed.
#
# Usage:
#   bash scripts/run_curriculum.sh REWARD=<reward> SEED=<seed> [KEY=VALUE ...]
#
# Phases run in order, each for NUM_ROLLOUT (12) iterations through
# scripts/train_policy.sh; every phase starts from the previous phase's final
# Hugging Face checkpoint (OUT_DIR/hf_iter<NUM_ROLLOUT - 1>), which is a fresh
# start of the optimizer and data order. A phase whose final checkpoint already
# exists is skipped, so an interrupted chain resumes where it stopped.
#
# Phases by reward (override with PHASES="rem40 rem80"):
#   sparse, scoreboard, civtelescope   rem20 rem40 rem80 rem120, from BASE_MODEL
#   the four hybrid rewards            rem120 only, from the civtelescope rem80
#                                      final checkpoint of the same seed
#   terminal_civtelescope              rem40 rem80, from BASE_MODEL
# INIT_CHECKPOINT overrides the checkpoint the first phase starts from.
#
# Run directories: RUN_ROOT/<REWARD>/seed<SEED>/<PHASE> (RUN_ROOT default
# runs/policy). Everything else (paths, CivTelescope, hyperparameters) is passed
# through to scripts/train_policy.sh; see its header.
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "unexpected argument '$arg' (expected KEY=VALUE)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
REWARD=${REWARD:?set REWARD}
SEED=${SEED:?set SEED}
BASE_MODEL=${BASE_MODEL:?set BASE_MODEL}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/policy}
NUM_ROLLOUT=${NUM_ROLLOUT:-12}
export NUM_ROLLOUT BASE_MODEL
LAST="hf_iter$((NUM_ROLLOUT - 1))"

case "$REWARD" in
  sparse | scoreboard | civtelescope)
    DEFAULT_PHASES="rem20 rem40 rem80 rem120"
    DEFAULT_INIT=$BASE_MODEL
    ;;
  scoreboard_to_civtelescope | sparse_to_civtelescope | civtelescope_to_sparse | scoreboard_to_civtelescope_to_sparse)
    DEFAULT_PHASES="rem120"
    DEFAULT_INIT=$RUN_ROOT/civtelescope/seed$SEED/rem80/$LAST
    ;;
  terminal_civtelescope)
    DEFAULT_PHASES="rem40 rem80"
    DEFAULT_INIT=$BASE_MODEL
    ;;
  *) echo "unknown REWARD '$REWARD'" >&2; exit 2 ;;
esac
PHASES=${PHASES:-$DEFAULT_PHASES}
init=${INIT_CHECKPOINT:-$DEFAULT_INIT}

for phase in $PHASES; do
  out="$RUN_ROOT/$REWARD/seed$SEED/$phase"
  if [ -d "$out/$LAST" ]; then
    echo "[$REWARD seed$SEED] $phase already complete: $out/$LAST"
  else
    [ -e "$init" ] || { echo "missing checkpoint $init for $phase" >&2; exit 1; }
    echo "[$REWARD seed$SEED] $phase from $init"
    INIT_CHECKPOINT=$init bash "$REPO_ROOT/scripts/train_policy.sh" \
      REWARD="$REWARD" PHASE="$phase" SEED="$SEED" OUT_DIR="$out"
    [ -d "$out/$LAST" ] || { echo "$phase ended without $out/$LAST" >&2; exit 1; }
  fi
  init="$out/$LAST"
done
echo "[$REWARD seed$SEED] curriculum complete: $init"
