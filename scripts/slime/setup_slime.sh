#!/usr/bin/env bash
# Install the pinned, patched slime used by the RL recipe.
#
# Usage: bash scripts/slime/setup_slime.sh [SLIME_DIR]
#
# Clones THUDM/slime into SLIME_DIR (default: ./third_party/slime), checks out
# the pinned commit, applies scripts/slime/slime.patch and installs the package
# in editable mode. Run it inside the slime docker image (slimerl/slime), which
# ships Megatron-LM, sglang and the CUDA stack; the install therefore uses
# --no-deps, as upstream recommends. Re-running on an already patched checkout
# is a no-op apart from the reinstall.
#
# Environment:
#   SLIME_REPO    git URL to clone from (default https://github.com/THUDM/slime)
#   SLIME_COMMIT  commit to pin (default 21b1b33d)
#   PIP_ARGS      extra arguments for `pip install` (default --no-deps)
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$HERE/../.." && pwd)"
SLIME_DIR="${1:-${SLIME_DIR:-$REPO_ROOT/third_party/slime}}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime}"
SLIME_COMMIT="${SLIME_COMMIT:-21b1b33d}"
PATCH="$HERE/slime.patch"
PIP_ARGS="${PIP_ARGS:---no-deps}"

if [ ! -d "$SLIME_DIR/.git" ]; then
  mkdir -p "$(dirname -- "$SLIME_DIR")"
  git clone "$SLIME_REPO" "$SLIME_DIR"
fi

cd "$SLIME_DIR"
head_commit="$(git rev-parse HEAD)"
pinned_commit="$(git rev-parse --verify "${SLIME_COMMIT}^{commit}")"
if [ "$head_commit" != "$pinned_commit" ]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: $SLIME_DIR has local changes and is not at $SLIME_COMMIT" >&2
    exit 1
  fi
  git checkout --quiet "$pinned_commit"
fi

if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "slime.patch already applied in $SLIME_DIR"
else
  git apply --check "$PATCH"
  git apply "$PATCH"
  echo "applied slime.patch to $SLIME_DIR"
fi

# shellcheck disable=SC2086  # PIP_ARGS is a whitespace-separated option list
python3 -m pip install $PIP_ARGS -e "$SLIME_DIR"
echo "slime installed from $SLIME_DIR at $(git rev-parse --short HEAD) (patched)"
