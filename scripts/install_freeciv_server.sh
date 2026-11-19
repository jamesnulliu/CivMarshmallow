#!/usr/bin/env bash
# Build and install the freeciv-server 3.2.5 used by civharness, and the civm41
# ruleset of environment v4.1.
#
# Usage: scripts/install_freeciv_server.sh
#   PREFIX=/path/to/prefix  install location (default: $HOME/opt/freeciv-3.2.5)
#   JOBS=N                  parallel make jobs (default: nproc)
#   BUILD_DIR=/path         where the source is checked out (default: a temp dir)
#   FREECIV_REPO=URL        source repository (default: github.com/jamesnulliu/freeciv)
#   FREECIV_REF=REF         commit, tag or branch (default: the pinned commit below)
#
# The source is freeciv 3.2.5 (upstream tag R3_2_5) plus two commits on branch
# civm-3.2.5 of github.com/jamesnulliu/freeciv:
#   - Hard bots get the H_FOG and H_MAP handicaps: they decide from what their
#     player can see, without sight through fog or knowledge of unexplored tiles.
#   - A conquest victory does not end the game: every game runs to 'endturn', so
#     the score at the turn limit stays comparable across games.
# FREECIV_REF=R3_2_5 builds the unmodified 3.2.5 release from the same repository.
#
# Only the headless server is built (no client, modpack installer, ruleset
# editor, NLS, readline or auth database). Building from git needs a C
# toolchain, autoconf, automake, libtool and pkg-config, plus the zlib, libcurl
# and ICU development headers. Known-good toolchain: gcc 13.3, autoconf 2.71,
# libcurl 8.5, zlib 1.3, ICU 74.2 (Ubuntu 24.04).
#
# Determinism is proven per server build, not per version: after installing,
# run the replay check before using the binary for experiments:
#
#   export CIVHARNESS_SERVER="$PREFIX/bin/freeciv-server"
#   pytest tests/civharness/test_determinism.py
#
# It seeds a game, loads the same mid-game save into two fresh servers, and
# requires byte-identical final saves after normalizing wall-clock fields.

set -euo pipefail

VERSION=3.2.5
FREECIV_REPO="${FREECIV_REPO:-https://github.com/jamesnulliu/freeciv.git}"
# branch civm-3.2.5: R3_2_5 + Hard-bot vision handicaps + no conquest end
FREECIV_REF="${FREECIV_REF:-8d7232bbac63d603bef689309326aab47fea34e6}"
PREFIX="${PREFIX:-$HOME/opt/freeciv-${VERSION}}"
JOBS="${JOBS:-$(nproc)}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${BUILD_DIR:-}" ]]; then
    mkdir -p "$BUILD_DIR"
else
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT
fi

src="$BUILD_DIR/freeciv"
rm -rf "$src"
echo "Fetching $FREECIV_REF from $FREECIV_REPO"
git init -q "$src"
git -C "$src" fetch -q --depth 1 "$FREECIV_REPO" "$FREECIV_REF"
git -C "$src" -c advice.detachedHead=false checkout -q FETCH_HEAD
commit="$(git -C "$src" rev-parse HEAD)"
if [[ "$FREECIV_REF" =~ ^[0-9a-f]{40}$ && "$commit" != "$FREECIV_REF" ]]; then
    echo "checked out $commit, expected $FREECIV_REF" >&2
    exit 1
fi
echo "Building freeciv at $commit"
cd "$src"

# --enable-fcdb=no: no auth database is needed for headless batch use, and
# configure fails without it when the sqlite3 headers are absent.
./autogen.sh --no-configure-run
./configure \
    --prefix="$PREFIX" \
    --disable-client \
    --enable-fcmp=no \
    --enable-ruledit=no \
    --disable-nls \
    --without-readline \
    --enable-fcdb=no
make -j"$JOBS"
make install

# Some filesystems leave the installed binary without execute permission.
chmod 755 "$PREFIX/bin/freeciv-server"

# The civm41 ruleset (environment v4.1), generated from the installed civ2civ3.
python3 "$HERE/make_civm41.py" --server "$PREFIX/bin/freeciv-server"

"$PREFIX/bin/freeciv-server" --version
echo
echo "Installed freeciv $commit. Point civharness at it with:"
echo "  export CIVHARNESS_SERVER=\"$PREFIX/bin/freeciv-server\""
