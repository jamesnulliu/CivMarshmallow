#!/usr/bin/env bash
# Build and install the pinned freeciv-server 3.2.5 used by civharness.
#
# Usage: scripts/install_freeciv_server.sh
#   PREFIX=/path/to/prefix  install location (default: $HOME/opt/freeciv-3.2.5)
#   JOBS=N                  parallel make jobs (default: nproc)
#   BUILD_DIR=/path         where the source is unpacked (default: a temp dir)
#
# Only the headless server is built (no client, modpack installer, ruleset
# editor, NLS, readline or auth database), so the build needs just a C
# toolchain plus the zlib, libcurl and ICU development headers. Known-good
# toolchain: gcc 13.3, autoconf 2.71, libcurl 8.5, zlib 1.3, ICU 74.2
# (Ubuntu 24.04).
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
URL="https://files.freeciv.org/stable/freeciv-${VERSION}.tar.xz"
SHA256=d32808f02a9b9f49ef159bcbf266b16ce2a3ce6ea8f71115d80f952c3cc609e8
PREFIX="${PREFIX:-$HOME/opt/freeciv-${VERSION}}"
JOBS="${JOBS:-$(nproc)}"

if [[ -n "${BUILD_DIR:-}" ]]; then
    mkdir -p "$BUILD_DIR"
else
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT
fi

tarball="$BUILD_DIR/freeciv-${VERSION}.tar.xz"
echo "Downloading $URL"
curl -fL --retry 3 -o "$tarball" "$URL"

echo "Verifying SHA-256"
echo "${SHA256}  ${tarball}" | sha256sum -c -

tar -xJf "$tarball" -C "$BUILD_DIR"
cd "$BUILD_DIR/freeciv-${VERSION}"

# --enable-fcdb=no: no auth database is needed for headless batch use, and
# configure fails without it when the sqlite3 headers are absent.
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

"$PREFIX/bin/freeciv-server" --version
echo
echo "Installed. Point civharness at it with:"
echo "  export CIVHARNESS_SERVER=\"$PREFIX/bin/freeciv-server\""
