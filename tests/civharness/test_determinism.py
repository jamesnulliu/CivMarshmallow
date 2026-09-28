"""Determinism invariant: byte-reproducible replay from a mid-game save.

Seed a game to turn 50, save, load that save into two fresh server
processes, run both to turn 100, and require the two final savegames to be
identical after timestamp normalization. This is the property every
branch-based experiment rests on; re-run it after any change to the server
build (scripts/install_freeciv_server.sh).

The test deliberately does not import civharness, so the invariant never
depends on the package it validates. The server binary is taken from
$CIVHARNESS_SERVER, falling back to ~/opt/freeciv-3.2.5/bin/freeciv-server.
"""

import glob
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

SERVER = os.environ.get(
    "CIVHARNESS_SERVER",
    str(Path.home() / "opt/freeciv-3.2.5/bin/freeciv-server"),
)

# Every generated .serv sets timeout -1 and minplayers 0: with the defaults a
# headless AI-only game silently never starts.
COMMON = """\
set timeout -1
set minplayers 0
set autosaves "GAMEOVER"
set compresstype PLAIN
"""

SEED_SERV = (
    COMMON
    + """\
set aifill 4
set endturn 50
set gameseed 42
set mapseed 42
set size 1
hard
start
"""
)

BRANCH_SERV = (
    COMMON
    + """\
set endturn 100
start
"""
)

# The only run-to-run differences are wall-clock metadata with no game-state
# content (on 3.2.5): Unix-epoch timestamps in event-cache rows, and
# last_turn_change_time (seconds the turn change took). The epoch pattern
# covers 2017-2043.
_TIMESTAMP_RE = re.compile(rb"\b(?:1[5-9]|2[0-2])\d{8}\b")
_TURNTIME_RE = re.compile(rb"last_turn_change_time=\d+")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(
    serv_text: str, workdir: Path, load: Path | None = None, timeout: int = 600
) -> Path:
    """Run one headless game to completion; return the gameover savegame."""
    savedir = workdir / "saves"
    savedir.mkdir(parents=True)
    script = workdir / "game.serv"
    script.write_text(serv_text)
    cmd = [
        SERVER,
        "--Announce",
        "none",
        "-e",
        "--read",
        str(script),
        "-p",
        str(_free_port()),
        "-s",
        str(savedir),
        "-R",
        str(workdir / "ranklog.txt"),
    ]
    if load is not None:
        cmd += ["-f", str(load)]
    with open(workdir / "server.log", "wb") as log:
        subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True
        )
    saves = sorted(glob.glob(str(savedir / "*.sav*")))
    assert saves, f"no savegame produced in {savedir} (see {workdir}/server.log)"
    return Path(saves[-1])


def normalize(raw: bytes) -> bytes:
    raw = _TIMESTAMP_RE.sub(b"TS", raw)
    return _TURNTIME_RE.sub(b"last_turn_change_time=T", raw)


def _turn_of(save: Path) -> int:
    m = re.search(r"-T(\d+)-", save.name)
    assert m, f"cannot parse turn number from {save.name}"
    return int(m.group(1))


@pytest.mark.server
def test_determinism(tmp_path):
    assert os.access(SERVER, os.X_OK), f"server binary not found: {SERVER}"
    root = tmp_path

    seed_save = run_server(SEED_SERV, root / "seed")
    branch_a = run_server(BRANCH_SERV, root / "a", load=seed_save)
    branch_b = run_server(BRANCH_SERV, root / "b", load=seed_save)

    # Guard against the trivial-pass failure mode: if the endturn override
    # were not applied after the load, the "branches" would end immediately
    # and compare equal without having replayed anything.
    assert _turn_of(branch_a) >= _turn_of(seed_save) + 50, (
        f"branch did not advance: {seed_save.name} -> {branch_a.name}"
    )

    raw_a, raw_b = branch_a.read_bytes(), branch_b.read_bytes()
    diff_lines = [
        (la, lb) for la, lb in zip(raw_a.splitlines(), raw_b.splitlines()) if la != lb
    ]
    sample = "\n".join(
        f"  A: {la[:120]!r}\n  B: {lb[:120]!r}" for la, lb in diff_lines[:10]
    )
    assert normalize(raw_a) == normalize(raw_b), (
        f"{len(diff_lines)} raw differing lines, divergence beyond wall-clock "
        f"metadata — determinism BROKEN:\n{sample}"
    )
