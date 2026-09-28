"""Concurrent batch execution: no port collisions, no orphan servers even
after SIGKILL of the driver, and manifest-based resume."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import civharness
from civharness import BatchRunner, GameConfig

pytestmark = pytest.mark.server

# The directory holding the civharness package, for the SIGKILL subprocess.
_PACKAGE_PARENT = str(Path(civharness.__file__).resolve().parent.parent)


def _orphans(marker: str) -> list[str]:
    out = subprocess.run(
        ["pgrep", "-af", "freeciv-server"], capture_output=True, text=True, check=False
    ).stdout
    return [line for line in out.splitlines() if marker in line]


def test_concurrent_batch(tmp_path):
    jobs = {
        f"g{i:02d}": GameConfig(
            aifill=3, endturn=15, mapseed=i, gameseed=i, scorelog=False
        )
        for i in range(12)
    }
    report = BatchRunner(tmp_path, workers=12).run(jobs)
    assert report.completed == 12, f"only {report.completed}/12 completed"
    assert not _orphans(str(tmp_path)), "servers left running after batch"
    # Distinct seeds must produce distinct games (port/savedir cross-talk check).
    shas = {r.save_sha256_normalized for r in report.results.values()}
    assert len(shas) == 12, f"only {len(shas)} distinct outcomes from 12 seeds"


def test_kill_leaves_no_orphans(tmp_path):
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import sys; sys.path.insert(0, {_PACKAGE_PARENT!r})
from civharness import BatchRunner, GameConfig
jobs = {{f"g{{i}}": GameConfig(aifill=4, endturn=200, mapseed=i, gameseed=i,
                               scorelog=False) for i in range(6)}}
BatchRunner({str(tmp_path)!r}, workers=6).run(jobs)
""",
        ]
    )
    time.sleep(12)  # let servers spawn and get busy
    assert _orphans(str(tmp_path)), "test invalid: no servers were running yet"
    os.kill(driver.pid, signal.SIGKILL)
    driver.wait()
    time.sleep(2)  # PDEATHSIG delivery
    left = _orphans(str(tmp_path))
    assert not left, f"orphan servers survived driver SIGKILL: {left}"


def test_resume(tmp_path):
    jobs = {
        f"g{i}": GameConfig(aifill=3, endturn=10, mapseed=i, gameseed=i, scorelog=False)
        for i in range(4)
    }
    first = BatchRunner(tmp_path, workers=4).run({k: jobs[k] for k in list(jobs)[:2]})
    assert first.completed == 2
    second = BatchRunner(tmp_path, workers=4).run(jobs)
    assert second.skipped == 2, f"expected 2 skipped, got {second.skipped}"
    assert second.completed == 4
    manifest = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert sum(1 for r in manifest if r["status"] == "done") == 4
