"""Run many games concurrently without leaking a single process.

- Pool sized from CPU *affinity* (the cores this process may actually run on,
  which under a scheduler or container can be far fewer than the machine has).
- Children get PR_SET_PDEATHSIG, so even `kill -9` of the driver reaps every
  server; hangs are killed by process group.
- Failed runs (ServerCrash / ServerHang) are retried in a fresh run directory.
- A JSONL manifest makes batches resumable: completed run_ids are skipped on
  restart, because a large batch *will* be interrupted.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from civharness import server
from civharness.config import DEFAULT_BINARY, GameConfig
from civharness.runner import GameResult, run_game


def default_workers() -> int:
    """Number of CPUs this process may run on."""
    return len(os.sched_getaffinity(0))


@dataclass
class BatchReport:
    """Per-run results of one BatchRunner.run call plus the counts."""

    results: dict  # run_id -> GameResult | None (failed)
    completed: int
    failed: int
    skipped: int  # already in manifest from a previous attempt


class BatchRunner:
    """Run {run_id: GameConfig} jobs on a thread pool under `workdir`, with
    `retries` extra attempts per failed run and a resumable manifest
    (`workdir/manifest.jsonl`)."""

    def __init__(
        self,
        workdir: Path,
        *,
        workers: int | None = None,
        binary: Path = DEFAULT_BINARY,
        retries: int = 1,
        progress=None,
    ):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.workers = workers or default_workers()
        self.binary = binary
        self.retries = retries
        self.progress = progress or (lambda msg: None)
        self.manifest = self.workdir / "manifest.jsonl"
        self._lock = threading.Lock()

    def _done_ids(self) -> dict[str, str]:
        done = {}
        if self.manifest.exists():
            for line in self.manifest.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("status") == "done":
                        done[rec["run_id"]] = rec["result_file"]
        return done

    def _record(self, rec: dict) -> None:
        with self._lock, open(self.manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _one(self, run_id: str, config: GameConfig) -> GameResult | None:
        last_err = None
        for attempt in range(self.retries + 1):
            rundir = (
                self.workdir
                / "runs"
                / (run_id if attempt == 0 else f"{run_id}.retry{attempt}")
            )
            try:
                result = run_game(config, rundir, binary=self.binary)
                self._record(
                    {
                        "run_id": run_id,
                        "status": "done",
                        "config_hash": config.hash(),
                        "result_file": str(rundir / "result.json"),
                    }
                )
                self.progress(f"done {run_id}")
                return result
            except (server.ServerCrash, server.ServerHang) as e:
                last_err = e
                self.progress(f"retry {run_id}: {e}")
        self._record(
            {
                "run_id": run_id,
                "status": "failed",
                "config_hash": config.hash(),
                "error": str(last_err),
            }
        )
        self.progress(f"FAILED {run_id}: {last_err}")
        return None

    def run(self, jobs: dict[str, GameConfig]) -> BatchReport:
        done = self._done_ids()
        results: dict[str, GameResult | None] = {}
        skipped = 0
        todo = {}
        for run_id, config in jobs.items():
            if run_id in done:
                skipped += 1
                rf = Path(done[run_id])
                results[run_id] = (
                    GameResult(**json.loads(rf.read_text())) if rf.exists() else None
                )
            else:
                todo[run_id] = config
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                run_id: pool.submit(self._one, run_id, config)
                for run_id, config in todo.items()
            }
            for run_id, fut in futures.items():
                results[run_id] = fut.result()
        completed = sum(1 for r in results.values() if r is not None)
        return BatchReport(
            results=results,
            completed=completed,
            failed=len(results) - completed,
            skipped=skipped,
        )
