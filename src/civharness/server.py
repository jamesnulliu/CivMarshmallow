"""Process lifecycle for one headless freeciv-server.

Servers are driven exclusively via `--read script` (never stdin — detached,
the server ignores it) and run in their own working directory so scorelogs
and saves cannot cross-talk. Each child gets its own process group so a
hung server, and anything it spawned, dies together.
"""

import glob
import os
import signal
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from civharness.config import DEFAULT_BINARY


def _child_preexec():
    """New session (so hangs die by process-group kill) + PR_SET_PDEATHSIG
    (so even a SIGKILLed driver leaves no orphan servers behind)."""
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1, signal.SIGKILL, 0, 0, 0
        )  # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001, S110
        pass  # no libc prctl: the process-group kill still applies


class ServerHang(RuntimeError):
    """A server exceeded its wall-clock budget (it is killed first)."""


class ServerCrash(RuntimeError):
    """A server exited non-zero, or loaded a fresh game instead of the save."""


_T = TypeVar("_T")


def free_port() -> int:
    """A port that was free a moment ago. The socket is closed before the
    server binds it, so concurrent launches can race for the same port; the
    loser exits with a bind failure (ServerCrash). See `with_port_retry`."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def with_port_retry(
    fn: Callable[[], _T],
    attempts: int = 5,
    *,
    on_retry: Callable[[int, ServerCrash], None] | None = None,
) -> _T:
    """Call `fn()` (any server-launching call: run_game, snapshot_series,
    branch, ...) and re-run it on ServerCrash, up to `attempts` calls in all.

    This absorbs the `free_port` race between concurrent workers: each retry
    launches with a freshly drawn port. `on_retry(next_attempt, exc)` is called
    before every retry (e.g. to log it). The last ServerCrash is re-raised.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    attempt = 1
    while True:
        try:
            return fn()
        except ServerCrash as exc:
            if attempt == attempts:
                raise
            attempt += 1
            if on_retry is not None:
                on_retry(attempt, exc)


@dataclass
class ServerHandle:
    """A launched server: its process, working directory and port."""

    proc: subprocess.Popen
    workdir: Path
    port: int
    _log: object = field(repr=False, default=None)

    @property
    def savedir(self) -> Path:
        return self.workdir / "saves"

    @property
    def ranklog(self) -> Path:
        return self.workdir / "ranklog.txt"

    @property
    def scorelog(self) -> Path:
        return self.workdir / "freeciv-score.log"

    @property
    def log(self) -> Path:
        return self.workdir / "server.log"

    def saves(self) -> list[Path]:
        return [Path(p) for p in sorted(glob.glob(str(self.savedir / "*.sav*")))]

    def assert_loaded(self) -> None:
        """A failed -f load does NOT abort the server — it silently starts a
        fresh random game, which would poison any branch experiment."""
        if b"Failure loading savegame" in self.log.read_bytes():
            raise ServerCrash(
                f"savegame failed to load; server started a fresh game "
                f"instead (port {self.port}). Log: {self.log}"
            )

    def wait(self, timeout: float) -> None:
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise ServerHang(
                f"server (port {self.port}) exceeded {timeout}s; killed. "
                f"Log: {self.log}"
            )
        finally:
            if self._log:
                self._log.close()
        if code != 0:
            raise ServerCrash(
                f"server exited {code} (port {self.port}). Log: {self.log}"
            )

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self._log:
            self._log.close()


def launch(
    serv_text: str,
    workdir: Path,
    *,
    binary: Path = DEFAULT_BINARY,
    port: int | None = None,
    load: Path | None = None,
) -> ServerHandle:
    """Start a server on `serv_text` (and savegame `load`, if given) in
    `workdir` and return immediately. `port` defaults to `free_port()`."""
    workdir = Path(workdir)
    (workdir / "saves").mkdir(parents=True, exist_ok=True)
    script = workdir / "game.serv"
    script.write_text(serv_text)
    port = port if port is not None else free_port()
    cmd = [
        str(binary),
        "--Announce",
        "none",
        "-e",
        "--read",
        str(script),
        "-p",
        str(port),
        "-s",
        str(workdir / "saves"),
        "-R",
        str(workdir / "ranklog.txt"),
    ]
    if load is not None:
        cmd += ["-f", str(load)]
    log = open(workdir / "server.log", "wb")  # noqa: SIM115 -- closed by the handle
    # preexec_fn is the only way to set PR_SET_PDEATHSIG in the child;
    # Popen's start_new_session would cover setsid alone.
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        preexec_fn=_child_preexec,  # noqa: PLW1509
    )
    return ServerHandle(proc=proc, workdir=workdir, port=port, _log=log)


def run(
    serv_text: str,
    workdir: Path,
    *,
    binary: Path = DEFAULT_BINARY,
    port: int | None = None,
    load: Path | None = None,
    timeout: float = 1200,
) -> ServerHandle:
    """Launch and wait; returns the finished handle."""
    h = launch(serv_text, workdir, binary=binary, port=port, load=load)
    h.wait(timeout)
    return h
