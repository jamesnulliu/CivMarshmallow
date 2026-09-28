"""Shared fixtures for the civharness tests.

Tests that launch the pinned freeciv-server carry the ``server`` marker (or
use the ``save`` fixture) and are skipped, not errored, when the binary is
absent — install it with scripts/install_freeciv_server.sh and point
CIVHARNESS_SERVER at it. A module may set a module-level
``SAVE_CONFIG = GameConfig(...)`` to control the snapshot ``save`` gives it;
otherwise a shared default mid-game position is used.
"""

from pathlib import Path

import pytest

from civharness import GameConfig, snapshot_series
from civharness.config import DEFAULT_BINARY

_DEFAULT_SAVE_CONFIG = GameConfig(aifill=3, endturn=20, mapseed=5, gameseed=5)


def _server_missing_reason() -> str | None:
    if Path(DEFAULT_BINARY).exists():
        return None
    return f"pinned freeciv-server not found at {DEFAULT_BINARY}"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "server: needs the pinned freeciv-server binary (CIVHARNESS_SERVER)"
    )


def pytest_collection_modifyitems(config, items):
    reason = _server_missing_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("server") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A throwaway working directory shared by a module's tests (each test uses
    its own named subdirectory under it)."""
    return tmp_path_factory.mktemp("civharness")


@pytest.fixture(scope="module")
def save(request, root):
    """A mid-game SaveRef to branch/drive from. Skips when the pinned server is
    unavailable."""
    reason = _server_missing_reason()
    if reason is not None:
        pytest.skip(reason)
    cfg = getattr(request.module, "SAVE_CONFIG", _DEFAULT_SAVE_CONFIG)
    refs = snapshot_series(cfg, root / "seed", every=10)
    return next(r for r in refs if r.turn >= 10)
