"""A player absent from the observation save's player table is not necessarily
dead: `_obs_save` returns the turn's autosave as soon as it appears, possibly
while it is still being written, so a truncated player table reads "absent"
for a live seat. `_player_id_of` must confirm absence against the
second-newest save (complete by construction) before classifying GAME;
anything unconfirmed is UNKNOWN.

No binary needed — pure save-text fixtures.
"""

import os
import time
from pathlib import Path

from civharness.agent import _player_id_of
from civharness.client import ControlLost


def _write_save(path: Path, players, mtime=None):
    body = "[game]\nturn=7\n"
    for i, (name, alive) in enumerate(players):
        body += f'[player{i}]\nname="{name}"\nis_alive={"TRUE" if alive else "FALSE"}\n'
    path.write_text(body)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _absent_raise(saves_dir):
    """Drive the absent path: empty player table, saves_dir wired through."""
    try:
        _player_id_of({}, "Alice", saves_dir=saves_dir)
    except ControlLost as e:
        return e
    raise AssertionError("absent player did not raise ControlLost")


def test_alive_in_previous_save_is_unknown(tmp_path):
    """Absent from the (possibly mid-write) newest save but alive in the
    previous one -> UNKNOWN, never GAME."""
    now = time.time()
    _write_save(
        tmp_path / "auto-T0006.sav", [("Alice", True), ("Bob", True)], mtime=now - 60
    )
    _write_save(tmp_path / "auto-T0007.sav", [("Bob", True)], mtime=now)  # truncated
    exc = _absent_raise(tmp_path)
    assert str(exc).startswith("UNKNOWN:"), str(exc)
    assert exc.kind == "UNKNOWN"
    assert "race" in str(exc), str(exc)


def test_absence_reproduced_is_game(tmp_path):
    """Absent (or dead) in the complete previous save too -> real elimination."""
    now = time.time()
    _write_save(
        tmp_path / "auto-T0006.sav", [("Alice", False), ("Bob", True)], mtime=now - 60
    )
    _write_save(tmp_path / "auto-T0007.sav", [("Bob", True)], mtime=now)
    exc = _absent_raise(tmp_path)
    assert str(exc).startswith("GAME:"), str(exc)
    assert exc.kind == "GAME"


def test_no_earlier_save_is_unknown(tmp_path):
    """Nothing complete to confirm against -> fail-safe UNKNOWN."""
    _write_save(tmp_path / "auto-T0007.sav", [("Bob", True)])
    exc = _absent_raise(tmp_path)
    assert str(exc).startswith("UNKNOWN:"), str(exc)


def test_no_saves_dir_is_unknown():
    """Call sites that cannot supply saves_dir stay fail-safe too."""
    try:
        _player_id_of({}, "Alice")
    except ControlLost as e:
        assert str(e).startswith("UNKNOWN:"), str(e)
        return
    raise AssertionError("absent player did not raise ControlLost")


def test_present_player_untouched():
    """The happy path must not consult any save."""
    pid = _player_id_of({"Alice": {"player_id": 3}}, "Alice", saves_dir="/nonexistent")
    assert pid == 3
