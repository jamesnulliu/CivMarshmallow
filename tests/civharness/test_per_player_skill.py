"""Per-player AI skill: command-script emission and save parsing (no server)."""

from civharness.config import resume_script
from civharness.parse import save_players


def test_per_player_lines_after_global():
    s = resume_script(
        80,
        skill="normal",
        skill_by_player={"Frederick": "hard", "Cleopatra II": "novice"},
    )
    lines = s.strip().splitlines()
    gi = lines.index("normal")
    assert lines[gi + 1] == 'hard "Frederick"'  # override AFTER global
    assert lines[gi + 2] == 'novice "Cleopatra II"'  # quoted (spaces safe)
    assert lines[-1] == "start"


def test_no_global_still_emits_overrides():
    s = resume_script(80, skill_by_player={"A": "hard"})
    assert 'hard "A"' in s


def test_bad_level_rejected():
    try:
        resume_script(80, skill_by_player={"A": "grandmaster"})
    except ValueError as e:
        assert "grandmaster" in str(e)
    else:
        raise AssertionError("bad level accepted")


def test_save_players_reads_playerN_sections():
    sections = {
        "player0": {"name": '"Frederick"'},
        "player1": {"name": '"Cleopatra II"'},
        "player12": {"name": '"Xerxes"'},
        "game": {"turn": "50"},
    }
    assert save_players(sections) == {0: "Frederick", 1: "Cleopatra II", 12: "Xerxes"}
