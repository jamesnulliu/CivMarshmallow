"""run_game(config) works end-to-end and re-running the same config produces
an identical result identity (turns, outcomes, scores, and normalized savegame
digest)."""

import pytest

from civharness import GameConfig, run_game
from civharness.server import ServerCrash, with_port_retry


@pytest.mark.server
def test_runner_deterministic(tmp_path):
    root = tmp_path
    config = GameConfig(aifill=3, endturn=30, mapseed=7, gameseed=7)

    r1 = run_game(config, root / "run1")
    r2 = run_game(config, root / "run2")

    assert r1.turns > 0, "no turns recorded"
    assert r1.winners or r1.losers, "ranklog parsed no players"
    assert r1.scores, "no scores parsed from final save"
    assert r1.scorelog_path, "scorelog missing"
    assert r1.identity() == r2.identity(), (
        f"same config, different results:\n{r1.identity()}\n{r2.identity()}"
    )


def test_with_port_retry():
    calls, retries = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ServerCrash("bind failure: Address already in use")
        return "ok"

    got = with_port_retry(flaky, on_retry=lambda n, e: retries.append(n))
    assert got == "ok" and len(calls) == 3 and retries == [2, 3]

    def always():
        raise ServerCrash("boom")

    with pytest.raises(ServerCrash):
        with_port_retry(always, attempts=2)
    with pytest.raises(ValueError):
        with_port_retry(always, attempts=0)
