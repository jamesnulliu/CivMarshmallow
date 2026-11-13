"""Determinism of the in-process games: a save plus a seed determines the
continuation — twice in one process, and across processes whatever
PYTHONHASHSEED is (Catanatron's own loop is hash-order dependent; the adapter
sorts every action list, see civharness/games/catan.py)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import civharness
from civharness.games import branch, play, snapshot_series

SRC = Path(civharness.__file__).resolve().parent.parent

HAVE_CATAN = importlib.util.find_spec("catanatron") is not None
GAMES = ["othello"] + (["catan"] if HAVE_CATAN else [])
SNAPS = {"othello": [36, 46], "catan": [20, 80]}

# what a subprocess computes: a snapshot series, branches of it, and a played episode
PROBE = """
import json, sys
from civharness.games import branch, play, snapshot_series
game, snaps = sys.argv[1], json.loads(sys.argv[2])
saves = snapshot_series(game, 17, snaps)
start = snapshot_series(game, 18, snaps[:1])[0]
decisions, result = play(start, lambda prompt: "invalid", 23)
print(json.dumps({
    "turns": [s["turn"] for s in saves],
    "branch": [branch(s, 3, 99) for s in saves],
    "episode": [[d["turn"], d["fallback"]] for d in decisions],
    "result": result,
}, sort_keys=True))
"""


def probe(game, hashseed):
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed), "PYTHONPATH": str(SRC)}
    out = subprocess.run(
        [sys.executable, "-c", PROBE, game, json.dumps(SNAPS[game])],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize("game", GAMES)
def test_same_save_same_seed_same_continuation(game):
    saves = snapshot_series(game, 17, SNAPS[game])
    assert [s["turn"] for s in saves] == SNAPS[game]
    first = []
    for save in saves:
        again = json.loads(json.dumps(save))  # through serialisation
        first.append(branch(save, 3, 99))
        assert branch(again, 3, 99) == first[-1]
    assert any(branch(saves[0], 3, other) != first[0] for other in (100, 101, 102, 103))
    start = saves[0]
    a = play(start, lambda prompt: "invalid", 23)
    assert a == play(start, lambda prompt: "invalid", 23)
    assert a != play(start, lambda prompt: "invalid", 24)


@pytest.mark.parametrize("game", GAMES)
def test_deterministic_across_processes_and_hash_seeds(game):
    first = probe(game, 0)
    assert first["turns"] == SNAPS[game]
    assert probe(game, 12345) == first


@pytest.mark.skipif(not HAVE_CATAN, reason="catanatron not installed")
def test_concurrent_catan_episodes_do_not_share_randomness():
    """Catanatron draws from the module-level RNG; with the state swap, an
    episode run alone and the same episode run among others in threads agree."""
    from concurrent.futures import ThreadPoolExecutor

    start = snapshot_series("catan", 18, [20])[0]
    alone = [play(start, lambda prompt: "invalid", seed) for seed in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        together = list(
            pool.map(lambda seed: play(start, lambda prompt: "invalid", seed), range(6))
        )
    assert together == alone
