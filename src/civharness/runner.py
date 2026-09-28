"""Run one configured game end-to-end and return a GameResult.

The result carries full provenance (binary digest, config hash, seeds) and a
normalized savegame digest, so determinism can be asserted at the API level:
same config -> same identity().
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from civharness import parse, server
from civharness.config import DEFAULT_BINARY, GameConfig, binary_sha256

_BINARY_SHA_CACHE: dict[str, str] = {}


def _binary_sha(binary: Path) -> str:
    key = str(binary)
    if key not in _BINARY_SHA_CACHE:
        _BINARY_SHA_CACHE[key] = binary_sha256(binary)
    return _BINARY_SHA_CACHE[key]


@dataclass(frozen=True)
class GameResult:
    """Outcome and provenance of one complete game."""

    config_hash: str
    binary_sha256: str
    mapseed: int
    gameseed: int
    turns: int
    winners: tuple[str, ...]
    losers: tuple[str, ...]
    scores: dict  # {player: {metric: int}} from the final save
    save_path: str
    save_sha256_normalized: str
    scorelog_path: str | None
    duration_s: float
    workdir: str

    def identity(self) -> dict:
        """The determinism-relevant fields: everything except wall-clock
        and filesystem location."""
        return {
            "config_hash": self.config_hash,
            "binary_sha256": self.binary_sha256,
            "turns": self.turns,
            "winners": self.winners,
            "losers": self.losers,
            "scores": self.scores,
            "save_sha256_normalized": self.save_sha256_normalized,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def default_timeout(endturn: int) -> float:
    """Wall-clock budget (seconds) for a game that runs to `endturn` with
    bot-speed turns. Late-game turns are much slower than early ones, so the
    budget is generous: its job is to detect a hang, not to pace the game."""
    return 120 + 5 * endturn


def run_game(
    config: GameConfig,
    workdir: Path,
    *,
    binary: Path = DEFAULT_BINARY,
    port: int | None = None,
    timeout: float | None = None,
) -> GameResult:
    """Play `config` to its end turn in `workdir` and summarise the final save.
    The result is also written to `workdir/result.json`."""
    t0 = time.monotonic()
    h = server.run(
        config.serv_script(),
        Path(workdir),
        binary=binary,
        port=port,
        timeout=timeout or default_timeout(config.endturn),
    )
    saves = h.saves()
    if not saves:
        raise server.ServerCrash(f"no savegame produced; log: {h.log}")
    final_save = saves[-1]
    rank = parse.parse_ranklog(h.ranklog)
    sections = parse.parse_save_sections(final_save)
    result = GameResult(
        config_hash=config.hash(),
        binary_sha256=_binary_sha(binary),
        mapseed=config.mapseed,
        gameseed=config.gameseed,
        turns=rank.turns,
        winners=tuple(p.name for p in rank.winners),
        losers=tuple(p.name for p in rank.losers),
        scores=parse.save_scores(sections),
        save_path=str(final_save),
        save_sha256_normalized=parse.normalized_save_sha(final_save),
        scorelog_path=str(h.scorelog) if h.scorelog.exists() else None,
        duration_s=round(time.monotonic() - t0, 3),
        workdir=str(h.workdir),
    )
    (h.workdir / "result.json").write_text(result.to_json())
    return result
