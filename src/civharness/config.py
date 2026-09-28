"""Typed game configuration and the server command scripts built from it.

Every generated script hard-codes the two settings without which a headless
AI-only game silently never starts (timeout -1, minplayers 0), saves plain
text for cheap diffing, and autosaves at game over so every run ends with a
loadable position.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

# The pinned freeciv-server 3.2.5 binary. scripts/install_freeciv_server.sh
# installs it under this default prefix; CIVHARNESS_SERVER overrides it.
DEFAULT_BINARY = Path(
    os.environ.get(
        "CIVHARNESS_SERVER", Path.home() / "opt/freeciv-3.2.5/bin/freeciv-server"
    )
)

SKILL_LEVELS = ("away", "novice", "easy", "normal", "hard", "cheating", "experimental")

# Settings applied to every autonomous game, seeded or resumed: without
# `timeout -1` and `minplayers 0` a headless AI-only game never starts.
_MANDATORY = (
    "set timeout -1",
    "set minplayers 0",
    'set autosaves "GAMEOVER"',
    "set compresstype PLAIN",
)


@dataclass(frozen=True)
class GameConfig:
    aifill: int = 4
    skill: str = "hard"
    endturn: int = 100
    mapseed: int = 42
    gameseed: int = 42
    size: int = 1  # map area in thousands of tiles
    ruleset: str | None = None  # None = server default (civ2civ3 on 3.2)
    scorelog: bool = True
    saveturns: int | None = None  # also autosave every N turns (position sampling)
    extra: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.skill not in SKILL_LEVELS:
            raise ValueError(f"skill {self.skill!r} not in {SKILL_LEVELS}")

    def serv_script(self) -> str:
        # rulesetdir must precede every `set`: switching rulesets reloads
        # the ruleset and resets server settings to defaults (on 3.2.5
        # minplayers snaps back to 1 and `start` is refused, hanging the
        # headless game). No ruleset -> no rulesetdir line.
        lines = []
        if self.ruleset:
            lines.append(f"rulesetdir {self.ruleset}")
        lines += list(_MANDATORY)
        lines += [
            f"set aifill {self.aifill}",
            f"set endturn {self.endturn}",
            f"set gameseed {self.gameseed}",
            f"set mapseed {self.mapseed}",
            f"set size {self.size}",
        ]
        if self.scorelog:
            lines.append("set scorelog enabled")
        if self.saveturns is not None:
            lines.append('set autosaves "TURN|GAMEOVER"')
            lines.append(f"set saveturns {self.saveturns}")
        lines.append(self.skill)  # bare skill command: all + future AIs
        lines += list(self.extra)
        lines.append("start")
        return "\n".join(lines) + "\n"

    def hash(self) -> str:
        return hashlib.sha256(self.serv_script().encode()).hexdigest()[:12]


def resume_script(
    endturn: int,
    *,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    gameseed: int | None = None,
    scorelog: bool = True,
    extra: tuple[str, ...] = (),
) -> str:
    """Command script for a server that loads a savegame (-f) and plays on.

    `gameseed` only takes effect if the save's stored RNG state has been
    invalidated first — see branch.reseed_save().

    `skill_by_player` maps player name -> AI level, emitted as
    `<level> "<name>"` AFTER the bare `skill` line, so the global level is
    applied to all players first and the named overrides win. This is what
    makes an opponent profile expressible: e.g. skill="normal",
    skill_by_player={"Frederick": "hard"}. Names must come from the save
    (parse.save_players), never guessed.
    """
    lines = list(_MANDATORY) + [f"set endturn {endturn}"]
    if gameseed is not None:
        lines.append(f"set gameseed {gameseed}")
    if scorelog:
        lines.append("set scorelog enabled")
    if skill is not None:
        if skill not in SKILL_LEVELS:
            raise ValueError(f"skill {skill!r} not in {SKILL_LEVELS}")
        lines.append(skill)
    for name, level in (skill_by_player or {}).items():
        if level not in SKILL_LEVELS:
            raise ValueError(f"skill {level!r} not in {SKILL_LEVELS}")
        lines.append(f'{level} "{name}"')
    lines += list(extra)
    lines.append("start")
    return "\n".join(lines) + "\n"


def client_resume_script(
    endturn: int,
    *,
    dump_lua_path: Path | None = None,
    skill: str | None = None,
    skill_by_player: dict[str, str] | None = None,
    scorelog: bool = True,
    saveturns: int | None = None,
    extra: tuple[str, ...] = (),
) -> str:
    """Command script for a save loaded to be driven by an external client
    (civharness.client). Unlike resume_script() it deliberately does NOT
    `start`: the controlling connection issues /start after taking its player.

    The two settings that make interactive control work (and differ from the
    autonomous path): `timeout 0` makes the server wait indefinitely for the
    player's phase-done instead of running rampant (`timeout -1` skips the
    turn-done check entirely — srv_main.c), so the game advances only on the
    client's cue and stays deterministic; `autotoggle enabled` drops AI control
    of a player the moment the client takes it, so only the client's orders
    apply. `cmdlevel hack first` grants the (first, only) connection the rights
    to /take and /start. A `lua file` line dumps the ruleset id<->name table
    to the log for the client's name resolution.
    """
    lines = [
        "set timeout 0",
        "set minplayers 0",
        "set compresstype PLAIN",
        "set autotoggle enabled",
        "cmdlevel hack first",
        f"set endturn {endturn}",
    ]
    if saveturns is not None:
        lines.append('set autosaves "TURN|GAMEOVER"')
        lines.append(f"set saveturns {saveturns}")
    else:
        lines.append('set autosaves "GAMEOVER"')
    if scorelog:
        lines.append("set scorelog enabled")
    if skill is not None:
        if skill not in SKILL_LEVELS:
            raise ValueError(f"skill {skill!r} not in {SKILL_LEVELS}")
        lines.append(skill)
    for name, level in (skill_by_player or {}).items():
        if level not in SKILL_LEVELS:
            raise ValueError(f"skill {level!r} not in {SKILL_LEVELS}")
        lines.append(f'{level} "{name}"')
    if dump_lua_path is not None:
        lines.append(f"lua file {dump_lua_path}")
    lines += list(extra)
    return "\n".join(lines) + "\n"


def binary_sha256(binary: Path = DEFAULT_BINARY) -> str:
    """SHA-256 of the server binary file, recorded as run provenance."""
    h = hashlib.sha256()
    with open(binary, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
