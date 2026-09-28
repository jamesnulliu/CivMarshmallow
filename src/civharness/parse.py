"""Parsers for the three machine-readable outputs of a run.

1. ranklog (`-R`): end-of-game turns / winners / losers.
2. savegame (plain text, INI-like): [scoreN] metric blocks, [playerN]
   identities, [game] turn — plus normalization for byte-level comparison.
3. scorelog (SCORELOG2): per-turn per-player time series (score, cities,
   techs, settlers, ...).
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# The only nondeterministic content in a savegame is wall-clock metadata:
# epoch timestamps in [event_cache] rows and last_turn_change_time.
# (Same patterns as tests/civharness/test_determinism.py, which stays
# import-free so the determinism check never depends on the package it
# validates.)
_TIMESTAMP_RE = re.compile(rb"\b(?:1[5-9]|2[0-2])\d{8}\b")
_TURNTIME_RE = re.compile(rb"last_turn_change_time=\d+")


def normalize_save_bytes(raw: bytes) -> bytes:
    """Replace the wall-clock fields of a savegame with fixed tokens."""
    raw = _TIMESTAMP_RE.sub(b"TS", raw)
    return _TURNTIME_RE.sub(b"last_turn_change_time=T", raw)


def normalized_save_sha(path: Path) -> str:
    """SHA-256 of a savegame after `normalize_save_bytes`: equal for two runs
    that reached the same game state."""
    return hashlib.sha256(normalize_save_bytes(Path(path).read_bytes())).hexdigest()


# --- ranklog ---------------------------------------------------------------


@dataclass(frozen=True)
class PlayerRank:
    name: str
    score: int | None
    raw: str


@dataclass(frozen=True)
class Ranklog:
    turns: int
    winners: tuple[PlayerRank, ...]
    losers: tuple[PlayerRank, ...]


def _parse_rank_entries(line: str) -> tuple[PlayerRank, ...]:
    # Entries look like "user,Name,user,score,," and are ",, "-separated.
    out = []
    for chunk in line.split(",,"):
        chunk = chunk.strip()
        if not chunk:
            continue
        f = chunk.split(",")
        score = int(f[3]) if len(f) > 3 and f[3].lstrip("-").isdigit() else None
        name = f[1] if len(f) > 1 else f[0]
        out.append(PlayerRank(name=name, score=score, raw=chunk))
    return tuple(out)


def parse_ranklog(path: Path) -> Ranklog:
    turns, winners, losers = 0, (), ()
    for line in Path(path).read_text().splitlines():
        if line.startswith("turns:"):
            turns = int(line.split(":", 1)[1])
        elif line.startswith("winners:"):
            winners = _parse_rank_entries(line.split(":", 1)[1])
        elif line.startswith("losers:"):
            losers = _parse_rank_entries(line.split(":", 1)[1])
    return Ranklog(turns=turns, winners=winners, losers=losers)


# --- savegame --------------------------------------------------------------

_SECTION_RE = re.compile(r"^\[([^\]]+)\]$")
_KV_RE = re.compile(r"^([A-Za-z_][\w.]*)=(.*)$")


def parse_save_sections(path: Path) -> dict[str, dict[str, str]]:
    """Tolerant INI pass: plain key=value lines per section; multi-line
    table values (`key={...}`) are captured raw under their key."""
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    table_key, table_lines = None, []
    for line in Path(path).read_text(errors="replace").splitlines():
        if table_key is not None:
            table_lines.append(line)
            if line.rstrip() == "}":
                current[table_key] = "\n".join(table_lines)
                table_key, table_lines = None, []
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = sections.setdefault(m.group(1), {})
            continue
        if current is None:
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if value.startswith("{") and not value.rstrip().endswith("}"):
            table_key, table_lines = key, [value]
        else:
            current[key] = value
    return sections


def unquote(s: str) -> str:
    """Strip surrounding whitespace and one pair of double quotes, if any."""
    s = s.strip()
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


_CELL_RE = re.compile(r'"[^"]*"|[^,]+')


def parse_table(raw: str) -> list[dict[str, str]]:
    """Freeciv save table: first line is quoted column names, then CSV rows.

    The header is tokenized with the SAME quote-aware splitter as the rows —
    a naive split(',') breaks on a quoted column name that itself contains a
    comma (e.g. "cma_minimal_surplus,1"), which then misaligns every column."""
    lines = [ln for ln in raw.splitlines() if ln.strip() and ln.strip() != "}"]
    if not lines:
        return []
    header = [unquote(c) for c in _CELL_RE.findall(lines[0].lstrip("{"))]
    rows = []
    for line in lines[1:]:
        cells = [unquote(c) for c in _CELL_RE.findall(line)]
        rows.append(dict(zip(header, cells)))
    return rows


def parse_vector(raw: str) -> list[str]:
    """A `*_vector="A_NONE","Alphabet",...` value -> ordered list of names, where
    list index == the id it maps (technology_vector / improvement_vector)."""
    return [unquote(c) for c in _CELL_RE.findall(raw or "")]


def bitstring_names(bits: str, names: list[str]) -> list[str]:
    """Names at the '1' positions of a 0/1 bitstring whose index == id (the
    savegame's `done` tech string and city `improvements` string)."""
    return [names[i] for i, ch in enumerate(bits or "") if ch == "1" and i < len(names)]


def save_research(sections: dict) -> dict[int, dict]:
    """{research number: {done, now, goal}} from the [research] `r={}` table.
    `done` is the known-tech bitstring; join to a player by player.team_no."""
    out: dict[int, dict] = {}
    sec = sections.get("research", {})
    if "r" in sec:
        for row in parse_table(sec["r"]):
            num = row.get("number", "")
            if not num.lstrip("-").isdigit():
                continue
            out[int(num)] = {
                "done": row.get("done", ""),
                "now": row.get("now_name"),
                "goal": row.get("goal_name"),
            }
    return out


def save_terrain(sections: dict) -> dict | None:
    """[map] terrain as {xsize, ysize, grid:[row strings], legend:{ident: name}}.
    Tile (x,y) = grid[y][x]; its terrain name = legend[grid[y][x]]. Dimensions
    are read off the rows themselves (row length = xsize, row count = ysize)."""
    mp = sections.get("map", {})
    rows, i = [], 0
    while f"t{i:04d}" in mp:
        rows.append(unquote(mp[f"t{i:04d}"]))
        i += 1
    if not rows:
        return None
    legend = {}
    terr = sections.get("savefile", {}).get("terrident")
    if terr:
        for row in parse_table(terr):
            legend[row.get("identifier")] = row.get("name")
    return {"xsize": len(rows[0]), "ysize": len(rows), "grid": rows, "legend": legend}


def save_turn(sections: dict) -> int:
    """The `[game] turn` of a parsed save."""
    return int(sections["game"]["turn"])


def save_players(sections: dict) -> dict[int, str]:
    """{player index: name} from [playerN] sections — the only safe source of
    names for per-player skill commands."""
    out = {}
    for key, sec in sections.items():
        m = re.fullmatch(r"player(\d+)", key)
        if m and "name" in sec:
            out[int(m.group(1))] = unquote(sec["name"])
    return out


def save_scores(sections: dict) -> dict[str, dict[str, int]]:
    """{player name: {metric: value}} joining [scoreN] with [playerN]."""
    out = {}
    for sec, kv in sections.items():
        m = re.fullmatch(r"score(\d+)", sec)
        if not m:
            continue
        player = sections.get(f"player{m.group(1)}", {})
        name = unquote(player.get("name", f"player{m.group(1)}"))
        out[name] = {k: int(v) for k, v in kv.items() if v.lstrip("-").isdigit()}
    return out


# --- scorelog --------------------------------------------------------------


@dataclass
class Scorelog:
    tags: dict[int, str] = field(default_factory=dict)
    players: dict[int, str] = field(default_factory=dict)
    # data[tag_name][player_id] -> list of (turn, value)
    data: dict[str, dict[int, list[tuple[int, int]]]] = field(default_factory=dict)

    def series(self, tag: str, player: int | str) -> list[tuple[int, int]]:
        if isinstance(player, str):
            ids = [i for i, n in self.players.items() if n == player]
            if not ids:
                raise KeyError(f"no player named {player!r}")
            player = ids[0]
        return self.data.get(tag, {}).get(player, [])


def parse_scorelog(path: Path) -> Scorelog:
    log = Scorelog()
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts or line.startswith("#"):
            continue
        if parts[0] == "tag":
            log.tags[int(parts[1])] = parts[2]
        elif parts[0] == "addplayer":
            log.players[int(parts[2])] = " ".join(parts[3:])
        elif parts[0] == "data":
            turn, tag_id, player_id, value = (
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
            )
            tag = log.tags.get(tag_id, str(tag_id))
            log.data.setdefault(tag, {}).setdefault(player_id, []).append((turn, value))
    return log
