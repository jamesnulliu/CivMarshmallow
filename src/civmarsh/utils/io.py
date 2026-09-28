"""JSONL reading and writing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_jsonl(path) -> Iterator[dict]:
    """Yield one record per non-blank line of a JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path) -> list[dict]:
    """All records of a JSONL file (blank lines are skipped)."""
    return list(iter_jsonl(path))


def write_jsonl(path, rows: Iterable[dict], *, sort_keys: bool = False) -> Path:
    """Write `rows` to `path`, one JSON object per line, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r, sort_keys=sort_keys) + "\n" for r in rows)
    return path


def append_jsonl(path, row: dict, *, sort_keys: bool = False) -> None:
    """Append one record to a JSONL file and flush it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=sort_keys) + "\n")
        f.flush()
