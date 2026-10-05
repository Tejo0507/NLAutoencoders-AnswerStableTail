"""Logging and resumable JSONL records.

Stages are long and interruptible, so every stage writes one JSON object per
unit of work to a ``.jsonl`` shard as it goes. On resume the shard is read back
and completed ids are skipped. This is the whole resume mechanism - there is no
separate checkpoint format to keep in sync.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

_CONFIGURED = False


def setup(run_dir: Path | None = None, level: int = logging.INFO,
          name: str = "nlaast") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def get(name: str = "nlaast") -> logging.Logger:
    return logging.getLogger(name)


# --- JSONL ---


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL shard, tolerating a truncated final line.

    A kill during a write leaves a partial last record. Dropping it is correct:
    the id it belonged to simply looks incomplete and gets redone on resume.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                get().warning("dropping truncated JSONL record in %s", path.name)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


class JsonlWriter:
    """Append-only writer that flushes per record.

    Flushing every record costs throughput but the units of work here are whole
    model generations taking seconds, so the cost is irrelevant next to losing
    an hour of generation to a buffered write.
    """

    def __init__(self, path: Path, key: str = "id"):
        self.path = path
        self.key = key
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = {
            str(r[key]) for r in read_jsonl(path) if key in r
        }
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def has(self, key_value: Any) -> bool:
        return str(key_value) in self.done

    def write(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter used outside its context manager")
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        if self.key in row:
            self.done.add(str(row[self.key]))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
