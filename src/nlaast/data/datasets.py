"""Benchmark loading, validation and deterministic splitting.

GSM8K and MATH are staged as JSONL under ``data/raw/`` by
``scripts/prepare_problems.py``. This module turns them into a single validated
``Problem`` list with a reproducible train/eval split.

The split is by problem id hash, not by shuffling: adding problems later does
not reshuffle the ones already assigned, so a probe fitted in an earlier run is
still evaluated on genuinely held-out problems in a later, larger run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal

from .. import paths
from ..config import DataConfig
from .answers import extract_gold

Split = Literal["train", "eval"]


@dataclass(frozen=True)
class Problem:
    id: str
    dataset: str
    question: str
    gold: str
    split: Split
    #: MATH only; ``None`` for GSM8K. Used by falsification test F2.
    level: str | None = None
    subject: str | None = None
    gold_solution: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _bucket(problem_id: str, salt: int) -> float:
    h = hashlib.sha256(f"{salt}:{problem_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _assign_split(problem_id: str, cfg: DataConfig) -> Split:
    return "train" if _bucket(problem_id, cfg.seed) < cfg.train_fraction else "eval"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python scripts/prepare_problems.py` first"
        )
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k(cfg: DataConfig) -> list[Problem]:
    rows = _read_jsonl(paths.DATA_RAW / "gsm8k" / f"{cfg.split}.jsonl")
    out: list[Problem] = []
    for i, r in enumerate(rows):
        gold = extract_gold(r, "gsm8k")
        if gold is None:
            continue  # counted by validate(); a gold-less record is unusable
        pid = f"gsm8k-{cfg.split}-{i:05d}"
        out.append(
            Problem(
                id=pid,
                dataset="gsm8k",
                question=r["question"].strip(),
                gold=gold,
                split=_assign_split(pid, cfg),
                gold_solution=r.get("answer"),
            )
        )
    return out


def load_math(cfg: DataConfig) -> list[Problem]:
    out: list[Problem] = []
    for subject in cfg.math_subjects:
        path = paths.DATA_RAW / "math" / cfg.split / f"{subject}.jsonl"
        for i, r in enumerate(_read_jsonl(path)):
            gold = extract_gold(r, "math")
            if gold is None:
                continue
            pid = f"math-{subject}-{cfg.split}-{i:05d}"
            out.append(
                Problem(
                    id=pid,
                    dataset="math",
                    question=r["problem"].strip(),
                    gold=gold,
                    split=_assign_split(pid, cfg),
                    level=r.get("level"),
                    subject=r.get("type") or subject,
                    gold_solution=r.get("solution"),
                )
            )
    return out


def _take_stratified(problems: list[Problem], n: int, salt: int) -> list[Problem]:
    """Deterministic subsample preserving the train/eval ratio.

    Sorting by a hash rather than slicing the head avoids inheriting whatever
    ordering the benchmark file happened to have (MATH files are grouped by
    level, so a head slice would be all easy problems).
    """
    if n <= 0 or n >= len(problems):
        return sorted(problems, key=lambda p: p.id)
    ordered = sorted(problems, key=lambda p: (_bucket(p.id, salt + 1), p.id))
    return sorted(ordered[:n], key=lambda p: p.id)


def load_problems(cfg: DataConfig) -> list[Problem]:
    """The study's problem set: validated, subsampled, deterministically split."""
    out: list[Problem] = []
    if "gsm8k" in cfg.datasets:
        out += _take_stratified(load_gsm8k(cfg), cfg.n_gsm8k, cfg.seed)
    if "math" in cfg.datasets:
        out += _take_stratified(load_math(cfg), cfg.n_math, cfg.seed)
    return out


def validate(problems: Iterable[Problem]) -> dict:
    """Integrity report written into the stage manifest."""
    problems = list(problems)
    ids = [p.id for p in problems]
    report = {
        "n": len(problems),
        "n_unique_ids": len(set(ids)),
        "duplicate_ids": sorted({i for i in ids if ids.count(i) > 1})[:20],
        "by_dataset": {},
        "by_split": {},
        "by_level": {},
        "empty_questions": sum(1 for p in problems if not p.question.strip()),
        "empty_golds": sum(1 for p in problems if not str(p.gold).strip()),
        "max_question_chars": max((len(p.question) for p in problems), default=0),
    }
    for p in problems:
        report["by_dataset"][p.dataset] = report["by_dataset"].get(p.dataset, 0) + 1
        report["by_split"][p.split] = report["by_split"].get(p.split, 0) + 1
        if p.level:
            report["by_level"][p.level] = report["by_level"].get(p.level, 0) + 1
    report["ok"] = (
        report["n"] > 0
        and report["n_unique_ids"] == report["n"]
        and report["empty_questions"] == 0
        and report["empty_golds"] == 0
    )
    return report
