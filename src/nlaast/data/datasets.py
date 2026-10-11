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
from typing import Iterable, Literal, Sequence

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


def interleave_by_dataset(groups: Sequence[Sequence[Problem]]) -> list[Problem]:
    """Order the problem set so that every prefix of it is representative.

    Every stage checkpoints per problem and resumes, so the achieved N is
    whatever completed - which is only a usable sample if stopping early
    leaves a balanced corpus. Concatenating the datasets does not: it puts all
    150 GSM8K problems before all 100 MATH ones, so a run halted at 120 has no
    MATH at all. That silently rules out F2, which compares tail fraction
    between benchmarks, and biases every pooled quantity towards the easier
    dataset.

    Each group is therefore spread evenly across the whole order by largest
    remainder on its own position, which keeps each prefix close to the
    configured proportions. The order is a deterministic function of the
    group sizes, so it does not depend on the seed and a resumed run keeps
    the order it started with.
    """
    present = [list(g) for g in groups if g]
    if len(present) <= 1:
        return [p for g in present for p in g]
    total = sum(len(g) for g in present)
    # Position each item at (its index + 0.5) / group size, so groups of
    # different sizes interleave in proportion rather than in blocks.
    keyed = [
        ((i + 0.5) / len(g), gi, i, p)
        for gi, g in enumerate(present)
        for i, p in enumerate(g)
    ]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    assert len(keyed) == total
    return [t[3] for t in keyed]


def stratified_cap(rows: Sequence[dict], cap: int, key: str = "dataset") -> list[dict]:
    """Take ``cap`` rows while keeping each stratum's share of the whole.

    An arm that can only afford a subset still has to be a subset *of this
    corpus*. Taking a prefix of the ids does not qualify: problem ids sort as
    ``gsm8k-...`` before ``math-...``, so capping 220 problems at 80 by sorted
    id yields 80 GSM8K problems and no MATH. The NLA arm is capped this way and
    the primary O3 head-to-head is computed on the problems it covers, so that
    prefix would have made the study's central comparison a GSM8K-only result
    reported as a corpus-wide one.

    Each stratum therefore gets its proportional share by largest remainder,
    filled with a deterministic prefix of its own sorted ids so that a resumed
    or extended run keeps the same subset instead of drifting onto new
    problems.
    """
    rows = list(rows)
    if cap <= 0 or len(rows) <= cap:
        return rows
    strata: dict[str, list[dict]] = {}
    for r in rows:
        strata.setdefault(str(r.get(key)), []).append(r)
    names = sorted(strata)
    exact = {s: cap * len(strata[s]) / len(rows) for s in names}
    take = {s: min(len(strata[s]), int(exact[s])) for s in names}
    # Largest remainder, then by stratum name, so the result is deterministic.
    while sum(take.values()) < cap:
        candidates = [s for s in names if take[s] < len(strata[s])]
        if not candidates:
            break
        s = max(candidates, key=lambda s: (exact[s] - take[s], -names.index(s)))
        take[s] += 1
    out: list[dict] = []
    for s in names:
        out += sorted(strata[s], key=lambda r: str(r.get("problem_id", "")))[:take[s]]
    return out


def load_problems(cfg: DataConfig) -> list[Problem]:
    """The study's problem set: validated, subsampled, deterministically split.

    Returned interleaved by dataset; see ``interleave_by_dataset`` for why the
    order is part of the design rather than an accident of concatenation.
    """
    groups: list[list[Problem]] = []
    if "gsm8k" in cfg.datasets:
        groups.append(_take_stratified(load_gsm8k(cfg), cfg.n_gsm8k, cfg.seed))
    if "math" in cfg.datasets:
        groups.append(_take_stratified(load_math(cfg), cfg.n_math, cfg.seed))
    return interleave_by_dataset(groups)


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
