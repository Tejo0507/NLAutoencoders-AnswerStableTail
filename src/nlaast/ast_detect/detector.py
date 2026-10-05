"""The Answer-Stable Tail.

Operational definition, fixed in PROJECT_PLAN.md section 5 before any data was
collected. Boundary ``i`` starts the tail iff:

1. **Truncation equivalence** - forcing an answer from prefix ``P_i`` gives an
   answer equivalent to the full trace's final answer.
2. **Resampled-continuation stability** - ``K`` independent continuations
   sampled from ``P_i`` all reach an equivalent answer. Mo et al. show the plain
   agreement rule stops on answers the trajectory later abandons at a rate that
   does not vanish, so this control is required rather than optional.
3. **Persistence** - 1 and 2 hold for every later boundary too.

The module takes evidence as data and decides. It never calls a model itself:
the caller (``scripts/detect_answer_stable_tail.py``) owns generation, which keeps the
decision logic pure and unit-testable, and keeps this package free of any
activation dependency.

Edge cases get a status code and stay in the corpus. Dropping ``incorrect_final``
problems would be the easy choice and the wrong one - the construct does not
presuppose correctness, and excluding them would bias every stopping comparison
toward problems the model already got right.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Sequence

from ..data.answers import equivalent


class TailStatus(str, Enum):
    OK = "ok"
    #: No boundary satisfies the criteria - the answer never settles.
    NO_STABLE_POINT = "no_stable_point"
    #: Stable from the very first chunk; there is no reasoning before the tail.
    STABLE_AT_ZERO = "stable_at_zero"
    #: The full trace has no parseable final answer.
    UNPARSEABLE_FINAL = "unparseable_final"
    #: Generation hit the token cap, so "the final answer" may not be final.
    TRUNCATED = "truncated"
    #: Fewer chunks than the criteria need.
    TOO_SHORT = "too_short"


@dataclass
class BoundaryEvidence:
    """Everything observed at one candidate boundary. Persisted for auditing."""

    index: int
    #: Intermediate answer parsed from the prefix as written.
    parsed_answer: str | None = None
    #: Answer obtained by forcing a conclusion from the prefix.
    forced_answer: str | None = None
    forced_matches_final: bool = False
    #: One answer per resampled continuation.
    continuation_answers: list[str | None] = field(default_factory=list)
    continuations_matching: int = 0
    prefix_tokens: int = 0

    @property
    def n_continuations(self) -> int:
        return len(self.continuation_answers)

    @property
    def continuations_unanimous(self) -> bool:
        return self.n_continuations > 0 and self.continuations_matching == self.n_continuations

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["continuations_unanimous"] = self.continuations_unanimous
        return d


@dataclass
class ASTResult:
    problem_id: str
    status: TailStatus
    n_chunks: int
    final_answer: str | None
    final_correct: bool
    #: First chunk index of the tail, or ``None`` when there is no tail.
    tail_start: int | None
    tail_fraction: float
    tail_tokens: int
    total_tokens: int
    evidence: list[BoundaryEvidence]
    #: Liu & Wang's agreement rule, recorded here so the baseline and the AST
    #: are computed from exactly the same parse of the same trace.
    convergence_start: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "problem_id": self.problem_id,
            "status": self.status.value,
            "n_chunks": self.n_chunks,
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "tail_start": self.tail_start,
            "tail_fraction": self.tail_fraction,
            "tail_tokens": self.tail_tokens,
            "total_tokens": self.total_tokens,
            "convergence_start": self.convergence_start,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }


def convergence_boundary(
    parsed: Sequence[str | None], window: int
) -> int | None:
    """Liu & Wang: first ``i`` where ``window`` consecutive answers agree.

    This is a *baseline*, not the AST. It sees only the parsed intermediate
    answers - no forcing, no resampling - which is exactly the weakness the AST
    definition is built to address.
    """
    if window < 1:
        raise ValueError("convergence window must be >= 1")
    n = len(parsed)
    for i in range(n - window + 1):
        block = parsed[i : i + window]
        if any(a is None for a in block):
            continue
        if all(equivalent(block[0], b) for b in block[1:]):
            return i + window - 1
    return None


def detect_ast(
    problem_id: str,
    evidence: Sequence[BoundaryEvidence],
    final_answer: str | None,
    gold: str | None,
    n_chunks: int,
    total_tokens: int,
    truncated: bool,
    require_unanimous: bool = True,
    convergence_window: int = 2,
    min_boundary_fraction: float = 0.0,
) -> ASTResult:
    """Decide the tail from collected boundary evidence."""
    parsed = [e.parsed_answer for e in evidence]
    conv = convergence_boundary(parsed, convergence_window)
    final_correct = equivalent(final_answer, gold)

    def _result(status: TailStatus, start: int | None, **notes: Any) -> ASTResult:
        if start is None:
            frac, tail_tokens = 0.0, 0
        else:
            frac = (n_chunks - start) / n_chunks if n_chunks else 0.0
            prefix = evidence[start - 1].prefix_tokens if start > 0 else 0
            tail_tokens = max(0, total_tokens - prefix)
        return ASTResult(
            problem_id=problem_id,
            status=status,
            n_chunks=n_chunks,
            final_answer=final_answer,
            final_correct=final_correct,
            tail_start=start,
            tail_fraction=frac,
            tail_tokens=tail_tokens,
            total_tokens=total_tokens,
            evidence=list(evidence),
            convergence_start=conv,
            notes=notes,
        )

    if final_answer is None:
        return _result(TailStatus.UNPARSEABLE_FINAL, None,
                       reason="no parseable final answer in the full trace")
    if n_chunks < 2:
        return _result(TailStatus.TOO_SHORT, None, reason=f"{n_chunks} chunk(s)")

    min_index = int(min_boundary_fraction * n_chunks)

    def qualifies(e: BoundaryEvidence) -> bool:
        if not e.forced_matches_final:
            return False
        if e.n_continuations == 0:
            # No resampling evidence means criterion 2 is untested, and an
            # untested criterion is not a satisfied one.
            return False
        if require_unanimous:
            return e.continuations_unanimous
        return e.continuations_matching > e.n_continuations / 2

    ok = [qualifies(e) for e in evidence]

    # Criterion 3: persistence. Scan backwards for the longest all-qualifying
    # suffix, so a single later failure correctly disqualifies everything
    # before it.
    start: int | None = None
    for i in range(len(ok) - 1, min_index - 1, -1):
        if not ok[i]:
            break
        start = evidence[i].index

    if start is None:
        return _result(TailStatus.NO_STABLE_POINT, None,
                       n_qualifying=sum(ok), reason="no persistent stable suffix")

    status = TailStatus.STABLE_AT_ZERO if start == 0 else TailStatus.OK
    res = _result(status, start, n_qualifying=sum(ok))
    if truncated:
        # Keep the measured tail but mark it: the "final" answer came from a
        # capped generation, so the tail may be an artefact of the cap.
        res.status = TailStatus.TRUNCATED
        res.notes["underlying_status"] = status.value
    return res


def matched_windows(
    tail_start: int,
    n_chunks: int,
    rng,
) -> dict[str, tuple[int, int]]:
    """Control windows for a detected tail, as chunk index ranges (inclusive).

    *matched_position* sits immediately before stabilisation in the same trace,
    so it controls for position-in-trace. *matched_length* sits at a random
    valid start, controlling for window length alone. Both are the same length
    as the tail; without them a difference measured on the tail could be a
    difference about where in a trace you look.
    """
    length = n_chunks - tail_start
    out: dict[str, tuple[int, int]] = {"tail": (tail_start, n_chunks - 1)}

    pre_end = tail_start - 1
    if pre_end >= 0:
        pre_start = max(0, pre_end - length + 1)
        out["matched_position"] = (pre_start, pre_end)

    max_start = n_chunks - length
    if max_start > 0:
        rand_start = int(rng.integers(0, max_start + 1))
        out["matched_length"] = (rand_start, rand_start + length - 1)
    return out
