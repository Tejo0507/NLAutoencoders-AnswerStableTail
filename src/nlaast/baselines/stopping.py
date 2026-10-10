"""Stopping rules and the matched-budget comparison they are judged by.

A stopping rule is a function from a trace's per-boundary evidence to a boundary
index (or ``None`` for "never stop"). Every candidate signal in the study -
answer convergence, the hidden-state probe, semantic entropy and the verbalised
NLA readout - is wrapped as one of these, so the comparison is between rules of
the same shape rather than between differently-shaped reported numbers.

**Matched budget.** Comparing each rule at its own natural operating point
would compare different token budgets, and a rule that stops later is trivially
safer. Instead each rule exposes a scalar score per boundary, the threshold is
swept, and rules are compared on the safety/saving curve - and at the specific
thresholds where mean tokens saved is equal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from ..data.answers import equivalent


@dataclass
class StopDecision:
    """Where a rule stopped on one problem, and whether that was safe."""

    problem_id: str
    rule: str
    #: Chunk index stopped at; ``None`` means the rule never fired.
    boundary: int | None
    tokens_used: int
    total_tokens: int
    answer_at_stop: str | None
    final_answer: str | None
    gold: str | None
    #: Outcome preserved: stopping did not change whether the answer is correct.
    safe: bool = False
    correct_at_stop: bool = False
    final_correct: bool = False
    score: float | None = None

    @property
    def tokens_saved_fraction(self) -> float:
        if not self.total_tokens:
            return 0.0
        return max(0.0, 1.0 - self.tokens_used / self.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tokens_saved_fraction"] = self.tokens_saved_fraction
        return d


def evaluate_stop(
    problem_id: str,
    rule: str,
    boundary: int | None,
    answer_at_stop: str | None,
    final_answer: str | None,
    gold: str | None,
    tokens_used: int,
    total_tokens: int,
    score: float | None = None,
) -> StopDecision:
    """Score one stopping decision.

    Safety is defined on *verified outcome*, not on answer identity: stopping is
    safe when the correctness of the answer you walk away with equals the
    correctness of the answer you would have got by letting the trace finish.
    Defining it as "same string" would count a rule that stops early on a wrong
    answer the model later fixes as safe, and would also count a lucky early
    stop on a problem the model later ruins as unsafe.
    """
    final_correct = equivalent(final_answer, gold)
    if boundary is None:
        return StopDecision(
            problem_id=problem_id, rule=rule, boundary=None,
            tokens_used=total_tokens, total_tokens=total_tokens,
            answer_at_stop=final_answer, final_answer=final_answer, gold=gold,
            safe=True, correct_at_stop=final_correct, final_correct=final_correct,
            score=score,
        )
    correct_at_stop = equivalent(answer_at_stop, gold)
    return StopDecision(
        problem_id=problem_id, rule=rule, boundary=boundary,
        tokens_used=tokens_used, total_tokens=total_tokens,
        answer_at_stop=answer_at_stop, final_answer=final_answer, gold=gold,
        safe=(correct_at_stop == final_correct),
        correct_at_stop=correct_at_stop, final_correct=final_correct, score=score,
    )


# --- Threshold sweeps at matched budget ---


@dataclass
class OperatingPoint:
    threshold: float
    n: int
    safe_rate: float
    accuracy: float
    mean_tokens_saved: float
    fire_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sweep_threshold(
    per_problem: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    higher_fires: bool = True,
) -> list[OperatingPoint]:
    """Safety/saving curve for a scored rule.

    Each element of ``per_problem`` is
    ``{'problem_id', 'scores': [...], 'boundary_tokens': [...], 'answers': [...],
       'final_answer', 'gold', 'total_tokens'}``. The rule fires at the first
    boundary whose score crosses the threshold.
    """
    points: list[OperatingPoint] = []
    for thr in thresholds:
        decisions: list[StopDecision] = []
        for row in per_problem:
            scores = row["scores"]
            fire_at = None
            for i, s in enumerate(scores):
                if s is None or not np.isfinite(s):
                    continue
                if (s >= thr) if higher_fires else (s <= thr):
                    fire_at = i
                    break
            if fire_at is None:
                decisions.append(
                    evaluate_stop(row["problem_id"], "sweep", None, None,
                                  row["final_answer"], row["gold"],
                                  row["total_tokens"], row["total_tokens"], thr)
                )
            else:
                decisions.append(
                    evaluate_stop(
                        row["problem_id"], "sweep", fire_at,
                        row["answers"][fire_at], row["final_answer"], row["gold"],
                        row["boundary_tokens"][fire_at], row["total_tokens"],
                        scores[fire_at],
                    )
                )
        n = len(decisions)
        if n == 0:
            continue
        points.append(
            OperatingPoint(
                threshold=float(thr),
                n=n,
                safe_rate=float(np.mean([d.safe for d in decisions])),
                accuracy=float(np.mean([d.correct_at_stop for d in decisions])),
                mean_tokens_saved=float(np.mean([d.tokens_saved_fraction for d in decisions])),
                fire_rate=float(np.mean([d.boundary is not None for d in decisions])),
            )
        )
    return points


#: How far an operating point may sit from the requested budget and still be
#: called matched, in mean fraction of tokens saved.
#:
#: Not a free parameter so much as the difference between a matched-budget
#: comparison and a mislabelled one. A rule's curve is a step function - it
#: fires at a boundary or it does not - and on a small corpus the steps are
#: wide. On the pilot, asking all four rules for a budget of 0.632 returned
#: convergence at 0.407 and the probe at 0.855: a 45-point spread in tokens
#: saved, compared as though the budgets were equal. Five points is tight
#: enough that the remaining difference is about safety, and loose enough to
#: keep adjacent grid points.
MATCH_TOLERANCE = 0.05


def at_matched_budget(
    curve: Sequence[OperatingPoint],
    target_saving: float,
    tolerance: float | None = MATCH_TOLERANCE,
) -> OperatingPoint | None:
    """The operating point closest to a given mean token saving.

    This is what makes the O3 comparison fair: every rule is read off its own
    curve at the *same* budget, so the reported difference is a difference in
    safety rather than in how aggressively the rule happens to be tuned.

    "Closest" is not the same as "close". A rule whose nearest achievable
    saving is far from the target cannot be compared at that target at all,
    and returning its nearest point anyway produces a table that looks matched
    and is not. ``None`` is returned instead; pass ``tolerance=None`` to get
    the old nearest-point behaviour for plotting a whole curve.
    """
    feasible = [p for p in curve if p.n > 0]
    if not feasible:
        return None
    best = min(feasible, key=lambda p: abs(p.mean_tokens_saved - target_saving))
    if tolerance is not None and abs(best.mean_tokens_saved - target_saving) > tolerance:
        return None
    return best


def budget_grid(curves: dict[str, Sequence[OperatingPoint]], n: int = 9) -> list[float]:
    """Savings levels every rule can actually reach.

    Taking the intersection of the per-rule achievable ranges avoids comparing
    one rule at an operating point another cannot reach, which would make the
    comparison an artefact of the threshold grid.
    """
    lows, highs = [], []
    for pts in curves.values():
        if not pts:
            continue
        vals = [p.mean_tokens_saved for p in pts]
        lows.append(min(vals))
        highs.append(max(vals))
    if not lows:
        return []
    lo, hi = max(lows), min(highs)
    if hi <= lo:
        return [float(lo)]
    return list(np.linspace(lo, hi, n))
