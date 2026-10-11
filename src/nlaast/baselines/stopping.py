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


def decisions_at(
    per_problem: Sequence[dict[str, Any]],
    threshold: float,
    higher_fires: bool = True,
) -> list[StopDecision]:
    """Where each problem's rule fires at one threshold, and whether it was safe.

    An operating point reports only rates, which is enough to draw a curve and
    not enough to test one: the matched-budget comparison's outcome is binary
    and per problem, so the test has to see the individual problems. This
    returns them, in the order given, for one point on the curve.
    """
    out: list[StopDecision] = []
    for row in per_problem:
        scores = row["scores"]
        fire_at = None
        for i, s in enumerate(scores):
            if s is None or not np.isfinite(s):
                continue
            if (s >= threshold) if higher_fires else (s <= threshold):
                fire_at = i
                break
        if fire_at is None:
            out.append(
                evaluate_stop(row["problem_id"], "sweep", None, None,
                              row["final_answer"], row["gold"],
                              row["total_tokens"], row["total_tokens"], threshold)
            )
        else:
            out.append(
                evaluate_stop(
                    row["problem_id"], "sweep", fire_at,
                    row["answers"][fire_at], row["final_answer"], row["gold"],
                    row["boundary_tokens"][fire_at], row["total_tokens"],
                    scores[fire_at],
                )
            )
    return out


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
        decisions = decisions_at(per_problem, thr, higher_fires)
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


def complete_threshold_grid(values: Sequence[float]) -> list[float]:
    """Every threshold that can produce a distinct operating point.

    A rule's curve changes only at a threshold that crosses an observed score:
    between two adjacent observed values no problem can change the boundary it
    fires at, so no threshold between them yields a new operating point.
    Sweeping the observed values therefore enumerates the *whole* curve, and
    does so exactly.

    This replaces a fixed-size linear grid, which samples the score axis
    uniformly and so depends on how the scores happen to be distributed rather
    than on the rule. The pilot shows why that mattered: the hidden-state
    probe's scores are logistic and saturate near 0 and 1, so 25 evenly spaced
    thresholds found 5 operating points and *none* within tolerance of a
    shared budget, and the probe's head-to-head was reported as unevaluable.
    The same 21 problems swept at their observed values give 86 operating
    points and match at all nine budgets. The comparison was never missing
    data; it was missing thresholds.

    The converse case is as important. With the complete grid, a curve's
    operating points are exhaustive, so a rule that still cannot be matched is
    not being failed by the sweep. ``granularity_bound`` is what distinguishes
    the two reasons that can happen.
    """
    finite = sorted({float(v) for v in values if v is not None and np.isfinite(v)})
    if not finite:
        return []
    eps = 1e-9
    # One threshold below the minimum gives the "fires wherever it can" end;
    # each observed value nudged upwards gives the point just past that score.
    return [finite[0] - eps] + [v + eps for v in finite]


def granularity_bound(values: Sequence[float]) -> int:
    """Most operating points a rule's curve can ever have, at any corpus size.

    A score taking ``d`` distinct values admits at most ``d + 1`` thresholds
    that behave differently, so the curve has at most ``d + 1`` points however
    many problems are added. This separates a comparison that a larger corpus
    would unblock from one no corpus size can.

    The agreement rule is the case in point: its score is 1.0 once the last
    ``window`` answers agree and 0.0 otherwise, so ``d = 2`` and its curve has
    three points at any N. On the pilot those are savings of 0.407 and 0.885,
    which straddle the matched-budget window without landing in it. That is a
    property of a binary score, not of 21 problems.
    """
    return len({float(v) for v in values if v is not None and np.isfinite(v)}) + 1


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

    Operating points that save nothing are excluded. Every complete curve has
    one - the threshold above which the rule never fires - and at that point
    no rule stops, so every rule is trivially safe and the difference between
    them is identically zero. Including it does not make the comparison
    conservative, it makes it vacuous: the budget the rules are matched at has
    to be a budget. The floor is therefore the smallest saving every rule can
    reach *while actually stopping somewhere*.
    """
    lows, highs = [], []
    for pts in curves.values():
        vals = [p.mean_tokens_saved for p in pts if p.mean_tokens_saved > 0]
        if not vals:
            continue
        lows.append(min(vals))
        highs.append(max(vals))
    if not lows:
        return []
    lo, hi = max(lows), min(highs)
    if hi <= lo:
        return [float(lo)]
    return list(np.linspace(lo, hi, n))
