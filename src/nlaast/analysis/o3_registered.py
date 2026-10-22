"""O3 re-scored on the registered outcome (D69). Pure numpy; no I/O.

Implements ``docs/O3_REGISTERED_REANALYSIS_PLAN.md``. Every rule is a
"stop at the first allowed boundary whose score reaches the threshold" rule, as
in ``baselines.stopping``. This module computes the same curves in matrix
form, so that a paired problem bootstrap can rebuild and re-match every curve
thousands of times.

**Why the matrix form is exact.** On one problem, a rule with threshold ``t``
stops at the first allowed boundary ``i`` with ``score_i >= t``. That boundary
changes only when ``t`` crosses a *record*: a score larger than every allowed
score before it. Record ``r`` is chosen for every ``t`` in
``(hi_{r-1}, hi_r]``, and nothing is chosen above the largest record. So for
the sorted thresholds ``T``, the stopping boundary is
``records[searchsorted(hi, T)]``. That is the same decision ``decisions_at``
makes, which ``tests/test_o3_registered.py`` checks against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..baselines.stopping import OutcomeUnknown


@dataclass
class Problem:
    """One problem's boundaries: what a stop there saves, and whether it is safe."""

    problem_id: str
    saving: np.ndarray      # float, per boundary: 1 - prefix_tokens / total
    safe: np.ndarray        # bool, per boundary, under the outcome in use
    known: np.ndarray       # bool, per boundary: was the outcome measured?

    @property
    def n(self) -> int:
        return len(self.saving)


@dataclass
class RuleMatrix:
    """A rule's whole curve, per problem: one column per distinct threshold."""

    thresholds: np.ndarray  # (K,), ascending; the last is +inf (never stops)
    saving: np.ndarray      # (n, K)
    safe: np.ndarray        # (n, K) bool
    fired: np.ndarray       # (n, K) bool
    boundary: np.ndarray    # (n, K) int, -1 where not fired

    def curve(self, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean saving and safe rate at every threshold, optionally with
        problem weights (bootstrap counts)."""
        if weights is None:
            return self.saving.mean(0), self.safe.mean(0)
        w = weights / weights.sum()
        return w @ self.saving, w @ self.safe


def _records(scores: Sequence[float | None], allowed: np.ndarray) -> tuple[list[int], list[float]]:
    best = -np.inf
    idx, hi = [], []
    for i, s in enumerate(scores):
        if not allowed[i] or s is None or not np.isfinite(s):
            continue
        if s > best:
            idx.append(i)
            hi.append(float(s))
            best = s
    return idx, hi


def rule_matrix(
    problems: Sequence[Problem],
    scores: Mapping[str, Sequence[float | None]],
    menu: Mapping[str, np.ndarray] | None = None,
) -> RuleMatrix:
    """Every operating point of a thresholded rule, per problem.

    ``menu[pid]`` is a boolean mask of the boundaries the rule may stop at
    (design A shares one menu across rules). ``None`` allows every boundary.
    A problem with no allowed, scored boundary never stops. Raises
    ``OutcomeUnknown`` if some threshold would stop the rule at a boundary
    whose outcome was not measured.
    """
    recs = []
    values: list[float] = []
    for p in problems:
        allowed = np.ones(p.n, bool) if menu is None else np.asarray(menu[p.problem_id], bool)
        idx, hi = _records(scores[p.problem_id], allowed)
        for i in idx:
            if not p.known[i]:
                raise OutcomeUnknown(f"{p.problem_id}: boundary {i} can be stopped at "
                                     f"but its outcome was not measured")
        recs.append((np.array(idx, int), np.array(hi, float)))
        values.extend(hi)
    T = np.array(sorted(set(values)) + [np.inf])
    n, K = len(problems), len(T)
    saving = np.zeros((n, K))
    safe = np.ones((n, K), bool)
    fired = np.zeros((n, K), bool)
    boundary = np.full((n, K), -1, int)
    for r, (p, (idx, hi)) in enumerate(zip(problems, recs)):
        if not len(idx):
            continue
        k = np.searchsorted(hi, T, side="left")
        f = k < len(idx)
        b = idx[np.minimum(k, len(idx) - 1)]
        fired[r] = f
        boundary[r] = np.where(f, b, -1)
        saving[r] = np.where(f, p.saving[b], 0.0)
        safe[r] = np.where(f, p.safe[b], True)
    return RuleMatrix(T, saving, safe, fired, boundary)


def match(saving_curve: np.ndarray, target: float, tol: float) -> int | None:
    """Column whose mean saving is closest to ``target``, if within ``tol``."""
    k = int(np.argmin(np.abs(saving_curve - target)))
    return k if abs(saving_curve[k] - target) <= tol else None


def oracle_bound(problems: Sequence[Problem], menu: Mapping[str, np.ndarray] | None = None
                 ) -> float:
    """Mean over problems of the largest saving at an allowed, safe boundary.

    An allowed boundary whose outcome is unknown makes the bound undefined
    rather than smaller, so it raises.
    """
    best = []
    for p in problems:
        allowed = np.ones(p.n, bool) if menu is None else np.asarray(menu[p.problem_id], bool)
        if np.any(allowed & ~p.known):
            raise OutcomeUnknown(f"{p.problem_id}: an allowed boundary has no measured outcome")
        ok = allowed & p.safe
        best.append(float(p.saving[ok].max()) if ok.any() else 0.0)
    return float(np.mean(best))


def no_signal_expectations(
    problems: Sequence[Problem],
    menu: Mapping[str, np.ndarray] | None,
    qgrid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected saving and safe rate of the random-score rule, per problem and ``q``.

    With i.i.d. Uniform(0, 1) scores and threshold ``1 - q``, each allowed
    boundary qualifies with probability ``q``. The rule stops at the ``k``-th
    allowed boundary (0-based) with probability ``q (1-q)^k`` and never with
    probability ``(1-q)^m``, which is safe. Returns two ``(n, len(qgrid))``
    arrays.
    """
    q = np.asarray(qgrid, float)
    ES = np.zeros((len(problems), len(q)))
    EQ = np.zeros((len(problems), len(q)))
    for r, p in enumerate(problems):
        allowed = np.ones(p.n, bool) if menu is None else np.asarray(menu[p.problem_id], bool)
        pos = np.flatnonzero(allowed)
        if np.any(~p.known[pos]):
            raise OutcomeUnknown(f"{p.problem_id}: an allowed boundary has no measured outcome")
        m = len(pos)
        if m == 0:
            EQ[r] = 1.0
            continue
        k = np.arange(m)[:, None]
        pk = q[None, :] * (1 - q[None, :]) ** k          # (m, G)
        ES[r] = p.saving[pos] @ pk
        EQ[r] = p.safe[pos].astype(float) @ pk + (1 - q) ** m
    return ES, EQ


def no_signal_monte_carlo(
    problems: Sequence[Problem],
    menu: Mapping[str, np.ndarray] | None,
    targets: Sequence[float],
    tol: float,
    draws: int,
    rng: np.random.Generator,
) -> list[float | None]:
    """The plan's original no-signal rule: random scores, matched per draw.

    Used only to check the closed form on the full sample (plan deviation 1).
    """
    out: list[list[float]] = [[] for _ in targets]
    for _ in range(draws):
        scores = {p.problem_id: list(rng.random(p.n)) for p in problems}
        rm = rule_matrix(problems, scores, menu)
        sav, saf = rm.curve()
        for j, b in enumerate(targets):
            k = match(sav, b, tol)
            if k is not None:
                out[j].append(float(saf[k]))
    return [float(np.mean(v)) if v else None for v in out]


def gap_over_budgets(
    curve_a: tuple[np.ndarray, np.ndarray],
    curve_b: tuple[np.ndarray, np.ndarray],
    budgets: Sequence[float],
    tol: float,
) -> tuple[float, int, list[float | None]]:
    """Mean safe-rate gap (a − b) over the budgets where both are matched."""
    diffs: list[float | None] = []
    for b in budgets:
        ka, kb = match(curve_a[0], b, tol), match(curve_b[0], b, tol)
        diffs.append(None if ka is None or kb is None
                     else float(curve_a[1][ka] - curve_b[1][kb]))
    got = [d for d in diffs if d is not None]
    return (float(np.mean(got)) if got else float("nan")), len(got), diffs


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm step-down adjusted p-values and decisions."""
    items = sorted(((k, v) for k, v in pvalues.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    for k, v in pvalues.items():
        if k not in out:
            out[k] = {"p": v, "p_holm": None, "reject": None}
    return out


def bootstrap_p(replicates: np.ndarray, floor: float) -> float:
    """Two-sided bootstrap p: 2 * min(P(Δ* <= 0), P(Δ* >= 0)), floored."""
    r = replicates[np.isfinite(replicates)]
    if not len(r):
        return float("nan")
    return float(max(floor, min(1.0, 2 * min(np.mean(r <= 0), np.mean(r >= 0)))))
