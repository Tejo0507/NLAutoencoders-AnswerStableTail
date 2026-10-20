"""Selective risk at matched coverage (D83 deviation 1). Pure numpy; no I/O.

The registered O3 estimand matches rules on *mean* saving and counts a rule
that never stops as safe. That rewards concentrating savings in a few
problems. Selective risk asks a different question: among the problems a rule
stops in before the end, how often is the stop unsafe, at the same share of
problems stopped (coverage)? Mean saving among the stops is reported beside
it, because a rule can lower its selective risk by stopping only late.

**No signal in closed form.** Under i.i.d. uniform scores and threshold
``1 − q``, a problem with ``m`` allowed boundaries fires with probability
``1 − (1 − q)^m``. Its expected safe rate (``no_signal_expectations``) counts
the no-stop event as safe, so the probability of an *unsafe* stop is
``1 − EQ``. Ratios of weighted sums then give coverage, selective risk and
saving among stops for any bootstrap weights.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .o3_registered import Problem, RuleMatrix


def fire_probability(problems: Sequence[Problem], menu: Mapping[str, np.ndarray] | None,
                     qgrid: np.ndarray) -> np.ndarray:
    """``(n, G)``: the no-signal rule's probability of stopping before the end."""
    q = np.asarray(qgrid, float)
    out = np.zeros((len(problems), len(q)))
    for r, p in enumerate(problems):
        m = p.n if menu is None else int(np.asarray(menu[p.problem_id], bool).sum())
        out[r] = 1 - (1 - q) ** m if m else 0.0
    return out


def _pick(cov_curve: np.ndarray, c: float, tol: float) -> int | None:
    k = int(np.argmin(np.abs(cov_curve - c)))
    return k if abs(cov_curve[k] - c) <= tol else None


def rule_selective(rm: RuleMatrix, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per threshold column: coverage, selective risk and saving among stops."""
    w = np.asarray(w, float)
    f = rm.fired.astype(float)
    fired = w @ f
    unsafe = w @ (f * (~rm.safe))
    sav = w @ rm.saving
    with np.errstate(invalid="ignore", divide="ignore"):
        return fired / w.sum(), np.where(fired > 0, unsafe / fired, np.nan), np.where(
            fired > 0, sav / fired, np.nan)


def no_signal_selective(FP: np.ndarray, ES: np.ndarray, EQ: np.ndarray, w: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(w, float)
    fired = w @ FP
    with np.errstate(invalid="ignore", divide="ignore"):
        return fired / w.sum(), np.where(fired > 0, (w @ (1 - EQ)) / fired, np.nan), np.where(
            fired > 0, (w @ ES) / fired, np.nan)


def at_coverage(curves: tuple[np.ndarray, np.ndarray, np.ndarray], c: float, tol: float
                ) -> dict | None:
    cov, risk, sav = curves
    k = _pick(cov, c, tol)
    if k is None:
        return None
    return {"coverage": float(cov[k]), "selective_risk": float(risk[k]),
            "saving_among_stops": float(sav[k])}


def selective_comparisons(mats: Mapping[str, RuleMatrix], ns: tuple[np.ndarray, np.ndarray, np.ndarray],
                          coverages: Sequence[float], tol: float, reps: int,
                          rng: np.random.Generator) -> dict:
    """Every rule's selective risk at each coverage, and its difference from no
    signal with a paired problem bootstrap (one draw shared by all rules)."""
    FP, ES, EQ = ns
    n = FP.shape[0]
    ones = np.ones(n)

    def all_curves(w):
        c = {r: rule_selective(m, w) for r, m in mats.items()}
        c["no_signal"] = no_signal_selective(FP, ES, EQ, w)
        return c

    point = all_curves(ones)
    W = rng.multinomial(n, np.full(n, 1 / n), size=reps).astype(float)
    boot = {(r, c): np.full(reps, np.nan) for r in mats for c in coverages}
    for i, w in enumerate(W):
        cur = all_curves(w)
        for c in coverages:
            b = at_coverage(cur["no_signal"], c, tol)
            for r in mats:
                a = at_coverage(cur[r], c, tol)
                if a is not None and b is not None:
                    boot[(r, c)][i] = a["selective_risk"] - b["selective_risk"]
    out = {}
    for c in coverages:
        row = {"no_signal": at_coverage(point["no_signal"], c, tol)}
        for r in mats:
            a = at_coverage(point[r], c, tol)
            entry = {"at": a}
            if a is not None and row["no_signal"] is not None:
                rep = boot[(r, c)]
                ok = rep[np.isfinite(rep)]
                entry.update({
                    "risk_minus_no_signal": a["selective_risk"] - row["no_signal"]["selective_risk"],
                    "ci95": ([float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
                             if len(ok) else None),
                    "n_replicates_evaluable": int(len(ok))})
            row[r] = entry
        out[str(c)] = row
    return out
