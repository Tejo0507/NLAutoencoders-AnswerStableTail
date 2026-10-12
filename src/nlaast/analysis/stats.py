"""Statistics for the study.

Three commitments, all made in advance:

* **Problem-level pairing.** Objective O3 asks for paired bootstrap confidence
  intervals computed at problem level. Boundaries within one trace are strongly
  dependent, so resampling boundaries would understate the interval badly. Every
  bootstrap here resamples *problems* and recomputes the statistic from the
  resampled problems' rows.
* **Effect sizes with every test.** A p-value at this sample size says little;
  the effect size and its interval say what was actually observed.
* **Multiple testing.** The pre-registered family is corrected with
  Benjamini-Hochberg at q = 0.05 (falsification test F9).

A wide interval is a result. Nothing here is permitted to report a point
estimate without its uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    confidence: float
    n: int
    method: str = "percentile_bootstrap"

    @property
    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.low) and np.isfinite(self.high)
                    and (self.low > 0 or self.high < 0))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["excludes_zero"] = self.excludes_zero
        return d


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    effect: float
    effect_name: str
    interval: Interval | None = None
    n: int = 0
    #: Filled in by ``benjamini_hochberg``.
    q_value: float | None = None
    significant: bool | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interval"] = self.interval.to_dict() if self.interval else None
        return d


# --- Bootstrap ---


def bootstrap_statistic(
    groups: Sequence[Any],
    statistic: Callable[[Sequence[int]], float],
    n_iterations: int,
    confidence: float,
    seed: int,
) -> Interval:
    """Resample *groups* (problems) with replacement and recompute.

    ``statistic`` receives the indices of the selected groups, so the caller
    decides how a group maps to rows. This is the whole reason the function
    takes indices rather than values: for paired comparisons the statistic has
    to see both arms of the same problem together.
    """
    groups = list(groups)
    n = len(groups)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, 0)

    point = float(statistic(list(range(n))))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_iterations, dtype=np.float64)
    for b in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        try:
            samples[b] = statistic(idx.tolist())
        except Exception:
            samples[b] = np.nan

    valid = samples[np.isfinite(samples)]
    if valid.size < max(50, n_iterations // 20):
        return Interval(point, float("nan"), float("nan"), confidence, n,
                        method="percentile_bootstrap (insufficient valid resamples)")
    alpha = (1 - confidence) / 2
    return Interval(
        estimate=point,
        low=float(np.percentile(valid, 100 * alpha)),
        high=float(np.percentile(valid, 100 * (1 - alpha))),
        confidence=confidence,
        n=n,
    )


def paired_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for ``mean(a - b)`` over paired observations."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, 0)
    return bootstrap_statistic(
        list(range(d.size)),
        lambda idx: float(np.mean(d[np.asarray(idx, dtype=int)])),
        n_iterations, confidence, seed,
    )


def grouped_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    group_key: str,
    value_a: str,
    value_b: str,
    n_iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired difference with the resampling unit set to ``group_key``.

    Rows are typically per-boundary; groups are problems. This is the function
    O3 calls, and the grouping is why its intervals are honest.
    """
    by_group: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(r[group_key], []).append(r)
    keys = sorted(by_group)

    def stat(idx: Sequence[int]) -> float:
        diffs = []
        for i in idx:
            for r in by_group[keys[i]]:
                a, b = r.get(value_a), r.get(value_b)
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    diffs.append(float(a) - float(b))
        return float(np.mean(diffs)) if diffs else float("nan")

    return bootstrap_statistic(keys, stat, n_iterations, confidence, seed)


# --- Effect sizes ---


def cohens_d(a: Sequence[float], b: Sequence[float], paired: bool = False) -> float:
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if paired:
        if a.size != b.size:
            return float("nan")
        d = a - b
        sd = d.std(ddof=1)
        return float(d.mean() / sd) if sd > 0 else float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1].

    Reported alongside Cohen's d because several of the study's outcomes
    (tokens saved, marker counts) are bounded or skewed, and d assumes neither.
    """
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    return float((gt - lt) / (a.size * b.size))


def rate_difference_ci(
    k_a: int, n_a: int, k_b: int, n_b: int, confidence: float = 0.95
) -> Interval:
    """Newcombe interval for a difference of two independent proportions.

    Preferred over the Wald interval because several arms here have rates near
    0 or 1 at small n, where Wald intervals leave the unit interval.
    """
    from scipy import stats

    if n_a == 0 or n_b == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, 0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    def wilson(k: int, n: int) -> tuple[float, float]:
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return centre - half, centre + half

    l1, u1 = wilson(k_a, n_a)
    l2, u2 = wilson(k_b, n_b)
    p1, p2 = k_a / n_a, k_b / n_b
    return Interval(
        estimate=p1 - p2,
        low=float((p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
        high=float((p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
        confidence=confidence, n=n_a + n_b, method="newcombe",
    )


def paired_rate_difference_ci(
    b_only: int, c_only: int, n: int, confidence: float = 0.95
) -> Interval:
    """Interval for a difference of two *paired* proportions.

    ``rate_difference_ci`` assumes the two proportions are independent, which
    is exactly what a matched comparison is not: both rules are scored on the
    same problems, and ignoring that pairing misstates the uncertainty.

    Only the discordant problems carry information. Writing ``d`` for their
    count, the difference in rates is ``(b - c) / n = (d / n) * (2 * phi - 1)``
    with ``phi = b / d``, which is monotone in ``phi``. A Clopper-Pearson
    interval for ``phi`` therefore maps straight onto one for the difference,
    and conditions on ``d`` just as McNemar's exact test does.
    """
    from scipy import stats

    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, 0)
    d = b_only + c_only
    est = (b_only - c_only) / n
    if d == 0:
        # No problem was handled differently, so the difference is exactly 0.
        return Interval(0.0, 0.0, 0.0, confidence, n, method="conditional_exact")
    ci = stats.binomtest(b_only, d, 0.5).proportion_ci(
        confidence_level=confidence, method="exact")
    scale = d / n
    return Interval(
        estimate=est,
        low=float(scale * (2 * ci.low - 1)),
        high=float(scale * (2 * ci.high - 1)),
        confidence=confidence, n=n, method="conditional_exact",
    )


# --- Tests ---


def paired_test(
    name: str,
    a: Sequence[float],
    b: Sequence[float],
    n_iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> TestResult:
    """Wilcoxon signed-rank plus a paired bootstrap interval and Cohen's d.

    Wilcoxon rather than a paired t-test because none of this study's paired
    outcomes are plausibly normal at n in the low hundreds.
    """
    from scipy import stats

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]

    if arr_a.size < 3 or np.allclose(arr_a, arr_b):
        return TestResult(
            name=name, statistic=float("nan"), p_value=float("nan"),
            effect=float("nan"), effect_name="cohens_d_paired", n=int(arr_a.size),
            notes={"skipped": "fewer than 3 usable pairs, or arms identical"},
        )
    try:
        stat, p = stats.wilcoxon(arr_a, arr_b)
    except ValueError as exc:
        return TestResult(name, float("nan"), float("nan"), float("nan"),
                          "cohens_d_paired", n=int(arr_a.size),
                          notes={"error": repr(exc)})
    return TestResult(
        name=name,
        statistic=float(stat),
        p_value=float(p),
        effect=cohens_d(arr_a, arr_b, paired=True),
        effect_name="cohens_d_paired",
        interval=paired_difference(arr_a, arr_b, n_iterations, confidence, seed),
        n=int(arr_a.size),
        notes={"test": "wilcoxon_signed_rank", "cliffs_delta": cliffs_delta(arr_a, arr_b)},
    )


def paired_binary_test(
    name: str,
    a: Sequence[bool],
    b: Sequence[bool],
    confidence: float = 0.95,
) -> TestResult:
    """McNemar's exact test on two binary outcomes measured on the same units.

    The matched-budget comparison needs this because its outcome is per
    problem and binary - stopping there either preserved the verified outcome
    or it did not - and the unit of analysis has to be the problem.

    Pairing over *budget levels* instead is what this replaces, and it does
    not merely lose power, it manufactures it. Each rule's curve is a step
    function, so a span of requested budgets resolves to a handful of distinct
    operating points, and asking for more budgets copies the same comparison
    rather than adding evidence. On the pilot all nine budgets landed on one
    operating-point pair, safe rates 0.095 against 0.286, and the signed-rank
    test over nine identical differences returned p = 2^-8 = 0.0039. The same
    data with five budgets gives 0.0625 and with twenty-five gives 1e-6: the
    p-value was a function of the grid size.

    Only the discordant problems carry information, so the exact binomial form
    is used rather than the chi-square approximation, which is unreliable when
    that count is small - and on 21 problems it is.

    **Two things this reports that the p-value alone does not.**

    ``notes['smallest_attainable_p']`` is the p-value this many discordant
    pairs would give if every one of them pointed the same way - the best the
    comparison could possibly have done. It separates "no difference was
    found" from "no difference could have been found": on the pilot's four
    discordant pairs it is 0.125, so that comparison could not have cleared a
    corrected threshold of 0.01 whichever way the pairs fell. That is a
    statement about the design, available without simulating anything.

    ``n_discordant == 0`` returns **p = 1.0**, not ``nan``. See the comment at
    that branch: conditioning on zero discordant pairs leaves one possible
    table, so the exact two-sided p-value is 1 and the hypothesis was tested
    and not rejected. ``notes['zero_discordant']`` marks it so a reader is not
    left to infer from a bare p = 1.0 that the comparison was informative.
    """
    from scipy import stats

    pairs = [(bool(x), bool(y)) for x, y in zip(a, b)]
    if not pairs:
        return TestResult(name, float("nan"), float("nan"), float("nan"),
                          "odds_ratio_discordant", n=0,
                          notes={"skipped": "no paired observations"})
    b_only = sum(1 for x, y in pairs if x and not y)   # a safe, b not
    c_only = sum(1 for x, y in pairs if y and not x)   # b safe, a not
    n_disc = b_only + c_only
    rate_a = sum(1 for x, _ in pairs if x) / len(pairs)
    rate_b = sum(1 for _, y in pairs if y) / len(pairs)
    notes = {
        "test": "mcnemar_exact",
        "n_problems": len(pairs),
        "n_discordant": n_disc,
        "a_safe_b_not": b_only,
        "b_safe_a_not": c_only,
        "safe_rate_a": rate_a,
        "safe_rate_b": rate_b,
        "unit_of_analysis": "problem",
    }
    if n_disc == 0:
        # p = 1.0, not "no result". The exact conditional test conditions on
        # the number of discordant pairs; given d = 0 the conditional sample
        # space is the single point b = 0, so P(as extreme or more) = 1 and the
        # two-sided exact p-value *is* 1.0. This is arithmetic, not a
        # convention.
        #
        # It used to return nan and mark itself skipped, which misclassified
        # the one case where the null holds exactly. Zero discordant pairs is
        # the strongest agreement the design can observe - both rules
        # preserved the verified outcome on exactly the same problems - so
        # treating it as missing evidence dropped a test from the
        # Benjamini-Hochberg denominator *because* its null was perfectly
        # satisfied. That is the one exclusion that correlates with the effect
        # being tested, and it biases every other q-value downwards (D38).
        #
        # The test is reported as conducted and not rejected. That it has no
        # power is a separate and separately reported fact: `power` below is
        # the smallest two-sided p this design could have produced, which at
        # d = 0 is 1.0 - there was no evidence available either way.
        notes["interpretation"] = (
            "no discordant problems: both rules preserved the verified "
            "outcome on exactly the same problems. This is maximal agreement, "
            "not missing data, and the exact conditional p-value is 1.0. The "
            "comparison is uninformative rather than unevaluable: no "
            "assignment of the discordant pairs could have rejected, because "
            "there are none")
        notes["zero_discordant"] = True
        notes["smallest_attainable_p"] = 1.0
        return TestResult(name, 0.0, 1.0, float("nan"),
                          "odds_ratio_discordant", n=len(pairs), notes=notes,
                          interval=paired_rate_difference_ci(
                              b_only, c_only, len(pairs), confidence))
    # The best case this many discordant pairs could ever do: all one way.
    notes["smallest_attainable_p"] = float(
        stats.binomtest(0, n_disc, 0.5).pvalue)
    p = float(stats.binomtest(b_only, n_disc, 0.5).pvalue)
    # Reported on the discordant pairs, which are the only ones that differ.
    effect = float("inf") if c_only == 0 else b_only / c_only
    return TestResult(
        name=name, statistic=float(b_only), p_value=p,
        effect=effect, effect_name="odds_ratio_discordant",
        interval=paired_rate_difference_ci(b_only, c_only, len(pairs), confidence),
        n=len(pairs), notes=notes,
    )


def unpaired_test(
    name: str,
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> TestResult:
    from scipy import stats

    arr_a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    arr_b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if arr_a.size < 3 or arr_b.size < 3:
        return TestResult(name, float("nan"), float("nan"), float("nan"),
                          "cliffs_delta", n=int(arr_a.size + arr_b.size),
                          notes={"skipped": "fewer than 3 observations in an arm"})
    stat, p = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    return TestResult(
        name=name, statistic=float(stat), p_value=float(p),
        effect=cliffs_delta(arr_a, arr_b), effect_name="cliffs_delta",
        n=int(arr_a.size + arr_b.size),
        notes={"test": "mann_whitney_u", "cohens_d": cohens_d(arr_a, arr_b),
               "mean_a": float(arr_a.mean()), "mean_b": float(arr_b.mean())},
    )


def _has_p(t: TestResult) -> bool:
    """Did this test actually produce a p-value?

    ``None`` and ``nan`` both mean "no p-value" and reach here from different
    places: a test that declined to run returns ``float('nan')``, while a
    result round-tripped through JSON comes back with ``None``, because the
    writer has no ``nan`` to serialise. Treating only one of them as missing
    made this function raise on the other.
    """
    return t.p_value is not None and bool(np.isfinite(t.p_value))


def benjamini_hochberg(tests: Sequence[TestResult], q: float = 0.05) -> list[TestResult]:
    """Annotate a family of tests with BH q-values, in place of raw p-values.

    **The correction is taken over the tests that produced a p-value**, so
    ``m`` is the number of *evaluable* tests rather than the number
    pre-registered. A test that could not be evaluated - no matched budget, no
    discordant pairs, a missing dependent variable - is carried through to the
    report with ``q_value`` and ``significant`` left ``None``, so it stays
    visible as planned-and-not-made. It does not enter ``m``.

    **Why that is defensible rather than merely convenient.** BH controls the
    false discovery rate, ``E[V/R]``, over the hypotheses actually tested. A
    hypothesis that produced no p-value cannot be rejected, so it can
    contribute neither to ``V`` nor to ``R``; including it in ``m`` buys no
    error control and only makes every other test conservative. What F9's
    "pre-registered family" guards against is choosing the denominator *after
    seeing which tests came out well*, and that guard is satisfied here
    provided every exclusion is for a reason independent of the direction and
    size of the effect under test. On this analysis plan they are:

    - **no matched budget exists** - decided by the rules' token-saving
      curves, not by their safe rates (O3 vs convergence);
    - **fewer than three observations in an arm** - decided by how many
      problems the model got right, not by the tail fractions compared (RQ1);
    - **the dependent variable has no variance** - decided by the
      instrument's floor, not by the arms' means (RQ3's marker count, which
      is identically zero on this model).

    **One exclusion was not independent of the effect, and has been removed.**
    A McNemar comparison with zero discordant pairs used to return ``nan`` and
    drop out of ``m`` - but zero discordance is exactly the null holding
    perfectly, so the family shrank *because* a null was true, which biases
    every other q-value downwards. ``paired_binary_test`` now returns the
    arithmetically correct ``p = 1.0`` there, so that test is counted. See
    D38.

    With that fixed, ``m = len(usable)`` is the convention this code
    implements and the one to report as primary. ``summarise_tests`` also
    reports ``q_values_at_planned_family_size`` so the stricter reading -
    ``m`` = every pre-registered test, evaluable or not - is visible as a
    sensitivity without re-running anything. D38 records the reasoning and the
    one judgement left to the author: which of the two to foreground.
    """
    usable = [t for t in tests if _has_p(t)]
    m = len(usable)
    if m == 0:
        return list(tests)
    order = sorted(usable, key=lambda t: t.p_value)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        t = order[i]
        adj = min(prev, t.p_value * m / (i + 1))
        t.q_value = float(min(1.0, adj))
        t.significant = bool(t.q_value <= q)
        prev = adj
    for t in tests:
        if not _has_p(t):
            t.q_value = None
            t.significant = None
    return list(tests)


def bh_at_family_size(
    tests: Sequence[TestResult], m: int, q: float = 0.05
) -> dict[str, Any]:
    """BH q-values for the same p-values under a *stated* family size.

    The sensitivity half of D38. ``benjamini_hochberg`` corrects over the
    tests that produced a p-value; this recomputes the same step-up with ``m``
    forced, so the stricter reading of F9 - every pre-registered test counts,
    evaluable or not - can be reported beside the primary numbers without
    re-running an analysis or mutating the results.

    Pure function: it does not touch the ``TestResult`` objects.
    """
    usable = sorted((t for t in tests if _has_p(t)), key=lambda t: t.p_value)
    k = len(usable)
    m = max(int(m), k)  # a family cannot be smaller than the tests in it
    out: dict[str, Any] = {}
    prev = 1.0
    for i in range(k - 1, -1, -1):
        t = usable[i]
        adj = min(prev, t.p_value * m / (i + 1))
        out[t.name] = {"q_value": float(min(1.0, adj)),
                       "significant": bool(min(1.0, adj) <= q)}
        prev = adj
    return {"family_size": m, "q": q,
            "n_significant": sum(1 for v in out.values() if v["significant"]),
            "q_values": out}


def summarise_tests(tests: Sequence[TestResult], q: float = 0.05) -> dict[str, Any]:
    """Family-level counts, the ``m`` used, and the alternative convention.

    ``bh_family_size`` is the one number a reader needs to reproduce the
    q-values, so it is stated rather than left to be inferred.
    ``n_planned_but_unevaluable`` is its complement: tests that were
    pre-registered, ran, and produced no p-value.

    ``q_values_at_planned_family_size`` is the D38 sensitivity - the same
    p-values corrected over every pre-registered test instead of only the
    evaluable ones. It is reported always, so the choice of convention changes
    which number is foregrounded rather than which numbers exist.
    """
    done = [t for t in tests if t.significant is not None]
    out = {
        "n_tests": len(tests),
        "n_evaluated": len(done),
        "n_significant": sum(1 for t in done if t.significant),
        #: The ``m`` used by ``benjamini_hochberg``: evaluable tests only.
        "bh_family_size": len(done),
        "n_planned_but_unevaluable": len(tests) - len(done),
        "bh_family_convention": "evaluable_tests",
        "tests": [t.to_dict() for t in tests],
    }
    if tests:
        out["q_values_at_planned_family_size"] = bh_at_family_size(
            tests, len(tests), q)
        out["verdict_is_robust_to_family_convention"] = (
            out["n_significant"]
            == out["q_values_at_planned_family_size"]["n_significant"])
    return out
