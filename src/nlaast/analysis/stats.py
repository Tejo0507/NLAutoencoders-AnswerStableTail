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


def benjamini_hochberg(tests: Sequence[TestResult], q: float = 0.05) -> list[TestResult]:
    """Annotate a family of tests with BH q-values, in place of raw p-values.

    Tests without a usable p-value are carried through unflagged rather than
    dropped: silently shrinking the family would make the correction look
    kinder than it is.
    """
    usable = [t for t in tests if np.isfinite(t.p_value)]
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
        if not np.isfinite(t.p_value):
            t.q_value = None
            t.significant = None
    return list(tests)


def summarise_tests(tests: Sequence[TestResult]) -> dict[str, Any]:
    done = [t for t in tests if t.significant is not None]
    return {
        "n_tests": len(tests),
        "n_evaluated": len(done),
        "n_significant": sum(1 for t in done if t.significant),
        "tests": [t.to_dict() for t in tests],
    }
