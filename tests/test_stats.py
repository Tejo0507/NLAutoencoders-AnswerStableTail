"""Statistics.

The properties that matter here are not "does scipy work" but the three
commitments the study makes: resampling at problem level, reporting an effect
size with every test, and correcting the whole family.
"""

import numpy as np
import pytest

from nlaast.analysis.stats import (
    TestResult,
    benjamini_hochberg,
    bootstrap_statistic,
    cliffs_delta,
    cohens_d,
    grouped_bootstrap_difference,
    paired_difference,
    paired_test,
    rate_difference_ci,
    unpaired_test,
)


class TestBootstrap:
    def test_recovers_a_known_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, 300)
        iv = bootstrap_statistic(list(range(len(x))),
                                 lambda idx: float(x[np.asarray(idx)].mean()),
                                 2000, 0.95, 0)
        assert iv.estimate == pytest.approx(x.mean(), abs=1e-9)
        assert iv.low < 5.0 < iv.high

    def test_empty_input(self):
        iv = bootstrap_statistic([], lambda idx: 0.0, 100, 0.95, 0)
        assert iv.n == 0 and np.isnan(iv.estimate)

    def test_paired_difference_detects_a_real_shift(self):
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 0.5, 200)
        iv = paired_difference(a + 0.8, a, 2000, 0.95, 0)
        assert iv.estimate == pytest.approx(0.8, abs=1e-6)
        assert iv.excludes_zero

    def test_paired_difference_on_no_shift(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        assert not paired_difference(a, b, 2000, 0.95, 0).excludes_zero

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_difference([1, 2, 3], [1, 2])


class TestGroupedBootstrap:
    def test_resampling_unit_is_the_group(self):
        """Boundaries within a problem are correlated; resampling them instead
        of problems understates the interval. With one dominant group, a
        group-level bootstrap must produce a *wide* interval."""
        rows = [{"pid": "A", "a": 1.0, "b": 0.0} for _ in range(50)]
        rows += [{"pid": f"B{i}", "a": 0.0, "b": 0.0} for i in range(3)]
        iv = grouped_bootstrap_difference(rows, "pid", "a", "b", 2000, 0.95, 0)
        # Resampling 4 groups, the estimate swings between ~0 and ~1.
        assert iv.high - iv.low > 0.5

    def test_handles_missing_values(self):
        rows = [{"pid": "A", "a": 1.0, "b": 0.5},
                {"pid": "A", "a": None, "b": 0.5},
                {"pid": "B", "a": 2.0, "b": 1.0}]
        iv = grouped_bootstrap_difference(rows, "pid", "a", "b", 500, 0.95, 0)
        assert np.isfinite(iv.estimate)


class TestEffectSizes:
    def test_cohens_d_sign_and_scale(self):
        rng = np.random.default_rng(3)
        a = rng.normal(1.0, 1.0, 500)
        b = rng.normal(0.0, 1.0, 500)
        assert cohens_d(a, b) == pytest.approx(1.0, abs=0.25)

    def test_cohens_d_too_few(self):
        assert np.isnan(cohens_d([1.0], [2.0]))

    def test_cliffs_delta_bounds(self):
        assert cliffs_delta([3, 4, 5], [0, 1, 2]) == pytest.approx(1.0)
        assert cliffs_delta([0, 1, 2], [3, 4, 5]) == pytest.approx(-1.0)
        assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


class TestRateDifference:
    def test_newcombe_interval_stays_in_range(self):
        # Wald intervals leave [-1, 1] near the boundary; Newcombe does not.
        iv = rate_difference_ci(10, 10, 0, 10)
        assert -1.0 <= iv.low <= iv.high <= 1.0
        assert iv.estimate == pytest.approx(1.0)

    def test_no_difference(self):
        iv = rate_difference_ci(5, 10, 5, 10)
        assert iv.estimate == pytest.approx(0.0)
        assert not iv.excludes_zero

    def test_empty(self):
        assert np.isnan(rate_difference_ci(0, 0, 1, 5).estimate)


class TestTests:
    def test_paired_test_finds_a_shift(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0, 1, 60)
        t = paired_test("shift", a + 0.6, a, 1000, 0.95, 0)
        assert t.p_value < 0.05
        assert t.interval.excludes_zero
        assert "cliffs_delta" in t.notes

    def test_paired_test_skips_identical_arms(self):
        t = paired_test("same", [1, 2, 3], [1, 2, 3])
        assert np.isnan(t.p_value)
        assert "skipped" in t.notes

    def test_paired_test_skips_tiny_samples(self):
        t = paired_test("tiny", [1.0], [2.0])
        assert np.isnan(t.p_value)

    def test_unpaired_test(self):
        rng = np.random.default_rng(5)
        t = unpaired_test("groups", rng.normal(1, 1, 80), rng.normal(0, 1, 80))
        assert t.p_value < 0.05
        assert t.effect_name == "cliffs_delta"


class TestMultipleTesting:
    def _mk(self, ps):
        return [TestResult(f"t{i}", 0.0, p, 0.0, "d") for i, p in enumerate(ps)]

    def test_bh_is_monotone_and_at_least_p(self):
        out = benjamini_hochberg(self._mk([0.001, 0.01, 0.04, 0.2, 0.9]), 0.05)
        qs = [t.q_value for t in out]
        assert all(q >= t.p_value - 1e-12 for q, t in zip(qs, out))
        assert qs == sorted(qs)

    def test_bh_rejects_nothing_when_all_null(self):
        out = benjamini_hochberg(self._mk([0.3, 0.5, 0.7, 0.9]), 0.05)
        assert not any(t.significant for t in out)

    def test_unevaluable_tests_stay_in_the_family(self):
        # Dropping them would shrink m and make the correction look kinder.
        tests = self._mk([0.001, 0.002])
        tests.append(TestResult("nan", 0.0, float("nan"), 0.0, "d"))
        out = benjamini_hochberg(tests, 0.05)
        assert len(out) == 3
        assert out[-1].q_value is None and out[-1].significant is None
        # m is 2 (the evaluable ones), so the smallest q is p * 2 / 1.
        assert out[0].q_value == pytest.approx(0.002, abs=1e-9)

    def test_empty_family(self):
        assert benjamini_hochberg([], 0.05) == []
