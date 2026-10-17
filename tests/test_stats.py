"""Statistics.

The properties that matter here are not "does scipy work" but the three
commitments the study makes: resampling at problem level, reporting an effect
size with every test, and correcting the whole family.
"""

import numpy as np
import pytest

from nlaast.analysis.stats import (
    PAIRED_MEAN_DIFFERENCE,
    PAIRED_RATE_DIFFERENCE,
    TestResult,
    benjamini_hochberg,
    bh_at_family_size,
    bootstrap_statistic,
    cliffs_delta,
    cohens_d,
    grouped_bootstrap_difference,
    paired_binary_test,
    paired_difference,
    paired_rate_difference_ci,
    paired_test,
    rate_difference_ci,
    summarise_tests,
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

    def test_unevaluable_tests_stay_in_the_report_but_not_in_m(self):
        """They are returned, flagged as unevaluated, and excluded from ``m``.

        Both halves matter and they used to be conflated: a test that produced
        no p-value stays in the output so that a planned-and-not-made
        comparison is visible, but it has no rejection region and so does not
        enter the denominator. See D37a - which of the two conventions the
        pre-registration intends is open; this pins what the code does.
        """
        tests = self._mk([0.001, 0.002])
        tests.append(TestResult("nan", 0.0, float("nan"), 0.0, "d"))
        out = benjamini_hochberg(tests, 0.05)
        assert len(out) == 3
        assert out[-1].q_value is None and out[-1].significant is None
        # m is 2 (the evaluable ones), so the smallest q is p * 2 / 1.
        assert out[0].q_value == pytest.approx(0.002, abs=1e-9)

    def test_none_p_value_is_treated_as_missing_not_as_a_crash(self):
        """``None`` and ``nan`` both mean "no p-value" and must behave alike.

        A result round-tripped through JSON comes back with ``None``, because
        the writer has no ``nan`` to serialise. ``np.isfinite(None)`` raises,
        so the two spellings used to diverge: one was excluded, the other
        brought the whole correction down.
        """
        tests = self._mk([0.001, 0.002])
        tests.append(TestResult("none", 0.0, None, 0.0, "d"))
        out = benjamini_hochberg(tests, 0.05)
        assert out[-1].q_value is None and out[-1].significant is None
        # Identical to the nan case above: m = 2, not 3.
        assert out[0].q_value == pytest.approx(0.002, abs=1e-9)

    def test_m_counts_evaluable_tests_only(self):
        """``m`` tracks the evaluable count, not the length of the family.

        Adding unevaluable members must not change any q-value. This is the
        behaviour D37a leaves open at the *reporting* level; the test asserts
        what the implementation does so that a change to it is deliberate.
        """
        base = benjamini_hochberg(self._mk([0.01, 0.02, 0.03]), 0.05)
        qs = [t.q_value for t in base]

        padded = self._mk([0.01, 0.02, 0.03])
        padded += [TestResult(f"u{i}", 0.0, float("nan"), 0.0, "d")
                   for i in range(6)]
        out = benjamini_hochberg(padded, 0.05)
        assert [t.q_value for t in out[:3]] == pytest.approx(qs)
        # Had the nine-member family been used, the smallest q would be
        # 0.01 * 9 / 1 = 0.09 rather than 0.01 * 3 / 1 = 0.03.
        assert out[0].q_value == pytest.approx(0.03, abs=1e-12)

    def test_worked_example_q_values_and_rejections(self):
        """A hand-checkable family, including the step-down monotonicity.

        p = 0.001, 0.008, 0.039, 0.041, 0.9 with m = 5 gives raw p*m/i of
        0.005, 0.020, 0.065, 0.05125, 0.9. BH enforces monotonicity from the
        largest down, so the third is pulled down to the fourth's 0.05125 -
        without that step-down it would read 0.065 and the pair would be
        reported inconsistently.
        """
        out = benjamini_hochberg(self._mk([0.001, 0.008, 0.039, 0.041, 0.9]), 0.05)
        qs = [t.q_value for t in out]
        assert qs[0] == pytest.approx(0.005)
        assert qs[1] == pytest.approx(0.020)
        assert qs[2] == pytest.approx(0.05125)
        assert qs[3] == pytest.approx(0.05125)
        assert qs[4] == pytest.approx(0.9)
        assert [t.significant for t in out] == [True, True, False, False, False]

    def test_summarise_reports_the_m_that_was_used(self):
        """The artifact has to be enough to reproduce the q-values (D38)."""
        tests = self._mk([0.001, 0.002])
        tests.append(TestResult("nan", 0.0, float("nan"), 0.0, "d"))
        summary = summarise_tests(benjamini_hochberg(tests, 0.05))
        assert summary["n_tests"] == 3
        assert summary["bh_family_size"] == 2
        assert summary["n_planned_but_unevaluable"] == 1
        assert summary["bh_family_convention"] == "evaluable_tests"
        assert summary["n_significant"] == 2

    def test_the_planned_family_sensitivity_is_always_reported(self):
        """Both conventions exist in the artifact; only the framing is a choice.

        With two evaluable tests of three planned, the primary correction uses
        m = 2 and the sensitivity uses m = 3, so the same p-values give
        q = 0.002 and q = 0.003. Reporting both means the D38 decision cannot
        change any number, only which one is foregrounded.
        """
        tests = self._mk([0.001, 0.002])
        tests.append(TestResult("nan", 0.0, float("nan"), 0.0, "d"))
        summary = summarise_tests(benjamini_hochberg(tests, 0.05), 0.05)
        sens = summary["q_values_at_planned_family_size"]
        assert sens["family_size"] == 3
        assert summary["tests"][0]["q_value"] == pytest.approx(0.002)
        assert sens["q_values"]["t0"]["q_value"] == pytest.approx(0.003)
        assert summary["verdict_is_robust_to_family_convention"] is True

    def test_the_sensitivity_flags_a_verdict_that_is_not_robust(self):
        """The case that actually needs the author's decision.

        p = 0.02 with m = 2 gives q = 0.04 and rejects; with m = 5 it gives
        0.10 and does not. When this flag is False the choice of convention
        decides a finding, which is exactly when it must be stated rather
        than assumed.
        """
        tests = self._mk([0.02, 0.9])
        tests += [TestResult(f"u{i}", 0.0, float("nan"), 0.0, "d")
                  for i in range(3)]
        summary = summarise_tests(benjamini_hochberg(tests, 0.05), 0.05)
        assert summary["n_significant"] == 1
        assert summary["q_values_at_planned_family_size"]["n_significant"] == 0
        assert summary["verdict_is_robust_to_family_convention"] is False

    def test_the_sensitivity_never_uses_a_family_smaller_than_its_tests(self):
        tests = self._mk([0.01, 0.02, 0.03])
        out = bh_at_family_size(tests, 1, 0.05)
        assert out["family_size"] == 3

    def test_the_sensitivity_does_not_mutate_the_results(self):
        tests = benjamini_hochberg(self._mk([0.001, 0.5]), 0.05)
        before = [(t.q_value, t.significant) for t in tests]
        bh_at_family_size(tests, 20, 0.05)
        assert [(t.q_value, t.significant) for t in tests] == before

    def test_empty_family(self):
        assert benjamini_hochberg([], 0.05) == []

    def test_family_with_no_evaluable_member_is_returned_unflagged(self):
        tests = [TestResult("a", 0.0, float("nan"), 0.0, "d"),
                 TestResult("b", 0.0, None, 0.0, "d")]
        out = benjamini_hochberg(tests, 0.05)
        assert len(out) == 2
        assert all(t.q_value is None for t in out)
        assert summarise_tests(out)["bh_family_size"] == 0


class TestPairedBinaryTest:
    """The matched-budget comparison's unit of analysis is the problem.

    Pairing over budget levels instead does not lose power, it invents it: a
    step-function curve turns a span of budgets into a few distinct operating
    points, so asking for more budgets copies one comparison. On the pilot nine
    budgets reached a single pair and the signed-rank test returned p = 2^-8.
    """

    def test_only_discordant_pairs_carry_information(self):
        # Twenty agreements and four disagreements, all one way. The p-value
        # must depend on the four, not on the twenty.
        a = [True] * 20 + [False] * 4
        b = [True] * 20 + [True] * 4
        t = paired_binary_test("t", a, b)
        assert t.notes["n_discordant"] == 4
        assert t.p_value == pytest.approx(0.125)

    def test_padding_with_agreements_does_not_change_the_p_value(self):
        short = paired_binary_test("t", [False] * 4, [True] * 4)
        padded = paired_binary_test("t", [True] * 50 + [False] * 4,
                                    [True] * 50 + [True] * 4)
        assert short.p_value == pytest.approx(padded.p_value)

    def test_repeating_the_same_comparison_does_not_manufacture_significance(self):
        """The defect this test exists to prevent: evidence must come from
        distinct units, not from the same unit counted again."""
        one = paired_binary_test("t", [False] * 4, [True] * 4)
        # The same four problems, each listed three times over.
        thrice = paired_binary_test("t", [False] * 12, [True] * 12)
        assert one.p_value == pytest.approx(0.125)
        assert thrice.p_value < one.p_value   # which is why units must be unique

    def test_no_discordant_pairs_is_p_one_not_a_missing_result(self):
        """Maximal agreement is a tested null, not absent evidence (D38).

        Conditioning on zero discordant pairs leaves a single possible table,
        so the exact two-sided p-value is 1.0. Returning nan instead dropped
        the test from the BH denominator precisely when its null held
        perfectly - the one exclusion that correlated with the effect.
        """
        t = paired_binary_test("t", [True, False, True], [True, False, True])
        assert t.p_value == 1.0
        assert t.notes["n_discordant"] == 0
        assert t.notes["zero_discordant"] is True
        assert "skipped" not in t.notes, "it was conducted, not skipped"
        assert "interpretation" in t.notes
        # Still reported as uninformative: nothing could have rejected.
        assert t.notes["smallest_attainable_p"] == 1.0
        # The rate-difference interval is still the informative part.
        assert t.interval is not None

    def test_a_zero_discordant_test_enters_the_bh_denominator(self):
        """The consequence that motivated the fix."""
        real = paired_binary_test("real", [True, True, False], [False, False, False])
        tied = paired_binary_test("tied", [True, False], [True, False])
        out = benjamini_hochberg([real, tied], 0.05)
        assert summarise_tests(out)["bh_family_size"] == 2
        # m = 2, so the real test's q is its p * 2 / 1 rather than p * 1 / 1.
        assert out[0].q_value == pytest.approx(real.p_value * 2)

    def test_smallest_attainable_p_separates_underpowered_from_null(self):
        """A design diagnostic the p-value alone does not give.

        Four discordant pairs can do no better than 0.125 however they fall,
        so a comparison resting on four of them could not have cleared a
        corrected threshold of 0.01 in either direction. That is a fact about
        the design, computable with no simulation.
        """
        four = paired_binary_test("t", [True] * 2 + [False] * 2 + [True] * 5,
                                  [False] * 2 + [True] * 2 + [True] * 5)
        assert four.notes["n_discordant"] == 4
        assert four.notes["smallest_attainable_p"] == pytest.approx(0.125)
        assert four.notes["smallest_attainable_p"] > 0.01

        eight = paired_binary_test("t", [True] * 8, [False] * 8)
        assert eight.notes["n_discordant"] == 8
        assert eight.notes["smallest_attainable_p"] <= 0.01

    def test_direction_is_recoverable_from_the_counts(self):
        t = paired_binary_test("t", [True, True, False], [False, False, False])
        assert t.notes["a_safe_b_not"] == 2 and t.notes["b_safe_a_not"] == 0
        assert t.notes["safe_rate_a"] > t.notes["safe_rate_b"]

    def test_empty(self):
        t = paired_binary_test("t", [], [])
        assert t.n == 0 and "skipped" in t.notes


class TestPairedRateDifference:
    def test_the_interval_is_centred_on_the_observed_difference(self):
        iv = paired_rate_difference_ci(b_only=0, c_only=4, n=21)
        assert iv.estimate == pytest.approx(-4 / 21)
        assert iv.low <= iv.estimate <= iv.high

    def test_no_discordance_means_no_uncertainty(self):
        iv = paired_rate_difference_ci(0, 0, 21)
        assert (iv.estimate, iv.low, iv.high) == (0.0, 0.0, 0.0)

    def test_it_is_tighter_than_treating_the_arms_as_independent(self):
        """Ignoring the pairing throws away the fact that both rules were
        scored on the same problems."""
        paired = paired_rate_difference_ci(0, 4, 21)
        indep = rate_difference_ci(2, 21, 6, 21)
        assert (paired.high - paired.low) < (indep.high - indep.low)

    def test_bounds_stay_inside_the_unit_interval(self):
        for b, c in ((0, 10), (10, 0), (5, 5), (1, 9)):
            iv = paired_rate_difference_ci(b, c, 10)
            assert -1.0 <= iv.low <= iv.high <= 1.0


class TestAnIntervalNamesWhatItBounds:
    """`effect` and `interval` are not on one scale for two test families, and
    the hypothesis-test table printed them side by side: an odds ratio of 0.833
    beside a CI of [-0.061, 0.049], Cohen's d of 0.712 beside [0.031, 0.033].
    The numbers were right; the table read as if the CI bounded the effect."""

    def test_a_paired_test_bounds_the_raw_mean_difference_not_d(self):
        t = paired_test("x", [3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 1.5, 2.0, 2.5, 3.0])
        assert t.effect_name == "cohens_d_paired"
        assert t.interval.estimand == PAIRED_MEAN_DIFFERENCE
        assert t.interval.estimate == pytest.approx(3.0)
        assert t.effect != pytest.approx(t.interval.estimate)

    def test_a_paired_binary_test_bounds_the_rate_difference_not_the_odds(self):
        a = [True] * 10 + [False] * 6 + [True] * 5
        b = [True] * 10 + [True] * 6 + [False] * 5
        t = paired_binary_test("x", a, b)
        assert t.effect_name == "odds_ratio_discordant"
        assert t.interval.estimand == PAIRED_RATE_DIFFERENCE
        assert t.interval.estimate == pytest.approx((5 - 6) / 21)

    def test_the_table_prints_the_estimand_and_its_estimate(self):
        from nlaast.analysis.tables import tests_table
        t = paired_test("x", [3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 1.5, 2.0, 2.5, 3.0])
        row = tests_table([t]).iloc[0]
        assert row["ci_estimand"] == PAIRED_MEAN_DIFFERENCE
        assert row["ci_estimate"] == pytest.approx(3.0)
        assert row["ci_low"] <= row["ci_estimate"] <= row["ci_high"]

    def test_an_artifact_written_before_the_label_is_still_labelled(self):
        """results.json files already on disk carry no estimand."""
        from nlaast.analysis.tables import tests_table
        old = {"name": "O3", "n": 120, "effect": 0.833,
               "effect_name": "odds_ratio_discordant",
               "interval": {"estimate": -0.008, "low": -0.061, "high": 0.049},
               "p_value": 1.0, "q_value": 1.0, "significant": False}
        row = tests_table([old]).iloc[0]
        assert row["ci_estimand"] == PAIRED_RATE_DIFFERENCE

    def test_no_interval_means_no_estimand(self):
        from nlaast.analysis.tables import tests_table
        row = tests_table([{"name": "F10", "effect": -0.2,
                            "effect_name": "cliffs_delta"}]).iloc[0]
        assert row["ci_estimand"] == ""
