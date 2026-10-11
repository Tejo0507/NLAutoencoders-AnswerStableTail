"""Stopping rules and the matched-budget comparison.

The safety definition is the subtle part and is tested directly: stopping is
safe when the *verified outcome* is unchanged, not when the answer string
matches.
"""

import numpy as np
import pytest

from nlaast.baselines.semantic_entropy import (
    confidence_score,
    normalised_entropy,
    semantic_entropy,
)
from nlaast.baselines.stopping import (
    OperatingPoint,
    at_matched_budget,
    budget_grid,
    complete_threshold_grid,
    evaluate_stop,
    granularity_bound,
    sweep_threshold,
)


class TestSafety:
    def test_safe_when_outcome_preserved(self):
        d = evaluate_stop("p", "r", 3, "42", "42", "42", 50, 100)
        assert d.safe and d.correct_at_stop and d.final_correct
        assert d.tokens_saved_fraction == pytest.approx(0.5)

    def test_unsafe_when_stopping_loses_a_correct_answer(self):
        d = evaluate_stop("p", "r", 2, "41", "42", "42", 40, 100)
        assert not d.safe
        assert not d.correct_at_stop and d.final_correct

    def test_safe_when_both_wrong(self):
        # Outcome is unchanged: the trace was going to be wrong anyway, so
        # stopping early cost nothing.
        d = evaluate_stop("p", "r", 2, "41", "40", "42", 40, 100)
        assert d.safe
        assert not d.correct_at_stop and not d.final_correct

    def test_stopping_early_on_an_answer_the_trace_would_have_ruined(self):
        # Correct at stop, wrong at the end: a *gain*, counted as unsafe only
        # if safety were defined as string identity. It is not.
        d = evaluate_stop("p", "r", 2, "42", "41", "42", 40, 100)
        assert not d.safe        # outcome did change
        assert d.correct_at_stop and not d.final_correct

    def test_never_firing_saves_nothing_and_is_safe(self):
        d = evaluate_stop("p", "r", None, None, "42", "42", 100, 100)
        assert d.safe and d.tokens_saved_fraction == 0.0
        assert d.boundary is None

    def test_equivalent_not_identical_answers(self):
        d = evaluate_stop("p", "r", 1, "0.5", r"\frac{1}{2}", "0.5", 20, 100)
        assert d.safe and d.correct_at_stop


def _problem(pid, scores, answers, final="42", gold="42"):
    n = len(scores)
    return {
        "problem_id": pid,
        "scores": scores,
        "answers": answers,
        "boundary_tokens": [(i + 1) * 10 for i in range(n)],
        "final_answer": final,
        "gold": gold,
        "total_tokens": n * 10,
    }


class TestSweep:
    def test_higher_threshold_fires_later_and_saves_less(self):
        rows = [_problem("p1", [0.1, 0.5, 0.9], ["1", "42", "42"]),
                _problem("p2", [0.2, 0.6, 0.95], ["2", "42", "42"])]
        curve = sweep_threshold(rows, [0.05, 0.55, 0.99])
        savings = [p.mean_tokens_saved for p in curve]
        assert savings == sorted(savings, reverse=True)
        assert curve[-1].fire_rate == 0.0

    def test_unparseable_scores_are_skipped(self):
        rows = [_problem("p1", [None, float("nan"), 0.9], ["1", "2", "42"])]
        curve = sweep_threshold(rows, [0.5])
        assert curve[0].fire_rate == 1.0
        assert curve[0].n == 1

    def test_empty_input(self):
        assert sweep_threshold([], [0.5]) == []


class TestMatchedBudget:
    def test_picks_the_nearest_operating_point(self):
        rows = [_problem("p1", [0.1, 0.5, 0.9], ["1", "42", "42"])]
        curve = sweep_threshold(rows, list(np.linspace(0, 1, 11)))
        pt = at_matched_budget(curve, 0.33)
        assert pt is not None
        assert abs(pt.mean_tokens_saved - 0.33) <= 0.4

    def test_budget_grid_is_the_intersection(self):
        """A budget one rule cannot reach must not be offered, or the
        comparison becomes an artefact of the threshold grid."""
        a = sweep_threshold([_problem("p", [0.9, 0.9, 0.9], ["42"] * 3)],
                            list(np.linspace(0, 1, 11)))
        b = sweep_threshold([_problem("p", [0.1, 0.1, 0.1], ["42"] * 3)],
                            list(np.linspace(0, 1, 11)))
        grid = budget_grid({"a": a, "b": b})
        for g in grid:
            assert min(p.mean_tokens_saved for p in a) - 1e-9 <= g
            assert g <= max(p.mean_tokens_saved for p in b) + 1e-9

    def test_empty_curves(self):
        assert budget_grid({}) == []
        assert at_matched_budget([], 0.5) is None

    def test_a_budget_of_no_saving_is_not_offered(self):
        """Every complete curve contains a never-fires point. Matching two
        rules there compares nothing: neither stops, both are trivially safe,
        and the difference is identically zero."""
        rows = [_problem("p1", [0.1, 0.5, 0.9], ["1", "2", "42"]),
                _problem("p2", [0.2, 0.6, 0.8], ["1", "2", "42"])]
        curve = sweep_threshold(rows, complete_threshold_grid(
            [0.1, 0.5, 0.9, 0.2, 0.6, 0.8]))
        assert any(p.mean_tokens_saved == 0.0 for p in curve)
        assert all(g > 0 for g in budget_grid({"a": curve, "b": curve}))

    def test_a_rule_that_only_ever_saves_nothing_drops_out(self):
        never = [OperatingPoint(threshold=0.0, n=5, safe_rate=1.0, accuracy=0.5,
                                mean_tokens_saved=0.0, fire_rate=0.0)]
        real = sweep_threshold(
            [_problem("p", [0.1, 0.5, 0.9], ["1", "2", "42"])],
            complete_threshold_grid([0.1, 0.5, 0.9]))
        assert all(g > 0 for g in budget_grid({"never": never, "real": real}))


class TestMatchedMeansClose:
    """"Closest" is not "close", and the difference decides whether a
    matched-budget table is matched.

    A rule's curve is a step function - it fires at a boundary or it does
    not - and on a small corpus the steps are wide. Asking all four pilot
    rules for a budget of 0.632 returned convergence at 0.407 tokens saved
    and the probe at 0.855: a 45-point spread, compared as though the budgets
    were equal.
    """

    @staticmethod
    def _curve(savings):
        return [OperatingPoint(threshold=float(i), n=10, safe_rate=0.5,
                               accuracy=0.5, mean_tokens_saved=float(s),
                               fire_rate=0.5)
                for i, s in enumerate(savings)]

    def test_a_point_within_tolerance_is_returned(self):
        pt = at_matched_budget(self._curve([0.40, 0.62]), 0.63)
        assert pt is not None
        assert pt.mean_tokens_saved == pytest.approx(0.62)

    def test_a_far_nearest_point_is_refused(self):
        """The convergence case: nothing near the target, so no comparison."""
        assert at_matched_budget(self._curve([0.40, 0.89]), 0.63) is None

    def test_the_tolerance_can_be_waived_for_plotting_a_whole_curve(self):
        pt = at_matched_budget(self._curve([0.40, 0.89]), 0.63, tolerance=None)
        assert pt is not None
        assert pt.mean_tokens_saved in (0.40, 0.89)

    def test_an_exact_match_is_always_returned(self):
        pt = at_matched_budget(self._curve([0.10, 0.63, 0.90]), 0.63)
        assert pt.mean_tokens_saved == pytest.approx(0.63)

    def test_the_default_tolerance_is_tight_enough_to_matter(self):
        from nlaast.baselines.stopping import MATCH_TOLERANCE

        assert 0 < MATCH_TOLERANCE <= 0.1


class TestCompleteThresholdGrid:
    """The sweep must enumerate the curve, not sample it.

    A fixed linear grid depends on how the scores happen to be distributed.
    On the pilot that cost a primary comparison: the probe's logistic scores
    saturate near 0 and 1, so 25 evenly spaced thresholds found 5 operating
    points and none within tolerance of a shared budget, and the head-to-head
    was reported unevaluable. Swept at the observed values the same problems
    give 86 points and match at every budget.
    """

    def test_every_observed_score_yields_an_operating_point(self):
        rows = [_problem("p1", [0.1, 0.5, 0.9], ["1", "2", "42"])]
        grid = complete_threshold_grid([0.1, 0.5, 0.9])
        curve = sweep_threshold(rows, grid)
        # Four distinct behaviours: fire at each of the three boundaries, or
        # never. Two of them save the same - stopping at the final boundary
        # saves nothing, exactly as not stopping does - so the behaviours are
        # counted, not the savings.
        assert len({(p.fire_rate, round(p.mean_tokens_saved, 9))
                    for p in curve}) == 4

    def test_it_finds_points_a_linear_grid_of_the_same_size_misses(self):
        # Scores clustered at the extremes: a linear grid spends its points
        # in the empty middle, where no problem changes its firing boundary.
        scores = [0.001, 0.002, 0.003, 0.997, 0.998, 0.999]
        rows = [_problem("p1", scores, ["1", "2", "3", "4", "5", "42"])]
        linear = sweep_threshold(rows, list(np.linspace(0.001, 0.999, 7)))
        complete = sweep_threshold(rows, complete_threshold_grid(scores))
        n_linear = len({round(p.mean_tokens_saved, 9) for p in linear})
        n_complete = len({round(p.mean_tokens_saved, 9) for p in complete})
        assert n_complete > n_linear

    def test_a_finer_grid_adds_nothing_once_the_values_are_swept(self):
        """Exhaustive means exhaustive: no threshold between two adjacent
        observed scores can produce a new operating point."""
        scores = [0.2, 0.4, 0.6, 0.8]
        rows = [_problem("p1", scores, ["1", "2", "3", "42"]),
                _problem("p2", [0.3, 0.5, 0.7, 0.9], ["1", "2", "3", "42"])]
        all_scores = scores + [0.3, 0.5, 0.7, 0.9]
        exhaustive = sweep_threshold(rows, complete_threshold_grid(all_scores))
        dense = sweep_threshold(rows, list(np.linspace(0.0, 1.0, 20001)))
        assert ({round(p.mean_tokens_saved, 9) for p in exhaustive}
                == {round(p.mean_tokens_saved, 9) for p in dense})

    def test_no_finite_scores(self):
        assert complete_threshold_grid([]) == []
        assert complete_threshold_grid([None, float("nan")]) == []

    def test_a_constant_score_still_has_both_ends(self):
        curve = sweep_threshold(
            [_problem("p", [0.5, 0.5, 0.5], ["42"] * 3)],
            complete_threshold_grid([0.5, 0.5]))
        assert max(p.fire_rate for p in curve) == 1.0
        assert min(p.fire_rate for p in curve) == 0.0


class TestGranularityBound:
    """How much of the budget axis a rule can reach, given the scores observed.

    A score taking d distinct values admits at most d + 1 thresholds that
    behave differently, so on *this* corpus the curve cannot have more points
    than that. What the bound does across corpus sizes depends on the score:

    - A score bounded by construction - the agreement rule's original binary
      indicator, 1.0 once the last `window` answers agree and 0.0 otherwise -
      has d = 2 on 21 problems and d = 2 on 21,000, so three operating points
      either way. More data genuinely cannot help.
    - A score whose range grows with the data - the agreement run length that
      replaced it (D36) - has a bound that grows too.

    Conflating the two is what made D31 claim no corpus size could make the
    convergence head-to-head evaluable; see D37b.
    """

    def test_a_binary_score_is_bounded_at_three(self):
        assert granularity_bound([0.0, 1.0, 1.0, 0.0, 1.0]) == 3

    def test_a_bounded_score_does_not_gain_points_from_more_problems(self):
        few = granularity_bound([0.0, 1.0] * 10)
        many = granularity_bound([0.0, 1.0] * 10_000)
        assert few == many == 3

    def test_an_unbounded_score_gains_points_as_its_range_widens(self):
        """The corrected contract, and the reason D31 was withdrawn.

        Agreement run lengths from short traces reach 3; longer traces reach
        further, and each new run length is another achievable `w` and so
        another operating point. The bound is read off the scores present, not
        fixed for all N.
        """
        short_traces = granularity_bound([0.0, 1.0, 2.0, 3.0])
        longer_traces = granularity_bound([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        assert short_traces == 5
        assert longer_traces == 7
        assert longer_traces > short_traces

    def test_more_problems_at_the_same_trace_length_add_nothing(self):
        """It is the score's *range* that matters, not the number of problems.

        Ten problems and ten thousand reach the same bound when they contain
        the same run lengths - which is why "a larger corpus may help" is a
        claim about trace length and answer behaviour, not about N alone.
        """
        assert (granularity_bound([0.0, 1.0, 2.0] * 10)
                == granularity_bound([0.0, 1.0, 2.0] * 10_000)
                == 4)

    def test_the_bound_is_honoured_by_an_exhaustive_sweep(self):
        scores = [0.0, 1.0]
        rows = [_problem(f"p{i}", [0.0, 1.0, 1.0], ["1", "42", "42"])
                for i in range(50)]
        curve = sweep_threshold(rows, complete_threshold_grid(scores))
        assert (len({round(p.mean_tokens_saved, 9) for p in curve})
                <= granularity_bound(scores))

    def test_a_continuous_score_is_not_bounded_low(self):
        assert granularity_bound(list(np.linspace(0, 1, 272))) == 273

    def test_non_finite_scores_do_not_count(self):
        assert granularity_bound([0.0, 1.0, None, float("nan")]) == 3


class TestSemanticEntropy:
    def test_unanimous_is_zero_entropy(self):
        r = semantic_entropy(["42", "42", "42"])
        assert r.n_clusters == 1
        assert r.semantic_entropy == pytest.approx(0.0)
        assert confidence_score(r) == pytest.approx(1.0)

    def test_equivalent_forms_collapse_but_lexical_does_not(self):
        # The whole point of the symbolic-equivalence adaptation.
        r = semantic_entropy(["0.5", r"\frac{1}{2}", "1/2"])
        assert r.n_clusters == 1
        assert r.semantic_entropy == pytest.approx(0.0)
        assert r.lexical_entropy > 0.0

    def test_disagreement_raises_entropy(self):
        r = semantic_entropy(["1", "2", "3", "4"])
        assert r.n_clusters == 4
        assert normalised_entropy(r) == pytest.approx(1.0)
        assert confidence_score(r) == pytest.approx(0.0)

    def test_unparsed_samples_lower_confidence(self):
        # Dropping them would make a problem the model mostly fails on look
        # confident - the signal would invert exactly where it matters.
        clean = semantic_entropy(["42", "42", "42", "42"])
        noisy = semantic_entropy(["42", None, None, None])
        assert confidence_score(noisy) < confidence_score(clean)

    def test_agreement_with_final(self):
        r = semantic_entropy(["42", "42", "7"], final_answer="42")
        assert r.agreement_with_final == pytest.approx(2 / 3)

    def test_empty(self):
        r = semantic_entropy([])
        assert r.n_samples == 0 and np.isnan(r.semantic_entropy)

    def test_no_samples_means_no_score_not_perfect_confidence(self):
        """A configuration can switch entropy sampling off entirely.

        Scoring that as confidence 1.0 would put a rule on the comparison
        curve that stops at the first boundary of every problem and label it
        semantic entropy. The arm has to be unavailable instead.
        """
        r = semantic_entropy([])
        assert np.isnan(normalised_entropy(r))
        assert np.isnan(confidence_score(r))

    def test_a_single_sample_is_confident_because_it_cannot_disagree(self):
        r = semantic_entropy(["42"])
        assert normalised_entropy(r) == pytest.approx(0.0)
        assert confidence_score(r) == pytest.approx(1.0)

    def test_method_is_labelled_as_the_adaptation(self):
        # It is not Farquhar et al.'s NLI clustering and must not be reported
        # as though it were.
        assert semantic_entropy(["1"]).method == "semantic_entropy_symbolic"
