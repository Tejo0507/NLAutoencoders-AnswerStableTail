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
    evaluate_stop,
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
