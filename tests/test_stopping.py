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

    def test_method_is_labelled_as_the_adaptation(self):
        # It is not Farquhar et al.'s NLI clustering and must not be reported
        # as though it were.
        assert semantic_entropy(["1"]).method == "semantic_entropy_symbolic"
