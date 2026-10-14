"""The primary comparison: the readout against one cheaper rule, at matched budget.

This is the test the study's headline comparison did not have. It lived inside
`aggregate_results.main`, so nothing exercised it without a run directory, and
one of its branches was substituting the wrong rule's data.

**The defect.** `decisions_at` decides where a rule fires by reading
`row['scores']`. The comparison therefore needs the *other* rule's own
per-problem scores, not merely its curve. When a rule had a curve but no
re-swept rows the code passed `nla_rows` instead - the baseline's threshold
applied to the NLA readout's own cosines - and reported the result as a
McNemar test between the two rules. It is reachable per rule:
`baseline_curves_on` drops a rule with no usable score on the problems the NLA
arm covers while keeping the others, and the loop guarded on the curve rather
than on the rows.

The pilot had all three rules re-swept, so no published number came from that
branch; the tests below pin that it can no longer be taken, and that the
substitution it made was not harmless.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_results import NLA_RULE, o3_head_to_head  # noqa: E402
from nlaast.baselines.stopping import (  # noqa: E402
    complete_threshold_grid, decisions_at, sweep_threshold)

LOG = logging.getLogger("test_o3")


def rows_for(scores_by_problem, *, correct_at=None):
    """Per-problem rows in the shape `decisions_at` reads.

    Four boundaries per problem at 25/50/75/100 tokens of a 100-token trace.
    The answer is right from `correct_at` onwards, so stopping before it loses
    a correct answer and stopping at or after it is safe.
    """
    out = []
    for pid, scores in scores_by_problem.items():
        at = (correct_at or {}).get(pid, 0)
        out.append({
            "problem_id": pid,
            "scores": list(scores),
            "answers": ["7" if i >= at else "1" for i in range(len(scores))],
            "boundary_tokens": [25 * (i + 1) for i in range(len(scores))],
            "final_answer": "7",
            "gold": "7",
            "total_tokens": 100,
        })
    return out


def curve_of(rows):
    vals = [s for r in rows for s in r["scores"] if s is not None]
    return sweep_threshold(rows, complete_threshold_grid(vals), higher_fires=True)


#: Six problems. The readout and the baseline fire in the same places on four
#: of them and differently on two, which is what makes the comparison non-
#: degenerate without being lopsided.
NLA_SCORES = {"p0": [0.1, 0.9, 0.9, 0.9], "p1": [0.1, 0.2, 0.9, 0.9],
              "p2": [0.9, 0.9, 0.9, 0.9], "p3": [0.1, 0.2, 0.3, 0.9],
              "p4": [0.1, 0.9, 0.9, 0.9], "p5": [0.1, 0.2, 0.9, 0.9]}
OTHER_SCORES = {"p0": [0.9, 0.9, 0.9, 0.9], "p1": [0.1, 0.2, 0.9, 0.9],
                "p2": [0.9, 0.9, 0.9, 0.9], "p3": [0.1, 0.2, 0.3, 0.9],
                "p4": [0.9, 0.9, 0.9, 0.9], "p5": [0.1, 0.2, 0.9, 0.9]}
#: The answer is only correct from the second boundary, so a rule that fires at
#: the first is unsafe on that problem.
CORRECT_AT = {"p0": 1, "p1": 2, "p2": 0, "p3": 3, "p4": 1, "p5": 2}


@pytest.fixture
def arms():
    nla = rows_for(NLA_SCORES, correct_at=CORRECT_AT)
    other = rows_for(OTHER_SCORES, correct_at=CORRECT_AT)
    return nla, other, curve_of(nla), curve_of(other)


class TestTheOtherRuleNeedsItsOwnRows:
    def test_a_rule_without_per_problem_scores_is_unevaluable(self, arms):
        nla, _, nla_curve, other_curve = arms
        t, unmatched = o3_head_to_head(
            "semantic_entropy", nla_curve, other_curve, nla, None,
            [0.5], "nla_problem_set", {}, 0.95, LOG)
        assert unmatched is not None
        assert unmatched["per_problem_scores_available"] is False
        assert t.n == 0
        assert "no per-problem scores" in t.notes["skipped"]

    def test_it_produces_no_p_value_so_it_cannot_be_rejected(self, arms):
        """An absent comparison must not look like a tested null."""
        import numpy as np
        nla, _, nla_curve, other_curve = arms
        t, _ = o3_head_to_head(
            "probe", nla_curve, other_curve, nla, [], [0.5],
            "nla_problem_set", {}, 0.95, LOG)
        assert not np.isfinite(t.p_value)
        assert t.q_value is None
        assert t.interval is None

    def test_an_empty_row_list_is_refused_like_a_missing_one(self, arms):
        nla, _, nla_curve, other_curve = arms
        _, unmatched = o3_head_to_head(
            "convergence", nla_curve, other_curve, nla, [], [0.5],
            "nla_problem_set", {}, 0.95, LOG)
        assert unmatched["per_problem_scores_available"] is False

    def test_the_substitution_it_used_to_make_was_not_harmless(self, arms):
        """Why the refusal matters: the two row sets give different decisions.

        If passing the readout's rows in place of the baseline's were benign,
        refusing would be pedantry. At the threshold this baseline is read at,
        it is not: the decisions differ on the problems where the two rules
        fire in different places, so the McNemar table built from the wrong
        rows is a different table.
        """
        nla, other, _, _ = arms
        thr = 0.5
        from_own = {d.problem_id: d.safe for d in decisions_at(other, thr, True)}
        from_nla = {d.problem_id: d.safe for d in decisions_at(nla, thr, True)}
        assert from_own != from_nla
        disagreeing = {p for p in from_own if from_own[p] != from_nla[p]}
        assert disagreeing, "the fixture must separate the two rule's decisions"


class TestTheComparisonItself:
    def test_it_is_paired_over_problems_and_reports_the_discordant_count(self, arms):
        nla, other, nla_curve, other_curve = arms
        t, unmatched = o3_head_to_head(
            "probe", nla_curve, other_curve, nla, other,
            [0.5], "nla_problem_set", {}, 0.95, LOG)
        assert unmatched is None
        assert t.notes["test"] == "mcnemar_exact"
        assert t.notes["unit_of_analysis"] == "problem"
        assert t.notes["n_problems"] == 6
        assert t.notes["n_discordant"] == (
            t.notes["a_safe_b_not"] + t.notes["b_safe_a_not"])

    def test_the_sample_counts_of_both_arms_are_recorded(self, arms):
        """The test runs on the intersection, so the two arms' own counts have
        to be visible or a silently lost problem shows up only as a smaller n."""
        nla, other, nla_curve, other_curve = arms
        t, _ = o3_head_to_head(
            "probe", nla_curve, other_curve, nla, other,
            [0.5], "nla_problem_set", {}, 0.95, LOG)
        assert t.notes["n_problems_nla_arm"] == 6
        assert t.notes["n_problems_probe_arm"] == 6
        assert t.notes["n_problems_shared"] == 6
        assert t.notes["n_problems"] == t.notes["n_problems_shared"]

    def test_a_problem_only_one_arm_covers_is_dropped_from_the_pairing(self, arms):
        nla, other, nla_curve, other_curve = arms
        extra = rows_for({"p9": [0.9, 0.9, 0.9, 0.9]})
        t, _ = o3_head_to_head(
            "probe", nla_curve, other_curve, nla, other + extra,
            [0.5], "nla_problem_set", {}, 0.95, LOG)
        assert t.notes["n_problems_probe_arm"] == 7
        assert t.notes["n_problems_nla_arm"] == 6
        assert t.notes["n_problems_shared"] == 6
        assert t.notes["n_problems"] == 6

    def test_the_direction_is_stated_in_the_outcomes_own_units(self, arms):
        nla, other, nla_curve, other_curve = arms
        t, _ = o3_head_to_head(
            "probe", nla_curve, other_curve, nla, other,
            [0.5], "nla_problem_set", {}, 0.95, LOG)
        assert t.notes["direction"] in {"nla_readout better", "nla_readout worse",
                                        "tied"}
        if t.notes["safe_rate_nla"] > t.notes["safe_rate_probe"]:
            assert t.notes["direction"] == "nla_readout better"

    def test_repeating_a_budget_does_not_multiply_the_evidence(self, arms):
        """Nine copies of one budget must give the same answer as one copy.

        This is D29 in the comparison itself rather than in `paired_binary_test`:
        a span of budgets that resolves to one operating-point pair is one
        comparison, and the p-value must not move with the length of the span.
        """
        nla, other, nla_curve, other_curve = arms
        one, _ = o3_head_to_head("probe", nla_curve, other_curve, nla, other,
                                 [0.5], "nla_problem_set", {}, 0.95, LOG)
        nine, _ = o3_head_to_head("probe", nla_curve, other_curve, nla, other,
                                  [0.5] * 9, "nla_problem_set", {}, 0.95, LOG)
        assert nine.p_value == one.p_value
        assert nine.notes["n_distinct_operating_points"] == 1
        assert nine.notes["n_budgets_requested"] == 9


class TestUnmatchedDiagnosticsDescribeTheProblemsCompared:
    def test_the_distinct_score_count_is_taken_from_the_rows_compared(self, arms):
        """Not from the corpus-wide `curve_resolution`.

        The two scopes differ - on the pilot the probe takes 272 distinct
        values corpus-wide and 203 on the 21 problems the NLA arm covers - and
        a reason that quotes one to explain a failure in the other describes a
        different curve from the one that could not be matched.
        """
        nla, other, nla_curve, other_curve = arms
        # A budget no pair of points can meet, so the unmatched branch runs.
        t, unmatched = o3_head_to_head(
            "convergence", nla_curve, other_curve, nla, other, [0.99],
            "nla_problem_set",
            {"n_distinct_scores": 9999, "max_operating_points_at_any_n": 10000,
             "granularity_limited": True},
            0.95, LOG)
        assert unmatched is not None
        observed = len({s for r in other for s in r["scores"] if s is not None})
        assert unmatched["n_distinct_scores"] == observed
        assert unmatched["max_operating_points_at_any_n"] == observed + 1
        assert unmatched["n_problems_compared"] == len(other)
        assert unmatched["diagnostics_basis"] == "nla_problem_set"
        # And the corpus-wide count is not what the reader is told.
        assert "9999" not in unmatched["reason"]
        assert "10000" not in unmatched["reason"]

    def test_the_corpus_wide_figures_are_kept_beside_it(self, arms):
        nla, other, nla_curve, other_curve = arms
        _, unmatched = o3_head_to_head(
            "convergence", nla_curve, other_curve, nla, other, [0.99],
            "nla_problem_set", {"n_distinct_scores": 9999}, 0.95, LOG)
        assert unmatched["full_corpus_resolution"] == {"n_distinct_scores": 9999}

    def test_a_corpus_wide_granularity_flag_does_not_decide_the_verdict(self, arms):
        """`granularity_limited` is about the curve that failed to match.

        A score with three or fewer reachable operating points on the problems
        compared is granularity-limited there; one with many is not, whatever
        the corpus-wide file says.
        """
        nla, other, nla_curve, other_curve = arms
        _, unmatched = o3_head_to_head(
            "convergence", nla_curve, other_curve, nla, other, [0.99],
            "nla_problem_set", {"granularity_limited": True}, 0.95, LOG)
        assert unmatched["granularity_limited"] is False

    def test_a_genuinely_coarse_score_is_reported_as_granularity_limited(self):
        binary = rows_for({f"p{i}": [0.0, 0.0, 1.0, 1.0] for i in range(6)},
                          correct_at={f"p{i}": 1 for i in range(6)})
        nla = rows_for({f"p{i}": [0.1, 0.4, 0.6, 0.9] for i in range(6)},
                       correct_at={f"p{i}": 1 for i in range(6)})
        _, unmatched = o3_head_to_head(
            "convergence", curve_of(nla), curve_of(binary), nla, binary,
            [0.99], "nla_problem_set", {}, 0.95, LOG)
        assert unmatched is not None
        assert unmatched["granularity_limited"] is True
        assert unmatched["max_operating_points_at_any_n"] == 3
        assert "needs a graded score" in unmatched["reason"]

    def test_an_unmatched_comparison_states_that_the_sweep_was_exhaustive(self, arms):
        nla, other, nla_curve, other_curve = arms
        _, unmatched = o3_head_to_head(
            "convergence", nla_curve, other_curve, nla, other, [0.99],
            "nla_problem_set", {}, 0.95, LOG)
        assert unmatched["thresholds_exhaustive"] is True
        assert unmatched["budgets_tried"] == 1


def test_the_test_name_is_stable_across_every_branch(arms):
    """The report finds these tests by name prefix, so every branch must agree."""
    nla, other, nla_curve, other_curve = arms
    expected = (f"O3: safe-stopping rate, {NLA_RULE} minus probe "
                f"at matched budget")
    made, _ = o3_head_to_head("probe", nla_curve, other_curve, nla, other,
                              [0.5], "nla_problem_set", {}, 0.95, LOG)
    norows, _ = o3_head_to_head("probe", nla_curve, other_curve, nla, None,
                                [0.5], "nla_problem_set", {}, 0.95, LOG)
    unmatched, _ = o3_head_to_head("probe", nla_curve, other_curve, nla, other,
                                   [0.99], "nla_problem_set", {}, 0.95, LOG)
    assert made.name == norows.name == unmatched.name == expected
    assert made.name.startswith("O3:")
