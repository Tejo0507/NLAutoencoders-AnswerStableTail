"""Answer-Stable Tail detection.

The detector takes evidence as data and decides, so its logic is fully testable
without a GPU. These tests pin the three criteria and the edge-case policy.
"""

import numpy as np
import pytest

from nlaast.ast_detect import (
    BoundaryEvidence,
    TailStatus,
    agreement_run_length,
    convergence_boundary,
    detect_ast,
    matched_windows,
)


def ev(index, parsed=None, forced=None, matches=False, k=3, matching=None,
       prefix_tokens=None):
    matching = k if matching is None else matching
    return BoundaryEvidence(
        index=index,
        parsed_answer=parsed,
        forced_answer=forced,
        forced_matches_final=matches,
        continuation_answers=["42"] * k,
        continuations_matching=matching,
        prefix_tokens=prefix_tokens if prefix_tokens is not None else (index + 1) * 10,
    )


def run(evidence, final="42", gold="42", n=None, total=100, truncated=False, **kw):
    return detect_ast(
        problem_id="p1", evidence=evidence, final_answer=final, gold=gold,
        n_chunks=n if n is not None else len(evidence), total_tokens=total,
        truncated=truncated, **kw,
    )


class TestConvergenceBaseline:
    def test_fires_when_window_agrees(self):
        assert convergence_boundary(["1", "2", "7", "7", "7"], 2) == 3

    def test_window_of_three(self):
        assert convergence_boundary(["1", "7", "7", "7"], 3) == 3

    def test_none_blocks_agreement(self):
        assert convergence_boundary(["7", None, "7", "7"], 2) == 3

    def test_never_agrees(self):
        assert convergence_boundary(["1", "2", "3"], 2) is None

    def test_equivalent_not_identical(self):
        # "0.5" and "1/2" are the same answer.
        assert convergence_boundary(["1", "0.5", "1/2"], 2) == 2

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError):
            convergence_boundary(["1"], 0)


class TestAgreementRunLength:
    """The graded score must *be* the pre-registered rule, re-parameterised.

    The O3 comparison sweeps a threshold, so the baseline was adapted to it by
    scoring 1.0 once `w` answers agree at one fixed `w` and 0.0 otherwise -
    two values, so three operating points at any corpus size. Scoring the
    agreement run length puts `w` back on the sweep axis, which is where the
    pre-registration puts it ("`w` configurable").

    That is only legitimate if thresholding the score at `w` is the same rule.
    This class asserts the identity instead of trusting the argument.
    """

    @staticmethod
    def _first_at_least(scores, w):
        for i, s in enumerate(scores):
            if s >= w:
                return i
        return None

    @pytest.mark.parametrize("parsed", [
        ["1", "2", "7", "7", "7"],
        ["1", "7", "7", "7"],
        ["7", None, "7", "7"],
        ["1", "2", "3"],
        ["1", "0.5", "1/2"],
        [None, None, None],
        ["7", "7", "7", "1", "7", "7"],
        ["7"],
        [],
    ])
    def test_threshold_w_is_convergence_boundary_w(self, parsed):
        runs = agreement_run_length(parsed)
        for w in range(1, len(parsed) + 2):
            assert self._first_at_least(runs, w) == convergence_boundary(parsed, w), (
                f"w={w} parsed={parsed} runs={runs}")

    def test_the_score_is_the_run_length(self):
        assert agreement_run_length(["1", "7", "7", "7"]) == [1.0, 1.0, 2.0, 3.0]

    def test_unparseable_scores_zero_and_breaks_the_run(self):
        assert agreement_run_length(["7", None, "7", "7"]) == [1.0, 0.0, 1.0, 2.0]

    def test_a_disagreement_restarts_the_run(self):
        assert agreement_run_length(["7", "7", "1", "7"]) == [1.0, 2.0, 1.0, 1.0]

    def test_equivalent_forms_extend_the_run(self):
        # "0.5" and "1/2" are the same answer, so the run continues.
        assert agreement_run_length(["1", "0.5", "1/2"]) == [1.0, 1.0, 2.0]

    def test_it_is_graded_where_the_binary_score_was_not(self):
        """The point of the change: enough distinct values to resolve a curve."""
        runs = agreement_run_length(["7"] * 8)
        assert runs == [float(i) for i in range(1, 9)]
        assert len(set(runs)) == 8

    def test_empty(self):
        assert agreement_run_length([]) == []


class TestCriteria:
    def test_persistent_suffix_is_the_tail(self):
        r = run([ev(0), ev(1), ev(2, matches=True), ev(3, matches=True),
                 ev(4, matches=True)])
        assert r.status is TailStatus.OK
        assert r.tail_start == 2
        assert r.tail_fraction == pytest.approx(3 / 5)

    def test_later_failure_disqualifies_everything_before_it(self):
        # Criterion 3: a boundary that qualifies but is followed by one that
        # does not cannot start the tail.
        r = run([ev(0), ev(1, matches=True), ev(2), ev(3, matches=True)])
        assert r.tail_start == 3

    def test_forced_mismatch_blocks(self):
        r = run([ev(0, matches=False), ev(1, matches=False)])
        assert r.status is TailStatus.NO_STABLE_POINT
        assert r.tail_start is None

    def test_non_unanimous_continuations_block_when_required(self):
        r = run([ev(0), ev(1, matches=True, k=5, matching=4)],
                require_unanimous=True)
        assert r.tail_start is None

    def test_majority_accepted_when_unanimity_not_required(self):
        r = run([ev(0), ev(1, matches=True, k=5, matching=4)],
                require_unanimous=False)
        assert r.tail_start == 1

    def test_no_continuations_means_criterion_untested(self):
        # An untested criterion is not a satisfied one.
        r = run([ev(0), ev(1, matches=True, k=0, matching=0)])
        assert r.tail_start is None


class TestEdgeCases:
    """All of these stay in the corpus with a status code."""

    def test_unparseable_final(self):
        r = run([ev(0, matches=True)], final=None)
        assert r.status is TailStatus.UNPARSEABLE_FINAL
        assert r.tail_start is None

    def test_too_short(self):
        r = run([ev(0, matches=True)])
        assert r.status is TailStatus.TOO_SHORT

    def test_stable_at_zero(self):
        r = run([ev(0, matches=True), ev(1, matches=True), ev(2, matches=True)])
        assert r.status is TailStatus.STABLE_AT_ZERO
        assert r.tail_start == 0

    def test_truncated_keeps_the_measurement_and_flags_it(self):
        r = run([ev(0), ev(1, matches=True), ev(2, matches=True)], truncated=True)
        assert r.status is TailStatus.TRUNCATED
        assert r.tail_start == 1
        assert r.notes["underlying_status"] == "ok"

    def test_incorrect_final_answers_are_kept(self):
        # Excluding them would bias every stopping comparison toward problems
        # the model already got right.
        r = run([ev(0), ev(1, matches=True), ev(2, matches=True)],
                final="41", gold="42")
        assert r.tail_start == 1
        assert r.final_correct is False


class TestTokenAccounting:
    def test_tail_tokens_are_the_remainder_after_the_prefix(self):
        e = [ev(0, prefix_tokens=10), ev(1, matches=True, prefix_tokens=40),
             ev(2, matches=True, prefix_tokens=70)]
        r = run(e, total=100)
        assert r.tail_start == 1
        assert r.tail_tokens == 90   # total 100 minus the 10-token prefix at index 0

    def test_minimum_boundary_fraction(self):
        e = [ev(i, matches=True) for i in range(10)]
        r = run(e, min_boundary_fraction=0.5)
        assert r.tail_start == 5


class TestMatchedWindows:
    def test_shapes(self):
        w = matched_windows(tail_start=6, n_chunks=10, rng=np.random.default_rng(0))
        assert w["tail"] == (6, 9)
        assert w["matched_position"] == (2, 5)       # same length, just before
        assert w["matched_length"][1] - w["matched_length"][0] == 3

    def test_no_matched_position_when_tail_starts_at_zero(self):
        w = matched_windows(0, 5, np.random.default_rng(0))
        assert "matched_position" not in w

    def test_matched_position_clipped_at_the_start(self):
        w = matched_windows(tail_start=2, n_chunks=10, rng=np.random.default_rng(0))
        assert w["matched_position"] == (0, 1)

    def test_matched_length_stays_in_range(self):
        for seed in range(20):
            w = matched_windows(7, 10, np.random.default_rng(seed))
            lo, hi = w["matched_length"]
            assert 0 <= lo <= hi <= 9


class TestSerialisation:
    def test_round_trips_to_dict(self):
        r = run([ev(0), ev(1, matches=True), ev(2, matches=True)])
        d = r.to_dict()
        assert d["id"] == "p1"
        assert d["status"] == "ok"
        assert len(d["evidence"]) == 3
        assert d["evidence"][1]["continuations_unanimous"] is True
