"""RQ3's member of the pre-registered hypothesis-test family.

That test used to be a Mann-Whitney U over every `ablate_dir` row against
every `random_dir` row, which was wrong twice over. The arms run on the same
problems, so discarding the pairing is the error D29 corrected for the primary
O3 comparison. And pooling the candidate's whole dose sweep put dose 0.0 - a
no-op edit by definition - into the candidate arm, diluting it against
undiluted controls; `summarise_arm` already documented that hazard, so the
registered test contradicted the verdict computed beside it.

It is now the same paired contrast the gate uses, read at the controls' dose,
with one extra condition: the candidate has to be demonstrated to have edited
something, or the arms' agreement is evidence about the intervention rather
than about the direction (D34).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def agg():
    return importlib.import_module("aggregate_results")


def row(arm, coeff, pid, markers, *, baseline=0, edited=40, did=None):
    return {
        "problem_id": pid, "intervention": arm, "coefficient": coeff,
        "answer": "1", "baseline_answer": "1", "gold": "1",
        "answer_changed": False, "correct": True, "baseline_correct": True,
        "tokens_generated": 20,
        "verification_markers": markers,
        "baseline_verification_markers": baseline,
        "direction_id": did,
        "notes": {} if edited is None else {"n_positions_edited": edited},
    }


def corpus(cand, ctrl, n=6, edited=40):
    """One candidate dose sweep plus a random-direction control per problem."""
    out = []
    for i in range(n):
        pid = f"p{i}"
        out += [row("ablate_dir", 0.0, pid, 0, edited=edited),
                row("ablate_dir", 1.0, pid, cand[i], edited=edited),
                row("ablate_dir", 2.0, pid, cand[i] * 2, edited=edited),
                row("random_dir", 1.0, pid, ctrl[i], edited=edited,
                    did="random_0")]
    return out


class TestTheContrastIsPairedAndDosed:
    def test_it_pairs_over_problems_not_rows(self, agg):
        out = agg.rq3_paired_contrast(corpus([4] * 6, [0] * 6), 0.95)
        assert out["n_pairs"] == 6, "six problems, not eighteen candidate rows"
        assert out["test_result"].n == 6
        assert out["test_result"].notes["unit_of_analysis"] == "problem"

    def test_only_the_controls_dose_enters_the_candidate_arm(self, agg):
        """Dose 0.0 is a no-op; pooling it would dilute the candidate."""
        out = agg.rq3_paired_contrast(corpus([6] * 6, [0] * 6), 0.95)
        assert out["dose"] == 1.0
        assert out["median_abs_effect_candidate"] == pytest.approx(6.0)
        assert out["test_result"].notes["dose"] == 1.0

    def test_a_consistent_candidate_win_is_significant(self, agg):
        out = agg.rq3_paired_contrast(corpus([5] * 6, [0] * 6), 0.95)
        t = out["test_result"]
        assert np.isfinite(t.p_value) and t.p_value <= 0.05
        assert out["beats"] is True
        assert t.notes["n_pairs_favouring_candidate"] == 6

    def test_a_mean_advantage_without_paired_support_does_not_win(self, agg):
        """The defect the old unpaired test could not see."""
        cand = [12, 0, 0, 0, 0, 0]
        ctrl = [0, 1, 1, 1, 1, 1]
        out = agg.rq3_paired_contrast(corpus(cand, ctrl), 0.95)
        assert out["beats"] is False
        assert out["median_paired_difference"] < 0


class TestTheInterventionHasToHaveHappened:
    def test_unrecorded_edit_counts_make_the_test_unevaluable(self, agg):
        """Pilot artifacts predate the counter; absence is not evidence."""
        out = agg.rq3_paired_contrast(corpus([4] * 6, [0] * 6, edited=None), 0.95)
        assert out["intervention_demonstrated"] is False
        t = out["test_result"]
        assert not np.isfinite(t.p_value)
        assert t.notes["broken_independent_variable"] is True

    def test_zero_edits_make_the_test_unevaluable(self, agg):
        out = agg.rq3_paired_contrast(corpus([4] * 6, [0] * 6, edited=0), 0.95)
        assert out["intervention_demonstrated"] is False
        assert not np.isfinite(out["test_result"].p_value)

    def test_identical_arms_with_a_real_edit_is_a_tested_null(self, agg):
        """D38's argument, but only once the edit is demonstrated.

        Exact agreement between arms that were genuinely intervened on is a
        null that was tested and not rejected, so it carries p = 1.0 and
        counts towards the BH denominator. Wilcoxon is undefined here, which
        is why this needs stating.
        """
        out = agg.rq3_paired_contrast(corpus([3] * 6, [3] * 6), 0.95)
        assert out["intervention_demonstrated"] is True
        assert out["test_result"].p_value == 1.0
        assert out["beats"] is False

    def test_identical_arms_without_a_real_edit_is_not_a_tested_null(self, agg):
        """The same numbers, and the opposite conclusion. This is the D34/D38
        interaction: agreement under a no-op says nothing about the direction."""
        out = agg.rq3_paired_contrast(
            corpus([3] * 6, [3] * 6, edited=0), 0.95)
        assert not np.isfinite(out["test_result"].p_value)


class TestDegenerateInputs:
    def test_no_rows_at_all(self, agg):
        out = agg.rq3_paired_contrast([], 0.95)
        assert out["available"] is False
        assert "test_result" not in out

    def test_too_few_shared_problems(self, agg):
        out = agg.rq3_paired_contrast(corpus([4] * 2, [0] * 2, n=2), 0.95)
        assert out["n_pairs"] == 2
        assert out["beats"] is None
        assert not np.isfinite(out["test_result"].p_value)

    def test_malformed_rows_are_counted_not_fatal(self, agg):
        rows = corpus([4] * 6, [0] * 6)
        rows.append({"intervention": "ablate_dir"})          # no problem_id
        out = agg.rq3_paired_contrast(rows, 0.95)
        assert out["malformed_rows_skipped"] == 1
        assert out["n_pairs"] == 6

    def test_a_missing_control_arm_is_uncomparable(self, agg):
        rows = [r for r in corpus([4] * 6, [0] * 6)
                if r["intervention"] != "random_dir"]
        out = agg.rq3_paired_contrast(rows, 0.95)
        assert out["n_pairs"] == 0
        assert out["beats"] is None
