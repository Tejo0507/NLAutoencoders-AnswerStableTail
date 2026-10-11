"""The causal arm: the edit, the controls, and the gate on RQ3.

Zhang & Nanda's point is that an activation-patching conclusion depends on the
metric, the corruption, the site and the normalisation, so this project turns
their recommendations into a gate in code: `direction_claim_supported` cannot
return True unless the candidate direction beats a matched-random direction
*and* the same direction at a matched pre-stabilisation position, with a
monotone dose-response, while leaving final answers intact.

A gate is only worth having if it actually refuses. These tests check that it
refuses for each reason separately, that the edit it is reading does what its
name says - removes a component over a span and nothing else - and that the
candidate and its controls are compared at the same projection coefficient.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nlaast.causal.interventions import (  # noqa: E402
    InterventionResult,
    count_verification_markers,
    direction_claim_supported,
    dose_response,
    filler_text,
    projection_editor,
    random_direction,
    screen_outcomes,
    summarise_arm,
)

D = 16


def unit(i: int, d: int = D) -> np.ndarray:
    v = np.zeros(d, dtype=np.float32)
    v[i] = 1.0
    return v


class TestProjectionEditor:
    def test_full_ablation_removes_the_component(self):
        edit = projection_editor(unit(3), 1.0, span=None)
        h = torch.randn(1, 6, D)
        out = edit(h)
        d = torch.from_numpy(unit(3))
        assert torch.allclose(out @ d, torch.zeros(1, 6), atol=1e-5)

    def test_the_orthogonal_part_is_untouched(self):
        edit = projection_editor(unit(3), 1.0, span=None)
        h = torch.randn(1, 6, D)
        out = edit(h)
        keep = [i for i in range(D) if i != 3]
        assert torch.allclose(out[:, :, keep], h[:, :, keep], atol=1e-6)

    def test_coefficient_zero_is_a_no_op(self):
        """The within-arm control: dose 0.0 must change nothing at all, or the
        dose-response curve has no baseline."""
        edit = projection_editor(unit(3), 0.0, span=None)
        h = torch.randn(1, 6, D)
        assert torch.allclose(edit(h), h)

    def test_the_dose_scales_the_removal(self):
        h = torch.zeros(1, 1, D)
        h[0, 0, 3] = 4.0
        d = torch.from_numpy(unit(3))
        for coeff, expected in ((0.5, 2.0), (1.0, 0.0), (2.0, -4.0)):
            out = projection_editor(unit(3), coeff, span=None)(h)
            assert float(out @ d) == pytest.approx(expected, abs=1e-5)

    def test_the_span_restricts_the_edit(self):
        """Without this the tail intervention is a whole-sequence
        intervention and cannot distinguish the tail from anything else."""
        edit = projection_editor(unit(3), 1.0, span=(2, 4))
        h = torch.randn(1, 6, D)
        out = edit(h)
        d = torch.from_numpy(unit(3))
        inside = (out[:, 2:4, :] @ d).abs().max()
        assert float(inside) < 1e-5
        assert torch.allclose(out[:, :2, :], h[:, :2, :])
        assert torch.allclose(out[:, 4:, :], h[:, 4:, :])

    def test_a_span_outside_the_sequence_is_clamped_not_an_error(self):
        edit = projection_editor(unit(3), 1.0, span=(100, 200))
        h = torch.randn(1, 6, D)
        assert torch.allclose(edit(h), h)

    def test_the_input_tensor_is_not_mutated(self):
        """The hook returns a new tensor; mutating in place would corrupt the
        unintervened forward pass that runs in the same process."""
        edit = projection_editor(unit(3), 1.0, span=None)
        h = torch.randn(1, 6, D)
        before = h.clone()
        edit(h)
        assert torch.allclose(h, before)

    def test_the_span_is_absolute_across_a_cached_decode(self):
        """The defect that made every direction arm a no-op.

        Cached decoding calls the hook once with the whole prompt and then once
        per new token with a sequence length of 1. The causal stage edits the
        *generated* continuation, so its span starts at the prompt length. Read
        as an index into each tensor, that span never intersects anything: the
        prompt call is clipped at its own end and every decode call is one
        position wide. The pilot's sweep returned identical answers and token
        counts at doses 0.0, 1.0 and 2.0 on all ten problems as a result.
        """
        d = torch.from_numpy(unit(3))
        prompt_len, n_new = 5, 4
        edit = projection_editor(unit(3), 1.0, span=(prompt_len, prompt_len + n_new))

        prompt = torch.randn(1, prompt_len, D)
        out_prompt = edit(prompt)
        assert torch.allclose(out_prompt, prompt), "the prompt must be untouched"

        for _ in range(n_new):
            step = torch.zeros(1, 1, D)
            step[0, 0, 3] = 2.0
            out = edit(step)
            assert float(out @ d) == pytest.approx(0.0, abs=1e-5)
        assert edit.n_positions_edited == n_new

    def test_the_edit_stops_at_the_end_of_the_span(self):
        edit = projection_editor(unit(3), 1.0, span=(2, 4))
        edit(torch.randn(1, 2, D))                 # positions 0-1, before
        for _ in range(2):                         # positions 2-3, inside
            step = torch.zeros(1, 1, D)
            step[0, 0, 3] = 1.0
            assert float(edit(step) @ torch.from_numpy(unit(3))) == pytest.approx(0.0)
        step = torch.zeros(1, 1, D)                # position 4, past the end
        step[0, 0, 3] = 1.0
        assert float(edit(step) @ torch.from_numpy(unit(3))) == pytest.approx(1.0)
        assert edit.n_positions_edited == 2

    def test_a_no_op_arm_is_detectable(self):
        """What the stage asserts on. An intervention that edits nothing
        reproduces the baseline exactly, which reads as 'the direction does
        nothing' rather than as 'the edit never happened'."""
        edit = projection_editor(unit(3), 1.0, span=(100, 200))
        edit(torch.randn(1, 6, D))
        assert edit.n_positions_edited == 0

    def test_reset_restarts_the_position_count(self):
        edit = projection_editor(unit(3), 1.0, span=(0, 2))
        edit(torch.randn(1, 4, D))
        assert edit.n_positions_edited == 2
        edit.reset()
        assert edit.n_positions_edited == 0
        edit(torch.randn(1, 4, D))
        assert edit.n_positions_edited == 2

    def test_a_whole_sequence_edit_still_follows_the_stream(self):
        edit = projection_editor(unit(3), 1.0, span=None)
        d = torch.from_numpy(unit(3))
        for width in (5, 1, 1):
            h = torch.zeros(1, width, D)
            h[0, :, 3] = 3.0
            assert float(edit(h).abs().max()) == pytest.approx(0.0, abs=1e-5)
            assert torch.allclose(edit(h) @ d, torch.zeros(1, width), atol=1e-5)

    def test_a_zero_direction_is_refused(self):
        with pytest.raises(ValueError, match="non-zero"):
            projection_editor(np.zeros(D), 1.0, span=None)

    def test_the_direction_is_normalised_so_its_scale_does_not_matter(self):
        h = torch.randn(1, 4, D)
        a = projection_editor(unit(3) * 17.0, 1.0, span=None)(h)
        b = projection_editor(unit(3), 1.0, span=None)(h)
        assert torch.allclose(a, b, atol=1e-5)


class TestRandomDirection:
    def test_it_is_a_unit_vector(self):
        rng = np.random.default_rng(0)
        assert np.linalg.norm(random_direction(D, rng)) == pytest.approx(1.0)

    def test_it_is_orthogonal_to_the_candidate(self):
        """A control that partially coincides with the candidate is weakest
        exactly when the candidate is real."""
        rng = np.random.default_rng(0)
        ref = random_direction(D, rng)
        for _ in range(10):
            r = random_direction(D, rng, reference=ref)
            assert abs(float(r @ ref)) < 1e-9

    def test_successive_draws_differ(self):
        rng = np.random.default_rng(0)
        a = random_direction(D, rng)
        b = random_direction(D, rng)
        assert not np.allclose(a, b)


class TestFiller:
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * max(1, len(text) // 4)}

    def test_it_matches_the_requested_token_budget_from_above(self):
        """Matched on tokens, because budget is counted in tokens everywhere
        else; never shorter, or the filler arm would also be a shorter arm."""
        tok = self._Tok()
        for want in (1, 10, 64):
            text = filler_text(want, "Let us consider this further. ", tok)
            got = len(tok(text)["input_ids"])
            assert got >= want

    def test_a_non_positive_budget_is_empty(self):
        assert filler_text(0, "x ", self._Tok()) == ""
        assert filler_text(-5, "x ", self._Tok()) == ""


class TestMarkers:
    def test_markers_are_counted_case_insensitively(self):
        assert count_verification_markers("Let me check. LET ME CHECK.") == 2

    def test_unrelated_text_scores_zero(self):
        assert count_verification_markers("The sum of two and three is five.") == 0

    def test_none_and_empty_are_zero(self):
        assert count_verification_markers(None) == 0
        assert count_verification_markers("") == 0


def result(arm, coeff, *, markers, baseline_markers=0, changed=False,
           correct=True, pid="p0", did=None, tokens=10, edited=None):
    return InterventionResult(
        problem_id=pid, intervention=arm, coefficient=coeff, answer="1",
        baseline_answer="1", gold="1", answer_changed=changed, correct=correct,
        baseline_correct=True, tokens_generated=tokens,
        verification_markers=markers,
        baseline_verification_markers=baseline_markers,
        direction_id=did,
        notes={} if edited is None else {"n_positions_edited": edited},
    )


class TestArmSummary:
    def test_pooling_every_dose_is_the_default(self):
        rows = [result("ablate_dir", 0.0, markers=0),
                result("ablate_dir", 1.0, markers=4)]
        s = summarise_arm(rows, "ablate_dir")
        assert s.n == 2
        assert s.coefficient is None
        assert s.mean_markers == pytest.approx(2.0)

    def test_one_dose_can_be_selected(self):
        rows = [result("ablate_dir", 0.0, markers=0),
                result("ablate_dir", 1.0, markers=4)]
        s = summarise_arm(rows, "ablate_dir", coefficient=1.0)
        assert s.n == 1
        assert s.mean_markers == pytest.approx(4.0)

    def test_an_absent_arm_summarises_to_nan_rather_than_raising(self):
        s = summarise_arm([], "ablate_dir")
        assert s.n == 0
        assert np.isnan(s.marker_delta)


class TestDoseResponse:
    def test_points_come_out_in_dose_order(self):
        rows = [result("ablate_dir", 2.0, markers=6),
                result("ablate_dir", 0.0, markers=0),
                result("ablate_dir", 1.0, markers=3)]
        curve = dose_response(rows, "ablate_dir")
        assert [p["coefficient"] for p in curve] == [0.0, 1.0, 2.0]
        assert [p["marker_delta"] for p in curve] == [0.0, 3.0, 6.0]


def supported_evidence(**overrides):
    """A result set that satisfies every condition, for one-at-a-time breaking."""
    rows = []
    for pid in range(6):
        rows += [
            result("ablate_dir", 0.0, markers=0, pid=f"p{pid}"),
            result("ablate_dir", 1.0, markers=3, pid=f"p{pid}"),
            result("ablate_dir", 2.0, markers=6, pid=f"p{pid}"),
            result("random_dir", 1.0, markers=overrides.get("random_markers", 0),
                   pid=f"p{pid}", did="random_0"),
            result("matched_position", 1.0,
                   markers=overrides.get("matched_markers", 0), pid=f"p{pid}"),
        ]
    if overrides.get("answers_change"):
        rows = [result(r.intervention, r.coefficient, markers=r.verification_markers,
                       changed=r.intervention == "ablate_dir", pid=r.problem_id,
                       did=r.direction_id)
                for r in rows]
    return rows


class TestTheGate:
    def test_it_can_be_satisfied(self):
        v = direction_claim_supported(supported_evidence())
        assert v["supported"] is True
        assert v["compared_at_coefficient"] == 1.0

    def test_the_candidate_is_read_at_the_controls_dose(self):
        """Pooling ablate_dir over its sweep would compare a diluted candidate
        against undiluted controls.

        Here the no-op dose drags the pooled mean *below* the dose the controls
        ran at, which is the direction that would make a real effect fail the
        gate. The gate must read 1.0; the pooled figure is still reported, but
        separately.
        """
        rows = []
        for pid in range(6):
            rows += [
                result("ablate_dir", 0.0, markers=0, pid=f"p{pid}"),
                result("ablate_dir", 1.0, markers=3, pid=f"p{pid}"),
                result("random_dir", 1.0, markers=2, pid=f"p{pid}", did="random_0"),
                result("matched_position", 1.0, markers=2, pid=f"p{pid}"),
            ]
        v = direction_claim_supported(rows)
        assert v["arms"]["ablate_dir"]["coefficient"] == 1.0
        assert v["arms"]["ablate_dir"]["mean_markers"] == pytest.approx(3.0)
        assert v["arms_pooled_over_doses"]["ablate_dir"]["mean_markers"] == \
            pytest.approx(1.5)
        # At the matched dose the candidate beats both controls; pooled it
        # would not have.
        assert v["beats_random_direction"] is True
        assert v["beats_matched_position"] is True

    def test_a_random_direction_doing_the_same_breaks_it(self):
        v = direction_claim_supported(supported_evidence(random_markers=3))
        assert v["beats_random_direction"] is False
        assert v["supported"] is False

    def test_the_same_direction_at_a_matched_position_doing_the_same_breaks_it(self):
        v = direction_claim_supported(supported_evidence(matched_markers=5))
        assert v["beats_matched_position"] is False
        assert v["supported"] is False

    def test_a_non_monotone_dose_response_breaks_it(self):
        rows = [r for r in supported_evidence()
                if not (r.intervention == "ablate_dir" and r.coefficient == 2.0)]
        rows += [result("ablate_dir", 2.0, markers=1, pid=f"p{p}") for p in range(6)]
        v = direction_claim_supported(rows)
        assert v["monotone_dose_response"] is False
        assert v["supported"] is False

    def test_too_few_dose_points_cannot_establish_monotonicity(self):
        rows = [r for r in supported_evidence() if r.coefficient != 2.0]
        v = direction_claim_supported(rows)
        assert v["monotone_dose_response"] is False

    def test_changing_the_answer_breaks_it(self):
        """RQ3 asks for a *selective* effect: behaviour moves, answer does not."""
        v = direction_claim_supported(supported_evidence(answers_change=True))
        assert v["answers_preserved"] is False
        assert v["supported"] is False

    def test_no_evidence_at_all_is_untested_not_refuted(self):
        v = direction_claim_supported([])
        assert v["supported"] is None
        assert v["outcome_varies"] is False
        assert v["untestable_reason"]


class TestAConstantOutcomeIsNotAResult:
    """The pilot's case, and the reason `supported` can be None.

    The readout counts lexical rechecking markers, and on the pilot corpus it
    is identically zero: not one of the twenty markers appears in any of the
    24 traces, because this model writes clean derivations with no "wait",
    "actually" or "let me check". With a constant outcome, |0| > |0| is false
    and every comparison in the gate fails - so it returned "not supported",
    which reads as evidence against a tail direction when there is none
    either way.
    """

    @staticmethod
    def _all_zero():
        rows = []
        for pid in range(6):
            rows += [
                result("ablate_dir", 0.0, markers=0, pid=f"p{pid}"),
                result("ablate_dir", 1.0, markers=0, pid=f"p{pid}"),
                result("ablate_dir", 2.0, markers=0, pid=f"p{pid}"),
                result("random_dir", 1.0, markers=0, pid=f"p{pid}", did="random_0"),
                result("matched_position", 1.0, markers=0, pid=f"p{pid}"),
            ]
        return rows

    def test_a_constant_readout_yields_untested_not_unsupported(self):
        v = direction_claim_supported(self._all_zero())
        assert v["supported"] is None
        assert v["outcome_varies"] is False
        assert "no test of the direction claim either way" in v["untestable_reason"]

    def test_the_control_comparisons_are_null_rather_than_false(self):
        """False would assert the candidate failed to beat the controls. It
        did not fail; nothing was measured."""
        v = direction_claim_supported(self._all_zero())
        assert v["beats_random_direction"] is None
        assert v["beats_matched_position"] is None

    def test_a_varying_readout_is_judged_normally(self):
        v = direction_claim_supported(supported_evidence())
        assert v["outcome_varies"] is True
        assert v["untestable_reason"] is None
        assert v["supported"] is True

    def test_the_readout_is_named_in_the_verdict(self):
        v = direction_claim_supported(self._all_zero())
        assert "verification_markers" in v["outcome"]


class TestOutcomeScreen:
    """Which readouts could test anything, reported beside the verdict.

    RQ3's pre-registered readout is identically zero on this model, so the gate
    has no power. Replacing it by whichever alternative gives the nicest answer
    would be choosing the hypothesis after seeing the data, so every candidate
    is screened for the one property a dependent variable needs first - that it
    varies between the candidate arm and its controls.
    """

    @staticmethod
    def _rows(tokens_by_arm, edited=5):
        out = []
        for arm, toks in tokens_by_arm.items():
            for i, t in enumerate(toks):
                out.append(result(arm, 1.0, markers=0, tokens=t,
                                  pid=f"p{i}", edited=edited))
        return out

    def test_a_constant_readout_is_not_usable(self):
        s = screen_outcomes(self._rows({
            "ablate_dir": [10, 10], "random_dir": [10, 10],
            "matched_position": [10, 10]}))
        assert s["outcomes"]["continuation_tokens"]["varies"] is False
        assert "continuation_tokens" not in s["usable_outcomes"]
        assert s["outcomes"]["marker_delta"]["varies"] is False

    def test_a_readout_that_separates_the_arms_is_usable(self):
        s = screen_outcomes(self._rows({
            "ablate_dir": [40, 50], "random_dir": [10, 12],
            "matched_position": [11, 13]}))
        entry = s["outcomes"]["continuation_tokens"]
        assert entry["varies"] and entry["separates_arms"]
        assert "continuation_tokens" in s["usable_outcomes"]
        assert entry["by_arm"]["ablate_dir"]["mean"] == pytest.approx(45.0)

    def test_variation_without_arm_separation_is_not_usable(self):
        """Noise inside every arm is not a signal between them."""
        s = screen_outcomes(self._rows({
            "ablate_dir": [10, 30], "random_dir": [30, 10],
            "matched_position": [20, 20]}))
        entry = s["outcomes"]["continuation_tokens"]
        assert entry["varies"] is True
        assert entry["separates_arms"] is False
        assert "continuation_tokens" not in s["usable_outcomes"]

    def test_a_broken_intervention_is_named_as_such(self):
        """The defect the screen exists to surface: if no arm edited anything,
        the arms equal the baseline by construction and no readout can
        distinguish them. That is not a negative result."""
        s = screen_outcomes(self._rows({
            "ablate_dir": [10, 10], "random_dir": [10, 10],
            "matched_position": [10, 10]}, edited=0))
        assert s["interventions_were_no_ops"] is True
        assert "broken independent variable" in s["warning"]
        assert s["arms_that_edited_nothing"] == 6

    def test_a_working_intervention_is_not_flagged(self):
        s = screen_outcomes(self._rows({
            "ablate_dir": [40, 50], "random_dir": [10, 12],
            "matched_position": [11, 13]}, edited=7))
        assert "interventions_were_no_ops" not in s
        assert s["arms_that_edited_nothing"] == 0

    def test_the_pre_registered_readout_is_marked(self):
        s = screen_outcomes(self._rows({"ablate_dir": [10], "random_dir": [10],
                                        "matched_position": [10]}))
        flagged = [n for n, e in s["outcomes"].items() if e["pre_registered"]]
        assert flagged == ["marker_delta"]

    def test_the_screen_rides_along_with_the_verdict(self):
        rows = self._rows({"ablate_dir": [40], "random_dir": [10],
                           "matched_position": [11]})
        v = direction_claim_supported(rows)
        assert "outcome_screen" in v
        assert v["supported"] is None          # marker readout still constant
        assert "continuation_tokens" in v["outcome_screen"]["usable_outcomes"]

    def test_no_results_at_all(self):
        s = screen_outcomes([])
        assert s["n_direction_results"] == 0
        assert s["usable_outcomes"] == []
