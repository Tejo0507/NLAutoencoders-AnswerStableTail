"""Causal validation of the tail (O5 / RQ3).

Zhang & Nanda show that the conclusion drawn from an activation-patching
experiment depends heavily on the metric, the corruption, the restoration site
and the normalisation, and recommend matched-random-direction controls,
matched-position controls and dose-response curves before an intervention is
read as mechanistic evidence. That recommendation is implemented here as a
structural constraint, not as a caveat in the write-up:

``direction_claim_supported`` will not return ``True`` unless the candidate
direction beats **both** a matched-random direction **and** the same direction
applied at a matched pre-stabilisation position, with a monotone dose-response.

The interventions:

| name | what it does | what it controls for |
|---|---|---|
| ``truncate`` | stop generating at the tail start | the behavioural upper bound on what the tail contributes |
| ``filler`` | replace tail text with neutral filler of matched token length | token count, as distinct from tail content |
| ``ablate_dir`` | project the candidate tail direction out of the layer-K residual stream over the tail span | the hypothesis under test |
| ``random_dir`` | project out a random direction of matched norm | that *any* rank-1 edit of this size would do it |
| ``matched_position`` | ``ablate_dir`` applied to the pre-stabilisation window | that the direction is about the tail, not about late positions |

The candidate direction is fitted on the **training split only**. Fitting it on
the problems it is then evaluated on would make the whole exercise circular.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Sequence

import numpy as np

from ..logging_utils import get

log = get(__name__)


@dataclass
class InterventionResult:
    problem_id: str
    intervention: str
    coefficient: float
    #: Answer produced under the intervention.
    answer: str | None
    #: Answer the unintervened trace produced.
    baseline_answer: str | None
    gold: str | None
    answer_changed: bool
    correct: bool
    baseline_correct: bool
    tokens_generated: int
    #: Surface markers of verification-like continuation, counted the same way
    #: in every arm so the comparison is like-for-like.
    verification_markers: int = 0
    baseline_verification_markers: int = 0
    direction_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = f"{self.problem_id}|{self.intervention}|{self.coefficient}|{self.direction_id}"
        return d


#: Lexical markers of rechecking. These are a *behavioural readout*, not a
#: label for an internal state - Project_Review_II.md section 2.1.6 is explicit
#: that surface text cannot be read as a log of the computation. They are used
#: only to compare arms against each other.
VERIFICATION_MARKERS: tuple[str, ...] = (
    "let me check", "let me verify", "double-check", "double check", "let's verify",
    "let me re-read", "verify that", "check my", "check that", "to confirm",
    "confirming", "wait", "actually", "hold on", "re-examine", "recheck",
    "make sure", "sanity check", "but let", "hmm",
)


def count_verification_markers(text: str) -> int:
    low = (text or "").lower()
    return sum(low.count(m) for m in VERIFICATION_MARKERS)


# --- Direction edits ---


def projection_editor(
    direction: np.ndarray,
    coefficient: float,
    span: tuple[int, int] | None,
    device_dtype_from: Any = None,
) -> Callable:
    """An edit that removes ``coefficient`` times the component along ``direction``.

    ``coefficient = 1.0`` is full ablation of that component; the sweep over
    coefficients is the dose-response curve. ``span`` restricts the edit to
    ``[start, end)`` absolute token positions, leaving the rest of the sequence
    untouched - without that, the "tail" intervention would be a whole-sequence
    intervention and could not distinguish the tail from anything else.

    **Absolute means absolute across the whole generation, not an index into
    whatever tensor this call happens to receive.** Cached decoding calls the
    hook once with the whole prompt and then once per new token with a
    sequence length of 1, so a span taken as an index into ``hidden`` is
    clipped away to nothing: the causal stage edits the *generated*
    continuation, whose positions all lie at or beyond the prompt length, and
    every arm silently became a no-op. The pilot's dose sweep returned
    byte-identical answers and token counts at coefficients 0.0, 1.0 and 2.0
    on all ten problems, which is what a no-op looks like when nothing checks.

    The editor therefore tracks how many positions it has been shown and maps
    the span onto each call. This assumes one contiguous left-to-right pass
    per ``reset()`` - true of greedy and sampled decoding, false of beam
    search, which is not used here. ``edit.n_positions_edited`` counts the
    positions actually modified so a caller can assert the intervention
    happened rather than trusting that it did.
    """
    import torch

    d = np.asarray(direction, dtype=np.float32).ravel()
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError("direction must be non-zero")
    d = d / n
    d_t = torch.from_numpy(d)
    state = {"seen": 0, "edited": 0}

    def edit(hidden):
        nonlocal d_t
        if d_t.device != hidden.device or d_t.dtype != hidden.dtype:
            d_t = d_t.to(hidden.device, hidden.dtype)
        width = hidden.shape[1]
        offset = state["seen"]
        state["seen"] = offset + width
        if span is None:
            lo, hi = 0, width
        else:
            # Intersect the absolute span with the absolute window this call
            # covers, then express it relative to the tensor.
            lo = max(span[0] - offset, 0)
            hi = min(span[1] - offset, width)
        if hi <= lo:
            return hidden
        out = hidden.clone()
        seg = out[:, lo:hi, :]
        proj = (seg @ d_t).unsqueeze(-1) * d_t
        out[:, lo:hi, :] = seg - coefficient * proj
        state["edited"] += hi - lo
        edit.n_positions_edited = state["edited"]
        return out

    def reset() -> None:
        state["seen"] = 0
        state["edited"] = 0
        edit.n_positions_edited = 0

    edit.n_positions_edited = 0
    edit.reset = reset
    return edit


def random_direction(d_model: int, rng, reference: np.ndarray | None = None) -> np.ndarray:
    """A random unit direction.

    If ``reference`` is given, the random direction is drawn in the orthogonal
    complement of it. That makes the control strictly a *different* direction
    rather than one that might partially coincide with the candidate, which
    would weaken the control exactly when the candidate is real.
    """
    v = rng.standard_normal(d_model).astype(np.float64)
    if reference is not None:
        r = np.asarray(reference, dtype=np.float64).ravel()
        r = r / max(np.linalg.norm(r), 1e-12)
        v = v - (v @ r) * r
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def filler_text(n_tokens: int, unit: str, tokenizer) -> str:
    """Neutral filler of approximately ``n_tokens`` tokens.

    Matched on token count rather than character count because the comparison
    the filler arm supports is "same budget, no tail content", and budget is
    counted in tokens everywhere else in the study.
    """
    if n_tokens <= 0:
        return ""
    per = max(1, len(tokenizer(unit, add_special_tokens=False)["input_ids"]))
    return (unit * max(1, (n_tokens + per - 1) // per)).strip()


# --- Reading the result ---


@dataclass
class ArmSummary:
    arm: str
    n: int
    answer_change_rate: float
    accuracy: float
    accuracy_delta: float
    mean_markers: float
    marker_delta: float
    #: ``None`` when the summary pools every dose of this arm.
    coefficient: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise_arm(results: Sequence[InterventionResult], arm: str,
                  coefficient: float | None = None) -> ArmSummary:
    """Aggregate one arm, optionally at one dose.

    ``coefficient`` matters for the gate below. ``ablate_dir`` is swept over a
    dose grid that includes 0.0 - a no-op edit, kept as the within-arm control
    - while ``random_dir`` and ``matched_position`` are run at full strength
    only. Pooling the sweep and comparing it against those would compare a
    diluted candidate against undiluted controls, and would understate the
    candidate for a reason that has nothing to do with the hypothesis.
    """
    rows = [r for r in results if r.intervention == arm]
    if coefficient is not None:
        rows = [r for r in rows if abs(r.coefficient - coefficient) < 1e-9]
    if not rows:
        return ArmSummary(arm, 0, float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), coefficient)
    return ArmSummary(
        arm=arm,
        coefficient=coefficient,
        n=len(rows),
        answer_change_rate=float(np.mean([r.answer_changed for r in rows])),
        accuracy=float(np.mean([r.correct for r in rows])),
        accuracy_delta=float(
            np.mean([r.correct for r in rows]) - np.mean([r.baseline_correct for r in rows])
        ),
        mean_markers=float(np.mean([r.verification_markers for r in rows])),
        marker_delta=float(
            np.mean([r.verification_markers - r.baseline_verification_markers for r in rows])
        ),
    )


def dose_response(results: Sequence[InterventionResult], arm: str) -> list[dict[str, float]]:
    rows = [r for r in results if r.intervention == arm]
    out = []
    for c in sorted({r.coefficient for r in rows}):
        sel = [r for r in rows if r.coefficient == c]
        out.append(
            {
                "coefficient": float(c),
                "n": len(sel),
                "marker_delta": float(
                    np.mean([r.verification_markers - r.baseline_verification_markers for r in sel])
                ),
                "answer_change_rate": float(np.mean([r.answer_changed for r in sel])),
                "accuracy": float(np.mean([r.correct for r in sel])),
            }
        )
    return out


#: Candidate behavioural readouts for RQ3, each a per-result scalar.
#:
#: The pre-registered one is the marker delta, and it stays the one the verdict
#: is computed from. It is also identically zero on this model, which writes
#: clean step-by-step derivations containing none of the twenty rechecking
#: markers, so it has no power here and the gate correctly declines to answer.
#:
#: Choosing a replacement by looking at which one gives the nicest answer would
#: be choosing the hypothesis after seeing the data. So every candidate is
#: screened on the same run for the one property a dependent variable must have
#: before it can test anything - that it varies - and the screen is reported
#: beside the verdict as exploratory. The next run's screen says which readout
#: a pre-registered test could use; it does not itself test the direction.
#:
#: ``continuation_tokens`` is the leading candidate because it is already
#: recorded, is continuous, and asks how much further the model writes before
#: concluding - a behavioural question that needs no judgement about what the
#: text means. Project_Review_II section 2.1.6 rules out reading surface text
#: as a log of the computation, and a token count does not.
OUTCOMES: dict[str, Callable[[InterventionResult], float]] = {
    "marker_delta": lambda r: float(
        r.verification_markers - r.baseline_verification_markers),
    "continuation_tokens": lambda r: float(r.tokens_generated),
    "answer_changed": lambda r: float(r.answer_changed),
    "correct": lambda r: float(r.correct),
}

#: Arms the screen looks at: the candidate direction and its two controls.
_DIRECTION_ARMS = ("ablate_dir", "random_dir", "matched_position")

#: Which registered construct each candidate readout operationalises.
#:
#: RQ3's construct *is* registered, in ``Project_Review_II.md`` line 122:
#: "do tail-window states carry a direction whose ablation selectively changes
#: **verification-like behaviour** without changing final answers?" What is
#: missing is an *operationalisation* of that construct, not the construct
#: (D41). That distinction constrains which readouts are even eligible: a
#: readout measuring something else cannot stand in, however much better
#: behaved it is.
#:
#: ``continuation_tokens`` is the case in point. It has variance where
#: ``marker_delta`` has none, which is why it kept being proposed - but "how
#: much further the model writes" is output length, not verification. Swapping
#: it in would answer a different question under RQ3's name. It is therefore
#: recorded as a *different* construct and is not eligible as the verdict
#: outcome without an amendment that re-states RQ3 itself.
OUTCOME_CONSTRUCT: dict[str, str] = {
    "marker_delta": "verification_like_behaviour",
    "continuation_tokens": "output_length",
    "answer_changed": "answer_preservation",
    "correct": "answer_correctness",
}

#: The construct RQ3 is registered against.
REGISTERED_CONSTRUCT = "verification_like_behaviour"

#: What each readout computes, in words, for the verdict artifact. The key in
#: ``OUTCOMES`` is the machine-readable name; this is what a reader needs.
_OUTCOME_DEFINITIONS: dict[str, str] = {
    "marker_delta": "verification_markers - baseline_verification_markers",
    "continuation_tokens": "tokens_generated",
    "answer_changed": "answer_changed (0/1)",
    "correct": "correct (0/1)",
}

#: Readouts that operationalise the registered construct, so are eligible to
#: be the verdict outcome at all. Eligibility is about construct validity; it
#: says nothing about whether a readout has the variance to test anything,
#: which is what ``screen_outcomes`` measures separately.
ELIGIBLE_OUTCOMES = tuple(
    name for name, c in OUTCOME_CONSTRUCT.items() if c == REGISTERED_CONSTRUCT)

#: The readout ``direction_claim_supported`` gates on - the single source of
#: truth, read through ``OUTCOMES`` so the gate and the screen cannot disagree.
#:
#: The pre-registration names no operationalisation (D40/D41), so this is the
#: implementation's choice among the eligible ones, not a registered
#: commitment. It is the face-valid reading of "verification-like behaviour":
#: a count of rechecking markers. Its known defect is a floor effect on this
#: model, which is a *measurement-power* problem rather than a construct
#: mismatch, and is reported as such rather than fixed by substituting a
#: readout that measures something else.
#:
#: Changing this is a scientific decision requiring an amendment, not an edit
#: here; ``_assert_eligible`` enforces that it at least stays in-construct.
VERDICT_OUTCOME = "marker_delta"

#: Significance level the two "beat" comparisons must clear. See
#: ``paired_arm_contrast`` for why a bare comparison of arm means was not
#: enough.
BEAT_ALPHA = 0.05

#: Fewest problems a paired contrast needs before it is attempted at all.
MIN_PAIRS = 3

#: Fewest distinct doses a dose-response curve needs.
MIN_DOSES = 3


def _assert_eligible(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    if outcome not in ELIGIBLE_OUTCOMES:
        raise ValueError(
            f"{outcome!r} measures {OUTCOME_CONSTRUCT.get(outcome)!r}, not the "
            f"registered construct {REGISTERED_CONSTRUCT!r}. Using it as RQ3's "
            f"verdict outcome would answer a different question under RQ3's "
            f"name; that needs an amendment restating RQ3, not a changed "
            f"constant (D41).")


#: Slack added to the candidate's span beyond the original tail's token
#: length, to cover a regeneration that runs longer than the tail it replaces.
#:
#: Measured on the pilot it never binds: the candidate's span was a median of
#: 148 positions while the regenerated continuation was 35.5, and the editor
#: can only edit positions the forward pass produces. So this is a safety
#: margin, not an extent, and trimming it would change nothing (D39).
SPAN_SLACK_TOKENS = 64


def intervention_spans(
    prompt_tokens: int,
    tail_tokens: int,
    pre_window_tokens: int,
    slack: int = SPAN_SLACK_TOKENS,
) -> dict[str, tuple[int, int]]:
    """The candidate and matched-position spans, in absolute token positions.

    Extracted from ``run_interventions.run_problem`` so the registered design
    is testable rather than only runnable. Behaviour is unchanged; this is the
    same arithmetic the stage already used.

    The two spans meet at the prompt boundary without overlapping:

    - ``tail`` starts at ``prompt_tokens``, so it covers the *generated*
      continuation - the regenerated tail - and nothing before it.
    - ``pre`` ends at ``prompt_tokens``, so it covers the pre-stabilisation
      window, which lies inside the prompt because the prefix up to the tail
      start is fed as prompt. The control editing prompt context is therefore
      the registered design, not a bug: PROJECT_PLAN.md §5 defines
      matched-position as "a window of length L ending at i*-1".

    That difference in *kind* - editing representations as they are generated
    versus editing already-fixed context - is why matching the two on extent
    would not make them the same experiment, and why D39 retains the asymmetry
    and reports it instead.
    """
    if prompt_tokens < 0 or tail_tokens < 0:
        raise ValueError("token counts must be non-negative")
    tail = (prompt_tokens, prompt_tokens + tail_tokens + max(0, slack))
    pre_len = max(1, int(pre_window_tokens))
    pre = (max(0, prompt_tokens - pre_len), prompt_tokens)
    return {"tail": tail, "pre": pre}


def spans_are_disjoint(spans: dict[str, tuple[int, int]]) -> bool:
    """Do the candidate and control spans share no position?

    Asserted by the stage rather than assumed, because an overlap would let
    the matched-position control edit part of the region the candidate edits,
    and the location contrast would then be comparing a window against itself.
    """
    (a0, a1), (b0, b1) = spans["tail"], spans["pre"]
    return min(a1, b1) <= max(a0, b0)


def paired_arm_contrast(
    results: Sequence[InterventionResult],
    candidate: str,
    control: str,
    outcome: str = VERDICT_OUTCOME,
    compare_at: float | None = 1.0,
    alpha: float = BEAT_ALPHA,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Does *candidate* move the outcome further than *control*, on the same problems?

    This is what "beat" means, and it used to mean much less. The gate compared
    two arm *means* - ``abs(ablate.marker_delta) > abs(random_.marker_delta)``
    - which has no inferential content at all: a candidate at 0.30 against a
    control at 0.29 satisfied it, on ten problems, with no test, no interval
    and no sensitivity to how many problems were behind either number.

    Every arm is run on the same problems, so the unit of analysis is the
    problem and the comparison is paired. This is the same correction D29 made
    to the primary O3 comparison, which had been pairing over budget levels
    instead of problems; the identical argument applies here and the two are
    now consistent.

    Per problem the quantity is ``|outcome(candidate)| - |outcome(control)|``:
    the absolute value because RQ3 asks whether ablation *changes* behaviour,
    not which way, so what has to exceed the control is the magnitude of the
    change. Replicate control directions (``random_dir`` is run at several
    ``direction_id``s) are averaged within a problem first, so a problem counts
    once however many replicates it has.

    ``beats`` is True only if the paired difference runs the right way *and*
    the paired test clears ``alpha``. It is ``None`` - not False - whenever the
    comparison could not be made: too few shared problems, or arms that are
    identical so the test is undefined. False is reserved for a comparison that
    was made and that the candidate did not win.
    """
    from ..analysis.stats import paired_test

    fn = OUTCOMES[outcome]

    def by_problem(arm: str) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for r in results:
            if r.intervention != arm:
                continue
            if compare_at is not None and abs(r.coefficient - compare_at) > 1e-9:
                continue
            v = fn(r)
            if not np.isfinite(v):
                continue
            acc.setdefault(r.problem_id, []).append(abs(float(v)))
        return {p: float(np.mean(vs)) for p, vs in acc.items()}

    a, b = by_problem(candidate), by_problem(control)
    shared = sorted(set(a) & set(b))
    out: dict[str, Any] = {
        "candidate": candidate,
        "control": control,
        "outcome": outcome,
        "dose": compare_at,
        "n_pairs": len(shared),
        "n_candidate_problems": len(a),
        "n_control_problems": len(b),
        "test": "wilcoxon_signed_rank_paired_on_problems",
    }
    if len(shared) < MIN_PAIRS:
        out["beats"] = None
        out["reason"] = (
            f"only {len(shared)} problems appear in both arms at dose "
            f"{compare_at}; {MIN_PAIRS} are needed before a paired contrast "
            f"can be attempted")
        return out

    av = [a[p] for p in shared]
    bv = [b[p] for p in shared]
    diffs = [x - y for x, y in zip(av, bv)]
    # The paired arrays themselves, so a caller can re-derive this test
    # exactly - the RQ3 family test does - and so the artifact is auditable
    # without re-reading interventions.jsonl.
    out["problem_ids"] = shared
    out["paired_candidate_values"] = av
    out["paired_control_values"] = bv
    out["median_abs_effect_candidate"] = float(np.median(av))
    out["median_abs_effect_control"] = float(np.median(bv))
    out["median_paired_difference"] = float(np.median(diffs))
    out["n_pairs_favouring_candidate"] = int(sum(1 for d in diffs if d > 0))
    out["n_pairs_favouring_control"] = int(sum(1 for d in diffs if d < 0))
    out["n_pairs_tied"] = int(sum(1 for d in diffs if d == 0))

    if all(x == 0 for x in av) and all(y == 0 for y in bv):
        # The readout is constant at zero in both arms: the instrument
        # registered no behaviour change anywhere, so there was nothing to
        # beat or to fail to beat. That is no test (D27), never evidence that
        # the direction does nothing - on this model the rechecking markers
        # are at their floor, so silence here is a property of the readout.
        out["beats"] = None
        out["readout_constant_zero"] = True
        out["reason"] = (
            f"the readout is zero for both arms on every shared problem at "
            f"dose {compare_at}, so the instrument registered no change in "
            f"either arm and this contrast tests nothing; it is not evidence "
            f"against the direction")
        return out

    if all(d == 0 for d in diffs):
        # Identical *non-zero* movement in both arms. Whether this is a
        # determinate refusal (below) or also untested is an open
        # researcher decision (DECISIONS.md D46); the behaviour is unchanged
        # pending it.
        #
        # Determinate, and False rather than None. Wilcoxon is undefined on
        # all-zero differences, but "beat" is one-sided and the substantive
        # answer does not need the test: the candidate moved the outcome by
        # exactly as much as its control on every single problem, which is the
        # strongest evidence available *against* direction-specificity. The
        # exact one-sided p for "candidate exceeds control" when no pair
        # favours the candidate is 1.0.
        #
        # Contrast the zero-discordance case in `paired_binary_test` (D38),
        # which is also p = 1.0 but there supports a *two-sided* null. Here the
        # claim is directional, so p = 1.0 is a refusal, not a tie.
        out["p_value"] = 1.0
        out["alpha"] = alpha
        out["direction_favours_candidate"] = False
        out["beats"] = False
        out["reason"] = (
            "the candidate and the control moved the outcome identically on "
            "every shared problem, so the candidate does not exceed it")
        return out

    t = paired_test(f"RQ3: {candidate} minus {control} ({outcome})",
                    av, bv, confidence=confidence)
    out["p_value"] = None if not np.isfinite(t.p_value) else float(t.p_value)
    out["effect"] = None if not np.isfinite(t.effect) else float(t.effect)
    out["effect_name"] = t.effect_name
    out["interval"] = t.interval.to_dict() if t.interval else None
    if out["p_value"] is None:
        out["beats"] = None
        out["reason"] = t.notes.get(
            "skipped", "the paired test returned no p-value")
        return out
    out["alpha"] = alpha
    out["direction_favours_candidate"] = bool(out["median_paired_difference"] > 0)
    out["beats"] = bool(out["direction_favours_candidate"]
                        and out["p_value"] <= alpha)
    return out


def screen_outcomes(
    results: Sequence[InterventionResult], compare_at: float = 1.0
) -> dict[str, Any]:
    """Which candidate readouts vary enough to be able to test anything.

    A readout that is constant across the direction arms provides no test, and
    that is a statement about power rather than about the hypothesis. This
    reports, per candidate: whether it varies at all, its spread, and its mean
    in each arm, so a reader can see which readouts were available and which
    were empty rather than being shown only the one that was chosen.

    It also reports whether the edit reached the computation. Three of the five
    arms are direction edits, and if their spans never intersected the forward
    pass they reproduce the baseline exactly - which looks like a readout with
    no effect instead of an intervention that did not happen.

    Finally it reports *how much* each arm edited
    (``edit_extent_by_arm``), because the two pre-registered controls are not
    matched on extent in the same way: ``random_dir`` shares the candidate's
    span exactly, while ``matched_position`` is the matched-length window
    before the tail and edits its whole span in one pass. See the comment at
    that block and D39.
    """
    rows = [r for r in results if r.intervention in _DIRECTION_ARMS]
    edited = [int((r.notes or {}).get("n_positions_edited", -1)) for r in rows]
    known = [e for e in edited if e >= 0]
    out: dict[str, Any] = {
        "n_direction_results": len(rows),
        "edit_positions_recorded": len(known),
        "arms_that_edited_nothing": sum(
            1 for r, e in zip(rows, edited)
            if e == 0 and abs(r.coefficient) > 1e-9),
        "outcomes": {},
    }
    if known and all(e == 0 for e in known):
        out["interventions_were_no_ops"] = True
        out["warning"] = (
            "every direction arm edited zero positions, so the arms are "
            "identical to the unintervened baseline by construction and no "
            "readout can distinguish them; this is a broken independent "
            "variable, not a negative result")

    # How *much* each arm edited, not just whether it edited at all.
    #
    # The pre-registration's two controls do different jobs, and only one of
    # them is matched on extent. `random_dir` is given the candidate's own
    # span, so that contrast isolates the direction exactly. `matched_position`
    # is the pre-registered matched-length window ending before the tail, which
    # lives inside the prompt - so it is delivered in a single forward pass and
    # edits its whole span, while the candidate edits only as far as generation
    # actually ran. On the pilot's spans that is a median of 72 positions
    # against 36: the location contrast also differs in extent, and in the
    # direction that makes the control the *stronger* edit.
    #
    # Reported rather than corrected, because matching the two would require
    # either redefining the pre-registered control window or a second
    # generation pass to learn the length first (D39). A reader comparing
    # `ablate_dir` with `matched_position` needs this beside the arm means.
    extent: dict[str, Any] = {}
    for arm in _DIRECTION_ARMS:
        vals = [int((r.notes or {}).get("n_positions_edited", -1))
                for r in rows if r.intervention == arm]
        vals = [v for v in vals if v >= 0]
        extent[arm] = {
            "n": len(vals),
            "median_positions_edited": float(np.median(vals)) if vals else float("nan"),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    out["edit_extent_by_arm"] = extent
    cand = extent.get("ablate_dir", {}).get("median_positions_edited", float("nan"))
    out["extent_matched_candidate_vs_random"] = bool(
        np.isfinite(cand)
        and cand == extent.get("random_dir", {}).get("median_positions_edited"))
    mp = extent.get("matched_position", {}).get("median_positions_edited", float("nan"))
    if np.isfinite(cand) and np.isfinite(mp) and cand > 0:
        out["extent_ratio_matched_position_over_candidate"] = float(mp / cand)
    for name, fn in OUTCOMES.items():
        vals = [fn(r) for r in rows]
        finite = [v for v in vals if np.isfinite(v)]
        entry: dict[str, Any] = {
            "varies": len(set(finite)) > 1,
            "n": len(finite),
            "n_distinct": len(set(finite)),
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "sd": float(np.std(finite)) if len(finite) > 1 else 0.0,
            "by_arm": {},
        }
        for arm in _DIRECTION_ARMS:
            sel = [fn(r) for r in rows
                   if r.intervention == arm
                   and abs(r.coefficient - compare_at) < 1e-9]
            sel = [v for v in sel if np.isfinite(v)]
            entry["by_arm"][arm] = {
                "n": len(sel),
                "mean": float(np.mean(sel)) if sel else float("nan"),
            }
        # Varying overall is necessary but not sufficient: the readout also has
        # to differ between the candidate and its controls to carry a test.
        means = [entry["by_arm"][a]["mean"] for a in _DIRECTION_ARMS]
        finite_means = [m for m in means if np.isfinite(m)]
        entry["separates_arms"] = (
            len(finite_means) >= 2
            and float(np.max(finite_means) - np.min(finite_means)) > 0.0)
        # Which readout the gate reads, *not* a claim that it was registered.
        #
        # This flag used to be `pre_registered`, set True for `marker_delta`.
        # That was false: "verification markers" appear nowhere in
        # PROJECT_PLAN.md or in either review. The pre-registration fixes the
        # five arms, the two control comparisons and the requirement of a
        # dose-response curve, and names no dependent variable at all - "beat"
        # is left undefined (D40). Recording marker_delta as registered made
        # an implementation default look like a commitment made in advance.
        entry["is_verdict_outcome"] = name == VERDICT_OUTCOME
        entry["pre_registered"] = False
        out["outcomes"][name] = entry
    usable = [n for n, e in out["outcomes"].items()
              if e["varies"] and e["separates_arms"]]
    out["usable_outcomes"] = usable
    out["preregistered_outcome"] = None
    out["verdict_outcome"] = VERDICT_OUTCOME
    out["note"] = (
        "exploratory. The pre-registration specifies the arms, both control "
        "comparisons and a dose-response requirement, but **no dependent "
        f"variable**; the verdict is computed from `{VERDICT_OUTCOME}`, which "
        "is the implementation's default and not a registered choice (D40). "
        "These candidates are reported so that a readout with power can be "
        "registered for a later run, not so that one can be chosen now "
        "because it gave the preferred answer.")
    return out


def direction_claim_supported(
    results: Sequence[InterventionResult],
    compare_at: float = 1.0,
    outcome: str = VERDICT_OUTCOME,
    alpha: float = BEAT_ALPHA,
) -> dict[str, Any]:
    """Does the evidence support "a tail direction exists"?

    RQ3 asks whether tail-window states carry a direction whose ablation
    selectively changes verification-like behaviour *without* changing final
    answers. All four conditions below must hold. The function returns the
    individual verdicts as well as the conjunction, so a partial result is
    still reportable - which matters, because a partial result is the likely
    outcome at this scale.

    The controls run at one dose, so the candidate is read at that same dose
    (``compare_at``) rather than pooled over its sweep. The sweep is still
    reported in full, as the dose-response curve.

    ``outcome`` must operationalise the registered construct - see
    ``OUTCOME_CONSTRUCT`` and ``_assert_eligible``. It is read through
    ``OUTCOMES`` rather than off ``ArmSummary``, so this function and
    ``screen_outcomes`` cannot disagree about which readout the verdict used;
    previously ``VERDICT_OUTCOME`` labelled the screen while the gate reached
    past it to a hard-coded ``marker_delta``.

    **Every condition is tri-state.** True, False, and ``None`` for "could not
    be compared", and the conjunction is ``None`` if any input is. The three
    ways that happens are distinct and all reported: the readout does not vary
    at all, fewer than ``MIN_PAIRS`` problems appear in both arms, or the two
    arms moved the readout identically so the paired test is undefined. None of
    these is a negative result, and collapsing them into False would read as
    evidence against a tail direction where there is none either way.
    """
    ablate = summarise_arm(results, "ablate_dir", coefficient=compare_at)
    random_ = summarise_arm(results, "random_dir", coefficient=compare_at)
    matched = summarise_arm(results, "matched_position", coefficient=compare_at)
    curve = dose_response(results, "ablate_dir")

    def _finite(x: float) -> bool:
        return bool(np.isfinite(x))

    # Is there anything here to beat?
    #
    # The readout is a count of lexical rechecking markers, and on the pilot
    # corpus it is identically zero: not one of the twenty markers appears in
    # any of the 24 traces, because this model writes clean step-by-step
    # derivations with no "wait", "actually" or "let me check". With a constant
    # outcome, |0| > |0| is false and every comparison below fails - so the
    # gate returned "not supported", which reads as evidence against a tail
    # direction when there is no evidence either way.
    #
    # A constant outcome means the experiment had no power, and that is a
    # different statement from a negative result. It is reported as one.
    _assert_eligible(outcome)
    fn = OUTCOMES[outcome]
    outcome_values = [fn(r) for r in results if r.intervention in _DIRECTION_ARMS]
    outcome_values = [v for v in outcome_values if np.isfinite(v)]
    outcome_varies = bool(outcome_values) and len(set(outcome_values)) > 1

    # Both "beat" comparisons, paired over the problems each arm shares with
    # the candidate and tested rather than eyeballed. See
    # `paired_arm_contrast` for what this replaces.
    contrast_random = paired_arm_contrast(
        results, "ablate_dir", "random_dir", outcome, compare_at, alpha)
    contrast_matched = paired_arm_contrast(
        results, "ablate_dir", "matched_position", outcome, compare_at, alpha)
    beats_random = contrast_random["beats"]
    beats_position = contrast_matched["beats"]

    deltas = [p[outcome] for p in curve
              if outcome in p and np.isfinite(p[outcome])]
    # A constant readout satisfies "non-decreasing in |delta|" vacuously, so an
    # identically-zero outcome used to report `monotone_dose_response: true` -
    # the pilot's own summary.json says so, next to `outcome_varies: false`. A
    # reader scanning the four conditions saw one of them met when the
    # dose-response curve was flat at zero and nothing had been established.
    #
    # Monotonicity is computed from the same readout as `beats_*`, so it is
    # gated the same way: unknown when the readout does not vary. It also
    # requires the curve itself to move, rather than only to not go down -
    # a dose-response claim needs a response.
    # Distinct *doses*, not finite readings: a duplicated dose in the grid
    # would otherwise count towards the minimum without adding a point.
    n_doses = len({p["coefficient"] for p in curve
                   if outcome in p and np.isfinite(p[outcome])})
    monotone = (
        n_doses >= MIN_DOSES
        and len(set(deltas)) > 1
        and all(abs(deltas[i + 1]) >= abs(deltas[i]) - 1e-9
                for i in range(len(deltas) - 1))
    )
    # "Selectively": behaviour moves, the answer does not.
    answer_preserved = _finite(ablate.answer_change_rate) and ablate.answer_change_rate < 0.2

    # The conjunction. `beats_*` are now tri-state, so `all()` over them would
    # read None as False and turn "could not be compared" into "did not win".
    conditions = (beats_random, beats_position, monotone, answer_preserved)
    if not outcome_varies or any(c is None for c in conditions):
        supported: bool | None = None          # untested, not refuted
    else:
        supported = bool(all(conditions))

    return {
        "beats_random_direction": beats_random if outcome_varies else None,
        "beats_matched_position": beats_position if outcome_varies else None,
        "monotone_dose_response": monotone if outcome_varies else None,
        "contrast_random_direction": contrast_random,
        "contrast_matched_position": contrast_matched,
        "n_distinct_doses": n_doses,
        # Not gated on the readout: it is measured on a different variable
        # (whether the final answer changed), so it stays meaningful even when
        # the behavioural readout is flat. Under a no-op intervention it is
        # vacuous for another reason, which `screen_outcomes` reports as
        # `interventions_were_no_ops`.
        "answers_preserved": answer_preserved,
        "supported": supported,
        "outcome_varies": outcome_varies,
        "outcome": outcome,
        "outcome_definition": _OUTCOME_DEFINITIONS.get(outcome, outcome),
        "outcome_construct": OUTCOME_CONSTRUCT.get(outcome),
        "registered_construct": REGISTERED_CONSTRUCT,
        "outcome_is_preregistered": False,
        "construct_is_preregistered": True,
        "beat_definition": (
            f"|outcome(ablate_dir)| exceeds |outcome(control)| on the same "
            f"problems, by a Wilcoxon signed-rank test over problems at "
            f"alpha = {alpha}, read at dose {compare_at}"),
        "untestable_reason": (
            "the behavioural readout is constant across every direction arm, "
            "so the controls cannot be beaten or missed: this corpus provides "
            "no test of the direction claim either way. On the pilot the "
            "rechecking markers the readout counts do not occur in any trace."
            if not outcome_varies else
            "; ".join(f"{c['control']}: {c['reason']}"
                      for c in (contrast_random, contrast_matched)
                      if c.get("readout_constant_zero")) or None
        ),
        "compared_at_coefficient": compare_at,
        "arms": {
            "ablate_dir": ablate.to_dict(),
            "random_dir": random_.to_dict(),
            "matched_position": matched.to_dict(),
        },
        "arms_pooled_over_doses": {
            "ablate_dir": summarise_arm(results, "ablate_dir").to_dict(),
        },
        "dose_response": curve,
        "outcome_screen": screen_outcomes(results, compare_at),
        "note": ("the three arms above are read at the same projection "
                 "coefficient; ablate_dir's full sweep is dose_response"),
    }
