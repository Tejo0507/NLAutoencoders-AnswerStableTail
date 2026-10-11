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
        entry["pre_registered"] = name == "marker_delta"
        out["outcomes"][name] = entry
    usable = [n for n, e in out["outcomes"].items()
              if e["varies"] and e["separates_arms"]]
    out["usable_outcomes"] = usable
    out["note"] = (
        "exploratory. The verdict is computed from the pre-registered "
        "marker_delta; these are reported so that a readout with power can be "
        "pre-registered for a later run, not so that one can be chosen now "
        "because it gave the preferred answer.")
    return out


def direction_claim_supported(
    results: Sequence[InterventionResult], compare_at: float = 1.0
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
    marker_values = [r.verification_markers - r.baseline_verification_markers
                     for r in results
                     if r.intervention in ("ablate_dir", "random_dir",
                                           "matched_position")]
    outcome_varies = bool(marker_values) and len(set(marker_values)) > 1

    beats_random = (
        _finite(ablate.marker_delta) and _finite(random_.marker_delta)
        and abs(ablate.marker_delta) > abs(random_.marker_delta)
    )
    beats_position = (
        _finite(ablate.marker_delta) and _finite(matched.marker_delta)
        and abs(ablate.marker_delta) > abs(matched.marker_delta)
    )
    deltas = [p["marker_delta"] for p in curve if np.isfinite(p["marker_delta"])]
    monotone = len(deltas) >= 3 and all(
        abs(deltas[i + 1]) >= abs(deltas[i]) - 1e-9 for i in range(len(deltas) - 1)
    )
    # "Selectively": behaviour moves, the answer does not.
    answer_preserved = _finite(ablate.answer_change_rate) and ablate.answer_change_rate < 0.2

    supported: bool | None = bool(
        beats_random and beats_position and monotone and answer_preserved)
    if not outcome_varies:
        # Untested, not refuted.
        supported = None

    return {
        "beats_random_direction": beats_random if outcome_varies else None,
        "beats_matched_position": beats_position if outcome_varies else None,
        "monotone_dose_response": monotone,
        "answers_preserved": answer_preserved,
        "supported": supported,
        "outcome_varies": outcome_varies,
        "outcome": "verification_markers - baseline_verification_markers",
        "untestable_reason": (
            None if outcome_varies else
            "the behavioural readout is constant across every direction arm, "
            "so the controls cannot be beaten or missed: this corpus provides "
            "no test of the direction claim either way. On the pilot the "
            "rechecking markers the readout counts do not occur in any trace."
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
