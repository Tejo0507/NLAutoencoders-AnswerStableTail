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
    """
    import torch

    d = np.asarray(direction, dtype=np.float32).ravel()
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError("direction must be non-zero")
    d = d / n
    d_t = torch.from_numpy(d)

    def edit(hidden):
        nonlocal d_t
        if d_t.device != hidden.device or d_t.dtype != hidden.dtype:
            d_t = d_t.to(hidden.device, hidden.dtype)
        if span is None:
            lo, hi = 0, hidden.shape[1]
        else:
            lo, hi = span
            lo = max(0, min(lo, hidden.shape[1]))
            hi = max(lo, min(hi, hidden.shape[1]))
        if hi <= lo:
            return hidden
        out = hidden.clone()
        seg = out[:, lo:hi, :]
        proj = (seg @ d_t).unsqueeze(-1) * d_t
        out[:, lo:hi, :] = seg - coefficient * proj
        return out

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise_arm(results: Sequence[InterventionResult], arm: str) -> ArmSummary:
    rows = [r for r in results if r.intervention == arm]
    if not rows:
        return ArmSummary(arm, 0, float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"))
    return ArmSummary(
        arm=arm,
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


def direction_claim_supported(results: Sequence[InterventionResult]) -> dict[str, Any]:
    """Does the evidence support "a tail direction exists"?

    RQ3 asks whether tail-window states carry a direction whose ablation
    selectively changes verification-like behaviour *without* changing final
    answers. All four conditions below must hold. The function returns the
    individual verdicts as well as the conjunction, so a partial result is
    still reportable - which matters, because a partial result is the likely
    outcome at this scale.
    """
    ablate = summarise_arm(results, "ablate_dir")
    random_ = summarise_arm(results, "random_dir")
    matched = summarise_arm(results, "matched_position")
    curve = dose_response(results, "ablate_dir")

    def _finite(x: float) -> bool:
        return bool(np.isfinite(x))

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

    return {
        "beats_random_direction": beats_random,
        "beats_matched_position": beats_position,
        "monotone_dose_response": monotone,
        "answers_preserved": answer_preserved,
        "supported": bool(beats_random and beats_position and monotone and answer_preserved),
        "arms": {
            "ablate_dir": ablate.to_dict(),
            "random_dir": random_.to_dict(),
            "matched_position": matched.to_dict(),
        },
        "dose_response": curve,
    }
