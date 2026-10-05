"""The claim-level audit (O4 / RQ2).

For each claim in a verbalisation, three perturbations and two controls:

| arm | question it answers |
|---|---|
| **delete** | does removing this claim degrade reconstruction? |
| **paraphrase** | how much does reconstruction move under a meaning-preserving edit? -> the **null** |
| **resample** | does an alternative claim the verbaliser produced for the *same* activation reconstruct as well? |
| **independent probe** | is the claim's content decodable from the activation by something that never saw the text? |
| **verbaliser-only** | does the verbaliser produce this claim from *noise* at matched norm? |

A claim is **reconstruction-dependent** iff its deletion effect exceeds the 95th
percentile of the paraphrase null for the same explanation. Using the
paraphrase arm as the null is what makes the threshold non-arbitrary: it is
calibrated to how much reconstruction moves when nothing semantic changed.

The verbaliser-only control is the one that can sink the whole NLA arm, and it
is run on every sampled verbalisation rather than a subset. Li et al. showed
that verbalisation methods can score well on standard benchmarks with no access
to target-model internals at all; if the claims here survive replacing the
activation with Gaussian noise, they are the verbaliser's priors and say nothing
about the target model.

This is a **RECAP-inspired** evaluation. It adopts RECAP's principle of
independent verification. It does not reproduce RECAP, which co-trains the
target model with linear decodability heads and therefore cannot be applied
retrospectively to released checkpoints (Project_Review_II.md section 2.1.9).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Sequence

import numpy as np

from ..config import FaithfulnessConfig
from ..logging_utils import get
from .claims import (
    Claim,
    delete_claim,
    keep_only,
    paraphrase_explanation,
    split_claims,
    substitute_claim,
)

log = get(__name__)

#: A reconstruction scorer: (explanation_text, activation) -> cosine similarity.
Scorer = Callable[[str, np.ndarray], float]


@dataclass
class ClaimAudit:
    claim_index: int
    claim_text: str
    base_cosine: float
    delete_cosine: float
    #: ``base - delete``. Positive means the claim carried reconstruction weight.
    delete_effect: float
    paraphrase_cosines: list[float] = field(default_factory=list)
    paraphrase_effects: list[float] = field(default_factory=list)
    resample_cosine: float | None = None
    resample_effect: float | None = None
    isolated_cosine: float | None = None
    null_threshold: float = float("nan")
    reconstruction_dependent: bool = False
    #: Verbaliser-only control: did a noise-driven verbalisation contain this?
    reproduced_under_noise: bool = False
    noise_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationAudit:
    problem_id: str
    window_kind: str
    sample_index: int
    explanation: str
    n_claims: int
    base_cosine: float
    claims: list[ClaimAudit] = field(default_factory=list)
    #: Headline for RQ2.
    dependent_fraction: float = float("nan")
    noise_reproduced_fraction: float = float("nan")
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = f"{self.problem_id}|{self.window_kind}|{self.sample_index}"
        return d


# --- Claim-level text similarity, for the noise control ---

_STOP = frozenset(
    "the a an of to and or in on for with that this is are was were be been it its "
    "as at by from vector activation concept text snippet about which".split()
)


def _tokens(text: str) -> set[str]:
    import re

    return {
        w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP
    }


def claim_similarity(a: str, b: str) -> float:
    """Jaccard over content words.

    Deliberately crude. An embedding model would be more sensitive, but it would
    also be a third model's opinion inserted into the control, and there is no
    VRAM or disk budget for one here. The threshold is reported alongside the
    rate so the reader can judge it.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def reproduced_in(claim: str, texts: Sequence[str], threshold: float = 0.5) -> tuple[bool, float]:
    best = 0.0
    for t in texts:
        for other in split_claims(t, max_claims=12):
            best = max(best, claim_similarity(claim, other.text))
    return best >= threshold, best


# --- The audit ---


def audit_explanation(
    problem_id: str,
    window_kind: str,
    sample_index: int,
    explanation: str,
    activation: np.ndarray,
    scorer: Scorer,
    cfg: FaithfulnessConfig,
    resample_alternatives: Sequence[str] = (),
    noise_explanations: Sequence[str] = (),
) -> ExplanationAudit:
    """Run every arm for one verbalisation."""
    claims = split_claims(explanation, cfg.max_claims_per_explanation)
    base = float(scorer(explanation, activation))
    audit = ExplanationAudit(
        problem_id=problem_id,
        window_kind=window_kind,
        sample_index=sample_index,
        explanation=explanation,
        n_claims=len(claims),
        base_cosine=base,
    )
    if not claims:
        audit.notes["skipped"] = "no claims parsed from the verbalisation"
        return audit

    # Paraphrase null, pooled across all claims of this explanation. Pooling is
    # the point: a per-claim null from three paraphrases would be far too noisy
    # a percentile estimate.
    null_effects: list[float] = []
    per_claim_para: dict[int, list[tuple[float, float]]] = {}
    for c in claims:
        rows = []
        for v in range(cfg.n_paraphrases):
            text = paraphrase_explanation(explanation, c, v)
            cos = float(scorer(text, activation))
            eff = base - cos
            rows.append((cos, eff))
            null_effects.append(abs(eff))
        per_claim_para[c.index] = rows

    threshold = (
        float(np.percentile(null_effects, cfg.null_percentile)) if null_effects else float("nan")
    )

    alt_claims: list[list[Claim]] = [split_claims(a, cfg.max_claims_per_explanation)
                                     for a in resample_alternatives]

    for c in claims:
        del_cos = float(scorer(delete_claim(explanation, c), activation))
        iso_cos = float(scorer(keep_only(explanation, c), activation))

        resample_cos = None
        for alt_list, alt_text in zip(alt_claims, resample_alternatives):
            # Pair by position: the AV emits claims in a stable order, so claim
            # j of another sample is the natural alternative to claim j here.
            match = next((a for a in alt_list if a.index == c.index), None)
            if match is not None:
                resample_cos = float(
                    scorer(substitute_claim(explanation, c, match.text), activation)
                )
                break

        para = per_claim_para[c.index]
        effect = base - del_cos
        noise_hit, noise_sim = reproduced_in(c.text, noise_explanations)

        audit.claims.append(
            ClaimAudit(
                claim_index=c.index,
                claim_text=c.text,
                base_cosine=base,
                delete_cosine=del_cos,
                delete_effect=effect,
                paraphrase_cosines=[p[0] for p in para],
                paraphrase_effects=[p[1] for p in para],
                resample_cosine=resample_cos,
                resample_effect=None if resample_cos is None else base - resample_cos,
                isolated_cosine=iso_cos,
                null_threshold=threshold,
                reconstruction_dependent=bool(np.isfinite(threshold) and effect > threshold),
                reproduced_under_noise=noise_hit,
                noise_similarity=noise_sim,
            )
        )

    audit.dependent_fraction = float(
        np.mean([c.reconstruction_dependent for c in audit.claims])
    )
    audit.noise_reproduced_fraction = float(
        np.mean([c.reproduced_under_noise for c in audit.claims])
    )
    audit.notes["null_threshold"] = threshold
    audit.notes["n_null_samples"] = len(null_effects)
    audit.notes["n_noise_explanations"] = len(noise_explanations)
    return audit


def summarise(audits: Sequence[ExplanationAudit]) -> dict[str, Any]:
    """Aggregate for the RQ2 table."""
    claims = [c for a in audits for c in a.claims]
    if not claims:
        return {"n_explanations": len(audits), "n_claims": 0}

    dep = np.array([c.reconstruction_dependent for c in claims], dtype=bool)
    noise = np.array([c.reproduced_under_noise for c in claims], dtype=bool)
    eff = np.array([c.delete_effect for c in claims], dtype=float)
    para = np.array([e for c in claims for e in c.paraphrase_effects], dtype=float)
    resample = np.array(
        [c.resample_effect for c in claims if c.resample_effect is not None], dtype=float
    )

    return {
        "n_explanations": len(audits),
        "n_claims": len(claims),
        "mean_base_cosine": float(np.mean([a.base_cosine for a in audits])),
        "dependent_fraction": float(dep.mean()),
        "noise_reproduced_fraction": float(noise.mean()),
        # The number that decides whether the NLA arm means anything: claims
        # that both carry reconstruction weight and are *not* reproducible from
        # noise.
        "dependent_and_not_noise_fraction": float((dep & ~noise).mean()),
        "mean_delete_effect": float(np.mean(eff)),
        "mean_paraphrase_effect": float(np.mean(np.abs(para))) if para.size else float("nan"),
        "mean_resample_effect": float(np.mean(resample)) if resample.size else float("nan"),
        "delete_vs_paraphrase_ratio": (
            float(np.mean(eff) / np.mean(np.abs(para))) if para.size and np.mean(np.abs(para)) else float("nan")
        ),
    }
