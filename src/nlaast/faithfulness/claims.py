"""Splitting a verbalisation into atomic claims, and perturbing them.

The audit template is Lanham et al.'s - delete, resample, paraphrase - applied
to verbalised activation descriptions rather than to chain-of-thought text.
Dingeto's observation is the reason it is needed: scoring an explanation by
reconstruction alone is structurally insensitive to individual false claims,
because flipping one claim often leaves the reconstruction unchanged.

Paraphrasing is rule-based rather than model-generated. Using a language model
to paraphrase would introduce that model's own semantics into the null
distribution, and the null is what the deletion threshold is measured against -
it has to be a *surface* perturbation that preserves meaning, not a semantic
rewrite. The transformations here are lexical and structural only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

#: The AV is prompted for "2-3 text snippets", and emits them as sentences,
#: bullets or quoted fragments. All three forms are handled.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")


@dataclass
class Claim:
    index: int
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_claims(explanation: str, max_claims: int = 8) -> list[Claim]:
    """Atomic claims from a verbalisation.

    Bullets take precedence over sentences: when the verbaliser emits a list,
    each item is one claim even if an item spans two sentences. Splitting a
    bullet internally would create claims whose deletion is not independently
    meaningful.
    """
    text = (explanation or "").strip()
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    bullets = list(_BULLET.finditer(text))
    if len(bullets) >= 2:
        for i, m in enumerate(bullets):
            start = m.end()
            end = bullets[i + 1].start() if i + 1 < len(bullets) else len(text)
            spans.append((start, end))
    else:
        pos = 0
        for m in _SENTENCE.finditer(text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(text)))

    claims: list[Claim] = []
    for s, e in spans:
        frag = text[s:e].strip()
        if len(frag) < 8:
            continue
        # Re-anchor after stripping so deletion removes exactly the claim.
        start = s + (len(text[s:e]) - len(text[s:e].lstrip()))
        claims.append(Claim(len(claims), frag, start, start + len(frag)))
        if len(claims) >= max_claims:
            break
    return claims


def delete_claim(explanation: str, claim: Claim) -> str:
    """Explanation with one claim removed."""
    out = explanation[: claim.char_start] + explanation[claim.char_end :]
    return re.sub(r"\s{2,}", " ", out).strip()


def keep_only(explanation: str, claim: Claim) -> str:
    """Just the one claim - the complement of deletion."""
    return claim.text.strip()


# --- Paraphrase: surface-only, meaning-preserving ---

_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"\brelates to\b", "concerns"),
    (r"\bconcerns\b", "relates to"),
    (r"\bdescribes\b", "characterises"),
    (r"\bindicates\b", "suggests"),
    (r"\bsuggests\b", "indicates"),
    (r"\brefers to\b", "points to"),
    (r"\binvolves\b", "includes"),
    (r"\bdiscussion of\b", "discourse about"),
    (r"\bmentions\b", "references"),
    (r"\babout\b", "regarding"),
    (r"\btext\b", "passage"),
    (r"\bcontext\b", "setting"),
    (r"\bthe concept of\b", "the notion of"),
)

_PREFIXES: tuple[str, ...] = (
    "In other words, {}",
    "Put differently, {}",
    "That is to say, {}",
)


def paraphrase_claim(claim_text: str, variant: int) -> str:
    """Deterministic surface paraphrase number ``variant``.

    Deterministic so the null distribution is reproducible from the saved
    config rather than from a stored RNG state.
    """
    text = claim_text.strip()
    if variant == 0:
        out = text
        for pat, rep in _SYNONYMS:
            new = re.sub(pat, rep, out, count=1, flags=re.IGNORECASE)
            if new != out:
                out = new
                break
        return out
    if variant == 1:
        body = text[0].lower() + text[1:] if text else text
        return _PREFIXES[0].format(body)
    if variant == 2:
        # Clause reordering around a comma - structural, not lexical.
        if "," in text:
            head, _, tail = text.partition(",")
            tail = tail.strip().rstrip(".")
            head = head.strip().rstrip(".")
            if tail:
                return f"{tail[0].upper()}{tail[1:]}, {head.lower()}."
        return _PREFIXES[1].format(text[0].lower() + text[1:] if text else text)
    return _PREFIXES[variant % len(_PREFIXES)].format(text)


def paraphrase_explanation(explanation: str, claim: Claim, variant: int) -> str:
    """Explanation with exactly one claim paraphrased."""
    return (
        explanation[: claim.char_start]
        + paraphrase_claim(claim.text, variant)
        + explanation[claim.char_end :]
    )


def substitute_claim(explanation: str, claim: Claim, replacement: str) -> str:
    """Explanation with one claim swapped for arbitrary text.

    Used for the resampling arm, where the replacement is a claim the AV
    produced on a different sample of the same activation.
    """
    return explanation[: claim.char_start] + replacement + explanation[claim.char_end :]
