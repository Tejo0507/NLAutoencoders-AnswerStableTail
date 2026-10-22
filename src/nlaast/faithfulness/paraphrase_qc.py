"""Is a paraphrase-null draw a valid perturbation? (D70)

The RQ2 threshold is the 95th percentile of the paraphrase null: how far
reconstruction moves when a claim is reworded without changing what it says.
A draw that does not change the text contributes a zero and narrows the
null. A draw that changes what the claim says is not a null draw at all.
This module classifies each draw by comparing it with the original. The
classes are fixed in D70 and are purely textual: no class certifies that
meaning is preserved, only that the content words are.
"""

from __future__ import annotations

import re
from collections import Counter

from .claims import _PREFIXES, _SYNONYMS

IDENTICAL = "identical"
NORMALISED_IDENTICAL = "normalised_identical"
PREFIX_ONLY = "prefix_only"
LEXICAL_SWAP = "lexical_swap"
REORDER = "reorder"
CONTENT_CHANGED = "content_changed"

#: D70's validity sets.
VALID_N1 = frozenset({PREFIX_ONLY, LEXICAL_SWAP, REORDER})
VALID_N2 = frozenset({PREFIX_ONLY, LEXICAL_SWAP})

_STOP = frozenset("""
a an the and or but of to in on at for by with as is are was were be been being it
its this that these those from into than then so such which who whom whose what
""".split())
_MARKERS = tuple(p.split("{}")[0].strip().rstrip(",").lower() for p in _PREFIXES)
_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def content_words(text: str) -> Counter:
    """Lowercased alphanumeric tokens minus stopwords and discourse markers."""
    low = text.lower()
    for m in _MARKERS:
        low = low.replace(m, " ")
    return Counter(t for t in _TOKEN.findall(low) if t not in _STOP)


def _synonym_pairs() -> set[tuple[frozenset, frozenset]]:
    pairs = set()
    for pat, rep in _SYNONYMS:
        src = re.sub(r"\\b", "", pat).replace("\\", "")
        pairs.add((frozenset(content_words(src).items()),
                   frozenset(content_words(rep).items())))
    return pairs


_PAIRS = _synonym_pairs()


def _is_prefix_only(original: str, variant: str) -> bool:
    o = original.strip()
    for p in _PREFIXES:
        head = p.split("{}")[0]
        if variant.startswith(head):
            body = variant[len(head):]
            if body == o or (o and body == o[0].lower() + o[1:]):
                return True
    return False


def classify(original: str, variant: str) -> str:
    """The D70 class of one paraphrase draw."""
    if variant == original:
        return IDENTICAL
    if _normalise(variant) == _normalise(original):
        return NORMALISED_IDENTICAL
    if _is_prefix_only(original, variant):
        return PREFIX_ONLY
    a, b = content_words(original), content_words(variant)
    if a == b:
        return REORDER
    removed, added = a - b, b - a
    if (frozenset(removed.items()), frozenset(added.items())) in _PAIRS:
        return LEXICAL_SWAP
    return CONTENT_CHANGED


def token_change(original: str, variant: str) -> float:
    """Share of the longer text's tokens not matched in order (1 − LCS ratio).

    A diversity measure only: 0 means no token changed. It says nothing about
    meaning.
    """
    a, b = _TOKEN.findall(original.lower()), _TOKEN.findall(variant.lower())
    if not a and not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))
