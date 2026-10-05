"""Sentence-level segmentation of a reasoning trace, with token alignment.

Liu & Wang segment a chain of thought into sentence-level chunks and parse an
intermediate answer at each boundary. Everything downstream - the convergence
baseline, the Answer-Stable Tail, the activation extraction positions and the
causal spans - is indexed by those boundaries, so the segmentation has to carry
exact token offsets, not just character offsets.

The hard part is not splitting English prose; it is not splitting inside
mathematics. ``$1.50``, ``Mr. Smith``, ``3.14`` and ``\\frac{1}{2}`` all contain
periods that are not sentence ends, and a naive split on ``.`` fragments a MATH
trace into dozens of meaningless chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Sequence

from ..config import ChunkConfig

#: Abbreviations whose trailing period never ends a sentence here.
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "st", "e.g", "i.e", "etc", "vs", "fig",
    "eq", "approx", "no", "cf", "al",
)

_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")

#: Mathematics that a sentence must never be split inside. ``$...$`` is handled
#: separately because these traces use ``$`` for both LaTeX and money.
_MATH_DELIMITED = re.compile(
    r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\\begin\{.*?\}.*?\\end\{.*?\}",
    re.DOTALL,
)
_DOLLAR_PAIR = re.compile(r"\$([^$\n]{1,200})\$")


def _math_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges occupied by mathematics.

    The naive rule - "an odd number of ``$`` so far means we are inside maths" -
    is wrong for this corpus and was observed merging chunks on GSM8K. These
    traces write money as ``$2.`` and ``$30``, so a single unpaired ``$`` is
    normal and every later sentence boundary would be suppressed.

    A ``$...$`` pair is therefore only treated as maths when its contents
    actually look like maths: a backslash command, or an operator together with
    a letter. ``$36`` and ``$2 per ticket`` are money; ``$x + 1$`` and
    ``$\\frac{1}{2}$`` are not.
    """
    spans = [(m.start(), m.end()) for m in _MATH_DELIMITED.finditer(text)]
    for m in _DOLLAR_PAIR.finditer(text):
        inner = m.group(1)
        looks_mathematical = "\\" in inner or (
            any(op in inner for op in "+-*/^=<>") and any(c.isalpha() for c in inner)
        )
        if looks_mathematical:
            spans.append((m.start(), m.end()))
    return spans


def _in_spans(idx: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= idx < e for s, e in spans)


def _is_boundary(text: str, idx: int, math_spans: list[tuple[int, int]] | None = None) -> bool:
    """Is the punctuation at ``idx`` a genuine sentence end?"""
    ch = text[idx]
    if ch in "!?":
        return True
    # Decimal point / version number: digit on both sides.
    if idx > 0 and idx + 1 < len(text) and text[idx - 1].isdigit() and text[idx + 1].isdigit():
        return False
    # Abbreviation.
    prefix = text[max(0, idx - 12) : idx].lower()
    word = re.split(r"[^a-z.]", prefix)[-1] if prefix else ""
    if word in _ABBREV or (word.endswith(".") and len(word) <= 4):
        return False
    # Single capital letter initial, "A. Smith".
    if idx >= 1 and text[idx - 1].isupper() and (idx < 2 or not text[idx - 2].isalpha()):
        return False
    if _in_spans(idx, math_spans if math_spans is not None else _math_spans(text)):
        return False
    return True


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int
    #: Index of the last *generated* token belonging to this chunk, relative to
    #: the start of the generated continuation (not the prompt).
    token_end: int
    #: Number of generated tokens in the prefix ending at this chunk.
    prefix_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str, cfg: ChunkConfig, mode: str | None = None) -> list[tuple[int, int]]:
    """Character spans of the chunks. ``mode`` overrides the primary splitter.

    ``mode='newline'`` is the alternative segmentation used by falsification
    tests F4/F5 to check that the Answer-Stable Tail is not an artefact of how
    the trace was cut up.
    """
    if not text.strip():
        return []
    mode = mode or "sentence"

    if mode == "newline":
        spans: list[tuple[int, int]] = []
        pos = 0
        for part in text.split("\n"):
            end = pos + len(part)
            if part.strip():
                spans.append((pos, end))
            pos = end + 1
        raw = spans
    else:
        raw = []
        start = 0
        spans = _math_spans(text)
        for m in _SENTENCE_END.finditer(text):
            idx = m.start(1)
            if not _is_boundary(text, idx, spans):
                continue
            end = m.end(1)
            if text[start:end].strip():
                raw.append((start, end))
            start = m.end()
        # A blank line is a paragraph break and a chunk boundary even when the
        # preceding line has no terminal punctuation - display maths such as
        # ``\[ 36 - 30 = 6 \]`` ends many reasoning steps.
        extra = [m.end() for m in re.finditer(r"\n[ \t]*\n", text)]
        if extra:
            cuts = sorted({c for c in extra if 0 < c < len(text)}
                          | {e for _, e in raw} | {len(text)})
            raw, prev = [], 0
            for c in cuts:
                if text[prev:c].strip():
                    raw.append((prev, c))
                prev = c
        elif start < len(text) and text[start:].strip():
            raw.append((start, len(text)))

    # Merge fragments below the minimum length. A two-character chunk has no
    # parseable intermediate answer and would only add noise to the boundary
    # index. Merging is backwards where possible and forwards for a short
    # leading fragment, which has no predecessor to absorb it.
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and len(text[s:e].strip()) < cfg.min_chunk_chars:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    while len(merged) > 1 and len(text[merged[0][0] : merged[0][1]].strip()) < cfg.min_chunk_chars:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    # Cap the count by fusing from the end, so early boundaries - where
    # stabilisation usually happens - keep their resolution.
    while len(merged) > cfg.max_chunks:
        merged[-2] = (merged[-2][0], merged[-1][1])
        merged.pop()
    return merged


def chunk_trace(
    text: str,
    cfg: ChunkConfig,
    token_offsets: Sequence[tuple[int, int]] | None = None,
    mode: str | None = None,
) -> list[Chunk]:
    """Segment ``text`` and attach token indices.

    ``token_offsets`` is the tokeniser's ``offset_mapping`` for the generated
    continuation: ``[(char_start, char_end), ...]`` per generated token. When it
    is absent the token fields fall back to character counts, which is only
    acceptable for tests - the activation stage always supplies real offsets,
    because an off-by-one there would read the residual stream at the wrong
    position and silently corrupt every NLA result.
    """
    spans = split_sentences(text, cfg, mode=mode)
    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(spans):
        if token_offsets is None:
            tok_end = e
        else:
            tok_end = _last_token_at_or_before(token_offsets, e)
        chunks.append(
            Chunk(
                index=i,
                text=text[s:e],
                char_start=s,
                char_end=e,
                token_end=tok_end,
                prefix_tokens=tok_end + 1,
            )
        )
    return chunks


def _last_token_at_or_before(offsets: Sequence[tuple[int, int]], char_end: int) -> int:
    """Index of the last token whose span starts before ``char_end``.

    Tokens can straddle a sentence boundary (" .\\nNext" is often one token), so
    the rule is "starts before the boundary" rather than "ends at or before it".
    That keeps the boundary token inside the prefix, which is what we want: the
    activation read at this position has seen the whole sentence.
    """
    lo, hi, best = 0, len(offsets) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid][0] < char_end:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def prefix_text(chunks: Sequence[Chunk], upto: int, full_text: str) -> str:
    """Text of chunks ``0..upto`` inclusive, sliced from the original string.

    Sliced rather than joined so whitespace between chunks is preserved exactly;
    re-joining would change the tokenisation of the prefix and therefore the
    activations.
    """
    if upto < 0 or not chunks:
        return ""
    upto = min(upto, len(chunks) - 1)
    return full_text[: chunks[upto].char_end]
