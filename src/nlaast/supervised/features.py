"""Feature construction for the two supervised analyses.

One table, built once, used by both scripts. Every column is derived from
either the problem statement or the *surface form* of the generated trace.
Nothing here touches an activation, and nothing here calls the model again.

Three nested blocks, so that each analysis can report what the *next* tier of
information is worth rather than only a single pooled score:

``A_problem``
    Known before a single token is generated: question length, numeric
    density, benchmark, MATH level, gold-answer magnitude. Blocks F1 (length)
    and F2 (difficulty) of the falsification battery live here.

``B_trace``
    ``A`` plus the trace's own surface statistics - chunk counts and lengths,
    arithmetic and LaTeX density, and discourse markers of the kind a reader
    would call "verification language". Requires generation, requires no
    forcing, no resampling, no gold answer.

``C_convergence``
    ``B`` plus the answer-convergence signal (Liu & Wang): the first boundary
    at which ``convergence_window`` consecutive parsed intermediate answers
    agree. This is the cheapest stopping signal in the study and the one the
    primary research question asks the verbalised readout to beat, so it is
    isolated in its own block.

A caveat carried openly: ``gold_abs_log10`` and ``math_level`` are problem
metadata, not things a live stopping rule would have. They are included because
F2 names them as the difficulty stratifiers, and the block structure means
their contribution can be read off separately and discounted.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get

log = get(__name__)

# --- Lexical cues ---
# Counted case-insensitively over the whole trace. These are deliberately
# plain word lists rather than a learned lexicon: the analysis has to be
# explainable, and a regex is explainable.

CUE_PATTERNS: dict[str, str] = {
    # "the answer therefore is" - inference / conclusion markers
    "cue_conclude": r"\b(therefore|thus|hence|so,|in total|altogether|overall)\b",
    # the language of re-checking, which is the behaviour the tail is often
    # *claimed* to consist of
    "cue_verify": r"\b(check|checking|verify|verifying|confirm|confirming|"
                  r"double[- ]check|make sure|re-?calculat\w*|re-?comput\w*|"
                  r"wait|actually|let me see|let's see)\b",
    # explicit answer announcements
    "cue_answer": r"\b(the answer is|final answer|the final answer)\b",
    # restatement of the question, a cheap proxy for padding
    "cue_restate": r"\b(we (are|were) (asked|told)|the (question|problem) (asks|states))\b",
}

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[A-Za-z']+")
_MATH_SPAN_RE = re.compile(r"\$[^$]{1,400}\$|\\\([^)]{1,400}?\\\)|\\\[[^]]{1,400}?\\\]")
_OPERATOR_RE = re.compile(r"[+\-*/^]|\\times|\\div|\\cdot|\\frac")

FEATURE_BLOCKS: dict[str, list[str]] = {
    "A_problem": [
        "q_chars",
        "q_words",
        "q_numbers",
        "q_number_density",
        "q_math_char_fraction",
        "prompt_tokens",
        "math_level",
        "gold_abs_log10",
        "gold_is_integer",
        "dataset",
        "subject",
    ],
    "B_trace": [
        "n_chunks",
        "n_tokens",
        "tokens_per_chunk_mean",
        "tokens_per_chunk_sd",
        "chunk_chars_mean",
        "chunk_chars_max",
        "short_chunk_fraction",
        "truncated",
        "equals_per_chunk",
        "operators_per_chunk",
        "digit_fraction",
        "math_char_fraction",
        "type_token_ratio",
        "cue_conclude_per_chunk",
        "cue_verify_per_chunk",
        "cue_answer_per_chunk",
        "cue_restate_per_chunk",
        "first_answer_mention",
        "answer_mention_count",
        "has_answer_cue",
    ],
    "C_convergence": [
        "convergence_position",
        "has_convergence",
        "n_parsed_boundaries",
        "parsed_boundary_fraction",
    ],
}

#: Cumulative blocks, in the order the nested comparison reports them.
BLOCKS: dict[str, list[str]] = {}
_acc: list[str] = []
for _name, _cols in FEATURE_BLOCKS.items():
    _acc = _acc + _cols
    BLOCKS[_name] = list(_acc)
del _acc, _name, _cols

#: A short, **pre-declared** subset for the inference models (OLS and Logit).
#:
#: The predictive models use every feature; the inference models do not, and
#: the reason is arithmetic rather than taste. Thirty-five features plus
#: imputation indicators against roughly a hundred training rows gives a design
#: matrix that is rank-deficient or close to it, and a coefficient table from
#: such a fit is not interpretable however confidently it is printed. This list
#: holds one representative per mechanism the falsification battery names -
#: length (F1), difficulty (F2), the convergence baseline, answer timing, and
#: the verification-language claim - and was fixed before any model was fitted,
#: not pruned to whatever turned out significant.
INFERENCE_FEATURES: list[str] = [
    "n_tokens",                # F1: trace length
    "n_chunks",
    "math_level",              # F2: difficulty
    "gold_abs_log10",          # F2: answer magnitude
    "dataset",
    "convergence_position",    # the cheap stopping baseline
    "first_answer_mention",    # when the answer value first surfaces
    "cue_verify_per_chunk",    # the "it is checking its work" claim
    "equals_per_chunk",        # arithmetic density
    "truncated",
]

CATEGORICAL = {"dataset", "subject"}
BOOLEAN = {"truncated", "gold_is_integer", "has_answer_cue", "has_convergence"}


def numeric_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c not in CATEGORICAL]


def categorical_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in CATEGORICAL]


# --- Per-trace feature extraction ---


def _math_char_fraction(text: str) -> float:
    """Fraction of characters sitting inside a LaTeX span."""
    if not text:
        return 0.0
    inside = sum(m.end() - m.start() for m in _MATH_SPAN_RE.finditer(text))
    return min(1.0, inside / len(text))


def _gold_magnitude(gold: str | None) -> tuple[float, int]:
    """``(log10(|gold|), is_integer)``, both NaN/0 when gold is not numeric.

    A magnitude feature rather than the value itself: F2 asks whether tail
    length tracks answer *size*, and 6277 and 6278 are the same difficulty.
    """
    if gold is None:
        return float("nan"), 0
    cleaned = str(gold).replace(",", "").replace("$", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return float("nan"), 0
    is_int = int(float(value).is_integer())
    if value == 0:
        return 0.0, is_int
    return math.log10(abs(value)), is_int


def _cue_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        name: len(re.findall(pattern, lowered))
        for name, pattern in CUE_PATTERNS.items()
    }


def _answer_mentions(chunks: Sequence[dict[str, Any]],
                     final_answer: str | None) -> tuple[float, int]:
    """Where the final answer's *value* first surfaces, and how often it recurs.

    Returns ``(relative chunk index of first mention, number of chunks
    mentioning it)``; the position is ``nan`` when the value never appears
    verbatim in the trace body.

    This is the cheapest possible "when did the answer appear" signal: a
    substring search over text the pipeline has already generated, with no
    forced continuation, no resampling and no gold label. The explicit
    announcement cue is *not* usable for this - the system prompt instructs the
    model to put 'The answer is X' on the last line, so that cue sits in the
    final chunk by construction and carries no information. The value's first
    appearance is not constrained that way.
    """
    n = len(chunks)
    if n == 0 or not final_answer:
        return float("nan"), 0
    needle = str(final_answer).strip().replace("$", "").replace(",", "")
    if not needle:
        return float("nan"), 0

    hits = [i for i, chunk in enumerate(chunks)
            if needle in str(chunk.get("text", "")).replace("$", "").replace(",", "")]
    if not hits:
        return float("nan"), 0
    position = hits[0] / (n - 1) if n > 1 else 0.0
    return position, len(hits)


def features_for_trace(trace: dict[str, Any], ast_row: dict[str, Any],
                       problem: dict[str, Any]) -> dict[str, Any]:
    """One row of the feature table, plus its targets and bookkeeping."""
    text = str(trace.get("trace") or "")
    chunks = list(trace.get("chunks") or [])
    boundaries = list(trace.get("boundaries") or [])
    n_chunks = int(trace.get("n_chunks") or len(chunks))
    question = str(trace.get("question") or problem.get("question") or "")

    chunk_chars = [len(str(c.get("text", ""))) for c in chunks] or [0]
    # Per-chunk token counts come from the recorded cumulative prefix lengths,
    # so they are the model's own tokenisation rather than a re-tokenisation.
    prefix = [int(c.get("prefix_tokens") or 0) for c in chunks]
    per_chunk_tokens = np.diff([0, *prefix]) if prefix else np.array([0])
    per_chunk_tokens = per_chunk_tokens[per_chunk_tokens >= 0]
    if per_chunk_tokens.size == 0:
        per_chunk_tokens = np.array([0])

    cues = _cue_counts(text)
    words = _WORD_RE.findall(text.lower())
    gold_log, gold_int = _gold_magnitude(trace.get("gold") or problem.get("gold"))

    parsed = [b.get("parsed_answer") for b in boundaries]
    n_parsed = sum(1 for a in parsed if a is not None)
    convergence_start = ast_row.get("convergence_start")
    first_mention, mention_count = _answer_mentions(chunks, trace.get("final_answer"))

    row: dict[str, Any] = {
        # --- identity and bookkeeping -----------------------------------
        "id": trace["id"],
        "dataset": str(trace.get("dataset") or problem.get("dataset") or "unknown"),
        "subject": str(problem.get("subject") or trace.get("dataset") or "unknown"),
        "split": str(problem.get("split") or trace.get("split") or "eval"),
        "ast_status": str(ast_row.get("status") or "missing"),

        # --- targets ------------------------------------------------------
        "tail_fraction": float(ast_row.get("tail_fraction") or 0.0),
        "final_correct": int(bool(trace.get("final_correct"))),

        # --- block A: known before generation -----------------------------
        "q_chars": len(question),
        "q_words": len(_WORD_RE.findall(question)),
        "q_numbers": len(_NUMBER_RE.findall(question)),
        "q_number_density": len(_NUMBER_RE.findall(question)) / max(1, len(question.split())),
        "q_math_char_fraction": _math_char_fraction(question),
        "prompt_tokens": int(trace.get("prompt_tokens") or 0),
        # MATH ships a 'Level N' string; GSM8K has no level, left missing and
        # imputed inside the pipeline rather than silently coded as zero.
        "math_level": _parse_level(problem.get("level") or trace.get("level")),
        "gold_abs_log10": gold_log,
        "gold_is_integer": gold_int,

        # --- block B: the trace's surface form ----------------------------
        "n_chunks": n_chunks,
        "n_tokens": int(trace.get("n_tokens") or 0),
        "tokens_per_chunk_mean": float(np.mean(per_chunk_tokens)),
        "tokens_per_chunk_sd": float(np.std(per_chunk_tokens)),
        "chunk_chars_mean": float(np.mean(chunk_chars)),
        "chunk_chars_max": float(np.max(chunk_chars)),
        "short_chunk_fraction": float(np.mean([c < 40 for c in chunk_chars])),
        "truncated": int(bool(trace.get("truncated"))),
        "equals_per_chunk": text.count("=") / max(1, n_chunks),
        "operators_per_chunk": len(_OPERATOR_RE.findall(text)) / max(1, n_chunks),
        "digit_fraction": sum(ch.isdigit() for ch in text) / max(1, len(text)),
        "math_char_fraction": _math_char_fraction(text),
        "type_token_ratio": (len(set(words)) / len(words)) if words else 0.0,
        "first_answer_mention": first_mention,
        "answer_mention_count": mention_count,
        "has_answer_cue": int(cues["cue_answer"] > 0),

        # --- block C: the answer-convergence baseline ---------------------
        "convergence_position": (
            convergence_start / max(1, n_chunks - 1)
            if convergence_start is not None and n_chunks > 1
            else float("nan")
        ),
        "has_convergence": int(convergence_start is not None),
        "n_parsed_boundaries": n_parsed,
        "parsed_boundary_fraction": n_parsed / max(1, len(boundaries)),
    }
    for name, count in cues.items():
        row[f"{name}_per_chunk"] = count / max(1, n_chunks)
    return row


def _parse_level(level: Any) -> float:
    """``'Level 3'`` -> ``3.0``; anything unrecognised -> ``nan``."""
    if level is None:
        return float("nan")
    match = re.search(r"\d", str(level))
    return float(match.group()) if match else float("nan")


# --- Table assembly ---


def build_feature_table(traces: Sequence[dict[str, Any]],
                        ast_rows: Sequence[dict[str, Any]],
                        problems: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Join traces, AST results and problem metadata into one row per problem.

    Traces that errored, or that have no AST record, are dropped here and the
    count is logged - they carry no target.
    """
    ast_by_id = {r["id"]: r for r in ast_rows}
    prob_by_id = {p["id"]: p for p in problems}

    rows, skipped = [], 0
    for trace in traces:
        tid = trace.get("id")
        if not trace.get("ok") or tid not in ast_by_id:
            skipped += 1
            continue
        rows.append(features_for_trace(trace, ast_by_id[tid], prob_by_id.get(tid, {})))

    log.info("feature table: %d rows, %d traces skipped (failed or no AST record)",
             len(rows), skipped)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    missing = [c for cols in FEATURE_BLOCKS.values() for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature table is missing declared columns: {missing}")
    return df
