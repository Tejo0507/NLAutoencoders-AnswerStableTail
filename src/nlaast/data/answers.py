"""Answer extraction and equivalence checking.

Everything in this study reduces to one question asked repeatedly: *is the answer
at this point the same answer as the final one?* Tail detection, stopping safety,
semantic-entropy clustering and the causal arm all call ``equivalent``. If this
module is wrong, every downstream number is wrong, so it is kept small, pure and
heavily tested (``tests/test_answers.py``).

Two extractors are provided on purpose. ``PERMISSIVE`` is the primary one;
``STRICT`` exists so falsification test F4 can ask whether the Answer-Stable Tail
is an artefact of the parser rather than of the model.

Windows note: ``math_verify`` implements its call timeout with ``multiprocessing``
and that path raises ``OSError`` on Windows. Every call passes
``timeout_seconds=None`` and is wrapped, with a deterministic fallback below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Mode = Literal["permissive", "strict"]

PERMISSIVE: Mode = "permissive"
STRICT: Mode = "strict"

# --- Extraction ---

#: GSM8K gold answers end with ``#### 18``.
_GSM8K_GOLD = re.compile(r"####\s*(-?[\d,]*\.?\d+)")

#: The phrasing the system prompt asks for. Only the marker is matched; the
#: candidate is taken from the text after the *last* marker. A single regex
#: with a capture group cannot do this: given "the answer is The answer is 6",
#: the first match swallows the second and captures "The answer is 6".
_ANSWER_MARKER = re.compile(
    r"(?:final\s+)?answer\s+(?:is|:)\s*", re.IGNORECASE
)

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

#: A number followed only by a short unit phrase: "180 minutes", "6 dollars",
#: "9 eggs per day". Observed in live output - the model answers the question
#: in words even when asked for a bare value, and scoring "180 minutes"
#: against a gold of "180" as a mismatch would understate accuracy and
#: manufacture spurious answer instability at chunk boundaries.
_NUMBER_WITH_UNITS = re.compile(
    r"^(-?[\d,]*\.?\d+)\s*(?:%|\$)?\s*((?:[A-Za-z][A-Za-z.-]*\s*){1,4})$"
)


def _strip_units(candidate: str) -> str:
    """``"180 minutes"`` -> ``"180"``. Leaves anything else untouched.

    Deliberately conservative: it fires only when the whole candidate is one
    number plus alphabetic words, so ``"2x + 1"``, ``"x = 3"`` and
    ``"\\frac{1}{2}"`` are unaffected.
    """
    m = _NUMBER_WITH_UNITS.match(candidate.strip())
    if not m:
        return candidate
    tail = m.group(2).strip().lower().rstrip(".")
    # A unit does not change the value; a mathematical operator or symbol
    # does. "3 apples" is 3; "3 pi", "2 squared" and "5 x" are not.
    MATHEMATICAL = {"pi", "e", "i", "x", "y", "z", "n", "sqrt", "times",
                    "squared", "cubed", "factorial", "choose", "mod", "log",
                    "ln", "sin", "cos", "tan", "over", "percent"}
    if any(w in MATHEMATICAL for w in tail.split()):
        return candidate
    return m.group(1).replace(",", "")

def _strip(text: str) -> str:
    """Remove presentation wrappers without damaging LaTeX.

    Stripping ``\\`` and ``{}`` indiscriminately turns ``\\frac{1}{2}`` into
    ``frac{1}{2`` - a silent corruption that then fails every equivalence test.
    So: whitespace and ``$`` go, ``\\text{...}``-style wrappers are unwrapped,
    *balanced* enclosing delimiters are peeled, and trailing sentence
    punctuation is dropped. A leading backslash that begins a LaTeX command is
    left alone.
    """
    out = text.strip()
    # Peeling and trailing-punctuation removal interleave: "\(\frac{3}{4}\)."
    # needs the period gone before the wrapper is recognisable, and the
    # wrapper gone before any inner punctuation is. Iterate to a fixed point.
    for _ in range(6):
        before = out
        out = out.strip().strip("$").strip().rstrip(".,:; \t").strip()

        new = re.sub(r"\\(?:text|textbf|mbox|mathrm|rm)\s*\{([^{}]*)\}", r"\1", out)
        if new != out:
            out = new
        out = re.sub(r"^\\(?:left|right)\s*", "", out).strip()

        # LaTeX inline / display wrappers around the whole candidate.
        for open_s, close_s in ((r"\(", r"\)"), (r"\[", r"\]")):
            if out.startswith(open_s) and out.endswith(close_s) and len(out) > 4:
                out = out[len(open_s) : -len(close_s)].strip()

        # Balanced enclosing delimiters, only when they wrap the whole string.
        for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
            while len(out) >= 2 and out[0] == open_c and out[-1] == close_c:
                depth, wraps = 0, True
                for i, ch in enumerate(out):
                    depth += (ch == open_c) - (ch == close_c)
                    if depth == 0 and i < len(out) - 1:
                        wraps = False
                        break
                if not wraps:
                    break
                out = out[1:-1].strip()

        if out == before:
            break

    # Thousands separators only. "(1,2)" is a coordinate pair and must keep its
    # comma, so the grouping pattern has to match exactly.
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", out):
        out = out.replace(",", "")
    return out


def find_boxed(text: str) -> str | None:
    """Last ``\\boxed{...}``, brace-balanced.

    A regex cannot do this: MATH solutions contain nested braces such as
    ``\\boxed{\\frac{1}{2}}``, and a non-greedy match truncates them.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        # ``\boxed 5`` form.
        m = re.match(r"\\(?:boxed|fbox)\s+(\S+)", text[idx:])
        return _strip(m.group(1)) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return _strip(text[i + 1 : j])
    return None  # unbalanced - a truncated generation


def extract_gold(record: dict, dataset: str) -> str | None:
    """Gold answer from a benchmark record."""
    if dataset == "gsm8k":
        m = _GSM8K_GOLD.search(record.get("answer", ""))
        return m.group(1).replace(",", "") if m else None
    if dataset == "math":
        return find_boxed(record.get("solution", ""))
    raise ValueError(f"unknown dataset {dataset!r}")


def extract_answer(text: str, mode: Mode = PERMISSIVE) -> str | None:
    """Pull a candidate answer out of generated text.

    Order matters and encodes a precedence claim: an explicit ``\\boxed`` or an
    "the answer is X" statement is what the model asserts; a bare trailing
    number is only a guess. ``STRICT`` refuses that last guess, which is exactly
    the difference F4 probes.
    """
    if not text:
        return None

    boxed = find_boxed(text)
    if boxed:
        return _strip_units(boxed)

    tail = text[-600:]
    last = None
    for m in _ANSWER_MARKER.finditer(tail):
        last = m
    if last is not None:
        rest = tail[last.end() :]
        # Re-anchor on a later marker, which the model emits when it echoes the
        # forced-answer suffix ("Therefore, the answer is The answer is 6.").
        for _ in range(3):
            nxt = None
            for m in _ANSWER_MARKER.finditer(rest):
                nxt = m
            if nxt is None:
                break
            rest = rest[nxt.end() :]
        cand = _strip(rest.split("\n", 1)[0])
        if cand:
            boxed = find_boxed(cand)
            if boxed:
                return _strip_units(boxed)
            # A short candidate is the answer. A long one is prose that
            # happened to follow the marker, so fall back to the number rule
            # rather than returning a sentence as an "answer".
            if len(cand) <= 40 and (any(c.isdigit() for c in cand) or "\\" in cand):
                return _strip_units(cand)
            if mode == PERMISSIVE:
                nums = _NUMBER.findall(cand)
                if nums:
                    return nums[-1].replace(",", "").rstrip(".")

    if mode == STRICT:
        return None

    # Permissive fallback: last number in the last non-empty line, then the
    # last number anywhere. This is what makes intermediate answers parseable
    # at chunk boundaries, where the model has not yet phrased a conclusion.
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    for line in reversed(lines[-3:]):
        nums = _NUMBER.findall(line)
        if nums:
            return nums[-1].replace(",", "").rstrip(".")
    nums = _NUMBER.findall(tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


# --- Equivalence ---


def _numeric(value: str) -> float | None:
    v = value.replace(",", "").replace("$", "").replace("%", "").strip()
    v = v.rstrip(".")
    try:
        return float(v)
    except ValueError:
        pass
    m = re.fullmatch(r"\\[dt]?frac\s*\{(-?[\d.]+)\}\s*\{(-?[\d.]+)\}", v)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den else None
        except ValueError:
            return None
    m = re.fullmatch(r"(-?[\d.]+)\s*/\s*(-?[\d.]+)", v)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den else None
        except ValueError:
            return None
    return None


def _fallback_equivalent(a: str, b: str) -> bool:
    """Deterministic comparison used when symbolic parsing is unavailable."""
    if a == b:
        return True
    na, nb = _numeric(a), _numeric(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(na), abs(nb))
    norm = lambda s: re.sub(r"[\s$\\{}(),]", "", s).lower().rstrip(".")
    return norm(a) == norm(b)


@lru_cache(maxsize=65536)
def _symbolic_equivalent(a: str, b: str) -> bool | None:
    """``math_verify`` comparison. ``None`` means it could not decide.

    Returning ``None`` rather than ``False`` on failure matters: a parse failure
    is not evidence of inequality, and treating it as such would systematically
    inflate the rate at which answers look unstable.
    """
    try:
        from math_verify import parse, verify
    except Exception:
        return None
    try:
        ga = parse(f"${a}$", parsing_timeout=None)
        gb = parse(f"${b}$", parsing_timeout=None)
        if not ga or not gb:
            return None
        return bool(verify(ga, gb, timeout_seconds=None))
    except TypeError:
        # Older math_verify without the timeout kwargs.
        try:
            from math_verify import parse, verify

            return bool(verify(parse(f"${a}$"), parse(f"${b}$")))
        except Exception:
            return None
    except Exception:
        return None


def equivalent(a: str | None, b: str | None, use_symbolic: bool = True) -> bool:
    """Are two extracted answers the same answer?

    ``None`` is never equivalent to anything, including another ``None``: an
    unparseable intermediate answer is not evidence that the answer has settled.
    """
    if a is None or b is None:
        return False
    a, b = _strip(str(a)), _strip(str(b))
    if not a or not b:
        return False
    if a == b:
        return True
    if _fallback_equivalent(a, b):
        return True
    if use_symbolic:
        sym = _symbolic_equivalent(a, b)
        if sym is not None:
            return sym
    return False


@dataclass(frozen=True)
class AnswerCluster:
    """A meaning class for semantic entropy (see baselines/semantic_entropy.py)."""

    representative: str
    members: tuple[int, ...]


def cluster_answers(answers: list[str | None]) -> list[AnswerCluster]:
    """Partition sampled answers into symbolic-equivalence classes.

    This is the adaptation of Farquhar et al.'s bidirectional-entailment
    clustering to mathematical outputs, where the right equivalence relation is
    symbolic equality rather than NLI entailment. Unparseable samples each form
    their own singleton cluster, which is the conservative choice: it raises
    measured entropy rather than hiding the failure.
    """
    clusters: list[list[int]] = []
    reps: list[str | None] = []
    for i, ans in enumerate(answers):
        placed = False
        if ans is not None:
            for ci, rep in enumerate(reps):
                if rep is not None and equivalent(ans, rep):
                    clusters[ci].append(i)
                    placed = True
                    break
        if not placed:
            clusters.append([i])
            reps.append(ans)
    return [
        AnswerCluster(representative=(r if r is not None else f"<unparsed:{c[0]}>"),
                      members=tuple(c))
        for r, c in zip(reps, clusters)
    ]
