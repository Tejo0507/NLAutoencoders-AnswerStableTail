"""Objective checks of what a verbalisation says about its own context (D71).

The verbaliser's explanations follow one template. 99.9% name the "Final
token" at the activation's position in quotes. Nearly all quote text they say
precedes it. 97.6% say what they "expect" to come next. Each of those three
statements has a ground truth in the stored trace, because the activation is
read at a known token (``chunk.token_end``, mapped to text by the trace's
``offsets``):

| claim | checked against | what a match shows |
|---|---|---|
| final token | the text ending at the activation's token | the claim names the position's content correctly |
| quoted context | the trace text up to that token | the quoted text exists, not that the claim about it is true |
| expected continuation | the text the model actually wrote next | the prediction agrees with the model's behaviour |

A match is a string comparison, not a judgement, so these checks need no
annotator. They cover only these three kinds of statement. What a claim says
the text *means* ("signals a concluding summation"), and anything about
mechanism, is out of their reach. ``docs/CLAIM_VALIDATION_PROTOCOL.md``
covers those.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_QUOTE = re.compile(r"[\"“]([^\"”]{1,300})[\"”]")
_FINAL = re.compile(r"[Ff]inal token\s*[\"“]([^\"”]{1,80})[\"”]")
_EXPECT = re.compile(r"(?:expect\w*|continuation|predict\w*|anticipat\w*|followed by)", re.I)
_WORD = re.compile(r"[a-z0-9]+")
#: Words a claim uses to assert mechanism or internal process, not content.
_MECHANISM = re.compile(
    r"\b(the model (?:is|was) (?:verifying|checking|computing|reasoning|planning|deciding)"
    r"|internally|mechanism|circuit|the model (?:believes|knows|intends))\b", re.I)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class ParsedClaims:
    final_token: str | None
    context_quotes: list[str]
    expected: list[str]
    mechanistic: bool


def parse(explanation: str) -> ParsedClaims:
    """Split an explanation's quotes into the three checkable kinds.

    The final-token quote is the one after "Final token". A quote is an
    *expectation* if an expectation word occurs earlier in the same sentence;
    every other quote is *context*.
    """
    m = _FINAL.search(explanation)
    final = m.group(1) if m else None
    ctx, exp = [], []
    for q in _QUOTE.finditer(explanation):
        if m and q.start() == m.start(1) - 1:
            continue
        sent_start = max(explanation.rfind(".", 0, q.start()),
                         explanation.rfind("\n", 0, q.start())) + 1
        if _EXPECT.search(explanation[sent_start:q.start()]):
            exp.append(q.group(1))
        else:
            ctx.append(q.group(1))
    return ParsedClaims(final, ctx, exp, bool(_MECHANISM.search(explanation)))


def final_token_match(quoted: str, context_up_to_token: str) -> dict[str, bool]:
    """Three nested levels of agreement between a quoted final token and the
    text that actually ends at the activation's position.

    * ``exact_suffix``: the context ends with the quote, after whitespace
      normalisation and case folding;
    * ``last_word``: the quote's last word is the context's last word;
    * ``last_char``: the last non-space character is the same.
    """
    q = " ".join(quoted.split()).lower()
    c = " ".join(context_up_to_token.split()).lower()
    qw, cw = words(quoted), words(context_up_to_token)
    qs, cs = quoted.rstrip(), context_up_to_token.rstrip()
    return {
        "exact_suffix": bool(q) and c.endswith(q),
        "last_word": bool(qw) and bool(cw) and qw[-1] == cw[-1],
        "last_char": bool(qs) and bool(cs) and qs[-1] == cs[-1],
    }


def longest_common_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous word run shared by ``a`` and ``b``."""
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            v = prev[j] + 1 if x == y else 0
            cur.append(v)
            best = max(best, v)
        prev = cur
    return best


def quote_support(quote: str, context: str, min_words: int = 4) -> float | None:
    """Share of a quote's words found as one contiguous run in the context.

    ``None`` for quotes shorter than ``min_words``: a two-word quote is found
    in most mathematical text by chance.
    """
    q = words(quote)
    if len(q) < min_words:
        return None
    return longest_common_run(q, words(context)) / len(q)


def expectation_hit(expected: list[str], continuation: str) -> dict[str, bool] | None:
    """Does any expected continuation agree with what the model wrote next?

    * ``first_word``: some expectation's first word is the continuation's
      first word;
    * ``any_word_in_first_10``: some expectation's first word occurs among the
      continuation's first ten words.
    """
    firsts = [w[0] for w in (words(e) for e in expected) if w]
    nxt = words(continuation)
    if not firsts or not nxt:
        return None
    return {"first_word": nxt[0] in firsts,
            "any_word_in_first_10": bool(set(firsts) & set(nxt[:10]))}


def cohens_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items (nominal labels)."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal-length, non-empty label lists")
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _agreement(a: list, b: list) -> tuple[float, float]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return po, pe


def kappa_summary(a: list, b: list, clusters: list, reps: int = 2000, seed: int = 20261019,
                  alpha: float = 0.05) -> dict:
    """Cohen's kappa with a cluster bootstrap interval, as the D71 protocol reports it.

    Unlike :func:`cohens_kappa`, a kappa whose chance agreement is 1 (both
    raters used one label throughout) is reported as **undefined**, not 1.0.
    Agreement on a constant label is no evidence of reliability. Replicates
    in which kappa is undefined are dropped and counted. ``clusters`` (one per
    label, e.g. the problem) are what the bootstrap resamples, because claims
    from one explanation are not independent.
    """
    import numpy as np

    if not (len(a) == len(b) == len(clusters)) or not a:
        raise ValueError("kappa needs equal-length, non-empty labels and clusters")
    po, pe = _agreement(a, b)
    point = None if pe >= 1 else (po - pe) / (1 - pe)
    ids = sorted(set(clusters), key=str)
    members = {c: [i for i, x in enumerate(clusters) if x == c] for c in ids}
    rng = np.random.default_rng(seed)
    vals, undefined = [], 0
    for _ in range(reps):
        idx = [i for c in rng.choice(len(ids), size=len(ids), replace=True)
               for i in members[ids[int(c)]]]
        ra, rb = [a[i] for i in idx], [b[i] for i in idx]
        q, e = _agreement(ra, rb)
        if e >= 1:
            undefined += 1
            continue
        vals.append((q - e) / (1 - e))
    ci = ([float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
          if point is not None and vals else None)
    return {"kappa": point, "ci95": ci, "observed_agreement": po, "chance_agreement": pe,
            "n_labels": len(a), "n_clusters": len(ids), "n_replicates_undefined": undefined,
            "defined": point is not None}
