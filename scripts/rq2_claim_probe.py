"""RQ2's independent-probe control, as proposed in amendment A1 (D56).

§5 registers, for the claim-level audit: "*Independent-probe control.* A probe
trained on held-out activations, never on the verbalised text, scores whether
the claim's content is decodable from the activation. Group-disjoint splits;
leakage is tested explicitly by the falsification battery." It does not say
what a free-text claim's "content" is as a probe target. Amendment A1
(``docs/RQ2_PROBE_AMENDMENT.md``) supplies that: three content attributes whose
ground truth comes from the trace and the problem record, never from the
verbaliser, and a fixed rule that reads which attribute value a claim asserts.

Attributes
    ``answer_stated``  has the trace already stated its final answer at this
                       window? (binary; strict extractor on the prefix)
    ``domain``         which benchmark subject the problem is from (GSM8K,
                       MATH algebra, MATH counting & probability)
    ``final_word``     the last word of the context at the read position,
                       numbers pooled as ``<num>``, top-K words plus ``<other>``

Probes are logistic (``cfg.probe``'s C, max_iter, standardisation), fitted on
layer-20 activations at every chunk boundary of **train-split** problems, with
labels from the text. Claims are scored only on **eval-split** windows, so no
problem contributes to both. Labels never touch the verbalisations.

Phases
    ``describe``         coverage only: how many claims assert each attribute,
                         base rates of the ground truth. Fits nothing.
    ``annotate-sample``  writes the blind annotation sheet for validating the
                         claim->assertion rule (A1 §4).
    ``evaluate``         fits the probes and scores claims. Refuses to run
                         unless the amendment is approved and the annotation
                         agreement has been computed and passed.

    python scripts/rq2_claim_probe.py --config main --phase describe
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _stage import base_parser

from nlaast import config as config_mod
from nlaast import logging_utils
from nlaast.data.answers import STRICT, equivalent, extract_answer
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.seeding import derive
from nlaast.trace.chunking import Chunk, prefix_text

ATTRIBUTES = ("answer_stated", "domain", "final_word")
#: Attributes the evaluation scores (A1 §2). ``final_word`` is described but not
#: scored: the words claims quote as the "final token" almost never end a
#: training boundary, so a probe has no class for them (27 claims at most).
EVALUATED = ("answer_stated", "domain")
#: What a rule-derived assertion of each attribute actually is (A1 §2-§3,
#: 2026-10-17 audit). Written into the evaluation output beside the numbers.
ATTRIBUTE_INTERPRETATION = {
    "answer_stated": ("the lexicon rule's reading of whether the claim says the "
                      "already-written context states the final answer (1) or that "
                      "only an answer statement is predicted next (0); rule-derived, "
                      "not human-validated; claims read as 1 sit at the final "
                      "boundary 85% of the time, so readings are largely positional"),
    "domain": ("the subject cue a claim names, mapped to a benchmark subject: "
               "money/price wording or 'word problem' -> gsm8k (418 of 463 gsm8k "
               "readings are money/price wording alone), algebraic objects -> "
               "math-algebra, probability/counting wording -> "
               "math-counting_and_probability. Not a validated problem-domain "
               "classification of the claim"),
}
DOMAINS = ("gsm8k", "math-algebra", "math-counting_and_probability")
NUM = "<num>"
OTHER = "<other>"
#: Final-word vocabulary size, fixed in A1 before any probe was fitted.
TOP_K_WORDS = 20
AMENDMENT = Path(__file__).resolve().parents[1] / "docs" / "RQ2_PROBE_AMENDMENT.md"


# --- ground truth (from text and metadata only) ----------------------------

def domain_of(problem_id: str) -> str:
    """Subject from the problem id: ``gsm8k-test-…`` / ``math-<subject>-test-…``."""
    head = problem_id.rsplit("-test", 1)[0]
    if head not in DOMAINS:
        raise ValueError(f"unknown domain for {problem_id!r}")
    return head


def normalise_word(text: str | None) -> str | None:
    """Last alphanumeric word, lower-cased; any number becomes ``<num>``."""
    if not text:
        return None
    words = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)*", text)
    if not words:
        return None
    w = words[-1].lower()
    return NUM if w[0].isdigit() else w


def boundary_labels(trace: dict, chunk_index: int) -> dict[str, Any]:
    """Ground truth for one read position. Text and metadata only."""
    chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                         "token_end", "prefix_tokens")})
              for c in trace["chunks"]]
    prefix = prefix_text(chunks, chunk_index, trace["trace"])
    final = trace.get("final_answer")
    stated = None
    if final is not None:
        # Stated at or before this boundary: some prefix up to here already
        # yields the final answer under the strict extractor.
        stated = any(equivalent(extract_answer(prefix_text(chunks, j, trace["trace"]), STRICT),
                                final)
                     for j in range(chunk_index + 1))
    return {"answer_stated": None if stated is None else int(stated),
            "domain": domain_of(trace["problem_id"]),
            "final_word": normalise_word(prefix)}


# --- the claim -> assertion rule (A1 §3) ----------------------------------

#: Opens an *expectation scope*: unquoted text from here to the end of the
#: (quote-aware) sentence describes what the model is predicted to write next.
_EXPECT = (r"(?:expect|anticipat|predict|about to|will (?:state|give|conclude|reach)"
           r"|leads? (?:in)?to|setting up|sets up"
           # prediction idioms found in a survey of quote lead-ins (2026-10-17):
           r"|likely (?:followed|complet|continu|end|clos|to\b)|(?:would|will|to) be followed"
           r"|requir(?:es|ing)\b|next tokens?\b|continuation (?:like|such as))")
#: An answer *statement*: text that states (or would state) the final answer.
_ANSWER_STMT = r"(?:final answer|answer is|the answer:|\\boxed|final result|final value is)"
#: A closing / confirming phrase: text that ends or confirms a solution rather
#: than stating its answer ("This concludes the answer", "The final answer is
#: confirmed", "end of solution"). Expecting one of these is *not* expecting
#: the answer (A1 §3, revision 2026-10-17).
_CLOSING = r"(?:conclu\w*|complet\w*|\bend\b|\bends\b|\bclos\w*|confirm\w*|finali[sz]\w*|verif\w*|</)"
#: Description patterns saying the context has concluded / stated its answer.
_STATED = [
    r"clos\w* (?:the|a|its) conclu",
    r"conclud\w* (?:summary|statement|sentence|clause|line|remark)",
    r"(?:has|having|already) (?:stated|given|reached|delivered) (?:the|its|a) (?:final )?(?:answer|result)",
    r"after (?:stating|giving) the (?:final )?answer",
]
_UNQUOTED_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_FINAL_TOKEN = re.compile(r"final token\s*[\"“']([^\"”']{1,60})[\"”']", re.I)
_DOMAIN_CUES = {
    "math-algebra": r"\balgebra|polynomial|quadratic|inequalit|system of (?:linear )?equations|solve for [a-z]\b",
    "math-counting_and_probability": r"probab|combinator|permutation|\bcounting\b|dice\b|\bcoin|arrangements",
    "gsm8k": r"word problem|\bdollars?\b|\$\d|\bprice|\bcosts?\b|\bshopping|everyday|real-world",
}


def quote_scope_map(text: str) -> tuple[list[bool], list[bool], bool]:
    """Per character: inside a quotation? inside an expectation scope?

    Computed over the **whole explanation**, because the claim splitter cuts
    inside quotations (its sentence regex splits after "time." in
    ``Final token "time. " closes``), so a claim can begin inside, or just
    after, a quotation opened in the previous claim. Straight double quotes
    toggle; the scan never resets at paragraph breaks because quotations do
    cross them (``"item:\\n\\n"``). An expectation scope opens at an unquoted
    ``_EXPECT`` match and closes at the next unquoted sentence end. The third
    value says whether the quotes balance; when they do not, quote state after
    the unmatched quote is unreliable.
    """
    in_q = [False] * len(text)
    q = False
    for i, ch in enumerate(text):
        if ch == '"':
            in_q[i] = True          # the quote mark belongs to the quotation
            q = not q
        else:
            in_q[i] = q
    balanced = not q
    in_s = [False] * len(text)
    ends = [m.start() for m in _UNQUOTED_SENTENCE_END.finditer(text) if not in_q[m.start()]]
    for m in re.finditer(_EXPECT, text, re.I):
        if in_q[m.start()]:
            continue
        stop = next((e for e in ends if e > m.start()), len(text))
        for i in range(m.start(), min(stop + 1, len(text))):
            in_s[i] = True
    return in_q, in_s, balanced


def _runs(text: str, mask: list[bool]) -> list[str]:
    out, cur = [], []
    for ch, m in zip(text, mask):
        if m:
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def answer_stated_evidence(explanation: str, start: int, end: int) -> dict[str, Any]:
    """What the claim ``explanation[start:end]`` says about the answer, and why.

    The claim text is never altered; the explanation supplies only quote and
    expectation state at each character. Four kinds of text in the claim:

    - *description* (unquoted, outside an expectation): ``_STATED`` patterns
      here say the context has concluded -> evidence for ``1``;
    - *context quotation* (quoted, outside an expectation): already-written
      text; an answer statement here -> evidence for ``1``;
    - *predicted quotation* (quoted, inside an expectation): what the model is
      expected to write. An answer statement that is not a closing phrase ->
      evidence for ``0``; a closing/confirming phrase -> **neutral** (it says
      the solution is being wrapped up, not that the answer is still to
      come, and on its own does not show the answer was stated either);
      a phrase that is both -> ambiguous;
    - *expectation text* (unquoted, inside an expectation): an answer cue that
      is not a closing phrase -> evidence for ``0``.

    Value: ``1`` with stated evidence and no not-yet evidence; ``0`` only when
    every predicted alternative that bears on the answer is an answer statement
    and there is no stated evidence; ``None`` (abstain) when both, neither,
    an ambiguous predicted phrase, a mix of answer-statement and closing
    alternatives, or unbalanced quotes in the explanation.
    """
    in_q, in_s, balanced = quote_scope_map(explanation)
    seg = explanation[start:end]
    q, s = in_q[start:end], in_s[start:end]
    desc = "".join(c if not a and not b else " " for c, a, b in zip(seg, q, s))
    ctx_quotes = _runs(seg, [a and not b for a, b in zip(q, s)])
    pred_quotes = _runs(seg, [a and b for a, b in zip(q, s)])
    exp_text = "".join(c if (not a and b) else " " for c, a, b in zip(seg, q, s))

    stated_desc = any(re.search(p, desc, re.I) for p in _STATED)
    stated_quote = any(re.search(_ANSWER_STMT, x, re.I) for x in ctx_quotes)
    not_yet, closing, ambiguous = False, False, False
    for x in pred_quotes:
        has_ans = bool(re.search(_ANSWER_STMT, x, re.I))
        has_close = bool(re.search(_CLOSING, x, re.I))
        if has_ans and has_close:
            ambiguous = True
        elif has_ans:
            not_yet = True
        elif has_close:
            closing = True
    for m in re.finditer(r"final answer|final result|the answer", exp_text, re.I):
        window = exp_text[max(0, m.start() - 40):m.end() + 20]
        if re.search(_CLOSING, window, re.I):
            closing = True
        else:
            not_yet = True
    stated = stated_desc or stated_quote
    if not balanced:
        value, reason = None, "unbalanced_quotes"
    elif ambiguous:
        value, reason = None, "ambiguous_predicted_phrase"
    elif not_yet and closing:
        # Alternatives such as 'expecting "The final answer is" or "This
        # concludes the problem"': the verbaliser predicts either a statement
        # or a wrap-up, so the claim does not say which.
        value, reason = None, "mixed_expected_alternatives"
    elif stated and not not_yet:
        value, reason = 1, "stated_description" if stated_desc else "stated_context_quote"
    elif not_yet and not stated:
        value, reason = 0, "answer_expected"
    elif stated and not_yet:
        value, reason = None, "both_stated_and_expected"
    else:
        value, reason = None, "closing_only" if closing else "no_answer_cue"
    return {"value": value, "reason": reason, "stated_description": stated_desc,
            "stated_context_quote": stated_quote, "expects_answer": not_yet,
            "expects_closing": closing, "quotes_balanced": balanced}


def assertions(claim: str, explanation: str | None = None, start: int | None = None
               ) -> dict[str, Any]:
    """Which attribute values a claim asserts. Absent means no assertion.

    A claim can assert more than one attribute. A claim whose cues for one
    attribute point both ways is ambiguous for that attribute and asserts
    nothing about it - it is counted, not guessed. ``explanation`` and
    ``start`` locate the claim in its explanation so quote and expectation
    state can be read from the source (``answer_stated_evidence``); without
    them the claim is read as a standalone text that starts unquoted.
    """
    out: dict[str, Any] = {}
    text = claim or ""

    # answer_stated
    if explanation is None:
        ev = answer_stated_evidence(text, 0, len(text))
    else:
        ev = answer_stated_evidence(explanation, start, start + len(text))
    if ev["value"] is not None:
        out["answer_stated"] = ev["value"]

    # domain
    hits = [d for d, p in _DOMAIN_CUES.items() if re.search(p, text, re.I)]
    if len(hits) == 1:
        out["domain"] = hits[0]

    # final_word
    m = _FINAL_TOKEN.search(text)
    if m:
        w = normalise_word(m.group(1))
        if w is not None:
            out["final_word"] = w
    return out


# --- data assembly --------------------------------------------------------

def load_inputs(cfg):
    from nlaast.activations.store import ActivationStore

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    store = ActivationStore(cfg.dir / "acts")
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    return traces, store, audits


def boundary_position(trace: dict, chunk_index: int) -> dict[str, float]:
    """Where a boundary sits in its trace - the position comparator's only input."""
    n = len(trace["chunks"])
    return {"relative_position": chunk_index / (n - 1) if n > 1 else 1.0,
            "is_final_boundary": float(chunk_index == n - 1),
            "boundary_index": float(chunk_index),
            "log_prefix_tokens": float(np.log1p(trace["chunks"][chunk_index]["prefix_tokens"]))}


def training_set(traces: dict, store, layer: int
                 ) -> tuple[np.ndarray, list[dict], list[str], list[dict]]:
    """Every chunk-boundary activation of every train-split problem, with labels.

    Rows are grouped by problem and, within a problem, in boundary order -
    the label shuffles rely on it. The fourth list is each row's position
    (``boundary_position``), for the position-only comparator.
    """
    X, labels, groups, positions = [], [], [], []
    for pid, t in sorted(traces.items()):
        if t.get("split") != "train" or not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        cis = list(meta.get("chunk_indices", []))
        for row, ci in sorted(enumerate(cis), key=lambda rc: rc[1]):
            if ci >= len(t["chunks"]):
                continue
            X.append(arr[row])
            labels.append(boundary_labels(t, ci))
            groups.append(pid)
            positions.append(boundary_position(t, ci))
    return (np.vstack(X) if X else np.empty((0, 0))), labels, groups, positions


def positional_predictability(y: Sequence[int], positions: Sequence[dict]) -> dict:
    """How much of a binary label boundary position alone explains (no model).

    The rank AUC of relative position for the label, and the label's rate at
    final versus non-final boundaries. Reported for ``answer_stated``
    (Decision 3, 2026-10-17) because it is close to "this is the final
    boundary" on this corpus.
    """
    from scipy.stats import mannwhitneyu

    y = np.asarray(y, dtype=int)
    rel = np.array([p["relative_position"] for p in positions])
    fin = np.array([p["is_final_boundary"] for p in positions]) == 1
    pos, neg = rel[y == 1], rel[y == 0]
    auc = (float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))
           if len(pos) and len(neg) else None)
    return {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else None,
            "auc_relative_position": auc,
            "rate_at_final_boundary": float(y[fin].mean()) if fin.any() else None,
            "rate_at_other_boundaries": float(y[~fin].mean()) if (~fin).any() else None,
            "note": "label statistics only; no model fitted"}


def word_vocabulary(labels: Sequence[dict], k: int = TOP_K_WORDS) -> list[str]:
    """Top-k final words in the training labels; ties broken alphabetically."""
    c = Counter(l["final_word"] for l in labels if l["final_word"] is not None)
    return [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def encode(attr: str, value: Any, vocab: Sequence[str]) -> Any:
    if attr == "final_word":
        return value if value in vocab else OTHER
    return value


def eval_claims(audits: list[dict], traces: dict) -> list[dict]:
    """One row per eval-split claim, with what it asserts and the ground truth."""
    rows = []
    for a in audits:
        t = traces.get(a["problem_id"])
        if t is None or t.get("split") != "eval":
            continue
        truth = boundary_labels(t, int(a["chunk_index"]))
        for c in a["claims"]:
            expl = a.get("explanation", "")
            start = claim_start(expl, int(c["claim_index"]), c["claim_text"])
            ev = answer_stated_evidence(expl, start, start + len(c["claim_text"]))
            rows.append({
                "problem_id": a["problem_id"], "window_kind": a["window_kind"],
                "chunk_index": int(a["chunk_index"]), "explanation_id": a.get("id"),
                "sample_index": a.get("sample_index"), "claim_index": c["claim_index"],
                "claim_text": c["claim_text"],
                "explanation": a.get("explanation", ""),
                "reconstruction_dependent": bool(c.get("reconstruction_dependent")),
                "asserts": assertions(c["claim_text"], expl, start),
                "answer_stated_reason": ev["reason"],
                "truth": truth,
            })
    return rows


def claim_start(explanation: str, claim_index: int, claim_text: str) -> int:
    """Character offset of a claim in its explanation, by the RQ2 splitter itself.

    Re-splitting with the same ``split_claims`` the audit used reproduces every
    claim on ``main`` (6,007 of 6,007), so the offset is the splitter's own,
    not a text search. Refuses on any mismatch rather than guessing.
    """
    from nlaast.faithfulness.claims import split_claims

    claims = split_claims(explanation, max_claims=claim_index + 1)
    if claim_index >= len(claims) or claims[claim_index].text != claim_text:
        raise ValueError(f"claim {claim_index} does not re-split from its explanation")
    return claims[claim_index].char_start


def check_disjoint(train_groups: Sequence[str], eval_rows: Sequence[dict]) -> None:
    """Leakage guard: no problem may supply both training and evaluated claims."""
    overlap = set(train_groups) & {r["problem_id"] for r in eval_rows}
    if overlap:
        raise AssertionError(f"train/eval problem overlap: {sorted(overlap)[:5]}")


# --- describe -------------------------------------------------------------

def describe(cfg, log) -> dict:
    traces, store, audits = load_inputs(cfg)
    X, labels, groups, positions = training_set(traces, store, cfg.target.layer)
    rows = eval_claims(audits, traces)
    check_disjoint(groups, rows)
    vocab = word_vocabulary(labels)
    stated = [(l["answer_stated"], p) for l, p in zip(labels, positions)
              if l["answer_stated"] is not None]
    out: dict[str, Any] = {
        "answer_stated_positional_predictability": positional_predictability(
            [s for s, _ in stated], [p for _, p in stated]),
        "phase": "describe (no probe fitted, no claim scored)",
        "n_train_vectors": int(X.shape[0]), "n_train_problems": len(set(groups)),
        "n_eval_claims": len(rows), "n_eval_problems": len({r["problem_id"] for r in rows}),
        "final_word_vocabulary": vocab,
        "train_label_base_rates": {
            "answer_stated": float(np.mean([l["answer_stated"] for l in labels
                                            if l["answer_stated"] is not None])),
            "domain": dict(Counter(l["domain"] for l in labels)),
            "final_word_in_vocab": float(np.mean([l["final_word"] in vocab for l in labels])),
        },
        "claims_asserting": {a: sum(1 for r in rows if a in r["asserts"]) for a in ATTRIBUTES},
        "claims_asserting_any": sum(1 for r in rows if r["asserts"]),
        "assertion_values": {a: dict(Counter(str(encode(a, r["asserts"][a], vocab))
                                             for r in rows if a in r["asserts"]))
                             for a in ATTRIBUTES},
        "eval_domains": dict(Counter(r["truth"]["domain"] for r in rows)),
    }
    write_json(cfg.dir.parent / f"{cfg.run_id}_rq2probe" / "describe.json", out)
    log.info("RQ2 probe describe: %s", {k: out[k] for k in
                                       ("n_train_vectors", "n_eval_claims", "claims_asserting")})
    return out


#: Annotation strata (A1 §4), fixed before any annotation: claims the rule
#: reads as asserting ``answer_stated``; claims it reads as asserting
#: ``domain`` but not ``answer_stated``; claims it reads as asserting neither.
ANNOTATION_STRATA = (("rule_answer_stated", 50), ("rule_domain_only", 50),
                     ("rule_neither", 50))


def annotation_stratum(asserts: dict) -> str:
    if "answer_stated" in asserts:
        return "rule_answer_stated"
    if "domain" in asserts:
        return "rule_domain_only"
    return "rule_neither"


def draw_annotation_sample(rows: Sequence[dict], seed: int) -> tuple[list[int], dict]:
    """Stratified, seeded, without replacement; returns row ids and the design.

    Each scored attribute gets claims the rule says assert it, and the third
    stratum supplies claims the rule says assert nothing - the only place a
    *missed* assertion can show up. Strata smaller than their quota are taken
    whole and the shortfall is recorded, not refilled from another stratum.
    """
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {s: [] for s, _ in ANNOTATION_STRATA}
    for i, r in enumerate(rows):
        by[annotation_stratum(r["asserts"])].append(i)
    pick: list[int] = []
    design: dict[str, Any] = {"strata": {}}
    for s, quota in ANNOTATION_STRATA:
        take = min(quota, len(by[s]))
        chosen = sorted(rng.choice(by[s], size=take, replace=False).tolist()) if take else []
        pick += chosen
        design["strata"][s] = {"quota": quota, "population": len(by[s]), "drawn": take}
    order = rng.permutation(len(pick))
    pick = [pick[i] for i in order]
    design["n"] = len(pick)
    design["rule_asserting_in_sample"] = {
        a: sum(1 for i in pick if a in rows[i]["asserts"]) for a in EVALUATED}
    design["sampling_note"] = (
        "strata are over-represented relative to the eval claims (the rule asserts "
        "answer_stated on 427 and domain on 770 of 3,779); kappa is computed on the "
        "sample as drawn, unweighted, as fixed in A1 §4")
    return pick, design


#: Annotation sheet schema v3 (A1 §4; guide: docs/RQ2_ANNOTATION_GUIDE.md).
#: Every code cell must be filled - an empty cell means "not annotated", never
#: "no assertion", which is the explicit code ``NA``.
SHEET_VERSION = 3
SHEET_COLUMNS = ("row_id", "claim_text", "explanation_context", "answer_stated",
                 "domain", "notes")
CODES = {"answer_stated": ("1", "0", "NA"),
         "domain": ("gsm8k", "math-algebra", "math-counting_and_probability", "NA")}
MARK_OPEN, MARK_CLOSE = "[[", "]]"


def mark_claim(explanation: str, claim: str) -> str:
    """The explanation with the claim to judge wrapped in ``[[ ]]``.

    Claims are segments of the explanation and are often cut inside quoted
    text, so a fragment alone can hide whether a quote is context or a
    predicted continuation. The surrounding explanation is shown to resolve
    that, and only that (guide §2). Refuses when the claim does not occur
    exactly once, rather than marking the wrong span.
    """
    c = claim.strip()
    if explanation.count(c) != 1:
        raise ValueError(f"claim occurs {explanation.count(c)} times in its explanation")
    i = explanation.index(c)
    return explanation[:i] + MARK_OPEN + c + MARK_CLOSE + explanation[i + len(c):]


def annotate_sample(cfg, log) -> Path:
    """Blind sheet v3: the claim, its explanation for context, explicit codes.

    Contains neither the rule's reading, the stratum, the window kind, nor the
    ground truth. Rows and their order are those of the v2 draw (same seed),
    so replacing the sheet changes what annotators see, not which claims.
    """
    traces, _, audits = load_inputs(cfg)
    rows = eval_claims(audits, traces)
    pick, design = draw_annotation_sample(rows, derive(cfg.seed, "rq2_annotation_v2"))
    root = cfg.dir.parent / f"{cfg.run_id}_rq2probe"
    out = root / f"annotation_sheet_v{SHEET_VERSION}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite an annotation sheet")
    root.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SHEET_COLUMNS)
        for i in pick:
            w.writerow([i, rows[i]["claim_text"],
                        mark_claim(rows[i]["explanation"], rows[i]["claim_text"]), "", "", ""])
    write_json(out.with_suffix(".key.json"), {
        "sheet_version": SHEET_VERSION, "row_ids": pick, "design": design,
        "codes": CODES, "seed_parts": [cfg.seed, "rq2_annotation_v2"],
        "guide": "docs/RQ2_ANNOTATION_GUIDE.md"})
    log.info("wrote %d-row blind annotation sheet to %s", len(pick), out)
    return out


def read_annotations(path: Path) -> list[dict[str, Any]]:
    """Parse a completed v3 sheet strictly: every code present and valid.

    Returns ``[{row_id, answer_stated, domain}]`` with ``NA`` as ``None`` and
    ``answer_stated`` as an int. Raises listing every empty or invalid cell, so
    a partly annotated sheet cannot enter the agreement as "no assertion".
    """
    out, problems = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        missing = [c for c in SHEET_COLUMNS if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"sheet lacks columns {missing}; expected schema v{SHEET_VERSION}")
        for rec in r:
            row = {"row_id": int(rec["row_id"])}
            for a, allowed in CODES.items():
                v = (rec[a] or "").strip()
                if v not in allowed:
                    problems.append(f"row {rec['row_id']}: {a}={v!r}")
                    continue
                row[a] = None if v == "NA" else (int(v) if a == "answer_stated" else v)
            out.append(row)
    if problems:
        raise ValueError(f"{len(problems)} empty or invalid cells: {problems[:10]}")
    ids = [x["row_id"] for x in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row_id in sheet")
    return out


def annotation_counts(rule: Sequence[Any], human: Sequence[Any]) -> dict[str, int]:
    """How much of the sample actually bears on an attribute.

    Kappa is computed over every annotated row with "no assertion" as a
    category, so rows where neither reader saw an assertion count as
    agreement. These counts say how many rows carried an assertion at all,
    so a kappa resting on few asserted rows is visible as such.
    """
    pairs = list(zip(rule, human))
    return {"rows": len(pairs),
            "rule_asserts": sum(1 for r, _ in pairs if r is not None),
            "annotator_asserts": sum(1 for _, h in pairs if h is not None),
            "either_asserts": sum(1 for r, h in pairs if r is not None or h is not None),
            "both_assert": sum(1 for r, h in pairs if r is not None and h is not None),
            "both_assert_same_value": sum(1 for r, h in pairs
                                          if r is not None and h is not None
                                          and str(r) == str(h))}


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa over paired labels, with ``None`` as its own category."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted({str(x) for x in a + b})
    po = sum(1 for x, y in zip(a, b) if str(x) == str(y)) / n
    pa = Counter(str(x) for x in a)
    pb = Counter(str(y) for y in b)
    pe = sum(pa[c] * pb[c] for c in cats) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


# --- evaluate (gated) -----------------------------------------------------

def amendment_approved(path: Path = AMENDMENT) -> bool:
    if not path.exists():
        return False
    return bool(re.search(r"^\*\*Status:\s*APPROVED", path.read_text(encoding="utf-8"), re.M))


def fit_probe(cfg, X: np.ndarray, y: Sequence[Any]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()] if cfg.probe.standardise else []
    clf = LogisticRegression(C=cfg.probe.C, max_iter=cfg.probe.max_iter)
    return make_pipeline(*steps, clf).fit(X, np.asarray(y, dtype=object).astype(str))


def support(probe, x: np.ndarray, value: Any) -> float:
    """P(asserted value | activation) under the probe; 0 for an unseen class."""
    classes = list(probe.classes_)
    p = probe.predict_proba(x[None, :])[0]
    v = str(value)
    return float(p[classes.index(v)]) if v in classes else 0.0


def confirms(probe, x: np.ndarray, value: Any) -> bool:
    return str(probe.predict(x[None, :])[0]) == str(value)


def derangement(items: Sequence[str], seed: int) -> dict[str, str]:
    """A seeded bijection with no fixed point: shuffle, then map each to the next.

    A single cycle through a random order is a permutation in which no item
    maps to itself, for any n >= 2. Every problem therefore donates its labels
    to exactly one other problem and receives exactly one problem's labels.
    """
    order = sorted(set(items))
    if len(order) < 2:
        raise ValueError("a derangement needs at least two problems")
    shuffled = [order[i] for i in np.random.default_rng(seed).permutation(len(order))]
    return {g: shuffled[(i + 1) % len(shuffled)] for i, g in enumerate(shuffled)}


def resample_by_position(src: Sequence[Any], n: int) -> list:
    """``src`` read at ``n`` evenly spaced relative positions (nearest index).

    How a donor's label sequence is fitted to a receiver with a different
    number of boundaries. Position along the trace is what structures these
    labels - ``answer_stated`` is 0 then 1, monotonically - so mapping by
    *relative* position keeps that structure and keeps the donor's class
    shares up to rounding. Copying by index instead (the earlier version)
    truncated a longer donor, dropping exactly the late boundaries where
    ``answer_stated`` is 1, and cycled a shorter one back to its start.
    """
    m = len(src)
    if m == 0 or n <= 0:
        return []
    if n == 1:
        return [src[m - 1]]
    # Half-up, not Python's half-to-even, so a tie always takes the later
    # (more-advanced) boundary regardless of parity.
    return [src[int(np.floor(j * (m - 1) / (n - 1) + 0.5))] for j in range(n)]


def permute_across_problems(y: Sequence[Any], groups: Sequence[str], seed: int) -> list:
    """Each problem's labels replaced by another problem's (leakage control).

    Moving labels *between* problems is what breaks a probe that has learned
    problem identity rather than the attribute; shuffling within a problem
    would leave a per-problem attribute such as ``domain`` untouched.

    Donors are assigned by ``derangement`` (a bijection with no problem keeping
    its own labels) and fitted to the receiver's length by
    ``resample_by_position``. For a per-problem attribute (``domain``) every
    receiver gets its donor's single value exactly. Rows of a problem must be
    in boundary order, which ``training_set`` guarantees.
    """
    target = derangement(groups, seed)
    by_group: dict[str, list] = {}
    for v, g in zip(y, groups):
        by_group.setdefault(g, []).append(v)
    fitted = {g: resample_by_position(by_group[target[g]], len(rows))
              for g, rows in by_group.items()}
    seen: Counter = Counter()
    out = []
    for g in groups:
        out.append(fitted[g][seen[g]])
        seen[g] += 1
    return out


def split_by_class_presence(claims: Sequence[dict], asserted, absent: Sequence[str]
                            ) -> tuple[list[dict], list[dict]]:
    """Claims whose asserted class the evaluation contains, and those it does not.

    A claim asserting a class with no true instance in the evaluation set
    cannot be confirmed by decoding - only by a probe error - so it is scored
    separately and never pooled into M2-M4.
    """
    absent = {str(a) for a in absent}
    present, missing = [], []
    for r in claims:
        (missing if str(asserted(r)) in absent else present).append(r)
    return present, missing


def class_shares(y: Sequence[Any]) -> dict[str, float]:
    c = Counter(str(v) for v in y)
    n = sum(c.values())
    return {k: v / n for k, v in sorted(c.items())} if n else {}


def mismatched_vector(partner_acts: np.ndarray, relative_position: float) -> np.ndarray:
    """The partner problem's boundary activation at the same relative position."""
    n = partner_acts.shape[0]
    i = int(round(min(1.0, max(0.0, relative_position)) * (n - 1)))
    return partner_acts[i]


def cluster_bootstrap_rate_diff(groups: Sequence[str], a: Sequence[float], mask_a: Sequence[bool],
                                n_boot: int, conf: float, seed: int) -> dict:
    """Rate among ``mask_a`` rows minus rate among the rest, resampling problems."""
    groups = np.asarray(groups)
    a = np.asarray(a, dtype=float)
    m = np.asarray(mask_a, dtype=bool)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}

    def stat(sel):
        s = np.concatenate([idx[g] for g in sel])
        if not m[s].any() or m[s].all():
            return np.nan
        return a[s][m[s]].mean() - a[s][~m[s]].mean()

    point = stat(ug)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(ug, size=len(ug), replace=True)) for _ in range(n_boot)])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2]) if bs.size else (np.nan, np.nan)
    return {"estimate": float(point), "low": float(lo), "high": float(hi), "n_rows": int(len(a)),
            "n_problems": int(len(ug)), "method": "problem-cluster percentile bootstrap"}


def cluster_bootstrap_mean(groups: Sequence[str], values: Sequence[float],
                           n_boot: int, conf: float, seed: int) -> dict:
    """Row-weighted mean with an interval from resampling whole problems.

    Claims, explanations, samples and windows are all nested inside problems
    and are not independent of each other; resampling problems keeps every
    such dependence inside the resampled unit. The estimand is the
    row-weighted (claim- or boundary-weighted) rate, so a problem with more
    rows weighs more in the point estimate, and the interval reflects that.
    """
    groups = np.asarray(groups)
    v = np.asarray(values, dtype=float)
    ug = np.unique(groups)
    if v.size == 0:
        return {"estimate": None, "low": None, "high": None, "n_rows": 0, "n_problems": 0,
                "method": "problem-cluster percentile bootstrap"}
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    bs = np.array([v[np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug))])].mean()
                   for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return {"estimate": float(v.mean()), "low": float(lo), "high": float(hi),
            "n_rows": int(v.size), "n_problems": int(len(ug)),
            "method": "problem-cluster percentile bootstrap"}


def cluster_bootstrap_paired_diff(groups: Sequence[str], a: Sequence[float], b: Sequence[float],
                                  n_boot: int, conf: float, seed: int) -> dict:
    """``mean(a - b)`` over paired rows, resampling problems.

    Used where both quantities are measured on the same rows: probe accuracy
    against majority-class accuracy on the same eval boundaries (M0), and a
    claim confirmed at its own activation against the same claim at a
    mismatched one (M3).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = cluster_bootstrap_mean(groups, d, n_boot, conf, seed)
    out["estimand"] = "mean paired difference (first minus second), row-weighted"
    return out


#: M0's pre-specified criterion (A1 §6), fixed before evaluation: the probe
#: decodes an attribute only if the lower bound of the problem-cluster
#: interval for (probe accuracy - majority-class accuracy) is above zero.
#: The leakage control fails if the same lower bound for the shuffled-label
#: probe is above zero.
def m0_verdict(probe_vs_majority: dict, shuffled_vs_majority: dict) -> dict:
    lo = probe_vs_majority.get("low")
    slo = shuffled_vs_majority.get("low")
    return {"decodes": bool(lo is not None and lo > 0),
            "leakage_detected": bool(slo is not None and slo > 0)}


def within_problem_shuffle(y: Sequence[Any], groups: Sequence[str], seed: int
                           ) -> tuple[list, dict]:
    """Permute labels inside each problem (F6's registered "within group" form).

    Each problem keeps exactly its own labels - so its prevalence, and the
    corpus prevalence, are unchanged - but which of its boundaries carries
    which label is randomised, which breaks both the label-activation link and
    the label-position link. Reports how many labels moved, as F6 does: a
    shuffle that moves nothing (a problem whose labels are all equal) is no
    control, and the count makes that visible.
    """
    rng = np.random.default_rng(seed)
    y = list(y)
    out = list(y)
    order = sorted(set(groups))
    for g in order:
        idx = [i for i, h in enumerate(groups) if h == g]
        vals = [y[i] for i in idx]
        for i, v in zip(idx, [vals[j] for j in rng.permutation(len(vals))]):
            out[i] = v
    changed = sum(1 for a, b in zip(y, out) if a != b)
    movable = sum(1 for g in order if len({y[i] for i, h in enumerate(groups) if h == g}) > 1)
    return out, {"scope": "within_problem", "n_labels_changed": changed,
                 "fraction_changed": changed / len(y) if y else 0.0,
                 "n_problems": len(order), "n_problems_with_mixed_labels": movable}


#: Which shuffle each attribute's leakage control uses (A1 §5, Decision 1).
SHUFFLE_SCOPE = {"answer_stated": "within_problem", "domain": "across_problems"}


def control_shuffle(attr: str, y: Sequence[Any], groups: Sequence[str], seed: int
                    ) -> tuple[list, dict]:
    """The registered-form leakage shuffle for ``attr``.

    ``answer_stated`` varies inside a problem, so the within-problem shuffle
    F6 registers applies and is not degenerate (it moves the stated
    boundary). ``domain`` is constant inside a problem, so a within-problem
    shuffle is the identity; it uses the across-problem swap
    (``permute_across_problems``), which transfers one value per problem
    exactly - the arm the project's F6 code also treats as primary.
    """
    scope = SHUFFLE_SCOPE[attr]
    if scope == "within_problem":
        return within_problem_shuffle(y, groups, seed)
    out = permute_across_problems(y, groups, seed)
    changed = sum(1 for a, b in zip(y, out) if str(a) != str(b))
    return out, {"scope": "across_problems", "n_labels_changed": changed,
                 "fraction_changed": changed / len(y) if y else 0.0,
                 "n_problems": len(set(groups))}


#: Position-only comparator features (A1 §6, Decision 1), fixed before evaluation.
POSITION_FEATURES = ("relative_position", "is_final_boundary", "boundary_index",
                     "log_prefix_tokens")


def position_matrix(positions: Sequence[dict]) -> np.ndarray:
    return np.array([[p[f] for f in POSITION_FEATURES] for p in positions], dtype=float)


def position_verdict(probe_minus_position: dict) -> dict:
    """A1 §7 criterion for ``answer_stated``, fixed before evaluation.

    Evidence of information *beyond the position features* requires the lower
    bound of the problem-cluster interval for (activation-probe accuracy -
    position-model accuracy) on the same eval boundaries to be strictly above
    zero. Even then it is evidence about decodability beyond these four
    features, not about semantic understanding. Not meeting it means only that
    predictive value beyond these features was not established - not that the
    activations contain no such information. Accuracy is the primary metric;
    the Brier comparison is secondary and never enters this verdict.
    """
    lo = probe_minus_position.get("low")
    return {"beats_position_model": bool(lo is not None and lo > 0)}


def answer_stated_reading(m0_verdict_: dict, position_verdict_: dict) -> str:
    """The pre-specified reading for ``answer_stated`` (A1 §7)."""
    if m0_verdict_.get("leakage_detected"):
        return "invalid: the within-problem shuffled probe beats the majority rate"
    if not m0_verdict_.get("decodes"):
        return "inconclusive: the probe does not beat the majority rate"
    if not position_verdict_.get("beats_position_model"):
        return ("decodable, but not distinguishable from boundary position: the "
                "experiment did not establish predictive value beyond the four "
                "position features (this does not show the activations lack "
                "such information)")
    return ("decodable beyond the position features used (not evidence of "
            "semantic understanding)")


def evaluate(cfg, log, annotation: Path | None) -> int:
    root = cfg.dir.parent / f"{cfg.run_id}_rq2probe"
    if not amendment_approved():
        log.error("amendment A1 is not marked APPROVED in %s; the evaluation does not run",
                  AMENDMENT)
        return 2
    # Decision 2 (2026-10-17): no independent human annotation. The
    # claim->assertion rule is therefore *unvalidated*; every attribute is
    # scored and every claim-level value it produces is labelled rule-derived.
    kept = list(EVALUATED)
    rule_validation = {
        "human_validation": "not performed",
        "decision": "researcher Decision 2, 2026-10-17: no human annotation",
        "kappa": None,
        "consequence": ("claim assertions are the lexicon rule's reading, not "
                        "validated labels; M1-M4 measure the rule's reading of "
                        "each claim, and a rule error is indistinguishable from "
                        "a verbaliser error"),
    }
    log.warning("claim assertions are rule-derived and not human-validated (Decision 2)")

    traces, store, audits = load_inputs(cfg)
    layer = cfg.target.layer
    X, labels, groups, positions = training_set(traces, store, layer)
    rows = eval_claims(audits, traces)
    check_disjoint(groups, rows)
    vocab = word_vocabulary(labels)
    conf, nb, seed = cfg.analysis.confidence, cfg.analysis.bootstrap_iterations, cfg.analysis.seed

    eval_problems = sorted({r["problem_id"] for r in rows})
    acts = {p: store.get(p, layer) for p in eval_problems}

    def vec(pid, ci):
        arr, meta = acts[pid]
        return arr[list(meta["chunk_indices"]).index(ci)]

    rng = np.random.default_rng(derive(cfg.seed, "rq2_mismatch"))
    partner = {p: rng.choice([q for q in eval_problems if q != p]) for p in eval_problems}

    result: dict[str, Any] = {"attributes": {}, "vocabulary": vocab, "kept": kept,
                              "rule_validation": rule_validation}
    for attr in kept:
        y = [encode(attr, l[attr], vocab) for l in labels]
        keep = [i for i, v in enumerate(y) if v is not None]
        y_keep = [y[i] for i in keep]
        probe = fit_probe(cfg, X[keep], y_keep)
        # Leakage control, F6's form per attribute (A1 §5): within-problem
        # for answer_stated, across-problem for domain. Probe refitted on the
        # shuffled labels must not beat the majority rate on held-out windows.
        y_perm, shuffle_info = control_shuffle(attr, y_keep, [groups[i] for i in keep], seed)
        shuf = fit_probe(cfg, X[keep], y_perm)

        # Probe validity on every eval boundary, not only audited windows.
        ev_X, ev_y, ev_g, ev_pos = [], [], [], []
        for p in eval_problems:
            arr, meta = acts[p]
            for row, ci in enumerate(meta["chunk_indices"]):
                v = encode(attr, boundary_labels(traces[p], ci)[attr], vocab)
                if v is not None:
                    ev_X.append(arr[row]); ev_y.append(str(v)); ev_g.append(p)
                    ev_pos.append(boundary_position(traces[p], ci))
        ev_X = np.vstack(ev_X)
        acc = (probe.predict(ev_X) == np.asarray(ev_y)).astype(float)
        acc_shuf = (shuf.predict(ev_X) == np.asarray(ev_y)).astype(float)
        majority = Counter(str(v) for v in y_keep).most_common(1)[0][0]
        acc_major = (np.asarray(ev_y) == majority).astype(float)
        m0 = {
            "probe_accuracy": cluster_bootstrap_mean(ev_g, acc, nb, conf, seed),
            "majority_class": majority,
            "majority_accuracy": cluster_bootstrap_mean(ev_g, acc_major, nb, conf, seed),
            "shuffled_label_accuracy": cluster_bootstrap_mean(ev_g, acc_shuf, nb, conf, seed),
            "probe_minus_majority": cluster_bootstrap_paired_diff(ev_g, acc, acc_major, nb, conf, seed),
            "shuffled_minus_majority": cluster_bootstrap_paired_diff(ev_g, acc_shuf, acc_major,
                                                                     nb, conf, seed),
            "train_label_shares": class_shares(y_keep),
            "permuted_label_shares": class_shares(y_perm),
            "shuffle": shuffle_info,
        }
        m0["verdict"] = m0_verdict(m0["probe_minus_majority"], m0["shuffled_minus_majority"])

        # Position-only comparator (answer_stated only; A1 §6-§7). Same
        # training rows and labels, position features only, same model class.
        position_model = None
        if attr == "answer_stated":
            position_model = fit_probe(cfg, position_matrix([positions[i] for i in keep]), y_keep)
            P_ev = position_matrix(ev_pos)
            acc_pos = (position_model.predict(P_ev) == np.asarray(ev_y)).astype(float)
            y_ev = (np.asarray(ev_y) == "1").astype(float)
            brier_probe = (probe.predict_proba(ev_X)[:, list(probe.classes_).index("1")] - y_ev) ** 2
            brier_pos = (position_model.predict_proba(P_ev)[:, list(position_model.classes_).index("1")]
                         - y_ev) ** 2
            m0["position_comparator"] = {
                "features": list(POSITION_FEATURES),
                "position_model_accuracy": cluster_bootstrap_mean(ev_g, acc_pos, nb, conf, seed),
                "probe_minus_position": cluster_bootstrap_paired_diff(ev_g, acc, acc_pos,
                                                                      nb, conf, seed),
                "secondary_brier_position_minus_probe": cluster_bootstrap_paired_diff(
                    ev_g, brier_pos, brier_probe, nb, conf, seed),
                "eval_positional_predictability": positional_predictability(
                    [int(v) for v in ev_y], ev_pos),
            }
            m0["position_verdict"] = position_verdict(m0["position_comparator"]["probe_minus_position"])
            m0["reading"] = answer_stated_reading(m0["verdict"], m0["position_verdict"])

        # Which classes the evaluation can test at all. The eval split's
        # NLA windows hold no MATH counting-and-probability problem (D54), so
        # for ``domain`` the probe's third class has no true instance there:
        # M0 measures two-class discrimination, and claims asserting the
        # absent class are kept apart - their "confirmation" can only be a
        # probe false positive, never evidence of decoding.
        train_classes = sorted(class_shares(y_keep))
        eval_classes = sorted(set(ev_y))
        absent = [c for c in train_classes if c not in eval_classes]
        coverage = {"train_classes": train_classes, "eval_classes": eval_classes,
                    "classes_absent_from_eval": absent,
                    "eval_boundaries_per_class": dict(Counter(ev_y)),
                    "eval_problems_per_class": {c: len({g for g, v in zip(ev_g, ev_y) if v == c})
                                                for c in eval_classes}}

        claims_all = [r for r in rows if attr in r["asserts"]]
        claims, absent_claims = split_by_class_presence(
            claims_all, lambda r: encode(attr, r["asserts"][attr], vocab), absent)
        absent_fp = [float(confirms(probe, vec(r["problem_id"], r["chunk_index"]),
                                    encode(attr, r["asserts"][attr], vocab)))
                     for r in absent_claims]
        own, mis, true_, pos_conf = [], [], [], []
        for r in claims:
            v = encode(attr, r["asserts"][attr], vocab)
            x = vec(r["problem_id"], r["chunk_index"])
            n_own = len(traces[r["problem_id"]]["chunks"])
            xm = mismatched_vector(acts[partner[r["problem_id"]]][0],
                                   r["chunk_index"] / max(1, n_own - 1))
            own.append(float(confirms(probe, x, v)))
            mis.append(float(confirms(probe, xm, v)))
            true_.append(float(str(v) == str(encode(attr, r["truth"][attr], vocab))))
            if position_model is not None:
                pf = position_matrix([boundary_position(traces[r["problem_id"]], r["chunk_index"])])
                pos_conf.append(float(str(position_model.predict(pf)[0]) == str(v)))
        g = [r["problem_id"] for r in claims]
        dep = [r["reconstruction_dependent"] for r in claims]
        claim_position = ({
            "m2_confirmation_rate_position_model": cluster_bootstrap_mean(g, pos_conf, nb, conf, seed),
            "m2_probe_minus_position_model": cluster_bootstrap_paired_diff(g, own, pos_conf,
                                                                           nb, conf, seed),
            "note": "descriptive; the decision criterion is M0's position comparison",
        } if position_model is not None else None)
        result["attributes"][attr] = {
            "n_train": len(keep), "n_eval_boundaries": int(len(ev_y)),
            "n_eval_problems": len(set(ev_g)),
            "class_coverage": coverage,
            "m0_probe_validity": m0,
            "n_claims_scored": len(claims), "n_claim_problems": len(set(g)),
            "n_claims_asserting_absent_class": len(absent_claims),
            "absent_class_claims_probe_false_positive_rate": (
                cluster_bootstrap_mean([r["problem_id"] for r in absent_claims], absent_fp,
                                       nb, conf, seed) if absent_claims else None),
            "assertion_source": "rule-derived (lexicon), not human-validated",
            "assertion_interpretation": ATTRIBUTE_INTERPRETATION[attr],
            "answer_stated_abstention_reasons": (
                dict(Counter(r["answer_stated_reason"] for r in rows))
                if attr == "answer_stated" else None),
            "ground_truth_source": "text-derived (trace and problem record)",
            "m1_rule_reading_matches_text_truth": cluster_bootstrap_mean(g, true_, nb, conf, seed),
            "claim_level_position_comparison": claim_position,
            "m2_confirmation_rate_own": cluster_bootstrap_mean(g, own, nb, conf, seed),
            "m2_confirmation_rate_mismatched": cluster_bootstrap_mean(g, mis, nb, conf, seed),
            "m3_own_minus_mismatched": cluster_bootstrap_paired_diff(g, own, mis, nb, conf, seed),
            "m4_dependent_minus_not": cluster_bootstrap_rate_diff(g, own, dep, nb, conf, seed),
        }
    write_json(root / "evaluation.json", result)
    log.info("wrote %s", root / "evaluation.json")
    return 0


def agreement(cfg, log, annotation: Path, threshold: float = 0.6) -> int:
    """Kappa between the rule and an *independent human* annotation of the v3 sheet.

    Not run under Decision 2 (2026-10-17: no human annotation). Kept for
    labels added later. Must never be given the AI annotation pass, which is a
    JSONL file precisely so that ``read_annotations`` (CSV, v3 schema) rejects it.
    """
    traces, _, audits = load_inputs(cfg)
    rows = eval_claims(audits, traces)
    got: dict[str, list] = {a: [] for a in EVALUATED}
    rule: dict[str, list] = {a: [] for a in EVALUATED}
    ann = read_annotations(annotation)
    key = read_json(annotation.with_suffix(".key.json"))
    if [x["row_id"] for x in ann] != key["row_ids"]:
        raise ValueError("sheet rows do not match its key; was the sheet re-ordered or edited?")
    for rec in ann:
        for a in EVALUATED:
            got[a].append(rec[a])
            rule[a].append(rows[rec["row_id"]]["asserts"].get(a))
    kappa = {a: cohen_kappa(rule[a], got[a]) for a in EVALUATED}
    out = {"kappa": kappa, "threshold": threshold, "n": len(got["domain"]),
           "usable": {a: annotation_counts(rule[a], got[a]) for a in EVALUATED},
           "passed": {a: bool(kappa[a] >= threshold) for a in EVALUATED}}
    write_json(cfg.dir.parent / f"{cfg.run_id}_rq2probe" / "annotation_agreement.json", out)
    log.info("annotation agreement: %s", out)
    return 0


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--phase", choices=["describe", "annotate-sample", "agreement", "evaluate"],
                   required=True)
    p.add_argument("--annotation", type=Path, default=None,
                   help="completed annotation sheet (agreement phase)")
    args = p.parse_args()
    cfg = config_mod.load(args.config, args.overrides)
    log = logging_utils.setup(cfg.dir.parent / f"{cfg.run_id}_rq2probe")
    if args.phase == "describe":
        describe(cfg, log)
        return 0
    if args.phase == "annotate-sample":
        annotate_sample(cfg, log)
        return 0
    if args.phase == "agreement":
        if args.annotation is None:
            log.error("--annotation is required")
            return 2
        return agreement(cfg, log, args.annotation)
    return evaluate(cfg, log, args.annotation)


if __name__ == "__main__":
    sys.exit(main())
