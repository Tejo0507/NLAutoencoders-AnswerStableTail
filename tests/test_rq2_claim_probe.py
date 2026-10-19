"""RQ2's claim-content probe (amendment A1): labels, assertions, splits, leakage.

The control is only worth running if four things hold, and each is pinned
here: the ground truth comes from the trace and never from a verbalisation;
the rule that reads a claim's assertion does what A1 says and abstains when
its cues conflict; no problem supplies both training activations and
evaluated claims; and the leakage control actually moves labels between
problems. The evaluation itself is gated on the amendment being approved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rq2_claim_probe as R  # noqa: E402


def trace(text, final, pid="gsm8k-test-00001", split="train"):
    from nlaast.config import from_mapping
    from nlaast.trace.chunking import chunk_trace
    chunks = chunk_trace(text, from_mapping({}).chunk)
    return {"problem_id": pid, "trace": text, "final_answer": final, "split": split,
            "chunks": [c.to_dict() for c in chunks]}


# --- ground truth -----------------------------------------------------------

def test_domain_comes_from_the_problem_id():
    assert R.domain_of("gsm8k-test-00007") == "gsm8k"
    assert R.domain_of("math-algebra-test-00002") == "math-algebra"
    assert R.domain_of("math-counting_and_probability-test-00144") == \
        "math-counting_and_probability"
    with pytest.raises(ValueError):
        R.domain_of("aime-test-1")


@pytest.mark.parametrize("text,word", [
    ("He buys 3 apples.", "apples"),
    ("The total is 160.", R.NUM),
    ("It costs 1,250.50 dollars.", "dollars"),
    ("x = 2", R.NUM),
    ("Therefore, X.", "x"),
    ("...", None),
    ("", None),
])
def test_normalise_word(text, word):
    assert R.normalise_word(text) == word


def test_answer_stated_is_cumulative_and_uses_the_strict_extractor():
    t = trace("First we add 2 and 2. That gives 4. The answer is 4. We are done here.", "4")
    labels = [R.boundary_labels(t, i)["answer_stated"] for i in range(len(t["chunks"]))]
    first = labels.index(1)
    assert labels[:first] == [0] * first
    assert all(v == 1 for v in labels[first:])   # stays stated once stated


def test_answer_stated_is_none_without_a_final_answer():
    t = trace("We try. We fail.", None)
    assert R.boundary_labels(t, 0)["answer_stated"] is None


# --- the assertion rule -----------------------------------------------------

@pytest.mark.parametrize("claim,expected", [
    # positive: already-written answer text, quoted as context
    ('Final token "answer." closes the concluding summary ("Therefore, our final answer is 55.")', 1),
    ('The phrase "Thus, the final answer is 12" ends the solution.', 1),
    # positive: description says the context concludes
    ('Final token "x" ends the concluding sentence of the solution.', 1),
    # negative: every predicted alternative is an answer statement
    ('ends mid-clause, strongly expecting "The final answer is" to follow.', 0),
    ('immediately expecting "Therefore, the final answer is" or "Thus, the final result is"', 0),
    # neither
    ('Structured math problem with numbered steps.', None),
])
def test_answer_stated_assertions(claim, expected):
    assert R.assertions(claim).get("answer_stated") == expected


def test_conflicting_answer_cues_abstain():
    c = ('closes the concluding summary, strongly expecting "The final answer is" next')
    assert "answer_stated" not in R.assertions(c)


# --- revision 2026-10-17: closing phrases ------------------------------------

@pytest.mark.parametrize("claim,reason", [
    # expecting only a closing remark: neutral, not "answer still to come"
    ('the clause ends, strongly expecting "This concludes the answer" or "end of solution"',
     "closing_only"),
    ('immediately expecting "The final answer is confirmed" to close it', "ambiguous_predicted_phrase"),
    # alternatives mixing an answer statement and a closing remark: abstain
    ('strongly expecting "The final answer is" or "This concludes the problem"',
     "mixed_expected_alternatives"),
])
def test_closing_phrases_are_never_read_as_answer_still_to_come(claim, reason):
    ev = R.answer_stated_evidence(claim, 0, len(claim))
    assert ev["value"] is None and ev["reason"] == reason


def test_closing_expectation_after_a_written_answer_reads_stated():
    # The old rule read this as 0 ("expecting ... the answer").
    c = ('Final token "x" closes the clause ("Thus, our final answer is: 7"), '
         'strongly expecting "This concludes the answer" or "end of the solution."')
    ev = R.answer_stated_evidence(c, 0, len(c))
    assert ev["value"] == 1 and ev["expects_closing"] and not ev["expects_answer"]


def test_a_closing_phrase_alone_is_not_relabelled_as_stated():
    c = 'The step ends, likely followed by "This completes the calculation."'
    assert R.answer_stated_evidence(c, 0, len(c))["value"] is None


def test_prediction_idioms_open_an_expectation_scope():
    # "likely followed by" predicts; its quotes are not already-written text.
    c = 'Final token "1.5." closes the number, likely followed by "Thus, the final answer is"'
    ev = R.answer_stated_evidence(c, 0, len(c))
    assert not ev["stated_context_quote"] and ev["expects_answer"]


# --- revision 2026-10-17: quotes cut by the claim splitter --------------------

def test_quote_scope_map_tracks_quotes_and_expectation_scope():
    t = 'A "b. c" d, expecting "e" or "f". Then "g".'
    in_q, in_s, bal = R.quote_scope_map(t)
    assert bal
    assert in_q[t.index("b")] and in_q[t.index("c")] and not in_q[t.index("d")]
    assert in_s[t.index('"e"')] and in_s[t.index('"f"')]   # alternatives stay in scope
    assert not in_s[t.index("Then")] and not in_s[t.index('"g"')]  # scope ends at sentence end


def test_quote_state_spans_paragraph_breaks():
    t = 'Final token "item:\n\n" closes a header.'
    in_q, _, bal = R.quote_scope_map(t)
    assert bal and in_q[t.index("item")] and not in_q[t.index("closes")]


def test_unbalanced_explanation_abstains():
    t = 'The sentence "Thus, the final answer is 4. Final token "x" closes it.'
    ev = R.answer_stated_evidence(t, 0, len(t))
    assert ev["value"] is None and ev["reason"] == "unbalanced_quotes"


# Real examples from runs/main/faithfulness/audits.jsonl (eval split), copied
# verbatim. Each claim starts with text cut out of a quotation by the splitter.
REAL_STATED_IN_CONTEXT = (
    "Structured math problem format with numbered steps and labeled clauses, presenting a "
    "logical derivation of a quantity's value with the given conditions about a chemical "
    "process.\n\nThe final sentence \"Thus, our final answer is: Therefore, the conclusion "
    "statement is: The answer is determined...\" signals a concluding closure pattern, likely "
    "completing the \"Final Answer\" label with the actual derived value or restating the final "
    "statement.\n\nFinal token \"value). \" closes the concluding summary clause (\"Thus, the "
    "answer is determined as above. Therefore, the final answer is...\"), strongly expecting "
    "\"The final answer is\" or \"This concludes the problem\" or \"The final statement is "
    "[boxed]\" to finish the answer format.")
REAL_CLOSING_ONLY = (
    "Structured math problem format with numbered steps and answer calculations, establishing "
    "a pattern of presenting a sequence of numerical values from a puzzle context.\n\nThe "
    "concluding sentence \"Therefore, after completing all calculations, the final statement "
    "is: My average value is approximately 5. Thus, the final answer is: Optimized. Therefore, "
    "answer.\" suggests a closing punctuation or confirmation of the final answer term.\n\n"
    "Final token \"term.\" closes the parenthetical clarification (\"Thus, final answer is: 3.5 "
    "and hence the answer. Final answer: Optimized. \"), strongly expecting \" \" to close the "
    "sentence, or \"This concludes the answer\" or \"Thus the integer value is adjusted.\"")


def real_claims(explanation):
    from nlaast.faithfulness.claims import split_claims
    return split_claims(explanation)


def test_real_claim_inside_a_written_quotation_reads_stated():
    cl = real_claims(REAL_STATED_IN_CONTEXT)
    target = next(c for c in cl if c.text.startswith('" closes the concluding summary clause'))
    # The claim begins with the closing quote of 'Final token "value). "'. Read
    # alone, that stray quote shifts every later pairing; read in its
    # explanation, ("Thus, the answer is determined as above...") is
    # already-written context and "closes the concluding" describes it.
    assert R.claim_start(REAL_STATED_IN_CONTEXT, target.index, target.text) == target.char_start
    ev = R.answer_stated_evidence(REAL_STATED_IN_CONTEXT, target.char_start,
                                  target.char_start + len(target.text))
    assert ev["value"] == 1 and ev["stated_context_quote"] and ev["stated_description"]
    alone = R.answer_stated_evidence(target.text, 0, len(target.text))
    assert not alone["stated_context_quote"]          # what the misaligned pairing loses


def test_real_fragment_starting_with_a_stray_quote_is_not_read_as_answer_to_come():
    cl = real_claims(REAL_CLOSING_ONLY)
    target = next(c for c in cl if c.text.startswith('"), strongly expecting'))
    ev = R.answer_stated_evidence(REAL_CLOSING_ONLY, target.char_start,
                                  target.char_start + len(target.text))
    # The old rule read this claim as 0. It expects only closing remarks.
    assert ev["value"] is None and ev["reason"] == "closing_only"
    assert R.assertions(target.text, REAL_CLOSING_ONLY, target.char_start).get("answer_stated") is None


def test_claim_text_is_never_modified():
    cl = real_claims(REAL_CLOSING_ONLY)
    for c in cl:
        assert REAL_CLOSING_ONLY[c.char_start:c.char_start + len(c.text)] == c.text


def test_claim_start_refuses_a_mismatch():
    with pytest.raises(ValueError):
        R.claim_start("One claim here. Another claim.", 0, "Not the claim.")


@pytest.mark.parametrize("claim,expected", [
    ("A word problem about a shop's prices.", "gsm8k"),
    ("Factoring a quadratic polynomial.", "math-algebra"),
    ("Computing the probability of two dice.", "math-counting_and_probability"),
    ("A word problem about the probability of rain.", None),     # two domains -> abstain
    ("Step-by-step breakdown.", None),
])
def test_domain_assertions(claim, expected):
    assert R.assertions(claim).get("domain") == expected


def test_final_word_assertion_is_normalised():
    assert R.assertions('Final token "time." closes a clause')["final_word"] == "time"
    assert R.assertions('final token "160" ends the line')["final_word"] == R.NUM
    assert "final_word" not in R.assertions("no token is quoted here")


# --- splits and leakage -----------------------------------------------------

def test_check_disjoint_rejects_overlap():
    R.check_disjoint(["a", "b"], [{"problem_id": "c"}])
    with pytest.raises(AssertionError):
        R.check_disjoint(["a", "b"], [{"problem_id": "b"}])


def test_eval_claims_take_only_eval_split_problems():
    tr = {"p1": trace("We get 4. The answer is 4.", "4", pid="gsm8k-test-1", split="train"),
          "p2": trace("We get 5. The answer is 5.", "5", pid="gsm8k-test-2", split="eval")}
    tr["p1"]["problem_id"], tr["p2"]["problem_id"] = "p1", "p2"
    tr = {k: {**v, "problem_id": k} for k, v in tr.items()}
    # domain_of needs a recognisable id
    tr = {f"gsm8k-test-{k}": {**v, "problem_id": f"gsm8k-test-{k}"} for k, v in tr.items()}
    audits = [{"problem_id": pid, "window_kind": "tail", "chunk_index": 0, "id": pid,
               "explanation": "A word problem.",
               "claims": [{"claim_index": 0, "claim_text": "A word problem.",
                           "reconstruction_dependent": True}]} for pid in tr]
    rows = R.eval_claims(audits, tr)
    assert [r["problem_id"] for r in rows] == ["gsm8k-test-p2"]
    assert rows[0]["asserts"] == {"domain": "gsm8k"}
    assert rows[0]["truth"]["domain"] == "gsm8k"


def test_permute_across_problems_moves_every_problem_off_its_own_labels():
    groups = [g for g in "abcdef" for _ in range(3)]
    y = [g.upper() for g in groups]           # a per-problem attribute
    out = R.permute_across_problems(y, groups, seed=0)
    assert len(out) == len(y)
    for g in "abcdef":
        mine = {v for v, h in zip(out, groups) if h == g}
        assert mine and g.upper() not in mine


def test_permute_is_deterministic():
    groups = list("aabbcc")
    y = list("xxyyzz")
    assert R.permute_across_problems(y, groups, 3) == R.permute_across_problems(y, groups, 3)


@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (7, 2), (101, 3), (101, 11)])
def test_derangement_is_a_bijection_without_fixed_points(n, seed):
    items = [f"p{i:03d}" for i in range(n)]
    d = R.derangement(items, seed)
    assert sorted(d) == sorted(items)
    assert sorted(d.values()) == sorted(items)          # every donor used exactly once
    assert all(k != v for k, v in d.items())


def test_derangement_needs_two_problems():
    with pytest.raises(ValueError):
        R.derangement(["only"], 0)


@pytest.mark.parametrize("src,n,expected", [
    ([0, 0, 1, 1], 4, [0, 0, 1, 1]),            # same length: identity
    ([0, 0, 0, 1], 2, [0, 1]),                  # shorter receiver keeps both ends
    ([0, 1], 5, [0, 0, 1, 1, 1]),               # longer receiver: no cycling back to 0
    ([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 3, [0, 0, 1]),
    (["a"], 3, ["a", "a", "a"]),
    ([0, 1, 1], 1, [1]),                        # one boundary takes the donor's last
])
def test_resample_by_position(src, n, expected):
    assert R.resample_by_position(src, n) == expected


def test_unequal_lengths_keep_monotone_structure_and_end_labels():
    # answer_stated is 0...0 then 1...1. The old index copy truncated a long
    # donor (losing its 1s) and cycled a short one (putting 0s after 1s).
    lengths = {"a": 3, "b": 12, "c": 5, "d": 20, "e": 2, "f": 9}
    groups, y = [], []
    for g, n in lengths.items():
        k = max(1, n // 4)                                  # last quarter stated
        groups += [g] * n
        y += [0] * (n - k) + [1] * k
    out = R.permute_across_problems(y, groups, seed=5)
    assert len(out) == len(y)
    for g in lengths:
        seq = [v for v, h in zip(out, groups) if h == g]
        assert len(seq) == lengths[g]
        assert seq == sorted(seq)                           # never 1 then 0
        assert seq[-1] == 1                                 # the donor's stated end survives


def test_unequal_lengths_preserve_class_shares_approximately():
    rng = np.random.default_rng(0)
    groups, y = [], []
    for i in range(60):
        n = int(rng.integers(3, 25))
        k = int(rng.integers(0, n))
        groups += [f"g{i}"] * n
        y += [0] * (n - k) + [1] * k
    before = R.class_shares(y)["1"]
    after = R.class_shares(R.permute_across_problems(y, groups, seed=1))["1"]
    # Lengths are re-matched to receivers, so shares move a little, not a lot.
    assert abs(after - before) < 0.05


def test_per_problem_attribute_is_transferred_exactly():
    groups = ["a"] * 2 + ["b"] * 7 + ["c"] * 4
    y = ["gsm8k"] * 2 + ["math-algebra"] * 7 + ["math-counting_and_probability"] * 4
    out = R.permute_across_problems(y, groups, seed=0)
    for g in "abc":
        vals = {v for v, h in zip(out, groups) if h == g}
        assert len(vals) == 1                               # one donor value, no mixing
    donated = sorted({v for v, h in zip(out, groups)})
    assert donated == sorted(set(y))                        # bijection: every value still used


# --- domain coverage ---------------------------------------------------------

def test_claims_asserting_an_absent_class_are_kept_apart():
    claims = [{"v": "gsm8k"}, {"v": "math-counting_and_probability"}, {"v": "math-algebra"}]
    present, missing = R.split_by_class_presence(
        claims, lambda r: r["v"], ["math-counting_and_probability"])
    assert [r["v"] for r in present] == ["gsm8k", "math-algebra"]
    assert [r["v"] for r in missing] == ["math-counting_and_probability"]
    p2, m2 = R.split_by_class_presence(claims, lambda r: r["v"], [])
    assert len(p2) == 3 and m2 == []


# --- clustered uncertainty ---------------------------------------------------

def test_cluster_bootstrap_mean_resamples_problems_not_rows():
    # Two problems, each internally constant: the row-level SE would be tiny,
    # but with two clusters the problem-level interval spans both values.
    groups = ["a"] * 50 + ["b"] * 50
    vals = [1.0] * 50 + [0.0] * 50
    out = R.cluster_bootstrap_mean(groups, vals, 500, 0.95, 0)
    assert out["estimate"] == pytest.approx(0.5)
    assert out["low"] == pytest.approx(0.0) and out["high"] == pytest.approx(1.0)
    assert out["n_problems"] == 2 and out["n_rows"] == 100


def test_cluster_bootstrap_mean_is_row_weighted():
    out = R.cluster_bootstrap_mean(["a"] * 3 + ["b"], [1, 1, 1, 0], 200, 0.95, 0)
    assert out["estimate"] == pytest.approx(0.75)


def test_cluster_bootstrap_mean_empty():
    out = R.cluster_bootstrap_mean([], [], 100, 0.95, 0)
    assert out["estimate"] is None and out["n_rows"] == 0


def test_paired_diff_keeps_pairs_inside_their_problem():
    groups = [g for g in "abcdefghij" for _ in range(5)]
    a = [1.0] * 50
    b = [0.0] * 50
    out = R.cluster_bootstrap_paired_diff(groups, a, b, 300, 0.95, 0)
    assert out["estimate"] == out["low"] == out["high"] == pytest.approx(1.0)
    assert "paired" in out["estimand"]


@pytest.mark.parametrize("lo,slo,decodes,leak", [
    (0.05, -0.02, True, False),
    (0.0, -0.02, False, False),       # lower bound must be strictly above zero
    (-0.01, 0.03, False, True),
    (None, None, False, False),
])
def test_m0_verdict_uses_the_lower_bounds(lo, slo, decodes, leak):
    v = R.m0_verdict({"low": lo}, {"low": slo})
    assert v == {"decodes": decodes, "leakage_detected": leak}


# --- annotation sample -------------------------------------------------------

def test_annotation_sample_is_stratified_seeded_and_records_its_design():
    rows = ([{"asserts": {"answer_stated": 1, "domain": "gsm8k"}}] * 60
            + [{"asserts": {"domain": "math-algebra"}}] * 80
            + [{"asserts": {}}] * 30                     # smaller than its quota
            + [{"asserts": {"final_word": "x"}}] * 40)   # counts as neither scored attribute
    pick, design = R.draw_annotation_sample(rows, seed=1)
    assert len(pick) == len(set(pick)) == 150
    assert design["strata"]["rule_answer_stated"] == {"quota": 50, "population": 60, "drawn": 50}
    assert design["strata"]["rule_domain_only"] == {"quota": 50, "population": 80, "drawn": 50}
    assert design["strata"]["rule_neither"] == {"quota": 50, "population": 70, "drawn": 50}
    assert design["rule_asserting_in_sample"] == {"answer_stated": 50, "domain": 100}
    assert R.draw_annotation_sample(rows, seed=1)[0] == pick


def test_annotation_sample_records_a_shortfall_instead_of_refilling():
    rows = [{"asserts": {"answer_stated": 0}}] * 10 + [{"asserts": {}}] * 100
    pick, design = R.draw_annotation_sample(rows, seed=0)
    assert design["strata"]["rule_answer_stated"]["drawn"] == 10
    assert design["strata"]["rule_domain_only"]["drawn"] == 0
    assert design["strata"]["rule_neither"]["drawn"] == 50
    assert len(pick) == 60


def test_mark_claim_wraps_the_one_occurrence():
    e = 'Structured problem. Final token "time." closes a clause.'
    assert R.mark_claim(e, 'Final token "time."') == \
        'Structured problem. [[Final token "time."]] closes a clause.'
    with pytest.raises(ValueError):
        R.mark_claim("a. a.", "a.")                 # ambiguous: two occurrences
    with pytest.raises(ValueError):
        R.mark_claim("abc", "xyz")                  # absent


def write_sheet(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(R.SHEET_COLUMNS)
        for r in rows:
            w.writerow(r)


def test_read_annotations_parses_codes(tmp_path):
    p = tmp_path / "s.csv"
    write_sheet(p, [[3, "c", "[[c]]", "1", "gsm8k", ""],
                    [7, "d", "[[d]]", "NA", "NA", "unclear"],
                    [9, "e", "[[e]]", "0", "math-counting_and_probability", ""]])
    assert R.read_annotations(p) == [
        {"row_id": 3, "answer_stated": 1, "domain": "gsm8k"},
        {"row_id": 7, "answer_stated": None, "domain": None},
        {"row_id": 9, "answer_stated": 0, "domain": "math-counting_and_probability"}]


@pytest.mark.parametrize("a,d", [("", "gsm8k"), ("1", ""), ("yes", "gsm8k"),
                                 ("1", "algebra"), ("1", "GSM8K"), (" ", "NA")])
def test_read_annotations_rejects_empty_or_invalid_cells(tmp_path, a, d):
    # An empty cell must never be read as "no assertion" - that is NA.
    p = tmp_path / "s.csv"
    write_sheet(p, [[1, "c", "[[c]]", a, d, ""]])
    with pytest.raises(ValueError):
        R.read_annotations(p)


def test_read_annotations_rejects_old_schema_and_duplicates(tmp_path):
    import csv
    old = tmp_path / "v2.csv"
    with open(old, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["row_id", "claim_text", "answer_stated (1/0/blank)",
                                   "domain (gsm8k/...)"], [1, "c", "", ""]])
    with pytest.raises(ValueError, match="schema"):
        R.read_annotations(old)
    dup = tmp_path / "dup.csv"
    write_sheet(dup, [[1, "c", "[[c]]", "1", "NA", ""], [1, "c", "[[c]]", "1", "NA", ""]])
    with pytest.raises(ValueError, match="duplicate"):
        R.read_annotations(dup)


def test_sheet_schema_has_no_rule_or_truth_columns():
    banned = ("asserts", "rule", "stratum", "truth", "window", "problem", "gold", "dependent")
    assert not [c for c in R.SHEET_COLUMNS if any(b in c for b in banned)]
    assert R.CODES["answer_stated"] == ("1", "0", "NA")
    assert set(R.CODES["domain"]) == set(R.DOMAINS) | {"NA"}


# --- what the current shuffled-label control does (facts, not validity) ------

def test_current_shuffle_keeps_labels_tied_to_relative_position():
    # A label that is a function of relative position ("is the last boundary")
    # stays a function of the *receiver's* relative position after the
    # permutation: the control moves labels between problems but does not
    # break the label-position association.
    lengths = [5, 8, 11, 6, 14, 9, 7, 12]
    groups, y = [], []
    for k, n in enumerate(lengths):
        groups += [f"g{k}"] * n
        y += [0] * (n - 1) + [1]
    out = R.permute_across_problems(y, groups, seed=2)
    for k, n in enumerate(lengths):
        seq = [v for v, h in zip(out, groups) if h == f"g{k}"]
        assert seq[-1] == 1                                   # last stays stated
        assert seq[0] == 0                                    # first stays not stated
        assert seq == sorted(seq)                             # only a late run of 1s


def test_current_shuffle_inflates_a_single_final_label_on_longer_receivers():
    # The mechanism behind 8.9% -> 10.3% on main: one stated boundary per
    # donor becomes several on a longer receiver.
    assert R.resample_by_position([0, 0, 1], 9).count(1) == 3
    assert R.resample_by_position([0] * 8 + [1], 3).count(1) == 1


# --- Decision 1 (option D): per-attribute shuffles and the position comparator

def final_only_labels(lengths):
    groups, y = [], []
    for k, n in enumerate(lengths):
        groups += [f"g{k:03d}"] * n
        y += [0] * (n - 1) + [1]
    return y, groups


def test_within_problem_shuffle_keeps_each_problems_labels_exactly():
    y, groups = final_only_labels([5, 8, 11, 6, 14, 9])
    out, info = R.within_problem_shuffle(y, groups, seed=0)
    for g in set(groups):
        assert sorted(v for v, h in zip(out, groups) if h == g) == \
            sorted(v for v, h in zip(y, groups) if h == g)
    assert R.class_shares(out) == R.class_shares(y)        # prevalence exact
    assert info["scope"] == "within_problem"
    assert info["n_problems_with_mixed_labels"] == 6


def test_within_problem_shuffle_breaks_the_label_position_link():
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(1)
    y, groups = final_only_labels([int(n) for n in rng.integers(5, 25, 200)])
    out, _ = R.within_problem_shuffle(y, groups, seed=3)
    n = {h: groups.count(h) for h in set(groups)}
    seen, rel = {}, []
    for h in groups:
        seen[h] = seen.get(h, -1) + 1
        rel.append(seen[h] / (n[h] - 1))
    rel, out, y = np.array(rel), np.array(out), np.array(y)

    def auc(lbl):
        return (mannwhitneyu(rel[lbl == 1], rel[lbl == 0]).statistic
                / ((lbl == 1).sum() * (lbl == 0).sum()))

    assert auc(y) == pytest.approx(1.0)                    # final-only labels
    assert 0.4 < auc(out) < 0.6                            # link removed by the shuffle


def test_within_problem_shuffle_reports_a_degenerate_problem():
    out, info = R.within_problem_shuffle([0, 0, 0, 1, 0], ["a", "a", "a", "b", "b"], seed=0)
    assert out[:3] == [0, 0, 0]                            # nothing to move in "a"
    assert info["n_problems_with_mixed_labels"] == 1


def test_control_shuffle_routes_each_attribute_to_its_registered_form():
    assert set(R.SHUFFLE_SCOPE) == set(R.EVALUATED)
    y, groups = final_only_labels([4, 6, 5])
    _, info = R.control_shuffle("answer_stated", y, groups, seed=0)
    assert info["scope"] == "within_problem"
    dom = ["gsm8k"] * 4 + ["math-algebra"] * 6 + ["math-counting_and_probability"] * 5
    out, info = R.control_shuffle("domain", dom, groups, seed=0)
    assert info["scope"] == "across_problems"
    for g in set(groups):                                  # one donor value per problem
        assert len({v for v, h in zip(out, groups) if h == g}) == 1
    assert info["n_labels_changed"] == 15                  # no problem keeps its own


def test_boundary_position_features():
    t = {"chunks": [{"prefix_tokens": 10}, {"prefix_tokens": 30}, {"prefix_tokens": 60}]}
    first, last = R.boundary_position(t, 0), R.boundary_position(t, 2)
    assert first["relative_position"] == 0.0 and last["relative_position"] == 1.0
    assert first["is_final_boundary"] == 0.0 and last["is_final_boundary"] == 1.0
    assert last["boundary_index"] == 2.0
    assert last["log_prefix_tokens"] == pytest.approx(np.log1p(60))
    assert R.boundary_position({"chunks": [{"prefix_tokens": 5}]}, 0)["relative_position"] == 1.0
    m = R.position_matrix([first, last])
    assert m.shape == (2, len(R.POSITION_FEATURES))


def test_position_comparator_sees_no_activation():
    # The comparator's inputs are these four position features and nothing else.
    assert R.POSITION_FEATURES == ("relative_position", "is_final_boundary",
                                   "boundary_index", "log_prefix_tokens")


@pytest.mark.parametrize("lo,beats", [(0.01, True), (0.0, False), (-0.02, False), (None, False)])
def test_position_verdict_needs_a_lower_bound_above_zero(lo, beats):
    assert R.position_verdict({"low": lo}) == {"beats_position_model": beats}


@pytest.mark.parametrize("m0,pos,start", [
    ({"decodes": True, "leakage_detected": True}, {"beats_position_model": True}, "invalid"),
    ({"decodes": False, "leakage_detected": False}, {"beats_position_model": True}, "inconclusive"),
    ({"decodes": True, "leakage_detected": False}, {"beats_position_model": False},
     "decodable, but not distinguishable from boundary position"),
    ({"decodes": True, "leakage_detected": False}, {"beats_position_model": True},
     "decodable beyond the position features"),
])
def test_answer_stated_reading(m0, pos, start):
    reading = R.answer_stated_reading(m0, pos)
    assert reading.startswith(start)
    assert "semantic understanding" in reading or not reading.startswith("decodable beyond")


def test_not_beating_the_comparator_is_not_read_as_absence_of_information():
    reading = R.answer_stated_reading({"decodes": True, "leakage_detected": False},
                                      {"beats_position_model": False})
    assert "did not establish predictive value beyond the four position features" in reading
    assert "does not show the activations lack" in reading


def test_position_verdict_reads_only_the_accuracy_interval():
    # Brier is secondary: a favourable Brier difference cannot flip the verdict.
    v = R.position_verdict({"low": -0.01, "high": 0.02,
                            "secondary_brier_position_minus_probe": {"low": 0.5}})
    assert v == {"beats_position_model": False}


def test_evaluation_never_reads_the_ai_annotation_or_computes_kappa():
    import inspect
    src = inspect.getsource(R.evaluate)
    assert "ai_claude" not in src and "jsonl" not in src
    assert "cohen_kappa" not in src and "read_annotations" not in src


def test_positional_predictability_of_a_final_only_label():
    y, groups = final_only_labels([5, 8, 11])
    pos = []
    for k, n in enumerate([5, 8, 11]):
        pos += [{"relative_position": j / (n - 1), "is_final_boundary": float(j == n - 1)}
                for j in range(n)]
    out = R.positional_predictability(y, pos)
    assert out["auc_relative_position"] == pytest.approx(1.0)
    assert out["rate_at_final_boundary"] == 1.0 and out["rate_at_other_boundaries"] == 0.0


def test_evaluation_no_longer_depends_on_an_annotation_agreement():
    import inspect
    src = inspect.getsource(R.evaluate)
    assert "annotation_agreement" not in src
    assert "human_validation" in src and "not performed" in src


def test_an_ai_annotation_jsonl_cannot_enter_the_agreement(tmp_path):
    import json
    p = tmp_path / "annotation_sheet_v3.ai_claude_not_independent.jsonl"
    p.write_text(json.dumps({"_provenance": {"annotator": "AI"}}) + "\n"
                 + json.dumps({"row_id": 1, "answer_stated": "1", "domain": "NA"}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError):
        R.read_annotations(p)


def test_annotation_counts():
    rule = [1, None, 0, None, 1]
    human = [1, 1, None, None, 0]
    assert R.annotation_counts(rule, human) == {
        "rows": 5, "rule_asserts": 3, "annotator_asserts": 3, "either_asserts": 4,
        "both_assert": 2, "both_assert_same_value": 1}


def test_mismatched_vector_matches_relative_position():
    acts = np.arange(10, dtype=float)[:, None] * np.ones((1, 3))
    assert R.mismatched_vector(acts, 0.0)[0] == 0
    assert R.mismatched_vector(acts, 1.0)[0] == 9
    assert R.mismatched_vector(acts, 0.5)[0] in (4, 5)
    assert R.mismatched_vector(acts, 7.0)[0] == 9


# --- evaluation helpers -----------------------------------------------------

def test_cohen_kappa():
    assert R.cohen_kappa([1, 0, 1, None], [1, 0, 1, None]) == pytest.approx(1.0)
    assert R.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0
    assert np.isnan(R.cohen_kappa([], []))


def test_cluster_bootstrap_rate_diff_point_and_bounds():
    groups = [g for g in "abcdefgh" for _ in range(4)]
    vals = [1.0, 1.0, 0.0, 0.0] * 8
    mask = [True, True, False, False] * 8
    out = R.cluster_bootstrap_rate_diff(groups, vals, mask, 300, 0.95, 0)
    assert out["estimate"] == pytest.approx(1.0)
    assert out["low"] == pytest.approx(1.0) and out["high"] == pytest.approx(1.0)
    assert out["n_problems"] == 8


def test_probe_support_and_confirmation():
    from nlaast.config import from_mapping
    cfg = from_mapping({})
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (30, 4)), rng.normal(2, 1, (30, 4))])
    y = [0] * 30 + [1] * 30
    probe = R.fit_probe(cfg, X, y)
    hi = np.full(4, 3.0)
    assert R.confirms(probe, hi, 1) and not R.confirms(probe, hi, 0)
    assert R.support(probe, hi, 1) > 0.9
    assert R.support(probe, hi, "unseen") == 0.0


def test_evaluation_refuses_without_an_approved_amendment(tmp_path):
    p = tmp_path / "A.md"
    p.write_text("**Status: PROPOSED**\n", encoding="utf-8")
    assert not R.amendment_approved(p)
    p.write_text("# A1\n\n**Status: APPROVED 2026-10-18 by the researcher**\n", encoding="utf-8")
    assert R.amendment_approved(p)
    assert not R.amendment_approved(tmp_path / "missing.md")
