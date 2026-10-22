"""The registered safety outcome (PROJECT_PLAN §5) beside the historical one (D68, D69).

§5 scores a stop on the answer *forced* from the prefix; the registered run
scored the answer *parsed* from it. These tests pin both, and pin what happens
where the forced answer was never measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nlaast.baselines.stopping import (
    OUTCOME_FORCED,
    OUTCOME_PARSED,
    OutcomeUnknown,
    decisions_at,
    forced_answer_sequence,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from force_all_boundaries import blocks, select_boundaries  # noqa: E402
from score_baselines import build_rows  # noqa: E402

# Chunk 0 sets up the problem; chunk 1 computes but writes no answer; chunk 2
# writes it. A forced answer from chunk 1 already gives 12.
TEXT = "We have 3 boxes of 4. Multiply them out. So the answer is 12."
CHUNKS = [
    {"index": 0, "text": "We have 3 boxes of 4.", "char_start": 0, "char_end": 21,
     "token_end": 6, "prefix_tokens": 6},
    {"index": 1, "text": " Multiply them out.", "char_start": 21, "char_end": 40,
     "token_end": 10, "prefix_tokens": 10},
    {"index": 2, "text": " So the answer is 12.", "char_start": 40, "char_end": 61,
     "token_end": 16, "prefix_tokens": 16},
]


def _trace(boundaries, gold="12", final="12"):
    return {"problem_id": "p", "trace": TEXT, "chunks": CHUNKS, "gold": gold,
            "final_answer": final, "n_tokens": 20, "boundaries": boundaries,
            "dataset": "gsm8k", "split": "eval"}


def _b(i, evaluated, forced=None):
    return {"index": i, "evaluated": evaluated, "forced_answer": forced,
            "prefix_tokens": CHUNKS[i]["prefix_tokens"]}


# --- which forced answer is used -------------------------------------------


def test_stored_answers_are_used_and_a_stored_none_is_a_measured_answer():
    t = _trace([_b(0, True, None), _b(1, True, "12"), _b(2, True, "12")])
    answers, known = forced_answer_sequence(t)
    assert answers == [None, "12", "12"] and known == [True, True, True]


def test_unevaluated_boundaries_come_from_the_completion_or_are_unknown():
    t = _trace([_b(0, False), _b(1, False), _b(2, True, "12")])
    answers, known = forced_answer_sequence(t, {("p", 1): "12"})
    assert answers == [None, "12", "12"]
    assert known == [False, True, True]


def test_prefer_completed_overrides_stored_only_where_completed_exists():
    t = _trace([_b(0, True, "3"), _b(1, True, "12"), _b(2, True, "12")])
    answers, known = forced_answer_sequence(t, {("p", 0): "4"}, prefer="completed")
    assert answers == ["4", "12", "12"] and all(known)
    with pytest.raises(ValueError):
        forced_answer_sequence(t, prefer="newest")


# --- forced versus parsed --------------------------------------------------


def test_parsed_outcome_is_still_the_default_and_unchanged():
    t = _trace([_b(0, True, "12"), _b(1, True, "12"), _b(2, True, "12")])
    row = build_rows([t])["p"]
    assert row["outcome"] == OUTCOME_PARSED
    # chunk 1's text has written no answer: parsing finds "4" (the last number)
    assert row["answers"][1] != "12" and row["answers"][2] == "12"


def test_forced_and_parsed_disagree_where_the_answer_is_recoverable_but_unwritten():
    t = _trace([_b(0, True, "3"), _b(1, True, "12"), _b(2, True, "12")])
    forced = build_rows([t], outcome=OUTCOME_FORCED)["p"]
    parsed = build_rows([t])["p"]
    stop_at_1 = {**forced, "scores": [0.0, 1.0, 1.0]}
    (d_forced,) = decisions_at([stop_at_1], 0.5)
    (d_parsed,) = decisions_at([{**parsed, "scores": [0.0, 1.0, 1.0]}], 0.5)
    assert d_forced.boundary == d_parsed.boundary == 1
    assert d_forced.safe is True and d_parsed.safe is False


def test_unknown_outcome_raises_instead_of_counting_as_incorrect():
    t = _trace([_b(0, False), _b(1, True, "12"), _b(2, True, "12")])
    row = {**build_rows([t], outcome=OUTCOME_FORCED)["p"], "scores": [1.0, 1.0, 1.0]}
    with pytest.raises(OutcomeUnknown):
        decisions_at([row], 0.5)


def test_an_unknown_boundary_the_rule_never_stops_at_is_harmless():
    t = _trace([_b(0, False), _b(1, True, "12"), _b(2, True, "12")])
    row = build_rows([t], outcome=OUTCOME_FORCED)["p"]
    (d,) = decisions_at([{**row, "scores": [0.0, 1.0, 1.0]}], 0.5)
    assert d.boundary == 1 and d.safe
    (never,) = decisions_at([{**row, "scores": [0.0, 0.0, 0.0]}], 0.5)
    assert never.boundary is None and never.safe


@pytest.mark.parametrize("final, safe", [("12", False), ("13", True)])
def test_a_forced_completion_that_names_no_answer_is_safe_only_if_the_final_is_wrong(final, safe):
    # §5: safe iff (forced == final and final correct) or both incorrect. A
    # completion that parses to nothing is not the gold answer.
    t = _trace([_b(0, True, None), _b(1, True, "12"), _b(2, True, final)], final=final)
    row = {**build_rows([t], outcome=OUTCOME_FORCED)["p"], "scores": [1.0, 1.0, 1.0]}
    (d,) = decisions_at([row], 0.5)
    assert d.boundary == 0 and d.safe is safe


def test_a_wrong_final_answer_makes_a_wrong_forced_stop_safe_and_a_right_one_unsafe():
    t = _trace([_b(0, True, "7"), _b(1, True, "12"), _b(2, True, "13")], final="13")
    row = build_rows([t], outcome=OUTCOME_FORCED)["p"]
    (d0,) = decisions_at([{**row, "scores": [1.0, 0.0, 0.0]}], 0.5)
    (d1,) = decisions_at([{**row, "scores": [0.0, 1.0, 0.0]}], 0.5)
    assert d0.safe is True       # both wrong
    assert d1.safe is False      # stopping would have fixed an answer the trace ruins


def test_tokens_used_are_the_prefix_through_the_stopping_chunk():
    t = _trace([_b(0, True, "12"), _b(1, True, "12"), _b(2, True, "12")])
    row = build_rows([t], outcome=OUTCOME_FORCED)["p"]
    (d,) = decisions_at([{**row, "scores": [0.0, 1.0, 1.0]}], 0.5)
    assert d.tokens_used == 10 and d.tokens_saved_fraction == pytest.approx(0.5)


def test_unknown_outcome_name_is_rejected():
    with pytest.raises(ValueError):
        build_rows([_trace([_b(0, True), _b(1, True), _b(2, True)])], outcome="judged")


# --- the completion script's selection -------------------------------------


def test_selection_covers_every_unevaluated_boundary_and_samples_stored_ones():
    traces = [
        {"problem_id": "a", "ok": True, "boundaries": [_b(0, False), _b(1, True, "1"), _b(2, True, "1")]},
        {"problem_id": "b", "ok": True, "boundaries": [_b(0, False), _b(1, False), _b(2, True, "2")]},
        {"problem_id": "c", "ok": False, "boundaries": [_b(0, False)]},
    ]
    missing, repro = select_boundaries(traces, 2, seed=0)
    assert missing == {"a": [0], "b": [0, 1]}
    stored = {("a", 1), ("a", 2), ("b", 2)}
    drawn = {(p, i) for p, v in repro.items() for i in v}
    assert len(drawn) == 2 and drawn <= stored
    assert select_boundaries(traces, 2, seed=0) == (missing, repro)


def test_cross_problem_batches_sort_by_length_and_cover_everything():
    from force_all_boundaries import cross_problem_batches
    items = [("a", 0, "missing"), ("b", 3, "missing"), ("a", 2, "repro"), ("c", 1, "missing")]
    lengths = {("a", 0): 50, ("b", 3): 10, ("a", 2): 30, ("c", 1): 30}
    out = cross_problem_batches(items, lengths, 3)
    assert out == [[("b", 3, "missing"), ("a", 2, "repro"), ("c", 1, "missing")],
                   [("a", 0, "missing")]]
    assert sorted(x for b in out for x in b) == sorted(items)


def test_blocks_are_descending_and_at_most_four():
    assert blocks([0, 1, 2, 3, 4, 5]) == [[5, 4, 3, 2], [1, 0]]
    assert blocks([3]) == [[3]]
