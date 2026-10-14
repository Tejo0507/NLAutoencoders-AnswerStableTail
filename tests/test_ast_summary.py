"""The `ast` stage's own accounting, and why `tail_rate` needs a caveat.

A tail is detected whenever *some* suffix of boundaries satisfies the three
criteria. At the **final** boundary the prefix is the whole trace, so forcing
an answer from it reproduces the final answer and continuations from it restate
the same answer: both criteria are close to tautological there. A tail
therefore almost always exists, and `tail_rate` sits near 1 whenever the final
answer parses and generation was not truncated.

That was visible on the first real traces — the final boundary qualified on
13 of 13 problems — and it means `tail_rate` on its own invites a reader to
conclude something the statistic cannot support. The summary therefore reports
how often the final boundary qualifies, how many tails are nothing but that
boundary, and the non-trivial rate alongside it.

These tests pin that accounting, including the degenerate case where every tail
is trivial.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from detect_answer_stable_tail import (  # noqa: E402
    evidence_from_row,
    forced_was_cut_off,
    run_sweep,
    summarise,
)


def row(pid, n_chunks, tail_start, *, status="ok", last_qualifies=True,
        total_tokens=100, convergence_start=None, stated_at="unset"):
    """One `ast.jsonl` row, with just the fields `summarise` reads."""
    evidence = [{"index": i, "forced_matches_final": False,
                 "continuations_unanimous": False} for i in range(n_chunks)]
    if last_qualifies and evidence:
        evidence[-1] = {"index": n_chunks - 1, "forced_matches_final": True,
                        "continuations_unanimous": True}
    frac = 0.0 if tail_start is None else (n_chunks - tail_start) / n_chunks
    if stated_at == "unset":
        stated_at = None if tail_start is None else tail_start
    gap = (None if (stated_at is None or tail_start is None)
           else stated_at - tail_start)
    return {
        "problem_id": pid, "status": status, "n_chunks": n_chunks,
        "tail_start": tail_start, "tail_fraction": frac,
        "tail_tokens": 0 if tail_start is None else total_tokens // 2,
        "total_tokens": total_tokens, "convergence_start": convergence_start,
        "final_correct": True, "evidence": evidence,
        "answer_stated_at_strict": stated_at,
        "chunks_from_tail_start_to_statement": gap,
        "tail_starts_before_answer_stated": (
            None if gap is None else bool(gap > 0)),
    }


class TestTrivialTails:
    def test_a_tail_of_only_the_final_chunk_is_counted_as_trivial(self):
        s = summarise([row("p0", 8, 7), row("p1", 8, 3)])
        assert s["n_with_tail"] == 2
        assert s["tail_rate"] == 1.0
        assert s["n_trivial_tail"] == 1
        assert s["trivial_tail_share"] == 0.5
        assert s["nontrivial_tail_rate"] == 0.5

    def test_all_trivial_is_reported_as_no_redundancy_found(self):
        """The case a bare tail_rate of 1.0 would misrepresent completely."""
        s = summarise([row(f"p{i}", 6, 5) for i in range(5)])
        assert s["tail_rate"] == 1.0
        assert s["nontrivial_tail_rate"] == 0.0
        assert s["trivial_tail_share"] == 1.0

    def test_problems_without_a_tail_are_not_counted_either_way(self):
        s = summarise([row("p0", 8, None, status="no_stable_point"),
                       row("p1", 8, 2)])
        assert s["n_with_tail"] == 1
        assert s["tail_rate"] == 0.5
        assert s["n_trivial_tail"] == 0
        assert s["nontrivial_tail_rate"] == 0.5


class TestFinalBoundary:
    def test_the_rate_at_which_the_final_boundary_qualifies_is_reported(self):
        s = summarise([row("p0", 8, 3, last_qualifies=True),
                       row("p1", 8, None, status="no_stable_point",
                           last_qualifies=False)])
        assert s["final_boundary_qualifies_rate"] == 0.5

    def test_the_caveat_is_carried_in_the_summary_not_only_in_prose(self):
        s = summarise([row("p0", 8, 3)])
        assert "by construction" in s["tail_rate_caveat"]
        assert "tail_fraction" in s["tail_rate_caveat"]


class TestDistribution:
    def test_the_spread_of_tail_fraction_is_reported_not_only_the_mean(self):
        """A mean alone cannot distinguish a corpus of uniform mid-length tails
        from one of trivial and whole-trace tails."""
        s = summarise([row("p0", 10, 9), row("p1", 10, 5), row("p2", 10, 1)])
        assert s["min_tail_fraction"] < s["mean_tail_fraction"] < s["max_tail_fraction"]
        assert s["min_tail_fraction"] == 0.1
        assert s["max_tail_fraction"] == 0.9

    def test_an_empty_corpus_does_not_divide_by_zero(self):
        s = summarise([])
        assert s["n"] == 0
        assert s["tail_rate"] == 0.0
        assert s["nontrivial_tail_rate"] == 0.0
        assert s["final_boundary_qualifies_rate"] == 0.0

    def test_the_median_is_the_median_at_an_even_count(self):
        """The middle element of a sorted list is not the median.

        Four tailed problems at fractions 0.1, 0.2, 0.8, 0.9 have a median of
        0.5. Indexing the sorted list at ``len // 2`` returns 0.8 - the upper
        of the two central values - which is a different and larger statistic
        reported under the median's name. The pilot's 21 tailed problems made
        the two agree; any even count makes them differ.
        """
        s = summarise([row("p0", 10, 9), row("p1", 10, 8),
                       row("p2", 10, 2), row("p3", 10, 1)])
        assert s["n_with_tail"] == 4
        assert s["median_tail_fraction"] == pytest.approx(0.5)
        assert s["median_tail_fraction"] < s["max_tail_fraction"]

    def test_the_median_still_works_at_an_odd_count(self):
        s = summarise([row("p0", 10, 9), row("p1", 10, 5), row("p2", 10, 1)])
        assert s["median_tail_fraction"] == pytest.approx(0.5)


class TestTailTokenShareDenominator:
    """The tail's token share has two denominators and they are not close.

    `tail_token_share` divides by the tokens generated in traces that *have* a
    tail, which is the right denominator for "how much of a tailed trace is
    tail". It is not the share of all generated tokens, and the traces without
    a tail are not a random sample of the corpus: they are the ones that hit
    the generation cap, so they are the longest. Excluding them therefore
    raises the ratio, and the two numbers were reported under one name.
    """

    def test_both_denominators_are_reported(self):
        # Two tailed traces of 100 tokens, half of each a tail, and one
        # 800-token trace with no tail at all.
        rows = [row("p0", 10, 5, total_tokens=100),
                row("p1", 10, 5, total_tokens=100),
                row("p2", 10, None, status="no_stable_point", total_tokens=800)]
        s = summarise(rows)
        assert s["total_tail_tokens"] == 100
        assert s["total_tokens"] == 200
        assert s["total_tokens_all_traces"] == 1000
        assert s["tail_token_share"] == pytest.approx(0.5)
        assert s["tail_token_share_all_traces"] == pytest.approx(0.1)

    def test_the_narrow_denominator_never_understates_the_share(self):
        rows = [row("p0", 10, 5, total_tokens=100),
                row("p1", 10, None, status="no_stable_point", total_tokens=500)]
        s = summarise(rows)
        assert s["tail_token_share"] >= s["tail_token_share_all_traces"]
        assert "tail_token_share_all_traces" in s["tail_token_share_denominator"]

    def test_the_two_agree_when_every_trace_has_a_tail(self):
        rows = [row("p0", 10, 5, total_tokens=100),
                row("p1", 10, 5, total_tokens=100)]
        s = summarise(rows)
        assert s["tail_token_share"] == pytest.approx(s["tail_token_share_all_traces"])


class TestDeterminacyVersusStatement:
    """The central interpretive caveat, measured.

    The criteria ask whether the answer is *determined* from a prefix, not
    whether the trace has said it: forcing can finish the arithmetic inside
    the forced answer. On the first real traces the tail began before the
    answer was stated on every single problem, by up to five chunks. That is
    correct for a stopping rule and wrong for reading the window as
    post-answer verification, so the gap is reported rather than implied.
    """

    def test_a_positive_gap_is_counted_and_averaged(self):
        rows = [row("p0", 10, 3, stated_at=6), row("p1", 10, 5, stated_at=6)]
        d = summarise(rows)["determinacy_vs_statement"]
        assert d["n"] == 2
        assert d["tail_starts_before_answer_stated"] == 1.0
        assert d["mean_chunks_before_statement"] == 2.0
        assert d["max_chunks_before_statement"] == 3

    def test_a_tail_starting_at_the_statement_is_not_counted_as_before(self):
        d = summarise([row("p0", 10, 6, stated_at=6)])["determinacy_vs_statement"]
        assert d["tail_starts_before_answer_stated"] == 0.0
        assert d["mean_chunks_before_statement"] == 0.0

    def test_problems_whose_answer_is_never_stated_are_counted_separately(self):
        rows = [row("p0", 10, 3, stated_at=5), row("p1", 10, 4, stated_at=None)]
        d = summarise(rows)["determinacy_vs_statement"]
        assert d["n"] == 1
        assert d["n_unstated"] == 1

    def test_the_block_is_absent_when_nothing_can_be_compared(self):
        assert "determinacy_vs_statement" not in summarise(
            [row("p0", 10, None, status="no_stable_point")])


class TestForcedBudget:
    """Criterion 1 can fail because 24 tokens ran out, not because the answer
    was undetermined.

    Observed on a real trace: forcing from a mid-derivation prefix produced
    "the answer is: Weight of sweet potatoes = 2 x 5 pounds = ", which the
    permissive extractor read as 5, so the boundary failed - while all three
    resampled continuations from the same prefix reached the correct answer.
    """

    def test_the_recorded_flag_is_preferred_when_present(self):
        assert forced_was_cut_off({"forced_finished": False}) is True
        assert forced_was_cut_off({"forced_finished": True}) is False

    def test_the_flag_wins_over_the_text_heuristic(self):
        """A trace that carries the exact signal must not be second-guessed."""
        b = {"forced_finished": True, "forced_text": "... = 2 \\times 5 = "}
        assert forced_was_cut_off(b) is False

    def test_older_traces_fall_back_to_the_text(self):
        assert forced_was_cut_off({"forced_text": "the answer is 7 pounds."}) is False
        assert forced_was_cut_off({"forced_text": "... pounds} = "}) is True

    def test_no_text_and_no_flag_is_unknown_not_false(self):
        assert forced_was_cut_off({}) is None
        assert forced_was_cut_off({"forced_text": "   "}) is None

    def test_relaxing_the_budget_only_rescues_unanimous_boundaries(self):
        row = {"boundaries": [
            # cut off, unanimous -> rescued
            {"index": 0, "forced_matches_final": False, "forced_finished": False,
             "continuation_answers": ["7", "7"], "continuations_matching": 2,
             "prefix_tokens": 4},
            # cut off, not unanimous -> left alone
            {"index": 1, "forced_matches_final": False, "forced_finished": False,
             "continuation_answers": ["7", "3"], "continuations_matching": 1,
             "prefix_tokens": 8},
            # finished and disagreed -> left alone, this is real evidence
            {"index": 2, "forced_matches_final": False, "forced_finished": True,
             "continuation_answers": ["7", "7"], "continuations_matching": 2,
             "prefix_tokens": 12},
        ]}
        strict = evidence_from_row(row)
        assert [e.forced_matches_final for e in strict] == [False, False, False]
        relaxed = evidence_from_row(row, relax_forced_budget=True)
        assert [e.forced_matches_final for e in relaxed] == [True, False, False]

    def test_relaxing_never_overturns_a_satisfied_criterion(self):
        row = {"boundaries": [
            {"index": 0, "forced_matches_final": True, "forced_finished": True,
             "continuation_answers": ["7"], "continuations_matching": 1,
             "prefix_tokens": 4},
        ]}
        assert evidence_from_row(row, relax_forced_budget=True)[0].forced_matches_final

    def test_the_summary_counts_blocking_cases_separately(self):
        rows = [{
            **row("p0", 4, 2),
            "n_boundaries_evaluated": 4,
            "n_forced_cut_off": 3,
            "n_forced_cut_off_blocking": 2,
        }]
        fab = summarise(rows)["forced_answer_budget"]
        assert fab["cut_off_rate"] == 0.75
        assert fab["blocking_rate"] == 0.5
        assert fab["n_problems_affected"] == 1


def sweep_trace(pid, n_chunks, stable_from, k, answer="42"):
    """A `traces.jsonl` row with enough structure for detection to run on it."""
    text = " ".join(f"Step {i} of the derivation here." for i in range(n_chunks - 1))
    text += f" The answer is {answer}."
    chunks, pos = [], 0
    for i in range(n_chunks):
        chunks.append({"index": i, "text": f"chunk {i}", "char_start": pos,
                       "char_end": pos + 10, "token_end": i * 4 + 3,
                       "prefix_tokens": (i + 1) * 4})
        pos += 10
    boundaries = []
    for i in range(n_chunks):
        stable = i >= stable_from
        boundaries.append({
            "index": i,
            "parsed_answer": answer if stable else None,
            "forced_answer": answer if stable else "0",
            "forced_matches_final": stable,
            "continuation_answers": [answer if stable else "0"] * k,
            "continuations_matching": k if stable else 0,
            "n_continuations": k,
            "prefix_tokens": (i + 1) * 4,
        })
    return {"id": pid, "problem_id": pid, "dataset": "gsm8k", "split": "eval",
            "level": None, "question": "q?", "gold": answer, "trace": text,
            "n_tokens": n_chunks * 4, "n_chunks": n_chunks, "chunks": chunks,
            "boundaries": boundaries, "final_answer": answer,
            "final_correct": True, "truncated": False, "ok": True}


class TestF5Sweep:
    """A sensitivity sweep must not report untested settings as stable.

    `k` is simulated by truncating the recorded continuations, which is exact
    downwards and impossible upwards: a requested k above the number generated
    silently reproduced the base result and was then reported as evidence of
    robustness. `convergence_window` is not an AST parameter at all - sweeping
    it leaves every tail untouched by construction, so an unchanged tail
    fraction across it is an identity, not a finding.
    """

    @staticmethod
    def _cfg(sweep_k):
        from nlaast import config as config_mod

        base = config_mod.load("pilot", []).to_dict()
        base["ast"]["k_continuations"] = 3
        base["robustness"]["ast_sweep_k"] = sweep_k
        return config_mod.from_mapping(base)

    @staticmethod
    def _log():
        import logging

        return logging.getLogger("test")

    def test_a_k_above_what_was_generated_is_not_applicable(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=3) for i in range(4)]
        sweep = run_sweep(self._cfg([3, 5]), traces, self._log())
        assert sweep["k=3"]["status"] == "ran"
        assert sweep["k=5"]["status"] == "not_applicable"
        assert "only 3 continuations" in sweep["k=5"]["reason"]
        assert "mean_tail_fraction" not in sweep["k=5"]

    def test_a_reachable_k_is_exercised(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=5) for i in range(4)]
        sweep = run_sweep(self._cfg([3, 5]), traces, self._log())
        assert sweep["k=3"]["status"] == sweep["k=5"]["status"] == "ran"

    def test_unanimity_is_always_swept(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=3) for i in range(4)]
        sweep = run_sweep(self._cfg([3]), traces, self._log())
        assert sweep["require_unanimous=False"]["status"] == "ran"

    def test_the_baseline_window_is_reported_separately_from_the_criteria(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=3) for i in range(4)]
        sweep = run_sweep(self._cfg([3]), traces, self._log())
        assert "convergence_window=2" not in sweep
        conv = sweep["baseline_convergence_window"]["settings"]
        assert set(conv) == {f"convergence_window={w}" for w in (1, 2, 3)}
        # Identical tails under every window - which is the point of the note.
        fracs = {c["mean_tail_fraction"] for c in conv.values()}
        assert len(fracs) == 1
        assert "not an AST parameter" in sweep["baseline_convergence_window"]["note"]

    def test_the_relaxed_budget_arm_is_present_and_marked_as_a_bound(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=3) for i in range(4)]
        arm = run_sweep(self._cfg([3]), traces, self._log())["forced_budget_relaxed"]
        assert arm["status"] == "ran"
        assert arm["is_upper_bound"] is True
        assert "not the pre-registered criterion" in arm["note"]
        assert "mean_tail_fraction" in arm

    def test_the_relaxed_arm_never_shortens_the_tail(self):
        """It only ever adds qualifying boundaries, so the tail can grow but
        not shrink - which is what makes the pair an interval."""
        traces = []
        for i in range(4):
            t = sweep_trace(f"p{i}", 8, 4, k=3)
            for b in t["boundaries"][2:4]:
                b["forced_matches_final"] = False
                b["forced_finished"] = False
                b["continuation_answers"] = ["42", "42", "42"]
                b["continuations_matching"] = 3
            traces.append(t)
        sweep = run_sweep(self._cfg([3]), traces, self._log())
        assert (sweep["forced_budget_relaxed"]["mean_tail_fraction"]
                >= sweep["k=3"]["mean_tail_fraction"])

    def test_the_meta_block_records_what_was_available(self):
        traces = [sweep_trace(f"p{i}", 8, 4, k=3) for i in range(4)]
        meta = run_sweep(self._cfg([3, 5]), traces, self._log())["_meta"]
        assert meta["max_continuations_recorded"] == 3
        assert meta["configured_k"] == 3
        assert meta["n_traces"] == 4


class TestConvergenceComparison:
    def test_the_agreement_rule_is_compared_where_both_fired(self):
        rows = [row("p0", 10, 5, convergence_start=3),
                row("p1", 10, 5, convergence_start=7),
                row("p2", 10, 5, convergence_start=5),
                row("p3", 10, 5, convergence_start=None)]
        c = summarise(rows)["convergence_vs_ast"]
        assert c["n"] == 3
        assert c["convergence_earlier"] == 1 / 3
        assert c["same"] == 1 / 3
        assert c["convergence_later"] == 1 / 3

    def test_the_signed_mean_cancels_and_is_not_the_gap(self):
        """One problem two chunks early and one two chunks late net to zero.

        `mean_gap_chunks` is signed, so on this corpus it is 0.0 - and not one
        of the three problems has a gap of zero. Printed beside "fires earlier
        on 33% of problems" it reads as "by 0 chunks", which is the opposite of
        what happened. The magnitude and the per-direction means are what a
        reader of that sentence needs.
        """
        rows = [row("p0", 10, 5, convergence_start=3),
                row("p1", 10, 5, convergence_start=7),
                row("p2", 10, 5, convergence_start=5)]
        c = summarise(rows)["convergence_vs_ast"]
        assert c["mean_gap_chunks"] == pytest.approx(0.0)
        assert c["mean_abs_gap_chunks"] == pytest.approx(4 / 3)
        assert c["mean_gap_when_earlier"] == pytest.approx(2.0)
        assert c["mean_gap_when_later"] == pytest.approx(2.0)

    def test_the_conditional_means_are_none_when_no_problem_falls_that_way(self):
        rows = [row("p0", 10, 5, convergence_start=3),
                row("p1", 10, 5, convergence_start=2)]
        c = summarise(rows)["convergence_vs_ast"]
        assert c["convergence_later"] == 0.0
        assert c["mean_gap_when_later"] is None
        assert c["mean_gap_when_earlier"] == pytest.approx(2.5)
        assert c["mean_gap_chunks"] == pytest.approx(c["mean_abs_gap_chunks"])

    def test_the_directions_are_distinguishable_from_the_summary_alone(self):
        """A corpus that fires uniformly late must not look like a cancelling one."""
        late = summarise([row("p0", 10, 5, convergence_start=7),
                          row("p1", 10, 5, convergence_start=7)])["convergence_vs_ast"]
        assert late["mean_gap_chunks"] == pytest.approx(-2.0)
        assert late["mean_abs_gap_chunks"] == pytest.approx(2.0)
        assert late["mean_gap_when_earlier"] is None
