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

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from detect_answer_stable_tail import summarise  # noqa: E402


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
