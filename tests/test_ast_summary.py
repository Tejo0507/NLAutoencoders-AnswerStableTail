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

from detect_answer_stable_tail import run_sweep, summarise  # noqa: E402


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
