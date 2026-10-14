"""The consistency checker, checked.

`scripts/check_run_consistency.py` exists because a run can be internally
reproducible and still not say what its code now says. The pilot is the case:
every number in `analysis/results.json` re-derives exactly from the artifacts,
and the probe comparison's stored p-value is `nan` where the current code gives
1.0 - which moves the Benjamini-Hochberg family from five evaluable tests to
six and every q-value with it.

A checker that misses that is worse than none, so these tests feed it runs with
each defect planted and assert it reports them. They build their own run
directories under `tmp_path`; nothing here reads or writes `runs/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_run_consistency as crc  # noqa: E402


def ast_row(pid, n_chunks, tail_start, *, total_tokens=100, convergence_start=None):
    evidence = [{"index": i, "forced_matches_final": False,
                 "continuations_unanimous": False} for i in range(n_chunks)]
    evidence[-1] = {"index": n_chunks - 1, "forced_matches_final": True,
                    "continuations_unanimous": True}
    return {
        "problem_id": pid, "status": "ok", "n_chunks": n_chunks,
        "tail_start": tail_start,
        "tail_fraction": (n_chunks - tail_start) / n_chunks,
        "tail_tokens": total_tokens // 2, "total_tokens": total_tokens,
        "convergence_start": convergence_start, "final_correct": True,
        "evidence": evidence, "answer_stated_at_strict": tail_start,
        "chunks_from_tail_start_to_statement": 0,
        "tail_starts_before_answer_stated": False,
        "n_boundaries_evaluated": n_chunks - 1,
        "n_forced_cut_off": 0, "n_forced_cut_off_blocking": 0,
    }


def mcnemar_test(name, *, n_problems, a_only, b_only, p, q, significant,
                 other="probe", interval=True):
    return {
        "name": name, "statistic": float(a_only), "p_value": p,
        "effect": 1.0, "effect_name": "odds_ratio_discordant",
        "interval": ({"estimate": 0.0, "low": -0.1, "high": 0.1,
                      "confidence": 0.95, "n": n_problems,
                      "method": "conditional_exact", "excludes_zero": False}
                     if interval else None),
        "n": n_problems, "q_value": q, "significant": significant,
        "notes": {"test": "mcnemar_exact", "n_problems": n_problems,
                  "n_discordant": a_only + b_only, "a_safe_b_not": a_only,
                  "b_safe_a_not": b_only, "safe_rate_nla": 0.1,
                  f"safe_rate_{other}": 0.1 if a_only == b_only else 0.3,
                  "direction": ("tied" if a_only == b_only
                                else "nla_readout worse"),
                  "unit_of_analysis": "problem"},
    }


def write_run(tmp_path: Path, *, ast_rows=None, results=None) -> Path:
    run = tmp_path / "run"
    (run / "analysis").mkdir(parents=True)
    if ast_rows is not None:
        (run / "ast").mkdir()
        (run / "ast" / "ast.jsonl").write_text(
            "\n".join(json.dumps(r) for r in ast_rows) + "\n", encoding="utf-8")
    (run / "analysis" / "results.json").write_text(
        json.dumps(results or {}), encoding="utf-8")
    return run


@pytest.fixture
def rep():
    return crc.Report()


class TestTheAnalysisAgainstItsArtifacts:
    def test_a_summary_that_does_not_re_derive_is_reported(self, tmp_path, rep):
        rows = [ast_row("p0", 10, 5), ast_row("p1", 10, 3)]
        run = write_run(tmp_path, ast_rows=rows,
                        results={"rq1": {"n": 2, "n_with_tail": 2,
                                         "mean_tail_fraction": 0.99}})
        crc.check_rq1(run, crc.read_json(run / "analysis/results.json"), rep)
        assert any("mean_tail_fraction" in f for f in rep.failures)
        assert not any("n_with_tail" in f for f in rep.failures)

    def test_a_summary_that_re_derives_passes(self, tmp_path, rep):
        rows = [ast_row("p0", 10, 5), ast_row("p1", 10, 3)]
        run = write_run(tmp_path, ast_rows=rows, results={})
        from detect_answer_stable_tail import summarise
        fresh = summarise(rows)
        (run / "ast" / "summary.json").write_text(json.dumps(fresh),
                                                  encoding="utf-8")
        crc.check_rq1(run, {}, rep)
        assert rep.failures == []
        assert rep.checked > 5

    def test_the_nested_blocks_are_checked_too(self, tmp_path, rep):
        """The report quotes these, so a scalar-only diff would miss them."""
        rows = [ast_row("p0", 10, 5, convergence_start=2),
                ast_row("p1", 10, 5, convergence_start=8)]
        run = write_run(tmp_path, ast_rows=rows, results={})
        from detect_answer_stable_tail import summarise
        fresh = summarise(rows)
        fresh["convergence_vs_ast"]["mean_gap_chunks"] = 99.0
        (run / "ast" / "summary.json").write_text(json.dumps(fresh),
                                                  encoding="utf-8")
        crc.check_rq1(run, {}, rep)
        assert any("convergence_vs_ast.mean_gap_chunks" in f
                   for f in rep.failures)


class TestTheAnalysisAgainstItself:
    def test_a_stale_q_value_is_recomputed_and_caught(self, rep):
        results = {"fdr_q": 0.05, "tests": {
            "n_tests": 2, "n_evaluated": 2, "n_significant": 1,
            "tests": [
                mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                             "at matched budget", n_problems=20, a_only=1,
                             b_only=5, p=0.21875, q=0.21875, significant=False),
                mcnemar_test("O3: safe-stopping rate, nla_readout minus "
                             "convergence at matched budget", n_problems=20,
                             a_only=0, b_only=9, p=0.001, q=0.5,
                             significant=False, other="convergence"),
            ]}}
        crc.check_test_family(results, rep)
        assert any("q_value" in f for f in rep.failures)

    def test_the_d38_invariant_is_enforced(self, rep):
        """Zero discordant problems is a tested null, so p must be 1.0.

        Recording nothing there drops the test from the BH denominator because
        its null held exactly, which biases every other q-value downwards.
        This is the defect the pilot's stored analysis still carries.
        """
        results = {"fdr_q": 0.05, "tests": {"tests": [
            mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=21, a_only=0,
                         b_only=0, p=None, q=None, significant=None),
        ]}}
        crc.check_test_family(results, rep)
        assert any(f.startswith("D38[") for f in rep.failures)

    def test_a_zero_discordant_test_with_p_one_passes(self, rep):
        results = {"fdr_q": 0.05, "tests": {"tests": [
            mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=21, a_only=0,
                         b_only=0, p=1.0, q=1.0, significant=False),
        ]}}
        crc.check_test_family(results, rep)
        assert not any(f.startswith("D38[") for f in rep.failures)

    def test_significance_must_follow_from_the_q_value(self, rep):
        results = {"fdr_q": 0.05, "tests": {"tests": [
            mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=20, a_only=9,
                         b_only=0, p=0.00390625, q=0.00390625,
                         significant=False),
        ]}}
        crc.check_test_family(results, rep)
        assert any("significant" in f for f in rep.failures)

    def test_a_missing_family_size_is_a_warning_not_a_failure(self, rep):
        """An older artifact lacks the field; that is worth saying and is not
        a numeric disagreement."""
        results = {"fdr_q": 0.05, "tests": {"tests": [
            mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=20, a_only=0,
                         b_only=0, p=1.0, q=1.0, significant=False),
        ]}}
        crc.check_test_family(results, rep)
        assert any("bh_family_size" in w for w in rep.warnings)
        assert not any("bh_family_size" in f for f in rep.failures)


class TestTheO3Accounting:
    def test_an_unevaluable_comparison_carrying_a_q_value_is_caught(self, rep):
        """It would mean the comparison entered the BH denominator."""
        results = {"tests": {"tests": [{
            "name": "O3: safe-stopping rate, nla_readout minus convergence "
                    "at matched budget",
            "p_value": None, "q_value": 0.3, "significant": False,
            "effect_name": "cohens_d_paired", "n": 0, "interval": None,
            "notes": {"skipped": "no matched budget"}}]}}
        crc.check_o3_accounting(results, rep)
        assert any("BH denominator" in f for f in rep.failures)

    def test_a_comparison_that_is_neither_tested_nor_explained_is_caught(self, rep):
        results = {"tests": {"tests": [{
            "name": "O3: safe-stopping rate, nla_readout minus convergence "
                    "at matched budget",
            "p_value": None, "q_value": None, "significant": None,
            "effect_name": "cohens_d_paired", "n": 0, "interval": None,
            "notes": {}}]}}
        crc.check_o3_accounting(results, rep)
        assert any("neither tested nor marked" in f for f in rep.failures)

    def test_a_direction_that_contradicts_the_rates_is_caught(self, rep):
        t = mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=20, a_only=4,
                         b_only=0, p=0.125, q=0.125, significant=False)
        t["notes"]["safe_rate_nla"] = 0.5
        t["notes"]["safe_rate_probe"] = 0.1
        t["notes"]["direction"] = "nla_readout worse"
        crc.check_o3_accounting({"tests": {"tests": [t]}}, rep)
        assert any("direction" in f for f in rep.failures)

    def test_the_discordant_count_must_be_the_sum_of_its_parts(self, rep):
        t = mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=20, a_only=2,
                         b_only=2, p=1.0, q=1.0, significant=False)
        t["notes"]["n_discordant"] = 9
        crc.check_o3_accounting({"tests": {"tests": [t]}}, rep)
        assert any("n_discordant" in f for f in rep.failures)

    def test_a_consistent_comparison_passes(self, rep):
        t = mcnemar_test("O3: safe-stopping rate, nla_readout minus probe "
                         "at matched budget", n_problems=20, a_only=0,
                         b_only=4, p=0.125, q=0.125, significant=False)
        crc.check_o3_accounting({"tests": {"tests": [t]}}, rep)
        assert rep.failures == []


class TestTheReportProse:
    def _run_with_report(self, tmp_path, text, results):
        run = write_run(tmp_path, results=results)
        (run / "report").mkdir()
        (run / "report" / "RESULTS.md").write_text(text, encoding="utf-8")
        return run

    def test_a_report_built_from_a_different_analysis_is_caught(self, tmp_path, rep):
        run = self._run_with_report(
            tmp_path, "- claims audited: **900**\n",
            {"rq2_headline": {"n_claims": 967}})
        crc.check_report_prose(run, crc.read_json(run / "analysis/results.json"),
                              rep)
        assert any("prose[rq2_headline.n_claims]" in f for f in rep.failures)

    def test_a_matching_report_passes(self, tmp_path, rep):
        run = self._run_with_report(
            tmp_path, "- claims audited: **967**\n",
            {"rq2_headline": {"n_claims": 967}})
        crc.check_report_prose(run, crc.read_json(run / "analysis/results.json"),
                              rep)
        assert rep.failures == []

    def test_the_verdict_table_is_checked_row_by_row(self, tmp_path, rep):
        t = mcnemar_test("O3: safe-stopping rate, nla_readout minus "
                         "semantic_entropy at matched budget", n_problems=21,
                         a_only=0, b_only=4, p=0.125, q=0.15625,
                         significant=False, other="semantic_entropy")
        table = ("| semantic_entropy | 0.100 | 0.300 | 99 | 4 (4 favour "
                 "semantic_entropy) | **nla_readout worse** | 0.1562 |\n")
        run = self._run_with_report(tmp_path, table, {"tests": {"tests": [t]}})
        crc.check_report_prose(run, crc.read_json(run / "analysis/results.json"),
                              rep)
        assert any("n_problems" in f for f in rep.failures)
        assert not any("direction" in f for f in rep.failures)


class TestStaleness:
    def test_a_run_without_a_manifest_is_skipped_not_failed(self, tmp_path, rep):
        run = write_run(tmp_path, results={})
        crc.check_staleness(run, rep)
        assert rep.failures == []

    def test_an_unknown_commit_is_a_warning(self, tmp_path, rep):
        run = write_run(tmp_path, results={})
        (run / "manifest.json").write_text(json.dumps({"stages": {
            "analysis": {"status": "complete",
                         "git": {"commit": "0" * 40, "dirty": False}}}}),
            encoding="utf-8")
        crc.check_staleness(run, rep)
        assert rep.failures == []
        assert any("not in this checkout" in w for w in rep.warnings)

    def test_a_stage_at_head_is_not_stale(self, tmp_path, rep):
        head = crc._git("rev-parse", "HEAD")
        if head is None:
            pytest.skip("not a git checkout")
        run = write_run(tmp_path, results={})
        (run / "manifest.json").write_text(json.dumps({"stages": {
            "analysis": {"status": "complete",
                         "git": {"commit": head, "dirty": False}}}}),
            encoding="utf-8")
        crc.check_staleness(run, rep)
        assert rep.failures == []

    def test_an_incomplete_stage_is_not_checked(self, tmp_path, rep):
        run = write_run(tmp_path, results={})
        (run / "manifest.json").write_text(json.dumps({"stages": {
            "analysis": {"status": "running",
                         "git": {"commit": "0" * 40, "dirty": False}}}}),
            encoding="utf-8")
        crc.check_staleness(run, rep)
        assert rep.failures == [] and rep.warnings == []


class TestTheHarness:
    def test_a_nan_matches_a_nan_but_not_a_number(self):
        assert crc._same(float("nan"), float("nan"))
        assert not crc._same(float("nan"), 1.0)

    def test_booleans_are_not_compared_as_numbers(self):
        """`True == 1` in Python, and a flag flipping to a count is a defect."""
        assert not crc._same(True, 1)
        assert crc._same(True, True)

    def test_floats_are_compared_with_a_tolerance(self):
        assert crc._same(0.1 + 0.2, 0.3)
        assert not crc._same(0.3001, 0.3)

    def test_the_exit_code_is_nonzero_only_on_a_mismatch(self, rep):
        assert not rep.failures
        rep.warn("a", "b")
        assert not rep.failures
        rep.fail("c", "d")
        assert rep.failures
