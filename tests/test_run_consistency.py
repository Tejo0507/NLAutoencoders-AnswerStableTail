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


class TestTheCausalLayers:
    """PW-23: the causal stage's rows are a measurement; its summary, and the
    analysis and report built on it, are derived. A file-level history check
    flagged the rows as stale after a correction to the derived layer only.
    The four cases below are now told apart from content."""

    @staticmethod
    def _rows():
        rows = []
        for i in range(6):
            for arm, coeff, markers, did in (
                    ("ablate_dir", 0.0, 0, None), ("ablate_dir", 1.0, 0, None),
                    ("ablate_dir", 2.0, 2, None), ("random_dir", 1.0, 0, "random_0"),
                    ("matched_position", 1.0, 0, None)):
                rows.append({
                    "id": f"p{i}|{arm}|{coeff}|{did}", "problem_id": f"p{i}",
                    "intervention": arm, "coefficient": coeff, "answer": "1",
                    "baseline_answer": "1", "gold": "1", "answer_changed": False,
                    "correct": True, "baseline_correct": True,
                    "tokens_generated": 20, "verification_markers": markers,
                    "baseline_verification_markers": 0, "direction_id": did,
                    "notes": {"n_positions_edited": 20}})
        return rows

    @pytest.fixture
    def run(self, tmp_path):
        import aggregate_results as agg
        import run_interventions as ri
        from nlaast.logging_utils import get

        run = tmp_path / "run"
        (run / "causal").mkdir(parents=True)
        (run / "analysis").mkdir()
        (run / "report").mkdir()
        rows = self._rows()
        (run / "causal" / "interventions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        summary = ri.summarise(run / "causal", None, get())
        basis = {k: v for k, v in agg.rq3_paired_contrast(rows, 0.95).items()
                 if k != "test_result"}
        results = {"rq3": {"verdict": summary["verdict"]},
                   "rq3_family_test_basis": basis}
        (run / "analysis" / "results.json").write_text(
            json.dumps(results, default=str), encoding="utf-8")
        (run / "report" / "RESULTS.md").write_text(
            "**Direction claim: not tested.**\n", encoding="utf-8")
        return run, results, ri

    def test_valid_current_outputs_pass(self, run, rep):
        run_dir, results, _ = run
        crc.check_causal(run_dir, results, rep)
        assert rep.failures == [] and rep.warnings == []

    def test_changed_rows_are_a_mismatch(self, run, rep):
        run_dir, results, _ = run
        p = run_dir / "causal" / "interventions.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        crc.check_causal(run_dir, results, rep)
        assert any(f.startswith("causal.rows_sha256") for f in rep.failures)

    def test_a_stale_summary_fails_and_a_rebuild_over_the_same_rows_passes(
            self, run, rep):
        run_dir, results, ri = run
        from nlaast.logging_utils import get
        path = run_dir / "causal" / "summary.json"
        s = json.loads(path.read_text(encoding="utf-8"))
        s["verdict"]["supported"] = False        # what the old gate wrote
        path.write_text(json.dumps(s), encoding="utf-8")
        crc.check_causal(run_dir, results, rep)
        assert any("causal.summary.verdict.supported" in f for f in rep.failures)

        ri.summarise(run_dir / "causal", None, get())
        again = crc.Report()
        crc.check_causal(run_dir, results, again)
        assert again.failures == [] and again.warnings == []

    def test_stale_downstream_analysis_and_report_fail(self, run, rep):
        run_dir, results, _ = run
        results["rq3"]["verdict"]["supported"] = False
        results["rq3_family_test_basis"]["intervention_demonstrated"] = False
        (run_dir / "report" / "RESULTS.md").write_text(
            "**Direction claim supported: False**\n", encoding="utf-8")
        crc.check_causal(run_dir, results, rep)
        failed = " ".join(rep.failures)
        assert "analysis.rq3.verdict.supported" in failed
        assert "analysis.rq3_family_test_basis.intervention_demonstrated" in failed
        assert "report.rq3.direction_claim" in failed

    def test_a_summary_without_a_row_hash_is_a_warning(self, run, rep):
        run_dir, results, _ = run
        path = run_dir / "causal" / "summary.json"
        s = json.loads(path.read_text(encoding="utf-8"))
        del s["rows_sha256"]
        path.write_text(json.dumps(s), encoding="utf-8")
        crc.check_causal(run_dir, results, rep)
        assert rep.failures == []
        assert any(w.startswith("causal.rows_sha256") for w in rep.warnings)


class TestRowCodeIsComparedBySymbol:
    def test_docstrings_and_comments_do_not_count(self):
        a = 'def f(x):\n    """Old."""\n    return x + 1\n'
        b = 'def f(x):\n    """New words."""\n    # a comment\n    return x + 1\n'
        assert crc._symbol_sources(a) == crc._symbol_sources(b)

    def test_a_changed_body_does(self):
        a = "def f(x):\n    return x + 1\n"
        b = "def f(x):\n    return x + 2\n"
        assert crc._symbol_sources(a)["f"] != crc._symbol_sources(b)["f"]

    def test_constants_are_compared_too(self):
        assert crc._symbol_sources("K = 64\n") != crc._symbol_sources("K = 32\n")

    def test_head_against_itself_changes_nothing(self):
        head = crc._git("rev-parse", "HEAD")
        if head is None:
            pytest.skip("not a git checkout")
        assert crc.causal_row_code_changes(head) == []

    def test_every_declared_symbol_exists_at_head(self):
        """A renamed generator would otherwise drop out of the check silently."""
        for path, names in crc.CAUSAL_ROW_SYMBOLS.items():
            src = crc._git("show", f"HEAD:{path}")
            if src is None:
                pytest.skip("not a git checkout")
            missing = set(names) - set(crc._symbol_sources(src))
            assert not missing, f"{path}: {sorted(missing)}"


class TestTheReplication:
    """`--replication`: rep1's outputs are separate files that record each
    other's hashes, so the defects to catch are a file changed after it was
    read, a reading edited by hand, a gate not applied, and a decision record
    that states a number the outputs do not."""

    @staticmethod
    def _gap(gap, lo, hi, p):
        return {"gap": gap, "ci95": [lo, hi], "p_bootstrap": p, "evaluable": True,
                "n_budgets_matched": 9}

    @pytest.fixture
    def tree(self, tmp_path):
        from reanalyse_o3_registered import confirmatory_verdict

        runs, docs = tmp_path / "runs", tmp_path / "docs"
        files = {
            runs / "rep1" / "traces" / "traces.jsonl": '{"problem_id": "p0"}\n',
            runs / "rep1" / "nla" / "reconstructions.jsonl": '{"id": "r0"}\n',
            runs / "rep1" / "baselines" / "boundary_scores.json": "{}",
            runs / "rep1_forced" / "forced.jsonl": '{"id": "f0"}\n',
            docs / "REPLICATION_PLAN.md": "# plan\n",
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        sha = {p: crc.sha256_file(p) for p in files}
        (runs / "rep1_forced" / "provenance.json").write_text(json.dumps(
            {"forced_sha256": sha[runs / "rep1_forced" / "forced.jsonl"]}), encoding="utf-8")
        design = {"gaps": {"nla_readout - probe": self._gap(0.047, 0.019, 0.075, 0.002),
                           "nla_readout - no_signal": self._gap(0.057, 0.039, 0.076, 0.0005)}}
        o3 = {"plan_sha256": sha[docs / "REPLICATION_PLAN.md"],
              "inputs_sha256": {
                  "traces": sha[runs / "rep1" / "traces" / "traces.jsonl"],
                  "forced": sha[runs / "rep1_forced" / "forced.jsonl"],
                  "reconstructions": sha[runs / "rep1" / "nla" / "reconstructions.jsonl"],
                  "boundary_scores": sha[runs / "rep1" / "baselines" / "boundary_scores.json"]},
              "instrument": {"reproducibility_agreement": 0.965},
              "design_A_shared_menu": design,
              "confirmatory": confirmatory_verdict(design)}
        (runs / "rep1_o3_registered").mkdir()
        self._write_o3(runs, o3)
        return runs, docs, o3

    @staticmethod
    def _write_o3(runs, o3):
        (runs / "rep1_o3_registered" / "o3_registered.json").write_text(
            json.dumps(o3), encoding="utf-8")

    def test_a_consistent_replication_passes(self, tree, rep):
        runs, docs, _ = tree
        crc.check_replication(runs, docs, rep)
        assert rep.failures == []
        assert rep.checked >= 15

    def test_an_input_changed_after_the_analysis_read_it_is_caught(self, tree, rep):
        runs, docs, _ = tree
        (runs / "rep1_forced" / "forced.jsonl").write_text('{"id": "f1"}\n', encoding="utf-8")
        crc.check_replication_hashes(runs, docs, rep)
        assert any("inputs.forced" in f for f in rep.failures)
        assert any("provenance.forced_sha256" in f for f in rep.failures)

    def test_an_edited_plan_is_caught(self, tree, rep):
        runs, docs, _ = tree
        (docs / "REPLICATION_PLAN.md").write_text("# plan, edited\n", encoding="utf-8")
        crc.check_replication_hashes(runs, docs, rep)
        assert any("REPLICATION_PLAN.md" in f for f in rep.failures)

    def test_a_reading_edited_by_hand_is_recomputed_and_caught(self, tree, rep):
        runs, docs, o3 = tree
        o3["confirmatory"]["H1_readout_minus_probe"]["reading"] = "inconclusive"
        self._write_o3(runs, o3)
        crc.check_replication_numbers(runs, rep)
        assert any("H1_readout_minus_probe.reading" in f for f in rep.failures)

    def test_a_blocked_or_ungated_analysis_is_caught(self, tree, rep):
        runs, docs, o3 = tree
        o3["blocked"] = "instrument"
        o3["instrument"]["reproducibility_agreement"] = 0.93
        self._write_o3(runs, o3)
        crc.check_replication_numbers(runs, rep)
        failed = " ".join(rep.failures)
        assert "not blocked" in failed and "agreement" in failed

    def test_the_record_must_state_the_outputs_numbers(self, tree, rep):
        runs, docs, _ = tree
        (docs / "DECISIONS.md").write_text(
            "## D94. Verdict\n\n| H1 | +0.047 | [0.019, 0.075] |\n"
            "H2: +0.058 [0.039, 0.076]\n\n## D95. Next\n", encoding="utf-8")
        crc.check_replication_record(runs, docs, rep)
        assert not any("D75 H1" in f for f in rep.failures)   # table form is read
        assert any("D75 H2" in f for f in rep.failures)       # 0.058 is not 0.057

    def test_a_number_in_another_entry_does_not_count(self, tree, rep):
        runs, docs, _ = tree
        (docs / "DECISIONS.md").write_text(
            "## D94. Verdict\n\nnothing\n\n## D95. Later\n\n+0.047 [0.019, 0.075]\n",
            encoding="utf-8")
        crc.check_replication_record(runs, docs, rep)
        assert any("D75 H1" in f for f in rep.failures)

    def test_record_numbers_are_written_with_a_true_minus(self):
        assert crc.fmt_signed(-0.0082) == "−0.008"
        assert crc.fmt_ci([-0.036, 0.024]) == "[−0.036, 0.024]"

    # --- empty and incomplete runs must not read as passes ---

    def test_an_empty_runs_directory_fails(self, tmp_path, rep):
        (tmp_path / "runs").mkdir()
        crc.check_replication(tmp_path / "runs", tmp_path / "docs", rep)
        assert any("required output" in f for f in rep.failures)
        assert crc.replication_exit_code(rep) == 1

    def test_zero_checks_is_not_a_pass(self, rep):
        assert rep.checked == 0 and not rep.failures
        assert crc.replication_exit_code(rep) == 1

    def test_the_minimal_valid_run_passes_and_lists_what_it_skipped(self, tree, rep):
        runs, docs, _ = tree
        crc.check_replication(runs, docs, rep)
        assert crc.replication_exit_code(rep) == 0
        assert "Stage 1 numbers" in rep.skipped

    def test_complete_mode_requires_every_output(self, tree, rep):
        runs, docs, _ = tree
        crc.check_replication(runs, docs, rep, complete=True)
        assert any("rep1_stage1/analysis.json" in f for f in rep.failures)
        assert crc.replication_exit_code(rep) == 1

    @staticmethod
    def _write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, str):
            path.write_text(obj, encoding="utf-8")
        else:
            path.write_text(json.dumps(obj), encoding="utf-8")

    def _d82_and_stage1(self, runs, docs):
        """A rep1 D82 analysis and a Stage 1 analysis, internally consistent,
        with no D75 output beside them."""
        w = self._write
        w(runs / "rep1" / "traces" / "traces.jsonl", '{"problem_id": "p0"}\n')
        w(runs / "rep1_forced" / "forced.jsonl", '{"id": "f0"}\n')
        w(runs / "rep1_forced" / "provenance.json",
          {"forced_sha256": crc.sha256_file(runs / "rep1_forced" / "forced.jsonl")})
        w(docs / "EXTENDED_FORCING_PLAN.md", "# d82\n")
        w(docs / "LEAKAGE_RESISTANT_DESIGN.md", "# design\n")
        ext = runs / "rep1_extforced"
        w(ext / "forced_ext.jsonl", "".join(f'{{"id": "e{i}"}}\n' for i in range(3)))
        w(ext / "labels_O256.jsonl", '{"id": "l0"}\n')
        traces = crc.sha256_file(runs / "rep1" / "traces" / "traces.jsonl")
        w(ext / "analysis.json", {
            "plan_sha256": crc.sha256_file(docs / "EXTENDED_FORCING_PLAN.md"),
            "ext_provenance": {
                "traces_sha256": traces,
                "registered_forced_sha256": crc.sha256_file(runs / "rep1_forced" / "forced.jsonl"),
                "forced_ext_sha256": crc.sha256_file(ext / "forced_ext.jsonl")},
            "labels_O256_sha256": crc.sha256_file(ext / "labels_O256.jsonl"),
            "instrument": {"stability": 1.0, "stability_gate": 0.95,
                           "check_token_count_max_abs_diff": 4,
                           "n_cut_off_forced": 2, "n_check": 1},
            "net_savings_used": False,
            "cut_off_meaning": {"f_256_equals_final": 0.867, "reading":
                                "the 48-token outcome mostly measured commitment within 48 tokens"},
            "cells": {}})
        st = runs / "rep1_stage1"
        w(st / "positions.jsonl", '{"p": 0}\n')
        w(st / "forced.jsonl", '{"id": "a", "purpose": "ext"}\n{"id": "b", "purpose": "ext"}\n'
                               '{"id": "c", "purpose": "check48"}\n')
        w(st / "provenance.json", {"positions_sha256": crc.sha256_file(st / "positions.jsonl"),
                                   "traces_sha256": traces, "check_rows": [1],
                                   "n_positions": 2})
        d1 = {"auc_a": 0.83, "auc_b": 0.83, "delta_auc": 0.0, "ci95": [-0.004, 0.004]}
        w(st / "analysis.json", {
            "positions_sha256": crc.sha256_file(st / "positions.jsonl"),
            "design_sha256": crc.sha256_file(docs / "LEAKAGE_RESISTANT_DESIGN.md"),
            "instrument": {"first48_vs_separate_48_agreement": 0.98, "gate": 0.95},
            "n_units": 2, "n_units_with_passing_verbalisation": 2,
            "primary": {"delta": 0.0, "ci95": [-0.004, 0.004],
                        "reading": "equivalent within ±0.01"},
            "Y_ext": {"contrasts": {"Δ1 E over pos+A": d1}}})

    def test_a_missing_d75_does_not_hide_the_other_checks(self, tmp_path, rep):
        runs, docs = tmp_path / "runs", tmp_path / "docs"
        self._d82_and_stage1(runs, docs)
        crc.check_replication(runs, docs, rep)
        required = [f for f in rep.failures if "required output" in f]
        assert required, "the missing D75 output must fail"
        assert rep.failures == required, rep.failures   # D82 and Stage 1 pass
        assert "D82 rep1 numbers" not in rep.skipped
        assert "Stage 1 numbers" not in rep.skipped
        assert rep.checked > len(required) + 15          # they actually ran
        assert crc.replication_exit_code(rep) == 1

    def test_a_planted_defect_in_stage1_is_caught_without_d75(self, tmp_path, rep):
        runs, docs = tmp_path / "runs", tmp_path / "docs"
        self._d82_and_stage1(runs, docs)
        p = runs / "rep1_stage1" / "analysis.json"
        st = json.loads(p.read_text(encoding="utf-8"))
        st["primary"]["reading"] = "practically relevant increment"
        p.write_text(json.dumps(st), encoding="utf-8")
        crc.check_replication_numbers(runs, rep)
        assert any("Stage 1 primary.reading" in f for f in rep.failures)

    def test_contrast_arithmetic_is_checked(self, rep):
        crc.check_contrast_arithmetic("x", {"c": {"auc_a": 0.8, "auc_b": 0.7,
                                                  "delta_auc": 0.2}}, rep)
        assert any("delta = a - b" in f for f in rep.failures)

    def test_f_must_be_stated_as_f_not_as_any_matching_digits(self, tree, rep):
        runs, docs, _ = tree
        self._write(runs / "rep1_extforced" / "analysis.json",
                    {"cut_off_meaning": {"f_256_equals_final": 0.867}})
        self._write(docs / "DECISIONS.md", "## D95. x\n\nagreement 0.867 elsewhere\n")
        crc.check_replication_record(runs, docs, rep)
        assert any("record[D82 f]" in f for f in rep.failures)

    @pytest.mark.integration
    def test_the_finished_replication_passes_in_complete_mode(self, rep):
        """The stored outputs, if present (NLAAST_TEST_RUNS), read only."""
        import os
        runs = Path(os.environ.get("NLAAST_TEST_RUNS",
                                   Path(__file__).resolve().parents[1] / "runs"))
        if not (runs / crc.REQUIRED_OUTPUT).exists():
            pytest.skip("needs the stored rep1 outputs (set NLAAST_TEST_RUNS)")
        crc.check_replication(runs, Path(__file__).resolve().parents[1] / "docs", rep,
                              complete=True)
        assert rep.failures == []
        assert crc.replication_exit_code(rep) == 0


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
