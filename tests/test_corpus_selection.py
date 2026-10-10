"""Which runs form the corpus for the root-level supervised scripts.

`common_data.build_table` used to glob every `runs/*/traces/traces.jsonl` and
keep the first row it saw for each problem id. Three things were wrong with
that, and all three put numbers into `outputs/` that no reader could have
caught:

* `runs/tiny`, written by the end-to-end test, holds traces from a randomly
  initialised 64-dimensional model. Globbing pooled them with real ones.
* `runs/smoke` runs under a different token cap, so its traces are not rows of
  the same corpus.
* Sorted-path order put `mlcorpus` before `pilot`, so traces generated *before*
  the chunker and answer-parser fixes took precedence over the corrected ones
  for every problem the two runs share - and the chunk boundaries baked into a
  stale trace cannot be repaired by reparsing.

These tests pin the replacement: comparability decided by the settings and the
code revision that produced the traces, the newest run as the reference, and
the decision recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common_data  # noqa: E402

BASE_CONFIG = {
    "target": {"repo_id": "Qwen/Qwen2.5-7B-Instruct", "precision": "nf4", "layer": 20},
    "generation": {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.95,
                   "canonical_greedy": True, "system_prompt": "sys",
                   "force_answer_suffix": "\n\nTherefore, the answer is",
                   "force_answer_max_new_tokens": 24},
    "chunk": {"min_chunk_chars": 12, "max_chunks": 40},
    "ast": {"k_continuations": 3, "continuation_temperature": 0.8,
            "continuation_max_new_tokens": 160, "require_unanimous": True,
            "min_boundary_fraction": 0.0, "max_boundaries_evaluated": 12},
}


def _deep_update(base: dict, overrides: dict) -> dict:
    import copy

    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_run(runs_root: Path, name: str, ids, *, commit="c" * 40, dirty=False,
             config_overrides=None, mtime=None, tails=None, with_config=True,
             fixture_marker=False):
    """A minimal run directory: traces, optional AST rows, config and manifest."""
    import os
    import yaml

    d = runs_root / name
    (d / "traces").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pid in enumerate(ids):
        rows.append({
            "id": pid, "problem_id": pid, "dataset": "gsm8k", "split": "eval",
            "question": f"question {pid}?", "gold": "1",
            "trace": f"Step one for {name}. The answer is 1.",
            "n_tokens": 20 + i, "n_chunks": 2, "truncated": False,
            "final_answer": "1", "final_correct": i % 2 == 0, "ok": True,
        })
    (d / "traces" / "traces.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    if tails is not None:
        (d / "ast").mkdir(parents=True, exist_ok=True)
        ast_rows = [{"id": pid, "problem_id": pid, "status": status,
                     "tail_fraction": frac}
                    for pid, status, frac in tails]
        (d / "ast" / "ast.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in ast_rows), encoding="utf-8")

    if with_config:
        cfg = _deep_update(BASE_CONFIG, config_overrides or {})
        (d / "config.resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (d / "manifest.json").write_text(
        json.dumps({"git": {"commit": commit, "dirty": dirty}}), encoding="utf-8")

    if fixture_marker:
        (d / common_data.FIXTURE_MARKER).write_text("synthetic", encoding="utf-8")

    if mtime is not None:
        p = d / "traces" / "traces.jsonl"
        os.utime(p, (mtime, mtime))
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(common_data, "RUNS", root)
    return root


class TestReferenceChoice:
    def test_the_newest_run_is_the_reference_not_the_largest(self, runs):
        make_run(runs, "mlcorpus", [f"p{i}" for i in range(30)], mtime=1_000_000)
        make_run(runs, "pilot", ["p0", "p1"], mtime=2_000_000)
        sel = common_data.select_runs()
        assert sel["reference_run"] == "pilot"

    def test_no_traces_anywhere_is_a_clear_error(self, runs):
        with pytest.raises(SystemExit, match="no run under runs/ holds usable traces"):
            common_data.select_runs()


class TestDisqualification:
    """Some runs can never be corpus data, and comparison is too late to say so.

    The end-to-end test writes its fixture run *last*, so "newest run wins"
    chose it as the reference - and then every real run was excluded for
    differing from a 64-dimensional random model. The supervised analyses
    fitted on six rows of that noise and wrote the result to `outputs/`. A run
    that cannot be corpus data has to be ruled out *before* it can become the
    thing everything else is compared against.
    """

    def test_a_marked_fixture_run_is_never_the_reference(self, runs):
        make_run(runs, "pilot", [f"p{i}" for i in range(5)], mtime=1_000_000)
        make_run(runs, "tiny", ["f0"], mtime=9_000_000, fixture_marker=True)
        sel = common_data.select_runs()
        assert sel["reference_run"] == "pilot"
        assert sel["runs"] == ["pilot"]
        assert "tiny" in sel["disqualified"]
        assert "synthetic fixture" in sel["disqualified"]["tiny"]

    def test_a_foreign_target_model_is_disqualified_without_a_marker(self, runs):
        """Second, independent guard: the released autoencoder is bound to one
        model, so a run against another is not a run of this study."""
        make_run(runs, "pilot", ["p0"], mtime=1_000_000)
        make_run(runs, "tiny", ["f0"], mtime=9_000_000,
                 config_overrides={"target": {"repo_id": "nlaast-fixture/target"}})
        sel = common_data.select_runs()
        assert sel["reference_run"] == "pilot"
        assert "not the model this study is bound to" in sel["disqualified"]["tiny"]

    def test_a_wrong_width_is_disqualified(self, runs):
        make_run(runs, "pilot", ["p0"], mtime=1_000_000)
        make_run(runs, "toy", ["t0"], mtime=9_000_000,
                 config_overrides={"target": {"d_model": 64}})
        sel = common_data.select_runs()
        assert sel["reference_run"] == "pilot"
        assert "d_model is 64" in sel["disqualified"]["toy"]

    def test_requesting_a_disqualified_run_is_refused(self, runs):
        make_run(runs, "pilot", ["p0"], mtime=1_000_000)
        make_run(runs, "tiny", ["f0"], mtime=9_000_000, fixture_marker=True)
        with pytest.raises(SystemExit, match="cannot be corpus data"):
            common_data.select_runs(["tiny"])

    def test_only_disqualified_runs_is_a_clear_error(self, runs):
        make_run(runs, "tiny", ["f0"], fixture_marker=True)
        with pytest.raises(SystemExit, match="no run under runs/ holds usable traces"):
            common_data.select_runs()

    def test_build_table_records_the_disqualifications(self, runs):
        make_run(runs, "pilot", ["p0", "p1"], mtime=1_000_000)
        make_run(runs, "tiny", ["f0"], mtime=9_000_000, fixture_marker=True)
        df, prov = common_data.build_table()
        assert prov["runs"] == ["pilot"]
        assert len(df) == 2
        assert "tiny" in prov["disqualified"]


class TestComparability:
    def test_a_different_token_cap_is_excluded(self, runs):
        make_run(runs, "pilot", ["p0"], mtime=2_000_000)
        make_run(runs, "smoke", ["p1"], mtime=1_000_000,
                 config_overrides={"generation": {"max_new_tokens": 256}})
        sel = common_data.select_runs()
        assert sel["runs"] == ["pilot"]
        assert "generation.max_new_tokens" in sel["excluded"]["smoke"]["differs_on"]

    def test_a_different_target_model_is_kept_out_entirely(self, runs):
        """The fixture run's case. A 64-wide random model is not merely
        incomparable with the corpus, it is disqualified from being corpus
        data at all - which is the only version of this check that survives
        the fixture run being the newest. See TestDisqualification."""
        make_run(runs, "pilot", ["p0"], mtime=2_000_000)
        make_run(runs, "tiny", ["f0"], mtime=1_000_000,
                 config_overrides={"target": {"repo_id": "nlaast-fixture/target",
                                              "layer": 1, "precision": "bf16"}})
        sel = common_data.select_runs()
        assert sel["runs"] == ["pilot"]
        assert "tiny" not in sel["excluded"]
        assert "tiny" in sel["disqualified"]

    def test_matching_settings_and_revision_do_pool(self, runs):
        make_run(runs, "mlcorpus", ["p1", "p2"], mtime=1_000_000, commit="a" * 40)
        make_run(runs, "pilot", ["p0"], mtime=2_000_000, commit="a" * 40)
        sel = common_data.select_runs()
        assert sel["runs"] == ["mlcorpus", "pilot"]

    def test_a_different_commit_is_not_poolable(self, runs):
        """The stale-trace case, which no config comparison can see."""
        make_run(runs, "mlcorpus", ["p1"], mtime=1_000_000, commit="a" * 40)
        make_run(runs, "pilot", ["p0"], mtime=2_000_000, commit="b" * 40)
        sel = common_data.select_runs()
        assert sel["runs"] == ["pilot"]
        assert "code_revision" in sel["excluded"]["mlcorpus"]["differs_on"]

    def test_a_dirty_tree_never_pools_automatically(self, runs):
        make_run(runs, "mlcorpus", ["p1"], mtime=1_000_000, commit="a" * 40, dirty=True)
        make_run(runs, "pilot", ["p0"], mtime=2_000_000, commit="a" * 40, dirty=True)
        sel = common_data.select_runs()
        assert sel["runs"] == ["pilot"]
        assert "code_revision" in sel["excluded"]["mlcorpus"]["differs_on"]

    def test_a_run_without_a_resolved_config_is_excluded(self, runs):
        make_run(runs, "pilot", ["p0"], mtime=2_000_000)
        make_run(runs, "orphan", ["p1"], mtime=1_000_000, with_config=False)
        sel = common_data.select_runs()
        assert sel["runs"] == ["pilot"]
        assert sel["excluded"]["orphan"] == "no config.resolved.yaml"


class TestExplicitSelection:
    def test_requested_runs_are_used_in_the_order_given(self, runs):
        make_run(runs, "mlcorpus", ["p1"], mtime=1_000_000, commit="a" * 40)
        make_run(runs, "pilot", ["p0"], mtime=2_000_000, commit="a" * 40)
        sel = common_data.select_runs(["mlcorpus", "pilot"])
        assert sel["runs"] == ["mlcorpus", "pilot"]
        assert sel["reference_run"] == "mlcorpus"
        assert sel["explicit"] is True

    def test_requesting_an_incomparable_run_is_refused_loudly(self, runs):
        make_run(runs, "pilot", ["p0"], mtime=2_000_000)
        make_run(runs, "smoke", ["p1"], mtime=1_000_000,
                 config_overrides={"generation": {"max_new_tokens": 256}})
        with pytest.raises(SystemExit, match="not poolable"):
            common_data.select_runs(["pilot", "smoke"])

    def test_requesting_a_run_with_no_traces_is_an_error(self, runs):
        make_run(runs, "pilot", ["p0"])
        with pytest.raises(SystemExit, match="no traces"):
            common_data.select_runs(["pilot", "nonexistent"])


class TestBuildTable:
    def test_the_earliest_listed_run_wins_a_collision(self, runs):
        """Explicit order decides, so a caller can prefer the corrected run."""
        make_run(runs, "mlcorpus", ["shared"], mtime=1_000_000, commit="a" * 40)
        make_run(runs, "pilot", ["shared"], mtime=2_000_000, commit="a" * 40)
        df, prov = common_data.build_table(run_ids=["pilot", "mlcorpus"])
        assert len(df) == 1
        assert prov["traces_per_run"] == {"pilot": 1, "mlcorpus": 0}

    def test_provenance_records_the_corpus(self, runs):
        make_run(runs, "pilot", ["p0", "p1"], mtime=2_000_000)
        df, prov = common_data.build_table()
        assert prov["n_rows"] == len(df) == 2
        assert prov["runs"] == ["pilot"]
        assert "code_revision" in prov["pooling_signature"]

    def test_only_problems_with_an_ok_tail_reach_the_regression(self, runs):
        make_run(runs, "pilot", ["p0", "p1", "p2"], mtime=2_000_000,
                 tails=[("p0", "ok", 0.4), ("p1", "no_stable_point", 0.0),
                        ("p2", "truncated", 0.3)])
        df, prov = common_data.build_table(with_ast=True)
        assert list(df["id"]) == ["p0"]
        assert prov["dropped_without_ok_tail"] == 2

    def test_write_provenance_round_trips(self, runs, tmp_path):
        make_run(runs, "pilot", ["p0"], mtime=2_000_000)
        _, prov = common_data.build_table()
        out = tmp_path / "out"
        common_data.write_provenance(out, prov, extra={"analysis": "demo"})
        written = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        assert written["analysis"] == "demo"
        assert written["runs"] == ["pilot"]
