"""The post-rep1 queue: approved order (D87) and what a re-run repeats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import post_rep1_queue as q  # noqa: E402


def test_fresh_data_work_runs_first_and_main_nla240_last():
    names = [n for n, _, _ in q.steps(stage1=True)]
    assert names[:4] == ["rep1_information", "rep1_information_posthoc", "rep1_lexicon",
                         "rep1_selective"]
    assert names.index("ext_force_rep1") < names.index("stage1_positions")
    assert names[-3:] == ["ext_force_main_nla240", "ext_analysis_main_nla240",
                          "info_O256_main_nla240"]
    assert not any(n.startswith("stage1_") for n, _, _ in q.steps(stage1=False))


def test_a_finished_reconstruction_is_not_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "R", tmp_path)
    gate = dict((n, g) for n, _, g in q.steps(stage1=True))["stage1_reconstruct"]
    assert gate() is not None            # not finished: it must run
    (tmp_path / "rep1_stage1").mkdir()
    (tmp_path / "rep1_stage1" / "reconstruct_summary.json").write_text(json.dumps({}))
    assert gate() is None                # finished: a re-run skips it


def test_a_forcing_pass_counts_only_when_its_end_of_run_fields_exist(tmp_path):
    d = tmp_path / "x_extforced"
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"n_cut_off": 3, "n_check": 2}))
    assert q._forcing_complete(d) is not None         # written at start only
    (d / "provenance.json").write_text(json.dumps(
        {"n_cut_off": 3, "n_check": 2, "n_rows": 4, "forced_ext_sha256": "x"}))
    assert "4 rows of 5" in q._forcing_complete(d)
    (d / "provenance.json").write_text(json.dumps(
        {"n_cut_off": 3, "n_check": 2, "n_rows": 5, "forced_ext_sha256": "x"}))
    assert q._forcing_complete(d) is None
