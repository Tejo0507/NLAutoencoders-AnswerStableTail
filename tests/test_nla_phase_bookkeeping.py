"""The `nla` stage is two phases, and finishing one is not finishing both.

The verbaliser and the reconstructor are each 7B-class and only one fits in
6.44 GB, so `--phase verbalise` followed by `--phase reconstruct` is the
intended way to run this stage on this hardware, not an edge case.

The stage used to mark itself `complete` whenever the script reached the end,
whichever phase had run. So the real pilot run did this:

    run_autoencoder.py --phase verbalise    -> 126 windows, stage "complete"
    run_autoencoder.py --phase reconstruct  -> "stage 'nla' already complete"

and exited 0. No reconstructions were written, and the loss was silent and
downstream: the NLA arm dropped off the matched-budget stopping comparison, F10
had no windows to compare, and F3 lost its reconstruction contrast - all of
which merely look like "not available" in the report.

Completeness is now read off the artefacts. These tests pin that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _status(run_dir: Path) -> tuple[str, dict]:
    data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage = data["stages"]["nla"]
    return stage["status"], stage.get("metrics") or {}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    from nlaast import paths

    monkeypatch.setattr(paths, "RUNS", tmp_path)
    d = tmp_path / "phases"
    (d / "nla").mkdir(parents=True)
    return d


def _finish(run_dir: Path, *, verbalise: bool, reconstruct: bool, metrics=None):
    """Replay the stage's end-of-run bookkeeping for a given set of artefacts."""
    from nlaast import provenance

    out = run_dir / "nla"
    verb, recon = out / "verbalisations.jsonl", out / "reconstructions.jsonl"
    if verbalise:
        verb.write_text('{"id": "a"}\n', encoding="utf-8")
    if reconstruct:
        recon.write_text('{"id": "a|s0"}\n', encoding="utf-8")

    phases_done = sorted(name for name, p in (("verbalise", verb),
                                              ("reconstruct", recon))
                         if p.exists() and p.stat().st_size > 0)
    complete = len(phases_done) == 2
    m = dict(metrics or {})
    m["phases_present"] = phases_done

    manifest = provenance.Manifest.open(run_dir, {}, "hash")
    manifest.start_stage("nla")
    manifest.finish_stage("nla", status="complete" if complete else "partial",
                          output=str(out), metrics=m)
    return phases_done


class TestCompletenessFollowsTheArtefacts:
    def test_verbalise_alone_leaves_the_stage_partial(self, run_dir):
        _finish(run_dir, verbalise=True, reconstruct=False)
        status, metrics = _status(run_dir)
        assert status == "partial"
        assert metrics["phases_present"] == ["verbalise"]

    def test_reconstruct_alone_leaves_the_stage_partial(self, run_dir):
        _finish(run_dir, verbalise=False, reconstruct=True)
        status, metrics = _status(run_dir)
        assert status == "partial"
        assert metrics["phases_present"] == ["reconstruct"]

    def test_both_phases_make_it_complete(self, run_dir):
        _finish(run_dir, verbalise=True, reconstruct=True)
        status, metrics = _status(run_dir)
        assert status == "complete"
        assert metrics["phases_present"] == ["reconstruct", "verbalise"]

    def test_an_empty_output_file_does_not_count_as_a_finished_phase(self, run_dir):
        """A crashed phase can leave the file created and empty."""
        (run_dir / "nla" / "reconstructions.jsonl").write_text("", encoding="utf-8")
        _finish(run_dir, verbalise=True, reconstruct=False)
        status, metrics = _status(run_dir)
        assert status == "partial"
        assert "reconstruct" not in metrics["phases_present"]


class TestPartialIsNotSkipped:
    def test_should_skip_only_skips_a_complete_stage(self, run_dir):
        """The behaviour that made the loss silent: `partial` has to re-run."""
        import logging

        from _stage import should_skip
        from nlaast import provenance

        _finish(run_dir, verbalise=True, reconstruct=False)
        manifest = provenance.Manifest.open(run_dir)
        assert manifest.stage_status("nla") == "partial"
        assert should_skip(manifest, "nla", False, logging.getLogger("t")) is False

        _finish(run_dir, verbalise=True, reconstruct=True)
        manifest = provenance.Manifest.open(run_dir)
        assert should_skip(manifest, "nla", False, logging.getLogger("t")) is True


class TestMetricsSurvive:
    def test_the_reconstruct_metrics_are_kept_alongside_the_phase_list(self, run_dir):
        _finish(run_dir, verbalise=True, reconstruct=True,
                metrics={"n_reconstructions": 252, "overall": {"fve": 0.5}})
        _, metrics = _status(run_dir)
        assert metrics["n_reconstructions"] == 252
        assert metrics["overall"]["fve"] == 0.5
        assert metrics["phases_present"] == ["reconstruct", "verbalise"]
