"""`run_interventions.py --summarise-only` rebuilds the verdict without a model.

The verdict is computed once, when the GPU stage finishes, and the analysis
copies it forward. On `main` that froze "supported: False" - built from a
readout that was zero in every arm - into the report after the gate itself
was corrected. The rows are the measurement; the summary is derived from them
and must be rebuildable on CPU.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from nlaast.logging_utils import get  # noqa: E402
from nlaast.provenance import Manifest  # noqa: E402


@pytest.fixture(scope="module")
def ri():
    return importlib.import_module("run_interventions")


def _row(arm, coeff, pid, markers, did=None):
    return {"id": f"{pid}|{arm}|{coeff}|{did}", "problem_id": pid,
            "intervention": arm, "coefficient": coeff, "answer": "1",
            "baseline_answer": "1", "gold": "1", "answer_changed": False,
            "correct": True, "baseline_correct": True, "tokens_generated": 20,
            "verification_markers": markers, "baseline_verification_markers": 0,
            "direction_id": did, "notes": {"n_positions_edited": 20}}


@pytest.fixture
def run(tmp_path, monkeypatch, ri):
    out = tmp_path / "causal"
    out.mkdir()
    rows = []
    for i in range(6):
        p = f"p{i}"
        rows += [_row("ablate_dir", 0.0, p, 0), _row("ablate_dir", 1.0, p, 0),
                 _row("ablate_dir", 2.0, p, 2),
                 _row("random_dir", 1.0, p, 0, "random_0"),
                 _row("matched_position", 1.0, p, 0)]
    (out / "interventions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # The stale summary a GPU run under the old gate would have left.
    (out / "summary.json").write_text(json.dumps(
        {"direction": {"available": True, "n_problems": 97},
         "verdict": {"supported": False}}), encoding="utf-8")
    m = Manifest.open(tmp_path, {"x": 1}, "h")
    m.start_stage("causal")
    m.finish_stage("causal", metrics={"verdict": {"supported": False}})

    def no_model(*a, **k):
        raise AssertionError("--summarise-only must not load a model")
    monkeypatch.setattr(ri, "TargetModel", no_model)
    cfg = SimpleNamespace(stage_dir=lambda s: tmp_path / s)
    return ri, cfg, m, out, tmp_path


def test_the_verdict_is_rebuilt_from_the_rows(run):
    ri, cfg, m, out, _ = run
    assert ri.summarise_only(cfg, m, get()) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"]["supported"] is None, "zero readout is untested, not refuted"
    assert s["n_results"] == 30
    assert s["direction"] == {"available": True, "n_problems": 97}, \
        "direction metadata is carried over, not refitted"


def test_the_gpu_runs_record_survives_and_the_rebuild_is_recorded(run):
    ri, cfg, m, _, run_dir = run
    before = dict(json.loads((run_dir / "manifest.json").read_text())["stages"]["causal"])
    ri.summarise_only(cfg, m, get())
    after = json.loads((run_dir / "manifest.json").read_text())["stages"]["causal"]
    for k in ("started", "finished", "elapsed_s", "git", "status"):
        assert after[k] == before[k]
    assert after["summary_rebuilt"]["model_loaded"] is False
    assert after["metrics"]["verdict"]["supported"] is None


def test_no_rows_is_an_error_not_an_empty_verdict(run):
    ri, cfg, m, out, _ = run
    (out / "interventions.jsonl").unlink()
    assert ri.summarise_only(cfg, m, get()) == 1
