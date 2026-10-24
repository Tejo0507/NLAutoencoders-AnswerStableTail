"""`scripts/stopping_pilot.py`: the protocol gate, phase order and CPU phases.

Builds synthetic pilot runs in a temporary directory; nothing here reads
`runs/`, loads a model or touches a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stopping_pilot as sp  # noqa: E402


def _protocol(tmp_path, **over):
    p = json.loads((sp.ROOT / "configs" / "stopping_pilot_protocol.template.json").read_text())
    p.update({"status": "frozen", "alpha": 0.10, "alpha_justification": "test only",
              "alpha_cal": 0.10, "delta": 0.05, "tau_grid_n": 50,
              "n_cal": sp.config_counts("pilot_cal"), "n_eval": sp.config_counts("pilot_eval"),
              "delta_save": 0.02, "gpu_hours_cap": 1})
    p.update(over)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    return path


def test_the_template_cannot_run():
    with pytest.raises(SystemExit, match="not 'frozen'"):
        sp.load_protocol(sp.ROOT / "configs" / "stopping_pilot_protocol.template.json")


def test_an_unfilled_decision_stops_everything(tmp_path):
    with pytest.raises(SystemExit, match="alpha_cal"):
        sp.load_protocol(_protocol(tmp_path, alpha_cal=None))


def test_configs_must_match_the_protocol(tmp_path):
    proto = sp.load_protocol(_protocol(tmp_path, n_eval={"gsm8k": 1, "math": 1}))
    with pytest.raises(SystemExit, match="pilot_eval"):
        sp.check_configs(proto)


def test_training_labels_follow_the_truncation_rule():
    u = {"Y_ext": True, "Y_ext_u": False, "forced_finished": False}
    assert sp.training_label(u, "cut_off_unsafe") is False
    assert sp.training_label(u, "as_parsed") is True
    assert sp.training_label(u, "as_no_answer") is False


def _run(root: Path, name: str, n_problems: int, rng, unsafe_low=0.4):
    d = root / name
    d.mkdir(parents=True)
    traces, forced, scores = [], [], []
    for k in range(n_problems):
        pid = f"{name}-{k}"
        n = 8
        traces.append({"problem_id": pid, "ok": True, "dataset": "gsm8k", "gold": "1",
                       "final_answer": "1", "n_tokens": 400,
                       "chunks": [{"prefix_tokens": 40 * (i + 1)} for i in range(n)]})
        for i in range(n):
            late = i >= 4
            correct = late or rng.random() > unsafe_low
            row = {"problem_id": pid, "chunk": i, "dataset": "gsm8k",
                   "forced_correct": correct, "finished": True, "final_correct": True,
                   "prefix_tokens": 40 * (i + 1), "forced_tokens": 12, "total_tokens": 400}
            forced.append({"id": f"{pid}|{i}|ext", "problem_id": pid, "chunk": i,
                           "purpose": "ext", "answer": "1" if correct else "2",
                           "finished": True, "n_tokens": 12, "prefix_tokens": 40 * (i + 1)})
            s = i / n + rng.normal(0, 0.05)
            scores.append({**row, "P0_position": s, "P1_same_label_probe": s + (0.2 if late else 0)})
        forced.append({"id": f"{pid}|0|repro", "problem_id": pid, "chunk": 0, "purpose": "repro",
                       "answer": forced[-8]["answer"], "finished": True, "n_tokens": 12,
                       "prefix_tokens": 40})
    for fname, rows in (("traces/traces.jsonl", traces), ("pilot_forced.jsonl", forced),
                        ("pilot_scores.jsonl", scores)):
        (d / fname).parent.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _args(tmp_path, protocol, **kw):
    return argparse.Namespace(runs=tmp_path / "runs", protocol=protocol, run=None,
                              train_runs=tmp_path / "train", **kw)


def test_calibrate_then_evaluate_on_synthetic_runs(tmp_path):
    rng = np.random.default_rng(0)
    path = _protocol(tmp_path)
    proto = sp.load_protocol(path)
    _run(tmp_path / "runs", "pilot_cal", 200, rng)
    a = _args(tmp_path, path)
    assert sp.cmd_calibrate(a, proto) == 0
    cal = json.loads((tmp_path / "runs" / "pilot_calibration" / "calibration.json").read_text())
    assert set(cal["policies"]) == set(sp.POLICIES)
    assert cal["protocol_sha256"] == sp.sha256(path)
    _run(tmp_path / "runs", "pilot_eval", 120, rng)
    assert sp.cmd_evaluate(a, proto) == 0
    ev = json.loads((tmp_path / "runs" / "pilot_evaluation" / "evaluation.json").read_text())
    assert ev["status"].startswith("exploratory pilot")
    for pol, res in ev["policies"].items():
        if res["tau"] is not None:
            assert {"coverage", "selective_risk", "safety_gate", "mean_net_saving"} <= set(res)
            assert set(ev["sensitivity"][pol]) == {"cut_off_unsafe", "as_parsed", "as_no_answer"}


def test_calibration_refuses_once_evaluation_forcing_exists(tmp_path):
    rng = np.random.default_rng(1)
    path = _protocol(tmp_path)
    _run(tmp_path / "runs", "pilot_cal", 50, rng)
    _run(tmp_path / "runs", "pilot_eval", 20, rng)
    with pytest.raises(SystemExit, match="calibration must come first"):
        sp.cmd_calibrate(_args(tmp_path, path), sp.load_protocol(path))


def test_evaluation_forcing_refuses_before_calibration(tmp_path):
    path = _protocol(tmp_path)
    a = _args(tmp_path, path)
    a.run = "pilot_eval"
    with pytest.raises(SystemExit, match="before the calibration is frozen"):
        sp.cmd_force(a, sp.load_protocol(path))


def test_a_changed_calibration_stops_evaluation(tmp_path):
    rng = np.random.default_rng(2)
    path = _protocol(tmp_path)
    proto = sp.load_protocol(path)
    _run(tmp_path / "runs", "pilot_cal", 100, rng)
    a = _args(tmp_path, path)
    sp.cmd_calibrate(a, proto)
    cal = tmp_path / "runs" / "pilot_calibration" / "calibration.json"
    cal.write_text(cal.read_text().replace('"tau"', '"tau" ', 1))
    _run(tmp_path / "runs", "pilot_eval", 20, rng)
    with pytest.raises(SystemExit, match="changed after it was frozen"):
        sp.cmd_evaluate(a, proto)
