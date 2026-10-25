"""`scripts/describe_pilot_stops.py`: where P1 stops, by the same rule as the pilot's walk."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import describe_pilot_stops as dps  # noqa: E402
from nlaast.analysis.stopping_pilot import build_problems, walk  # noqa: E402


def _row(pid, chunk, score, prefix, total=100, forced=5):
    return {"problem_id": pid, "chunk": chunk, dps.POLICY: score, "prefix_tokens": prefix,
            "forced_tokens": forced, "total_tokens": total, "forced_correct": True,
            "finished": True, "final_correct": True}


ROWS = [
    # stored out of chunk order, as the pilot's files are
    _row("a", 2, 0.97, 60), _row("a", 1, 0.90, 30), _row("a", 3, 0.99, 100),
    _row("b", 1, 0.96, 20), _row("b", 2, 0.50, 80),
    _row("c", 1, 0.10, 50), _row("c", 2, 0.20, 100),
]


def test_stops_where_the_walk_stops():
    d = dps.describe(ROWS, 0.95)
    w = walk(build_problems(ROWS, lambda r: r[dps.POLICY], "cut_off_unsafe"), 0.95)
    assert d["n_stopped"] == int(w.stopped.sum()) == 2
    assert d["n_problems"] == 3
    # a stops at chunk 2 (0.60), b at chunk 1 (0.20); the shares follow
    assert np.isclose(d["prefix_share_mean"], 0.40)
    assert np.isclose(d["forced_share_mean"], 0.05)
    assert d["stop_boundary_mean"] == 1.5
    assert d["stops_at_first_boundary"] == 1


def test_stops_that_save_nothing_are_counted():
    rows = ROWS + [_row("d", 1, 0.99, 100),            # at the trace's end
                   _row("e", 1, 0.99, 97, forced=5)]   # prefix + forced answer exceed the trace
    d = dps.describe(rows, 0.95)
    assert d["n_stopped"] == 4
    assert d["stops_with_zero_saving"] == 2 and d["stops_at_trace_end"] == 1
    assert d["problems_with_a_saving_stop"] == 2
    assert np.isclose(d["share_with_a_saving_stop"], 2 / 5)


def test_an_existing_output_is_never_overwritten(tmp_path):
    import hashlib
    import json
    import pytest
    runs = tmp_path / "runs"
    (runs / "pilot_calibration").mkdir(parents=True)
    (runs / "pilot_eval").mkdir()
    cal = runs / "pilot_calibration" / "calibration.json"
    cal.write_text(json.dumps({"policies": {dps.POLICY: {"tau": 0.95}}}), encoding="utf-8")
    (runs / "pilot_calibration" / "calibration.sha256").write_text(
        hashlib.sha256(cal.read_bytes()).hexdigest() + "\n")
    (runs / "pilot_eval" / "pilot_scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    assert dps.main([str(tmp_path), "--out", "x.json"]) == 0
    first = (runs / "pilot_posthoc" / "x.json").read_bytes()
    with pytest.raises(SystemExit):
        dps.main([str(tmp_path), "--out", "x.json"])
    assert (runs / "pilot_posthoc" / "x.json").read_bytes() == first


def test_a_threshold_nobody_reaches_reports_no_stops():
    assert dps.describe(ROWS, 1.5) == {"n_problems": 3, "n_stopped": 0}
