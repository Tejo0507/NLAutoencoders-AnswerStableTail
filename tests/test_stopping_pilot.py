"""`nlaast.analysis.stopping_pilot`: labels, walks, bounds and SGR calibration.

Synthetic problems only; nothing here reads `runs/`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nlaast.analysis import stopping_pilot as sp


def _p(pid, scores, safe, saving=None, finished=None):
    n = len(scores)
    return sp.Problem(pid, np.array(scores, float), np.array(safe, bool),
                      np.array(saving if saving is not None else [0.1 * (n - i) for i in range(n)], float),
                      np.array(finished if finished is not None else [True] * n, bool))


def test_safety_labels_follow_the_truncation_rule():
    # final answer correct; forced answer correct but cut off at 256
    assert sp.is_safe(True, False, True, "cut_off_unsafe") is False
    assert sp.is_safe(True, False, True, "as_parsed") is True
    assert sp.is_safe(True, False, True, "as_no_answer") is False
    # final answer wrong; no forced answer: correctness agrees (both not correct)
    assert sp.is_safe(None, True, False, "cut_off_unsafe") is True
    with pytest.raises(ValueError):
        sp.is_safe(True, True, True, "lenient")


def test_net_saving_charges_every_generated_token_and_floors_at_zero():
    assert sp.net_saving(100, 20, 400) == pytest.approx(0.7)
    assert sp.net_saving(100, 20, 400, signal_tokens=290) == 0.0


def test_the_walk_stops_at_the_first_boundary_reaching_the_threshold():
    probs = [_p("a", [0.1, 0.6, 0.9], [False, True, True], [0.9, 0.5, 0.1], [True, False, True]),
             _p("b", [0.1, 0.2], [True, True])]
    w = sp.walk(probs, 0.5)
    assert w.stopped.tolist() == [True, False]
    assert w.saving.tolist() == [0.5, 0.0]
    assert w.cut_off.tolist() == [True, False]
    assert w.safe.tolist() == [True, True]          # not stopping counts as safe


def test_clopper_pearson_matches_known_values():
    assert sp.cp_upper(0, 120, 0.95) == pytest.approx(0.0247, abs=1e-4)
    assert sp.cp_upper(0, 0, 0.95) == 1.0
    lo, hi = sp.cp_interval(3, 120)
    assert lo == pytest.approx(0.0052, abs=1e-3) and hi == pytest.approx(0.0713, abs=1e-3)


def test_sgr_finds_the_most_permissive_threshold_with_a_valid_bound():
    rng = np.random.default_rng(0)
    probs = []
    for i in range(400):
        s = np.sort(rng.random(8))
        # boundaries with high scores are safe; low-score boundaries are unsafe 30% of the time
        safe = [(v > 0.5) or (rng.random() > 0.3) for v in s]
        probs.append(_p(str(i), s, safe))
    grid = np.linspace(0, 1, 200)
    res = sp.sgr_threshold(probs, grid, target=0.05, delta=0.05)
    assert res["k"] == math.ceil(math.log2(200))
    assert res["conf_per_step"] == pytest.approx(1 - 0.05 / res["k"])
    assert res["tau"] is not None
    accepted = [s for s in res["steps"] if s["accepted"]]
    assert all(s["bound"] <= 0.05 for s in accepted)
    assert res["tau"] == min(s["tau"] for s in accepted)
    # the chosen threshold's own walk respects the target in sample
    x, n = sp.selective_counts(sp.walk(probs, res["tau"]))
    assert sp.cp_upper(x, n, res["conf_per_step"]) <= 0.05


def test_sgr_returns_none_when_no_threshold_is_safe_enough():
    probs = [_p(str(i), [0.2, 0.8], [False, False]) for i in range(50)]
    res = sp.sgr_threshold(probs, np.linspace(0, 1, 50), target=0.05, delta=0.05)
    assert res["tau"] is None


def test_summaries_and_gates_report_what_they_measure():
    probs = [_p("a", [0.9], [False]), _p("b", [0.9], [True]), _p("c", [0.1], [True])]
    w = sp.walk(probs, 0.5)
    s = sp.summarise(w)
    assert s["n_stopped"] == 2 and s["n_unsafe"] == 1 and s["coverage"] == pytest.approx(2 / 3)
    assert sp.safety_gate(w, 0.05)["reading"] == "bound above alpha"
    never = sp.walk(probs, 2.0)
    assert sp.safety_gate(never, 0.05)["reading"].startswith("never stops")


def test_paired_saving_is_per_problem():
    a = sp.Walk(np.array([True, True]), np.array([True, True]), np.array([0.3, 0.1]), np.array([False, False]))
    b = sp.Walk(np.array([True, False]), np.array([True, True]), np.array([0.1, 0.0]), np.array([False, False]))
    d = sp.paired_saving(a, b, 200, np.random.default_rng(1))
    assert d["mean_diff"] == pytest.approx(0.15) and d["n"] == 2


def test_build_problems_orders_boundaries_and_labels_them():
    rows = [{"problem_id": "p", "chunk": 1, "forced_correct": True, "finished": True,
             "final_correct": True, "prefix_tokens": 50, "forced_tokens": 10, "total_tokens": 100},
            {"problem_id": "p", "chunk": 0, "forced_correct": False, "finished": False,
             "final_correct": True, "prefix_tokens": 20, "forced_tokens": 256, "total_tokens": 100}]
    (p,) = sp.build_problems(rows, lambda r: r["chunk"], "cut_off_unsafe")
    assert p.safe.tolist() == [False, True]
    assert p.saving.tolist() == [0.0, pytest.approx(0.4)]
    assert p.finished.tolist() == [False, True]
