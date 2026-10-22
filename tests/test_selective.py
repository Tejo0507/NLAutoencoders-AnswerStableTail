"""Selective risk at matched coverage (D83 deviation 1), on synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.analysis.o3_registered import Problem, RuleMatrix, no_signal_expectations, rule_matrix
from nlaast.analysis.selective import (
    at_coverage,
    fire_probability,
    no_signal_selective,
    rule_selective,
    selective_comparisons,
)


def test_rule_selective_by_hand():
    # 4 problems, one threshold column: 3 fire (2 safe, 1 unsafe), 1 abstains
    rm = RuleMatrix(thresholds=np.array([0.5]),
                    saving=np.array([[0.6], [0.2], [0.4], [0.0]]),
                    safe=np.array([[True], [True], [False], [True]]),
                    fired=np.array([[True], [True], [True], [False]]),
                    boundary=np.array([[1], [2], [0], [-1]]))
    cov, risk, sav = rule_selective(rm, np.ones(4))
    assert cov[0] == pytest.approx(0.75)
    assert risk[0] == pytest.approx(1 / 3)
    assert sav[0] == pytest.approx(0.4)
    # weights act as replicated problems
    cov2, risk2, _ = rule_selective(rm, np.array([2.0, 0, 1, 1]))
    assert cov2[0] == pytest.approx(0.75) and risk2[0] == pytest.approx(1 / 3)


def test_no_signal_closed_form_matches_simulation():
    rng = np.random.default_rng(0)
    probs = [Problem(f"p{k}", np.linspace(0.9, 0.0, 6), np.array([0, 0, 1, 0, 1, 1], bool),
                     np.ones(6, bool)) for k in range(3)]
    menu = {p.problem_id: np.array([1, 1, 1, 0, 1, 1], bool) for p in probs}
    q = np.array([0.3])
    FP = fire_probability(probs, menu, q)
    ES, EQ = no_signal_expectations(probs, menu, q)
    cov, risk, sav = no_signal_selective(FP, ES, EQ, np.ones(3))
    fired = unsafe = saving = 0.0
    draws = 40000
    for _ in range(draws):
        for p in probs:
            for i in np.flatnonzero(menu[p.problem_id]):
                if rng.random() < q[0]:
                    fired += 1
                    unsafe += not p.safe[i]
                    saving += p.saving[i]
                    break
    assert cov[0] == pytest.approx(fired / (3 * draws), abs=0.005)
    assert risk[0] == pytest.approx(unsafe / fired, abs=0.01)
    assert sav[0] == pytest.approx(saving / fired, abs=0.01)


def test_comparisons_pick_the_matched_column_and_bootstrap_is_paired():
    rng = np.random.default_rng(1)
    probs = [Problem(f"p{k}", np.linspace(0.9, 0.0, 8), np.arange(8) >= 4, np.ones(8, bool))
             for k in range(60)]
    scores_good = {p.problem_id: list(np.arange(8) / 8 + rng.normal(0, 0.01, 8)) for p in probs}
    rm = rule_matrix(probs, scores_good, None)
    q = np.linspace(0, 1, 2001)
    ES, EQ = no_signal_expectations(probs, None, q)
    FP = fire_probability(probs, None, q)
    out = selective_comparisons({"good": rm}, (FP, ES, EQ), [0.5], 0.05, 200,
                                np.random.default_rng(0))
    row = out["0.5"]
    # stopping at the latest boundaries first, a rising score is always safe when it fires
    assert row["good"]["at"]["selective_risk"] == 0.0
    assert row["no_signal"]["selective_risk"] > 0.4
    assert row["good"]["ci95"][1] < 0
    assert at_coverage((np.array([0.1, 0.9]), np.array([0, 0]), np.array([0, 0])), 0.5, 0.05) is None
