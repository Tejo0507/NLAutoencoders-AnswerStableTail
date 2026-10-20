"""The post-hoc audit quantities (D68), on cases whose answers are known."""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.analysis.posthoc import (
    cluster_bootstrap_mean,
    decompose_decisions,
    oracle_safe_saving,
    paired_problem_means,
    random_first_boundary_null,
)
from nlaast.baselines.stopping import StopDecision, decisions_at


def _dec(pid, boundary, safe, used=10, total=100):
    return StopDecision(problem_id=pid, rule="r", boundary=boundary,
                        tokens_used=used if boundary is not None else total,
                        total_tokens=total, answer_at_stop=None, final_answer=None,
                        gold=None, safe=safe)


def test_decomposition_separates_abstention_from_safe_stops():
    d = [_dec("a", None, True), _dec("b", None, True),
         _dec("c", 0, True), _dec("d", 0, False), _dec("e", 3, False)]
    out = decompose_decisions(d, {"c": 2, "d": 0, "e": 1})
    assert out["safe_rate"] == pytest.approx(3 / 5)
    assert out["n_abstained"] == 2 and out["n_safe_stops"] == 1
    assert out["share_of_safe_that_is_abstention"] == pytest.approx(2 / 3)
    assert out["safe_given_fired"] == pytest.approx(1 / 3)
    assert out["share_fired_at_first_boundary"] == pytest.approx(2 / 3)
    # c fired 2 before its tail, d at it, e 2 after
    assert out["share_fired_before_tail"] == pytest.approx(1 / 3)


def test_never_firing_rule_is_all_abstention():
    # The property the decomposition exists to expose: safe by definition.
    out = decompose_decisions([_dec(p, None, True) for p in "abc"])
    assert out["safe_rate"] == 1.0 and out["n_safe_stops"] == 0
    assert out["mean_tokens_saved"] == 0.0


def test_decomposition_agrees_with_decisions_at():
    rows = [{"problem_id": "p", "scores": [0.1, 0.9], "answers": ["1", "2"],
             "boundary_tokens": [10, 20], "final_answer": "2", "gold": "2",
             "total_tokens": 40},
            {"problem_id": "q", "scores": [0.1, 0.2], "answers": ["3", "4"],
             "boundary_tokens": [10, 20], "final_answer": "4", "gold": "4",
             "total_tokens": 40}]
    out = decompose_decisions(decisions_at(rows, 0.5, True))
    # p fires at boundary 1 on the right answer (safe); q never fires (safe).
    assert out["n_safe_stops"] == 1 and out["n_abstained"] == 1


def test_random_first_boundary_null_has_the_expected_mean():
    # Every first-boundary stop saves 0.5 and is unsafe. To reach a mean
    # saving of 0.25 the rule stops on half the problems, so the safe rate is
    # exactly the abstaining half - in every draw.
    out = random_first_boundary_null([0.5] * 10, [False] * 10, 0.25, 200,
                                     np.random.default_rng(0))
    assert out["mean_safe_rate"] == pytest.approx(0.5)
    assert out["interval"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert out["mean_achieved_saving"] == pytest.approx(0.25)


def test_random_first_boundary_null_counts_safe_stops():
    # Same budget, but every stop is safe: the rule is always fully safe.
    out = random_first_boundary_null([0.5] * 10, [True] * 10, 0.25, 50,
                                     np.random.default_rng(1))
    assert out["mean_safe_rate"] == pytest.approx(1.0)


def test_oracle_takes_the_largest_safe_saving_and_abstains_otherwise():
    rows = [{"boundary_tokens": [10, 50, 90], "total_tokens": 100, "safe": [False, True, True]},
            {"boundary_tokens": [10, 50], "total_tokens": 100, "safe": [False, False]}]
    out = oracle_safe_saving(rows, lambda r, i: r["safe"][i])
    # problem 1: safe at 50 -> saves 0.5; problem 2: nowhere safe -> abstain, 0.
    assert out["mean_max_safe_saving"] == pytest.approx(0.25)
    assert out["is_lower_bound"] is False


def test_oracle_with_unknown_boundaries_is_a_lower_bound():
    rows = [{"boundary_tokens": [10, 50], "total_tokens": 100, "safe": [None, True]}]
    out = oracle_safe_saving(rows, lambda r, i: r["safe"][i])
    assert out["mean_max_safe_saving"] == pytest.approx(0.5)
    assert out["n_boundaries_unknown"] == 1 and out["is_lower_bound"] is True


def test_cluster_bootstrap_is_wider_than_treating_units_as_independent():
    rng = np.random.default_rng(0)
    # 20 clusters of 50 near-identical units: the effective n is about 20.
    centres = rng.normal(0, 1, 20)
    vals = np.concatenate([c + rng.normal(0, 0.01, 50) for c in centres])
    clus = np.repeat(np.arange(20), 50)
    out = cluster_bootstrap_mean(vals, clus, 2000, np.random.default_rng(1))
    naive_half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    assert (out["high"] - out["low"]) / 2 > 3 * naive_half
    assert out["n_clusters"] == 20 and out["n_units"] == 1000


def test_paired_problem_means_pairs_and_averages_within_problem():
    recs = [{"problem_id": "p", "k": "tail", "v": 1.0},
            {"problem_id": "p", "k": "tail", "v": 3.0},
            {"problem_id": "p", "k": "ctrl", "v": 1.0},
            {"problem_id": "q", "k": "tail", "v": 5.0},   # no control: dropped
            {"problem_id": "r", "k": "ctrl", "v": 4.0},   # no tail: dropped
            {"problem_id": "s", "k": "tail", "v": 0.0},
            {"problem_id": "s", "k": "ctrl", "v": 2.0}]
    pids, a, b = paired_problem_means(recs, "k", "v", "tail", "ctrl")
    assert pids == ["p", "s"]
    assert a.tolist() == [2.0, 0.0] and b.tolist() == [1.0, 2.0]
