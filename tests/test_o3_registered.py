"""The matrix form of the stopping curves (D69) agrees with ``decisions_at``."""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.analysis.o3_registered import (
    Problem,
    bootstrap_p,
    gap_over_budgets,
    holm,
    match,
    no_signal_expectations,
    no_signal_monte_carlo,
    oracle_bound,
    rule_matrix,
)
from nlaast.baselines.stopping import OutcomeUnknown, decisions_at


def _random_problems(rng, n=12):
    probs, scores = [], {}
    for j in range(n):
        m = int(rng.integers(3, 9))
        tokens = np.sort(rng.integers(5, 200, m))
        total = int(tokens[-1] + rng.integers(1, 50))
        safe = rng.random(m) < 0.5
        pid = f"p{j}"
        probs.append(Problem(pid, 1 - tokens / total, safe, np.ones(m, bool)))
        sc = list(rng.random(m).round(2))           # ties on purpose
        sc[int(rng.integers(0, m))] = None           # an unscored boundary
        scores[pid] = sc
    return probs, scores


def _as_rows(probs, scores):
    """The same problems in ``decisions_at``'s row format: safety is encoded
    by making the stop answer equal the gold when safe (final is correct)."""
    rows = []
    for p in probs:
        total = 1000
        rows.append({"problem_id": p.problem_id, "scores": scores[p.problem_id],
                     "answers": ["1" if s else "2" for s in p.safe],
                     "boundary_tokens": [int(round((1 - s) * total)) for s in p.saving],
                     "final_answer": "1", "gold": "1", "total_tokens": total})
    return rows


def test_matrix_curve_matches_decisions_at_at_every_threshold():
    rng = np.random.default_rng(3)
    probs, scores = _random_problems(rng)
    rm = rule_matrix(probs, scores)
    rows = _as_rows(probs, scores)
    for k, t in enumerate(rm.thresholds):
        dec = decisions_at(rows, t if np.isfinite(t) else 2.0, True)
        assert [d.boundary if d.boundary is not None else -1 for d in dec] == list(rm.boundary[:, k])
        assert [d.safe for d in dec] == list(rm.safe[:, k])


def test_menu_restricts_where_a_rule_can_stop():
    p = Problem("a", np.array([0.9, 0.5, 0.1]), np.array([False, True, True]), np.ones(3, bool))
    rm = rule_matrix([p], {"a": [0.9, 0.1, 0.5]}, {"a": np.array([False, True, True])})
    assert set(rm.boundary[0]) == {1, 2, -1}         # never boundary 0
    unrestricted = rule_matrix([p], {"a": [0.9, 0.1, 0.5]})
    assert set(unrestricted.boundary[0]) == {0, -1}  # 0.9 is the first record


def test_unmeasured_outcome_that_could_be_chosen_raises():
    p = Problem("a", np.array([0.9, 0.5]), np.array([True, True]), np.array([False, True]))
    with pytest.raises(OutcomeUnknown):
        rule_matrix([p], {"a": [0.9, 0.1]})
    # unreachable unknown (score never a record) is fine
    rule_matrix([p], {"a": [None, 0.1]})
    with pytest.raises(OutcomeUnknown):
        oracle_bound([p])


def test_oracle_is_the_mean_best_safe_saving():
    a = Problem("a", np.array([0.9, 0.5, 0.1]), np.array([False, True, True]), np.ones(3, bool))
    b = Problem("b", np.array([0.8, 0.2]), np.array([False, False]), np.ones(2, bool))
    assert oracle_bound([a, b]) == pytest.approx(0.25)
    assert oracle_bound([a], {"a": np.array([False, False, True])}) == pytest.approx(0.1)


def test_no_signal_closed_form_agrees_with_monte_carlo():
    rng = np.random.default_rng(5)
    probs, _ = _random_problems(rng, n=40)
    qgrid = np.linspace(0, 1, 2001)
    ES, EQ = no_signal_expectations(probs, None, qgrid)
    targets = [0.15, 0.3]
    closed = []
    for b in targets:
        k = match(ES.mean(0), b, 0.02)
        closed.append(EQ.mean(0)[k])
    mc = no_signal_monte_carlo(probs, None, targets, 0.02, 400, np.random.default_rng(6))
    for c, m in zip(closed, mc):
        assert c == pytest.approx(m, abs=0.03)


def test_no_signal_never_stops_at_q_zero_and_always_at_the_first_boundary_at_q_one():
    p = Problem("a", np.array([0.9, 0.5]), np.array([False, True]), np.ones(2, bool))
    ES, EQ = no_signal_expectations([p], None, np.array([0.0, 1.0]))
    assert ES[0].tolist() == [0.0, 0.9] and EQ[0].tolist() == [1.0, 0.0]


def test_gap_counts_only_budgets_both_rules_reach():
    a = (np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.8, 0.7]))
    b = (np.array([0.1, 0.3]), np.array([0.5, 0.5]))
    mean, n, diffs = gap_over_budgets(a, b, [0.1, 0.2, 0.3], tol=0.02)
    assert n == 2 and diffs[1] is None
    assert mean == pytest.approx((0.4 + 0.2) / 2)


def test_holm_and_bootstrap_p():
    h = holm({"x": 0.01, "y": 0.04, "z": float("nan")})
    assert h["x"]["p_holm"] == pytest.approx(0.02) and h["y"]["p_holm"] == pytest.approx(0.04)
    assert h["z"]["reject"] is None
    assert bootstrap_p(np.array([1.0] * 99 + [-1.0]), 1e-3) == pytest.approx(0.02)
    assert bootstrap_p(np.ones(10), 0.05) == 0.05


def _confirmatory():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from reanalyse_o3_registered import confirmatory_verdict
    return confirmatory_verdict


@pytest.mark.parametrize("gap, ci, p, reading", [
    (0.06, [0.03, 0.09], 0.001, "supported"),
    (-0.06, [-0.09, -0.03], 0.001, "reversed: the comparator is better"),
    (0.01, [-0.03, 0.04], 0.6, "equivalent within ±0.05"),
    (-0.04, [-0.09, 0.02], 0.3, "advantage of 0.05 or more excluded"),
    (0.03, [-0.02, 0.08], 0.2, "inconclusive"),
])
def test_confirmatory_readings(gap, ci, p, reading):
    cv = _confirmatory()
    entry = {"gap": gap, "ci95": ci, "p_bootstrap": p, "evaluable": True, "n_budgets_matched": 9}
    other = {"gap": 0.05, "ci95": [0.02, 0.08], "p_bootstrap": 0.001, "evaluable": True,
             "n_budgets_matched": 9}
    out = cv({"gaps": {"nla_readout - probe": entry, "nla_readout - no_signal": other}})
    assert out["H1_readout_minus_probe"]["reading"] == reading
    assert out["H2_readout_minus_no_signal"]["reading"] == "supported"


def test_weighted_curve_equals_duplicating_problems():
    rng = np.random.default_rng(9)
    probs, scores = _random_problems(rng, n=5)
    rm = rule_matrix(probs, scores)
    w = np.array([2, 0, 1, 1, 1])
    dup = [probs[0], probs[0], probs[2], probs[3], probs[4]]
    rm2 = rule_matrix(dup, scores)
    s1, q1 = rm.curve(w)
    s2, q2 = rm2.curve()
    # same thresholds except possibly values only problem 1 contributed
    for k, t in enumerate(rm2.thresholds):
        j = int(np.flatnonzero(rm.thresholds == t)[0])
        assert s1[j] == pytest.approx(s2[k]) and q1[j] == pytest.approx(q2[k])
