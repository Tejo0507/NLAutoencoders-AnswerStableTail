"""The position-only control and the truncation scenarios (D79)."""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.analysis.o3_registered import (
    Problem,
    bootstrap_p,
    gap_over_budgets,
    no_signal_expectations,
    rule_matrix,
)
from nlaast.analysis.position import (
    majority_answer,
    paired_comparisons,
    position_reading,
    position_scores,
    rate,
    relabelled,
    residualise_on_position,
    safe_vector,
    scenario_answers,
    stop_cut_off_share,
    within_problem_spearman,
)
from nlaast.baselines.stopping import OutcomeUnknown


def eq(a, b):
    return a is not None and b is not None and a == b


def _menu_world(n_problems=40, seed=0):
    """Problems shaped like design A's menu: symmetric about a tail start,
    safe exactly at and after it. Returns problems, menu, prefix tokens."""
    rng = np.random.default_rng(seed)
    probs, menu, prefix = [], {}, {}
    for j in range(n_problems):
        n = int(rng.integers(8, 16))
        tokens = np.cumsum(rng.integers(10, 40, n))
        tail = int(rng.integers(n // 2, n - 1))
        L = n - tail
        m = np.zeros(n, bool)
        for i in (tail - L, tail - 1, tail, n - 1):
            if 0 <= i < n:
                m[i] = True
        pid = f"p{j}"
        probs.append(Problem(pid, 1 - tokens / tokens[-1], np.arange(n) >= tail, np.ones(n, bool)))
        menu[pid] = m
        prefix[pid] = [int(t) for t in tokens]
    return probs, menu, prefix


# ---- position scores and eligibility ----

def test_position_scores_kinds():
    assert position_scores([5, 9, 20], 40, "tokens") == [5.0, 9.0, 20.0]
    assert position_scores([5, 9, 20], 40, "index") == [0.0, 1.0, 2.0]
    assert position_scores([5, 9, 20], 40, "relative") == [0.125, 0.225, 0.5]
    assert position_scores([5, 9, 20], 40, "tokens_falling") == [-5.0, -9.0, -20.0]
    with pytest.raises(ValueError):
        position_scores([5], 0, "relative")
    with pytest.raises(ValueError):
        position_scores([5], 10, "nla")


def test_position_rule_stops_only_on_the_menu_at_the_first_chunk_past_the_threshold():
    probs, menu, prefix = _menu_world()
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs}
    rm = rule_matrix(probs, sc, menu)
    for r, p in enumerate(probs):
        allowed = np.flatnonzero(menu[p.problem_id])
        for k, t in enumerate(rm.thresholds):
            b = rm.boundary[r, k]
            if not np.isfinite(t):
                assert b == -1                               # +inf never stops
                continue
            past = [i for i in allowed if prefix[p.problem_id][i] >= t]
            assert b == (past[0] if past else -1)


def test_position_rule_ignores_outcomes():
    """Where it stops cannot depend on the labels: relabelling safety leaves
    every stopping boundary unchanged."""
    probs, menu, prefix = _menu_world()
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs}
    flipped = [relabelled(p, ~p.safe) for p in probs]
    a, b = rule_matrix(probs, sc, menu), rule_matrix(flipped, sc, menu)
    assert np.array_equal(a.boundary, b.boundary)
    assert np.array_equal(a.saving, b.saving)
    assert np.array_equal(a.safe, ~b.safe | ~a.fired)


def test_position_rule_covers_the_whole_saving_range_on_the_menu():
    probs, menu, prefix = _menu_world()
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs}
    sav, saf = rule_matrix(probs, sc, menu).curve()
    first = np.mean([p.saving[np.flatnonzero(menu[p.problem_id])[0]] for p in probs])
    assert sav[0] == pytest.approx(first)          # lowest threshold: first menu chunk
    assert sav[-1] == 0.0 and saf[-1] == 1.0       # +inf: never stops, safe


def test_falling_position_stops_only_at_the_first_menu_chunk_or_never():
    """A score that falls along the trace has one record, the first menu
    chunk, so the rule chooses *which problems* to stop in, not where."""
    probs, menu, prefix = _menu_world()
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens_falling") for p in probs}
    rm = rule_matrix(probs, sc, menu)
    for r, p in enumerate(probs):
        first = int(np.flatnonzero(menu[p.problem_id])[0])
        assert set(rm.boundary[r].tolist()) <= {first, -1}
        assert first in rm.boundary[r]


def test_unmeasured_menu_outcome_raises_for_the_position_rule():
    probs, menu, prefix = _menu_world(3)
    p = probs[0]
    i = int(np.flatnonzero(menu[p.problem_id])[-1])
    known = p.known.copy()
    known[i] = False
    probs[0] = Problem(p.problem_id, p.saving, p.safe, known)
    sc = {q.problem_id: position_scores(prefix[q.problem_id], 0, "tokens") for q in probs}
    with pytest.raises(OutcomeUnknown):
        rule_matrix(probs, sc, menu)


def test_position_reaches_the_oracle_when_every_tail_starts_at_the_same_place():
    """With identical problems, the threshold just past i*-1 stops every
    problem exactly at the tail start: all safe, at the oracle saving."""
    probs, menu, prefix = [], {}, {}
    tokens = np.arange(1, 11) * 10                      # 10 chunks, tail at 6
    for j in range(5):
        pid = f"p{j}"
        probs.append(Problem(pid, 1 - tokens / 100, np.arange(10) >= 6, np.ones(10, bool)))
        menu[pid] = np.isin(np.arange(10), [2, 5, 6, 9])
        prefix[pid] = list(tokens)
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 100, "relative") for p in probs}
    rm = rule_matrix(probs, sc, menu)
    sav, saf = rm.curve()
    k = int(np.flatnonzero(np.isclose(rm.thresholds, 0.7))[0])
    assert np.all(rm.boundary[:, k] == 6) and saf[k] == 1.0
    assert sav[k] == pytest.approx(0.3)


def test_position_need_not_beat_no_signal_when_tail_starts_vary():
    """Why D79's X2 is an empirical question: the no-signal rule's abstentions
    are safe, while one position threshold stops late-tail problems before
    their tail. In this world position *loses* to no signal."""
    probs, menu, prefix = _menu_world(60)
    sc = {p.problem_id: position_scores(prefix[p.problem_id], prefix[p.problem_id][-1], "relative")
          for p in probs}
    rm = rule_matrix(probs, sc, menu)
    ES, EQ = no_signal_expectations(probs, menu, np.linspace(0, 1, 2001))
    g, k, _ = gap_over_budgets(rm.curve(), (ES.mean(0), EQ.mean(0)), [0.05, 0.1, 0.15], 0.02)
    assert k == 3 and g < 0


# ---- residualisation ----

def _resid_inputs(rng, n=8, exact=True):
    scores, prefix, menu = {}, {}, {}
    for j in range(n):
        t = np.sort(rng.integers(10, 400, 5))
        y = 0.3 + 0.1 * np.log(t) + (0 if exact else rng.normal(0, 0.05, 5))
        s = [float(v) for v in y]
        s[1] = None                                     # an unscored boundary
        pid = f"p{j}"
        scores[pid], prefix[pid] = s, [int(v) for v in t]
        menu[pid] = np.array([True, True, True, False, True])
    return scores, prefix, menu


def test_residuals_vanish_when_the_score_is_exactly_a_position_trend():
    scores, prefix, menu = _resid_inputs(np.random.default_rng(1))
    res = residualise_on_position(scores, prefix, menu)
    for pid, r in res.items():
        assert r[1] is None and r[3] is None             # unscored, off menu
        assert all(abs(v) < 1e-9 for v in r if v is not None)


def test_residualisation_is_leave_one_problem_out():
    """A problem's own scores never set its trend: shifting them shifts its
    residuals by exactly the shift."""
    scores, prefix, menu = _resid_inputs(np.random.default_rng(2), exact=False)
    base = residualise_on_position(scores, prefix, menu)
    shifted = {**scores, "p0": [None if v is None else v + 0.5 for v in scores["p0"]]}
    after = residualise_on_position(shifted, prefix, menu)
    for a, b in zip(base["p0"], after["p0"]):
        assert (a is None) == (b is None)
        if a is not None:
            assert b - a == pytest.approx(0.5)


def test_residualisation_refuses_an_unidentified_trend():
    scores = {"a": [0.1, 0.2], "b": [0.3, 0.4]}
    prefix = {"a": [10, 20], "b": [10, 10]}             # b alone: one position
    menu = {"a": np.array([True, True]), "b": np.array([True, True])}
    with pytest.raises(ValueError):
        residualise_on_position(scores, prefix, menu)


def test_spearman_with_position():
    scores = {"a": [1, 2, 3, None], "b": [3, 2, 1, 0], "c": [1, 1, 1, 1], "d": [1, 2]}
    pos = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4], "d": [1, 2]}
    menu = {k: np.ones(len(v), bool) for k, v in pos.items()}
    s = within_problem_spearman(scores, pos, menu)
    assert s["n_problems"] == 2 and s["n_skipped"] == 2
    assert s["mean_rho"] == pytest.approx(0.0) and s["share_positive"] == 0.5


# ---- the shared bootstrap ----

def test_paired_comparisons_reproduce_the_registered_bootstrap_loop():
    probs, menu, prefix = _menu_world(30, seed=3)
    rng0 = np.random.default_rng(4)
    sc = {"a": {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs},
          "b": {p.problem_id: list(rng0.random(p.n)) for p in probs}}
    mats = {r: rule_matrix(probs, sc[r], menu) for r in sc}
    ES, EQ = no_signal_expectations(probs, menu, np.linspace(0, 1, 2001))
    budgets, tol, reps = [0.05, 0.1, 0.15], 0.02, 200
    got = paired_comparisons(len(probs), mats, (ES, EQ), {"a - b": ("a", "b"),
                             "a - no_signal": ("a", "no_signal")}, budgets, tol, reps,
                             np.random.default_rng(9), 3)
    # the loop reanalyse_o3_registered.analyse runs, written out
    rng = np.random.default_rng(9)
    n = len(probs)
    boot = np.full(reps, np.nan)
    for i in range(reps):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        g, k, _ = gap_over_budgets(mats["a"].curve(w), mats["b"].curve(w), budgets, tol)
        if k >= 3:
            boot[i] = g
    ok = boot[np.isfinite(boot)]
    assert got["a - b"]["p_bootstrap"] == bootstrap_p(boot, 1 / reps)
    assert got["a - b"]["ci95"] == [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
    assert got["a - b"]["gap"] == gap_over_budgets(mats["a"].curve(), mats["b"].curve(),
                                                   budgets, tol)[0]


def test_a_rule_against_itself_has_zero_gap():
    probs, menu, prefix = _menu_world(20)
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs}
    m = rule_matrix(probs, sc, menu)
    got = paired_comparisons(len(probs), {"a": m, "b": m}, None, {"a - b": ("a", "b")},
                             [0.05, 0.1, 0.15], 0.02, 50, np.random.default_rng(0), 3)
    assert got["a - b"]["gap"] == 0.0 and got["a - b"]["ci95"] == [0.0, 0.0]


def test_too_few_matched_budgets_is_not_evaluable():
    probs, menu, prefix = _menu_world(20)
    sc = {p.problem_id: position_scores(prefix[p.problem_id], 0, "tokens") for p in probs}
    m = rule_matrix(probs, sc, menu)
    got = paired_comparisons(len(probs), {"a": m, "b": m}, None, {"a - b": ("a", "b")},
                             [5.0, 6.0, 7.0], 0.02, 20, np.random.default_rng(0), 3)
    assert not got["a - b"]["evaluable"] and got["a - b"]["ci95"] is None
    assert np.isnan(got["a - b"]["p_bootstrap"])


# ---- readings ----

def _gaps(x1, x1ci, x4):
    return {"readout - position_tokens": {"gap": x1, "ci95": x1ci},
            "position_tokens - no_signal": {"gap": 0.05, "ci95": [0.01, 0.09]},
            "readout_residualised - no_signal": {"gap": x4, "ci95": [0.0, 0.1]}}


@pytest.mark.parametrize("x1,ci,rej1,x4,rej4,overall", [
    (0.06, [0.02, 0.1], True, 0.04, True, "beyond position"),
    (0.06, [0.02, 0.1], True, 0.04, False, "unresolved"),
    (0.01, [-0.03, 0.04], False, 0.04, True, "excluded"),
    (-0.06, [-0.1, -0.02], True, 0.04, True, "excluded"),
    (0.02, [-0.02, 0.07], False, 0.04, True, "unresolved"),
])
def test_position_reading(x1, ci, rej1, x4, rej4, overall):
    h = {"readout - position_tokens": {"reject": rej1},
         "position_tokens - no_signal": {"reject": True},
         "readout_residualised - no_signal": {"reject": rej4}}
    r = position_reading(_gaps(x1, ci, x4), h)
    assert overall in r["overall"]
    assert r["position_beats_no_signal"] is True


@pytest.mark.parametrize("y1,ci,rej1,y2,rej2,overall", [
    (0.01, [-0.03, 0.05], False, 0.05, True, "reproduces"),
    (0.08, [0.03, 0.12], True, 0.05, True, "beats the falling"),
    (0.01, [-0.03, 0.05], False, 0.05, False, "unresolved"),
    (0.01, [-0.03, 0.05], False, -0.05, True, "unresolved"),     # falling loses to chance
    (-0.08, [-0.12, -0.03], True, 0.05, True, "unresolved"),     # readout worse
])
def test_falling_reading(y1, ci, rej1, y2, rej2, overall):
    from nlaast.analysis.position import falling_reading
    g = {"readout - position_tokens_falling": {"gap": y1, "ci95": ci},
         "position_tokens_falling - no_signal": {"gap": y2, "ci95": None}}
    h = {"readout - position_tokens_falling": {"reject": rej1},
         "position_tokens_falling - no_signal": {"reject": rej2}}
    assert overall in falling_reading(g, h)["overall"]


# ---- truncation ----

def test_scenario_answers_change_only_known_cut_offs():
    ans, cut = ["3", "4", None, "5"], [True, False, None, True]
    assert scenario_answers(ans, cut, "9", "registered") == ans
    assert scenario_answers(ans, cut, "9", "uncommitted") == [None, "4", None, None]
    assert scenario_answers(ans, cut, "9", "committed_to_final") == ["9", "4", None, "9"]
    with pytest.raises(ValueError):
        scenario_answers(ans, cut[:3], "9", "uncommitted")
    with pytest.raises(ValueError):
        scenario_answers(ans, cut, "9", "optimistic")


def test_safe_vector_is_the_registered_definition():
    # final correct: safe iff the forced answer is correct
    assert safe_vector(["7", "8", None], [True] * 3, "7", "7", eq).tolist() == [True, False, False]
    # final incorrect: safe iff the forced answer is incorrect too, including none
    assert safe_vector(["7", "8", None], [True] * 3, "7", "8", eq).tolist() == [False, True, True]
    # unknown is never labelled safe (rule_matrix refuses to score it)
    assert safe_vector(["7"], [False], "7", "7", eq).tolist() == [False]


def test_uncommitted_cut_off_is_unsafe_only_when_the_final_answer_is_right():
    ans = scenario_answers(["7", "7"], [True, False], "7", "uncommitted")
    assert safe_vector(ans, [True, True], "7", "7", eq).tolist() == [False, True]
    ans = scenario_answers(["7", "7"], [True, False], "8", "uncommitted")
    assert safe_vector(ans, [True, True], "7", "8", eq).tolist() == [True, False]


def test_committed_to_final_makes_every_cut_off_safe():
    for final, gold in (("7", "7"), ("8", "7")):
        ans = scenario_answers(["1", "2"], [True, True], final, "committed_to_final")
        assert safe_vector(ans, [True, True], gold, final, eq).all()


def test_majority_answer():
    assert majority_answer(["5", "5", "5", "6", None], eq) == "5"
    assert majority_answer(["5", "5", "6", "6", "7"], eq) is None
    assert majority_answer(["5", "5", None, None, None], eq) is None    # 2 of 5
    assert majority_answer([], eq) is None


def test_majority_uses_answer_equivalence():
    from nlaast.data.answers import equivalent
    assert majority_answer(["160", "160 minutes.**", "160.0", "40", None], equivalent) == "160"


def test_stop_cut_off_share_excludes_unknown_status_from_the_denominator():
    probs = [Problem(f"p{j}", np.array([0.5, 0.2]), np.array([True, True]), np.ones(2, bool))
             for j in range(4)]
    sc = {"p0": [1, 0], "p1": [1, 0], "p2": [1, 0], "p3": [None, None]}
    rm = rule_matrix(probs, sc, None)
    cut = {("p0", 0): True, ("p1", 0): False, ("p2", 0): None}
    s = stop_cut_off_share(rm, [0, None], probs, cut)
    assert s[1] is None
    assert s[0] == {"n_stops": 3, "n_status_unknown": 1, "share_cut_off": 0.5}


def test_rate_groups_and_skips_unknown_flags():
    recs = [{"d": "a", "f": True}, {"d": "a", "f": None}, {"d": "b", "f": False}]
    assert rate(recs, "f")["all"] == {"n": 3, "n_unknown": 1, "rate": 0.5}
    by = rate(recs, "f", "d")
    assert by["a"] == {"n": 2, "n_unknown": 1, "rate": 1.0}
    assert by["b"]["rate"] == 0.0
    assert rate([{"f": None}], "f")["all"]["rate"] is None
