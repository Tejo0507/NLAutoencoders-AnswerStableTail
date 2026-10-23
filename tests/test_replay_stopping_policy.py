"""The stopping-policy replay's arithmetic, on synthetic problems.

No stored run is read: these check the walk, the threshold choices, the
paired spread, the sample-size helpers and the guard that keeps the replay
from writing inside the runs directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import replay_stopping_policy as rsp  # noqa: E402


def unit(pid, chunk, safe, prefix, forced=20, total=200):
    return {"problem_id": pid, "chunk": chunk, "as_stored": safe, "prefix_tokens": prefix,
            "forced_tokens": forced, "total_tokens": total}


@pytest.fixture
def probs():
    units = [unit("a", 0, False, 40), unit("a", 1, True, 100),
             unit("b", 0, True, 50), unit("b", 1, True, 150)]
    scores = [0.3, 0.9, 0.8, 0.95]
    return rsp.problems_from_units(units, scores, "as_stored")


def test_the_walk_stops_at_the_first_candidate_over_tau(probs):
    st, sf, sv = rsp.walk(probs, 0.85, ["a", "b"])
    assert st.tolist() == [True, True]
    assert sf.tolist() == [True, True]
    # a stops at chunk 1 (prefix 100), b at chunk 1 (prefix 150; chunk 0 is 0.8)
    assert sv.tolist() == pytest.approx([1 - 120 / 200, 1 - 170 / 200])


def test_candidates_are_walked_in_chunk_order_not_score_order():
    units = [unit("a", 1, True, 100), unit("a", 0, False, 40)]
    p = rsp.problems_from_units(units, [0.9, 0.9], "as_stored")
    st, sf, _ = rsp.walk(p, 0.5, ["a"])
    assert st[0] and not sf[0]                 # chunk 0 comes first, and is unsafe


def test_a_problem_that_never_stops_saves_nothing_and_is_not_covered(probs):
    st, sf, sv = rsp.walk(probs, 0.99, ["a", "b"])
    assert not st.any() and sv.tolist() == [0.0, 0.0]
    s = rsp.summary(st, sf, sv)
    assert s["coverage"] == 0 and s["selective_risk"] is None


def test_net_saving_charges_the_forced_tokens_and_is_floored_at_zero():
    p = rsp.problems_from_units([unit("a", 0, True, 190, forced=48)], [1.0], "as_stored")
    assert rsp.walk(p, 0.5, ["a"])[2][0] == 0.0


def test_the_risk_threshold_is_the_most_permissive_that_meets_the_target(probs):
    grid = rsp.tau_grid(probs)
    tau = rsp.tau_at_risk(probs, ["a", "b"], 0.0, grid)
    st, sf, _ = rsp.walk(probs, tau, ["a", "b"])
    assert (~sf[st]).mean() == 0.0
    lower = [t for t in grid if t < tau]
    assert all((lambda w: w[0].any() and (~w[1][w[0]]).mean() > 0)(rsp.walk(probs, t, ["a", "b"]))
               or not rsp.walk(probs, t, ["a", "b"])[0].any() for t in lower)


def test_the_paired_spread_is_the_sd_of_per_problem_differences():
    r = rsp.paired(np.array([0.5, 0.2, 0.0]), np.array([0.1, 0.2, 0.3]))
    assert r["mean_diff"] == pytest.approx(0.1 / 3)
    assert r["sd_diff"] == pytest.approx(np.std([0.4, 0.0, -0.3], ddof=1))


def test_the_sample_size_helpers_match_the_proposal_tables():
    assert rsp.c1_n_stopped(0.025, 0.05) == 386
    assert rsp.c1_n_stopped(0.0125, 0.05) == 153
    assert rsp.c1_n_stopped(0.06, 0.05) is None
    assert rsp.c2_n(0.25, 0.05) == {"n_power80": 197, "n_halfwidth": 385}


def test_the_replay_refuses_to_write_inside_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(sys, "argv", ["replay", "--runs", str(runs), "--out", str(runs / "x")])
    assert rsp.main() == 1
    assert not (runs / "x").exists()


def test_the_replay_refuses_to_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["replay", "--runs", str(tmp_path / "runs"),
                                      "--out", str(out)])
    assert rsp.main() == 1
