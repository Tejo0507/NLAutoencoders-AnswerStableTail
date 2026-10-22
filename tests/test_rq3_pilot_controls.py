"""The RQ3 pilot's controls and outcome reading (D74), checked on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.causal.controls import (
    degenerate,
    mean_removed_norm,
    permuted_label_directions,
    read_outcome,
    second_moment,
    variance_matched_directions,
    within_tolerance,
)
from nlaast.causal.interventions import projection_editor, random_direction


def _anisotropic(rng, n=400, d=64):
    scales = np.geomspace(10.0, 0.1, d)
    X = rng.standard_normal((n, d)) * scales
    X[:, 0] += 30.0                     # a large shared offset, as in residual streams
    return X


def test_isotropic_random_directions_remove_far_less_than_a_high_variance_candidate():
    # The registered control's defect, reproduced: same coefficient, much smaller edit.
    rng = np.random.default_rng(0)
    X = _anisotropic(rng)
    cand = X[:200].mean(0) - X[200:].mean(0) + np.eye(64)[0] * 5
    iso = random_direction(64, rng, reference=cand)
    assert second_moment(X, cand) > 20 * second_moment(X, iso)


def test_variance_matched_directions_match_the_candidate_and_are_orthogonal_to_it():
    rng = np.random.default_rng(1)
    X = _anisotropic(rng)
    cand = np.eye(64)[1] + 0.3 * np.eye(64)[0]
    dirs, info = variance_matched_directions(X, cand, 5, rng, tol=0.10)
    target = second_moment(X, cand)
    for u in dirs:
        assert abs(second_moment(X, u) / target - 1) <= 0.10
        assert abs(u @ (cand / np.linalg.norm(cand))) < 1e-8
        assert np.linalg.norm(u) == pytest.approx(1.0)
    assert len(info["accepted_ratios"]) == 5


def test_too_few_matching_draws_is_refused_not_padded():
    # A tolerance no draw can meet must fail loudly, never return
    # unmatched directions as if they were controls.
    rng = np.random.default_rng(2)
    X = _anisotropic(rng)
    with pytest.raises(ValueError, match="within"):
        variance_matched_directions(X, np.eye(64)[3], 3, rng, tol=1e-9, max_draws=64)


def test_permuted_label_directions_keep_counts_and_differ_from_the_candidate():
    rng = np.random.default_rng(3)
    tail = {f"p{i}": rng.standard_normal((3, 16)) + 2 for i in range(20)}
    pre = {f"p{i}": rng.standard_normal((2, 16)) for i in range(20)}
    cand = np.vstack(list(tail.values())).mean(0) - np.vstack(list(pre.values())).mean(0)
    cand /= np.linalg.norm(cand)
    dirs, _ = permuted_label_directions(tail, pre, 4, rng)
    assert len(dirs) == 4
    assert all(abs(u @ cand) < 0.9 for u in dirs)


def test_permuted_controls_can_be_orthogonalised_and_size_filtered():
    rng = np.random.default_rng(8)
    tail = {f"p{i}": rng.standard_normal((3, 16)) * np.geomspace(4, 0.5, 16) + 1
            for i in range(30)}
    pre = {f"p{i}": rng.standard_normal((2, 16)) * np.geomspace(4, 0.5, 16) for i in range(30)}
    X = np.vstack(list(tail.values()) + list(pre.values()))
    cand = np.vstack(list(tail.values())).mean(0) - np.vstack(list(pre.values())).mean(0)
    dirs, info = permuted_label_directions(tail, pre, 3, rng, X=X, candidate=cand, tol=0.5)
    c = cand / np.linalg.norm(cand)
    for u in dirs:
        assert abs(u @ c) < 1e-8
    assert all(abs(r - 1) <= 0.5 for r in info["accepted_ratios"])


def test_position_direction_points_from_early_to_late():
    from nlaast.causal.controls import position_direction
    V = {"a": np.array([[0.0, 0], [0, 0], [2, 0], [2, 0]])}
    rel = {"a": np.array([0.0, 0.3, 0.6, 0.9])}
    assert position_direction(V, rel).tolist() == [1.0, 0.0]


def test_editor_reports_the_norm_it_removed():
    torch = pytest.importorskip("torch")
    d = np.zeros(4, np.float32)
    d[0] = 1.0
    edit = projection_editor(d, 1.0, (0, 2))
    h = torch.tensor([[[3.0, 1, 0, 0], [-4.0, 0, 1, 0], [5.0, 0, 0, 1]]])
    out = edit(h)
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 0 and out[0, 2, 0] == 5
    assert edit.n_positions_edited == 2
    assert edit.removed_norm_sum == pytest.approx(7.0)
    assert mean_removed_norm(edit.removed_norm_sum, edit.n_positions_edited) == pytest.approx(3.5)
    edit.reset()
    assert edit.removed_norm_sum == 0.0


def test_a_capped_continuation_is_read_from_its_forced_answer_not_its_text():
    assert read_outcome(True, "18", "99") == ("18", "finished")
    assert read_outcome(False, "5", "18") == ("18", "forced_after_cap")


def test_degeneration_flags_repetition_and_non_ascii():
    loop = "so we add the two numbers to get the total again. " * 3
    assert degenerate(loop)
    assert not degenerate("First add 3 and 4 to get 7. Then multiply by 2 to get 14.")
    assert degenerate("答案是" * 10)


def _pilot():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_rq3_pilot
    return run_rq3_pilot


def _row(pid, arm, dose, changed, norm=10.0, finished=True, degenerate=False, n_tokens=80):
    return {"problem_id": pid, "arm": arm, "dose": dose, "answer_changed": changed,
            "mean_removed_norm": norm, "finished": finished, "degenerate": degenerate,
            "n_tokens": n_tokens}


def test_stage0_stops_when_the_candidate_changes_too_few_answers():
    rp = _pilot()
    rows = [_row(f"p{i}", "ablate_dir", 0.0, False) for i in range(60)]
    rows += [_row(f"p{i}", "ablate_dir", 2.0, i < 3) for i in range(60)]   # 5%
    assert rp.stage0_decision(rows)["verdict"].startswith("no-go")
    rows = [r if r["dose"] == 0.0 else {**r, "answer_changed": int(r["problem_id"][1:]) < 12}
            for r in rows]                                                 # 20%
    assert rp.stage0_decision(rows)["verdict"].startswith("go to stage 1")
    bad = [{**rows[0], "answer_changed": True}] + rows[1:]
    assert rp.stage0_decision(bad)["verdict"].startswith("invalid")


def _stage1_rows(cand_rate, ctrl_rate, size=10.0, degenerate=False, n=60):
    rows = []
    for i in range(n):
        pid = f"p{i}"
        rows.append(_row(pid, "ablate_dir", 0.0, False))
        for d in (1.5, 2.0):
            rows.append(_row(pid, "ablate_dir", d, i < cand_rate * n, degenerate=degenerate))
            for arm in ("variance_matched", "permuted_label"):
                for k in range(5):
                    rows.append(_row(pid, arm, d, (i + k) % n < ctrl_rate * n, norm=size))
    return rows


def test_go_no_go_rules():
    rp = _pilot()
    go = rp.summarise(_stage1_rows(0.30, 0.03), reps=500)
    assert go["decision"]["verdict"] == "go"
    same = rp.summarise(_stage1_rows(0.10, 0.10), reps=500)
    assert same["decision"]["verdict"] == "no-go"
    degen = rp.summarise(_stage1_rows(0.30, 0.03, degenerate=True), reps=500)
    assert degen["decision"]["verdict"] == "no-go" and "degenerate" in degen["decision"]["reasons"][0]
    unmatched = rp.summarise(_stage1_rows(0.30, 0.03, size=20.0), reps=500)
    assert unmatched["decision"]["verdict"] == "invalid manipulation"


def test_cost_estimate_counts_the_arms_given():
    rp = _pilot()
    stage0 = [("ablate_dir", 0.0, None, "t"), ("ablate_dir", 2.0, None, "t")]
    c = rp.cost_estimate(60, stage0)
    assert c["generations"] == 120
    assert c["tokens_low"] == pytest.approx(60 * (64.6 + 183.6))
    assert c["tokens_high"] == 120 * rp.MAX_NEW_TOKENS


def test_within_tolerance():
    assert within_tolerance([10.5, 12.0, None], 10.0) == [True, False, False]
    assert within_tolerance([1.0], None) == [False]
