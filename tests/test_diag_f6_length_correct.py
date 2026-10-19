"""Unit tests for the F6 length-correct diagnostic (scripts/diag_f6_length_correct.py).

Synthetic data and a fake cross-validation function only: no research probe is
fitted, no research data is read, and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import floor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diag_f6_length_correct as D  # noqa: E402

from nlaast.baselines.probe import permute_labels  # noqa: E402


# --- synthetic F6-shaped inputs -----------------------------------------------

LENGTHS = {"p00": 4, "p01": 4, "p02": 5, "p03": 5, "p04": 5, "p05": 6, "p06": 6,
           "p07": 7, "p08": 7, "p09": 8, "p10": 9, "p11": 11}       # 8, 9, 11 unique


def synth(lengths=LENGTHS, seed=0) -> D.F6Inputs:
    rng = np.random.default_rng(seed)
    X, y, groups, j, rel, P = [], [], [], [], [], []
    for g, n in lengths.items():
        t = int(rng.integers(1, n))                       # monotone: wrong, then right
        for k in range(n):
            X.append(rng.normal(size=3))
            y.append(int(k >= t))
            groups.append(g)
            j.append(k)
            rel.append(k / (n - 1))
            P.append([k / (n - 1), float(k == n - 1), float(k), np.log1p(k + 1)])
    return D.F6Inputs(X=np.array(X), y=np.array(y), groups=groups, row_index=np.array(j),
                      rel=np.array(rel), P=np.array(P))


def fake_cv(X, y, groups, cfg, shuffle_labels=False, shuffle_scope="across_groups"):
    """Stands in for cross_validate: deterministic scores, no model."""
    y = np.asarray(y, dtype=int)
    notes = {}
    if shuffle_labels:
        y2, _ = permute_labels(y, groups, shuffle_scope, cfg.seed + 1)
        notes = {"shuffle": {"n_labels_changed": int(np.sum(y2 != y))}}
        y = y2
    s = np.asarray(X, dtype=float)[:, 0] + 0.5 * y
    return SimpleNamespace(auc=D.auc(y, s), oof=list(s), notes=notes)


CFG = SimpleNamespace(seed=11)


def synth_registered(inp: D.F6Inputs) -> D.Registered:
    shuf = fake_cv(inp.X, inp.y, inp.groups, CFG, shuffle_labels=True)
    real = fake_cv(inp.X, inp.y, inp.groups, CFG)
    eligible, _ = D.exact_length_eligible(inp.lengths)
    return D.Registered(n_labels_changed=shuf.notes["shuffle"]["n_labels_changed"],
                        shuffled_auc=shuf.auc, real_auc=real.auc, n_rows=len(inp.y),
                        n_problems=len(inp.order), n_eligible_problems=len(eligible),
                        n_eligible_rows=sum(inp.lengths[g] for g in eligible))


@pytest.fixture
def paths(tmp_path, monkeypatch) -> D.Paths:
    runs = tmp_path / "runs"
    src = runs / "main"
    (src / "robustness").mkdir(parents=True)
    rob = src / "robustness" / "robustness.json"
    rob.write_text('{"F6_probe_leakage": {"leak_suspected": false}}', encoding="utf-8")
    proto = tmp_path / "protocol.md"
    proto.write_text("approved protocol\n", encoding="utf-8")
    script = tmp_path / "script.py"
    script.write_text("print('x')\r\n", encoding="utf-8")
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text(f"D59 code SHA-256 {D.sha256_text_lf(script)}\n", encoding="utf-8")
    monkeypatch.setattr(D, "ROBUSTNESS_SHA256", D.sha256_file(rob))
    monkeypatch.setattr(D, "PROTOCOL_SHA256", D.sha256_file(proto))
    return D.Paths(root=tmp_path, source_dir=src, out_dir=runs / D.RUN_ID, robustness=rob,
                   protocol=proto, decisions=decisions, script=script)


# --- the length-correct mapping -------------------------------------------------

def test_phi_is_the_identity_for_equal_lengths():
    for n in range(2, 15):
        assert np.array_equal(D.phi(n, n), np.arange(n))


def test_phi_longer_and_shorter_receivers():
    assert D.phi(3, 5).tolist() == [0, 1, 1, 2, 2]
    assert D.phi(5, 3).tolist() == [0, 2, 4]
    assert D.phi(9, 4).tolist() == [0, 3, 5, 8]


def test_phi_matches_the_protocol_formula_exactly_with_bounds_and_endpoints():
    for n_d in range(1, 25):
        for n_r in range(2, 25):
            got = D.phi(n_d, n_r)
            want = [floor(Fraction(j * (n_d - 1), n_r - 1) + Fraction(1, 2)) for j in range(n_r)]
            assert got.tolist() == want
            assert got[0] == 0 and got[-1] == n_d - 1
            assert got.min() >= 0 and got.max() <= n_d - 1
            assert np.all(np.diff(got) >= 0)                    # order-preserving
            if n_r >= n_d:                                      # every donor row used
                assert set(got.tolist()) == set(range(n_d))


def test_phi_rejects_undefined_inputs():
    with pytest.raises(ValueError):
        D.phi(5, 1)
    with pytest.raises(ValueError):
        D.phi(0, 4)


def test_s1_preserves_order_endpoints_and_monotonicity():
    rng = np.random.default_rng(3)
    for n_d in range(2, 20):
        t = int(rng.integers(0, n_d + 1))
        donor = (np.arange(n_d) >= t).astype(int)
        for n_r in range(2, 20):
            out = D.fit_to_length("S1", donor, n_r)
            assert len(out) == n_r
            assert out[0] == donor[0] and out[-1] == donor[-1]
            assert np.all(np.diff(out) >= 0)


def test_s1_downsampling_can_drop_a_short_run_and_change_transitions():
    donor = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1])
    out = D.fit_to_length("S1", donor, 4)
    assert out.tolist() == [0, 0, 0, 1]
    assert np.count_nonzero(np.diff(donor)) == 3 and np.count_nonzero(np.diff(out)) == 1


def test_s0_is_np_resize_and_all_rules_agree_at_equal_length():
    donor = np.array([0, 0, 1, 1, 1])
    assert D.fit_to_length("S0", donor, 7).tolist() == [0, 0, 1, 1, 1, 0, 0]   # tiled
    assert D.fit_to_length("S0", donor, 3).tolist() == [0, 0, 1]               # truncated
    assert D.fit_to_length("S1", donor, 3).tolist() == [0, 1, 1]
    for rule in ("S0", "S1", "S2"):
        assert D.fit_to_length(rule, donor, 5).tolist() == donor.tolist()
    with pytest.raises(ValueError):
        D.fit_to_length("S2", donor, 4)


def test_label_statistics_count_duplicated_and_dropped_labels():
    inp = synth({"a": 4, "b": 6, "c": 6})
    donors = {"a": "b", "b": "c", "c": "a"}
    s0 = D.label_statistics(inp, D.shuffled_labels(inp, donors, "S0"), donors, "S0")
    # a<-b: 2 dropped; b<-c: none; c<-a (np.resize tiles 4 to 6): 2 duplicated
    assert (s0["n_labels_dropped"], s0["n_labels_duplicated"]) == (2, 2)
    s1 = D.label_statistics(inp, D.shuffled_labels(inp, donors, "S1"), donors, "S1")
    assert (s1["n_labels_dropped"], s1["n_labels_duplicated"]) == (2, 2)
    assert s1["fixed_points"] == 0
    assert len(s1["per_problem_prevalence"]) == 3


# --- donor assignments --------------------------------------------------------

def test_rejection_derangement_has_no_fixed_points_and_is_reproducible():
    for b in range(1, 30):
        p = D.rejection_derangement(13, D.stream(D.STREAM_FULL, b))
        assert not np.any(p == np.arange(13))
        assert np.array_equal(p, D.rejection_derangement(13, D.stream(D.STREAM_FULL, b)))
    assert not np.array_equal(D.rejection_derangement(13, D.stream(0, 1)),
                              D.rejection_derangement(13, D.stream(0, 2)))
    with pytest.raises(ValueError):
        D.rejection_derangement(1, D.stream(0, 1))


def test_streams_are_the_protocol_seed_sequences():
    ref = np.random.default_rng(np.random.SeedSequence([11, 0xF6, 2, 7]))
    assert np.array_equal(D.stream(D.STREAM_STRATA, 7).permutation(20), ref.permutation(20))


def test_draw0_reproduces_permute_labels_exactly():
    inp = synth()
    donors = D.full_donors(inp.order, 0)
    want, _ = permute_labels(inp.y, inp.groups, "across_groups", 12)
    assert np.array_equal(D.shuffled_labels(inp, donors, "S0"), want)


def test_exact_length_eligibility_excludes_unique_lengths():
    eligible, excluded = D.exact_length_eligible(LENGTHS)
    assert excluded == ["p09", "p10", "p11"]
    assert eligible == sorted(set(LENGTHS) - set(excluded))


def test_stratum_donors_stay_within_length_and_swap_pairs():
    sub = synth().subset(D.exact_length_eligible(LENGTHS)[0])
    for b in range(5):
        don = D.stratum_donors(sub.lengths, b)
        assert all(sub.lengths[d] == sub.lengths[r] for r, d in don.items())
        assert all(d != r for r, d in don.items())
        assert don["p00"] == "p01" and don["p01"] == "p00"           # the size-2 stratum
        assert don == D.stratum_donors(sub.lengths, b)


def test_paired_arms_share_one_donor_assignment():
    full = synth()
    sub = full.subset(D.exact_length_eligible(full.lengths)[0])
    dl = D.draw_labels(full, sub, 3)
    assert np.array_equal(dl.y["S0"], D.shuffled_labels(full, dl.donors_full, "S0"))
    assert np.array_equal(dl.y["S1"], D.shuffled_labels(full, dl.donors_full, "S1"))
    assert np.array_equal(dl.y["sub_S0"], D.shuffled_labels(sub, dl.donors_subset, "S0"))
    assert np.array_equal(dl.y["sub_S1"], D.shuffled_labels(sub, dl.donors_subset, "S1"))
    assert dl.donors_full == D.full_donors(full.order, 3)
    assert D.draw_labels(full, sub, 4).donors_full != dl.donors_full


# --- the positional oracle ------------------------------------------------------

def test_oracle_matches_the_definition_and_excludes_receiver_and_donor():
    inp = synth()
    donors = D.full_donors(inp.order, 2)
    for rule in ("S0", "S1"):
        g = D.oracle_scores(inp, donors, rule)
        for r, idx in inp.rows.items():
            others = [d for d in inp.order if d not in (r, donors[r])]
            want = np.mean([D.fit_to_length(rule, inp.labels[d], len(idx)) for d in others],
                           axis=0)
            assert np.allclose(g[idx], want)


def test_oracle_refuses_a_receiver_that_is_its_own_donor():
    inp = synth()
    donors = D.full_donors(inp.order, 2)
    assert all(d != r for r, d in donors.items())
    D.oracle_scores(inp, donors, "S1")                       # a derangement is accepted
    a = inp.order[0]
    fixed = {r: (r if r == a else d) for r, d in donors.items()}
    with pytest.raises(ValueError, match="its own donor"):
        D.oracle_scores(inp, fixed, "S1")


def test_data_checks_pass_on_a_derangement_at_draw_0():
    inp = synth()
    checks = D.data_checks(inp, {"chunk_indices_ascending_within_problem": True},
                           synth_registered(inp))
    assert checks["draw0_fixed_points"] == 0


def test_data_checks_refuse_a_fixed_point_in_draw_0(monkeypatch):
    inp = synth()
    reg = synth_registered(inp)
    real = D.f6_permutation(len(inp.order))
    k = int(np.flatnonzero(real == 0)[0])                     # the receiver that had donor 0
    with_fixed_point = real.copy()
    with_fixed_point[0], with_fixed_point[k] = 0, real[0]     # receiver 0 keeps its own labels
    assert sorted(with_fixed_point.tolist()) == list(range(len(real)))
    assert int(np.sum(with_fixed_point == np.arange(len(real)))) >= 1
    monkeypatch.setattr(D, "f6_permutation", lambda n, seed=12: with_fixed_point)
    with pytest.raises(D.PreflightFailure, match="draw0_fixed_points"):
        D.data_checks(inp, {"chunk_indices_ascending_within_problem": True}, reg)


def test_oracle_ignores_the_receivers_and_donors_own_labels():
    inp = synth()
    donors = D.full_donors(inp.order, 1)
    r = "p05"
    base = D.oracle_scores(inp, donors, "S1")[inp.rows[r]]
    for who in (r, donors[r]):
        y2 = inp.y.copy()
        y2[inp.rows[who]] = 1 - y2[inp.rows[who]]
        alt = D.F6Inputs(X=inp.X, y=y2, groups=inp.groups, row_index=inp.row_index,
                         rel=inp.rel, P=inp.P)
        assert np.allclose(D.oracle_scores(alt, donors, "S1")[inp.rows[r]], base)
    third = next(d for d in inp.order if d not in (r, donors[r]))
    y3 = inp.y.copy()
    y3[inp.rows[third]] = 1 - y3[inp.rows[third]]
    alt = D.F6Inputs(X=inp.X, y=y3, groups=inp.groups, row_index=inp.row_index,
                     rel=inp.rel, P=inp.P)
    assert not np.allclose(D.oracle_scores(alt, donors, "S1")[inp.rows[r]], base)


# --- AUC, bootstrap, summaries -------------------------------------------------

def test_auc_valid_and_degenerate():
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.4, 0.4, 0.8])
    assert D.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(D.auc(np.ones(4), np.arange(4)))
    assert np.isnan(D.auc(np.array([]), np.array([])))


def test_weighted_auc_equals_roc_auc_with_unit_and_duplicated_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.normal(size=60), 1)                  # many ties
    prob = np.repeat(np.arange(12), 5)
    assert D.weighted_auc(y, s, prob, np.ones((1, 12)))[0] == pytest.approx(roc_auc_score(y, s))
    counts = rng.integers(0, 3, size=(1, 12))
    keep = np.repeat(np.arange(60), counts[0][prob])
    assert D.weighted_auc(y, s, prob, counts)[0] == pytest.approx(roc_auc_score(y[keep], s[keep]))
    only_pos = np.zeros((1, 12)); only_pos[0, 0] = 1
    y_one = np.ones(60, dtype=int)
    assert np.isnan(D.weighted_auc(y_one, s, prob, only_pos)[0])


def test_bootstrap_counts_are_reproducible_resamples():
    a, b = D.bootstrap_counts(9, 40, 2024), D.bootstrap_counts(9, 40, 2024)
    assert np.array_equal(a, b) and a.shape == (40, 9) and np.all(a.sum(axis=1) == 9)
    assert not np.array_equal(a, D.bootstrap_counts(9, 40, 2025))


def test_paired_bootstrap_uses_one_resample_for_both_arms():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 40), rng.normal(size=40)
    prob = np.repeat(np.arange(8), 5)
    counts = D.bootstrap_counts(8, 30, 7)
    a, b = D.weighted_auc(y, s, prob, counts), D.weighted_auc(y, s, prob, counts)
    assert np.array_equal(a - b, np.zeros(30))


def test_monte_carlo_se_is_sd_over_root_b():
    v = [0.1, 0.3, 0.2, 0.6]
    assert D.monte_carlo_se(v) == pytest.approx(np.std(v, ddof=1) / 2)
    assert np.isnan(D.monte_carlo_se([0.5]))


def _draws(pos, neg, B):
    return D.draw_summary([0.01] * pos + [-0.01] * neg + [0.0] * (B - pos - neg))


def test_wording_follows_the_predeclared_table():
    boot_pos = {"mean": 0.01, "n_undefined": 0, "p2_5": 0.001, "p97_5": 0.02}
    boot_neg = {"mean": -0.01, "n_undefined": 0, "p2_5": -0.02, "p97_5": -0.001}
    boot_zero = {"mean": 0.0, "n_undefined": 0, "p2_5": -0.01, "p97_5": 0.01}
    assert D.wording(_draws(100, 0, 100), boot_pos)["case"] == "both"
    assert D.wording(_draws(100, 0, 100), boot_neg)["case"] == "conflicting"
    assert D.wording(_draws(98, 2, 100), boot_zero)["case"] == "draws_only"
    assert D.wording(_draws(97, 3, 100), boot_pos)["case"] == "bootstrap_only"
    assert D.wording(_draws(50, 50, 100), boot_zero)["case"] == "neither"
    assert D.wording(_draws(49, 1, 50), boot_zero)["case"] == "draws_only"   # 0.98 >= 0.975
    assert D.wording(_draws(48, 2, 50), boot_zero)["case"] == "neither"      # 0.96
    assert "not a joint confidence statement" in D.WORDING["both"]


def test_summaries_flag_undefined_values_instead_of_dropping_them():
    s = D.draw_summary([0.1, float("nan"), 0.3])
    assert s["n_undefined"] == 1 and s["n_draws"] == 3
    w = D.wording(s, {"mean": 0.1, "n_undefined": 0, "p2_5": 0.01, "p97_5": 0.2})
    assert w["complete"] is False


# --- gate and budget --------------------------------------------------------

def test_gate_passes_on_exact_reproduction():
    inp = synth()
    reg = synth_registered(inp)
    out = D.run_gate(inp, CFG, fake_cv, reg)
    assert out["passed"] and out["n_labels_changed"] == reg.n_labels_changed


@pytest.mark.parametrize("change", [
    {"n_labels_changed": -1}, {"shuffled_auc": 2e-9}, {"real_auc": 2e-9}])
def test_gate_fails_on_any_mismatch(change):
    inp = synth()
    reg = synth_registered(inp)
    fields = {k: getattr(reg, k) for k in reg.__dataclass_fields__}
    for k, d in change.items():
        fields[k] = fields[k] + d
    with pytest.raises(D.GateFailure):
        D.run_gate(inp, CFG, fake_cv, D.Registered(**fields))


def test_gate_tolerance_is_strictly_below_1e_9():
    inp = synth()
    reg = synth_registered(inp)
    ok = D.Registered(**{**reg.__dict__, "shuffled_auc": reg.shuffled_auc + 5e-10})
    assert D.run_gate(inp, CFG, fake_cv, ok)["passed"]


def test_registered_constants_are_the_protocols():
    r = D.Registered()
    assert (r.n_labels_changed, r.shuffled_auc, r.real_auc) == (718, 0.6142873504, 0.9460944231)
    assert (r.n_eligible_problems, r.n_eligible_rows) == (246, 2577)
    assert D.GATE_TOLERANCE == 1e-9 and D.F6_SHUFFLE_SEED == 12


def test_draw_budget_fallback():
    assert D.choose_draws(30.0)["B"] == 100                 # 30*5*100*1.25 = 5.2 h
    assert D.choose_draws(40.0)["B"] == 50                  # 6.9 h at 100, 3.5 h at 50
    with pytest.raises(D.BudgetStop):
        D.choose_draws(150.0)                               # 13 h at 50


# --- preflight and registered-artifact checks --------------------------------------

def _clean_git(root, files):
    return {"head": "x", "files": {str(f): {"tracked": True, "unmodified": True} for f in files}}


def _ok_fingerprints():
    return dict(D.SOURCE_FINGERPRINTS)


def _no_torch():
    return False


CV = {"probe": {"kind": "logistic", "C": 1.0, "max_iter": 2000, "standardise": True,
                "n_folds": 5, "seed": 11}}


def test_preflight_fails_if_torch_is_loaded(paths):
    with pytest.raises(D.PreflightFailure, match="torch_not_imported"):
        D.preflight(paths, CV, _clean_git, _ok_fingerprints, lambda: True)


def test_preflight_passes_when_every_check_holds(paths):
    checks = D.preflight(paths, CV, _clean_git, _ok_fingerprints, _no_torch)
    assert all(c["ok"] for c in checks.values())


def test_preflight_refuses_an_existing_output_directory(paths):
    paths.out_dir.mkdir()
    with pytest.raises(D.PreflightFailure, match="output_dir_absent"):
        D.preflight(paths, CV, _clean_git, _ok_fingerprints, _no_torch)


def test_preflight_fails_on_a_changed_registered_artifact(paths):
    paths.robustness.write_text("{}", encoding="utf-8")
    with pytest.raises(D.PreflightFailure, match="registered_robustness_sha256"):
        D.preflight(paths, CV, _clean_git, _ok_fingerprints, _no_torch)


@pytest.mark.parametrize("breakage, name", [
    ("protocol", "approved_protocol_sha256"),
    ("decisions", "script_sha256_recorded_in_decisions"),
    ("git", "script_protocol_decisions_committed"),
    ("fingerprint", "registered_functions_unchanged"),
    ("probe", "probe_config_as_protocol"),
    ("max_iter", "probe_config_as_protocol"),
])
def test_preflight_fails_on_each_guard(paths, breakage, name):
    git, fp, cv = _clean_git, _ok_fingerprints, CV
    if breakage == "protocol":
        paths.protocol.write_text("edited\n", encoding="utf-8")
    elif breakage == "decisions":
        paths.decisions.write_text("no hash here\n", encoding="utf-8")
    elif breakage == "git":
        def git(root, files):
            return {"head": "x", "files": {str(f): {"tracked": True, "unmodified": False}
                                           for f in files}}
    elif breakage == "fingerprint":
        def fp():
            return {**D.SOURCE_FINGERPRINTS, "nlaast.baselines.probe.permute_labels": "0" * 64}
    elif breakage == "max_iter":
        cv = {"probe": {**CV["probe"], "max_iter": 1000}}
    else:
        cv = {"probe": {**CV["probe"], "seed": 12}}
    with pytest.raises(D.PreflightFailure, match=name):
        D.preflight(paths, cv, git, fp, _no_torch)


def test_registered_functions_match_their_recorded_fingerprints():
    assert D.source_fingerprints() == D.SOURCE_FINGERPRINTS


def test_script_hash_ignores_crlf(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\r\ny = 2\r\n")
    b.write_bytes(b"x = 1\ny = 2\n")
    assert D.sha256_text_lf(a) == D.sha256_text_lf(b)


def test_no_torch_import():
    import subprocess

    src = Path(D.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src
    code = ("import sys; sys.path.insert(0, r'%s'); import diag_f6_length_correct; "
            "print('torch' in sys.modules)" % SCRIPTS)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- orchestration on synthetic data (fake fits) -------------------------------

def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _run(paths, cv_fn=fake_cv, reg=None, budget=None, inp=None):
    inp = inp or synth()
    reg = reg or synth_registered(inp)
    return D.run_diagnostic(inp, {}, paths, CFG, {"note": "test"}, cv_fn, reg, analysis_seed=5,
                            draws_fn=budget or (lambda t: {"B": 3}), n_boot=40, max_draws=3)


def test_full_orchestration_completes_and_leaves_the_registered_file_alone(paths):
    before = paths.robustness.read_bytes()
    out = _run(paths)
    assert out == {"status": "complete", "B": 3, "n_draws_completed": 3}
    assert paths.robustness.read_bytes() == before
    status = json.loads((paths.out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    summary = json.loads((paths.out_dir / "summary.json").read_text(encoding="utf-8"))
    stats = summary["statistics"]
    assert set(stats) == set(D.STATISTICS)
    for name, s in stats.items():
        assert s["donor_draws"]["n_draws"] == 3
        assert s["problem_bootstrap"]["n_resamples"] == 40
        assert ("wording" in s) == (s["kind"] == "difference")
    assert not any("leak" in k.lower() for k in _keys(summary))     # no verdict field
    assert "no leakage verdict" in summary["reading"]
    caveats = summary["caveats"]
    assert "not an interval for the mean" in caveats["donor_draw_percentiles"]
    assert "not a guaranteed upper bound" in caveats["oracle"]
    assert "formal null distribution" in caveats["oracle"]
    assert "not proof of leakage" in caveats["probe_above_oracle"]
    assert "does not rule out leakage" in caveats["probe_at_or_below_oracle"]
    draws = (paths.out_dir / "draws.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(draws) == 3
    s0 = json.loads((paths.out_dir / "stage0_label_stats.json").read_text(encoding="utf-8"))
    assert s0["computed_before_any_fit"] and len(s0["draws"]) == 3


def test_orchestration_is_reproducible(paths, tmp_path):
    _run(paths)
    first = json.loads((paths.out_dir / "summary.json").read_text(encoding="utf-8"))
    second_paths = D.Paths(**{**paths.__dict__, "out_dir": tmp_path / "runs2" / D.RUN_ID})
    second_paths.out_dir.parent.mkdir()
    _run(second_paths)
    second = json.loads((second_paths.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert first["statistics"] == second["statistics"]


def test_orchestration_refuses_an_existing_output_directory(paths):
    paths.out_dir.mkdir()
    calls = []

    def counting_cv(*a, **k):
        calls.append(1)
        return fake_cv(*a, **k)

    with pytest.raises(D.PreflightFailure):
        _run(paths, cv_fn=counting_cv)
    assert calls == []                                       # no fit was made


def test_gate_failure_writes_nothing(paths):
    inp = synth()
    reg = synth_registered(inp)
    bad = D.Registered(**{**reg.__dict__, "n_labels_changed": reg.n_labels_changed + 1})
    before = paths.robustness.read_bytes()
    with pytest.raises(D.GateFailure):
        _run(paths, reg=bad, inp=inp)
    assert not paths.out_dir.exists()
    assert paths.robustness.read_bytes() == before


def test_budget_stop_writes_nothing(paths):
    def stop(t):
        raise D.BudgetStop("too slow", {})

    with pytest.raises(D.BudgetStop):
        _run(paths, budget=stop)
    assert not paths.out_dir.exists()


def test_a_failure_while_writing_early_outputs_is_marked_failed(paths, monkeypatch):
    real_write = D.write_json

    def failing_write(path, obj):
        if Path(path).name == "stage0_label_stats.json":
            raise OSError("synthetic disk failure")
        real_write(path, obj)

    monkeypatch.setattr(D, "write_json", failing_write)
    before = paths.robustness.read_bytes()
    with pytest.raises(D.RunFailure, match="outputs"):
        _run(paths)
    status = json.loads((paths.out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["failed_stage"] == "outputs"
    assert status["failed_draw"] is None and status["n_draws_completed"] == 0
    assert not (paths.out_dir / "summary.json").exists()
    assert not (paths.out_dir / "draws.jsonl").exists()
    assert paths.robustness.read_bytes() == before


def test_a_failed_draw_is_recorded_not_omitted(paths):
    calls = []

    def failing_cv(*a, **k):
        calls.append(1)
        if len(calls) == 2 + 7 + 3:            # gate (2), draw 0 (7), third fit of draw 1
            raise RuntimeError("synthetic fit failure")
        return fake_cv(*a, **k)

    before = paths.robustness.read_bytes()
    with pytest.raises(D.RunFailure):
        _run(paths, cv_fn=failing_cv)
    status = json.loads((paths.out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["failed_stage"] == "draws"
    assert status["failed_draw"] == 1 and status["n_draws_completed"] == 1
    assert status["robustness_unchanged"] is True
    assert not (paths.out_dir / "summary.json").exists()
    assert paths.robustness.read_bytes() == before


def test_a_changed_registered_artifact_fails_the_run(paths):
    calls = []

    def tampering_cv(*a, **k):
        calls.append(1)
        if len(calls) == 5:
            paths.robustness.write_text("{}", encoding="utf-8")
        return fake_cv(*a, **k)

    with pytest.raises(D.RunFailure, match="changed"):
        _run(paths, cv_fn=tampering_cv)
    status = json.loads((paths.out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["robustness_unchanged"] is False
    assert not (paths.out_dir / "summary.json").exists()


def test_an_undefined_auc_fails_the_draw(paths):
    calls = []

    def one_class_cv(X, y, groups, cfg, **k):
        calls.append(1)
        rep = fake_cv(X, y, groups, cfg, **k)
        if len(calls) == 3:                    # first fit of draw 0
            return SimpleNamespace(auc=float("nan"), oof=[float("nan")] * len(y), notes={})
        return rep

    with pytest.raises(D.RunFailure, match="undefined"):
        _run(paths, cv_fn=one_class_cv)
    assert not (paths.out_dir / "summary.json").exists()


def test_oracle_rows_follow_the_probes_scored_rows(paths):
    calls = []

    def partial_cv(X, y, groups, cfg, **k):
        calls.append(1)
        rep = fake_cv(X, y, groups, cfg, **k)
        oof = np.asarray(rep.oof, dtype=float)
        if len(calls) > 2 + 7:                 # gate and draw 0 as registered; then partial
            oof[:4] = np.nan                   # the probe leaves four rows unscored
            v = np.isfinite(oof)
            return SimpleNamespace(auc=D.auc(np.asarray(y)[v], oof[v]), oof=list(oof),
                                   notes=rep.notes)
        return rep

    _run(paths, cv_fn=partial_cv)
    rec = json.loads((paths.out_dir / "draws.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert rec["probe_S0"]["n_unscored"] == 4
    assert rec["oracle_S0"]["n_scored"] == rec["probe_S0"]["n_scored"]
