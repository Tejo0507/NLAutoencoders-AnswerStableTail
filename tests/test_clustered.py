"""The corrected F10/RQ2 contrasts (D72) do not mistake copies for evidence."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nlaast.analysis.clustered import mixed_contrast, mixed_mean, paired_problem_contrast
from nlaast.analysis.posthoc import cluster_bootstrap_mean


def _records(rng, n_problems=30, per=2, effect=0.0):
    out = []
    for p in range(n_problems):
        base = rng.normal(0, 1)                      # problem-level variation dominates
        for kind, shift in (("tail", effect), ("ctrl", 0.0)):
            for s in range(per):
                out.append({"problem_id": f"p{p}", "kind": kind, "window": f"p{p}-{kind}",
                            "v": base + shift + rng.normal(0, 0.1)})
    return out


def _duplicate(records, k):
    return [dict(r) for r in records for _ in range(k)]


def test_problem_contrast_is_invariant_to_duplicating_observations():
    rng = np.random.default_rng(0)
    recs = _records(rng, effect=0.05)
    a = paired_problem_contrast(recs, "kind", "v", "tail", "ctrl", 500, np.random.default_rng(1))
    b = paired_problem_contrast(_duplicate(recs, 10), "kind", "v", "tail", "ctrl", 500,
                                np.random.default_rng(1))
    assert a.keys() == b.keys()
    for k in a:                      # equal up to summation order (~1e-17)
        assert np.allclose(a[k], b[k], rtol=1e-12, atol=1e-15), k
    assert a["n_problems"] == b["n_problems"] == 30


def test_the_registered_style_pooled_test_is_not():
    # The defect in one assertion: tenfold copies of the same evidence make a
    # pooled Mann-Whitney far more "significant".
    rng = np.random.default_rng(2)
    recs = _records(rng, effect=0.3)
    def pooled_p(rs):
        t = [r["v"] for r in rs if r["kind"] == "tail"]
        c = [r["v"] for r in rs if r["kind"] == "ctrl"]
        return stats.mannwhitneyu(t, c).pvalue
    assert pooled_p(_duplicate(recs, 10)) < pooled_p(recs) / 10


def test_cluster_bootstrap_interval_is_unchanged_in_width_by_duplication():
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 1, 50)
    clus = np.repeat(np.arange(10), 5)
    a = cluster_bootstrap_mean(vals, clus, 2000, np.random.default_rng(4))
    b = cluster_bootstrap_mean(np.repeat(vals, 4), np.repeat(clus, 4), 2000,
                               np.random.default_rng(4))
    assert a["high"] - a["low"] == pytest.approx(b["high"] - b["low"], rel=0.05)


def test_paired_contrast_detects_a_within_problem_effect_that_pooling_hides():
    # Large between-problem spread, small consistent within-problem shift.
    rng = np.random.default_rng(5)
    recs = _records(rng, n_problems=40, per=2, effect=0.1)
    r = paired_problem_contrast(recs, "kind", "v", "tail", "ctrl", 1000, np.random.default_rng(6))
    assert r["wilcoxon_p"] < 0.001 and r["ci95"][0] > 0
    assert r["rank_biserial_paired"] > 0.6
    pooled = stats.mannwhitneyu([x["v"] for x in recs if x["kind"] == "tail"],
                                [x["v"] for x in recs if x["kind"] == "ctrl"]).pvalue
    assert pooled > 0.05             # the between-problem spread swamps it unpaired


def test_mixed_models_recover_the_within_problem_effect():
    rng = np.random.default_rng(7)
    recs = _records(rng, n_problems=40, per=2, effect=0.1)
    m = mixed_contrast(recs, "kind", "v", "tail", "ctrl")
    assert m["ci95"][0] < 0.1 < m["ci95"][1] and m["n_problems"] == 40
    mm = mixed_mean([r["v"] for r in recs], [r["problem_id"] for r in recs],
                    [r["window"] for r in recs])
    assert mm["converged"]
