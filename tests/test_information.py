"""The information audit's estimation (D81), on synthetic data with known structure.

These fixtures test the implementation, not the study: each one is built so
that the right answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.analysis.information import (
    Design,
    MentionUnit,
    cluster_weights,
    compare,
    grouped_folds,
    icc1,
    mean_difference,
    mention_contrast,
    normalise_number,
    numbers_in,
    out_of_fold,
    plain_number,
    reading,
    weighted_auc,
    weighted_log_loss,
    within_problem_concordance,
)


def test_weighted_auc_matches_sklearn_and_the_replicated_resample():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.4
    s = np.round(rng.normal(size=200) + y, 1)          # rounded: many ties
    assert weighted_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 4, size=200).astype(float)
    rep = np.repeat(np.arange(200), w.astype(int))
    assert weighted_auc(y, s, w) == pytest.approx(roc_auc_score(y[rep], s[rep]))


def test_weighted_auc_is_nan_with_one_class_and_log_loss_weights():
    assert np.isnan(weighted_auc(np.ones(5, bool), np.arange(5.0)))
    y = np.array([1, 0, 1.0])
    p = np.array([0.9, 0.2, 0.6])
    w = np.array([2.0, 1, 0])
    rep = np.array([0, 0, 1])
    assert weighted_log_loss(y, p, w) == pytest.approx(weighted_log_loss(y[rep], p[rep]))


def test_folds_never_split_a_problem_and_cover_every_unit():
    rng = np.random.default_rng(1)
    groups = np.repeat([f"p{i}" for i in range(60)], 5)
    y = rng.random(300) < 0.5
    folds = grouped_folds(y, groups, 10, seed=3)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(300))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
    assert [list(te) for _, te in folds] == [list(te) for _, te in grouped_folds(y, groups, 10, 3)]


def _problems(n_prob=80, per=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i}" for i in range(n_prob)], per)
    pos = np.tile(np.arange(per, dtype=float), n_prob) + rng.normal(0, 0.3, n_prob * per)
    signal = rng.normal(size=n_prob * per)
    noise = rng.normal(size=n_prob * per)
    logit = 1.2 * (pos - 2) + 1.5 * signal
    y = rng.random(n_prob * per) < 1 / (1 + np.exp(-logit))
    return groups, pos, signal, noise, y


def test_an_informative_feature_is_detected_and_a_noise_feature_is_not():
    groups, pos, signal, noise, y = _problems()
    folds = grouped_folds(y, groups, 10, seed=0)
    base = out_of_fold(Design(pos[:, None]), y, folds)
    good = out_of_fold(Design(np.c_[pos, signal]), y, folds)
    junk = out_of_fold(Design(np.c_[pos, noise]), y, folds)
    W = cluster_weights(groups, 400, np.random.default_rng(0))
    g = compare(y, good, base, W)
    j = compare(y, junk, base, W)
    assert g["delta_auc"] > 0.03 and g["ci95"][0] > 0
    assert j["ci95"][0] < 0 < j["ci95"][1]
    assert abs(j["delta_auc"]) < 0.02


def test_identical_predictions_compare_to_exactly_zero():
    groups, pos, _, _, y = _problems(40, 4)
    p = 1 / (1 + np.exp(-pos))
    W = cluster_weights(groups, 50, np.random.default_rng(0))
    c = compare(y, p, p, W)
    assert c["delta_auc"] == 0 and c["ci95"] == [0.0, 0.0] and c["p_bootstrap"] == 1.0


def test_a_text_token_that_carries_the_label_is_picked_up_out_of_fold():
    rng = np.random.default_rng(2)
    groups = np.repeat([f"p{i}" for i in range(100)], 4)
    y = rng.random(400) < 0.5
    filler = ["alpha beta", "gamma delta", "beta gamma", "delta alpha"]
    texts = [f"{filler[i % 4]} {'concluded' if (yy and rng.random() < 0.7) else 'working'}"
             for i, yy in enumerate(y)]
    folds = grouped_folds(y, groups, 10, seed=0)
    zero = np.zeros((400, 0))
    p_text = out_of_fold(Design(zero, (texts,)), y, folds)
    p_none = out_of_fold(Design(np.ones((400, 1))), y, folds)
    assert weighted_auc(y, p_text) > 0.75
    assert weighted_auc(y, p_none) == pytest.approx(0.5, abs=0.1)


def test_within_problem_concordance_ignores_between_problem_order():
    y = np.array([0, 1, 0, 1], bool)
    s = np.array([10.0, 11.0, 0.0, 1.0])          # problem a high, b low
    g = np.array(["a", "a", "b", "b"])
    assert within_problem_concordance(y, s, g) == 1.0
    # pooled: of the pairs (11,10) (11,0) (1,10) (1,0), only (1,10) is misordered
    assert weighted_auc(y, s) == 0.75


@pytest.mark.parametrize("tok,norm", [("1,200", "1200"), ("3.50", "3.5"), ("4.0", "4"),
                                      ("-2", "-2"), ("007", "7"), ("0.25", "0.25"),
                                      ("-0", "0"), ("100.00", "100"), ("abc", None)])
def test_number_normalisation(tok, norm):
    assert normalise_number(tok) == norm


def test_numbers_in_text_and_plain_answers():
    assert numbers_in("costs $1,200, then 3.50 and 18.") == {"1200", "3.5", "18"}
    assert plain_number("$18") == "18" and plain_number("1,000") == "1000"
    assert plain_number("\\frac{4}{13}") is None and plain_number(None) is None


def test_mention_contrast_against_other_problems_of_the_same_dataset():
    answers = {"a": ("gsm8k", "18"), "b": ("gsm8k", "7"), "c": ("gsm8k", "40"),
               "d": ("math", "18")}
    u = MentionUnit("a", "gsm8k", "18", context_numbers={"7"},
                    explanation_numbers={"18", "40"})
    own, other = mention_contrast([u], answers)
    # "b"'s 7 is in a's context, so only "c" (40) is a fair comparison; "d" is MATH
    assert own.tolist() == [1.0] and other.tolist() == [1.0]
    u2 = MentionUnit("a", "gsm8k", "18", set(), {"5"})
    own2, other2 = mention_contrast([u2], answers)
    assert own2.tolist() == [0.0] and other2.tolist() == [0.0]


def test_mean_difference_and_reading_language():
    W = cluster_weights(np.array(["a", "b", "c", "d"]), 200, np.random.default_rng(0))
    m = mean_difference(np.array([0.0, 0.0, 0.0, 0.0]), W)
    assert m["mean"] == 0 and m["ci95"] == [0.0, 0.0]
    assert reading([0.01, 0.05], True, 0.03, 0.02) == "increment detected"
    assert reading([-0.01, 0.015], False, 0.002, 0.02) == "no practically relevant increment"
    assert reading([-0.01, 0.05], False, 0.02, 0.02) == "unresolved"
    assert reading(None, None, float("nan"), 0.02) == "not evaluable"


def test_nested_probe_finds_a_planted_direction_and_nothing_in_noise():
    from nlaast.analysis.information import nested_probe
    rng = np.random.default_rng(4)
    n, d = 300, 40
    groups = np.repeat([f"p{i}" for i in range(75)], 4)
    X = rng.normal(size=(n, d))
    y_sig = (X[:, 3] + 0.5 * rng.normal(size=n)) > 0
    y_null = rng.random(n) < 0.5
    for y, lo, hi in ((y_sig, 0.8, 1.0), (y_null, 0.35, 0.65)):
        folds = grouped_folds(y, groups, 5, seed=0)
        extras, Cs = nested_probe(X, y, groups, folds, Cs=(1e-2, 1.0), inner_splits=3)
        p = out_of_fold(Design(np.zeros((n, 0))), y, folds, extras)
        assert lo < weighted_auc(y, p) < hi
        assert len(Cs) == 5 and set(Cs) <= {1e-2, 1.0}
        # train-row scores are inner out-of-fold: one per training row
        for (tr, te), (a, b) in zip(folds, extras):
            assert a.shape == (len(tr), 1) and b.shape == (len(te), 1)


def test_icc_is_high_for_group_constant_values_and_low_for_noise():
    rng = np.random.default_rng(0)
    g = np.repeat([f"p{i}" for i in range(50)], 4)
    level = np.repeat(rng.normal(size=50), 4)
    assert icc1(level + rng.normal(0, 0.05, 200), g) > 0.95
    assert abs(icc1(rng.normal(size=200), g)) < 0.15
