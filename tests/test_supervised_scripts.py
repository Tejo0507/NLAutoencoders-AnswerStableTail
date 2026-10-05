"""The modelling paths of the two supervised analysis scripts.

The real corpus is expensive to grow and, early on, too small to exercise every
branch - at N = 21 the training split happened to be single-class, so the
classification path could not run at all. These tests drive the same functions
on synthetic frames with a *known* relationship, so a wiring fault is caught
here rather than discovered on a sample that took hours to generate.

The data is obviously synthetic and is never written anywhere. What is being
checked is plumbing and sign conventions, not a scientific claim.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nlaast.supervised import BLOCKS, INFERENCE_FEATURES

# ``scripts/`` is not a package: each stage script imports ``_stage`` as a
# sibling top-level module, which only resolves with this directory on the path.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def regression_script():
    return importlib.import_module("predict_tail_fraction")


@pytest.fixture(scope="module")
def classification_script():
    return importlib.import_module("predict_correctness")


# --- Synthetic frames with a signal that is actually there ---


def synthetic(n=200, seed=0, signal=True):
    """A frame where ``n_tokens`` and ``convergence_position`` drive both targets."""
    rng = np.random.default_rng(seed)
    columns = BLOCKS["C_convergence"]
    data = {c: rng.normal(size=n) for c in columns
            if c not in ("dataset", "subject", "truncated")}
    data["dataset"] = rng.choice(["gsm8k", "math"], size=n)
    data["subject"] = rng.choice(["algebra", "gsm8k"], size=n)
    data["truncated"] = rng.integers(0, 2, size=n)
    frame = pd.DataFrame(data)

    driver = 0.6 * frame["convergence_position"] - 0.4 * frame["n_tokens"]
    noise = rng.normal(scale=0.3, size=n)
    latent = (driver + noise) if signal else rng.normal(size=n)

    frame["tail_fraction"] = np.clip(0.35 + 0.12 * latent, 0.0, 1.0)
    frame["final_correct"] = (latent > np.quantile(latent, 0.25)).astype(int)

    frame["id"] = [f"synthetic-{i:04d}" for i in range(n)]
    frame["split"] = np.where(np.arange(n) % 5 < 2, "train", "eval")
    frame["ast_status"] = "ok"
    frame["math_level"] = np.where(frame["dataset"] == "math",
                                   rng.integers(1, 6, size=n), np.nan)
    return frame


# --- Regression paths ---


def test_cohorts_partition_as_documented(regression_script):
    frame = synthetic(120, seed=1)
    frame.loc[:19, "ast_status"] = "no_stable_point"
    frame.loc[20:39, "truncated"] = 1
    frame.loc[40:, "truncated"] = 0

    cohorts = regression_script.define_cohorts(frame)

    # primary excludes both the convention rows and the truncated ones
    assert not cohorts["primary"]["ast_status"].eq("no_stable_point").any()
    assert cohorts["primary"]["truncated"].eq(0).all()
    # each sensitivity cohort puts exactly one exclusion back
    assert cohorts["with_no_stable_point"]["ast_status"].eq("no_stable_point").any()
    assert cohorts["with_truncated"]["truncated"].eq(1).any()
    assert len(cohorts["primary"]) <= len(cohorts["with_no_stable_point"])
    assert len(cohorts["primary"]) <= len(cohorts["with_truncated"])


def test_ridge_beats_the_mean_on_a_frame_with_real_signal(regression_script):
    from sklearn.metrics import mean_absolute_error

    frame = synthetic(200, seed=2, signal=True)
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "eval"]
    columns = BLOCKS["C_convergence"]

    model = regression_script.ridge_pipeline(columns).set_params(model__alpha=1.0)
    model.fit(train[columns], train["tail_fraction"])
    pred = model.predict(test[columns])

    ridge_mae = mean_absolute_error(test["tail_fraction"], pred)
    mean_mae = mean_absolute_error(test["tail_fraction"],
                                   np.full(len(test), train["tail_fraction"].mean()))
    assert ridge_mae < mean_mae


def test_ridge_does_not_beat_the_mean_on_a_frame_without_signal(regression_script):
    """The complement of the test above - a model that 'wins' either way is broken."""
    from sklearn.metrics import r2_score

    frame = synthetic(200, seed=3, signal=False)
    frame["tail_fraction"] = np.random.default_rng(9).normal(0.35, 0.1, len(frame))
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "eval"]
    columns = BLOCKS["C_convergence"]

    model = regression_script.ridge_pipeline(columns).set_params(model__alpha=1.0)
    model.fit(train[columns], train["tail_fraction"])

    assert r2_score(test["tail_fraction"], model.predict(test[columns])) < 0.15


def test_ols_inference_returns_a_usable_coefficient_table(regression_script, caplog):
    import logging

    frame = synthetic(200, seed=4)
    train = frame[frame["split"] == "train"]
    log = logging.getLogger("test")

    coef, diag, resid = regression_script.ols_inference(
        INFERENCE_FEATURES, train[INFERENCE_FEATURES],
        train["tail_fraction"].to_numpy(), log)

    assert "(intercept)" in set(coef["feature"])
    assert coef["q_value"].notna().sum() >= 1       # BH applied to the family
    assert 0.0 <= diag["r_squared"] <= 1.0
    assert diag["observations_per_parameter"] > 2   # not rank-deficient
    assert len(resid) == len(train)
    assert abs(float(np.mean(resid))) < 1e-8        # OLS residuals centre on zero


# --- Classification paths ---


def test_logistic_separates_a_frame_with_real_signal(classification_script):
    from sklearn.metrics import roc_auc_score

    frame = synthetic(200, seed=5, signal=True)
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "eval"]
    columns = BLOCKS["C_convergence"]

    model = classification_script.logistic_pipeline(columns, 0).set_params(model__C=1.0)
    model.fit(train[columns], train["final_correct"])
    auroc = roc_auc_score(test["final_correct"], model.predict_proba(test[columns])[:, 1])

    assert auroc > 0.75


def test_threshold_is_chosen_on_the_frame_it_is_given(classification_script):
    """Youden's J on a perfectly separable score must land between the classes."""
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])

    chosen = classification_script.choose_threshold(y, p)
    assert 0.1 < chosen["threshold"] <= 0.9
    assert chosen["train_youden_j"] == pytest.approx(1.0)
    assert chosen["train_fpr"] == pytest.approx(0.0)


def test_scores_on_a_perfect_and_a_useless_classifier(classification_script):
    y = np.array([0] * 40 + [1] * 60)
    perfect = np.concatenate([np.full(40, 0.05), np.full(60, 0.95)])
    useless = np.full(100, 0.6)

    good = classification_script.classification_scores(y, perfect, 0.5, 200, 0)
    bad = classification_script.classification_scores(y, useless, 0.5, 200, 0)

    assert good["auroc"] == pytest.approx(1.0)
    assert good["balanced_accuracy"] == pytest.approx(1.0)
    assert good["brier"] < bad["brier"]
    assert bad["auroc"] == pytest.approx(0.5)
    # AUPRC of a constant score is the prevalence, not zero
    assert bad["auprc"] == pytest.approx(0.6, abs=0.02)


def test_error_analysis_counts_the_two_error_types_separately(classification_script):
    frame = synthetic(120, seed=6)
    test = frame[frame["split"] == "eval"].copy()
    y = test["final_correct"].to_numpy()
    # a score that is exactly wrong, so every error is attributable
    p = 1.0 - y.astype(float)

    errors = classification_script.error_analysis(test, y, p, 0.5)

    assert set(errors["stratum"]) == {"dataset", "trace length", "difficulty",
                                      "truncated"}
    for _, group in errors.groupby("stratum"):
        assert group["n"].sum() == len(test)
    # with an inverted score every row is an error, split across the two types
    assert errors["error_rate"].eq(1.0).all()
    totals = errors[errors["stratum"] == "dataset"]
    assert totals["false_correct"].sum() + totals["false_incorrect"].sum() == len(test)


def test_odds_ratio_table_is_finite_and_corrected(classification_script, caplog):
    import logging

    frame = synthetic(200, seed=7)
    train = frame[frame["split"] == "train"]

    odds = classification_script.odds_ratio_table(
        INFERENCE_FEATURES, train[INFERENCE_FEATURES],
        train["final_correct"].to_numpy(), 0, logging.getLogger("test"))

    assert not odds.empty
    assert np.isfinite(odds["odds_ratio"]).all()
    assert (odds["odds_ratio"] > 0).all()
    # on a well-conditioned design the unpenalised fit must be the one used,
    # because the penalised fallback cannot report an interval at all
    assert not odds["penalised_no_ci"].any()
    assert odds["q_value"].notna().sum() >= 1

    inner = odds[odds["feature"] != "(intercept)"]
    assert np.isfinite(inner["or_ci_low"]).all()
    assert (inner["or_ci_low"] <= inner["odds_ratio"]).all()
    assert (inner["odds_ratio"] <= inner["or_ci_high"]).all()
