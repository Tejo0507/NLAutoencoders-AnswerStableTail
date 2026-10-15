"""The odds-ratio table's penalised fallback must not crash its own callers.

`odds_ratio_table` falls back to an L2-penalised logit when the unpenalised fit
gives no usable standard errors - under separation, for example - and then
deliberately returns no `q_value` column, because there are no p-values to
correct. Both consumers selected `q_value` unconditionally:
`predict_correctness.py` crashed with `KeyError: ['q_value']` right after
writing the table, and `write_supervised_report.py` would have crashed the same
way. Found by running the classification stage on the 216 main-run traces
available on 2026-10-15, where the MLE does not converge.

The report also called every table "the penalised logistic fit", which was
wrong for the unpenalised one.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from nlaast.supervised import INFERENCE_FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def clf():
    return importlib.import_module("predict_correctness")


@pytest.fixture(scope="module")
def report():
    return importlib.import_module("write_supervised_report")


def separated_frame(n=120, seed=3):
    """A design where one feature separates the label perfectly."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in INFERENCE_FEATURES
                          if c not in ("dataset", "subject", "truncated")})
    if "dataset" in INFERENCE_FEATURES:
        frame["dataset"] = rng.choice(["gsm8k", "math"], size=n)
    if "subject" in INFERENCE_FEATURES:
        frame["subject"] = rng.choice(["algebra", "gsm8k"], size=n)
    if "truncated" in INFERENCE_FEATURES:
        frame["truncated"] = rng.integers(0, 2, size=n)
    numeric = next(c for c in INFERENCE_FEATURES
                   if c not in ("dataset", "subject", "truncated"))
    y = (frame[numeric] > 0).astype(int).to_numpy()
    return frame[list(INFERENCE_FEATURES)], y


def penalised_table():
    return pd.DataFrame({
        "feature": ["(intercept)", "n_tokens"],
        "coefficient": [0.1, -0.4],
        "odds_ratio": [1.1, 0.67],
        "std_error": [np.nan, np.nan],
        "p_value": [np.nan, np.nan],
        "or_ci_low": [np.nan, np.nan],
        "or_ci_high": [np.nan, np.nan],
        "penalised_no_ci": [True, True],
    })


def mle_table():
    t = penalised_table()
    t["penalised_no_ci"] = False
    t["p_value"] = [0.5, 0.01]
    t["or_ci_low"], t["or_ci_high"] = [0.8, 0.5], [1.5, 0.9]
    t["q_value"] = [np.nan, 0.02]
    t["significant_bh"] = [False, True]
    return t


class TestTheFallbackIsReachableAndHasNoQValue:
    def test_separation_takes_the_penalised_branch(self, clf):
        X, y = separated_frame()
        odds = clf.odds_ratio_table(INFERENCE_FEATURES, X, y, 0,
                                    logging.getLogger("test"))
        assert not odds.empty
        assert odds["penalised_no_ci"].all()
        assert "q_value" not in odds.columns

    def test_the_display_columns_are_the_ones_the_fit_produced(self, clf):
        cols = clf.odds_columns(penalised_table())
        assert "q_value" not in cols
        assert "significant_bh" not in cols
        penalised_table()[cols]  # the selection that used to raise KeyError

    def test_an_mle_table_keeps_its_inference_columns(self, clf):
        cols = clf.odds_columns(mle_table())
        assert {"q_value", "significant_bh", "p_value"} <= set(cols)


class TestTheReportSaysWhichFitItIs:
    def test_a_penalised_table_renders_without_crashing(self, report):
        text = "\n".join(report.odds_section(penalised_table()))
        assert "L2-penalised" in text
        assert "no intervals and no tests" in text
        assert "q_value" not in text

    def test_an_unpenalised_table_is_not_called_penalised(self, report):
        text = "\n".join(report.odds_section(mle_table()))
        assert "unpenalised" in text
        assert "L2-penalised" not in text
        assert "q_value" in text

    def test_an_empty_table_adds_nothing(self, report):
        assert report.odds_section(pd.DataFrame()) == []
