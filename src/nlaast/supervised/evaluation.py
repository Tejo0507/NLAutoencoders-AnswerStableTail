"""Shared evaluation machinery for the two supervised analyses.

Everything here is deliberately small and inspectable. The preprocessing is a
scikit-learn ``Pipeline``, which is not a stylistic choice: fitting the imputer
and the scaler inside the pipeline is what stops the test fold's mean leaking
into the training fold's standardisation. A manual ``StandardScaler`` applied
to the whole table before splitting is the single most common way this kind of
analysis quietly inflates its own scores, and it is the reason the split is
never crossed outside a ``fit``.

The uncertainty reporting reuses the study's existing convention - percentile
bootstrap over problems, and a *paired* bootstrap when two models are compared
on the same problems - so these numbers sit beside the O3 numbers without a
change of method.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..logging_utils import get
from .features import BLOCKS, categorical_columns, numeric_columns

log = get(__name__)


def make_preprocessor(columns: Sequence[str], *, scale: bool = True) -> ColumnTransformer:
    """Impute, optionally standardise, one-hot the two categoricals.

    ``scale=False`` for tree ensembles, which are invariant to monotone
    rescaling and only pay the cost of it.
    """
    num = numeric_columns(columns)
    cat = categorical_columns(columns)

    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median",
                                                             add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [("num", Pipeline(steps), num)]
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                         sparse_output=False)),
            ]),
            cat,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names after imputation indicators and one-hot expansion."""
    try:
        return [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - sklearn version differences
        return [f"f{i}" for i in range(preprocessor.transform_count_)]


def independent_columns(design: np.ndarray, names: Sequence[str],
                        *, tolerance: float = 0.999) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop constant and linearly redundant columns, keeping the earlier one.

    The inference models need a design matrix of full rank: a rank-deficient
    one makes statsmodels return a pseudo-inverse solution, whose coefficients
    and standard errors look ordinary and mean nothing.

    One redundancy is structural here rather than accidental. ``math_level`` is
    missing for exactly the GSM8K problems, so the imputer's missingness
    indicator for it is *identical* to the ``dataset=gsm8k`` dummy. Two columns
    carrying the same information cannot both get a coefficient.

    Constant columns go first, then any column almost perfectly correlated with
    a column already kept. Keeping the earlier one makes the choice depend on
    the declared feature order rather than on the data, so it is stable across
    runs and across cohorts. Returns the pruned design, its names, and the
    names dropped.
    """
    design = np.asarray(design, dtype=np.float64)
    kept: list[int] = []
    dropped: list[str] = []

    for j in range(design.shape[1]):
        column = design[:, j]
        if column.std() <= 1e-10:
            dropped.append(f"{names[j]} (constant)")
            continue
        redundant = next(
            (names[i] for i in kept
             if abs(np.corrcoef(column, design[:, i])[0, 1]) >= tolerance),
            None,
        )
        if redundant is not None:
            dropped.append(f"{names[j]} (duplicates {redundant})")
            continue
        kept.append(j)

    return design[:, kept], [names[i] for i in kept], dropped


# --- Uncertainty ---


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a metric, resampling problems with replacement.

    Resamples that happen to contain a single class (so AUROC is undefined) are
    discarded rather than coerced; the count of usable draws is reported, which
    is the honest way to say "this interval rests on fewer draws than asked".
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.size
    point = float(metric(y_true, y_pred))

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        try:
            value = float(metric(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        return {"estimate": point, "low": float("nan"), "high": float("nan"),
                "confidence": confidence, "n": n, "n_draws": 0}

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {"estimate": point, "low": float(low), "high": float(high),
            "confidence": confidence, "n": n, "n_draws": len(draws)}


def paired_metric_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for ``metric(a) - metric(b)`` on the *same* resampled problems.

    Paired, because the two models are scored on identical problems and the
    problem-to-problem variance is shared. An unpaired interval on a difference
    of this kind is wider than the data warrants and would hide a real gap.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = y_true.size
    point = float(metric(y_true, pred_a)) - float(metric(y_true, pred_b))

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        try:
            value = (float(metric(y_true[idx], pred_a[idx]))
                     - float(metric(y_true[idx], pred_b[idx])))
        except ValueError:
            continue
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        return {"estimate": point, "low": float("nan"), "high": float("nan"),
                "confidence": confidence, "n": n, "n_draws": 0, "excludes_zero": False}

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    # Two-sided bootstrap p-value: how often the sign of the resampled
    # difference disagrees with the sign of the point estimate.
    p = 2.0 * min(np.mean(np.asarray(draws) <= 0.0), np.mean(np.asarray(draws) >= 0.0))
    return {"estimate": point, "low": float(low), "high": float(high),
            "confidence": confidence, "n": n, "n_draws": len(draws),
            "p_value": float(min(1.0, p)),
            "excludes_zero": bool(low > 0 or high < 0)}


def permutation_null(
    fit_score: Callable[[np.ndarray], float],
    y: np.ndarray,
    *,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Refit on shuffled labels and report where the real score sits.

    This is the analysis-level counterpart of falsification test F6: a model
    whose score does not collapse when the labels are destroyed is reading
    something other than the signal it claims to read.
    """
    observed = float(fit_score(y))
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_permutations):
        value = float(fit_score(rng.permutation(y)))
        if np.isfinite(value):
            null.append(value)

    arr = np.asarray(null) if null else np.asarray([np.nan])
    # (#{null >= observed} + 1) / (B + 1) - the +1 keeps p away from exactly 0,
    # which a finite permutation set cannot licence.
    p = (np.sum(arr >= observed) + 1) / (arr.size + 1)
    return {"observed": observed, "null_mean": float(np.nanmean(arr)),
            "null_sd": float(np.nanstd(arr)), "null_max": float(np.nanmax(arr)),
            "n_permutations": int(arr.size), "p_value": float(p)}


# --- Nested feature blocks ---


def nested_block_scores(
    fit_score: Callable[[list[str]], dict[str, float]],
) -> pd.DataFrame:
    """Score each cumulative feature block and report the increment it buys.

    The headline question of the study is incremental: does an expensive signal
    add anything over cheap ones? Asking the same question *within* the cheap
    tier is the honest way to present these models - a pooled R-squared hides
    which tier of information actually carried it.
    """
    records = []
    previous: float | None = None
    for name, columns in BLOCKS.items():
        scores = fit_score(columns)
        primary = scores["primary"]
        records.append({
            "block": name,
            "n_features": len(columns),
            **scores,
            "increment": float("nan") if previous is None else primary - previous,
        })
        previous = primary
    return pd.DataFrame(records)
