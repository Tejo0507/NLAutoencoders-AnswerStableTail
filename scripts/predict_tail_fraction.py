"""Stage `regression_tail`: predicting tail fraction.

**Target.** ``tail_fraction`` - the proportion of a reasoning trace's chunks
that lie inside the Answer-Stable Tail, i.e. after the point from which the
answer no longer changes under truncation or resampling.

**Why this target and not another.** The project's question is whether a
verbalised readout of layer-20 activations says anything useful about the
post-answer region of a trace *beyond what much cheaper signals already say*.
``tail_fraction`` is the study's own operational measure of that region - it is
defined in ``PROJECT_PLAN.md`` section 5, computed by a stage that is forbidden
from reading an activation, and it is the denominator of the token-saving
figure the whole O3 comparison is scored on. A regression of it on cheap,
activation-free features is therefore not a side quest: it is falsification
tests F1 ("the readout adds nothing over length") and F2 ("...over difficulty")
turned into an estimate with a confidence interval instead of a yes/no.

**What the answer means either way.** A high cheap-tier R-squared raises the bar
the expensive readout has to clear and is a genuine threat to the study's
interest. A low one says the redundancy is not a surface-form phenomenon, which
is the premise the rest of the pipeline rests on. Both are reportable.

Model ladder, in increasing capacity:

1. ``mean`` - predict the training mean. The reference every R-squared is against.
2. ``ridge`` - regularised linear, the simple interpretable model.
3. ``ols`` - unregularised linear, fitted for *inference*: coefficients,
   confidence intervals, and the residual diagnostics a linear model licenses.
4. ``gbm`` - histogram gradient boosting, the stronger model. Justified only if
   it beats ridge on held-out error by a paired-bootstrap interval that
   excludes zero; if it does not, the linear model is the reported one.

Splitting. The run's existing ``train``/``eval`` assignment is used unchanged.
It is a hash of the problem id fixed in ``configs/base.yaml`` long before this
analysis existed, which makes it the one split in the project that cannot have
been chosen to flatter a result. Hyperparameters are selected by 5-fold CV
*inside* ``train``; ``eval`` is scored exactly once per model.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _stage import base_parser, setup, should_skip  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from nlaast.analysis.stats import TestResult, benjamini_hochberg  # noqa: E402
from nlaast.analysis.tables import markdown, write_table  # noqa: E402
from nlaast.logging_utils import read_jsonl, write_json  # noqa: E402
from nlaast.supervised import (  # noqa: E402
    BLOCKS,
    FEATURE_BLOCKS,
    INFERENCE_FEATURES,
    bootstrap_metric,
    build_feature_table,
    make_preprocessor,
    nested_block_scores,
    paired_metric_difference,
    permutation_null,
)
from nlaast.supervised.evaluation import feature_names, independent_columns  # noqa: E402

STAGE = "regression_tail"
TARGET = "tail_fraction"
DPI = 200
#: Statuses that carry a measured tail. ``no_stable_point`` records
#: ``tail_fraction = 0.0`` as a *convention*, not a measurement, so it is held
#: out of the primary sample and reported as a sensitivity cohort instead.
PRIMARY_STATUSES = ("ok", "stable_at_zero")


# --- Sample definition ---


def define_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The primary analysis sample plus the two pre-declared sensitivity cohorts.

    Truncated traces are excluded from the primary sample because their "final"
    answer is the answer at the token cap, so tail-ness relative to it is not
    the quantity we mean. They are not deleted - cohort 3 puts them back.
    """
    has_tail = df["ast_status"].isin(PRIMARY_STATUSES)
    return {
        "primary": df[has_tail & (df["truncated"] == 0)].copy(),
        "with_no_stable_point": df[df["truncated"] == 0].copy(),
        "with_truncated": df[has_tail].copy(),
    }


# --- Models ---


def ridge_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(columns, scale=True)),
        ("model", Ridge()),
    ])


def gbm_pipeline(columns: list[str], seed: int) -> Pipeline:
    return Pipeline([
        # Trees do not need the scaler; the imputer stays, because
        # ``math_level`` is genuinely missing for every GSM8K problem and the
        # indicator column is informative.
        ("pre", make_preprocessor(columns, scale=False)),
        ("model", HistGradientBoostingRegressor(random_state=seed)),
    ])


RIDGE_GRID = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
GBM_GRID = {
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.03, 0.1],
    "model__max_iter": [200, 400],
    # A leaf floor this high on ~100 training rows is not timidity: with fewer,
    # a leaf is memorising single problems and the CV estimate stops meaning
    # anything.
    "model__min_samples_leaf": [8, 15],
}


def tune(pipeline: Pipeline, grid: dict, X: pd.DataFrame, y: np.ndarray,
         seed: int, log) -> tuple[Pipeline, dict]:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid, scoring="neg_mean_absolute_error",
                          cv=cv, n_jobs=1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    log.info("  tuned %s: %s (cv MAE %.4f)",
             type(pipeline.named_steps["model"]).__name__,
             search.best_params_, -search.best_score_)
    return search.best_estimator_, {
        "best_params": {k: v for k, v in search.best_params_.items()},
        "cv_mae": float(-search.best_score_),
    }


# --- Inference on the linear model ---


def ols_inference(columns: list[str], X: pd.DataFrame, y: np.ndarray,
                  log) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Fit OLS on the design matrix and return coefficients plus diagnostics.

    The OLS fit exists for interpretation, not prediction - ridge is the
    predictive linear model - and it is fitted on the pre-declared
    ``INFERENCE_FEATURES`` subset rather than all thirty-five columns, because
    a rank-deficient design produces a coefficient table that looks
    interpretable and is not.

    Zero-variance columns are dropped first: an all-constant imputation
    indicator makes the design singular and statsmodels would quietly return a
    pseudo-inverse solution instead of failing.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    pre = make_preprocessor(columns, scale=True)
    design, names, dropped = independent_columns(pre.fit_transform(X),
                                                 feature_names(pre))
    log.info("  OLS design: %d rows x %d columns%s",
             design.shape[0], design.shape[1],
             f" (dropped {', '.join(dropped)})" if dropped else "")

    exog = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(y, exog).fit()

    coef = pd.DataFrame({
        "feature": ["(intercept)", *names],
        "coefficient": fit.params,
        "std_error": fit.bse,
        "t": fit.tvalues,
        "p_value": fit.pvalues,
        "ci_low": fit.conf_int()[:, 0],
        "ci_high": fit.conf_int()[:, 1],
    })

    # Benjamini-Hochberg over the coefficient family, matching the project's
    # convention for any pre-registered family of tests (F9).
    tests = [TestResult(name=r.feature, statistic=float(r.t), p_value=float(r.p_value),
                        effect=float(r.coefficient), effect_name="beta")
             for r in coef.itertuples() if r.feature != "(intercept)"]
    corrected = {t.name: t for t in benjamini_hochberg(tests, q=0.05)}
    coef["q_value"] = [corrected[f].q_value if f in corrected else np.nan
                       for f in coef["feature"]]
    coef["significant_bh"] = [bool(corrected[f].significant) if f in corrected else False
                              for f in coef["feature"]]

    resid = np.asarray(fit.resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, exog)

    # VIF only where it can be computed - with p close to n the design is
    # near-singular and a VIF of 1e9 is an artefact, not a finding.
    vifs: list[float] = []
    if design.shape[0] > design.shape[1] + 2:
        for i in range(exog.shape[1]):
            try:
                vifs.append(float(variance_inflation_factor(exog, i)))
            except Exception:
                vifs.append(float("nan"))
    coef["vif"] = vifs if len(vifs) == len(coef) else [float("nan")] * len(coef)

    diagnostics = {
        "n": int(design.shape[0]),
        "n_parameters": int(exog.shape[1]),
        "r_squared": float(fit.rsquared),
        "r_squared_adj": float(fit.rsquared_adj),
        "f_statistic": float(fit.fvalue),
        "f_p_value": float(fit.f_pvalue),
        "condition_number": float(fit.condition_number),
        "durbin_watson": float(sm.stats.durbin_watson(resid)),
        "jarque_bera_p": float(sm.stats.jarque_bera(resid)[1]),
        "breusch_pagan_lm": float(bp_lm),
        "breusch_pagan_p": float(bp_p),
        "max_vif": float(np.nanmax(coef["vif"].to_numpy())) if vifs else float("nan"),
        "observations_per_parameter": float(design.shape[0] / exog.shape[1]),
        "dropped_columns": dropped,
    }
    return coef, diagnostics, resid


# --- Figures ---


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)


def figure_predictions(out_dir, y_true, preds: dict[str, np.ndarray], scores) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4.0),
                             squeeze=False)
    lo, hi = 0.0, max(1.0, float(np.max(y_true)) * 1.05)
    for ax, (name, pred) in zip(axes[0], preds.items()):
        ax.scatter(y_true, pred, s=22, alpha=0.65, edgecolor="none", color="#4C72B0")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, label="perfect")
        row = scores[scores["model"] == name].iloc[0]
        _style(ax, f"{name}  (R² = {row['r2']:.3f}, MAE = {row['mae']:.3f})",
               "observed tail fraction", "predicted")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Held-out predictions (eval split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_predictions.png", dpi=DPI)
    plt.close(fig)


def figure_diagnostics(out_dir, fitted, resid, n_tokens) -> None:
    import scipy.stats as st

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    axes[0, 0].scatter(fitted, resid, s=20, alpha=0.6, color="#4C72B0", edgecolor="none")
    axes[0, 0].axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    _style(axes[0, 0], "Residuals vs fitted", "fitted", "residual")

    st.probplot(resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set(markersize=3.5, alpha=0.7)
    _style(axes[0, 1], "Normal Q-Q", "theoretical quantile", "ordered residual")

    axes[1, 0].hist(resid, bins=min(25, max(8, len(resid) // 6)),
                    color="#55A868", edgecolor="white")
    _style(axes[1, 0], "Residual distribution", "residual", "count")

    axes[1, 1].scatter(n_tokens, resid, s=20, alpha=0.6, color="#DD8452",
                       edgecolor="none")
    axes[1, 1].axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    _style(axes[1, 1], "Residuals vs trace length\n(F1: is length doing the work?)",
           "generated tokens", "residual")

    fig.suptitle("OLS diagnostics (train split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_diagnostics.png", dpi=DPI)
    plt.close(fig)


def figure_importance(out_dir, imp: pd.DataFrame, title: str, fname: str) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(top) + 1.6))
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#4C72B0",
            error_kw={"linewidth": 0.8, "ecolor": "#444444"})
    ax.axvline(0.0, color="k", linewidth=0.8)
    _style(ax, title, "increase in MAE when permuted", "")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=DPI)
    plt.close(fig)


def figure_blocks(out_dir, blocks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(blocks))
    ax.bar(x, blocks["cv_r2"], color="#4C72B0", width=0.55)
    for i, (value, inc) in enumerate(zip(blocks["cv_r2"], blocks["increment"])):
        label = f"{value:.3f}" if not np.isfinite(inc) else f"{value:.3f}\n({inc:+.3f})"
        ax.text(i, value + 0.012, label, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in blocks["block"]], fontsize=8)
    ax.axhline(0.0, color="k", linewidth=0.8)
    _style(ax, "Cross-validated R² by cumulative feature block\n"
               "(train split, 5-fold; increment over the previous block in brackets)",
           "", "CV R²")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_blocks.png", dpi=DPI)
    plt.close(fig)


def figure_target(out_dir, cohorts: dict[str, pd.DataFrame]) -> None:
    primary = cohorts["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].hist(primary[TARGET], bins=18, color="#C44E52", edgecolor="white")
    axes[0].axvline(float(primary[TARGET].mean()), color="k", linestyle="--",
                    linewidth=1.0, label=f"mean = {primary[TARGET].mean():.3f}")
    axes[0].legend(fontsize=8, frameon=False)
    _style(axes[0], "Target distribution (primary cohort)", "tail fraction", "count")

    groups = [primary.loc[primary["dataset"] == d, TARGET].to_numpy()
              for d in sorted(primary["dataset"].unique())]
    axes[1].boxplot(groups, tick_labels=sorted(primary["dataset"].unique()),
                    widths=0.5)
    _style(axes[1], "Tail fraction by benchmark", "", "tail fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_target.png", dpi=DPI)
    plt.close(fig)


# --- Entry point ---


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--permutations", type=int, default=200,
                   help="label permutations for the F6-style null (default 200)")
    p.add_argument("--allow-small", action="store_true",
                   help="run below the minimum sample guard (smoke testing only; "
                        "the numbers it produces are not reportable)")
    args = p.parse_args(argv)
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    traces = read_jsonl(cfg.stage_dir("traces") / "traces.jsonl")
    ast_rows = read_jsonl(cfg.stage_dir("ast") / "ast.jsonl")
    problems = read_jsonl(cfg.dir / "data" / "problems.jsonl")
    if not traces or not ast_rows:
        log.error("need the `traces` and `ast` stages first (traces=%d, ast=%d)",
                  len(traces), len(ast_rows))
        return 1

    manifest.start_stage(STAGE, target=TARGET)
    out_dir = cfg.stage_dir(STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_feature_table(traces, ast_rows, problems)
    write_table(table, out_dir, "feature_table")
    cohorts = define_cohorts(table)
    for name, cohort in cohorts.items():
        log.info("cohort %-22s n=%3d  mean %s=%.3f",
                 name, len(cohort), TARGET, cohort[TARGET].mean())

    df = cohorts["primary"]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "eval"]
    if (len(train) < 30 or len(test) < 20) and not args.allow_small:
        log.error("primary cohort too small to model: train=%d eval=%d "
                  "(--allow-small overrides, for smoke testing only)",
                  len(train), len(test))
        return 1

    all_columns = BLOCKS["C_convergence"]
    X_train, y_train = train[all_columns + ["id"]].drop(columns="id"), train[TARGET].to_numpy()
    X_test, y_test = test[all_columns + ["id"]].drop(columns="id"), test[TARGET].to_numpy()
    log.info("train n=%d, eval n=%d, %d features", len(train), len(test), len(all_columns))

    # ---- model ladder --------------------------------------------------
    log.info("fitting models")
    ridge, ridge_info = tune(ridge_pipeline(all_columns), RIDGE_GRID,
                             X_train, y_train, cfg.seed, log)
    gbm, gbm_info = tune(gbm_pipeline(all_columns, cfg.seed), GBM_GRID,
                         X_train, y_train, cfg.seed, log)

    preds = {
        "mean": np.full(len(y_test), float(np.mean(y_train))),
        "ridge": ridge.predict(X_test),
        "gbm": gbm.predict(X_test),
    }

    boot = cfg.analysis.bootstrap_iterations
    rows = []
    for name, pred in preds.items():
        rows.append({
            "model": name,
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            **{f"mae_{k}": v for k, v in
               bootstrap_metric(y_test, pred, mean_absolute_error,
                                n_iterations=boot, seed=cfg.seed).items()
               if k in ("low", "high")},
            **{f"r2_{k}": v for k, v in
               bootstrap_metric(y_test, pred, r2_score,
                                n_iterations=boot, seed=cfg.seed).items()
               if k in ("low", "high")},
        })
    scores = pd.DataFrame(rows)
    write_table(scores, out_dir, "model_scores")
    log.info("held-out scores:\n%s", markdown(scores))

    # ---- is the stronger model justified? ------------------------------
    comparisons = {
        "gbm_minus_ridge_mae": paired_metric_difference(
            y_test, preds["gbm"], preds["ridge"], mean_absolute_error,
            n_iterations=boot, seed=cfg.seed),
        "ridge_minus_mean_mae": paired_metric_difference(
            y_test, preds["ridge"], preds["mean"], mean_absolute_error,
            n_iterations=boot, seed=cfg.seed),
    }
    gbm_justified = bool(comparisons["gbm_minus_ridge_mae"]["estimate"] < 0
                         and comparisons["gbm_minus_ridge_mae"]["excludes_zero"])
    reported = "gbm" if gbm_justified else "ridge"
    log.info("gbm - ridge MAE = %+.4f [%+.4f, %+.4f] -> reported model: %s",
             comparisons["gbm_minus_ridge_mae"]["estimate"],
             comparisons["gbm_minus_ridge_mae"]["low"],
             comparisons["gbm_minus_ridge_mae"]["high"], reported)

    # ---- nested blocks (CV on train only) ------------------------------
    def block_score(columns: list[str]) -> dict[str, float]:
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = cross_val_score(ridge_pipeline(columns).set_params(
                                     model__alpha=ridge_info["best_params"]["model__alpha"]),
                                 train[columns], y_train, cv=cv, scoring="r2")
            mae = cross_val_score(ridge_pipeline(columns).set_params(
                                      model__alpha=ridge_info["best_params"]["model__alpha"]),
                                  train[columns], y_train, cv=cv,
                                  scoring="neg_mean_absolute_error")
        return {"primary": float(np.mean(r2)), "cv_r2_sd": float(np.std(r2)),
                "cv_mae": float(-np.mean(mae))}

    blocks = nested_block_scores(block_score).rename(columns={"primary": "cv_r2"})
    write_table(blocks, out_dir, "block_increments")
    log.info("block increments:\n%s", markdown(blocks))

    # ---- inference and diagnostics on the linear model -----------------
    coef, diagnostics, resid = ols_inference(INFERENCE_FEATURES,
                                             train[INFERENCE_FEATURES], y_train, log)
    write_table(coef, out_dir, "ols_coefficients")
    log.info("OLS: R²=%.3f adj=%.3f  BP p=%.3g  JB p=%.3g  cond=%.3g  n/p=%.2f",
             diagnostics["r_squared"], diagnostics["r_squared_adj"],
             diagnostics["breusch_pagan_p"], diagnostics["jarque_bera_p"],
             diagnostics["condition_number"], diagnostics["observations_per_parameter"])

    # ---- feature analysis ----------------------------------------------
    best = {"ridge": ridge, "gbm": gbm}[reported]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(best, X_test, y_test, n_repeats=30,
                                      random_state=cfg.seed,
                                      scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({
        "feature": all_columns,
        # permutation_importance reports the *drop* in the score; with a
        # negated-error scorer a harmful permutation gives a negative drop, so
        # the sign is flipped to read as "MAE increase".
        "importance_mean": -perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    write_table(imp, out_dir, "permutation_importance")
    log.info("top features (%s):\n%s", reported, markdown(imp.head(8)))

    # ---- label-shuffle null (F6 analogue) ------------------------------
    def fit_and_score(y_shuffled: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ridge_pipeline(all_columns).set_params(
                model__alpha=ridge_info["best_params"]["model__alpha"])
            cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
            return float(np.mean(cross_val_score(model, X_train, y_shuffled,
                                                 cv=cv, scoring="r2")))

    null = permutation_null(fit_and_score, y_train,
                            n_permutations=args.permutations, seed=cfg.seed)
    log.info("label-shuffle null: observed CV R²=%.3f vs null %.3f±%.3f (p=%.4f)",
             null["observed"], null["null_mean"], null["null_sd"], null["p_value"])

    # ---- sensitivity cohorts -------------------------------------------
    sens = []
    for name, cohort in cohorts.items():
        tr = cohort[cohort["split"] == "train"]
        te = cohort[cohort["split"] == "eval"]
        if len(tr) < 25 or len(te) < 15:
            sens.append({"cohort": name, "n_train": len(tr), "n_eval": len(te),
                         "r2": float("nan"), "mae": float("nan")})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ridge_pipeline(all_columns).set_params(
                model__alpha=ridge_info["best_params"]["model__alpha"])
            model.fit(tr[all_columns], tr[TARGET].to_numpy())
            pred = model.predict(te[all_columns])
        sens.append({"cohort": name, "n_train": len(tr), "n_eval": len(te),
                     "mean_target": float(cohort[TARGET].mean()),
                     "r2": float(r2_score(te[TARGET], pred)),
                     "mae": float(mean_absolute_error(te[TARGET], pred))})
    sensitivity = pd.DataFrame(sens)
    write_table(sensitivity, out_dir, "sensitivity_cohorts")
    log.info("sensitivity:\n%s", markdown(sensitivity))

    # ---- figures --------------------------------------------------------
    figure_target(out_dir, cohorts)
    figure_predictions(out_dir, y_test, {k: v for k, v in preds.items() if k != "mean"},
                       scores)
    figure_diagnostics(out_dir, y_train - resid, resid, train["n_tokens"].to_numpy())
    figure_importance(out_dir, imp, f"Permutation importance - {reported} (eval split)",
                      "fig_importance.png")
    figure_blocks(out_dir, blocks)

    summary = {
        "target": TARGET,
        "n_total_traces": len(table),
        "cohort_sizes": {k: int(len(v)) for k, v in cohorts.items()},
        "n_train": int(len(train)),
        "n_eval": int(len(test)),
        "n_features": len(all_columns),
        "feature_blocks": {k: len(v) for k, v in FEATURE_BLOCKS.items()},
        "target_mean": float(df[TARGET].mean()),
        "target_sd": float(df[TARGET].std()),
        "tuning": {"ridge": ridge_info, "gbm": gbm_info},
        "scores": scores.to_dict(orient="records"),
        "comparisons": comparisons,
        "reported_model": reported,
        "gbm_justified": gbm_justified,
        "block_increments": blocks.to_dict(orient="records"),
        "ols_features": INFERENCE_FEATURES,
        "ols_diagnostics": diagnostics,
        "label_shuffle_null": null,
        "sensitivity": sensitivity.to_dict(orient="records"),
        "top_features": imp.head(10).to_dict(orient="records"),
    }
    write_json(out_dir / "summary.json", summary)
    manifest.finish_stage(STAGE, status="complete", output=str(out_dir),
                          metrics={"r2_reported": float(scores.loc[
                              scores["model"] == reported, "r2"].iloc[0]),
                              "n_eval": int(len(test))})
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
