"""Stage `classify_correct`: predicting final-answer correctness.

**Target.** ``final_correct`` - whether the trace's final answer is equivalent
to the gold answer under ``math_verify``. Binary, and verified by the same
external checker the rest of the pipeline uses, so the label is not this
analysis's own opinion.

**Why this target and not another.** Safe stopping, the outcome the whole O3
comparison is scored on, is *defined* in terms of verified correctness:
``PROJECT_PLAN.md`` section 5 calls a stop safe iff the forced answer and the
full-trace answer are either both right or both wrong. A rule that stops early
on a trace heading for a wrong answer has not saved tokens, it has locked in an
error. So the question "can correctness be predicted at all, and from what?"
sits directly underneath the stopping comparison.

It is also the target of an existing arm. ``probe.kind: logistic`` in
``configs/base.yaml`` is a hidden-state correctness probe on layer-20
activations (O3, after Zhang et al.). This stage fits the *same* label with the
*same* model family on *no activations at all*. Its AUROC is therefore the floor
the layer-20 probe has to clear before decodability can be claimed as the thing
doing the work - which is falsification tests F1 and F2 again, on the other
target.

**What the answer means either way.** A cheap classifier near chance says
correctness is not written on the surface of the trace, and the probe's AUROC
is interpretable as its own. A strong cheap classifier means any probe result
has to be reported as an increment over it, not as an absolute.

Model ladder:

1. ``majority`` - the prevalence baseline, so no metric is read without one.
2. ``logistic`` - L2 logistic regression, the interpretable model, and the same
   family as the project's own probe.
3. ``gbm`` - histogram gradient boosting, the stronger model, reported as the
   headline only if it beats logistic by a paired-bootstrap AUROC interval
   excluding zero.

Splitting and leakage are handled exactly as in the regression analysis: the run's
pre-existing hash-based ``train``/``eval`` assignment, 5-fold CV inside train
for hyperparameters, ``eval`` scored once. All preprocessing lives inside the
pipeline, so no fold ever sees another fold's imputation or scaling statistics.
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

from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
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

STAGE = "classify_correct"
TARGET = "final_correct"
DPI = 200


# --- Models ---


def logistic_pipeline(columns: list[str], seed: int) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(columns, scale=True)),
        # ``class_weight='balanced'`` because the model is good at this task and
        # accuracy on an 80/20 split is uninformative; balanced accuracy and
        # AUROC are the metrics, so the fit should optimise for both classes.
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                     random_state=seed)),
    ])


def gbm_pipeline(columns: list[str], seed: int) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(columns, scale=False)),
        ("model", HistGradientBoostingClassifier(random_state=seed)),
    ])


LOGISTIC_GRID = {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]}
GBM_GRID = {
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.03, 0.1],
    "model__max_iter": [200, 400],
    "model__min_samples_leaf": [8, 15],
}


def tune(pipeline: Pipeline, grid: dict, X, y, seed: int, log) -> tuple[Pipeline, dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    log.info("  tuned %s: %s (cv AUROC %.4f)",
             type(pipeline.named_steps["model"]).__name__,
             search.best_params_, search.best_score_)
    return search.best_estimator_, {"best_params": dict(search.best_params_),
                                    "cv_auroc": float(search.best_score_)}


# --- Threshold choice ---


def choose_threshold(y_train: np.ndarray, p_train: np.ndarray) -> dict[str, float]:
    """Pick the operating point on the *training* split, never on eval.

    Maximising Youden's J (sensitivity + specificity - 1) rather than accuracy,
    because the two error types are not interchangeable here: a false "correct"
    is a stopping rule endorsing a wrong answer. Choosing the threshold on eval
    would make every number downstream of it optimistic, which is why it is
    fitted here and then frozen.
    """
    fpr, tpr, thresholds = roc_curve(y_train, p_train)
    j = tpr - fpr
    k = int(np.argmax(j))
    return {"threshold": float(thresholds[k]), "train_youden_j": float(j[k]),
             "train_tpr": float(tpr[k]), "train_fpr": float(fpr[k])}


def classification_scores(y: np.ndarray, p: np.ndarray, threshold: float,
                          boot: int, seed: int) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    auroc = bootstrap_metric(y, p, roc_auc_score, n_iterations=boot, seed=seed)
    auprc = bootstrap_metric(y, p, average_precision_score, n_iterations=boot, seed=seed)
    return {
        "auroc": auroc["estimate"], "auroc_low": auroc["low"], "auroc_high": auroc["high"],
        "auprc": auprc["estimate"], "auprc_low": auprc["low"], "auprc_high": auprc["high"],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(np.mean(pred == y)),
    }


# --- Error analysis ---


def error_analysis(test: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                   threshold: float) -> pd.DataFrame:
    """Error rate per stratum, so a pooled AUROC cannot hide a broken subgroup."""
    pred = (p >= threshold).astype(int)
    frame = test.copy()
    frame["y"], frame["pred"], frame["p"] = y, pred, p
    frame["error"] = (frame["y"] != frame["pred"]).astype(int)
    frame["length_tertile"] = pd.qcut(frame["n_tokens"], 3,
                                      labels=["short", "medium", "long"],
                                      duplicates="drop")
    frame["level_group"] = frame["math_level"].map(
        lambda v: "gsm8k (no level)" if not np.isfinite(v) else f"MATH level {int(v)}")

    out = []
    for name, key in (("dataset", "dataset"), ("trace length", "length_tertile"),
                      ("difficulty", "level_group"), ("truncated", "truncated")):
        for value, group in frame.groupby(key, observed=True):
            if len(group) == 0:
                continue
            out.append({
                "stratum": name, "value": str(value), "n": int(len(group)),
                "prevalence": float(group["y"].mean()),
                "error_rate": float(group["error"].mean()),
                "false_correct": int(((group["pred"] == 1) & (group["y"] == 0)).sum()),
                "false_incorrect": int(((group["pred"] == 0) & (group["y"] == 1)).sum()),
                "mean_p": float(group["p"].mean()),
            })
    return pd.DataFrame(out)


def odds_ratio_table(columns: list[str], X: pd.DataFrame, y: np.ndarray,
                     seed: int, log) -> pd.DataFrame:
    """Logistic coefficients as odds ratios, with Wald CIs and BH correction.

    Fitted with statsmodels rather than read off the sklearn estimator, because
    sklearn does not expose standard errors and an interpretable model without
    an interval on its coefficients is not actually interpretable.

    Fitted on the pre-declared ``INFERENCE_FEATURES`` subset, for the same
    reason as the OLS fit in the regression analysis: with thirty-five features against
    roughly a hundred rows the design is near-singular, and a logistic fit on a
    near-singular design reports odds ratios that are numerical artefacts.
    """
    import statsmodels.api as sm

    pre = make_preprocessor(columns, scale=True)
    design, names, dropped = independent_columns(pre.fit_transform(X),
                                                 feature_names(pre))
    if dropped:
        log.info("  logit design: dropped %s", ", ".join(dropped))

    exog = sm.add_constant(design, has_constant="add")
    # Unpenalised MLE first, because it is the only fit that yields usable
    # standard errors: statsmodels' regularized results leave `bse` and
    # `conf_int` undefined for shrunk coefficients, which would silently fill
    # the table with odds ratios of exactly 1.0 and missing intervals. The
    # penalised fit is kept only as a fallback for perfect separation, where
    # the MLE diverges and reports an infinite odds ratio - a numerical
    # artefact, not an effect - and in that case no interval is claimed.
    penalised = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sm.Logit(y, exog).fit(disp=0, maxiter=200)
        if not np.isfinite(np.asarray(fit.bse)).all():
            raise ValueError("non-finite standard errors")
        conf, bse, pvals = fit.conf_int(), fit.bse, fit.pvalues
    except Exception as exc:
        log.warning("  unpenalised logit did not give usable errors (%s); "
                    "falling back to an L2-penalised fit without intervals", exc)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.Logit(y, exog).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0)
        except Exception as inner:  # pragma: no cover - singular beyond repair
            log.warning("  logit inference failed (%s); table omitted", inner)
            return pd.DataFrame()
        penalised = True
        nan = np.full(exog.shape[1], np.nan)
        conf, bse, pvals = np.column_stack([nan, nan]), nan, nan

    frame = pd.DataFrame({
        "feature": ["(intercept)", *names],
        "coefficient": np.asarray(fit.params),
        "odds_ratio": np.exp(np.asarray(fit.params)),
        "std_error": np.asarray(bse),
        "p_value": np.asarray(pvals),
        "or_ci_low": np.exp(np.asarray(conf)[:, 0]),
        "or_ci_high": np.exp(np.asarray(conf)[:, 1]),
        "penalised_no_ci": penalised,
    })
    if penalised:
        # Without p-values there is nothing for BH to correct, and inventing
        # columns that look like inference would be worse than omitting them.
        return frame.reset_index(drop=True)
    tests = [TestResult(name=r.feature, statistic=float(r.coefficient / max(r.std_error, 1e-12)),
                        p_value=float(r.p_value), effect=float(r.odds_ratio),
                        effect_name="odds_ratio")
             for r in frame.itertuples() if r.feature != "(intercept)"]
    corrected = {t.name: t for t in benjamini_hochberg(tests, q=0.05)}
    frame["q_value"] = [corrected[f].q_value if f in corrected else np.nan
                        for f in frame["feature"]]
    frame["significant_bh"] = [bool(corrected[f].significant) if f in corrected else False
                               for f in frame["feature"]]
    return frame.sort_values("p_value").reset_index(drop=True)


# --- Figures ---


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)


COLOURS = {"logistic": "#4C72B0", "gbm": "#C44E52", "majority": "#888888"}


def figure_curves(out_dir, y, probs: dict[str, np.ndarray], scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    for name, p in probs.items():
        row = scores[scores["model"] == name].iloc[0]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=COLOURS[name], linewidth=1.6,
                     label=f"{name} (AUROC {row['auroc']:.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, color=COLOURS[name], linewidth=1.6,
                     label=f"{name} (AUPRC {row['auprc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.9, label="chance")
    axes[1].axhline(float(np.mean(y)), color="k", linestyle="--", linewidth=0.9,
                    label=f"prevalence ({np.mean(y):.3f})")
    _style(axes[0], "ROC (eval split)", "false positive rate", "true positive rate")
    _style(axes[1], "Precision-recall (eval split)", "recall", "precision")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="lower left" if ax is axes[0] else "best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_curves.png", dpi=DPI)
    plt.close(fig)


def figure_confusion(out_dir, y, p, threshold: float, model_name: str) -> None:
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = ["incorrect", "correct"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#222222")
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.set_yticks([0, 1], labels, fontsize=9)
    ax.set_title(f"Confusion matrix - {model_name}\n"
                 f"threshold {threshold:.3f}, chosen on train", fontsize=10, pad=10)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("actual", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_confusion.png", dpi=DPI)
    plt.close(fig)


def figure_calibration(out_dir, y, probs: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    bins = max(4, min(8, len(y) // 12))
    for name, p in probs.items():
        try:
            true_frac, pred_mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        except ValueError:
            continue
        ax.plot(pred_mean, true_frac, "o-", color=COLOURS[name], linewidth=1.4,
                markersize=5, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="perfect")
    _style(ax, f"Calibration (eval split, {bins} quantile bins)",
           "mean predicted probability", "observed fraction correct")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_calibration.png", dpi=DPI)
    plt.close(fig)


def figure_importance(out_dir, imp: pd.DataFrame, title: str) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(top) + 1.6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="#4C72B0", error_kw={"linewidth": 0.8, "ecolor": "#444444"})
    ax.axvline(0.0, color="k", linewidth=0.8)
    _style(ax, title, "drop in AUROC when permuted", "")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_importance.png", dpi=DPI)
    plt.close(fig)


def figure_blocks(out_dir, blocks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(blocks))
    ax.bar(x, blocks["cv_auroc"], color="#4C72B0", width=0.55)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.9, label="chance")
    for i, (value, inc) in enumerate(zip(blocks["cv_auroc"], blocks["increment"])):
        label = f"{value:.3f}" if not np.isfinite(inc) else f"{value:.3f}\n({inc:+.3f})"
        ax.text(i, value + 0.012, label, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in blocks["block"]], fontsize=8)
    ax.set_ylim(0.3, 1.0)
    _style(ax, "Cross-validated AUROC by cumulative feature block\n"
               "(train split, 5-fold stratified; increment in brackets)", "", "CV AUROC")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_blocks.png", dpi=DPI)
    plt.close(fig)


def figure_errors(out_dir, errors: pd.DataFrame) -> None:
    if errors.empty:
        return
    groups = list(errors.groupby("stratum", observed=True))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.9),
                             squeeze=False)
    for ax, (name, frame) in zip(axes[0], groups):
        ax.bar(range(len(frame)), frame["error_rate"], color="#DD8452", width=0.6)
        ax.set_xticks(range(len(frame)))
        ax.set_xticklabels([f"{v}\n(n={n})" for v, n in
                            zip(frame["value"], frame["n"])], fontsize=7, rotation=20)
        ax.set_ylim(0, max(0.6, float(frame["error_rate"].max()) * 1.25))
        _style(ax, f"by {name}", "", "error rate")
    fig.suptitle("Held-out error rate by stratum", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_errors.png", dpi=DPI)
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
    # Every successfully generated trace has a correctness label, including the
    # truncated ones - truncation is itself a plausible predictor of being
    # wrong, so excluding them would remove signal rather than noise.
    df = table.copy()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "eval"]
    log.info("n=%d (train %d, eval %d); prevalence train %.3f eval %.3f",
             len(df), len(train), len(test),
             train[TARGET].mean(), test[TARGET].mean())
    too_small = len(train) < 30 or len(test) < 20
    if train[TARGET].nunique() < 2 or (too_small and not args.allow_small):
        log.error("sample too small or single-class: train=%d eval=%d classes=%d "
                  "(--allow-small overrides, for smoke testing only)",
                  len(train), len(test), train[TARGET].nunique())
        return 1

    columns = BLOCKS["C_convergence"]
    X_train, y_train = train[columns], train[TARGET].to_numpy()
    X_test, y_test = test[columns], test[TARGET].to_numpy()

    log.info("fitting models")
    logistic, logistic_info = tune(logistic_pipeline(columns, cfg.seed), LOGISTIC_GRID,
                                   X_train, y_train, cfg.seed, log)
    gbm, gbm_info = tune(gbm_pipeline(columns, cfg.seed), GBM_GRID,
                         X_train, y_train, cfg.seed, log)

    probs = {
        "logistic": logistic.predict_proba(X_test)[:, 1],
        "gbm": gbm.predict_proba(X_test)[:, 1],
    }
    train_probs = {
        "logistic": logistic.predict_proba(X_train)[:, 1],
        "gbm": gbm.predict_proba(X_train)[:, 1],
    }
    thresholds = {k: choose_threshold(y_train, v) for k, v in train_probs.items()}

    boot = cfg.analysis.bootstrap_iterations
    rows = [{
        "model": "majority",
        "auroc": 0.5, "auroc_low": np.nan, "auroc_high": np.nan,
        "auprc": float(np.mean(y_test)), "auprc_low": np.nan, "auprc_high": np.nan,
        "balanced_accuracy": 0.5, "f1": float(f1_score(
            y_test, np.full(len(y_test), int(round(y_train.mean()))), zero_division=0)),
        "brier": float(np.mean((y_test - y_train.mean()) ** 2)),
        "accuracy": float(np.mean(y_test == int(round(y_train.mean())))),
        "threshold": np.nan,
    }]
    for name, prob in probs.items():
        rows.append({"model": name,
                     **classification_scores(y_test, prob,
                                             thresholds[name]["threshold"], boot, cfg.seed),
                     "threshold": thresholds[name]["threshold"]})
    scores = pd.DataFrame(rows)
    write_table(scores, out_dir, "model_scores")
    log.info("held-out scores:\n%s", markdown(scores))

    comparisons = {
        "gbm_minus_logistic_auroc": paired_metric_difference(
            y_test, probs["gbm"], probs["logistic"], roc_auc_score,
            n_iterations=boot, seed=cfg.seed),
    }
    # AUROC against chance: a one-sample interval on the model's own AUROC that
    # excludes 0.5 is the claim, and the bootstrap interval already gives it.
    auroc_ci = {name: bootstrap_metric(y_test, prob, roc_auc_score,
                                       n_iterations=boot, seed=cfg.seed)
                for name, prob in probs.items()}
    gbm_justified = bool(comparisons["gbm_minus_logistic_auroc"]["estimate"] > 0
                         and comparisons["gbm_minus_logistic_auroc"]["excludes_zero"])
    reported = "gbm" if gbm_justified else "logistic"
    log.info("gbm - logistic AUROC = %+.4f [%+.4f, %+.4f] -> reported model: %s",
             comparisons["gbm_minus_logistic_auroc"]["estimate"],
             comparisons["gbm_minus_logistic_auroc"]["low"],
             comparisons["gbm_minus_logistic_auroc"]["high"], reported)

    # ---- nested blocks --------------------------------------------------
    def block_score(cols: list[str]) -> dict[str, float]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        model = logistic_pipeline(cols, cfg.seed).set_params(
            model__C=logistic_info["best_params"]["model__C"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auroc = cross_val_score(model, train[cols], y_train, cv=cv, scoring="roc_auc")
            ap = cross_val_score(model, train[cols], y_train, cv=cv,
                                 scoring="average_precision")
        return {"primary": float(np.mean(auroc)), "cv_auroc_sd": float(np.std(auroc)),
                "cv_auprc": float(np.mean(ap))}

    blocks = nested_block_scores(block_score).rename(columns={"primary": "cv_auroc"})
    write_table(blocks, out_dir, "block_increments")
    log.info("block increments:\n%s", markdown(blocks))

    # ---- inference ------------------------------------------------------
    odds = odds_ratio_table(INFERENCE_FEATURES, train[INFERENCE_FEATURES],
                            y_train, cfg.seed, log)
    if not odds.empty:
        write_table(odds, out_dir, "odds_ratios")
        log.info("strongest coefficients:\n%s",
                 markdown(odds.head(8)[["feature", "odds_ratio", "or_ci_low",
                                        "or_ci_high", "p_value", "q_value"]]))

    # ---- feature analysis ----------------------------------------------
    best = {"logistic": logistic, "gbm": gbm}[reported]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(best, X_test, y_test, n_repeats=30,
                                      random_state=cfg.seed, scoring="roc_auc")
    imp = pd.DataFrame({"feature": columns,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std}) \
        .sort_values("importance_mean", ascending=False).reset_index(drop=True)
    write_table(imp, out_dir, "permutation_importance")
    log.info("top features (%s):\n%s", reported, markdown(imp.head(8)))

    # ---- error analysis -------------------------------------------------
    errors = error_analysis(test, y_test, probs[reported],
                            thresholds[reported]["threshold"])
    write_table(errors, out_dir, "error_analysis")
    log.info("error analysis:\n%s", markdown(errors))

    cm = confusion_matrix(y_test, (probs[reported] >= thresholds[reported]["threshold"])
                          .astype(int), labels=[0, 1])

    # ---- label-shuffle null (F6 analogue) ------------------------------
    def fit_and_score(y_shuffled: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = logistic_pipeline(columns, cfg.seed).set_params(
                model__C=logistic_info["best_params"]["model__C"])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
            return float(np.mean(cross_val_score(model, X_train, y_shuffled,
                                                 cv=cv, scoring="roc_auc")))

    null = permutation_null(fit_and_score, y_train,
                            n_permutations=args.permutations, seed=cfg.seed)
    log.info("label-shuffle null: observed CV AUROC=%.3f vs null %.3f±%.3f (p=%.4f)",
             null["observed"], null["null_mean"], null["null_sd"], null["p_value"])

    # ---- figures --------------------------------------------------------
    figure_curves(out_dir, y_test, probs, scores)
    figure_confusion(out_dir, y_test, probs[reported],
                     thresholds[reported]["threshold"], reported)
    figure_calibration(out_dir, y_test, probs)
    figure_importance(out_dir, imp, f"Permutation importance - {reported} (eval split)")
    figure_blocks(out_dir, blocks)
    figure_errors(out_dir, errors)

    summary = {
        "target": TARGET,
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_eval": int(len(test)),
        "n_features": len(columns),
        "feature_blocks": {k: len(v) for k, v in FEATURE_BLOCKS.items()},
        "prevalence": {"train": float(y_train.mean()), "eval": float(y_test.mean())},
        "tuning": {"logistic": logistic_info, "gbm": gbm_info},
        "thresholds": thresholds,
        "scores": scores.to_dict(orient="records"),
        "auroc_intervals": auroc_ci,
        "comparisons": comparisons,
        "reported_model": reported,
        "gbm_justified": gbm_justified,
        "block_increments": blocks.to_dict(orient="records"),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "label_shuffle_null": null,
        "top_features": imp.head(10).to_dict(orient="records"),
        "error_analysis": errors.to_dict(orient="records"),
        "inference_features": INFERENCE_FEATURES,
        "odds_ratios": (odds.head(15).to_dict(orient="records")
                        if not odds.empty else []),
    }
    write_json(out_dir / "summary.json", summary)
    manifest.finish_stage(STAGE, status="complete", output=str(out_dir),
                          metrics={"auroc_reported": float(scores.loc[
                              scores["model"] == reported, "auroc"].iloc[0]),
                              "n_eval": int(len(test))})
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
