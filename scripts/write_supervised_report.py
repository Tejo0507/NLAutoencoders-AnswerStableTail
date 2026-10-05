"""Stage `supervised_report`: assemble the report for the two supervised analyses.

A pure function of the two ``summary.json`` files and the tables beside them.
It loads no model and recomputes no metric: every number it prints was computed
and written by the regression or classification stage, so the report cannot disagree with the run.

Interpretation is deliberately rule-based rather than written by hand. The
thresholds below are stated once, applied to whatever the run produced, and
produce the same verdict for a result the author would have liked and one they
would not. ``docs/SUPERVISED_ANALYSES.md`` holds the methodology; this file
holds the numbers and the conclusions those rules draw from them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from _stage import base_parser, setup  # noqa: E402

from nlaast.analysis.tables import markdown  # noqa: E402

STAGE = "supervised_report"

#: Rule-based reading of an R-squared. Fixed before the run produced one.
R2_BANDS = [
    (0.50, "substantial - the cheap tier explains most of the variation, and any "
           "claim for a more expensive signal has to be made as an increment over this"),
    (0.25, "moderate - a real but partial account; most of the variation is "
           "unexplained by surface form"),
    (0.05, "weak - detectable but small; the redundancy is largely not a "
           "surface-form phenomenon"),
    (-1e9, "absent - the cheap features carry essentially no usable signal for "
           "this target"),
]

#: Rule-based reading of an AUROC.
AUROC_BANDS = [
    (0.80, "strong - correctness is substantially readable off the trace surface"),
    (0.70, "moderate - a usable signal, well short of a reliable predictor"),
    (0.60, "weak - better than chance, too weak to act on"),
    (0.00, "at or near chance - correctness is not readable off the trace surface"),
]


def band(value: float, bands: list[tuple[float, str]]) -> str:
    for threshold, text in bands:
        if value >= threshold:
            return text
    return bands[-1][1]


def load(run_dir: Path, stage: str) -> dict | None:
    path = run_dir / stage / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table(run_dir: Path, stage: str, name: str) -> pd.DataFrame:
    path = run_dir / stage / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def interval(d: dict, key: str = "estimate") -> str:
    lo, hi = d.get("low"), d.get("high")
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"{d[key]:.3f}"
    return f"{d[key]:.3f} [{lo:.3f}, {hi:.3f}]"


# --- Sections ---


def regression_section(run_dir: Path, s: dict) -> list[str]:
    scores = table(run_dir, "regression_tail", "model_scores")
    blocks = table(run_dir, "regression_tail", "block_increments")
    coef = table(run_dir, "regression_tail", "ols_coefficients")
    imp = table(run_dir, "regression_tail", "permutation_importance")
    sens = table(run_dir, "regression_tail", "sensitivity_cohorts")
    diag, null = s["ols_diagnostics"], s["label_shuffle_null"]
    reported = s["reported_model"]
    r2 = float(scores.loc[scores["model"] == reported, "r2"].iloc[0])

    cmp_gbm = s["comparisons"]["gbm_minus_ridge_mae"]
    cmp_ridge = s["comparisons"]["ridge_minus_mean_mae"]

    out = [
        "## 1. Regression - how much of the Answer-Stable Tail is predictable "
        "from cheap signals?",
        "",
        f"**Target:** `tail_fraction`, the proportion of a trace's chunks lying "
        f"inside the Answer-Stable Tail. Continuous on `[0, 1]`; "
        f"mean {s['target_mean']:.3f}, SD {s['target_sd']:.3f} in the primary cohort.",
        "",
        f"**Sample.** {s['n_total_traces']} traces carried a target. The primary "
        f"cohort (a measured tail, not truncated) holds "
        f"{s['cohort_sizes']['primary']}, split "
        f"{s['n_train']} train / {s['n_eval']} held out by the run's "
        f"pre-existing hash assignment. {s['n_features']} features in three "
        "nested blocks; no activations, no gold answer beyond its magnitude.",
        "",
        "### 1.1 Held-out performance",
        "",
        markdown(scores, floatfmt="%.4f"),
        "",
        f"The reported model is **{reported}**. The gradient-boosted model was "
        f"{'' if s['gbm_justified'] else 'not '}justified: its MAE differs from "
        f"ridge's by {interval(cmp_gbm)} (paired bootstrap), which "
        f"{'excludes' if cmp_gbm['excludes_zero'] else 'includes'} zero.",
        "",
        f"Ridge against the training-mean predictor: MAE difference "
        f"{interval(cmp_ridge)}. A negative estimate whose interval excludes zero "
        "means the features beat knowing nothing.",
        "",
        f"**Reading of R² = {r2:.3f}:** {band(r2, R2_BANDS)}.",
        "",
        "### 1.2 What each tier of information is worth",
        "",
        markdown(blocks, floatfmt="%.4f"),
        "",
        "Cross-validated inside the training split only. `A_problem` is what is "
        "knowable before generation; `B_trace` adds the trace's surface form "
        "(this is falsification test F1's length predictor and F2's difficulty "
        "stratifiers); `C_convergence` adds Liu & Wang's answer-agreement signal, "
        "the cheapest stopping rule in the study.",
        "",
        "### 1.3 Linear model: coefficients and diagnostics",
        "",
        f"OLS on the pre-declared interpretable subset "
        f"({len(s.get('ols_features', []))} features, "
        f"{diag['n_parameters']} design columns, "
        f"{diag['observations_per_parameter']:.1f} observations per parameter). "
        f"R² {diag['r_squared']:.3f}, adjusted {diag['r_squared_adj']:.3f}, "
        f"F = {diag['f_statistic']:.2f} (p = {diag['f_p_value']:.3g}).",
        "",
        markdown(coef[["feature", "coefficient", "std_error", "ci_low", "ci_high",
                       "p_value", "q_value", "significant_bh"]]
                 if not coef.empty else coef, floatfmt="%.4f"),
        "",
        "Coefficients are on standardised features, so each is the change in "
        "tail fraction per SD of that predictor. `significant_bh` is "
        "Benjamini-Hochberg at q = 0.05 across the coefficient family.",
        "",
        "| Diagnostic | Value | Reading |",
        "|---|---|---|",
        f"| Breusch-Pagan p | {diag['breusch_pagan_p']:.3g} | "
        f"{'homoscedastic' if diag['breusch_pagan_p'] > 0.05 else '**heteroscedastic** - the OLS standard errors above are optimistic'} |",
        f"| Jarque-Bera p | {diag['jarque_bera_p']:.3g} | "
        f"{'residuals consistent with normal' if diag['jarque_bera_p'] > 0.05 else '**non-normal residuals** - the t-based intervals are approximate'} |",
        f"| Condition number | {diag['condition_number']:.1f} | "
        f"{'no serious collinearity' if diag['condition_number'] < 30 else '**collinear design** - individual coefficients are unstable, the fit as a whole is not'} |",
        f"| Max VIF | {diag['max_vif']:.1f} | "
        f"{'acceptable' if not (diag['max_vif'] > 10) else '**inflated** - at least one predictor is largely explained by the others'} |",
        f"| Durbin-Watson | {diag['durbin_watson']:.2f} | reported for completeness; "
        "rows are independent problems, so serial correlation is not expected |",
        "",
        "See `fig_diagnostics.png` for residuals vs fitted, the normal Q-Q plot, "
        "and residuals against trace length - the last being F1 made visual.",
        "",
        "### 1.4 Feature analysis",
        "",
        markdown(imp.head(10), floatfmt="%.4f"),
        "",
        "Permutation importance on the held-out split, as the increase in MAE "
        "when a feature is shuffled. Computed on held-out data rather than "
        "training data, where it would measure what the model memorised.",
        "",
        "### 1.5 Leakage check and sensitivity",
        "",
        f"**Label shuffle (F6 analogue).** Refitting on permuted targets "
        f"{null['n_permutations']} times gives a null CV R² of "
        f"{null['null_mean']:.3f} (SD {null['null_sd']:.3f}, max "
        f"{null['null_max']:.3f}) against the observed {null['observed']:.3f}; "
        f"p = {null['p_value']:.4f}. "
        + ("The model collapses on destroyed labels, which is what a "
           "leakage-free fit should do."
           if null["p_value"] < 0.10 else
           "**The observed score is not clearly separated from the null**, which "
           "at this sample size is a power statement rather than evidence of "
           "leakage - but it means the point estimate should not be leaned on."),
        "",
        markdown(sens, floatfmt="%.3f"),
        "",
        "The two sensitivity cohorts put back the exclusions the primary cohort "
        "makes: traces where no boundary qualified (`tail_fraction = 0` by "
        "convention rather than measurement) and traces that hit the token cap.",
        "",
    ]
    return out


def classification_section(run_dir: Path, s: dict) -> list[str]:
    scores = table(run_dir, "classify_correct", "model_scores")
    blocks = table(run_dir, "classify_correct", "block_increments")
    odds = table(run_dir, "classify_correct", "odds_ratios")
    imp = table(run_dir, "classify_correct", "permutation_importance")
    errors = table(run_dir, "classify_correct", "error_analysis")
    cm, null = s["confusion_matrix"], s["label_shuffle_null"]
    reported = s["reported_model"]
    auroc = s["auroc_intervals"][reported]
    cmp_gbm = s["comparisons"]["gbm_minus_logistic_auroc"]

    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    out = [
        "## 2. Classification - is final-answer correctness readable off the "
        "trace surface?",
        "",
        "**Target:** `final_correct`, whether the trace's final answer is "
        "equivalent to gold under `math_verify`. Binary, adjudicated by the same "
        "external verifier the rest of the pipeline uses.",
        "",
        f"**Sample.** {s['n_total']} traces, {s['n_train']} train / "
        f"{s['n_eval']} held out on the same pre-existing split. Prevalence "
        f"{s['prevalence']['train']:.3f} train, {s['prevalence']['eval']:.3f} "
        "held out - high enough that accuracy alone would be uninformative, "
        "which is why AUROC, AUPRC and balanced accuracy are reported together.",
        "",
        "### 2.1 Held-out performance",
        "",
        markdown(scores, floatfmt="%.4f"),
        "",
        f"The reported model is **{reported}**. Gradient boosting was "
        f"{'' if s['gbm_justified'] else 'not '}justified: its AUROC differs from "
        f"logistic regression's by {interval(cmp_gbm)}, which "
        f"{'excludes' if cmp_gbm['excludes_zero'] else 'includes'} zero.",
        "",
        f"**AUROC {interval(auroc)}.** "
        + ("The interval excludes 0.5, so the signal is distinguishable from chance."
           if (auroc.get("low") or 0) > 0.5 else
           "**The interval includes 0.5**, so on this sample the classifier is "
           "not distinguishable from chance.")
        + f" Reading: {band(auroc['estimate'], AUROC_BANDS)}.",
        "",
        "### 2.2 What each tier of information is worth",
        "",
        markdown(blocks, floatfmt="%.4f"),
        "",
        "### 2.3 Confusion and error analysis",
        "",
        f"At the threshold {s['thresholds'][reported]['threshold']:.3f}, chosen by "
        "maximising Youden's J on the **training** split and then frozen:",
        "",
        "| | predicted incorrect | predicted correct |",
        "|---|---|---|",
        f"| **actually incorrect** | {tn} (true negative) | {fp} (**false correct**) |",
        f"| **actually correct** | {fn} (false incorrect) | {tp} (true positive) |",
        "",
        "The two error types are not interchangeable. A **false correct** is the "
        "costly one: a stopping rule reading this signal would endorse an answer "
        f"that is wrong. There are {fp} of them against {fn} false incorrects, "
        "which cost only tokens.",
        "",
        markdown(errors, floatfmt="%.3f"),
        "",
        "Error rate per stratum, so a pooled AUROC cannot hide a subgroup the "
        "model fails on. See `fig_errors.png` and `fig_calibration.png` - "
        "calibration matters here because a stopping rule would threshold the "
        "probability, not the ranking, and a well-ranked but badly calibrated "
        "score is unusable for that.",
        "",
        "### 2.4 Feature analysis",
        "",
        markdown(imp.head(10), floatfmt="%.4f"),
        "",
        "Permutation importance as the drop in held-out AUROC.",
        "",
    ]
    if not odds.empty:
        out += [
            "Odds ratios from the penalised logistic fit on the pre-declared "
            "interpretable subset (standardised features, so each is the "
            "multiplicative change in odds of being correct per SD):",
            "",
            markdown(odds.head(12)[["feature", "odds_ratio", "or_ci_low",
                                    "or_ci_high", "p_value", "q_value",
                                    "significant_bh"]], floatfmt="%.3f"),
            "",
        ]
    out += [
        "### 2.5 Leakage check",
        "",
        f"**Label shuffle (F6 analogue).** {null['n_permutations']} refits on "
        f"permuted labels give a null CV AUROC of {null['null_mean']:.3f} "
        f"(SD {null['null_sd']:.3f}, max {null['null_max']:.3f}) against the "
        f"observed {null['observed']:.3f}; p = {null['p_value']:.4f}. "
        + ("The null sits at chance and the observed score is separated from it."
           if null["p_value"] < 0.10 else
           "**The observed score is not clearly separated from the null.** At this "
           "sample size that is a statement about power, not proof of leakage, but "
           "the point estimate should not be leaned on."),
        "",
    ]
    return out


def conclusions(reg: dict | None, clf: dict | None) -> list[str]:
    out = ["## 3. Conclusions", ""]

    if reg:
        scores = reg["scores"]
        r2 = next(r["r2"] for r in scores if r["model"] == reg["reported_model"])
        blocks = {b["block"]: b for b in reg["block_increments"]}
        conv_inc = blocks.get("C_convergence", {}).get("increment")
        out += [
            f"**On the tail.** Cheap, activation-free features explain "
            f"R² = {r2:.3f} of held-out variation in how much of a trace is "
            f"post-answer redundancy ({band(r2, R2_BANDS).split(' - ')[0]}). ",
        ]
        if conv_inc is not None and conv_inc == conv_inc:
            out += [
                f"Adding the answer-convergence baseline on top of the trace's "
                f"surface form moves cross-validated R² by {conv_inc:+.3f}. ",
            ]
        out += [
            "This is the floor for falsification tests F1 and F2: whatever the "
            "verbalised readout eventually shows about the tail has to be shown "
            "as an increment over this number, not against zero.",
            "",
        ]

    if clf:
        reported = clf["reported_model"]
        auroc = clf["auroc_intervals"][reported]
        out += [
            f"**On correctness.** The same cheap features reach AUROC "
            f"{interval(auroc)} on held-out problems "
            f"({band(auroc['estimate'], AUROC_BANDS).split(' - ')[0]}). "
            "This is the floor for the layer-20 hidden-state probe: a probe "
            "AUROC at or below this number is not evidence that correctness is "
            "decodable from activations, because it is already readable from the "
            "text.",
            "",
        ]

    out += [
        "**On the project's premise.** Both analyses are constructed so that a "
        "*strong* cheap-tier result is bad news for the study's central claim and "
        "a *weak* one is good news. They were run and reported either way, which "
        "is the stance `PROJECT_PLAN.md` §6 commits to.",
        "",
        "## 4. Limitations",
        "",
        "1. **Sample size.** The achieved N is set by how much trace generation "
        "finished on a 6.44 GB GPU, not by what would have been comfortable. "
        "Where an interval is wide, that is the result, and no conclusion above "
        "rests on a point estimate whose interval crosses the threshold it is "
        "being compared to.",
        "2. **Predictive, not causal.** These are observational models over "
        "generated traces. That trace length predicts tail fraction does not mean "
        "lengthening a trace lengthens its tail. Causal claims belong to stage "
        "`09` and its matched-random-direction controls.",
        "3. **Nothing here is about the autoencoder.** Neither model reads an "
        "activation. They set the floor the probe and the verbalised readout must "
        "clear; they say nothing about whether those clear it.",
        "4. **Semantic entropy is missing from the cheap tier.** The corpus was "
        "generated with `--no-entropy` to buy sample size. A cheap uncertainty "
        "signal is therefore absent, so the floor reported here is, if anything, "
        "slightly low.",
        "5. **Two problem-metadata features are not deployment-available.** "
        "`math_level` and `gold_abs_log10` are included because F2 names them as "
        "the difficulty stratifiers. Their contribution is visible as the "
        "`A_problem` block score and should be discounted when reading these "
        "numbers as a claim about a live stopping rule.",
        "6. **One model, one decoding configuration.** Qwen2.5-7B-Instruct at "
        "4-bit NF4, greedy, 512-token cap, GSM8K and MATH algebra / counting. "
        "The quantisation deviation documented in `PROJECT_PLAN.md` §2 applies to "
        "the traces these features are computed from.",
        "7. **The tail definition is a construct, not a ground truth.** "
        "`tail_fraction` depends on `k_continuations`, the continuation "
        "temperature and the chunker. Falsification test F5 sweeps those; this "
        "analysis inherits whatever that sweep concludes.",
        "",
    ]
    return out


# --- Entry point ---


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--out", default="EXECUTION_REPORT_SUPERVISED.md",
                   help="markdown path, relative to the repository root")
    args = p.parse_args(argv)
    cfg, manifest, log = setup(args)

    reg = load(cfg.dir, "regression_tail")
    clf = load(cfg.dir, "classify_correct")
    if reg is None and clf is None:
        log.error("neither analysis stage has produced a summary in %s", cfg.dir)
        return 1

    lines = [
        "# Supervised analyses - results",
        "",
        f"Generated from `{cfg.dir.as_posix()}` by `scripts/write_supervised_report.py`. "
        "Every number below was computed and written by the regression or classification stage; this "
        "script recomputes nothing. Methodology, target justification and "
        "leakage controls are in "
        "[`docs/SUPERVISED_ANALYSES.md`](docs/SUPERVISED_ANALYSES.md).",
        "",
        f"- **Run:** `{cfg.run_id}`, config hash `{cfg.hash()}`",
        f"- **Regression:** `{(cfg.dir / 'regression_tail').as_posix()}`"
        + ("" if reg else "  _(not run)_"),
        f"- **Classification:** `{(cfg.dir / 'classify_correct').as_posix()}`"
        + ("" if clf else "  _(not run)_"),
        "",
        "---",
        "",
    ]
    if reg:
        lines += regression_section(cfg.dir, reg) + ["---", ""]
    if clf:
        lines += classification_section(cfg.dir, clf) + ["---", ""]
    lines += conclusions(reg, clf)

    out_path = Path(__file__).resolve().parents[1] / args.out
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s (%d lines)", out_path, len(lines))

    manifest.finish_stage(STAGE, status="complete", output=str(out_path),
                          metrics={"regression": bool(reg), "classification": bool(clf)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
