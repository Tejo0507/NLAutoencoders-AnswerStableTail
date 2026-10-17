# Supervised analyses - results

Generated from `D:/NLAutoencoders-AnswerStableTail/runs/main` by `scripts/write_supervised_report.py`. Every number below was computed and written by the regression or classification stage; this script recomputes nothing. Methodology, target justification and leakage controls are in [`docs/SUPERVISED_ANALYSES.md`](docs/SUPERVISED_ANALYSES.md).

- **Run:** `main`, config hash `ea82fa845911b70f`
- **Regression:** `D:/NLAutoencoders-AnswerStableTail/runs/main/regression_tail`
- **Classification:** `D:/NLAutoencoders-AnswerStableTail/runs/main/classify_correct`

---

## 1. Regression - how much of the Answer-Stable Tail is predictable from cheap signals?

**Target:** `tail_fraction`, the proportion of a trace's chunks lying inside the Answer-Stable Tail. Continuous on `[0, 1]`; mean 0.378, SD 0.160 in the primary cohort.

**Sample.** 250 traces carried a target. The primary cohort (a measured tail, not truncated) holds 240, split 97 train / 143 held out by the run's pre-existing hash assignment. 35 features in three nested blocks; no activations, no gold answer beyond its magnitude.

### 1.1 Held-out performance

| model | r2 | mae | rmse | mae_low | mae_high | r2_low | r2_high |
|---|---|---|---|---|---|---|---|
| mean | -0.0210 | 0.1316 | 0.1647 | 0.1158 | 0.1482 | -0.1151 | -0.0001 |
| ridge | 0.3329 | 0.1018 | 0.1331 | 0.0881 | 0.1161 | 0.1884 | 0.4369 |
| gbm | 0.3188 | 0.1009 | 0.1345 | 0.0868 | 0.1162 | 0.1451 | 0.4403 |

The reported model is **ridge**. The gradient-boosted model was not justified: its MAE differs from ridge's by -0.001 [-0.012, 0.011] (paired bootstrap), which includes zero.

Ridge against the training-mean predictor: MAE difference -0.030 [-0.042, -0.018]. A negative estimate whose interval excludes zero means the features beat knowing nothing.

**Reading of R² = 0.333:** moderate - a real but partial account; most of the variation is unexplained by surface form.

### 1.2 What each tier of information is worth

| block | n_features | cv_r2 | cv_r2_sd | cv_mae | increment |
|---|---|---|---|---|---|
| A_problem | 11 | -0.9238 | 1.8259 | 0.1364 |  |
| B_trace | 31 | -0.3898 | 1.1428 | 0.1167 | 0.5340 |
| C_convergence | 35 | -0.3584 | 1.1159 | 0.1160 | 0.0314 |

Cross-validated inside the training split only. `A_problem` is what is knowable before generation; `B_trace` adds the trace's surface form (this is falsification test F1's length predictor and F2's difficulty stratifiers); `C_convergence` adds Liu & Wang's answer-agreement signal, the cheapest stopping rule in the study.

### 1.3 Linear model: coefficients and diagnostics

OLS on the pre-declared interpretable subset (10 features, 12 design columns, 8.1 observations per parameter). R² 0.490, adjusted 0.424, F = 7.43 (p = 8.92e-09).

| feature | coefficient | std_error | ci_low | ci_high | p_value | q_value | significant_bh |
|---|---|---|---|---|---|---|---|
| (intercept) | 0.3924 | 0.0119 | 0.3687 | 0.4161 | 0.0000 |  | False |
| n_tokens | -0.0503 | 0.0253 | -0.1007 | 0.0001 | 0.0504 | 0.2771 | False |
| n_chunks | -0.0676 | 0.0179 | -0.1032 | -0.0321 | 0.0003 | 0.0032 | True |
| math_level | -0.0216 | 0.0157 | -0.0529 | 0.0097 | 0.1731 | 0.4335 | False |
| gold_abs_log10 | -0.0019 | 0.0133 | -0.0284 | 0.0246 | 0.8880 | 0.9768 | False |
| convergence_position | 0.0062 | 0.0134 | -0.0205 | 0.0329 | 0.6456 | 0.7891 | False |
| first_answer_mention | -0.0155 | 0.0141 | -0.0436 | 0.0126 | 0.2758 | 0.4335 | False |
| cue_verify_per_chunk | 0.0169 | 0.0131 | -0.0091 | 0.0429 | 0.2003 | 0.4335 | False |
| equals_per_chunk | -0.0004 | 0.0194 | -0.0390 | 0.0382 | 0.9843 | 0.9843 | False |
| missingindicator_math_level | -0.0250 | 0.0207 | -0.0662 | 0.0162 | 0.2309 | 0.4335 | False |
| missingindicator_gold_abs_log10 | -0.0171 | 0.0153 | -0.0476 | 0.0133 | 0.2666 | 0.4335 | False |
| missingindicator_convergence_position | 0.0085 | 0.0153 | -0.0219 | 0.0389 | 0.5789 | 0.7891 | False |

Coefficients are on standardised features, so each is the change in tail fraction per SD of that predictor. `significant_bh` is Benjamini-Hochberg at q = 0.05 across the coefficient family.

| Diagnostic | Value | Reading |
|---|---|---|
| Breusch-Pagan p | 0.484 | homoscedastic |
| Jarque-Bera p | 0.698 | residuals consistent with normal |
| Condition number | 4.6 | no serious collinearity |
| Max VIF | 4.5 | acceptable |
| Durbin-Watson | 2.30 | reported for completeness; rows are independent problems, so serial correlation is not expected |

See `fig_diagnostics.png` for residuals vs fitted, the normal Q-Q plot, and residuals against trace length - the last being F1 made visual.

### 1.4 Feature analysis

| feature | importance_mean | importance_std |
|---|---|---|
| cue_verify_per_chunk | 0.0008 | 0.0013 |
| parsed_boundary_fraction | 0.0005 | 0.0003 |
| chunk_chars_mean | 0.0003 | 0.0002 |
| has_answer_cue | 0.0002 | 0.0001 |
| answer_mention_count | 0.0001 | 0.0002 |
| tokens_per_chunk_mean | 0.0001 | 0.0002 |
| has_convergence | 0.0001 | 0.0001 |
| digit_fraction | 0.0000 | 0.0001 |
| cue_conclude_per_chunk | 0.0000 | 0.0001 |
| cue_restate_per_chunk | 0.0000 | 0.0000 |

Permutation importance on the held-out split, as the increase in MAE when a feature is shuffled. Computed on held-out data rather than training data, where it would measure what the model memorised.

### 1.5 Leakage check and sensitivity

**Label shuffle (F6 analogue).** Refitting on permuted targets 200 times gives a null CV R² of -0.277 (SD 0.211, max 0.020) against the observed -0.358; p = 0.7313. **Not separated from the null at α = 0.05.** This check is run on the cross-validated *training-split* score, and with 97 training rows, about 19 per CV fold that score is unstable from one fold assignment to the next, so the check has little power: it shows no sign of leakage, and it neither corroborates nor contradicts the held-out estimate above, which rests on its own bootstrap interval over the larger held-out split.

| cohort | n_train | n_eval | mean_target | r2 | mae |
|---|---|---|---|---|---|
| primary | 97 | 143 | 0.378 | 0.333 | 0.102 |
| with_no_stable_point | 98 | 144 | 0.375 | 0.313 | 0.104 |
| with_truncated | 97 | 143 | 0.378 | 0.333 | 0.102 |

The two sensitivity cohorts put back the exclusions the primary cohort makes: traces where no boundary qualified (`tail_fraction = 0` by convention rather than measurement) and traces that hit the token cap.

---

## 2. Classification - is final-answer correctness readable off the trace surface?

**Target:** `final_correct`, whether the trace's final answer is equivalent to gold under `math_verify`. Binary, adjudicated by the same external verifier the rest of the pipeline uses.

**Sample.** 250 traces, 101 train / 149 held out on the same pre-existing split. Prevalence 0.941 train, 0.839 held out - high enough that accuracy alone would be uninformative, which is why AUROC, AUPRC and balanced accuracy are reported together.

### 2.1 Held-out performance

| model | auroc | auroc_low | auroc_high | auprc | auprc_low | auprc_high | balanced_accuracy | f1 | brier | accuracy | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.5000 |  |  | 0.8389 |  |  | 0.5000 | 0.9124 | 0.1455 | 0.8389 |  |
| logistic | 0.8177 | 0.7269 | 0.8962 | 0.9596 | 0.9313 | 0.9822 | 0.6338 | 0.9242 | 0.1372 | 0.8658 | 0.4803 |
| gbm | 0.6960 | 0.5747 | 0.8078 | 0.9021 | 0.8342 | 0.9637 | 0.6275 | 0.8800 | 0.1664 | 0.7987 | 0.9765 |

The reported model is **logistic**. Gradient boosting was not justified: its AUROC differs from logistic regression's by -0.122 [-0.233, -0.013], which excludes zero.

**AUROC 0.818 [0.727, 0.896].** The interval excludes 0.5, so the signal is distinguishable from chance. Reading: strong - correctness is substantially readable off the trace surface.

### 2.2 What each tier of information is worth

| block | n_features | cv_auroc | cv_auroc_sd | cv_auprc | increment |
|---|---|---|---|---|---|
| A_problem | 11 | 0.6368 | 0.3044 | 0.9658 |  |
| B_trace | 31 | 0.7684 | 0.1901 | 0.9818 | 0.1316 |
| C_convergence | 35 | 0.7474 | 0.1924 | 0.9789 | -0.0211 |

### 2.3 Confusion and error analysis

At the threshold 0.480, chosen by maximising Youden's J on the **training** split and then frozen:

| | predicted incorrect | predicted correct |
|---|---|---|
| **actually incorrect** | 7 (true negative) | 17 (**false correct**) |
| **actually correct** | 3 (false incorrect) | 122 (true positive) |

The two error types are not interchangeable. A **false correct** is the costly one: a stopping rule reading this signal would endorse an answer that is wrong. There are 17 of them against 3 false incorrects, which cost only tokens.

| stratum | value | n | prevalence | error_rate | false_correct | false_incorrect | mean_p |
|---|---|---|---|---|---|---|---|
| dataset | gsm8k | 81 | 0.852 | 0.160 | 12 | 1 | 0.665 |
| dataset | math | 68 | 0.824 | 0.103 | 5 | 2 | 0.655 |
| trace length | short | 50 | 0.960 | 0.040 | 2 | 0 | 0.727 |
| trace length | medium | 49 | 0.857 | 0.163 | 7 | 1 | 0.659 |
| trace length | long | 50 | 0.700 | 0.200 | 8 | 2 | 0.597 |
| difficulty | MATH level 1 | 5 | 1.000 | 0.000 | 0 | 0 | 0.728 |
| difficulty | MATH level 2 | 14 | 1.000 | 0.000 | 0 | 0 | 0.738 |
| difficulty | MATH level 3 | 16 | 1.000 | 0.000 | 0 | 0 | 0.752 |
| difficulty | MATH level 4 | 16 | 0.812 | 0.250 | 2 | 2 | 0.619 |
| difficulty | MATH level 5 | 17 | 0.471 | 0.176 | 3 | 0 | 0.510 |
| difficulty | gsm8k (no level) | 81 | 0.852 | 0.160 | 12 | 1 | 0.665 |
| truncated | 0 | 144 | 0.868 | 0.139 | 17 | 3 | 0.677 |
| truncated | 1 | 5 | 0.000 | 0.000 | 0 | 0 | 0.196 |

Error rate per stratum, so a pooled AUROC cannot hide a subgroup the model fails on. See `fig_errors.png` and `fig_calibration.png` - calibration matters here because a stopping rule would threshold the probability, not the ranking, and a well-ranked but badly calibrated score is unusable for that.

### 2.4 Feature analysis

| feature | importance_mean | importance_std |
|---|---|---|
| q_words | 0.0481 | 0.0178 |
| truncated | 0.0409 | 0.0126 |
| n_tokens | 0.0205 | 0.0091 |
| q_chars | 0.0149 | 0.0088 |
| cue_verify_per_chunk | 0.0087 | 0.0176 |
| math_level | 0.0077 | 0.0050 |
| q_number_density | 0.0076 | 0.0110 |
| prompt_tokens | 0.0046 | 0.0029 |
| chunk_chars_mean | 0.0044 | 0.0040 |
| gold_abs_log10 | 0.0044 | 0.0100 |

Permutation importance as the drop in held-out AUROC.

Odds ratios from an **L2-penalised** logistic fit on the pre-declared interpretable subset. The unpenalised fit did not give usable standard errors, so these are shrunk point estimates with **no intervals and no tests**: read them as directions, not effects. Features are standardised, so each is the multiplicative change in odds of being correct per SD:

| feature | odds_ratio | or_ci_low | or_ci_high | p_value |
|---|---|---|---|---|
| (intercept) | 21.892 |  |  |  |
| n_tokens | 1.000 |  |  |  |
| n_chunks | 1.000 |  |  |  |
| math_level | 1.000 |  |  |  |
| gold_abs_log10 | 0.808 |  |  |  |
| convergence_position | 1.000 |  |  |  |
| first_answer_mention | 1.000 |  |  |  |
| cue_verify_per_chunk | 1.000 |  |  |  |
| equals_per_chunk | 1.000 |  |  |  |
| truncated | 0.299 |  |  |  |
| missingindicator_math_level | 0.668 |  |  |  |
| missingindicator_gold_abs_log10 | 1.000 |  |  |  |

### 2.5 Leakage check

**Label shuffle (F6 analogue).** 200 refits on permuted labels give a null CV AUROC of 0.509 (SD 0.172, max 0.911) against the observed 0.747; p = 0.0796. **Not separated from the null at α = 0.05.** This check is run on the cross-validated *training-split* score, and with 101 training rows, about 20 per CV fold, and only 6 minority-class examples in the whole training split (1.2 per fold) that score is unstable from one fold assignment to the next, so the check has little power: it shows no sign of leakage, and it neither corroborates nor contradicts the held-out estimate above, which rests on its own bootstrap interval over the larger held-out split.

---

## 3. Conclusions

**On the tail.** Cheap, activation-free features explain R² = 0.333 of held-out variation in how much of a trace is post-answer redundancy (moderate). 
Adding the answer-convergence baseline on top of the trace's surface form moves cross-validated R² by +0.031. 
That figure is one estimate on one pre-existing split; cross-validation inside the 97-row training split does not corroborate it (CV R² -0.358, label-shuffle p = 0.73), because at that size the CV score is too unstable to say anything either way (§1.5). 
This is the floor for falsification tests F1 and F2: whatever the verbalised readout eventually shows about the tail has to be shown as an increment over this number, not against zero.

**On correctness.** The same cheap features reach AUROC 0.818 [0.727, 0.896] on held-out problems (strong). This is the floor for the layer-20 hidden-state probe: a probe AUROC at or below this number is not evidence that correctness is decodable from activations, because it is already readable from the text. 
The ranking claim rests on the held-out interval alone: cross-validation inside the training split is not separated from its label-shuffle null (p = 0.08) with only 6 minority-class training examples, and the same 6 fix the decision threshold, so threshold-based metrics (balanced accuracy, the confusion matrix) and calibration are less secure than the AUROC.

**On the project's premise.** Both analyses are constructed so that a *strong* cheap-tier result is bad news for the study's central claim and a *weak* one is good news. They were run and reported either way, which is the stance `PROJECT_PLAN.md` §6 commits to.

## 4. Limitations

1. **Sample size.** The achieved N is set by how much trace generation finished on a 6.44 GB GPU, not by what would have been comfortable. Where an interval is wide, that is the result, and no conclusion above rests on a point estimate whose interval crosses the threshold it is being compared to.
2. **Predictive, not causal.** These are observational models over generated traces. That trace length predicts tail fraction does not mean lengthening a trace lengthens its tail. Causal claims belong to stage `09` and its matched-random-direction controls.
3. **Nothing here is about the autoencoder.** Neither model reads an activation. They set the floor the probe and the verbalised readout must clear; they say nothing about whether those clear it.
4. **Semantic entropy is missing from the cheap tier.** The corpus was generated with `--no-entropy` to buy sample size. A cheap uncertainty signal is therefore absent, so the floor reported here is, if anything, slightly low.
5. **Two problem-metadata features are not deployment-available.** `math_level` and `gold_abs_log10` are included because F2 names them as the difficulty stratifiers. Their contribution is visible as the `A_problem` block score and should be discounted when reading these numbers as a claim about a live stopping rule.
6. **One model, one decoding configuration.** Qwen2.5-7B-Instruct at 4-bit NF4, greedy, 512-token cap, GSM8K and MATH algebra / counting. The quantisation deviation documented in `PROJECT_PLAN.md` §2 applies to the traces these features are computed from.
7. **The tail definition is a construct, not a ground truth.** `tail_fraction` depends on `k_continuations`, the continuation temperature and the chunker. Falsification test F5 sweeps those; this analysis inherits whatever that sweep concludes.
