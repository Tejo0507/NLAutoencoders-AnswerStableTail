# Two supervised analyses over the trace corpus

**Stages:** `scripts/predict_tail_fraction.py`, `scripts/predict_correctness.py`
**Library:** `src/nlaast/supervised/`
**Run:** `runs/mlcorpus/` (config `configs/mlcorpus.yaml`)
**Output:** `runs/mlcorpus/regression_tail/`, `runs/mlcorpus/classify_correct/`

This document states what the two analyses are, why their targets are the
right ones for *this* project rather than two convenient columns, how leakage
is prevented, and what the results do and do not license.

**It contains no results.** `scripts/write_supervised_report.py` assembles
`EXECUTION_REPORT_SUPERVISED.md` from whatever a run produced; the corpus is
not yet large enough for that report to say anything, so it has not been
generated. See [`STATUS.md`](STATUS.md) §2.

---

## 1. Why these two targets

The project's primary research question (`README.md`, `Project_Review_II.md`
§1.3) is:

> Does a verbalised readout of residual-stream activations tell us anything
> useful about the part of a reasoning trace that comes after the answer has
> already settled — **beyond what much cheaper signals already say**?

The clause in bold is what makes these analyses part of the study rather than
an exercise bolted onto it. A claim of *incremental* value is only as good as
the baseline it is incremental over. The falsification battery
(`PROJECT_PLAN.md` §6) already names the two cheap explanations that have to be
ruled out:

| | Alternative explanation | Test |
|---|---|---|
| **F1** | the readout adds nothing over trace **length** | add a length-only predictor to the baseline set |
| **F2** | the readout adds nothing over **difficulty** | stratify by MATH level and gold-answer magnitude |

F1 and F2 as specified are yes/no checks. These two analyses are the same two
questions asked quantitatively, with an interval: *how much* of each outcome do
the cheap, activation-free signals actually explain?

### Regression target: `tail_fraction`

The proportion of a trace's chunks lying inside the Answer-Stable Tail — the
region from which the answer no longer changes under truncation or under `K`
independently resampled continuations.

It qualifies on four counts:

1. **It is the study's own construct**, defined in advance in `PROJECT_PLAN.md`
   §5 and computed by the `ast` stage, which the stage DAG forbids from reading an
   activation. It was not chosen after seeing the data.
2. **It is continuous and bounded** in `[0, 1]`, and genuinely varies — in the
   corpus it spans roughly 0.14 to 0.63 with a mean near 0.34.
3. **It is the quantity the practical payoff is denominated in.** `tokens_saved`
   in the O3 stopping comparison is a function of where the tail starts; a model
   of `tail_fraction` is a model of how much there is to save.
4. **Predicting it from cheap features is exactly F1 and F2.** The feature set
   contains trace length and the difficulty proxies by construction, so the
   regression answers both at once and reports an interval rather than a verdict.

### Classification target: `final_correct`

Whether the trace's final answer is equivalent to the gold answer under
`math_verify`. Binary, and adjudicated by the same external verifier the rest
of the pipeline uses, so the label is not the analysis's own opinion of itself.

It qualifies on three counts:

1. **Safe stopping is defined in terms of it.** `PROJECT_PLAN.md` §5: a stop at
   boundary `s` is *safe* iff the forced answer and the full-trace answer are
   both correct or both incorrect. A rule that stops early on a trace heading
   for a wrong answer has not saved tokens — it has locked in an error. So
   whether correctness is predictable at all, and from what, sits directly
   underneath the comparison the study is scored on.
2. **It is already a target in the project.** `probe.kind: logistic` in
   `configs/base.yaml` is a hidden-state correctness probe on layer-20
   activations (O3, after Zhang et al.). This stage fits the *same label* with
   the *same model family* on *no activations at all*. Its AUROC is therefore
   the floor the layer-20 probe has to clear before decodability can be claimed
   as the thing doing the work.
3. **It is F1 and F2 again**, on the other outcome.

### What was deliberately not used as a target

- `tail_tokens`, `tail_start` — monotone functions of `tail_fraction` and the
  trace length already in the features; modelling them would restate the
  regression with a scale confound added.
- `ast_status` as a multi-class target — the classes are an edge-case taxonomy
  (`unparseable`, `truncated_generation`), not a scientific contrast.
- Reconstruction FVE, faithfulness scores, intervention effects — all of these
  require the autoencoder, which at the time of writing has not been downloaded
  (`STATUS.md` §P4). Targets that do not exist yet were not invented.

---

## 2. Unit of analysis and sample

One row per problem: one greedy trace, one AST record, one correctness label.

The **regression** runs on a primary cohort and two pre-declared sensitivity
cohorts, because two exclusions are judgement calls and both are reported
rather than resolved silently:

| Cohort | Definition | Why |
|---|---|---|
| `primary` | `ast_status ∈ {ok, stable_at_zero}` and not truncated | a measured tail, and a final answer that is actually final |
| `with_no_stable_point` | primary **+** traces where no boundary qualified | `no_stable_point` records `tail_fraction = 0.0` by *convention*, not measurement; including it conflates "no redundancy" with "detector found none" |
| `with_truncated` | primary **+** traces that hit the token cap | a truncated trace's "final" answer is the answer at the cap, so tail-ness relative to it is not the quantity we mean |

The **classification** runs on every successfully generated trace, truncated
ones included: truncation is itself a plausible predictor of being wrong, so
excluding those rows would remove signal rather than noise.

---

## 3. Features

Built once in `src/nlaast/supervised/features.py` and shared by both scripts.
Every column comes from either the problem statement or the *surface form* of
the generated trace. **Nothing here reads an activation and nothing calls the
model again** — that is the point, since these models are the cheap tier whose
job is to set a floor.

Three **nested** blocks, so each analysis reports what the next tier of
information is worth rather than one pooled score. Asking the incremental
question inside the cheap tier is the same discipline the study applies to the
expensive one.

### `A_problem` — known before a single token is generated

`q_chars`, `q_words`, `q_numbers`, `q_number_density`, `q_math_char_fraction`,
`prompt_tokens`, `math_level`, `gold_abs_log10`, `gold_is_integer`, `dataset`,
`subject`.

This is where F2's difficulty stratifiers live.

### `B_trace` — `A` plus the trace's own surface statistics

Shape: `n_chunks`, `n_tokens`, `tokens_per_chunk_mean`, `tokens_per_chunk_sd`,
`chunk_chars_mean`, `chunk_chars_max`, `short_chunk_fraction`, `truncated`.
This is where F1's length predictor lives.

Content density: `equals_per_chunk`, `operators_per_chunk`, `digit_fraction`,
`math_char_fraction`, `type_token_ratio`.

Discourse cues, counted by plain case-insensitive regex (a deliberately
explainable lexicon — see `CUE_PATTERNS`): `cue_conclude_per_chunk`,
`cue_verify_per_chunk`, `cue_answer_per_chunk`, `cue_restate_per_chunk`.
`cue_verify` matters most: the tail is commonly *claimed* to be the model
checking its work, and this counts the words that claim would predict.

Answer timing: `first_answer_mention` (relative chunk index at which the final
answer's *value* first appears verbatim in the trace body),
`answer_mention_count`, `has_answer_cue`.

> A note on why the timing feature is the value and not the announcement. The
> system prompt instructs the model to put `The answer is X` on the last line,
> so the announcement cue sits in the final chunk by construction and carries
> no information — measured at 1.0 or missing on every trace in the pilot. The
> value's first appearance is not constrained that way and does vary. This was
> found by inspecting the feature, not assumed.

### `C_convergence` — `B` plus the answer-convergence signal

`convergence_position`, `has_convergence`, `n_parsed_boundaries`,
`parsed_boundary_fraction`.

This is Liu & Wang's agreement rule: the first boundary at which
`convergence_window` consecutive parsed intermediate answers agree. It is the
cheapest stopping signal in the study — it parses prefixes and makes no extra
model calls, no forcing and no resampling — and it is the rule the primary
research question asks the verbalised readout to beat. It gets its own block so
its contribution can be read off separately.

### Carried caveats

- `math_level` is missing for every GSM8K problem by construction. It is
  imputed inside the pipeline with `add_indicator=True`, so "GSM8K, no level"
  is an explicit feature rather than a silent zero.
- `math_level` and `gold_abs_log10` are **problem metadata, not things a live
  stopping rule would have.** They are included because F2 names them as the
  difficulty stratifiers. The block structure means their contribution is
  visible as the `A_problem` score and can be discounted when reading the
  result as a deployment claim.
- Semantic entropy is **not** available as a predictor. The corpus was
  generated with `--no-entropy` (≈1k tokens per problem saved, and that arm is
  a separate O3 baseline). This is a real gap: a cheap uncertainty signal is
  absent from the cheap tier, so the floor reported here is, if anything, a
  little low.

---

## 4. Leakage prevention

Four measures, in descending order of how easy they are to get wrong.

**1. The split is not ours to choose.** Both analyses use the run's existing
`train`/`eval` assignment, a hash of the problem id fixed in
`configs/base.yaml` (`data.train_fraction: 0.4`) long before these analyses
existed. It is the one split in the project that cannot have been picked to
flatter a result. Hyperparameters are selected by 5-fold CV *inside* `train`;
`eval` is scored exactly once per model.

**2. All preprocessing lives inside the `Pipeline`.** Imputation, scaling and
one-hot encoding are pipeline steps, so each CV fold fits them on its own
training part. Standardising the whole table before splitting is the single
most common way an analysis like this inflates its own scores; here the split
is never crossed outside a `fit`.

**3. The operating point is chosen on train.** The classification threshold
maximises Youden's J on the *training* split and is then frozen. Choosing it on
`eval` would make the confusion matrix and every rate derived from it
optimistic.

**4. No feature is a function of the target.** For `tail_fraction` this means no
feature touches boundary *forcing* or *resampled continuations* — the two
measurements the AST criteria are made of. `convergence_position` is admissible
precisely because it uses neither: it reads parsed prefixes only. For
`final_correct` it means no feature touches the gold answer except
`gold_abs_log10`/`gold_is_integer`, which are magnitude and type of the gold
*value* and carry no information about whether the model found it.

### The check that this worked

Both scripts run a **label-shuffle null** (`permutation_null`), the
analysis-level counterpart of falsification test **F6**: the model is refit
from scratch on permuted labels, 200 times, and the real cross-validated score
is placed against that null. A model whose score does not collapse when the
labels are destroyed is reading something other than the signal it claims to.
The reported p-value is `(#{null ≥ observed} + 1) / (B + 1)`.

---

## 5. Models and how the stronger one has to earn its place

Both analyses run a ladder, and the headline model is **not** chosen by which
number is biggest.

| | Regression | Classification |
|---|---|---|
| trivial reference | predict the training mean | predict the majority class |
| interpretable | `Ridge` (tuned α) | `LogisticRegression` (tuned C, `class_weight='balanced'`) |
| inference | `OLS` — coefficients, CIs, residual diagnostics | `Logit` — odds ratios with Wald CIs |
| stronger | `HistGradientBoostingRegressor` | `HistGradientBoostingClassifier` |

**The gradient-boosted model is reported as the headline only if it beats the
linear model by a paired-bootstrap interval that excludes zero** (MAE for the
regression, AUROC for the classification). If it does not, the linear model is
the reported one and the boosting result is recorded as not justified. This is
decided in code (`gbm_justified`) before any write-up, not by taste afterwards.

The bootstrap is **paired** — the two models are scored on identical problems,
so the problem-to-problem variance is shared and an unpaired interval on the
difference would be wider than the data warrants.

Tree hyperparameter grids keep `min_samples_leaf ∈ {8, 15}`. On ~100 training
rows a smaller leaf is memorising individual problems and the CV estimate stops
meaning anything; this is a constraint, not timidity.

### The inference models get a smaller feature set, and why

`OLS` and `Logit` are fitted on `INFERENCE_FEATURES` — ten pre-declared columns,
one per mechanism the falsification battery names — rather than all thirty-five.
This is arithmetic, not modesty: thirty-five features plus imputation indicators
against roughly a hundred training rows gives a design matrix that is
rank-deficient or close to it, and statsmodels answers a rank-deficient design
with a pseudo-inverse solution whose coefficients and standard errors look
entirely ordinary and mean nothing. The list was fixed before any model was
fitted, not pruned to whatever came out significant.

Two further guards, both added because a test caught the failure rather than
because it was anticipated:

**A structural collinearity.** `math_level` is missing for *exactly* the GSM8K
problems, so the imputer's missingness indicator for it is **identical** to the
`dataset=gsm8k` dummy. A design holding both cannot give either a coefficient.
`independent_columns` drops constant columns and then any column almost
perfectly correlated with one already kept, preferring the earlier one so the
choice depends on the declared feature order rather than on the data. What it
dropped is logged and recorded in `summary.json` under
`ols_diagnostics.dropped_columns`.

**Standard errors that do not exist.** The logistic inference fit tries
unpenalised MLE first, because statsmodels' *regularized* results leave `bse`
and `conf_int` undefined for shrunk coefficients — a penalised fit would have
quietly filled the odds-ratio table with values of exactly 1.0 and missing
intervals. The penalised fit survives only as a fallback for perfect
separation, where the MLE diverges; in that case the table is flagged
`penalised_no_ci` and no interval or p-value is claimed at all.

---

## 6. Metrics and statistical evaluation

**Regression.** R², MAE, RMSE on the held-out split, each with a percentile
bootstrap CI over problems. R² is reported against the training-mean predictor
so a negative value is readable as "worse than knowing nothing".

**Classification.** AUROC and AUPRC (both with bootstrap CIs), balanced
accuracy, F1, Brier score, and a calibration curve. AUPRC and prevalence are
reported together because AUROC alone is misleading at ~80% prevalence.
Bootstrap resamples that happen to contain a single class are discarded rather
than coerced, and the count of usable draws is reported — the honest way to say
an interval rests on fewer draws than requested.

**Inference.** Coefficient families get Benjamini–Hochberg at q = 0.05, the
same correction the project applies to its pre-registered test family (F9).

**Linear-model diagnostics** (regression, on the OLS fit): residuals vs fitted,
normal Q–Q, residual histogram, residuals vs trace length (F1 made visual),
Breusch–Pagan for heteroscedasticity, Jarque–Bera for residual normality,
Durbin–Watson, condition number, VIF, and observations-per-parameter. VIF is
computed only when the design has enough rows for it to mean anything; with `p`
close to `n` a VIF of 1e9 is an artefact, not a finding.

**Error analysis** (classification): held-out error rate, false-correct and
false-incorrect counts per stratum — benchmark, trace-length tertile,
difficulty group, truncation — so a pooled AUROC cannot hide a broken subgroup.
The two error types are not interchangeable: a **false "correct"** is a
stopping rule endorsing a wrong answer, which is the failure that matters.

**Feature analysis**: permutation importance on the held-out split for the
reported model, plus the coefficient/odds-ratio table for the linear one.
Permutation importance is computed on `eval`, not on `train`, because on train
it measures what the model memorised.

---

## 7. How to reproduce

```powershell
$env:NLAAST_QUANT_DIR = "<volume>/nf4"
$env:HF_HOME = "<volume>/hf_cache"

# corpus (resumable; reuses runs/pilot traces, whose settings are identical)
.venv\Scripts\python.exe scripts\prepare_problems.py          --config mlcorpus
.venv\Scripts\python.exe scripts\generate_traces.py           --config mlcorpus --no-entropy
.venv\Scripts\python.exe scripts\detect_answer_stable_tail.py --config mlcorpus

# the two analyses
.venv\Scripts\python.exe scripts\predict_tail_fraction.py     --config mlcorpus
.venv\Scripts\python.exe scripts\predict_correctness.py       --config mlcorpus
```

Each analysis stage writes `summary.json`, tables as both `.csv` and `.json`,
and PNG figures into its own directory under `runs/mlcorpus/`. Both are pure
functions of the saved stage outputs — neither loads a model — so they
regenerate in seconds without a GPU.

---

## 8. What these analyses cannot do

- **No causal claim.** These are predictive models on observational trace data.
  That `n_chunks` predicts `tail_fraction` does not mean lengthening a trace
  lengthens its tail. The project's causal claims belong to the `causal` stage and its
  matched-random-direction controls.
- **No statement about the autoencoder.** Neither model reads an activation.
  They set the floor the layer-20 probe and the verbalised readout must clear;
  they say nothing about whether those clear it.
- **Single-model, single-decoding-config corpus.** Qwen2.5-7B-Instruct at 4-bit
  NF4, greedy, 512-token cap, two benchmarks. Nothing here transfers to another
  model, and the quantisation deviation documented in `PROJECT_PLAN.md` §2
  applies to the traces these features are computed from.
- **The achieved N is the reported N.** Sample size is set by how much
  generation finished, not by what would have been comfortable. Where an
  interval is wide, that is the result.
