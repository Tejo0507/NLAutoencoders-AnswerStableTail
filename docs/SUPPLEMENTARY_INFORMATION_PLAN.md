# What the readout knows: an information audit beside the stopping metric (D81)

**Frozen before any quantity below was computed on any run.** D81 records
this file's SHA-256. Later changes are appended under *Deviations*, dated,
and never applied silently.

**What had been seen when this was written.** On `main` and `main_nla240`:
every D69 and D79 result. That includes the readout score's falling position
trend (median within-problem ρ −0.37), the stop counts at 0.5·B\*, F10's
tail-versus-matched-position cosine (0.873 vs 0.878) and the truncation
rates. None of Q1–Q5 had been computed. On `rep1`, no outcome of any kind
existed: 166 of 400 traces, no tail detection, no verbalisation, no forcing.

## 1. Why this is needed

The registered O3 estimand asks whether a *thresholded stopping rule* breaks
fewer answers per token saved. D79 showed that a rule using no readout
matches the readout's advantage there. The reason is that the estimand
rewards abstaining plus a few very early stops. That says little about
whether the readout *contains* information about the model's state. A
readout could carry real information and still lose under that estimand, or
win it for the wrong reason. This plan asks the information question
directly, with cheaper information sources as controls:
- **Q1:** is the readout's score informative about a chunk's safety beyond
  position and the probe?
- **Q2:** is the verbalised text informative beyond the trace text the model
  has already written?
- **Q3:** does the text name the eventual answer before the trace writes it?

**None of this replaces or re-reads the registered O3 test, D69 or D75.**

## 2. Units, outcomes and features

**Unit:** a distinct menu chunk `(problem, chunk i)`, design A's menu. That
is a chunk with at least one integrity-passing, non-control verbalisation.
Every menu chunk has a measured forced outcome.

**Limitation, stated now.** The menu is selected from the tail start: the
matched-position window ends at `i* − 1`, and the tail window starts at
`i*`. So every result below is about information *at outcome-selected
points*. It is not about a deployable rule. `docs/LEAKAGE_RESISTANT_DESIGN.md`
removes that selection.

**Outcomes:**

| name | definition | role |
|---|---|---|
| `Y_reg` | the registered safety label: the forced answer's correctness equals the final answer's (D69 `problems_for`) | **primary** |
| `Y_tail` | `i ≥ i*` (AST tail membership; no forcing involved) | secondary |
| `Y_commit` | D79's *committed-to-final* reading: a cut-off forced completion counts as the final answer | sensitivity |
| `Y_ext` | the extended-budget label (`docs/EXTENDED_FORCING_PLAN.md`), once measured | sensitivity, added when it exists |

**Features**, all computed per unit without outcome information:

| block | features | available online? |
|---|---|---|
| `pos` | `log(prefix_tokens)`, chunk index, `is_MATH` | yes |
| `probe` | `logit(p)` clipped to [1e-6, 1 − 1e-6], 0 where the probe has no score, plus a missing indicator | yes (where an intermediate answer exists) |
| `readout` | the registered readout score: mean reconstruction cosine of the unit's integrity-passing real samples, as returned by `reanalyse_o3_registered.rule_scores` | yes, at verbaliser cost |
| `diag` | `prefix_tokens / total_tokens`, `log(total_tokens)` | **no**: uses the trace's length; a diagnostic for confounding by length and difficulty |
| `E` | the explanation texts of the unit's integrity-passing real samples, concatenated | yes, at verbaliser cost |
| `R` | the last 600 characters of the prefix text through chunk `i` | yes, free |
| `N` | the Gaussian-control (`noise_samples`) text at the same unit (`main`, `main_nla240` only) | null control |

## 3. Models and estimation

**Classifier, identical everywhere:**
- a standardised logistic regression with C = 1.0, L2 and `max_iter` 5000;
- each text block goes through its own TF-IDF (word 1–2-grams, `min_df` 3,
  sublinear tf, lowercase) and is stacked with the numeric block;
- the vectoriser and scaler are fitted inside each training fold.

There is no tuning.

**Cross-validation:** `StratifiedGroupKFold`, 10 folds, grouped by problem,
stratified on the outcome, seed 20261020. Every number is computed on
out-of-fold predictions.

**Uncertainty:** a problem-cluster bootstrap of the out-of-fold predictions,
2,000 replicates, seed 20261020, percentile 95% CI. The two-sided bootstrap
p-value is `2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))`, floored at 1/2000.

**Metric:** pooled AUC over units. Mean log loss is reported beside it.
Within-problem concordance (the share of a problem's safe/unsafe unit pairs
ordered correctly) is descriptive only.

**CV-seed robustness:** the primary contrasts are recomputed with seeds
20261021–20261025 and their range is reported. The inference stays the
seed-20261020 one.

## 4. Questions and contrasts

### Q1. Readout score beyond position and probe

| contrast | role |
|---|---|
| **Q1-P:** AUC(`pos+probe+readout`) − AUC(`pos+probe`), on `Y_reg` | **primary** |
| AUC(`pos+readout`) − AUC(`pos`) | secondary |
| AUC(`pos+probe`) − AUC(`pos`) | secondary (the probe's own increment) |
| AUC(`pos+diag+probe+readout`) − AUC(`pos+diag+probe`) | secondary: is the increment explained by length or difficulty? |
| the same four on `Y_tail` and `Y_commit` | secondary |

### Q2. Verbalised text beyond the trace text

| contrast | role |
|---|---|
| **Q2-P:** AUC(`pos+R+E`) − AUC(`pos+R`), on `Y_reg` | **primary** |
| AUC(`pos+E`) − AUC(`pos`) | secondary |
| AUC(`pos+R`) − AUC(`pos`) | secondary (what the free text alone gives) |
| AUC(`pos+N`) − AUC(`pos`) | null check: should be ≈ 0; if its CI excludes 0 the text pipeline is leaking, and Q2 is not interpreted |
| the same on `Y_tail` | secondary |

### Q3. Does the text name the answer before the trace does?

**Eligibility:**
- the final answer parses as a plain number (integer or decimal, after
  removing `$`, `,` and a trailing `.0`);
- that number does not occur as a number in the question or in the prefix
  text through chunk `i`.

**Number matching:** every number in a text is found with
`-?\d[\d,]*(?:\.\d+)?`, has its commas removed and trailing zeros
normalised, and is compared as a string.

**Statistic:**
- `own(u)` is 1 if any of the unit's real explanations contains its
  problem's final answer;
- `other(u)` is the mean, over every other problem of the same dataset whose
  answer is also absent from `u`'s question and prefix, of whether `u`'s
  explanation contains that problem's answer;
- **Q3-P = mean over units of `own − other`** (primary). This controls for
  how often an explanation contains an arbitrary plausible answer.

**Secondary:**
- the noise null: `own` computed on the Gaussian-control text (`main`,
  `main_nla240`);
- Q3-P separately for pre-tail (`i < i*`) and tail units;
- among eligible units, `P(Y_reg = 1 | mention)` against
  `P(Y_reg = 1 | no mention)`, which asks whether the text naming the answer
  goes with the answer being recoverable by forcing.

**What it is not.** This is string matching. A mention is evidence that the
number is decodable from the verbalisation of this activation. It is not
evidence that the explanation is faithful to the computation.

### Q4. Subgroups

Q1-P, Q2-P and Q3-P are reported separately for GSM8K and MATH (2 strata),
with the difference between strata and its bootstrap CI.

- **Why this split:** MATH traces are longer and have more cut-off forced
  completions (D79), so it is the one split with a stated mechanism.
- **Status:** exploratory. No other subgroup is examined.

### Q5. Descriptive: is the readout score reliable?

- Pearson and Spearman correlation of the two verbaliser samples' cosines at
  the same unit.
- The share of the readout's variance that lies between problems (a one-way
  ICC).

A readout whose two samples barely agree has a low ceiling on the
information it can carry, whatever it encodes.

## 5. Multiplicity and decision language

**Holm over {Q1-P, Q2-P, Q3-P}** at α = 0.05 on the primary run. Everything
else is secondary: its 95% CI is reported, unadjusted and labelled.

| reading | rule |
|---|---|
| **increment detected** | Holm-significant with Δ > 0 |
| **no practically relevant increment** | the 95% CI of ΔAUC lies inside (−0.02, +0.02) |
| **unresolved** | otherwise |

For Q3-P the margin is 0.02 in rate.

## 6. Runs and their status

| run | status |
|---|---|
| `main_nla240` (240 problems) | **primary for this plan; exploratory**, because the run's other results had been seen |
| `main` (its first 120) | consistency check; not independent of the above |
| `rep1` | **pre-specified, secondary to D75**, computed only after D75's confirmatory analysis exists. This is the independent test of any Q1–Q3 result from `main_nla240`. `N` is unavailable there (no Gaussian controls), so Q2's null check and Q3's noise null are omitted |

**Code:**
- `src/nlaast/analysis/information.py` (pure functions, tested on synthetic
  data with known structure);
- `scripts/information_audit.py` → `runs/<run>_information/information.json`.

The script refuses to overwrite its output, and nothing under `runs/<run>/`
is written.

## Deviations

1. **2026-10-20, after the plan's results on `main_nla240` were seen: four
   post-hoc checks of Q2-P.** Q2-P came out large: +0.073 AUC
   [0.058, 0.087]. The plan's controls do not rule out three explanations:
   - **Sample multiplicity.** A chunk in two windows (for example tail and
     matched-length) has four samples, not two. Window membership comes from
     the tail start. The `N` null shows no leak through that channel, but it
     is not the same text.
   - **A weak raw-text control.** `R` sees 600 characters.
   - **A mismatched cheap baseline.** The verbaliser reads only the layer-20
     activation. The registered probe is trained for intermediate-answer
     correctness, not for this label. The fair cheap competitor is a linear
     probe on the *same* activation, trained for the *same* label.

   Added:
   - **`E2`:** the texts of exactly one verbalisation row per unit. The row is
     the one whose id sorts first, at most 2 samples. All rows at a chunk
     verbalise the same activation, so the choice is label-neutral.
   - **`Rfull`:** the whole prefix text in place of `R`.
   - **`A`:** a layer-20 activation probe for the unit's own label.
     - A standardised logistic regression whose C is chosen from
       {1e-4, 1e-3, 1e-2, 1e-1, 1} by inner 5-fold grouped CV on log loss,
       within each outer training fold. The baseline is tuned and the text
       models are not, which favours the baseline.
     - It enters the outer models as one stacked feature: inner out-of-fold
       scores on the training rows, and scores from the fold's fitted probe
       on the test rows.
   - **Descriptive:** the 20 most positive and most negative `E` terms of a
     model fitted on all units.

   Contrasts, on `Y_reg` and `Y_tail`:

   | contrast | question |
   |---|---|
   | `pos+R+E2` − `pos+R` | Q2-P without multiplicity |
   | `pos+Rfull+E` − `pos+Rfull` | Q2-P against the whole prefix |
   | `pos+A` − `pos` | what the activation itself holds |
   | **`pos+A+E` − `pos+A`** | does the text add anything to its own input? |
   | `pos+R+A+E` − `pos+R+A` | the same, beside the raw text |
   | `pos+E` − `pos+A` | text against probe, head to head |

   - **Status:** post hoc on `main` and `main_nla240`, so these are exploratory.
     They are **specified before any outcome for `rep1`**, where they are
     secondary.
   - **Output:** `runs/<run>_information/information_posthoc.json`. The plan's
     output is not modified.
2. **2026-10-20, after deviation 1's results were seen: an unfitted ensemble.**
   In `pos+A+E` a single strong stacked column sits beside thousands of TF-IDF
   columns under one penalty, and it could crowd the text out.
   - **Added check:** the equal-weight mean of the logits of `pos+A` and
     `pos+E`, which fits nothing. It is compared against `pos+A`.
   - **Reading:** if the text carries information the probe lacks, the
     ensemble beats the probe.
   - The output file is regenerated with this field added. The other fields
     are recomputed by the same deterministic code and must equal the first
     run's values. The script checks this.
3. **2026-10-20, before `rep1` had any verbalisation: a frozen-lexicon
   transfer test.** The heaviest `E` terms on `main_nla240` name a reasoning
   phase. Whether that is a property of the verbaliser or of one fitted
   model can be tested out of sample.
   - **The lexicon:** the 20 positive and 20 negative terms in
     `runs/main_nla240_information/information_posthoc.json` →
     `top_E_terms_Y_tail`, copied unchanged with their SHA-256 into
     `runs/main_nla240_information/lexicon.json`.
   - **The score of a unit:** the number of positive terms present in its
     `E` text, minus the number of negative terms. Unigrams and bigrams are
     matched after the TF-IDF tokenisation (lowercase, `\b\w+\b`). Nothing is
     fitted on `rep1`.
   - **On `rep1`**, specified now:
     - **L-P:** AUC of the lexicon score for `Y_tail`, with a
       problem-cluster CI;
     - secondary: the same for `Y_reg`, and AUC(`pos+lex`) − AUC(`pos`) by
       D81's cross-validation.
   - **Reading:** if L-P's CI lies above 0.5, the phase vocabulary transfers
     to unseen problems.
     - This is evidence that the verbaliser's text describes something about
       the state that recurs across problems.
     - It is **not** evidence of faithfulness to the computation, and not of
       information beyond the activation (deviation 1 already bounds that on
       the menu).
