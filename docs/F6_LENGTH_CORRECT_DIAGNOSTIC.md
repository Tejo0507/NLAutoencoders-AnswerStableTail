# F6 diagnostic: length-correct label shuffle (PROPOSED, not approved)

Separate from F6. F6's registered result (real AUC 0.9460944231, across-groups
shuffled AUC 0.6142873504, within-group 0.5385), its threshold
(|shuffled AUC - 0.5| > 0.15), `leak_suspected = false`, its interpretation,
`permute_labels` and every registered artifact are unchanged. Exploratory and
descriptive; no leakage verdict.

## Fixed inputs

F6's own assembly (2,681 rows, 250 problems; no new exclusions); folds
`grouped_folds(groups, 5, seed=11)` (label-free, identical in every arm);
`CorrectnessProbe` (standardised logistic, C = 1, lbfgs); pooled out-of-fold
AUC.

## Gate

Rebuild F6's inputs; run the registered `np.resize` shuffle (seed 12) with the
unmodified `permute_labels`. Require 718 labels changed and AUC 0.6142873504
(|diff| < 1e-9), and real AUC 0.9460944231. On mismatch: stop, write no result,
confirm the `robustness.json` SHA-256 is unchanged.

## Arms

One derangement pi_b per draw b, shared by S0 and S1.

- **S0** (full sample): registered `np.resize`.
- **S1** (full sample; sensitivity analysis for Question A): receiver row j
  of n_r takes donor row phi(j) = floor(j * (n_d - 1) / (n_r - 1) + 1/2);
  positions are F6 row indices. S1 is one reasonable length-fitting rule, not
  a uniquely correct or definitive null (see "What S1 preserves").
- **S2** (secondary, Question A'): exact-length subset; donor from the same
  length stratum, copied verbatim. S0 and S1 also refitted on the subset with
  their own subset derangement. When n_d = n_r, S0 = S1 = S2.
- **Positional oracle** (Question B primary; no fitting; defined below).
  Full sample, S0 and S1 only.
- **P** (secondary): A1's four position features (relative position, boundary
  index, log prefix tokens, final-boundary flag), linear logistic, same folds.

## What S1 preserves

Write M_s(y_d, n_r) for arm s's length-fitting rule (S0: `np.resize`; S1:
y_d[phi(j)]). The shuffled label of receiver r, row j, in draw b is
y~_{r,j} = M_s(y_{pi_b(r)}, n_r)[j].

- **Preserved exactly:** receiver length n_r; the donor assignment pi_b (the
  same as S0's in that draw); donor label order (phi is non-decreasing); the
  donor's first and last labels; monotonicity of a monotone donor sequence;
  the donor sequence itself when n_d = n_r; every donor label at least once
  when n_r >= n_d.
- **Preserved approximately:** per-problem and overall prevalence; the
  relative location of each transition (within half a donor step); the
  label-relative-position association.
- **May be altered:** the exact positive count; the number of transitions
  and runs when n_r < n_d (runs shorter than the step can disappear); run
  lengths (stretched or compressed, unevenly through rounding); positions
  counted from the start or end of the trace; labels duplicated when
  n_r > n_d.
- **Destroyed (intended):** the correspondence between a receiver's
  activations and its own labels.

**What Delta_A isolates.** S0 and S1 share donors, sample, rows, folds and
model, and differ only in the length-fitting rule. Delta_A is therefore the
effect of replacing `np.resize`'s rule with phi, given the same donors. It is
not the effect of tiling and truncation measured against an ideal null:
phi also duplicates and drops labels, spread along the sequence rather
than at its end. Equal-length receivers contribute identical labels to both
arms.

## Positional oracle

For arm s in {S0, S1}, draw b, receiver r and row j (0 <= j < n_r):

    D_r    = all problems except r and pi_b(r)          (|D_r| = 248)
    g_r(j) = (1 / |D_r|) * sum over d in D_r of M_s(y_d, n_r)[j]

g uses only the real label sequences of other problems and the receiver's row
positions (j, n_r): no activations, no fitted model. **Oracle AUC** = the
pooled AUC of g_r(j) against y~_{r,j}, over the same rows the probe scores
out of fold (rows the probe leaves unscored are dropped from both). It is
computed once per arm and draw, like the probe's pooled out-of-fold AUC. g
is not fitted per fold, and D_r is not restricted by fold (see "Decisions").

**Why it is position-derived, not an upper bound.** Given pi_b's
independence from the receiver, y~_{r,j} depends on the receiver only through
(j, n_r), and g_r(j) estimates its expectation from those alone. Ranking by g
is still not guaranteed to maximise pooled AUC for a position-only predictor:

- rows of one problem share a donor, so their labels are dependent, and the
  AUC-maximising ranking of dependent pairs need not follow marginal
  expectations;
- pooled AUC normalises by the realised positive and negative counts, which
  vary by draw;
- g averages over the 248 other sequences, while the realised labels come from
  one draw without replacement; a predictor can beat g on a draw by chance;
- the pooled AUC mixes rows across problems, where g's scale is shared but
  the probe's fold models are not.

So a position-only predictor, or the probe, can score above g without any
defect. The oracle is not a guaranteed upper bound, not a formal null
distribution and not a definitive leakage test.

**The probe and the oracle aggregate information differently.** g is an
analytic average: for each row it combines the real label sequences of 248
other problems through the exact length-fitting rule, using the receiver's
position (j, n_r) directly. The probe sees position only as far as it is
encoded in the activations, learns from one realised set of shuffled training
labels, and its pooled AUC concatenates scores from five fold-specific
models with different fits. A gap between the two can therefore come from
how each aggregates, not only from what information is available.

**Reading (conservative).** Probe AUC above the oracle means the probe
outperforms this particular position-derived benchmark on the observed
shuffled labels: a diagnostic discrepancy to investigate (donor
independence, the assembly, fold handling, aggregation, chance), not proof
of leakage. Probe AUC similar to or below the oracle does not establish that
position fully explains the predictability, and does not establish that
leakage is absent.

## Seeds and draws

Draw 0: pi_0 = F6's own derangement (seed 12). Draws b >= 1:
`default_rng(SeedSequence([11, 0xF6, 0, b]))`, rejection-sampled until
fixed-point-free; subset stream `[11, 0xF6, 1, b]`; S2 strata `[11, 0xF6, 2, b]`.

B = 100. Projected time = t (gate S0 fit) x 5 activation fits/draw x B x 1.25.
B = 100 if <= 6 h, else B = 50; if the B = 50 projection > 12 h, stop before
any new fit. No early stop, no extension. Estimate <= 75 min, low confidence.

## Stage 0: label statistics before any fit

Per arm and draw: prevalence (overall, per problem); labels changed /
duplicated / dropped; transitions; monotone share; label-relative-position
AUC. Recorded, never used to choose a mapping.

## Questions

| Question | Comparison | Metric | Uncertainty (two separate summaries) | Limit |
|---|---|---|---|---|
| A: sensitivity of the shuffled AUC to np.resize's length-fitting rule | S1 vs S0, same pi_b, full | Delta_A = mean_b[AUC(S1) - AUC(S0)] | donor-draw summary; problem-bootstrap summary | compares two fitting rules, not np.resize vs an ideal null; S1's no-leak expectation is not 0.5 |
| A': exact pattern kept | S2 vs S1 vs S0 on subset | per-arm AUC; differences | as A, on subset | selected subset (4 longest problems out); no automatic generalisation |
| B: predictability relative to a position-derived benchmark | probe vs oracle; P secondary | AUC(oracle); E = AUC(probe) - AUC(oracle); (AUC(probe) - 0.5) / (AUC(oracle) - 0.5) | as A | oracle is a benchmark: not an upper bound, null distribution or leakage test |
| C: evidence of leakage | follows from B | none | none | no verdict; never shows absence of leakage |

## Uncertainty

Two summaries, always reported separately, plus one supplementary
Monte Carlo figure. None of them includes model-refitting variability, which
remains unmeasured, and they are not combined into one interval or test.

- **Donor-draw variability.** The distribution of the per-draw statistic
  (AUC, Delta_A, E) over the B draws: mean, SD, and 2.5th-97.5th percentiles.
  It conditions on the observed problems, labels, activations, folds and
  model specification, and varies only the donor derangements. It omits
  problem-sampling, fold-assignment and refitting variability. The percentile
  range describes how single draws vary (the registered F6 used one); it is
  not an interval for the mean.
- **Monte Carlo standard error (supplementary).** SD / sqrt(B): the
  precision of the mean over donor draws as an estimate of the mean over all
  possible draws, for this dataset and these fits. It is not the spread of
  individual draws (above), not problem-level sampling uncertainty (below),
  and not refitting uncertainty (unmeasured). It is reported beside the
  draw summary and is not used in the wording table.
- **Problem-level sampling variability.** Cluster bootstrap: 2,000 problem
  resamples with replacement, fixed seed (`analysis.seed`), no refit. Each
  replicate recomputes, on the fixed out-of-fold predictions and fixed
  shuffled labels of every draw, the draw-averaged statistic. The same
  resample is used for all arms, so differences stay paired. It conditions on
  the fitted predictions, the folds and the B draws, and omits refitting
  variability (so it is probably too narrow) and treats problems as
  exchangeable.
- No interval for full-minus-subset differences (overlapping samples).

**Predeclared wording.** Every statistic is reported with its effect size and
both summaries. "Excludes 0" for the draw summary means at least 97.5% of
draws share the sign.

| Draw summary | Bootstrap summary | Wording |
|---|---|---|
| excludes 0 | excludes 0, same sign | "consistent in sign across donor draws, and the draw-averaged value is stable under problem resampling (two separate summaries, not a joint confidence statement)" |
| excludes 0 | includes 0 | "consistent across donor draws for this dataset, but not distinguishable from problem-sampling variability" |
| includes 0 | excludes 0 | "the draw-averaged value is stable under problem resampling, but individual draws vary in sign; a single draw would not reliably show it" |
| includes 0 | includes 0 | "no consistent difference in either summary" |
| exclude 0, opposite signs | | reported as conflicting; no direction stated |

## Exact-length subset (read-only count)

Lengths with >= 2 problems: 18 lengths, 246 / 250 problems, 2,577 / 2,681
rows. Excluded: the 4 problems whose length no other problem shares (lengths
20, 21, 29, 34): 104 rows, and also the four longest traces. Size-2 strata
(lengths 4, 23, 24; 6 problems) are forced swaps, so they contribute no
donor-draw variability.

S2 stays secondary. The subset is selected by length, so a difference between
subset and full-sample results can reflect sample composition (no long traces)
as well as the same-length donor constraint; the design cannot separate the
two. Subset results are reported as subset results and are not generalised
to the full sample.

## Reading rules

Uncertainty wording as in the table above; no "material" label.

- Delta_A distinguishable from 0: "replacing np.resize's length-fitting rule
  with phi, given the same donors, changes the shuffled AUC by Delta_A"; the
  registered 0.6143 depends on the fitting rule as well as on donor
  reassignment. Not about leakage, and not a revision of F6.
- Delta_A not distinguishable from 0: no detectable sensitivity to the
  fitting rule at this precision; not equivalence, and the audit's
  label-statistic findings stand.
- Probe above oracle: the probe outperforms this position-derived benchmark
  on the observed shuffled labels; a diagnostic discrepancy to investigate
  (see "Positional oracle"); not proof of leakage; not a verdict.
- Probe similar to or below oracle: no discrepancy from this benchmark; does
  not establish that position explains the predictability or that leakage is
  absent.
- P: "linear explicit features reach X"; no inference from probe vs P.
- Full vs subset disagree: report both; composition and the donor constraint
  are both possible causes and are not separated; no pooling or choosing.

## Decisions (researcher, 2026-10-18, before any fit)

1. **Oracle donor pool:** the definition above is kept (all problems except r
   and pi_b(r)). A fold-restricted pool is not used in this experiment.
2. **Monte Carlo standard error:** SD / sqrt(B) is retained as a
   supplementary figure, as described under "Uncertainty".

## Safeguards

Run ID `main_f6_diag`; output `runs/main_f6_diag/`; script
`scripts/diag_f6_length_correct.py` (no torch); this document and a dated
DECISIONS entry with the code SHA-256 committed before execution. The script
refuses to run if the output directory exists, verifies the `robustness.json`
SHA-256 before and after (stops on mismatch), and never writes to `runs/main`.
