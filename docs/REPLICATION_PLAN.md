# Confirmatory replication of design A on fresh problems: analysis plan (D75)

**Frozen before any trace was generated for these problems.** D75 records this
file's SHA-256. `scripts/reanalyse_o3_registered.py --confirmatory` refuses to
run if the file has changed. Any later deviation is appended below, dated, and
never applied silently.

## 1. Question

D69 found, post hoc, that at budgets where safe stopping is possible, and
given the same positions to stop at, the readout's reconstruction-cosine score
selects safer stops than chance. Whether it beats the hidden-state probe was
unresolved: +0.027 [−0.026, 0.076] at N = 120, and +0.057 [0.023, 0.091] on
the post-hoc 240-problem arm. This study tests both on problems that no
earlier run has seen.

## 2. Hypotheses (confirmatory)

| | hypothesis | estimand |
|---|---|---|
| **H1 (primary)** | the readout selects safer stops than the **layer-20** probe | Δ₁ = mean safe-rate gap, readout − probe, over the budget grid, design A |
| **H2** | the readout selects safer stops than a rule with no signal | Δ₂ = the same, readout − no-signal |

- **Why layer 20.** It is the registered probe: `target.layer`, `fit_probe`,
  as used by every registered and corrected O3 analysis. Layers 14 and 24
  are reported as secondary (D73 showed the comparison is sensitive to the
  choice) and do not enter the verdict.
- **The margin, 0.05,** is D69's.

## 3. Population and measurement

- **Problems:** `runs/rep1/data/problems.jsonl`, chosen by
  `scripts/prepare_fresh_problems.py`. That is 400 problems: 240 GSM8K and
  160 MATH (123 algebra, 37 counting and probability). Every id used by any
  earlier run (278 ids) is excluded, and the selection takes the next
  problems in the registered hash order.
- **Measurement:** identical to `main` (`configs/replication.yaml`): model,
  layer, greedy 768-token traces, chunking, the AST criteria (K = 5,
  T = 0.8, 200 tokens, 48-token forcing), and the verbaliser (2 windows per
  kind, 2 samples, 256 tokens). Three differences, none affecting a quantity
  in this analysis:
  - no semantic-entropy samples;
  - no Gaussian-control verbalisations;
  - no cap on the NLA arm.
- **Outcome:** registered (forced answer, PROJECT_PLAN §5). Forced answers
  come from the traces stage where its scan evaluated a boundary, and
  otherwise from `force_all_boundaries.py --menu-only`, which forces exactly
  the boundaries design A can stop at.
- **Instrument checks, before any outcome is used:**
  - 100 stored boundaries are re-forced. If agreement is below 0.95, the
    analysis is blocked.
  - Cross-problem batching (an engineering choice, D75) is used only if a
    benchmark on `main`'s stored boundaries first shows agreement ≥ 0.95.

## 4. Analysis

As in D69 plan §4–§6, design A only. Every rule may stop only at the
verbalised chunks; the scores are as registered.
- **Budgets:** {0.1, …, 0.9} × the menu's oracle bound; matching tolerance
  0.02; at least 3 matched budgets.
- **Uncertainty:** a paired problem bootstrap, 2,000 replicates with every
  curve rebuilt, seed 20261019. Two-sided bootstrap p-values, Holm over
  {H1, H2} at α = 0.05.
- **Decisions** (`confirmatory_verdict`, tested):
  - *supported* if Holm-significant with Δ > 0;
  - *reversed* if Holm-significant with Δ < 0;
  - *equivalent within ±0.05* if the 95% CI lies inside (−0.05, 0.05);
  - *advantage of 0.05 or more excluded* if the upper bound is below 0.05;
  - otherwise *inconclusive*.
- **Secondary, not in the verdict:** probes at layers 14 and 24; convergence;
  strata by subject; McNemar at 0.5 × B\*; the cut-off share of stops.

## 5. Sample size and stopping

- **Configured N:** 400. **Achieved N:** the longest prefix of the
  interleaved problem order with every stage complete when generation stops.
- Generation stops when all 400 are done or when compute is exhausted. That
  decision is taken without computing any O3 outcome. No forced-answer
  completion or analysis runs before it.
- Below 200 problems with a tail, the result is reported as an underpowered
  replication, not a confirmatory test.
- **Precision expected** from D69's SE (0.026 at N = 120, scaled by
  1/√N): at N = 400, SE ≈ 0.014 and the CI half-width ≈ 0.028. Power at
  two-sided α = 0.05 is about 0.45 for a true Δ₁ of 0.027, 0.80 for 0.040 and
  0.98 for 0.057.

## Deviations after the hash

1. **2026-10-20, before any outcome was computed, after 3 traces: batch size 2
   instead of 4 (D78).** A decode benchmark on this GPU gave 9.2 tok/s at
   batch 4, 33.0 at batch 2 and 20.9 at batch 1: batch 4 spills to shared
   memory.
   - **What changes:** the batch for sampled continuations, block forcing and
     verbalisation. The measurement does not change: K, T, the token budgets,
     the forcing call and the scores are all the same.
   - **The 3 traces already made at batch 4** are archived as
     `traces.batch4_superseded.jsonl` and regenerated, so the whole run uses
     one setting.
   - **Hashes:** the plan's SHA-256 above stays the hash of the plan as
     frozen. The analysis script is given the hash of this file as amended,
     and both are recorded in D78.
