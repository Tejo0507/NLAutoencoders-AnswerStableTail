# 6. Limitations and threats to validity (draft v3, 2026-10-23)

*From `paper/CLAIM_EVIDENCE_MATRIX.md` (weakest-point column and §E).*

**One model, one layer, one NLA.**
- Every result concerns Qwen2.5-7B-Instruct in 4-bit NF4 and the released
  layer-20 NLA. We found no NLA for another model or layer that would let us
  test generality (E3).
- The bf16 check covered only the target's activations (D55), not the
  verbaliser or reconstructor.
- Reasoning models that verify in their prose, or that end thinking with a
  delimiter, may behave differently. The instruction-tuned model studied
  here rarely rechecks in prose (D62) and has no end-of-thinking token, so
  delimiter-based stopping signals [35, 36] were not compared.
- Traces are greedy with one generation seed. Only the stochastic stages
  vary with a seed.

**Outcome-selected points.**
- Every NLA stopping comparison, including the confirmatory one, was
  measured on design A's menu, which contains the chunks on both sides of
  the answer change (§3.4). This is why the replication's advantage cannot
  be attributed to content (R2).
- Stage 1 removed the selection for the *information* question (R6), but
  no NLA stopping *policy* was evaluated at outcome-blind positions.

**The outcome depends on a budget.**
- The registered outcome is a 48-token forced answer, which is cut off for
  most pre-tail positions. The registered verdict uses it, and that verdict
  is not replaced.
- The 256-token sensitivity analysis (D82) changes the reading: under it,
  the registered hypotheses are not supported (R4). Its intervals exclude
  effects of 0.05 or more, not smaller ones.
- A 256-token outcome still depends on a budget: 9.1% of the re-forced
  cut-offs on rep1 were still cut off at 256 (D95).
- The token-count check failed on both D82 arms (maximum differences of 4
  and 5 tokens, against ±2), so net savings were not used under O256, and
  every O256 comparison uses registered savings.
- The tail itself was not recomputed under the longer budget (A1).

**Featurisation.**
- Explanations are represented with TF-IDF and a linear model, and the
  activation probes are linear.
- "No practically relevant increment" means none that these models can
  extract, within ±0.01 AUC, for these labels. It does not show that
  everything the text says is linearly available in the activation. A
  stronger text model could find more, at a cost the comparison would have
  to charge.

**No faithfulness or semantic claim (E1).**
- The phase vocabulary and the answer-mention test are string-level. The
  context checks (C7) compare strings with the trace.
- They establish predictive content, not that an explanation describes the
  computation.
- No human rated the explanations (D87). The two-annotator protocol exists
  but was not run, and the blinded annotation sheets are withheld.
- The reconstruction-dependence rate (C6, 0.749) rests on a registered
  paraphrase null that is mostly no edit or an added discourse marker
  (D70). Reconstruction dependence is not faithfulness.

**Stopping at deployment is untested (E4).**
- The selective-risk results (R7) use the outcome-selected menu and
  48-token labels. No selective-risk result exists for the same-label probe
  on rep1.
- The CPU planning replay that motivates a stopping-policy study chose its
  thresholds in sample and walked four positions per problem. It is
  planning evidence only, not evidence of out-of-sample safety.

**Post-hoc analyses.**
- The falling-position rule, the same-activation probe, the multiplicity
  and whole-prefix checks, the unfitted ensemble and selective risk were
  added after related results on the main run were seen.
- Each was specified before any outcome existed for the replication, which
  is their test.
- The main run's corrected O3 analysis (D69) has a provenance gap: the
  amended plan's hash was not recorded before the data (D91).

**Independence.** The 240-problem arm contains the main run's 120
problems, so its results are exploratory and do not replicate the main run
independently. Only rep1's 400 problems are fresh.

**Power.**
- The registered H1 had power of about 0.45 for the gap D69 estimated
  (D75 §5). H1 was supported, but its interval spans the 0.05 margin, so an
  advantage of 0.05 or more is neither shown nor excluded (R1).
- Stage 1's power was audited on stored-data proxies before any Stage 1
  data existed (D85), and N was raised to all 400 problems in response.

**Engineering.**
- Batch sizes differ between `main` (4) and the replication (2) for
  continuations and forcing (D78). Only the random draws differ.
- The forcing instrument was checked for reproducibility before any
  outcome was used (0.965 on 200 re-forced boundaries for rep1).

**Novelty is bounded by a focused review.** The related-work review read
abstracts and selected full texts (§2). Where we say prior work does not
report something, we mean the papers named.
