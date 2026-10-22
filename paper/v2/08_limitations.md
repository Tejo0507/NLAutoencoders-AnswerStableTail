# 9. Limitations and threats to validity (draft v2, 2026-10-21)

**One model, one layer, one NLA.**
- Every result concerns Qwen2.5-7B-Instruct in 4-bit NF4 and the released
  layer-20 NLA. We found no NLA for another model or layer that would let us
  test generality.
- The bf16 check covered only the target's activations (D55), not the
  verbaliser or reconstructor.
- Reasoning models that verify in their prose may behave differently. The
  instruction-tuned model studied here rarely does (D62).

**Outcome-selected points.**
- Every NLA result except Stage 1 was measured at chunks chosen around the
  tail start (§3.4).
- The cheap-signal results were reproduced without that selection
  (Stage 0). Stage 1 does the same for the NLA [pending].

**The outcome depends on a budget.**
- Safety is defined by a 48-token forced answer, which is cut off for most
  pre-tail positions.
- The registered verdict uses it, and so do we. The 256-token measurement
  says how much the wording must narrow [pending]. It does not replace the
  registered outcome.
- A 256-token outcome still depends on a budget.

**Featurisation.**
- Explanations are represented with TF-IDF and a linear model, and the
  activation probes are linear.
- "No increment" means no increment that these models can extract. A
  stronger text model could find more, at a cost the comparison would have
  to charge.

**No faithfulness or semantic claim.**
- The phase vocabulary, and the test of whether explanations name the final
  answer, are string-level.
- They establish predictive content, not that an explanation describes the
  computation.
- No human rated the explanations (D87). The two-annotator protocol exists
  but was not run.

**Post-hoc analyses.**
- The falling-position rule, the same-activation probe, the multiplicity
  and whole-prefix checks, the unfitted ensemble and selective risk were
  added after related results were seen.
- Each was specified before any outcome existed for the replication, which
  is their test.

**Power.**
- The registered H1 has power of about 0.45 for the gap D69 estimated
  (D75 §5). An inconclusive H1 is not evidence of equivalence.
- Stage 1's power was audited on stored-data proxies (D85). Under the
  pessimistic proxy, 200 × 4 resolves only 40% of the time, which is why N
  was raised before any Stage 1 data existed.

**Engineering.**
- Batch sizes differ between `main` (4) and the replication (2) for
  continuations and forcing (D78). Only the random draws differ.
- The forcing instrument is checked for reproducibility before any outcome
  is used.

**Novelty is provisional.** The related-work review read abstracts and five
full texts (§2). Where we say prior work does not report something, we mean
the papers named.
