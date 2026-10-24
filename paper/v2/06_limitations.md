# 6. Limitations and threats to validity

*Revised candidate, 2026-10-24. From the claim matrix's weakest-point
column and §E.*

**One model, one layer, one NLA.** Every result concerns
Qwen2.5-7B-Instruct in 4-bit NF4 and the released layer-20 NLA; we found
no NLA for another model or layer with which to test generality. The bf16
check covered only the target model's activations, not the verbaliser or
reconstructor. Models that verify in prose, or that end a thinking phase
with a delimiter, may behave differently; the model studied here rarely
rechecks in prose and has no end-of-thinking token, so delimiter-based
signals [6, 18] were not compared. Traces are greedy with a single
generation seed, and the problems come from two public mathematical
benchmarks, which may appear in the model's training data. That would
affect every compared signal alike, but answers might settle earlier than
on genuinely new problems.

**Multiplicity.** Multiplicity is controlled only within pre-specified
families: Holm over the two confirmatory hypotheses, and within the
position audit's and the information audit's families. The many secondary
and exploratory intervals reported alongside them are not adjusted
jointly, so an isolated interval that excludes zero should be read as
descriptive. The paper's conclusions rest on the confirmatory test and on
the pre-specified analyses read by their fixed rules.

**Outcome-selected candidate points and position.** Every NLA stopping
comparison, including the confirmatory one, was made on design A's menu,
which contains the chunks on both sides of the answer change. This is why
the replication's advantage cannot be attributed to content (§4.2).
Stage 1 removes the selection for the information question, but no NLA
stopping *policy* was evaluated at outcome-blind positions.

**Dependence on the outcome and its budget.** The registered outcome is a
48-token forced answer, which is cut off for most pre-tail positions. The
registered verdict uses it and is not replaced, but under a 256-token
outcome the registered hypotheses are not supported, and those intervals
exclude only effects of 0.05 or more. A 256-token outcome still depends on
a budget: 9.1% of the re-forced cut-offs on the replication were still
cut off at 256 tokens. The token-count check failed on both arms, so net
savings were not used under the 256-token outcome, and the tail itself was
not recomputed under the longer budget.

**Prediction is not explanation.** The text features are TF-IDF with a
linear model, and the activation probes are linear. "No practically
relevant increment" means none that these models can extract, within
±0.01 AUC, for these labels; a stronger text model could find more, at a
cost the comparison would have to charge. Predictive information in an
explanation does not show that the explanation describes the computation.

**Limited evidence about faithfulness.** The phase lexicon, the
answer-mention test and the context checks against the trace are
string-level. No person rated the explanations; a two-annotator protocol
exists but was not run, and its blinded annotation sheets are withheld.
The reconstruction-dependence rate rests on a registered paraphrase null
that is mostly unchanged text. The exploratory rewording null changed the
rate little on 60 explanations, but its rewordings were checked for form,
not meaning, and only 117 of 246 claims received one. Reconstruction
dependence is not faithfulness.

**No established safe stopping policy.** The selective-risk results use
the outcome-selected menu and 48-token labels, and none exists for the
same-label probe on the replication. The planning replay that motivates a
stopping-policy study chose its thresholds in sample on four positions
per problem. No out-of-sample estimate of unsafe-stop risk exists for any
policy studied here.

**Post-hoc analyses and provenance.** The falling-position rule, the
same-label probe, the multiplicity and whole-prefix checks, the unfitted
ensemble and selective risk were added after related results on the main
run were seen. Each was specified before any replication outcome existed,
and the replication is their test. The main run's corrected comparison
(D69) has a provenance gap: the amended plan's hash was not recorded
before the data (D91).

**Independence and power.** The 240-problem arm contains the main run's
120 problems, so its results are exploratory and do not replicate the main
run independently; only the replication's 400 problems are fresh. The
registered H1 had a power of about 0.45 for the gap estimated on the main
run. H1 was supported, but its interval spans the 0.05 margin. Stage 1's
power was audited on proxies from stored data before any Stage 1 data
existed, and its sample was raised to all 400 problems in response.

**Engineering.** Batch sizes for continuations and forcing differ between
the main run (4) and the replication (2); only the random draws differ.
The forcing instrument was checked for reproducibility before any outcome
was used.

**A focused review.** The related-work review read abstracts and selected
full texts (§2). Where we say that prior work does not report something,
we mean the papers named.
