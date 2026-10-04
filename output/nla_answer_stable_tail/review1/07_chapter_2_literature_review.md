# Chapter 2 · Literature Review

## 2.1 Review of Existing Work

The literature relevant to this project sits in two historically separate
communities — *reasoning efficiency and faithfulness* and *internal
interpretability of LLMs* — that have only recently begun to intersect. This
section synthesises the state of each along six themes, closes each theme with
its limitation for the Answer-Stable Tail (AST) question, and ends with the
positioning of the proposed project. Each cited work is one of the ten papers
selected in `04_literature_comparison.md`; a small number of adjacent works are
discussed in prose for context and listed in the references. Preprints and
project reports are marked.

### 2.1.1 Chain-of-thought reasoning and its surface non-literality

Chain-of-thought prompting established the dominant interface for reasoning in
LLMs: a model is induced to emit a textual sequence of intermediate steps
before the answer [Wei et al. 2022]. CoT raises accuracy on multi-step tasks
and has become the computational substrate of subsequent work. However, the
same text that enables reasoning is **not a literal account** of the
computation. Turpin et al. (2023 preprint) show that CoT can be influenced by
information the model never acknowledges — hidden prompt features change the
answer without changing the stated reasoning. Lanham et al. (2023 preprint)
provide a measurement framework: deleting, resampling, or paraphrasing CoT
tokens often leaves the final answer unchanged, while some settings do produce
a measurable CoT-answer coupling. The consequence for AST is immediate:
*surface text cannot be used as a ground-truth label for internal state*. A
tail that contains the string "let me check" is not thereby a verification
event; the model's underlying computation may bear no literal relationship to
its words.

### 2.1.2 Overthinking, answer convergence, and dynamic early exit

A second strand of recent work directly measures the behavioural post-answer
regularity. Liu and Wang (EMNLP 2025) show that CoT traces can be chunked and
interrogated for intermediate answers; across five open models and five
reasoning benchmarks, stopping when consecutive chunks agree saves tokens
substantially with limited average accuracy loss. Caldarella et al. (2026
preprint) formalise the *first-correct prefix*: a trace position where the
extracted answer first matches the verifier label. They distinguish *verbose
overthinking* (harmless surplus) from *harmful overthinking* (post-correct
drift). Wu et al. (2026, ICLR) and the Sui et al. (2025 preprint) survey
consolidate this field, and Yang et al. (2026, ICLR) and Min et al. (2026
preprint) propose dynamic early-exit mechanisms tied to specific reasoning
models.

These works establish that *the tail is real* and *behaviourally measurable*;
they do not establish what the tail *is* at a representational level. They
also share two common limitations: answer agreement is a necessary but not
sufficient condition for *safe* stopping (a stable agreement can precede a
missed constraint), and none of them pair early-exit decisions with
matched-control causal interventions on the suffix.

### 2.1.3 Hidden-state probing and the availability–use distinction

A third strand asks what reasoning-model activations themselves know. Burns
et al. (2022/2023 preprints) show that latent-knowledge directions can be
recovered without supervision. For reasoning specifically, Zhang et al. (COLM
2025) train probes on intermediate-answer-position activations: correctness
is linearly decodable, in-distribution AUC exceeds 0.7 (as reported), and
probe-guided early exit reduces tokens with calibrated scores, though
cross-domain transfer degrades. The classical interpretability literature on
causal tracing (Meng et al. 2022; Goldowsky-Dill et al. 2023 preprint)
emphasises an important distinction: a decodable direction is evidence of
*availability* of information, not evidence that the model *uses* the
direction. Converting a probe into a mechanistic claim requires activation
patching with matched-random-direction, matched-position, and dose-response
controls — the methodological standard that Goldowsky-Dill et al. 2023
spells out.

The implication for AST is that any proposed internal "verification signal"
must clear two bars: decoded with sufficient AUC in-distribution *and*
survive causal interventions with matched controls. The reasoning-tail
literature has met the first bar; it has not yet met the second.

### 2.1.4 Uncertainty quantification as a cheap but strong baseline

A fourth strand provides the comparator any new tail-stopping method must
beat. Farquhar et al. (2024, *Nature*) propose semantic entropy, in which
multiple model generations are clustered by meaning and entropy is computed
over clusters; this outperforms lexical entropy and several confidence
baselines for confabulation detection across the models and tasks they
report. Semantic entropy measures *outcome uncertainty* rather than *tail
type*, so it is not itself a verification detector — but it is the current
best cheap signal with which to compare one. Related process-reward work
(Lightman et al. 2023 preprint; Setlur et al. 2024 preprint) and
self-consistency confidence further populate this baseline family.

### 2.1.5 Self-correction and the risk of romanticising the tail

A fifth strand examines whether LLMs can intrinsically correct their own
reasoning. The current consensus is mixed. Huang et al. (ICLR 2024) show in
controlled studies that *intrinsic* repeated self-correction often fails or
degrades accuracy, and that many prior gains collapse under controlled
conditions. Self-Refine (Madaan et al. 2023, NeurIPS) and Reflexion (Shinn
et al. 2023, NeurIPS) report gains, but typically depend on external
feedback or specific prompting. The TACL survey (Liu et al. 2024)
consolidates the regime. The implication for AST is a *prior of
skepticism*: a suffix that superficially looks like "checking the answer"
should not be assumed to improve or even to be independent of the final
answer, and no project should promote a "verification-looking" surface
label into a positive outcome without external evidence.

### 2.1.6 Mechanistic interpretability on the residual stream

A sixth strand concerns the methods available to read, isolate, and
manipulate information in the residual stream. Sparse autoencoders have
become a mature quantitative method for isolating interpretable directions
[Cunningham et al. 2023 preprint; Templeton et al. 2024 Anthropic project
report; Marks et al. 2024 preprint, feature circuits]. Causal tracing /
activation patching [Meng et al. 2022; Goldowsky-Dill et al. 2023 preprint]
remains the methodological reference for intervention-based claims. Into
this landscape, Fraser-Taliente et al. (2026, Transformer Circuits
project report) introduce **Natural Language Autoencoders**: a verbaliser
(AV) that samples text from a residual-stream state and a reconstructor
(AR) that maps text back to a vector. Joint training with squared
reconstruction loss and a KL constraint, warm-started from context
summaries, produces AV outputs that preserve a substantial fraction of
variance. Released AV/AR pairs exist for Qwen2.5-7B-Instruct L20,
Gemma-3-12B/27B-IT, and Llama-3.3-70B-Instruct at specified layers. The
authors themselves document confabulation, layer sensitivity, decoder
prior, and ~500 generated tokens per activation, framing the method as a
*hypothesis generator*.

Almost immediately, this method encountered a direct challenge. Dingeto
(2026 preprint, "RECAP") shows that reconstruction can remain high while
individual AV claims are not reconstruction-dependent — i.e., the gist is
preserved but specific statements are not individually constrained by the
vector. The paper's constructive proposal is to co-train the *target
model* with linear decodability heads, so that pre-specified facts remain
auditable against a fresh probe. RECAP therefore *supervises* faithfulness
for designated variables; it does not retroactively validate free-form AV
prose on released NLA checkpoints.

### 2.1.7 Where these themes converge — the Answer-Stable Tail

The AST question sits at the intersection of all six strands. The tail is
*behaviourally identifiable* (§2.1.2), *representationally non-trivial*
(§2.1.3), *cheap-signal-measurable* (§2.1.4), *not safely text-labelled*
(§2.1.1), *not presumptively useful computation* (§2.1.5), and
*accessible to a free-form language readout whose faithfulness is
contested* (§2.1.6). None of the surveyed works, individually or in
combination, has:

- defined the AST without reference to NLA output;
- compared an NLA readout against semantic entropy, hidden-state probes,
  and answer-agreement at matched compute on safe stopping;
- subjected AV sentences on tails to the Lanham et al. (2023 preprint)
  deletion / resample / paraphrase template;
- or applied matched-control activation interventions
  [Goldowsky-Dill et al. 2023 preprint] to the tail windows.

This is the composite that the proposed project is designed to supply.

### 2.1.8 Positioning of the proposed project

Fraser-Taliente et al. (2026) is used as *the method*; the project does not
propose a new NLA. Liu and Wang (2025), Zhang et al. (2025), Caldarella
et al. (2026 preprint), and Farquhar et al. (2024) are used as the
*comparators and operational scaffolding*. Dingeto (2026 preprint),
Turpin et al. (2023 preprint), Lanham et al. (2023 preprint), and Huang
et al. (2024) are used as *prior-of-skepticism constraints* that shape the
methodology. Goldowsky-Dill et al. (2023 preprint) governs the *causal
validation ladder*. The project's contribution is a composite protocol
and the first head-to-head benchmark, not a new interpretability method.

## 2.2 Research Gap Identification

### 2.2.1 Established findings

The ten-paper comparison in `04_literature_comparison.md` establishes:

1. **Behavioural fact.** Reasoning models often produce suffixes after
   their answer is behaviourally stable [Liu and Wang 2025, EMNLP;
   Caldarella et al. 2026 preprint].
2. **Representational fact.** Correctness information is linearly
   decodable from reasoning-model activations with usable
   in-distribution AUC [Zhang et al. 2025, COLM].
3. **Uncertainty fact.** Semantic entropy is a strong meaning-level
   uncertainty signal [Farquhar et al. 2024, Nature].
4. **Faithfulness fact.** CoT text is not a literal account of the
   computation [Turpin et al. 2023 preprint; Lanham et al. 2023
   preprint].
5. **Correction fact.** Intrinsic repeated self-correction is often
   neutral or harmful under controlled conditions [Huang et al. 2024,
   ICLR].
6. **Readout fact.** Natural Language Autoencoders can produce free-
   form, activation-conditioned descriptions of residual-stream states
   with substantial reconstruction fidelity, under explicit
   confabulation and layer-sensitivity caveats [Fraser-Taliente et al.
   2026, project report].
7. **Faithfulness-of-readout fact.** High reconstruction does not imply
   claim-level faithfulness of AV sentences; supervised auditability
   requires target-model co-training (RECAP) [Dingeto 2026 preprint].
8. **Causal methodology fact.** Any causal claim on residual-stream
   state requires matched-control activation interventions
   [Goldowsky-Dill et al. 2023 preprint].

### 2.2.2 Limitations persisting across these works

- **Limitation A (operational).** No work in §2.1 defines the AST without
  reference to NLA output, under K-continuation + verifier controls.
- **Limitation B (comparative).** No work in §2.1 compares an NLA-derived
  signal against semantic entropy, hidden-state correctness probes, and
  answer-convergence stopping at matched compute budgets on verified
  reasoning tasks.
- **Limitation C (faithfulness-audit).** No work in §2.1 applies the
  Lanham-template to AV sentences on reasoning-tail windows; Dingeto
  2026 is the general challenge but is not applied here.
- **Limitation D (causal).** No work in §2.1 subjects the putative
  verification tail to matched-control activation interventions in the
  methodology of Goldowsky-Dill et al. 2023.
- **Limitation E (labels).** No work in §2.1 provides a blinded,
  two-rater, Krippendorff-α-reported semantic annotation for suffix
  type (verification / restatement / derivation / correction) on
  reasoning traces.

### 2.2.3 Unresolved research issue

Taken together, these limitations state a specific unresolved issue:

> **It is not currently known whether an NLA-derived readout of residual-
> stream activations carries safe-stopping information for reasoning
> LLMs that is not already available from cheap uncertainty, probe, or
> agreement signals, nor whether any such information corresponds to a
> causally meaningful "redundant verification" state as distinct from
> answer stability itself.**

### 2.2.4 Evidence that this is a gap rather than mere absence

The gap is *not* an "it has not been done yet" claim. It is supported by
positive evidence:

- Fraser-Taliente et al. 2026 themselves list confabulation, decoder
  prior, and layer sensitivity as unresolved risks, and they do not
  evaluate on safe-stopping tasks.
- Dingeto (2026 preprint) predicts that claim-level faithfulness of
  free-form AV text will not follow from reconstruction fidelity and
  demonstrates this in controlled runs.
- Huang et al. (2024, ICLR) and the CoT-faithfulness literature
  [Turpin 2023 preprint; Lanham 2023 preprint] independently rule out
  the two naive shortcuts (treating a "verification-looking" suffix as
  positive label; treating CoT as literal).
- The strongest cheap baselines (Liu and Wang 2025; Zhang et al. 2025;
  Farquhar et al. 2024) have not been compared against NLA at matched
  budget on this task.

Each of these is independent of the others; together they are evidence
that the gap is substantive, not merely a gap in effort.

### 2.2.5 Classification of the gap

Using the gap-type schema of the prompt and the limitation matrix in
`05_research_gap.md`:

- **Methodological gap.** A composite protocol combining independent AST
  definition, blinded labels, truncation / K-continuation controls, and
  Goldowsky-Dill-grade activation interventions is **absent**.
- **Evaluation gap.** A head-to-head benchmark of NLA against
  semantic-entropy, hidden-state probe, and answer-convergence
  baselines on safe stopping is **absent**.
- **Causal gap.** No matched-control activation-intervention evidence
  on verification-like directions exists in the reasoning-tail
  setting.
- **Faithfulness gap.** No claim-level audit of AV sentences on
  reasoning tails has been performed.

The gap is **not** an implementation gap (releasing yet another NLA for
GSM8K would address none of the above), **not** a mere application gap
(simply running NLA on reasoning tasks would still leave the comparison
and causal layers missing), and **not** a theoretical gap in the sense
of a missing formal framework. It is primarily *methodological* and
*evaluation*, with a secondary *causal* and *faithfulness* component.

### 2.2.6 Proposed research direction

The project proposes to:

- Define AST operationally and NLA-independently (chunk-level answer
  equivalence under truncation and K-continuation, with matched-position
  controls).
- Run NLA (AV + AR, released Qwen2.5-7B-Instruct L20 checkpoint) at AST
  candidate windows, alongside semantic entropy, hidden-state
  correctness probes, and answer-convergence stopping, on verified
  GSM8K and MATH.
- Audit a stratified AV sample under the Lanham deletion / resample /
  paraphrase template, scored by AR reconstruction change.
- Run a causal-validation ladder — truncation, tail replacement,
  matched-random-direction, matched-position, and dose-response
  controls — on the AST window.
- Report paired bootstrap confidence intervals by problem, Krippendorff
  α on all semantic labels, matched-control results on every
  intervention, and both negative and positive outcomes with equal
  weight.

### 2.2.7 Honest statement of risk

If, in Phase 1, Liu and Wang (2025) alone matches the NLA-derived AST
detector within confidence intervals at matched budget, the honest
conclusion will be that the project contributes an evaluation protocol
and a semantic interface rather than a new detector. This is a
scientifically useful outcome and is explicitly provided for by the
project's publication strategy.

### References

See the reference list of `06_chapter_1_introduction.md`. Only the
papers used in Chapter 2.1 directly are repeated below in a thematic
grouping to aid the examiner. All URLs/DOIs are as recorded in the
project paper database on 2026-09-25; preprint URLs have not been
independently re-fetched for Review 1 and are flagged accordingly.

**Natural Language Autoencoders and faithfulness.**
Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). *Natural
Language Autoencoders Produce Unsupervised Explanations of LLM
Activations*. Transformer Circuits (project report).
Dingeto, H. (2026). *Train the Model, Not the Reader: Decodability
Supervision for Verifiable Activation Explanations* (preprint).

**Answer convergence, overthinking, dynamic early exit.**
Liu, Y., & Wang, X. (2025). *Answer Convergence as a Signal for Early
Stopping in Reasoning*. EMNLP.
Caldarella, S., et al. (2026). *Thinking Past the Answer* (preprint).
Yang, Z., et al. (2026). *Dynamic Early Exit in Reasoning Models*.
ICLR.
Min, J., et al. (2026). *Stop When Reasoning Converges* (preprint).
Wu, Y., et al. (2026). *When More is Less: Understanding
Chain-of-Thought Length in LLMs*. ICLR.
Sui, R., et al. (2025). *Stop Overthinking: A Survey on Efficient
Reasoning for Large Language Models* (preprint).

**Hidden-state probing and uncertainty.**
Zhang, A., Chen, Y., Pan, J., Zhao, C., Panda, A., Li, J., & He, H.
(2025). *Reasoning Models Know When They're Right*. COLM.
Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting
hallucinations in large language models using semantic entropy*.
Nature. DOI 10.1038/s41586-024-07421-0.

**CoT faithfulness and self-correction.**
Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023).
*Language Models Don't Always Say What They Think* (preprint).
Lanham, T., et al. (2023). *Measuring Faithfulness in Chain-of-Thought
Reasoning* (preprint).
Huang, J., et al. (2024). *Large Language Models Cannot Self-Correct
Reasoning Yet*. ICLR.
Madaan, A., et al. (2023). *Self-Refine*. NeurIPS.
Shinn, N., et al. (2023). *Reflexion*. NeurIPS.
Liu, Y., et al. (2024). *When Can LLMs Actually Correct Their Own
Mistakes?* TACL.

**Causal interpretability and sparse representations.**
Meng, K., et al. (2022). *Locating and Editing Factual Associations in
GPT*. NeurIPS.
Goldowsky-Dill, N., et al. (2023). *Towards Best Practices of
Activation Patching in Language Models* (preprint).
Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly
Interpretable Directions in Language Model Residual Stream Space*
(preprint).
Templeton, A., et al. (2024). *Scaling Monosemanticity* (Anthropic
project report).
Marks, S., et al. (2024). *Sparse Feature Circuits* (preprint).

**Benchmarks (background only, not entries in the comparison table).**
Hendrycks, D., et al. (2021). *MATH*. NeurIPS.
Cobbe, K., et al. (2021). *GSM8K* (preprint).
Rein, D., et al. (2023). *GPQA*. COLM.
Chen, M., et al. (2021). *HumanEval* (preprint).
Suzgun, M., et al. (2022). *BBH* (preprint).

**Methodological acknowledgement.**
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents* (preprint). arXiv:2609.00065. The research-workflow
tooling (research-planning, deep-research, scientific-writing,
related-work-writing, novelty-assessment, peer-review, self-review,
citation-management, table-generation, scientific-slides) used to
prepare this review is part of this library.
