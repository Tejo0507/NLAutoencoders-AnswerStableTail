# 01 · Project Research Profile

**Working title (repository):** NLAutoencoders-AnswerStableTail
**Proposed research-oriented title:** *Beyond Readable Hypotheses: An NLA-Independent, Causally Validated Benchmark for the Answer-Stable Tail in LLM Reasoning*
**Short handle:** NLA-AST

**Compiled:** 2026-10-03.
**Status:** Review 1 preparation. The repository currently contains data staging and prior-research notes; no experimental pipeline exists yet. All numeric claims in this profile originate from the referenced literature or from direct repository inspection, not from in-house experiments.

---

## 1. Domain and sub-domain

- **Primary domain.** Machine learning / natural language processing.
- **Sub-domain 1.** Mechanistic / internal interpretability of large language models (LLMs) — specifically, methods that attempt to render residual-stream activations in human-readable form.
- **Sub-domain 2.** Efficient and faithful reasoning in LLMs — specifically, chain-of-thought (CoT) termination, answer convergence, overthinking, and self-correction.
- **Methodological neighbourhood.** Activation patching and sparse autoencoders (SAEs); hidden-state correctness probes; uncertainty quantification; verifier-based evaluation of mathematical reasoning.

## 2. Problem under investigation

LLMs frequently continue generating text *after* their answer has already settled. In some cases these suffixes are harmless restatement ("verbose overthinking"); in other cases they derail an already-correct trajectory ("harmful overthinking"). The repository asks whether this post-answer suffix — the **Answer-Stable Tail (AST)** — is a distinct cognitive event (e.g. "redundant self-verification"), whether its internal signature can be *read out* from activations by a Natural Language Autoencoder (NLA), and whether this reading can enable safe early termination.

The scientific question therefore has three nested parts:

- **Q1 (phenomenological).** Does an AST exist as a reliably identifiable regime, operationalisable without reference to NLA output?
- **Q2 (representational).** Do model activations in the AST window carry information distinct from entropy, answer-token confidence, or a correctness probe?
- **Q3 (causal / applicative).** Does an NLA-derived readout add non-trivial, causally supported value for safe stopping over cheap baselines (truncation + K-continuation agreement, semantic entropy, correctness probes)?

## 3. Proposed approach (as reconstructed from the repository and prior-research notes)

The project proposes:

1. Generate reasoning traces on GSM8K and MATH with a model that has a *released* NLA checkpoint — primarily Qwen2.5-7B-Instruct at layer 20 (the only released NLA for which training-scale replication is affordable).
2. Define candidate ASTs via an **NLA-independent** criterion (answer-equivalence under truncation plus K sampled continuations).
3. Extract residual-stream activations across the candidate tail.
4. Apply the NLA activation verbaliser (AV) and activation reconstructor (AR) to produce language-conditioned descriptions and reconstruction fidelity (FVE).
5. Compare the NLA-derived signal to strong cheap baselines (semantic entropy, answer-agreement, hidden-state correctness probes, token-position/length heuristics) on the task of predicting a *safe stopping point*.
6. Test causal validity via truncation, matched-position controls, tail replacement, activation patching, and (optionally) NLA-edit steering.

**Inputs.** A frozen prompt and decoding configuration; reasoning traces and token-level log probabilities; residual-stream activations at the released layer; verifier labels for GSM8K/MATH; two trained annotators for blinded semantic labels.
**Outputs.** A labelled trace corpus; comparative accuracy–compute curves; incremental-value tests for NLA relative to baselines; a causal-validation ladder report.
**Assumptions.** Compatible NLA checkpoint; stable prompt matrix; verifier reliability on test items; feasibility of two-rater semantic annotation.

## 4. What is implemented, what is not

**Implemented in the repository.**
- `scripts/download_datasets.py` — downloads GSM8K and the MATH subjects into `data/raw/`.
- `data/raw/gsm8k/{train,test}.jsonl` — full GSM8K.
- `data/raw/math/{train,test}/{algebra,counting_and_probability}.jsonl` — two of the seven MATH subjects.
- `output/nla_answer_stable_tail/phase{1..6}` — a six-phase prior-research artifact dated 2026-09-25: frontier scan, survey, deep-reading notes, code/checkpoint landscape, cross-paper synthesis, and a 28 KB methodological critique and roadmap (`phase6_report/report.md`).
- `requirements.txt` — currently names only `datasets` and `huggingface_hub`.

**Not implemented.**
- Model inference, prompt matrix, decoding configuration, EOS policy.
- Activation extraction, hooks, caching schema.
- NLA loading, AV/AR invocation.
- Semantic-entropy and probe baselines.
- Verifier integration (math answer parsing, symbolic equivalence).
- Any causal-intervention code (truncation, patching, steering).
- Any annotation protocol, inter-rater reliability tooling, or logs.
- Any evaluation harness or statistical analysis scaffolding.
- Dependencies for all of the above are missing from `requirements.txt`.

**Supported-by-code vs documentation-only claims.** No scientific claim in the repository is currently supported by in-house experimental code. Every methodological commitment is at the design-document stage.

## 5. Evaluation surface available today

- **Datasets.** GSM8K (full); MATH-algebra and MATH-counting-and-probability only. The downloader is configured for seven MATH subjects; five remain uncollected.
- **Metrics planned.** Verified final-answer accuracy; tokens / FLOPs saved; semantic-entropy; probe AUC; paired bootstrap CIs by problem; annotation Krippendorff α.
- **Baselines planned.** Static length cutoff; token-entropy threshold; semantic entropy (Farquhar 2024); self-consistency; answer-position logit margin; linear correctness probe (Zhang 2025); answer-convergence stopping (Liu & Wang 2025).
- **Experiments planned but not yet run.** All of them.

## 6. Claimed contribution — audited before literature test

The repository's intuitive framing ("NLAs read the internal self-verification state and let us stop early") decomposes into claims of varying scientific weight. The following table is an **internal audit** of those claims against the current literature *before* producing the formal research gap in `05_research_gap.md`; the gap section tests them against the final paper comparison.

| Candidate claim | Audit status | Justification |
|---|---|---|
| Reasoning models often produce suffixes after the answer is behaviorally stable. | **Already established** | Liu & Wang 2025; Caldarella et al. 2026; Zhang et al. 2025. Not a novel contribution. |
| NLAs can supply activation-conditioned, human-readable hypotheses about hidden states. | **Established, with important caveats** | Fraser-Taliente et al. 2026 (project report). The authors themselves document confabulation and layer sensitivity; not a claim of ground-truth decoding. |
| NLA explanations are *faithful* at the individual-claim level. | **Contested by recent work** | Dingeto 2026 (preprint, RECAP) directly shows that high reconstruction need not make individual claims reconstruction-dependent. Cannot be claimed without new evidence. |
| There exists an internal "redundant self-verification" state detectable from activations. | **Not established** | No current paper demonstrates a distinct cognitive mechanism, independent of answer-token identity, position, length, or decoder prior. |
| NLA output can enable *safe* early termination better than cheap baselines. | **Not established** | No head-to-head comparison exists against semantic entropy, correctness probes, or answer-agreement at matched compute budgets. |
| An NLA-independent, causally validated AST benchmark exists in the literature. | **Does not exist** | No published benchmark combines independent AST definition, blinded semantic labels, truncation/continuation evidence, and matched-control activation interventions. |

**Scientifically defensible candidate contribution** (to be tested against the final 10-paper comparison before being asserted): a *methodological contribution* consisting of (i) the first NLA-independent operational definition of the Answer-Stable Tail; (ii) a reproducible evaluation protocol combining behavioural, representational, and causal evidence; and (iii) a direct, matched-budget comparison of NLA-derived readouts against entropy, probe, and agreement baselines on GSM8K / MATH, treating NLA as a blinded hypothesis generator rather than a detector.

## 7. Honest statement of risk

- The NLA checkpoint is *exact* to model-and-layer. The target model is therefore effectively fixed to the released set (Qwen2.5-7B L20 for near-term work).
- Semantic labels for "verification vs. restatement vs. derivation" may fail inter-rater reliability thresholds; this would be a finding, not a failure, and would narrow the scientific claim.
- A plausible outcome is that cheap baselines match or exceed NLA on safe-stopping performance. If so, the project must be re-cast as *a semantic interface over equivalent detectability*, not as a new detector.

## 8. Research-integrity notes relevant to this profile

- Several papers cited above are preprints (Fraser-Taliente et al. 2026 is a Transformer Circuits project report; Dingeto 2026, Caldarella et al. 2026 are arXiv preprints). They are marked as such throughout and are not treated as settled evidence.
- No numeric experimental result in this profile is in-house. In particular, no FVE, probe-AUC, token-savings, or accuracy figure is produced by code in this repository as of 2026-10-03.
- The 2026-09-25 phase-reports in `output/nla_answer_stable_tail/` are the sole source of recent-literature claims used here; where this profile goes beyond those reports, it is clearly flagged.
