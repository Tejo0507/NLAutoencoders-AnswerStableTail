# Chapter 1 · Introduction

## 1.1 Background of the Study

Large language models (LLMs) have become the dominant architecture for complex
reasoning tasks, from grade-school arithmetic in GSM8K [Cobbe et al. 2021] through
competition-level problems in the MATH benchmark [Hendrycks et al. 2021] to
expert-level question answering on GPQA [Rein et al. 2023]. The core interface
through which these systems reason is the *chain-of-thought* (CoT): the model is
induced to emit an explicit sequence of intermediate steps before producing an
answer [Wei et al. 2022]. CoT prompting materially improves accuracy on multi-step
problems, and reasoning-specialised models — including those trained with
reinforcement learning on verifiable rewards, e.g. DeepSeek-R1 [DeepSeek-AI 2025] —
have raised the frontier further.

With this capability, however, has come a well-documented inefficiency: reasoning
models frequently continue generating tokens *after* their answer has already
settled. A substantial recent literature has formalised this behaviour.
Liu and Wang (EMNLP 2025) show that CoT traces can be segmented into chunks whose
intermediate answers converge well before generation ends, and that stopping at
the convergence point preserves most of the final-answer accuracy while cutting
tokens substantially. Zhang et al. (COLM 2025) show that correctness information
is linearly decodable from reasoning-model hidden states, so that a cheap probe
can gate early exit. Caldarella et al. (2026 preprint) separate the suffix into
*verbose overthinking* — harmless restatement after a first-correct prefix — and
*harmful overthinking*, in which later tokens drift away from a correct answer.
Together, these results establish that the *post-answer suffix* in reasoning
traces is a real, measurable regularity.

What these studies do **not** establish is *why* a reasoning model produces the
suffix. One intuitively compelling candidate is that the model is performing a
form of internal self-verification — a redundant check on an answer it has
already computed. If this were literally true, and if a readout of the model's
hidden state could *report* the check, two things would follow: an
interpretability advance (an internal cognitive state becomes readable) and a
practical advance (the suffix becomes safe to truncate on a trace-by-trace
basis). The hypothesis is attractive precisely because it couples a scientific
claim to a deployment-relevant control.

The emergence of **Natural Language Autoencoders (NLAs)** in 2026 appears to make
this hypothesis testable. Fraser-Taliente et al. (2026, Transformer Circuits
project report) train a pair of networks: an *activation verbaliser* (AV) that
samples free-form text from a model's residual-stream activation, and an
*activation reconstructor* (AR) that maps text back to a vector. Joint training
with a reconstruction loss and a KL constraint produces AV outputs that preserve
a substantial fraction of the activation's information, as measured by fraction
of variance explained (FVE). The authors release AV/AR checkpoints for
Qwen2.5-7B-Instruct, Gemma-3-12B/27B-IT, and Llama-3.3-70B-Instruct at specific
layers. On the face of it, this seems to supply the missing instrument: an LLM
whose internal states can be read in natural language.

However, the same literature that makes this project attractive has also moved
quickly to constrain it. Dingeto (2026 preprint, "RECAP") shows that
reconstruction fidelity does not imply that individual AV sentences are *claim-
level* faithful — a verbaliser can preserve the gist of an activation while
making unaudited statements about specific contents. Independent CoT-faithfulness
work — Turpin et al. (2023 preprint) and Lanham et al. (2023 preprint) — shows
that textual reports produced by LLMs are often not literal accounts of the
underlying computation. Huang et al. (ICLR 2024) show that *intrinsic*
self-correction, under controlled conditions, often fails to improve or even
degrades reasoning accuracy. And classical interpretability guidance
[Goldowsky-Dill et al. 2023 preprint] insists that any *causal* claim about an
internal state requires matched-control interventions, not merely decodability
evidence.

The scientific situation at the time of this proposal is therefore unusual.
The relevant pieces exist — released NLA checkpoints, strong cheap baselines
for early stopping, a formal overthinking taxonomy, a methodology for causal
interventions on residual states — but they have not been **combined** into a
study that could decide whether an NLA-derived readout actually adds safe-
stopping value over cheap signals, nor whether a putative *redundant self-
verification tail* has any causal reality distinct from answer stability
itself. This absence motivates the present study.

## 1.2 Problem Statement

The specific research-and-engineering problem addressed by this project is the
following. Reasoning LLMs produce suffixes after their answer has become
behaviourally stable; the literature does not currently settle whether these
suffixes carry any detectable, causally relevant *internal verification signal*,
nor whether reading such a signal via natural-language activation
interpretation adds value over the cheap uncertainty, probe, and answer-
agreement signals already known to work. Four concrete limitations persist:

1. **No NLA-independent operational definition of the Answer-Stable Tail
   (AST).** Caldarella et al. (2026 preprint) define a first-correct prefix,
   but their boundary is sensitive to the answer parser and to lucky early
   answers; Liu and Wang (2025) use answer agreement, but agreement is
   necessary rather than sufficient for *safe* stopping.
2. **No comparison of NLA readouts against cheap baselines on safe stopping.**
   Fraser-Taliente et al. (2026) report AV/AR quality and audit case studies;
   they do not run NLA against semantic entropy [Farquhar et al. 2024, Nature]
   or against hidden-state correctness probes [Zhang et al. 2025, COLM] at
   matched compute on the stopping task.
3. **No claim-level faithfulness audit of AV sentences in a reasoning-tail
   setting.** Dingeto (2026 preprint) shows this audit is non-trivial; the
   deletion / resample / paraphrase template of Lanham et al. (2023 preprint)
   has not been applied to AV outputs on reasoning traces.
4. **No causal validation of the putative verification tail with matched
   controls.** Early-exit studies [Liu and Wang 2025; Yang et al. 2026, ICLR;
   Min et al. 2026 preprint] report accuracy–token trade-offs; none run
   tail-replacement or matched-control activation interventions in the style
   recommended by Goldowsky-Dill et al. (2023 preprint).

The practical consequence of this unresolved state is that early-stopping policies
are deployed today either on heuristic length cutoffs or on behavioural proxies
whose relationship to internal state is unknown. The scientific consequence is
that strong interpretability claims — "the model is internally verifying its
answer" — are being made or implied on the basis of textual appearance alone,
without faithfulness or causal evidence.

## 1.3 Aim and Objectives

### Aim

To build and empirically test a pre-registered, NLA-independent, causally
validated protocol for evaluating whether natural-language readouts of
residual-stream activations add safe-stopping value in reasoning LLMs, and to
compare NLA against strong cheap baselines on this task.

### Objectives

The objectives below are each connected to a problem-statement limitation
(L1–L4), to the research gap of Chapter 2.2, to a method, and to an
evaluation. They are traced in `08_research_traceability.md`.

**Objective 1 (addresses L1).** Define the Answer-Stable Tail
operationally without reference to NLA output, using (i) chunk-level answer
equivalence under truncation, (ii) K independently sampled continuations with a
verifier, and (iii) matched token-position / length controls on GSM8K and MATH.

**Objective 2 (addresses L2).** Implement a reproducible pipeline that
loads the released Qwen2.5-7B-Instruct L20 NLA (Fraser-Taliente et al. 2026)
and emits AV text + AR vectors at the AST candidate windows, with versioned
decoding configuration and activation hashes.

**Objective 3 (addresses L2 and L4).** Measure the *incremental value* of the
NLA-derived signal over semantic entropy [Farquhar et al. 2024], hidden-state
correctness probes [Zhang et al. 2025], and answer-convergence stopping
[Liu and Wang 2025] on safe-stopping accuracy vs. tokens-saved at matched
compute, with paired bootstrap confidence intervals by problem.

**Objective 4 (addresses L3).** Audit a stratified sample of AV sentences
using the deletion / resample / paraphrase protocol of Lanham et al.
(2023 preprint), scored against AR reconstruction change, to produce
claim-level faithfulness statistics for AV outputs on reasoning tails.

**Objective 5 (addresses L4).** Run a causal-validation ladder — truncation,
tail replacement, matched-random-direction and matched-position controls, and
activation patches — in the methodology of Goldowsky-Dill et al. (2023
preprint), to test whether removing or altering the AST changes verified
behaviour.

**Objective 6 (reporting).** Release the labelled AST corpus, the evaluation
harness, and a preregistered analysis plan with Krippendorff α on all
semantic labels, so that incremental follow-up (RECAP-supervised NLAs; other
model families; code / expert QA tasks) can be run against a stable baseline.

### Primary Research Question

> Does an NLA-derived readout of residual-stream activations provide
> incremental, causally supported value for safe stopping in reasoning LLMs
> on GSM8K and MATH, over strong cheap baselines (semantic entropy,
> hidden-state correctness probes, answer-convergence stopping)?

### Supporting Research Questions

- **SRQ1.** Can the Answer-Stable Tail be identified by an NLA-independent
  criterion that passes K-continuation and verifier controls?
- **SRQ2.** At claim-level audit, what fraction of AV sentences on AST windows
  are reconstruction-dependent in the sense of Dingeto (2026 preprint)?
- **SRQ3.** Under matched-control activation interventions, do AST-window
  states carry a direction whose ablation selectively changes
  verification-like behaviour without changing final answers?

### Research Contribution

New empirical evidence for or against the claim that free-form, language-
level readouts of reasoning-model activations supply safe-stopping value
beyond cheap baselines, with the matched evaluation / audit / causal
protocol required to interpret that evidence.

### Technical Contribution

A reproducible pipeline — prompt matrix, decoding configuration, activation
cache schema, NLA integration, verifier integration, probe / entropy / agreement
baselines, intervention harness, and preregistered analysis — released alongside
the labelled AST corpus.

### Experimental Contribution

The first head-to-head comparison of NLA against semantic entropy, hidden-state
correctness probes, and answer-convergence stopping on verified GSM8K and
MATH, at matched compute budgets, with blinded semantic labels and
matched-control activation interventions.

### Practical Contribution

A calibrated, honestly bounded safe-stopping policy for reasoning LLMs — one
that reports tokens saved per percentage-point of accuracy risk and that
abstains when the AST criterion is not met — along with a null-result pathway
that is a scientifically useful outcome.

### Honest statement of what will not be claimed

- The project will **not** claim that NLAs "read the model's thoughts".
- The project will **not** claim to have proven that redundant self-
  verification exists as a distinct internal state unless the full causal
  ladder returns matched-control-supported evidence.
- The project will **not** present a surface AV sentence containing the word
  "check" as positive label for a verification state.

### References

Caldarella, S., Talon, D., Aljundi, R., Ricci, E., & Mancini, M. (2026).
*Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning
Models* (preprint). https://arxiv.org/abs/2606.02835 — preprint, URL as recorded
in the project paper database; not independently re-fetched for Review 1.

Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*
(preprint).

DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning*. Nature.

Dingeto, H. (2026). *Train the Model, Not the Reader: Decodability Supervision
for Verifiable Activation Explanations* (preprint).
https://arxiv.org/abs/2607.20379 — preprint, URL as recorded in the project
paper database; not independently re-fetched for Review 1.

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting hallucinations
in large language models using semantic entropy*. Nature, 630, 625–630.
https://doi.org/10.1038/s41586-024-07421-0

Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). *Natural Language
Autoencoders Produce Unsupervised Explanations of LLM Activations*. Transformer
Circuits (project report). https://transformer-circuits.pub/2026/nla/

Goldowsky-Dill, N., MacLeod, C., Huang, A., et al. (2023). *Towards Best
Practices of Activation Patching in Language Models* (preprint).
https://arxiv.org/abs/2309.16042

Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving With the
MATH Dataset*. NeurIPS.

Huang, J., Chen, X., Mishra, S., et al. (2024). *Large Language Models Cannot
Self-Correct Reasoning Yet*. ICLR.

Lanham, T., Chen, A., Radhakrishnan, A., et al. (2023). *Measuring Faithfulness
in Chain-of-Thought Reasoning* (preprint). https://arxiv.org/abs/2307.13702

Liu, Y., & Wang, X. (2025). *Answer Convergence as a Signal for Early Stopping
in Reasoning*. EMNLP. https://aclanthology.org/2025.emnlp-main.904/

Min, J. et al. (2026). *Stop When Reasoning Converges* (preprint).

Rein, D. et al. (2023). *GPQA*. COLM.

Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). *Language Models
Don't Always Say What They Think* (preprint). https://arxiv.org/abs/2305.04388

Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models*. NeurIPS.

Yang, Z. et al. (2026). *Dynamic Early Exit in Reasoning Models*. ICLR.

Zhang, A., Chen, Y., Pan, J., Zhao, C., Panda, A., Li, J., & He, H. (2025).
*Reasoning Models Know When They're Right: Probing Hidden States for Self-
Verification*. COLM. https://arxiv.org/abs/2504.05419
