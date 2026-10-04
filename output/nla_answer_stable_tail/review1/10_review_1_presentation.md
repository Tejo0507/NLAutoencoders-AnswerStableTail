# 10 · Review 1 Presentation (Slide-Ready Content)

**Compiled:** 2026-10-03. Each slide below contains (i) a title, (ii) bullet
content, and (iii) one presenter note. The deck targets ~16 slides including
title and references, consistent with the project brief. If a
`scientific-slides` export is required later, each `## Slide N` block maps
one-to-one to a slide.

---

## Slide 1 — Title

- **Beyond Readable Hypotheses: An NLA-Independent, Causally Validated
  Benchmark for the Answer-Stable Tail in LLM Reasoning**
- Project Review 1
- Repository: `NLAutoencoders-AnswerStableTail`
- Date: 2026-10-03

*Presenter note.* Open by naming the two literatures the project bridges —
mechanistic interpretability (NLAs) and efficient reasoning (early stopping /
overthinking) — before stating the specific research question.

---

## Slide 2 — Background

- Reasoning LLMs often continue generating tokens **after** the answer has
  behaviourally stabilised (Liu & Wang 2025 EMNLP; Caldarella 2026 preprint).
- Why? Candidate hypothesis: a **redundant internal self-verification state**.
- If testable, it would unite an interpretability advance with a
  deployment-relevant control (safe early termination).
- Natural Language Autoencoders (NLAs) (Fraser-Taliente 2026, Transformer
  Circuits project report) appear to supply the readout tool: residual
  activations → free-form text, with reconstruction fidelity (FVE).

*Presenter note.* State clearly that NLA is a **project report**, not a
peer-reviewed paper, and that FVE is vector-level evidence, not claim-level.

---

## Slide 3 — Problem Statement

- No published study **jointly** supplies:
  1. an NLA-independent operational definition of the Answer-Stable Tail
     (AST);
  2. a head-to-head NLA-vs-cheap-baseline comparison on safe stopping;
  3. a claim-level faithfulness audit of AV sentences on reasoning tails;
  4. matched-control activation interventions on the tail window.
- Consequence: strong interpretability claims — "the model is checking
  its answer" — are being implied on textual appearance alone.

*Presenter note.* This is where the research gap is first motivated. Keep
language careful: *claims are being implied*, not *falsified*; the project's
job is to **test**, not to **debunk**.

---

## Slide 4 — Research Motivation

- Early-stopping is a current operational question: compute cost of
  reasoning LLMs is dominated by long chains of thought.
- The scientific question underneath — *what is the tail?* — is one of
  the few cases where interpretability can be audited against a
  behavioural outcome (verified accuracy).
- Released NLA checkpoints (Qwen2.5-7B-L20 and others) make the
  experiment feasible without training.

*Presenter note.* The motivation combines a practical driver (compute) and a
scientific driver (auditable interpretability).

---

## Slide 5 — Aim

- **Aim.** Build and empirically test a pre-registered, NLA-independent,
  causally validated protocol for evaluating whether natural-language
  readouts of residual-stream activations add safe-stopping value in
  reasoning LLMs, and compare NLA against strong cheap baselines on this
  task.

*Presenter note.* One sentence. The aim is deliberately framed around
**evaluation + protocol**, not around "proving" an internal state exists.

---

## Slide 6 — Objectives

- **O1.** Define AST operationally without NLA (truncation + K
  continuations + verifier + matched controls).
- **O2.** Implement reproducible NLA pipeline on Qwen2.5-7B-L20.
- **O3.** Measure NLA's incremental value over semantic entropy,
  hidden-state probes, and answer-convergence stopping.
- **O4.** Audit AV sentences under the Lanham (2023 preprint) template.
- **O5.** Run matched-control activation interventions
  (Goldowsky-Dill 2023 preprint).
- **O6.** Release labelled AST corpus + evaluation harness.

*Presenter note.* Each objective maps to a specific limitation in the
problem statement; the mapping is in `08_research_traceability.md`.

---

## Slide 7 — Research Questions

- **Primary RQ.** Does an NLA-derived readout provide incremental,
  causally supported safe-stopping value for reasoning LLMs on GSM8K and
  MATH, over strong cheap baselines?
- **SRQ1.** Can AST be identified by an NLA-independent criterion that
  passes K-continuation and verifier controls?
- **SRQ2.** What fraction of AV sentences on AST windows are
  reconstruction-dependent (Dingeto 2026 preprint sense)?
- **SRQ3.** Under matched-control interventions, do AST-window states
  carry a verification-selective direction?

*Presenter note.* Keep the primary RQ in a single sentence; examiners will
test wording.

---

## Slide 8 — Research Landscape (Taxonomy)

- **Reasoning efficiency / termination.** Static length; token entropy;
  semantic entropy; answer convergence; dynamic early exit.
- **Overthinking.** Verbose vs. harmful overthinking (first-correct prefix).
- **CoT faithfulness.** Perturbation; deletion; resample; paraphrase.
- **Self-correction.** Intrinsic failures; externally-grounded gains.
- **Internal interpretability.** Activation patching; SAEs; probes;
  **NLAs**; RECAP challenge.
- **Benchmarks.** GSM8K; MATH; GPQA; HumanEval; BBH.

*Presenter note.* One visual table on the slide; narrate the taxonomy from
left to right.

---

## Slide 9 — Existing Approaches (Methodological Families)

- **Behavioural stopping.** Liu & Wang 2025 (EMNLP) — answer agreement.
- **Representational.** Zhang 2025 (COLM) — hidden-state correctness
  probe.
- **Uncertainty.** Farquhar 2024 (Nature) — semantic entropy.
- **Tail taxonomy.** Caldarella 2026 preprint — first-correct prefix.
- **Readout.** Fraser-Taliente 2026 (project report) — NLA.
- **Faithfulness challenge.** Dingeto 2026 preprint — RECAP.
- **Causal interpretability.** Goldowsky-Dill 2023 preprint — patching
  best-practice.

*Presenter note.* Emphasise that **none of these** supply the composite
protocol the project proposes.

---

## Slide 10 — Literature Comparison (Compact View)

Compact overview matrix from `04_literature_comparison.md` §3.

| Paper | Activation? | Text? | Causal? | Peer-reviewed? |
|---|---|---|---|---|
| P1 NLA (Fraser-Taliente 2026) | Yes | **Free-form** | No | PR |
| P2 RECAP (Dingeto 2026) | Yes | Partial | No | preprint |
| P3 Answer convergence (Liu & Wang 2025) | Partial | Yes | No | **EMNLP** |
| P4 Hidden-state probe (Zhang 2025) | Yes | No | No | **COLM** |
| P5 First-correct prefix (Caldarella 2026) | No | Yes | No | preprint |
| P6 Semantic entropy (Farquhar 2024) | No | Yes | No | **Nature** |
| P7 Self-correction limits (Huang 2024) | No | Yes | No | **ICLR** |
| P8 CoT non-literal (Turpin 2023) | No | Yes | No | preprint |
| P9 CoT faithfulness measurement (Lanham 2023) | Partial | Yes | Partial | preprint |
| P10 Activation patching (Goldowsky-Dill 2023) | Yes | No | **Yes** | preprint |

*Presenter note.* The full 12-column comparison is in
`04_literature_comparison.md` §2 and should be printed/handed out.

---

## Slide 11 — Critical Findings from the Literature

- Answer stability ≠ redundancy (Liu & Wang 2025; Caldarella 2026
  preprint).
- Decodability ≠ causal use (Zhang 2025; Goldowsky-Dill 2023 preprint).
- Reconstruction ≠ claim-level faithfulness (Dingeto 2026 preprint;
  Fraser-Taliente 2026).
- CoT text ≠ literal account of computation (Turpin 2023 preprint;
  Lanham 2023 preprint).
- Intrinsic self-correction ≠ reliably useful (Huang 2024).

*Presenter note.* Each "≠" is a boundary the project is designed to respect.

---

## Slide 12 — Limitations (Recurring)

- No NLA-independent AST definition with K-continuation + verifier
  controls.
- No head-to-head NLA-vs-cheap-baseline evaluation on safe stopping.
- No claim-level audit of AV sentences on reasoning tails.
- No matched-control activation interventions on tail windows.
- No blinded semantic labels with Krippendorff α on tail types.

*Presenter note.* These five bullets are the exact five Limitations A–E in
Chapter 2.2 §2.2.2.

---

## Slide 13 — Research Gap

- **Composite gap:** methodological + evaluation + causal +
  faithfulness.
- **Not** an implementation or application gap: adding NLA to GSM8K
  without the above components leaves the gap intact.
- **Evidence** is positive, not merely absent: Fraser-Taliente 2026's
  own stated limits; Dingeto 2026 preprint's direct challenge; Huang
  2024 ICLR and the CoT-faithfulness literature.

*Presenter note.* Say explicitly: "A gap of absence that would be
scientifically weak is distinguished here from a gap of **documented
unresolved tension**, which this is."

---

## Slide 14 — Proposed Research Direction

- Define AST without NLA → K-continuation + verifier + matched
  controls.
- Run NLA (AV + AR, Qwen2.5-7B-L20) at AST candidates.
- Compare against semantic entropy, hidden-state probe,
  answer-convergence stopping at matched compute.
- Audit AV sentences under Lanham 2023 preprint template.
- Causal-validation ladder under Goldowsky-Dill 2023 preprint
  standards.
- Pre-registered analysis; null results published.

*Presenter note.* Emphasise pre-registration and the honest publication
pathway for null results.

---

## Slide 15 — Expected Contribution and Risk

- **Contribution.** The first head-to-head safe-stopping benchmark of
  NLA against cheap signals, with blinded labels and matched-control
  interventions on reasoning tails.
- **Novelty character.** Combination + evaluation, confidence medium.
- **Honest risk.** If cheap baselines match NLA, the project becomes an
  evaluation protocol + semantic-interface + negative-result paper on
  NLA as a detector.
- **Non-claims.** Does **not** claim NLAs "read the model's thoughts".
  Does **not** claim redundant self-verification is proven to exist as
  a distinct internal state.

*Presenter note.* State both the ambition and the honest downside in the
same slide.

---

## Slide 16 — References (summary)

- Fraser-Taliente et al. 2026 (NLA, project report).
- Dingeto 2026 (RECAP, preprint).
- Liu & Wang 2025 (Answer Convergence, EMNLP).
- Zhang et al. 2025 (Hidden-state probe, COLM).
- Caldarella et al. 2026 (Harmful Overthinking, preprint).
- Farquhar et al. 2024 (Semantic Entropy, Nature).
- Huang et al. 2024 (Self-Correction limits, ICLR).
- Turpin et al. 2023 (CoT non-literal, preprint).
- Lanham et al. 2023 (CoT faithfulness measurement, preprint).
- Goldowsky-Dill et al. 2023 (Activation patching, preprint).
- Kassis et al. 2026 (Scientific Agent Skills; methodology
  acknowledgement, preprint).

*Presenter note.* The full reference list is in
`06_chapter_1_introduction.md` and `07_chapter_2_literature_review.md`.
All preprint URLs are as recorded in the project paper database on
2026-09-25 and have not been independently re-fetched today;
peer-reviewed venues (EMNLP, COLM, Nature, ICLR) are clearly labelled.
