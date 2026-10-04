# 08 · Research Traceability

**Compiled:** 2026-10-03. Every component below links a problem-statement
limitation to a gap, a research question, an objective, a method, an
experiment, a metric, and the evidence class that will decide the question.
If any row cannot be filled, it is flagged.

Problem-statement limitations L1–L4 are the four limitations named in
`06_chapter_1_introduction.md` §1.2.

---

## 1. Primary traceability table

| Problem | Research Gap | Research Question | Objective | Method | Experiment | Metric | Evidence class |
|---|---|---|---|---|---|---|---|
| **L1** No NLA-independent AST definition. | **Operational gap A** (`05_research_gap.md`). | **SRQ1.** Can AST be identified by an NLA-independent criterion that passes K-continuation + verifier controls? | **O1.** Define AST operationally without NLA. | Chunk-level answer equivalence under truncation; K independently sampled continuations with external verifier; matched token-position / length controls. | **E1.** Boundary identification on GSM8K + MATH-algebra/C&P using Qwen2.5-7B-Instruct generations. | Fraction of traces with ≥1 AST candidate; AST false-positive rate under matched-position shuffle; agreement with Caldarella-style first-correct prefix. | Behavioural; descriptive. |
| **L2** No comparison of NLA readouts to cheap baselines on safe stopping. | **Evaluation gap B** (`05_research_gap.md`). | **PRQ.** Does an NLA-derived readout provide incremental, causally supported value for safe stopping over cheap baselines? | **O2.** Build a reproducible NLA pipeline on the released Qwen2.5-7B-L20 checkpoint. **O3.** Measure incremental value vs. baselines. | NLA (AV+AR) at AST windows; semantic entropy (Farquhar 2024); hidden-state correctness probe (Zhang 2025); answer-convergence stopping (Liu & Wang 2025); position/length heuristics. | **E2.** Safe-stopping evaluation at matched compute budgets, with and without each signal. | Verified final-answer accuracy; tokens / FLOPs saved; AUROC for safe-stopping; **incremental AUC of NLA over baselines**; paired bootstrap CI by problem. | Behavioural + representational; comparative. |
| **L3** No claim-level faithfulness audit of AV sentences on reasoning tails. | **Faithfulness gap C** (`05_research_gap.md`). | **SRQ2.** What fraction of AV sentences on AST windows are reconstruction-dependent in the sense of Dingeto (2026 preprint)? | **O4.** Audit a stratified AV sample using the Lanham (2023 preprint) template. | AV-sentence deletion / resample / paraphrase; score AR reconstruction change; blind rater adequacy of alternative. | **E3.** Stratified audit on N≥200 AV sentences from AST windows. | Fraction of claims that materially change AR reconstruction; blinded rater adequacy; AV-prose genericness rate. | Faithfulness audit. |
| **L4** No causal validation with matched controls. | **Causal gap D** (`05_research_gap.md`). | **SRQ3.** Under matched-control interventions, do AST-window states carry a direction whose ablation selectively changes verification-like behaviour without changing final answers? | **O5.** Run the causal-validation ladder in Goldowsky-Dill (2023 preprint) methodology. | Truncation at AST; tail replacement by neutral filler; matched-random-direction, matched-position, dose-response controls; optional NLA-edit steering. | **E4.** Intervention ladder on a held-out subset of traces. | Paired change in verified accuracy; direction specificity (effect on verification vs. derivation); dose-response slope; matched-control invariance. | Causal. |
| **Reporting** | — | — | **O6.** Release labelled AST corpus + preregistered harness. | Problem-grouped train/dev/test splits; versioned configs; checksums; CI logs. | **E5.** Reproducibility pack. | Krippendorff α on all labels; software/data manifests. | Reproducibility. |

## 2. Secondary traceability — each paper → where it enters the project

| Paper (`04_literature_comparison.md`) | Role in the project | Objective it touches | Experiment it touches |
|---|---|---|---|
| P1 Fraser-Taliente 2026 (NLA, project report) | **Method.** Supplies AV + AR + released checkpoints. | O2 | E2, E3 |
| P2 Dingeto 2026 (RECAP, preprint) | **Constraint on claims.** Framing of faithfulness audit; motivation for Lanham-template on AV sentences. | O4 | E3 |
| P3 Liu & Wang 2025 (EMNLP) | **Primary cheap baseline.** Supplies chunk framework and answer-convergence stopping. | O1, O3 | E1, E2 |
| P4 Zhang 2025 (COLM) | **Representational baseline.** Hidden-state correctness probe. | O3 | E2 |
| P5 Caldarella 2026 (preprint) | **Operational tail definition.** First-correct prefix extended with K-continuation controls. | O1 | E1 |
| P6 Farquhar 2024 (Nature) | **Uncertainty baseline.** Semantic entropy. | O3 | E2 |
| P7 Huang 2024 (ICLR) | **Prior of skepticism.** Rules out treating "verification-looking" surface text as positive outcome. | O1, O4 | E1, E3 |
| P8 Turpin 2023 (preprint) | **CoT non-literality constraint.** Forbids using tail text as ground-truth label. | O1 | E1 |
| P9 Lanham 2023 (preprint) | **Audit template.** Deletion / resample / paraphrase on AV sentences. | O4 | E3 |
| P10 Goldowsky-Dill 2023 (preprint) | **Causal-validation standard.** Matched controls on every intervention. | O5 | E4 |

## 3. Falsification table (what counts as a negative result)

| Expected positive | Falsification condition | Scientific value of the negative |
|---|---|---|
| AST candidates preserve verified answers under truncation. | More than a pre-registered threshold of candidates fail verifier non-inferiority under truncation. | Finding: *answer stability is not safely sufficient for stopping*; project becomes a cautionary benchmark. |
| NLA adds incremental AUC over Liu & Wang 2025 + Farquhar 2024 + Zhang 2025 at matched budget. | NLA ≤ baseline-max within paired bootstrap CI. | Finding: *NLA is a semantic interface, not a detector*; project becomes an evaluation + interface contribution. |
| Lanham-style audit of AV sentences shows meaningful claim-level dependence. | Low reconstruction-change under deletion / resample / paraphrase. | Finding: *AV prose is reconstruction-equivalent at claim level* on reasoning tails, strengthening RECAP's challenge; project becomes a methodological warning paper. |
| Matched-control activation interventions show selective effect of a verification direction. | Matched-random-direction and matched-position controls produce comparable effect; no dose-response. | Finding: *no verification mechanism is identifiable*; project retracts the mechanism claim and reports the null. |
| Inter-rater reliability on semantic labels passes threshold. | Krippendorff α below pre-registered threshold. | Finding: *the construct does not survive annotation*; project narrows semantic claims to label categories that pass α. |

## 4. Validity audit (self-check)

- Does every objective address a problem-statement limitation? **Yes** —
  O1→L1, O2+O3→L2, O4→L3, O5→L4, O6 is reporting.
- Does every research question map to a planned experiment? **Yes** —
  PRQ→E2; SRQ1→E1; SRQ2→E3; SRQ3→E4.
- Does every planned experiment have a metric able to answer its question?
  **Yes** — see the metric column of §1.
- Are there objectives that do not address a gap? **No.** Any component
  that cannot be defended under this test will be dropped before Review 2.
- Are there research questions that cannot be answered by the proposed
  experiments? **No** — SRQ1–SRQ3 and PRQ each have a specific experiment,
  though statistical power depends on the chosen K and problem budget,
  which will be pre-registered.
- Is the proposed method aligned with the stated gap? **Yes** —
  methodological + evaluation + causal + faithfulness, matched to the
  gap classification in `05_research_gap.md` §2.
