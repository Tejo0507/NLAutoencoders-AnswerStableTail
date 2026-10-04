# 02 · Research Landscape

**Compiled:** 2026-10-03. The landscape below is reconstructed from the prior-research
artefacts in `output/nla_answer_stable_tail/phase{1..6}/` (search date 2026-09-25) and
cross-checked against the local paper database `output/nla_answer_stable_tail/paper_db.jsonl`.
Preprints are explicitly marked. Where a claim is not independently re-verifiable from
within the repository on 2026-10-03, this document says so.

---

## 1. Scope of the landscape

The project sits at the intersection of two bodies of literature that have, until
recently, developed in parallel:

- **Internal interpretability of LLM activations** — mechanistic interpretability,
  sparse autoencoders (SAEs), hidden-state probes, activation patching, and the
  recent natural-language-autoencoder (NLA) line of work.
- **Reasoning efficiency and faithfulness** — chain-of-thought (CoT) faithfulness,
  answer convergence, overthinking, dynamic early exit, self-correction, and
  uncertainty estimation.

A **taxonomy** is given in §3. §4 names the state of the art along each branch, §5
lists the major open problems, and §6 records datasets, metrics, and evaluation
conventions.

## 2. Methodological evolution — a short history

1. **CoT prompting establishes a dominant interface for reasoning** (Wei et al. 2022).
   The textual reasoning trace becomes both the computational substrate and the
   candidate interpretability target.
2. **Faithfulness challenges to CoT** (Turpin et al. 2023; Lanham et al. 2023, both
   arXiv preprints at time of writing) show that CoT text can be influenced by
   information the model never acknowledges and that *removing* content from CoT
   often does not change the final answer. The CoT trace is therefore not a safe
   literal record of the computation.
3. **Mechanistic interpretability matures.** Causal tracing / activation patching
   (Meng et al. 2022; methodological best-practice notes by Goldowsky-Dill et al.
   2023, preprint) and SAEs on the residual stream (Cunningham et al. 2023, preprint;
   Templeton et al. 2024, Anthropic project report) show that distributed features
   can be isolated and manipulated, but that choices of metric, corruption, site,
   and threshold materially change interpretation.
4. **Probing hidden states for correctness.** Burns et al. (2022/2023 preprints) and,
   more directly for reasoning, Zhang et al. (COLM 2025) show that correctness or
   answer identity is linearly decodable from intermediate activations in reasoning
   models, with usable AUC in-distribution but weaker cross-domain transfer.
5. **Semantic uncertainty.** Farquhar et al. (Nature 2024) quantify uncertainty over
   *meanings* by clustering sampled answers, outperforming lexical entropy as a
   confabulation detector.
6. **Self-correction is not a free lunch.** Huang et al. (ICLR 2024) show that
   *intrinsic* repeated correction often degrades or does not improve accuracy under
   controlled settings; Madaan et al. 2023 (Self-Refine, NeurIPS) and Shinn et al.
   2023 (Reflexion, NeurIPS) show gains conditional on external feedback or
   specific prompting; Liu et al. 2024 (TACL) consolidate the field.
7. **Answer convergence and dynamic early exit become live research directions.**
   Liu & Wang (EMNLP 2025) use answer-chunk agreement to stop early; Yang et al.
   (ICLR 2026) and Min et al. (2026 preprint) propose dynamic exits in reasoning
   models; Caldarella et al. (2026 preprint) separate harmless "verbose
   overthinking" from "harmful overthinking" via a first-correct-prefix evaluation;
   the Sui et al. (2025 preprint) survey consolidates "efficient reasoning".
8. **Natural Language Autoencoders arrive.** Fraser-Taliente et al. (2026,
   Transformer Circuits project report) jointly train an activation verbaliser (AV)
   and reconstructor (AR) so that free-form text passes a reconstruction bottleneck.
   The authors themselves document confabulation, layer sensitivity, and roughly
   500 generated tokens per activation — i.e. the method is a hypothesis generator,
   not a semantic ground-truth instrument.
9. **An immediate challenge: decodability vs. faithfulness.** Dingeto (2026
   preprint; "RECAP") shows reconstruction may remain high while individual
   verbalised claims are not reconstruction-dependent, and proposes
   *target-model* co-training to make specified facts auditable. RECAP therefore
   strengthens evidence for pre-specified readouts but does not validate arbitrary
   free-form NLA prose.

## 3. Conceptual taxonomy

```
Reasoning in LLMs
├── Reasoning efficiency & termination
│   ├── Static length control (max-new-tokens heuristics)
│   ├── Entropy / confidence thresholds
│   │   ├── Token entropy and answer-token logit margin
│   │   └── Semantic entropy (Farquhar 2024)
│   ├── Answer-convergence stopping (Liu & Wang 2025)
│   ├── Dynamic early exit in reasoning models (Yang 2026; Min 2026 preprint)
│   └── Overthinking taxonomy
│       ├── Verbose overthinking (benign suffix)
│       └── Harmful overthinking (post-correct drift) (Caldarella 2026 preprint;
│           Wu 2026)
├── CoT faithfulness & self-correction
│   ├── Perturbation / hint tests (Turpin 2023 preprint)
│   ├── Deletion / resample tests (Lanham 2023 preprint)
│   ├── Intrinsic self-correction limits (Huang 2024)
│   └── Externally-grounded correction (Self-Refine 2023; Reflexion 2023;
│       Liu 2024 TACL survey)
├── Internal interpretability
│   ├── Causal tracing / activation patching (Meng 2022; Goldowsky-Dill 2023
│   │   preprint)
│   ├── Sparse autoencoders (Cunningham 2023 preprint; Templeton 2024 project
│   │   report; Marks 2024 preprint, feature circuits)
│   ├── Hidden-state probes for correctness (Burns 2023 preprint;
│   │   Zhang 2025 COLM)
│   └── Natural Language Autoencoders (NLA)
│       ├── Fraser-Taliente 2026 (project report) — AV + AR, released
│       │   checkpoints
│       └── RECAP (Dingeto 2026 preprint) — target-model decodability
│           supervision
└── Benchmarks & verifiers
    ├── Math: GSM8K (Cobbe 2021 preprint), MATH (Hendrycks 2021), MATH-500,
    │   AIME-like sets
    ├── Logic: BBH (Suzgun 2022 preprint)
    ├── Code: HumanEval (Chen 2021 preprint), MBPP (with execution)
    └── Expert QA: GPQA (Rein 2023 COLM)
```

## 4. State of the art along each branch

### 4.1 Reasoning efficiency and termination
- **Baseline that beats most heuristics today.** Answer-convergence stopping with
  learned hidden-state termination (Liu & Wang, EMNLP 2025) reports substantial
  token reductions with limited average accuracy loss across five open models and
  five benchmarks. This is the current cheap-and-strong reference point.
- **Representational reference point.** Hidden-state correctness probes
  (Zhang et al., COLM 2025) achieve usable in-distribution AUC on reasoning-model
  intermediate answers and support probe-guided early exit, with weaker transfer
  across domains.
- **Overthinking taxonomy.** Caldarella et al. (2026 preprint) is the closest
  current operational definition of "the tail".

### 4.2 CoT faithfulness
- The dominant finding is **non-literal**: CoT text can be systematically influenced
  by prompt or context features the model never acknowledges (Turpin 2023 preprint),
  and much CoT content can be deleted without changing the answer (Lanham 2023
  preprint). This makes CoT text *per se* unusable as ground truth for internal
  state.

### 4.3 Interpretability
- SAEs and feature circuits (Templeton 2024; Marks 2024 preprint) remain the
  dominant quantitative method for isolating features from the residual stream.
- NLAs (Fraser-Taliente 2026 project report) are the first method to produce free-
  form *language-level* explanations of activations with a reconstruction loss, and
  the current state of the art for that specific interface.
- RECAP (Dingeto 2026 preprint) establishes that reconstruction fidelity is not
  equivalent to claim-level faithfulness.

### 4.4 Uncertainty estimation
- Semantic entropy (Farquhar et al., Nature 2024) is currently the strongest
  general-purpose uncertainty signal for free-form generation.

### 4.5 Causal validation methodology
- Activation patching best-practices (Goldowsky-Dill et al. 2023 preprint) is the
  methodological reference for interpretation-robust interventions on residual
  states.

## 5. Major open problems

1. **No construct validity for "redundant self-verification".** No paper
   demonstrates a distinct, causally relevant internal state corresponding to the
   intuitive notion of a model "checking its answer again".
2. **No head-to-head benchmark of NLA vs. cheap signals on safe stopping.** The
   NLA literature establishes aggregate reconstruction/audit utility; it does not
   compare NLA against semantic entropy, probes, or answer-agreement at matched
   compute budgets on the specific task of safely stopping reasoning.
3. **Faithfulness certification remains partial.** RECAP shows how to make a
   *specified* variable reliably decodable, but the method requires target-model
   co-training and does not retrofit onto released NLA checkpoints.
4. **CoT text is not a ground-truth label for internal state.** This limits label
   availability for any supervised "verification detector".
5. **Semantic labels for suffix type (verification / restatement / derivation /
   correction) have no established annotation protocol or inter-rater benchmark**
   on reasoning traces.
6. **Causal evidence for early stopping is thin.** Most early-exit papers report
   accuracy–token trade-offs but not matched-control activation interventions.

## 6. Datasets, benchmarks, metrics, standard experimental settings

- **Math reasoning.** GSM8K (grade-school, ~8.5k problems); MATH (12.5k problems
  across 7 subjects); MATH-500 and AIME-like subsets for harder evaluation.
- **Logic / mixed.** BBH (BIG-Bench Hard) chain-of-thought subset.
- **Code with execution verification.** HumanEval and MBPP.
- **Expert QA.** GPQA (Google-proof QA).
- **Standard metrics.** Verified final-answer accuracy; tokens or FLOPs budget;
  self-consistency vote; log-prob calibration (ECE); semantic entropy; probe AUC;
  Krippendorff α for human labels.
- **Standard controls** (increasingly used, often omitted). Problem-grouped
  splits; matched token position and length; shuffled activation-text pairings;
  random-direction and random-layer controls for interventions; context-only
  summariser baseline.

## 7. Known limitations recurrent across the field

| Limitation theme | Where it appears | Why it matters for NLA-AST |
|---|---|---|
| CoT text is not literal | Turpin 2023; Lanham 2023; Jacovi & Goldberg 2020 | Any "I am checking" label is unsafe as ground truth. |
| Reconstruction ≠ faithfulness | Dingeto 2026 preprint | The AV output cannot be assumed claim-faithful. |
| Probes decode availability, not use | Zhang 2025; Burns 2023 preprint | A decodable verification signal is not evidence the model uses it. |
| Early exit is accuracy-trade | Liu & Wang 2025; Yang 2026; Min 2026 preprint | Safe-stopping gains must be measured, not assumed. |
| Intrinsic self-correction often fails | Huang 2024 ICLR | A verifying-looking suffix is not presumptively useful. |
| Activation patching is choice-sensitive | Goldowsky-Dill 2023 preprint | Causal AST tests need matched controls. |
| NLA checkpoints are model/layer exact | Fraser-Taliente 2026 project report | Scope of a near-term study is effectively Qwen2.5-7B L20. |

## 8. Recent (last ~24 months) developments that most affect this project

- Appearance of the NLA method and its released AV/AR checkpoints
  (Fraser-Taliente 2026 project report).
- Appearance of a direct faithfulness challenge (Dingeto 2026 preprint).
- Formalisation of post-answer overthinking (Caldarella 2026 preprint;
  Wu 2026 ICLR).
- Peer-reviewed answer-convergence stopping (Liu & Wang EMNLP 2025) and
  hidden-state correctness probing (Zhang COLM 2025).
- Nature-published semantic entropy as a strong, principled baseline
  (Farquhar 2024).

## 9. Where this project proposes to intervene — a preview

The project does **not** aim to produce a new interpretability method or a new
NLA. It aims to produce a *comparative, causally validated evaluation protocol*
for the specific hypothesis that NLA readouts carry information about an
answer-stable, verification-like tail that is not available from strong cheap
baselines. The literature above does not currently supply such a comparison; the
research gap in `05_research_gap.md` makes this precise.

## 10. Preregistration-style constraints inherited from the field

- Problem-grouped train/dev/test splits.
- Matched position / length controls for any claimed detector.
- Reporting *both* paired accuracy change and tokens / FLOPs saved.
- Reporting Krippendorff α for any semantic label used as outcome.
- Reporting matched-control results for every activation intervention.
- Marking preprints and project reports as such throughout.
