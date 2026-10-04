# 03 · Literature Candidates and Screening

**Compiled:** 2026-10-03.
**Source of candidates:** `output/nla_answer_stable_tail/paper_db.jsonl` (36 primary
entries, dated to field milestones through 2026-09-25) and the deep-reading notes in
`output/nla_answer_stable_tail/phase3_deep_dive/`.
**Screening rule:** inclusion requires a primary paper or an official technical / project
report that bears on at least one of {NLA, activation explanation, CoT faithfulness,
answer convergence, overthinking, hidden-state probes, self-correction, causal
interpretability, uncertainty estimation, SAEs, reasoning benchmarks}. Preprints are
retained but explicitly marked and are not treated as settled evidence. Blog posts and
secondary summaries are excluded unless they are an official project report with
inspectable code or checkpoints.

---

## 1. Candidate pool (36 entries)

The IDs below are the keys used in `paper_db.jsonl`. The "cluster" column places each
work in the taxonomy of `02_research_landscape.md`. "Peer-reviewed" is as recorded in
the paper database. URLs/DOIs follow the project's existing `references.bib` and the
2026-09-25 phase reports; where this review did not re-verify a DOI/URL on 2026-10-03,
the status column says so.

| # | ID (paper_db) | Cluster | Peer-reviewed? | Verified in this review? |
|---|---|---|---|---|
| 1 | fraser-taliente2026nla | NLA | No (project report, Transformer Circuits) | Metadata consistent with phase6 report; URL not re-fetched here. |
| 2 | dingeto2026recap | NLA faithfulness | No (preprint) | arXiv ID format `2607.20379` is unusual; retained as recorded, flagged. |
| 3 | liu-wang2025convergence | Early stop | Yes (EMNLP 2025) | ACL anthology link recorded in phase6 report. |
| 4 | zhang2025right | Hidden-state probe | Yes (COLM 2025) | arXiv:2504.05419 recorded. |
| 5 | caldarella2026past | Overthinking | No (preprint) | arXiv ID `2606.02835`; retained, flagged. |
| 6 | farquhar2024semantic | Uncertainty | Yes (Nature 2024) | DOI 10.1038/s41586-024-07421-0 recorded. |
| 7 | huang2024selfcorrect | Self-correction | Yes (ICLR 2024) | OpenReview ID recorded. |
| 8 | turpin2023unfaithful | CoT faithfulness | No (preprint) | arXiv:2305.04388 recorded. |
| 9 | lanham2023faithfulness | CoT faithfulness | No (preprint) | arXiv:2307.13702 recorded. |
| 10 | jacovi-goldberg2020faithful | Faithfulness theory | Yes (ACL 2020) | Not re-verified in this review. |
| 11 | meng2022rome | Causal tracing | Yes (NeurIPS 2022) | Not re-verified in this review. |
| 12 | goldowsky-dill2023patching | Causal tracing methodology | No (preprint) | arXiv:2309.16042 recorded. |
| 13 | cunningham2023sae | SAE | No (preprint) | Not re-verified. |
| 14 | templeton2024mono | SAE | No (Anthropic project report) | Not re-verified. |
| 15 | wei2022cot | CoT | Yes (NeurIPS 2022) | Not re-verified. |
| 16 | lightman2023prm | Process reward | No (preprint) | Not re-verified. |
| 17 | setlur2024progress | Process reward | No (preprint) | Not re-verified. |
| 18 | deepseek2025r1 | Reasoning model | Yes (Nature 2025) | Not re-verified. |
| 19 | liu2025mind | CoT limits | Yes (ICML 2025) | Not re-verified. |
| 20 | yang2026dynamic | Dynamic exit | Yes (ICLR 2026) | Not re-verified. |
| 21 | min2026converges | Dynamic exit | No (preprint) | Not re-verified. |
| 22 | wu2026less | Overthinking | Yes (ICLR 2026) | Not re-verified. |
| 23 | sui2025survey | Survey | No (preprint) | Not re-verified. |
| 24 | madaan2023selfrefine | Self-correction | Yes (NeurIPS 2023) | Not re-verified. |
| 25 | shinn2023reflexion | Self-correction | Yes (NeurIPS 2023) | Not re-verified. |
| 26 | liu2024surveyselfcorrection | Self-correction survey | Yes (TACL 2024) | Not re-verified. |
| 27 | marks2024sae | SAE / circuits | No (preprint) | Not re-verified. |
| 28 | hendrycks2021math | Benchmark | Yes (NeurIPS 2021) | Standard benchmark. |
| 29 | cobbe2021gsm8k | Benchmark | No (preprint) | Standard benchmark. |
| 30 | rein2023gpqa | Benchmark | Yes (COLM 2023/24) | Not re-verified. |
| 31 | chen2021humaneval | Benchmark | No (preprint) | Not re-verified. |
| 32 | suzgun2022bbh | Benchmark | No (preprint) | Not re-verified. |
| 33 | burns2023discover | Probe | No (preprint) | Not re-verified. |
| 34 | burns2022discover | Probe | No (preprint) | Not re-verified; near-duplicate of (33). |
| 35 | gurnee2024representation | Representation | No (preprint) | Not re-verified. |
| 36 | kassis2026skills | Agent skills library | No (preprint) | Only relevant as a methodological acknowledgement for this project's workflow, not as scientific evidence for NLA-AST. |

## 2. Screening decisions

The 10 final papers for Chapter 2's comparison table (see
`04_literature_comparison.md`) are chosen so that they *collectively* cover every
causal step in the AST inference chain:

```
NLA readout (what we read)
 → representation evidence (what else is encoded)
 → answer stability (behavioural signal)
 → self-correction / overthinking (what the tail might be)
 → faithfulness of textual reports (how far text can be trusted)
 → causal validation (how to prove the signal is used)
 → uncertainty baseline (what we must beat)
```

**Final 10 (justified in `04_literature_comparison.md`):**

1. Fraser-Taliente et al. 2026 (NLA, project report).
2. Dingeto 2026 (RECAP, preprint).
3. Liu & Wang 2025 (Answer Convergence, EMNLP).
4. Zhang et al. 2025 (Hidden-state correctness probe, COLM).
5. Caldarella et al. 2026 (Harmful Overthinking, preprint).
6. Farquhar et al. 2024 (Semantic Entropy, Nature).
7. Huang et al. 2024 (Self-Correction limits, ICLR).
8. Turpin et al. 2023 (CoT not literal, preprint).
9. Lanham et al. 2023 (CoT faithfulness measurement, preprint).
10. Goldowsky-Dill et al. 2023 (Activation patching best practices, preprint).

### 2.1 Why these ten rather than other strong candidates

- **Wu 2026 "When More is Less" (ICLR)** overlaps with Caldarella 2026 on the
  overthinking taxonomy; Caldarella is retained because its first-correct-prefix
  construction is closest to the AST definition.
- **Yang 2026 (dynamic early exit, ICLR)** and **Min 2026 (preprint)** overlap with
  Liu & Wang 2025 on the early-exit baseline; Liu & Wang is retained because its
  method is the simplest, strongest baseline the project must beat and because it
  is peer-reviewed.
- **Templeton 2024 and Marks 2024 (SAE / circuits)** could enter the table as a
  mechanistic-interpretability baseline. They are *not* retained in the final 10
  because (i) the project does not currently include an SAE arm, (ii) their
  method is well surveyed in Fraser-Taliente 2026's related work, and (iii)
  retaining them would displace a paper that bears directly on the AST
  inference chain. They are discussed in prose in Chapter 2.1 instead.
- **Burns 2023 / 2022 (CCS probe)** is subsumed by Zhang 2025 for the reasoning
  setting.
- **Huang / Madaan / Shinn / Liu 2024 TACL (self-correction)** family: Huang 2024
  is retained as the strongest controlled negative result; the others are
  discussed in prose.
- **Benchmarks (GSM8K, MATH, GPQA, HumanEval, BBH)** are not selected into the
  comparison table — they are datasets, not competing methods — and are
  discussed in Chapter 2.1 as the evaluation substrate.
- **Jacovi & Goldberg 2020** is a definitional paper for faithfulness; cited in
  prose but not in the comparison table.
- **Kassis et al. 2026 (Scientific Agent Skills)** is a methodological
  acknowledgement for the research-workflow tooling and does not enter the
  evidence base for scientific claims.

## 3. Screening outcome summary

- Candidate pool size: 36.
- Final selection: 10 (justified above).
- Mixture: 1 foundational NLA source, 1 recent challenge to NLA faithfulness,
  3 competing operational signals (answer-convergence, hidden-state probe,
  semantic entropy), 1 overthinking taxonomy, 1 self-correction negative
  result, 2 CoT-faithfulness primers, 1 causal-interpretability methodology.
- Preprint fraction in final 10: 6 of 10 (NLA source, RECAP, Caldarella,
  Turpin, Lanham, Goldowsky-Dill). Each is marked as preprint throughout.

## 4. Integrity notes

- Items 33 and 34 (`burns2023discover` and `burns2022discover`) in the paper
  database appear to be near-duplicates of the same work (Burns et al.,
  "Discovering Latent Knowledge in Language Models Without Supervision"); only
  one logical entry is used in prose.
- Several 2026 arXiv IDs in the paper database have unusual month fields
  (e.g. `2607.*`, `2606.*`, `2609.*`); given today's date (2026-10-03) these are
  plausibly real but have not been re-fetched here. They are kept as recorded
  and flagged as preprints.
- No new candidate papers have been added to the pool during this review-1
  preparation. Any further literature search before Review 2 should be
  pre-registered.
