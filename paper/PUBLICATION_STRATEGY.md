# Publication strategy, backlog and go/no-go (2026-10-20)

**Basis.** The evidence as of 2026-10-20 (`paper/CLAIM_EVIDENCE_MATRIX.md`,
`docs/UNDEREXPLORED_RESULTS_2026-10-20.md`). The `rep1` replication is still
generating traces (175/400 at 10:52 IST). Every recommendation below says
how `rep1` and the D82 forcing pass could change it.

## 1. Where the evidence stands

| | established on stored data | still open |
|---|---|---|
| **NLA as a stopping signal** | Its registered score adds nothing beyond position and the probe (C1). Its apparent advantage comes from early stops and abstention (B4, B6). An online check costs 0.83 of a trace (D2) | D75's registered H1/H2 on fresh problems |
| **NLA as a description of state** | The explanation text carries reasoning-phase information beyond position and the raw trace text, and its content is legible ("concluding" against "setting up") (C2, C3) | Out-of-sample transfer (`rep1` lexicon); off-menu (Stage 1) |
| **NLA against cheap representations** | A same-label linear probe on the same activation subsumes the text (C4) | `rep1`; Stage 1 |
| **Evaluation methodology** | The menu leaks the outcome (B5). The matched-mean-saving estimand is dominated by abandonment, even against an AUC-0.86 signal (B7). 48-token forced answers are reliable but invalid before the tail (A2, A3) | D82 decides how much the forcing budget matters |

## 2. Candidate paper directions

| | central claim | novelty | evidence now | main gap | weaknesses | done with existing + `rep1` + D82? |
|---|---|---|---|---|---|---|
| **A.** NLA readouts improve stopping | — | — | Refuted on stored data (C1, B6, D2) | — | — | **Not viable** as a positive paper. Survives only as a negative result inside E |
| **B.** Methodology: leakage, candidate-point selection and outcome definitions in reasoning-efficiency evaluation | Three common evaluation choices can manufacture or erase early-stopping gains: outcome-derived candidate points, matched-mean-saving estimands with abstention counted safe, and short forced-answer budgets | Plausibly high. Probe-based early exit and forced-exit labelling are active topics (self-verification probes [arXiv 2504.05419]; LYNX [2512.05325]; ThinkBrake's sentence-level forced stopping [2510.00546]). A search found no critique of their evaluation protocols, but **a systematic related-work review is required** before claiming novelty | Strong: B4–B7 and A2–A3, plus a clean demonstration (an information-free rule beats an AUC-0.86 probe) | D82 (measured truncation effect); the same analyses on `rep1` (replication) | One model, one task family. The critique is demonstrated on our own pipeline, so reviewers will ask whether it transfers to published methods | **Yes** |
| **C.** Answer stability: the tail is recoverability within a budget, not redundant computation | The answer-stable tail marks where the answer becomes recoverable under a forcing budget, and that point moves with the budget | Moderate | A1–A3 | D82 and tail starts under 256-token forcing (backlog H3) | Close to existing "overthinking" analyses; one model | Mostly; H3 needs about 2 extra GPU-hours |
| **D.** What NLA verbalisations carry about reasoning state, against cheap probes | NLA text legibly encodes reasoning phase, but carries no information a linear probe on the same activation lacks, and an online check costs nearly a trace | High for the interpretability audience: NLAs are new (Anthropic, 2026), and an incremental-value test against same-activation probes is the obvious skeptical question | C1–C5, D2 on stored data, with tight intervals | `rep1` replication (pre-specified); Stage 1 off the menu (4–6 GPU-hours) | One NLA pair and layer; TF-IDF featurisation; menu-only so far | Yes with Stage 1; without it, a reviewer will ask about the menu |
| **E.** Combined: B as the frame, D as the case study, A's negative result inside it | "Evaluating whether verbalised activations help a model know when it is done: what the readout carries, what cheap probes carry, and how evaluation choices decide the answer" | Union of B and D | All of the above | `rep1`, D82, Stage 1 | Length and focus. It must not read as a list of failures | Yes |

**Novelty update (2026-10-20, `docs/NOVELTY_REVIEW_2026-10-20.md`):**
- Two general lessons are **already published**: charge the signal's
  compute, and compare at matched risk (LearnStop, arXiv 2606.30852; Thought
  Calibration, EMNLP 2025).
- Direction B's novel core narrows to three specific failure modes:
  outcome-derived candidate points, the matched-mean-saving/abstention
  artefact, and forced-answer budget validity.
- Direction D's closest neighbour is Li et al., ICML 2026, on verbalizers
  conveying privileged information. Our same-activation, same-label probe
  control extends theirs.

## 3. Recommendation

**Primary: E, framed around D's question and B's lessons.**

The paper answers a question the interpretability community will ask about
NLAs: does natural-language access to activations tell you anything a cheap
probe does not, here about whether reasoning is done? The answer is
*legible, yes; incremental, no*. It also shows why the stopping-evaluation
setup would have suggested otherwise, and that lesson transfers to
probe-based early exit generally.

Each half protects the other:
- The methodological findings explain the earlier positive-looking result.
  That makes the negative result credible, not merely null.
- The NLA case study gives the methodological points a concrete, reproducible
  demonstration.

**Fallback: B alone,** with NLA as one of several signals evaluated. Use it if:
- `rep1` or Stage 1 overturns C4, so the NLA story is unresolved; or
- the NLA part has to shrink to fit a venue.

**What would change this recommendation:**

| outcome | consequence |
|---|---|
| D75 H1 *supported* on `rep1` (the readout beats the probe under the registered estimand) | Report it as registered. But B4–B7 mean it would show the estimand's behaviour, not the readout's information. The selective-risk and information analyses on `rep1` decide which |
| `rep1` text-over-probe > 0, or Stage 1 Δ₁ > 0 | D becomes a positive result. Promote it and keep B as context |
| D82 finds `f ≥ 0.80` (cut-offs are mostly the final answer) | The 48-token outcome measured commitment. B's truncation lesson becomes central, and the A-series claims are reworded |

## 4. Prioritised backlog

### Essential before submission

| # | item | cost | state |
|---|---|---|---|
| E1 | Finish `rep1` and run D75's confirmatory analysis (registered H1/H2) | about 13–20 GPU-hours remaining, unattended (`rep1_runbook.py`) | running |
| E2 | `rep1` secondary: D79 position audit; D81 information audit, post-hoc checks and lexicon transfer; D83 selective risk | CPU, about 30 min | pre-specified; commands in §6 |
| E3 | D82 extended forcing on `main_nla240`, then `rep1`, and its analysis | 1–2 GPU-hours each | plan frozen; code tested |
| E4 | Systematic related-work review: probe-based early exit, forced-exit labelling, overthinking metrics, NLA/LatentQA-style verbalisers | 1–2 days | one exploratory search done |
| E5 | Rewrite the manuscript to the E framing (`paper/MANUSCRIPT_OUTLINE.md`). The frozen report stays as the registered record | days | outline written |
| E6 | Correct statements that exceed the evidence in living summaries (matrix §F) | hours | README corrected 2026-10-20 |

### High-value if inexpensive

| # | item | cost | why |
|---|---|---|---|
| H1 | **Stage 1** of D83: NLA at outcome-blind positions | 4–6 GPU-hours | Removes the "menu only" objection to C2–C4 |
| H2 | Stage 0 on `rep1` traces | CPU (needs `rep1` acts and D82-style forcing at every boundary, about 1 GPU-hour) | Replicates B7 and D1 on fresh problems |
| H3 | Tail starts recomputed under 256-token forcing at every boundary of `main` | about 2 GPU-hours | Settles C/A1 directly |
| H4 | A small human check of the phase lexicon (2 annotators, 100 explanations) | about 2 annotator-hours | Turns C3 from string evidence into rated content. Not a faithfulness test |
| H5 | Activation probes at layers 14 and 24 against the text | CPU | Shows where the phase signal lives |
| H6 | Publication polish of `paper/figures_v2/` | hours | Overlapping labels in fig 5; consistent palette |

### Optional future work

- A second model or NLA pair (checkpoint availability unverified).
- A stronger text featuriser (sentence embeddings), with its cost charged.
- RQ3, which stays closed at Stage 0 (D77).
- F5 sweeps (60–150 GPU-hours, D52).
- The two-annotator claim study (`docs/ANNOTATOR_PACKET.md`), needed only if
  the paper makes a faithfulness claim. E does not.

## 5. Go / no-go for further experiments

| experiment | decision | the uncertainty it resolves | condition |
|---|---|---|---|
| `rep1` remaining stages | **go** (running, gated) | D75's registered H1/H2 | — |
| D82 extended forcing | **go** | Whether a cut-off 48-token answer is "no answer" or "not yet reached", and so whether every O3 reading depends on the budget | after `rep1`'s GPU stages; `main_nla240` first |
| D83 Stage 1 | **conditional go** | Whether the text adds nothing beyond a same-activation probe at outcome-blind positions, with an equivalence margin of ±0.01 AUC | researcher review; `rep1` information audit done first |
| H3 tail under 256-token forcing | **go if E3 shows `f` in the mixed band or above** | Whether the tail itself moves with the budget | after D82 |
| Any larger NLA run aimed at efficiency | **no-go** | none: one check costs 0.83 of a trace, so no outcome could show a compute saving here | — |
| RQ3 Stage 1 | **no-go** | none: Stage 0 changed no answers (D77) | — |

## 6. Commands for the `rep1` secondary analyses (after D75)

```
python scripts/information_audit.py --config replication --forced-dir runs/rep1_forced
python scripts/information_audit.py --config replication --forced-dir runs/rep1_forced --post-hoc
python scripts/information_audit.py --config replication --forced-dir runs/rep1_forced \
    --lexicon runs/main_nla240_information/lexicon.json
python scripts/selective_risk_menu.py --config replication --forced-dir runs/rep1_forced
python scripts/force_extended.py --config replication --forced-dir runs/rep1_forced      # GPU
python scripts/extended_forcing_analysis.py --config replication --forced-dir runs/rep1_forced --confirmatory
```

`main_nla240`'s extended pass:

```
python scripts/force_extended.py --config main --run-id main_nla240 --forced-dir runs/main_forced   # GPU
python scripts/extended_forcing_analysis.py --config main --run-id main_nla240 --forced-dir runs/main_forced
```

Sources consulted for §2's novelty note:
- [Probing for self-verification](https://arxiv.org/pdf/2504.05419)
- [LYNX](https://arxiv.org/html/2512.05325v1)
- [NEAT](https://arxiv.org/html/2602.02010v1)
- [ThinkBrake](https://arxiv.org/pdf/2510.00546)
- [EAT](https://arxiv.org/abs/2509.26522)
- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/)
