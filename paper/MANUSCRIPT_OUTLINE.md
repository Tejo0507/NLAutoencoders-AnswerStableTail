# Manuscript outline, revised framing (direction E, 2026-10-20)

**Working title:** *Do verbalised activations know when reasoning is done?
Natural-language autoencoders, cheap probes, and how stopping evaluations
decide the answer*

**The registered report stays frozen** (`paper/REPORT.md`, D66). It is the
record of what was pre-registered and found. This outline is for a new
manuscript. It cites the frozen report and labels every result by its status
in `paper/CLAIM_EVIDENCE_MATRIX.md`.

**Markers:**
- **[R]**: registered;
- **[C]**: confirmatory (`rep1`, pending);
- **[X]**: exploratory;
- **[P]**: post hoc;
- **[A]**: analytic;
- **[pending]**: not yet computed.

## 1. Introduction (about 1 page)

- Reasoning models keep writing after their answer is settled. Stopping
  earlier needs a signal for "done".
- NLAs (Anthropic, 2026) offer natural-language access to activations.
  Can such a readout tell when the answer is settled, and does it add
  anything beyond cheap probes?
- **Contributions:**
  1. **A pre-registered comparison and a confirmatory replication.**
     - The verbalised readout's stopping score carries no information
       beyond position and a probe [X; C pending].
     - Its earlier apparent advantage is reproduced by a rule with no
       readout [X].
  2. **What the text does carry.** The explanation text legibly encodes
     reasoning phase [X]. But a linear probe on the same activation, trained
     for the same label, subsumes it [P], and an online check costs about a
     whole trace [A].
  3. **Three evaluation pitfalls, each demonstrated:**
     - candidate points derived from the outcome;
     - matched-mean-saving estimands that reward abandoning problems;
     - short forced-answer budgets that are reliable but invalid.

## 2. Related work

- Probe-based and training-free early exit: self-verification probes, LYNX,
  NEAT, EAT, DEER, Dynasor, ThinkBrake.
- Overthinking analyses.
- NLAs and activation verbalisers (LatentQA-style).
- Selective prediction and risk–coverage evaluation.
- **Backlog E4:** a systematic review is required before the novelty claims
  in §1 are final.

## 3. Setting and definitions (Methods I)

**Model and corpus:** Qwen2.5-7B-Instruct (NF4), greedy 768-token traces,
GSM8K and MATH (algebra, counting and probability), chunking (`main`: 250
problems).

**Answer-stable tail [R]:**
- K = 5 continuations at T = 0.8, 200 tokens;
- 48-token forcing;
- the tail start is `i*`.

**Registered safety outcome O48 [R]:** a stop at `s` is safe iff the answer
forced from `P_s` has the same correctness as the final answer.

**NLA:**
- the released L20 verbaliser and reconstructor;
- 2 samples at 256 tokens;
- the readout score is reconstruction cosine [R].

**Baselines [R]:**
- convergence;
- semantic entropy;
- a correctness probe trained on intermediate parsed answers.

**Table 1** (setting summary): from `configs/main.yaml`, `configs/replication.yaml` and D75/D78.

## 4. The registered comparison and why it was uninformative (Results I)

- The registered O3 test [R] used budgets above the oracle bound, where a
  no-signal rule wins (D68).
- Rescored on O48 (D69) [P]: readout − no signal +0.052; readout − probe
  unresolved.
- **The menu (design A) contains `i* − 1` and `i*` by construction** [A].
  This is figure 1, a schematic still to be made.

## 5. What the readout score carries (Results II)

**Falling rule (D79) [X]:**
- falling rule − no signal +0.082 and +0.100;
- readout − falling −0.024 [−0.070, 0.015].

**Information (D81 Q1) [X]:** ΔAUC −0.001 [−0.002, −0.000].

**Selective risk [P]:** when the readout stops, the stop is more often
unsafe than chance: +0.14 [0.05, 0.21].

**Figures:**
- **Fig 2** = `paper/figures_v2/fig_v2_2_registered_estimand.png`;
- **Fig 5 (left)** = `fig_v2_5_selective_risk.png`.

**Confirmatory (`rep1`, D75):** H1 and H2 as registered [C, pending].
Reported first and verbatim, then read with the controls above [pending].

## 6. What the verbalised text carries (Results III)

**Text over position and raw text [X]:**
- +0.073 [0.058, 0.087] on safety;
- robust to multiplicity, to the whole prefix as control, and to a noise
  null [P].

**The phase vocabulary [X]:** **Table 2**, the frozen lexicon. Transfer to
`rep1` [pending, pre-specified].

**Text against a same-activation probe [P]:**
- −0.001 [−0.004, 0.002];
- the probe is better head to head;
- the ensemble is no better.

**No latent answer (Q3) [X]:** 3.1% against 4.0%.

**Figure:** **Fig 3** = `fig_v2_1_information_increments.png`.

**Pending:**
- the `rep1` replication of all of the above [pre-specified];
- Stage 1 off the menu, if run (H1).

## 7. Evaluation pitfalls, demonstrated (Results IV)

1. **Outcome-derived candidate points** (§4–5) [A, X].
2. **The matched-mean-saving estimand** (Stage 0, every boundary) [X]:
   - an information-free rule beats an AUC-0.861 activation probe:
     −0.093 [−0.147, −0.043] on net savings;
   - selective risk at matched coverage reverses the ranking;
   - **Fig 4** = `fig_v2_4_stage0_policies.png`, **Fig 5 (right)**.
3. **Short forced-answer budgets** [X]:
   - 54–57% of pre-tail completions are cut off;
   - 98–99% of those disagree with the continuations;
   - **Fig 6** = `fig_v2_3_truncation.png`;
   - **D82 measured effect** [pending].
4. **Cost** [A]: one NLA check costs 0.83 of a mean trace.

**Recommendations for evaluating stopping signals:**
- choose candidate points blind to the outcome;
- report the saving–risk frontier and selective risk, not one matched
  estimand;
- report the forcing budget and the truncation rate, and charge the forced
  tokens;
- compare against a same-label probe on the same activations.

## 8. Discussion

- **Legible but not incremental.** What NLAs add is interpretive access, not
  information.
- **Why a good state classifier saves little safely.** Its confident-safe
  region is the end of the trace (Stage 0 selective risk).
- **What the tail is:** recoverability within a budget (A2; D82 pending).

## 9. Limitations and threats to validity

The draft paragraphs below can be used as written once the pending items
report.

- **Single model, layer and NLA.** Every result concerns Qwen2.5-7B-Instruct
  in NF4 and the released layer-20 autoencoder. Other models, and reasoning
  models that verify in prose, may differ. The bf16 check covered only the
  target's activations (D55).
- **Outcome-selected points.** Every NLA result was measured at chunks
  chosen around the tail start. The cheap-signal results were reproduced
  without that selection (Stage 0); the NLA results were not, unless Stage 1
  runs.
- **Outcome definition.** Safety uses a 48-token forced answer that is cut
  off for most pre-tail positions. Conclusions that change under the
  extended budget are flagged (D82).
- **Featurisation.** The text is represented with TF-IDF and a linear model,
  and the activation probes are linear. "No increment" means no increment
  that these models can extract.
- **Post-hoc analyses.** The falling-position control, the same-activation
  probe and selective risk were added after related results were seen. Each
  was specified before any outcome existed for the fresh-problem
  replication, which is their test.
- **String-level content checks.** The lexicon and the answer-mention test
  match strings. Neither establishes that an explanation is faithful to the
  computation, and the paper makes no faithfulness claim.
- **Power.** The registered H1 has power of about 0.45 for the gap D69
  estimated. An inconclusive H1 is not evidence of equivalence.

## 10. Reproducibility statement

- Every number traces to a JSON output with its SHA-256 recorded in
  `docs/DECISIONS.md` (D69–D84).
- Every plan was hashed before computation.
- Commands are in `paper/PUBLICATION_STRATEGY.md` §6 and in each script's
  docstring.

## Figure and table plan

| | content | source | state |
|---|---|---|---|
| Fig 1 | Schematic: trace, chunks, `i*`, the menu's windows, forcing | to draw | todo |
| Fig 2 | Registered estimand curves; stops at 0.5·B\* | `fig_v2_2` | done (exploratory data) |
| Fig 3 | Information increments, two labels | `fig_v2_1` | done |
| Fig 4 | Stage 0 policy gaps to no signal | `fig_v2_4` | done |
| Fig 5 | Selective risk against coverage, menu and Stage 0 | `fig_v2_5` | done (polish labels) |
| Fig 6 | Truncation | `fig_v2_3` | done; add the D82 panel when measured |
| Table 1 | Setting and preregistration timeline (D66–D84) | DECISIONS | todo |
| Table 2 | Frozen phase lexicon and its `rep1` transfer AUC | `lexicon.json`; `rep1` | pending |
| Table 3 | D75 confirmatory H1/H2 verbatim, plus readings under O256 | `rep1_o3_registered`, `rep1_extforced` | pending |

`python scripts/make_paper_figures_v2.py` regenerates every figure from
stored JSON.
