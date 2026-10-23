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

### Contribution statement (draft, 2026-10-20; conditional on the pending results)

Prior work already establishes four things (`docs/NOVELTY_REVIEW_2026-10-20.md`):
- stopping signals should be charged for their compute and compared at
  matched risk (LearnStop; Thought Calibration);
- hidden-state probes predict intermediate correctness (2504.05419);
- verbalizer evaluations need input and probe controls (Li et al., ICML 2026);
- truncate-and-answer reveals when answers commit (Lanham et al. 2023 and its
  successors).

**This paper does not claim any of these.** It claims:
1. **A measured validity problem in the forced-answer instrument itself.**
   With a 48-token budget, the setting a 2026 early-exit method also uses,
   54–57% of pre-tail forced answers are cut off. They are reproducible but
   disagree with the model's own continuations 98–99% of the time. A
   256-token re-forcing shows how much this changes stopping conclusions
   [D82, pending].
2. **Two evaluation choices that can manufacture a stopping advantage,
   each demonstrated with a control:**
   - Candidate points chosen from the outcome: a rule using no readout
     matches the NLA [D79].
   - A matched-mean-saving estimand that counts abstention as safe: an
     information-free rule beats an AUC-0.86 probe, at every boundary,
     without the menu [D83 Stage 0].
   - Selective risk reverses the rankings in both cases.
3. **The first incremental-value test of NLA verbalisations for
   reasoning-state information, against a same-activation, same-label
   probe.** The control is Li et al.'s.
   - The text carries phase information beyond the raw input text (+0.073
     AUC).
   - It carries none beyond the probe: −0.001 [−0.004, 0.002] on the menu,
     [rep1 replication and Stage 1 off the menu, pending].

The claims are bounded to one target model and one released NLA, and to
string-level predictive content. There is no faithfulness claim (D87).

### Anticipated objections and where each is answered

| objection | answer in the paper | evidence |
|---|---|---|
| not novel | The contribution statement above. General ideas are credited, and only the three specific findings are claimed. Novelty is stated as "to our knowledge", after a full-text review | `NOVELTY_REVIEW` |
| unfair stopping comparison | Every estimand is reported side by side: registered (authoritative), selective risk, position controls, no-menu Stage 0, O48 against O256, registered against net savings. No ranking rests on one estimand | D79, D83, D82; rep1 pre-specified |
| explanations not faithful | No faithfulness or semantic claim. The lexicon is predictive string content, in a separate section, tested out of sample on rep1 | D81 deviation 3, D87 |
| one model / dataset | Claims limited to Qwen2.5-7B-Instruct with its released L20 NLA. Rep1 gives 400 unseen problems; GSM8K and MATH are reported separately | limitations §6 |
| weak implementation | Fidelity: cosine 0.875 against a 0.39 noise floor; FVE 0.696 against the card's 0.752; integrity 92%; sample agreement 0.94. Strong baselines. The conclusion is limited to "what linear and TF-IDF models can extract" | `runs/main_nla240/nla/summary.json`; D81 Q5 |

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

**Fig 1** (`fig_v2_0_schematic`) and **Table 1** (setting and
preregistration timeline, generated from the configs and the decision
record) present the design before any result.

## 4. Results (reordered 2026-10-23; draft: `paper/v2/04_results.md`)

The replication leads; the main run's own comparison is context (§4.6).

| § | content | class | carried by |
|---|---|---|---|
| 4.1 | The registered replication: H1 and H2 supported on O48 | [C] | Table 3a, Fig 7 (left) |
| 4.2 | Qualification 1: candidate position. A position-only rule beats the readout; position-residualised, the readout does not beat no signal | [S] (main: [X]) | Table 3c, Fig 1 |
| 4.3 | Qualification 2: the forcing budget. f = 0.867; under O256 neither hypothesis is supported (effects ≥ 0.05 excluded) | [S] | Table 3b, Fig 7 (left), Fig 6 |
| 4.4 | What the text carries. Predictive on and off the menu; no practically relevant increment beyond a same-label probe (Stage 1, ±0.01) | [S] (main: [X], [P]) | Tables 2, 3d; Figs 3, 7 (right) |
| 4.5 | Selective risk on the candidate menu (O48; not deployment evidence) | [S], [P] | Fig 5 |
| 4.6 | How evaluation choices manufacture an advantage (main run): the uninformative registered O3, outcome-derived menus, the matched-mean-saving estimand, short forcing budgets, cost | [R] read post hoc, [A], [X] | Figs 2, 4, 6 |
| 4.7 | The 240-problem arm under O256 | [X] | — |
| 4.8 | Summary, separating prediction, incremental value and faithfulness | — | — |

**Recommendations for evaluating stopping signals** (§4.6, Discussion):
- choose candidate points blind to the outcome;
- report the saving–risk frontier and selective risk, not one matched
  estimand;
- report the forcing budget and the truncation rate, and charge the forced
  tokens;
- compare against a same-label probe on the same activations.

## 5. Discussion

- **Legible but not incremental.** What NLAs add is interpretive access, not
  information.
- **Why a good state classifier saves little safely.** Its confident-safe
  region is the end of the trace (Stage 0 selective risk).
- **What the tail is:** recoverability within a budget (A2; D82 pending).

## 6. Limitations and threats to validity (draft: `paper/v2/08_limitations.md`)

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

## 7. Reproducibility statement

- Every number traces to a JSON output with its SHA-256 recorded in
  `docs/DECISIONS.md` (D69–D84).
- Every plan was hashed before computation.
- Commands are in `paper/PUBLICATION_STRATEGY.md` §6 and in each script's
  docstring.

## Figure and table plan

| | content | source | state |
|---|---|---|---|
| Fig 1 | Schematic: trace, chunks, `i*`, the menu's windows (from the code's rules), Stage 1's outcome-blind draw, 48- and 256-token forcing, the NLA path | `fig_v2_0` | done (2026-10-23) |
| Fig 2 | Registered estimand curves; stops at 0.5·B\* | `fig_v2_2` | done (exploratory data) |
| Fig 3 | Information increments, two labels | `fig_v2_1` | done |
| Fig 4 | Stage 0 policy gaps to no signal | `fig_v2_4` | done; rule names in words (2026-10-23) |
| Fig 5 | Selective risk against coverage, menu and Stage 0 (`main` arms; post hoc). Not deployment evidence | `fig_v2_5` | done; labels polished (2026-10-23) |
| Fig 6 | Truncation | `fig_v2_3` | done; add the D82 panel when measured |
| Fig 7 | rep1: D75 H1/H2 on the 48-token (confirmatory) and 256-token (sensitivity) outcomes with the ±0.05 margin; Stage 1's increments with the ±0.01 margin. The panels have different units | `fig_v2_6` | done (2026-10-23) |
| Table 1 | Setting (main and rep1) and preregistration timeline (D66–D96) | configs; DECISIONS | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |
| Table 2 | Frozen phase lexicon and its `rep1` transfer AUC | `lexicon.json`; `rep1` | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |
| Table 3 | D75 confirmatory H1/H2 verbatim, plus readings under O256, D79 and Stage 1 | `rep1_o3_registered`, `rep1_extforced`, `rep1_position_audit`, `rep1_stage1` | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |

`python scripts/make_paper_figures_v2.py` regenerates every figure, and
`python scripts/make_paper_tables_v2.py` regenerates Tables 2–3, from stored
JSON. To read the outputs from another checkout, add
`--runs <path>`; the figures are still written to this checkout.

**Caption notes for Fig 7:**
- The 256-token points are the D82 sensitivity analysis, which never
  replaces the registered verdict.
- "Equivalent within ±0.05" excludes effects of 0.05 or more, not smaller
  ones.
- Stage 1's comparator is the same-label probe, not the registered
  correctness probe.
