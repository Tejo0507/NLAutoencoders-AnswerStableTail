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
- **[C]**: confirmatory (`rep1`, D94);
- **[S]**: pre-specified secondary or sensitivity (`rep1`, D94–D96);
- **[X]**: exploratory;
- **[P]**: post hoc;
- **[A]**: analytic.

**Manuscript files (`paper/v2/`), all drafted as of 2026-10-23:**

| § | file | state |
|---|---|---|
| Abstract | `00_abstract.md` | draft v1 (2026-10-23) |
| 1 | `01_introduction.md` | draft v3; contributions led by the replication; §1.1 contribution statement |
| 2 | `02_related_work.md` | draft v3; DEER, Dynasor, ThinkBrake, EAT, NEAT, LYNX added and verified ([33]–[38]) |
| 3 | `03_methods.md` | draft v3; Stage 1 and D82 described as run |
| 4 | `04_results.md` | draft v3 (reordered around the replication) |
| 5 | `05_discussion.md` | draft v1; includes the conclusion (§5.6) |
| 6 | `06_limitations.md` | draft v3 (renamed from `08_limitations.md`) |
| 7 | `07_reproducibility.md` | draft v1; the rep1 archive identifier is added once published |
| Refs | `../references.md` [1]–[27], `references_additions.md` [28]–[42] | checked against primary sources; [39]–[42] (model, NF4, Holm, TOST) added 2026-10-23 as missing citations |

## 1. Introduction (about 1 page)

Final text: `paper/v2/01_introduction.md`. The five reported findings lead
with the registered replication [C] and its two qualifications [S], then
the score [X, S], the text [X, S], the three evaluation pitfalls [A, X, S]
and cost [A].

### Contribution statement (final draft, 2026-10-23)

In `paper/v2/01_introduction.md` §1.1. It credits prior work for charging
signal cost and matching risk [28, 29], correctness probes [6],
same-activation probe controls [12], truncate-and-answer [16, 31] and
persistence of agreement [22], and claims four things:
1. the confirmatory test with its pre-specified qualifications [C, S];
2. the measured budget dependence of the forced-answer instrument (f =
   0.867 on rep1, 0.881 on the 240 arm) [S, X];
3. two demonstrated ways an evaluation manufactures a stopping advantage
   [A, X];
4. an incremental-value test of NLA text against a same-activation,
   same-label probe at outcome-blind positions (equivalent within ±0.01)
   [S].

The claims are bounded to one target model and one released NLA, and to
predictive content. There is no faithfulness claim (D87; matrix E1).

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
  NEAT, EAT, DEER, Dynasor, ThinkBrake (swept 2026-10-23; [33]–[38]).
- Overthinking analyses.
- NLAs and activation verbalisers (LatentQA-style).
- Selective prediction and risk–coverage evaluation.
- **Backlog E4:** the review is focused, not systematic. Novelty is stated
  as "to our knowledge, within the review of §2".

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

## 5. Discussion (draft: `paper/v2/05_discussion.md`)

- **5.1** What the replication supports: the confirmatory verdict as
  registered, qualified by R2 (not content) and R4 (budget-dependent).
- **5.2** Prediction, incremental value and faithfulness kept apart:
  predictive yes, incremental no (±0.01), faithfulness untested.
- **5.3** Four recommendations for evaluating stopping signals.
- **5.4** Stopping: menu selective risk is not deployment evidence; the
  NLA's cost rules out an online check; the in-sample CPU replay is
  planning evidence only; the stopping-policy study is proposed, not run.
- **5.5** What the tail is: recoverability within a budget (A2, R4).
- **5.6** Conclusion.

## 6. Limitations and threats to validity (draft: `paper/v2/06_limitations.md`)

Written from the matrix's weakest-point column and §E: one model, layer
and NLA; outcome-selected points; budget dependence (with the failed
token-count check); featurisation; no faithfulness or semantic claim
(including C6's null); stopping at deployment untested (R7, the in-sample
replay); post-hoc analyses and D91's provenance gap; independence of the
240 arm; power; engineering; a focused, not systematic, review.

## 7. Reproducibility statement (draft: `paper/v2/07_reproducibility.md`)

- Every number traces to a JSON output named in `docs/DECISIONS.md`
  (D69–D96), and every plan was hashed before computation.
- `check_run_consistency.py --replication --complete` checks the readings,
  gates and listed numbers (169 checks). It does not read the manuscript.
- The rep1 archive identifier is added once the archive is published and
  verified.

## Figure and table plan

| | content | source | state |
|---|---|---|---|
| Fig 1 | Schematic: trace, chunks, `i*`, the menu's windows (from the code's rules), Stage 1's outcome-blind draw, 48- and 256-token forcing, the NLA path | `fig_v2_0` | done (2026-10-23) |
| Fig 2 | Registered estimand curves; stops at 0.5·B\* | `fig_v2_2` | done (exploratory data) |
| Fig 3 | Information increments, two labels | `fig_v2_1` | done |
| Fig 4 | Stage 0 policy gaps to no signal | `fig_v2_4` | done; rule names in words (2026-10-23) |
| Fig 5 | Selective risk against coverage, menu and Stage 0 (`main` arms; post hoc). Not deployment evidence | `fig_v2_5` | done; labels polished (2026-10-23) |
| Fig 6 | Truncation: main's 48-token cut-off and disagreement rates; D82 panel (pre-tail cut-offs at 256 tokens: f = 0.867 on rep1 [S], 0.881 on the 240 arm [X]) | `fig_v2_3` | done; D82 panel added (2026-10-23) |
| Fig 7 | rep1: D75 H1/H2 on the 48-token (confirmatory) and 256-token (sensitivity) outcomes with the ±0.05 margin; Stage 1's increments with the ±0.01 margin. The panels have different units | `fig_v2_6` | done (2026-10-23) |
| Table 1 | Setting (main and rep1) and preregistration timeline (D66–D96) | configs; DECISIONS | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |
| Table 2 | Frozen phase lexicon and its `rep1` transfer AUC | `lexicon.json`; `rep1` | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |
| Table 3 | D75 confirmatory H1/H2 verbatim, plus readings under O256, D79 and Stage 1 | `rep1_o3_registered`, `rep1_extforced`, `rep1_position_audit`, `rep1_stage1` | done; generated in `paper/v2/tables_generated.md` (2026-10-23) |

`python scripts/make_paper_figures_v2.py` regenerates every figure, and
`python scripts/make_paper_tables_v2.py` regenerates Tables 1–3, from the
configs, the decision record and stored JSON. To read the outputs from another checkout, add
`--runs <path>`; the figures are still written to this checkout.

**Caption notes for Fig 7:**
- The 256-token points are the D82 sensitivity analysis, which never
  replaces the registered verdict.
- "Equivalent within ±0.05" excludes effects of 0.05 or more, not smaller
  ones.
- Stage 1's comparator is the same-label probe, not the registered
  correctness probe.
