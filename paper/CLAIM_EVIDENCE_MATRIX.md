# Claim–evidence matrix (2026-10-20; rep1 results and D76 added 2026-10-23; the stopping-policy pilot added 2026-10-25)

Every claim the revised paper could make is mapped here to its evidence, its
status, its weakest point and the test that would settle it. **No claim
should appear in the manuscript in a stronger form than its row allows.**

**Status labels:**

| label | meaning |
|---|---|
| **registered** | in the pre-registered plan, computed as planned |
| **confirmatory** | pre-specified before the data (D75); computed 2026-10-22 (D94) |
| **sensitivity** | pre-specified to qualify a confirmatory result; never replaces it (D82) |
| **corrected / post hoc** | the analysis was fixed after related results were seen |
| **exploratory** | specified before computation, but on data whose other results were known |
| **exploratory (frozen protocol)** | the stopping-policy pilot: fresh data and a protocol frozen before any data (D99), designated exploratory by that protocol |
| **analytic** | follows from measured costs or from the construction of the code |

All `main_nla240` results include `main`'s 120 problems. They are not
independent replications of `main`.

## A. The answer-stable tail and its measurement

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| A1 | Tails are common and long on this model | Non-trivial on 0.948 of 250 problems; mean tail fraction 0.378 (`runs/main/analysis/results.json`) | registered | The tail is defined with 48-token forcing, so its start is a floor that moves with the budget | Tail starts recomputed under 256-token forcing (backlog H3; R4 now indicates it) |
| A2 | The tail marks where the answer becomes *recoverable within 48 tokens*, not redundant post-answer text | Tails begin before the answer is written (D68). Before `i*`, 54–56% of forced completions are cut off at 48 tokens (0.543 on `main`, 0.565 on the 240 arm; recorded as "54–57%" until D98 corrected a rounding error), and 98–99% of those disagree with the 200-token continuations' majority, which equals the final answer 96–98% of the time (D79 B) | exploratory | Whether a cut-off answer is "no answer" or "an answer not yet reached" is unmeasured | Done: R4 (f = 0.867 on rep1; 0.881 on the 240 arm) |
| A3 | The 48-token forced outcome is reliable but not valid before the tail | Cut-off answers re-force identically 100% of the time (22 and 52), against 94–96% for finished ones; validity as in A2 (D79) | exploratory | As A2 | Done: R4 |

## B. The registered stopping comparison (O3)

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| B1 | The registered O3 test is uninformative about incremental value | Its budgets (0.61, 0.72) lie above the oracle's 0.17, where a no-signal rule beats every compared rule (D68) | registered result, read post hoc | — | none needed |
| B2 | On the registered outcome, the readout's score beats a no-signal rule | +0.052 [0.022, 0.082] at N = 120; +0.069 [0.049, 0.093] at N = 240 (D69) | corrected / post hoc | Reproduced by a rule with no readout (B4); the menu leaks the outcome (B5) | Done: R1, read with R2 and R4 |
| B3 | Readout against probe | Unresolved: +0.027 [−0.026, 0.076]; +0.006 against a layer-24 probe (D69, D73) | corrected / post hoc | Power ≈ 0.45 for a true gap of 0.027 at N = 400 | Done: R1, read with R4 |
| B4 | A rule with no readout matches the readout's advantage over no signal | Falling rule − no signal: +0.082 and +0.100; readout − falling: −0.024 [−0.070, 0.015] (D79) | exploratory; pre-specified for `rep1` | Post hoc on these runs | Done: R2 |
| B5 | Design A's menu leaks the outcome | By construction the menu contains `i* − 1` and `i*` (`ast_detect.matched_windows`, `run_autoencoder.select_windows`) | analytic | — | none; it is a property of the code |
| B6 | The readout "selects safer stops" (D69's wording) is **false** | At 0.5·B\*: the readout's stops are safe in 12 of 32 cases (D79). Selective risk is *higher* than no signal's at every coverage: +0.16 [0.04, 0.26] at 0.5 on `main`, +0.14 [0.05, 0.21] on the 240 arm (D83 deviation 1) | post hoc | — | Done: R7 |
| B7 | The registered estimand rewards concentrating savings in a few problems, whatever the signal | Every boundary, no menu (Stage 0): an information-free falling rule beats a same-label activation probe with boundary AUC 0.861. Probe − falling: −0.032 [−0.083, 0.009] on registered savings, −0.093 [−0.147, −0.043] on net savings (D83) | exploratory | One model and one corpus; the effect follows from short safe regions (B\* ≈ 0.40–0.47) | Same analysis on `rep1` traces (cheap) |
| B8 | Semantic entropy cannot be compared | It has no operating point below the oracle bound: 82 of 120 problems tie at confidence 1.0 (D69) | corrected / post hoc | — | — |

## C. What the NLA readout contains (D81)

All C rows are at the menu's outcome-selected points (B5). None says
anything about a deployable rule. C1–C5 are from D81's primary run, the
**240-problem arm** (`main_nla240`, 1,206 menu chunks), which includes
`main`'s 120 problems; `main` alone agrees on every point (D81). They are
not results of `main` alone (D98).

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| C1 | The readout *score* (reconstruction cosine) adds no stopping information beyond position and the probe | ΔAUC −0.001 [−0.002, −0.000], *no practically relevant increment* (Holm p 0.052). On tail membership −0.000 [−0.002, 0.001]. Its two samples correlate 0.94, so it is not noise | exploratory | Menu only; one scalar reduction of the readout | Done: R5 |
| C2 | The explanation *text* carries information about the reasoning state beyond position and the raw trace text | +0.073 [0.058, 0.087] on safety and +0.078 [0.065, 0.091] on tail membership. Survives one row per unit (+0.073), the whole prefix as control (+0.078) and a noise-text null (−0.003) | exploratory (deviation 1 checks post hoc) | Menu only; TF-IDF featurisation | Done: R5, R6 |
| C3 | That information describes a reasoning **phase**: concluding against setting up | The heaviest terms are "final answer", "concluding", "therefore", "summary" against "first", "let's", "worked example", "expecting continuation". A frozen 40-term lexicon reaches AUC 0.925 for tail membership *in sample* | descriptive, in sample | Not yet tested out of sample. Not a faithfulness claim | Done: R5 (transfers out of sample). Human labels optional |
| C4 | The text adds **nothing** beyond a linear probe on its own input activation trained for the same label | Text over probe: −0.001 [−0.004, 0.002] (safety) and −0.000 [−0.002, 0.001] (tail). Probe beats text head to head by 0.029 and 0.031. An unfitted ensemble does not beat the probe (−0.005) | post hoc (deviations 1–2) | Menu only; a linear probe and TF-IDF text | Done: R6 |
| C5 | Explanations do not reveal the final answer before the trace writes it | Own-answer mention 3.1% against 4.0% for other problems' answers: −0.010 [−0.022, 0.004]. Above noise text (+0.028) only because real explanations contain more numbers | exploratory | String matching; a paraphrased answer would be missed | Done: R5 |
| C6 | Most explanation claims carry reconstruction weight | 0.749 of 6,007, robust to clustering and invalid nulls (0.73–0.78). D76 (exploratory, 60 explanations, 246 claims): under a model-written rewording null the rate is 0.764 [0.685, 0.834], against 0.797 [0.729, 0.856] under the registered null on the same claims; rewordings move reconstruction about 3× as much as rule-based paraphrases (mean \|effect\| 0.0045 vs 0.0015), deletions 0.0267 (D97) | registered, with post-hoc corrections (D70–D72); D76 exploratory | Reconstruction dependence is not faithfulness. The rewordings' meaning preservation is unvalidated by people, and only 117 of 246 claims received an accepted rewording | Human judgement of the 313 rewordings (`rewording_validation_sheet.csv`); human annotation (`docs/ANNOTATOR_PACKET.md`) |
| C7 | Some explanation content checks out against the trace | The next word is right on 52% of explanations, against 3–10% under controls; "quoted" context is rarely verbatim (D71) | post hoc | String checks, not semantics | Human annotation |

## D. Cheap signals and cost

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| D1 | At outcome-blind positions, a same-label linear probe on the layer-20 activation carries substantial safety information | Every boundary of 250 problems: +0.091 [0.066, 0.115] over position (AUC 0.861); +0.123 on tail membership (AUC 0.975). Raw trace text adds nothing (−0.002) (D83 Stage 0) | exploratory | One model and layer | `rep1` traces (cheap) |
| D2 | An online rule that verbalises at each candidate cannot save compute on this setup | One check generates 2 × 144.9 verbaliser tokens, 0.89 of the mean 324.8-token trace, before reconstruction (D98). D83's "2 × 134 tokens, 0.83" counted only the extracted explanation text (134.3 tokens on average), not the delimiters and end token also generated | analytic | Holds for this verbaliser length and these trace lengths only | none |
| D3 | The registered correctness probe is a weak representation of the state | It adds +0.026 over position at every boundary, against +0.091 for the same-label probe, and +0.018 AUC above a positional floor (registered) | registered + exploratory | It was trained for a different target | — |

## R. The replication (`rep1`, 400 fresh problems), computed 2026-10-22/23

Every number below was copied from the output named in D94–D96 and checked
by review. The numbers in this table are **not** automatically checked.
`scripts/check_run_consistency.py --replication` checks the readings, the
gates and a listed subset of the numbers as written in `docs/DECISIONS.md`
(D96 §4).

The classes stay separate: R1 is the only confirmatory result, and nothing
else may be reported as one.

Two probes appear below:
- the **registered probe** is the layer-20 correctness probe;
- the **same-label probe** is a linear probe on the same layer-20
  activation, trained for the label being predicted.

| # | claim | evidence | status | weakest point | what it allows |
|---|---|---|---|---|---|
| R1 | On the registered 48-token outcome, the readout beats the registered probe (H1) and no signal (H2) | H1 +0.047 [0.019, 0.075], Holm p 0.002. H2 +0.057 [0.039, 0.076], Holm p 0.001. N = 380; re-forcing agreement 0.965 (D94) | **confirmatory** | H1's CI spans the 0.05 margin. Qualified by R2 and R4 | "supported as registered, on the 48-token outcome". No more |
| R2 | That advantage cannot be attributed to explanation content | A position-only falling rule beats the readout by 0.025 [0.005, 0.044]. The position-residualised readout − no signal is +0.007 [−0.016, 0.029] (D79 families X and Y: unresolved; D94) | secondary, pre-specified | Linear-in-log-position residualisation only | "not shown to come from content" |
| R3 | The readout's *score* adds nothing beyond position and the registered probe; the *text* does add information beyond position and trace text | Q1-P +0.0005 [−0.0015, 0.0025]; Q2-P +0.077 [0.063, 0.090]; Q3-P −0.002 [−0.016, 0.012] (D95) | secondary, pre-specified | Menu points (B5) | Reproduces C1, C2, C5 out of sample |
| R4 | The registered outcome measures commitment within 48 tokens, and the registered hypotheses are not supported under a 256-token budget | f = 0.867: 520 of the 600 cut-off boundaries before the tail reach the final answer at 256 tokens. Under O256: H1 −0.008 [−0.036, 0.024], H2 −0.004 [−0.024, 0.016], both *equivalent within ±0.05*. The 48-token estimates lie outside these intervals (D95). Same on the 240 arm: f = 0.881; readout − registered probe +0.001 [−0.031, 0.040] (D96, exploratory) | sensitivity (rep1); exploratory (240 arm) | Net savings unusable (token-count check failed); registered savings used. Effects smaller than 0.05 are not ruled out | R1 stands as registered, worded as being about 48-token commitment. No claim of zero effect at 256 |
| R5 | The phase lexicon transfers out of sample | Position plus lexicon over position: +0.091 [0.076, 0.106] (tail), +0.079 [0.059, 0.099] (safety); 0 problems shared (D95) | secondary, pre-specified | Menu points; string features | C3 holds out of sample. Not a faithfulness claim |
| R6 | At outcome-blind positions, for predicting these stopping labels, the text adds no practically relevant information beyond position and the same-label probe | Δ₁ +0.00002 [−0.004, 0.004], *equivalent within ±0.01*. Text over position +0.071; same-label probe over position +0.102; text − same-label probe −0.031 [−0.047, −0.015] (Stage 1, D96) | pre-specified (D83, approved D85) | One model, layer and NLA; a linear probe and TF-IDF text | "no practically relevant predictive information for these labels beyond the same-label probe". Not that everything the text says is in the activation. Says nothing on faithfulness |
| R7 | On the candidate menu with O48 labels, no rule dominates on stopping risk and saving together | At 50% coverage: registered probe risk 0.011, saving 0.028; readout 0.632, 0.711; no signal 0.472, 0.413 (D95, D96). No selective-risk result exists for the same-label probe | secondary, pre-specified | Outcome-selected menu and O48 labels: **not deployment evidence** | Report risk and saving together; do not rank. Do not cite as evidence about deployed stopping, or about the same-label probe |

## S. The stopping-policy pilot (fresh problems; D99–D102), computed 2026-10-25

Exploratory by its frozen protocol (D99): it estimates the inputs a
confirmatory study would need and **is not a safety certification**. The
NLA is not part of it. Every number below is bound to the pilot's outputs
in `paper/v2/evidence/key_numbers.json` (N85–N111) and checked by
`scripts/verify_manuscript_numbers.py --pilot`.

| # | claim | evidence | status | weakest point | what it allows |
|---|---|---|---|---|---|
| S1 | Position alone cannot be calibrated to a 0.05 selective-risk target with 300 calibration problems | No SGR step accepted; lowest bound 0.158 (201 stops, 19 unsafe), so P0 never stops (D101) | exploratory (frozen protocol) | One calibration set; SGR at δ = 0.05 | "P0 could not be calibrated at α = 0.05", not "position carries no stopping information" |
| S2 | A same-label-probe policy calibrated on fresh problems keeps its observed out-of-sample risk below 0.05, but the pilot cannot bound it there | τ = 0.945. Evaluation: 148 of 150 stop; 6 unsafe; risk 0.041 [0.015, 0.086]; one-sided upper bound 0.078 > α = 0.05; criterion (1) of protocol §3.9 **not met** (D101) | exploratory (frozen protocol) | 148 stops; one model under NF4; two benchmarks; α set for the pilot without a use case | "the safety criterion was not met"; "a confirmatory study as designed is not warranted". **Not** "safe", and not "unsafe" |
| S3 | At that threshold the policy saves a modest share of tokens | Net saving 0.055 [0.046, 0.065] against P0 (which never stops), above δ_save = 0.02 (`c2_completed.json`). Stops come late: 0.861 of the trace on average (post hoc, D102); 11 of 148 stops save nothing once the forced answer is charged, so 137 of 150 problems (0.913) have a saving stop (post hoc, D103) | exploratory; the paired estimand completed outside `evaluate` (D101) | The probe's own compute is not charged; coverage (0.987) overstates useful stops | Report with S2, never alone; quote the saving-stop share beside coverage |
| S4 | The 256-token forced-answer label has a small batch-composition component | Single-boundary re-forcing 0.92 (46 of 50); all 4 disagreements were pair-forced boundaries; paired re-forcing 50/50 on both sets (D100) | instrument measurement | 36 pair-forced boundaries | A caveat on every forced-answer label under NF4 batching |

## E. Not tested, and must not be claimed

| # | not tested | why |
|---|---|---|
| E1 | That the NLA explanations are **faithful** to the computation | No human or causal validation. Reconstruction dependence and string checks do not establish it |
| E2 | Any **causal** role of a tail direction (RQ3) | Not tested on this model (D62, D77) |
| E3 | Generalisation to other models, layers or autoencoders | One target model, one released NLA pair |
| E4 | That the readout improves stopping **at deployment** | The NLA stopping results are on the outcome-selected menu, and D2 rules out compute savings. Off the menu (R6), the text adds no practically relevant predictive information over the same-label probe. Selective risk (R7) is on the menu |
| E6 | That **any** stopping policy studied here is safe, or that the pilot certifies one | The only out-of-sample estimate (S2) did not meet its safety criterion, and the pilot was designed as exploratory |
| E5 | That R1 shows the explanations' content helps stopping | R2 and R4: a position rule beats the readout, and under a 256-token budget the hypotheses are not supported (effects of 0.05 or more excluded) |

## F. Statements in the record that exceed the evidence

Corrected in new summaries. Historical documents are left as written.

| where | statement | correction |
|---|---|---|
| `docs/REANALYSIS_2026-10-19.md` | "the readout's score selects safer stops than chance" | Its stops are mostly unsafe. The advantage comes from abstention and a few early stops (B4, B6). Corrected in `docs/POSITION_TRUNCATION_RESULTS.md` and here |
| `README.md` (uncommitted D69 text) | "So the readout carries stopping-relevant information" | Its score's advantage is reproduced by a rule with no readout, and it carries no information beyond position and the probe (C1). Corrected in the README update of 2026-10-20 |
| `docs/REPLICATION_PLAN.md` §1 | "selects safer stops than chance" (frozen plan text) | Frozen and not edited. The `rep1` result is read with B4–B7 |
| D73 | "The readout's score beating chance is robust" | Robust across seeds and arms, but not to the falling-rule control (B4) or the forcing budget (A2) |
