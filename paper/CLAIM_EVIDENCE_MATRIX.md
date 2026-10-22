# Claim–evidence matrix (2026-10-20)

Every claim the revised paper could make is mapped here to its evidence, its
status, its weakest point and the test that would settle it. **No claim
should appear in the manuscript in a stronger form than its row allows.**

**Status labels:**

| label | meaning |
|---|---|
| **registered** | in the pre-registered plan, computed as planned |
| **confirmatory** | pre-specified before the data (D75), not yet computed |
| **corrected / post hoc** | the analysis was fixed after related results were seen |
| **exploratory** | specified before computation, but on data whose other results were known |
| **analytic** | follows from measured costs or from the construction of the code |

All `main_nla240` results include `main`'s 120 problems. They are not
independent replications of `main`.

## A. The answer-stable tail and its measurement

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| A1 | Tails are common and long on this model | Non-trivial on 0.948 of 250 problems; mean tail fraction 0.378 (`runs/main/analysis/results.json`) | registered | The tail is defined with 48-token forcing, so its start is a floor that moves with the budget | Tail starts recomputed under 256-token forcing (not yet planned; backlog H3) |
| A2 | The tail marks where the answer becomes *recoverable within 48 tokens*, not redundant post-answer text | Tails begin before the answer is written (D68). Before `i*`, 54–57% of forced completions are cut off at 48 tokens, and 98–99% of those disagree with the 200-token continuations' majority, which equals the final answer 96–98% of the time (D79 B) | exploratory | Whether a cut-off answer is "no answer" or "an answer not yet reached" is unmeasured | **D82 extended pass**: its `f` statistic decides the wording |
| A3 | The 48-token forced outcome is reliable but not valid before the tail | Cut-off answers re-force identically 100% of the time (22 and 52), against 94–96% for finished ones; validity as in A2 (D79) | exploratory | As A2 | D82 |

## B. The registered stopping comparison (O3)

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| B1 | The registered O3 test is uninformative about incremental value | Its budgets (0.61, 0.72) lie above the oracle's 0.17, where a no-signal rule beats every compared rule (D68) | registered result, read post hoc | — | none needed |
| B2 | On the registered outcome, the readout's score beats a no-signal rule | +0.052 [0.022, 0.082] at N = 120; +0.069 [0.049, 0.093] at N = 240 (D69) | corrected / post hoc | Reproduced by a rule with no readout (B4); the menu leaks the outcome (B5) | **D75 H2 on `rep1`** (confirmatory); read with B4–B6 |
| B3 | Readout against probe | Unresolved: +0.027 [−0.026, 0.076]; +0.006 against a layer-24 probe (D69, D73) | corrected / post hoc | Power ≈ 0.45 for a true gap of 0.027 at N = 400 | **D75 H1 on `rep1`** (confirmatory) |
| B4 | A rule with no readout matches the readout's advantage over no signal | Falling rule − no signal: +0.082 and +0.100; readout − falling: −0.024 [−0.070, 0.015] (D79) | exploratory; pre-specified for `rep1` | Post hoc on these runs | D79 family Y on `rep1` |
| B5 | Design A's menu leaks the outcome | By construction the menu contains `i* − 1` and `i*` (`ast_detect.matched_windows`, `run_autoencoder.select_windows`) | analytic | — | none; it is a property of the code |
| B6 | The readout "selects safer stops" (D69's wording) is **false** | At 0.5·B\*: the readout's stops are safe in 12 of 32 cases (D79). Selective risk is *higher* than no signal's at every coverage: +0.16 [0.04, 0.26] at 0.5 on `main`, +0.14 [0.05, 0.21] on the 240 arm (D83 deviation 1) | post hoc | — | Selective risk on `rep1` (pre-specified) |
| B7 | The registered estimand rewards concentrating savings in a few problems, whatever the signal | Every boundary, no menu (Stage 0): an information-free falling rule beats a same-label activation probe with boundary AUC 0.861. Probe − falling: −0.032 [−0.083, 0.009] on registered savings, −0.093 [−0.147, −0.043] on net savings (D83) | exploratory | One model and one corpus; the effect follows from short safe regions (B\* ≈ 0.40–0.47) | Same analysis on `rep1` traces (cheap) |
| B8 | Semantic entropy cannot be compared | It has no operating point below the oracle bound: 82 of 120 problems tie at confidence 1.0 (D69) | corrected / post hoc | — | — |

## C. What the NLA readout contains (D81)

All C rows are at the menu's outcome-selected points (B5). None says
anything about a deployable rule.

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| C1 | The readout *score* (reconstruction cosine) adds no stopping information beyond position and the probe | ΔAUC −0.001 [−0.002, −0.000], *no practically relevant increment* (Holm p 0.052). On tail membership −0.000 [−0.002, 0.001]. Its two samples correlate 0.94, so it is not noise | exploratory | Menu only; one scalar reduction of the readout | `rep1` Q1-P (pre-specified) |
| C2 | The explanation *text* carries information about the reasoning state beyond position and the raw trace text | +0.073 [0.058, 0.087] on safety and +0.078 [0.065, 0.091] on tail membership. Survives one row per unit (+0.073), the whole prefix as control (+0.078) and a noise-text null (−0.003) | exploratory (deviation 1 checks post hoc) | Menu only; TF-IDF featurisation | `rep1` Q2-P; Stage 1 off the menu |
| C3 | That information describes a reasoning **phase**: concluding against setting up | The heaviest terms are "final answer", "concluding", "therefore", "summary" against "first", "let's", "worked example", "expecting continuation". A frozen 40-term lexicon reaches AUC 0.925 for tail membership *in sample* | descriptive, in sample | Not yet tested out of sample. Not a faithfulness claim | **Lexicon transfer L-P on `rep1`** (pre-specified, deviation 3); human labels (optional) |
| C4 | The text adds **nothing** beyond a linear probe on its own input activation trained for the same label | Text over probe: −0.001 [−0.004, 0.002] (safety) and −0.000 [−0.002, 0.001] (tail). Probe beats text head to head by 0.029 and 0.031. An unfitted ensemble does not beat the probe (−0.005) | post hoc (deviations 1–2) | Menu only; a linear probe and TF-IDF text | `rep1` (pre-specified); **Stage 1** at outcome-blind positions |
| C5 | Explanations do not reveal the final answer before the trace writes it | Own-answer mention 3.1% against 4.0% for other problems' answers: −0.010 [−0.022, 0.004]. Above noise text (+0.028) only because real explanations contain more numbers | exploratory | String matching; a paraphrased answer would be missed | `rep1` Q3-P |
| C6 | Most explanation claims carry reconstruction weight | 0.749 of 6,007, robust to clustering and invalid nulls (0.73–0.78) | registered, with post-hoc corrections (D70–D72) | Reconstruction dependence is not faithfulness; the rewording null is unvalidated by people | Human annotation (`docs/ANNOTATOR_PACKET.md`) |
| C7 | Some explanation content checks out against the trace | The next word is right on 52% of explanations, against 3–10% under controls; "quoted" context is rarely verbatim (D71) | post hoc | String checks, not semantics | Human annotation |

## D. Cheap signals and cost

| # | claim | evidence | status | weakest point | next test |
|---|---|---|---|---|---|
| D1 | At outcome-blind positions, a same-label linear probe on the layer-20 activation carries substantial safety information | Every boundary of 250 problems: +0.091 [0.066, 0.115] over position (AUC 0.861); +0.123 on tail membership (AUC 0.975). Raw trace text adds nothing (−0.002) (D83 Stage 0) | exploratory | One model and layer | `rep1` traces (cheap) |
| D2 | An online rule that verbalises at each candidate cannot save compute on this setup | One check (2 × 134 verbaliser tokens) costs 0.83 of the mean 325-token trace, before reconstruction (D83) | analytic | Holds for this verbaliser length and these trace lengths only | none |
| D3 | The registered correctness probe is a weak representation of the state | It adds +0.026 over position at every boundary, against +0.091 for the same-label probe, and +0.018 AUC above a positional floor (registered) | registered + exploratory | It was trained for a different target | — |

## E. Not tested, and must not be claimed

| # | not tested | why |
|---|---|---|
| E1 | That the NLA explanations are **faithful** to the computation | No human or causal validation. Reconstruction dependence and string checks do not establish it |
| E2 | Any **causal** role of a tail direction (RQ3) | Not tested on this model (D62, D77) |
| E3 | Generalisation to other models, layers or autoencoders | One target model, one released NLA pair |
| E4 | That the readout improves stopping **at deployment** | Every NLA result is on the outcome-selected menu, and D2 rules out compute savings |

## F. Statements in the record that exceed the evidence

Corrected in new summaries. Historical documents are left as written.

| where | statement | correction |
|---|---|---|
| `docs/REANALYSIS_2026-10-19.md` | "the readout's score selects safer stops than chance" | Its stops are mostly unsafe. The advantage comes from abstention and a few early stops (B4, B6). Corrected in `docs/POSITION_TRUNCATION_RESULTS.md` and here |
| `README.md` (uncommitted D69 text) | "So the readout carries stopping-relevant information" | Its score's advantage is reproduced by a rule with no readout, and it carries no information beyond position and the probe (C1). Corrected in the README update of 2026-10-20 |
| `docs/REPLICATION_PLAN.md` §1 | "selects safer stops than chance" (frozen plan text) | Frozen and not edited. The `rep1` result is read with B4–B7 |
| D73 | "The readout's score beating chance is robust" | Robust across seeds and arms, but not to the falling-rule control (B4) or the forcing budget (A2) |
