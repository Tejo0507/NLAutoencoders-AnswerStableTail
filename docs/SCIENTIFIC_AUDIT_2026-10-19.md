# Scientific audit of the frozen report, 2026-10-19

> **Followed up the same day (D69–D74):**
> - O3 has been re-scored on the registered outcome (§5.6's first
>   requirement, §11 rank 1). The readout beats a no-signal rule; its value
>   beyond the probe is unresolved.
> - §8's RQ3 dose–response is a truncation artefact (D74).
> - §6's units have corrected analyses (D72).
>
> See [`REANALYSIS_2026-10-19.md`](REANALYSIS_2026-10-19.md). This document is
> kept as written.

**What this is.** An audit of the frozen report (`32b5539`, D66). For each
headline claim it traces the chain from research question, through the
registered definition, the code, the configuration and the stored artifacts,
to the reported number and its interpretation. **Every analysis here is post
hoc and exploratory** (D68). None changes a registered result, and the report's
section files are not edited. Corrections to interpretation are collected in
[`paper/ADDENDUM_2026-10-19.md`](../paper/ADDENDUM_2026-10-19.md).

**Where the numbers come from.** Every number not otherwise attributed was
produced by:

```powershell
.venv\Scripts\python.exe scripts\audit_claims_posthoc.py --config main
```

The script is read-only and CPU-only, and runs in roughly 20–30 minutes. It writes
`runs/main_claims_audit/claims_audit.json`, with the SHA-256 of every input
and the seed (20261019). It **reproduces the two registered O3 head-to-heads
from the run's own functions before it reports anything**, and stops if they
differ. The pure functions it uses are in `src/nlaast/analysis/posthoc.py`,
tested by `tests/test_posthoc_audit.py`.

---

## 1. What each document is

| Kind | Documents | May change? |
|---|---|---|
| Pre-registration | `PROJECT_PLAN.md` §5–§6 (2026-10-05) | No |
| Frozen report | `paper/0*.md`, `paper/REPORT.md` at `32b5539` (D66) | Only by a dated decision. This audit adds an addendum and edits no section |
| Registered results | `runs/main/analysis/results.json` and the stage outputs | No |
| Exploratory, partial, post hoc | N = 240 arm (D54), F8 in part (D55), A1 (D56–D57), F6 diagnostic (D58–D60), supervised analyses (D51), **this audit (D68)** | Each is labelled where it appears |
| Decision record | `docs/DECISIONS.md` | Append only |
| Living register | `docs/AUDIT_2026-10-17.md` | Rows added; existing rows kept as written |

## 2. Claim–evidence audit

| Claim as reported | Status after audit | Basis |
|---|---|---|
| Tails occur non-trivially on 0.948 of problems, covering 0.378 of the chunks (a floor) | **Established, as an operational quantity** | Reproduced. It depends on the forcing budget: the pilot moved from 0.356 to 0.486 when the budget changed, and the main run spans 0.378–0.484 between the registered and relaxed criteria (§3) |
| The tail "measures determinacy, not post-answer checking" | **Narrowed** | Where the tail starts before the answer is written, the forced completion does the remaining arithmetic itself on 0.595 of problems (§3). The tail start marks *recoverability within the budget*, not a state in which the model's answer is already fixed |
| Tail fraction is higher on correct answers (q = 0.036) | **Suggested, not established** | The association disappears once log trace length is a covariate: coefficient 0.86, p = 0.68 (§4). Under the 9-test family its significance also depends on the multiplicity convention (§6) |
| The readout "did not beat" the cheaper signals | **True, but uninformative** | The comparison was made at budgets where an information-free rule beats every compared rule, and where even an oracle must stop unsafely (§5). It could not have shown incremental value |
| The readout is "directionally worse than semantic entropy"; at N = 240 "the interval excludes zero" | **Withdrawn as an interpretation** | Semantic entropy's advantage is abstention: 38 of its 39 safe problems are problems it never stopped on, and it is safe on 1 of 82 problems where it stops (§5) |
| The comparison is at matched budget | **Narrowed to matched *trace-token* budget** | The plan's O3 says "matched compute". Sampling semantic entropy costs 8.2× the trace's tokens, and the verbaliser 9.6× (§5.5) |
| Safe stopping as registered | **Implemented differently, undisclosed** | The code scores the answer *parsed* from the prefix; §5 registers the answer *forced* from it (§5.1) |
| 0.749 of claims are reconstruction-dependent and not reproduced from noise | **Established, as reconstruction dependence** | Survives problem clustering (§7). It establishes no semantic accuracy or faithfulness (§7.2) |
| F10: tail reconstruction is significantly worse than both matched windows | **Narrowed** | The registered tests are sample-level. At the problem level the matched-length contrast is p = 0.053, and the mean difference against matched-position has a CI that includes 0 (§6) |
| "Correcting over 7 or 9 tests gives the same 5 significant results" | **Does not survive the unit correction** | With F10 and RQ2 at the problem level, the counts are 4 under the evaluable convention and 3 under the planned family; RQ1 is the test that differs (§6) |
| RQ3 is not tested | **Confirmed** | 2 of 274 traces (§8). The diagnosis is extended, and a follow-up is designed in `docs/RQ3_FOLLOWUP_DESIGN.md` |
| F8: bf16 target activations match NF4 at cosine 0.990 | **Established, partial** | Only the target's activations were bf16. Nothing about bf16 verbalisation or generation is shown (§9) |

## 3. The AST operationalisation

The registered definition (PROJECT_PLAN §5) is behavioural and implemented as
written (`ast_detect/detector.py`, a backward scan with persistence, capped at
12 boundaries by D4). Three points constrain what the measurement means.

1. **The tail start depends on a generation budget.** Criterion 1 forces an
   answer with "Therefore, the answer is" and up to 48 new tokens. On 0.475 of
   tail starts the forced completion contains arithmetic. On the 173 problems
   where the tail starts before the answer appears in the text (permissive
   extractor), it computes on 0.595 of them. For example: "Therefore, the
   answer is: 40 minutes + 20 minutes + 40 minutes + 60 minutes = 160
   minutes". The criterion therefore asks whether the model *can finish
   reliably from here within 48 tokens*. The budget sensitivity the project
   already measured follows from that: tail fraction 0.356 → 0.486 on the
   pilot when the budget was raised from 24 to 48 (§4.3), and 0.378 against
   0.484 on the main run when cut-off forced answers are relaxed.
2. **The genuinely post-answer region is small.** Only a mean 0.117 of a
   trace's tokens come after the chunk where the answer is first written
   (median 0.066). That text is mostly restatement and `\boxed{}`. Most of the
   tail (a mean 0.378 of the chunks) is the computation that produces the
   answer.
3. **Answer stability ≠ redundant computation ≠ removable region.** The
   report already keeps these apart (§3.3, §8.6). The audit adds that, on this
   model, the tail is mostly *answer-producing* computation, so a "redundant
   tokens" reading would be wrong twice over. The `truncate` arm is close to
   tautological with criterion 1: stopping at the tail start and forcing an
   answer is the criterion itself, so its result (2 of 60 answers changed,
   by a fresh forced generation under a different seed) is a reproducibility
   check of criterion 1, not causal evidence.

**What the measurement supports:** an operationally defined region over which
the final answer is recoverable from the prefix, at the stated budgets and
under five sampled continuations. Its location is a *lower bound*, sensitive to
the forcing budget. The F5 sweeps over temperature and chunker, which would
test the other sensitivities, were not run (D52).

## 4. RQ1: tail fraction and correctness

The registered test (Mann–Whitney over 240 problems with a tail, Cliff's δ
+0.295, q = 0.036) reproduces. Tail fraction and trace length are strongly
related (Spearman ρ = −0.569 against log tokens), and incorrect traces are
longer (median 299 tokens against 278). Logistic regression of correctness on
tail fraction (post hoc; 21 incorrect, so the estimates are imprecise):

| covariates | coefficient on tail fraction | p | 95% CI |
|---|---|---|---|
| none | 3.76 | 0.029 | [0.38, 7.15] |
| log chunks + dataset | 4.23 | 0.046 | [0.07, 8.38] |
| log tokens + dataset | 0.86 | 0.68 | [−3.17, 4.89] |

**Reading.** The association cannot be separated from trace length on this
corpus. Correct traces are shorter, and shorter traces have proportionally
longer tails. Whether tail fraction carries information about correctness
beyond length is **untested**. With 21 events, this check could not establish
the absence of such information either.

## 5. The primary comparison (O3)

### 5.1 The safety outcome deviates from the registration

PROJECT_PLAN §5 defines a stop at boundary *s* as safe "iff the answer
**forced** from `P_s` is equivalent to the full-trace answer and the full-trace
answer is correct, or both are incorrect". Report §3.5 restates it.
`score_baselines.build_rows` scores instead the answer **parsed from the
prefix text** (`extract_answer(prefix, PERMISSIVE)`). This was not recorded as
a deviation. D8 describes outcome-based safety but not the substitution. The
difference is large wherever the tail has started and the text has not yet
written the answer:

| stopping at the AST start, NLA problem set (120) | safe rate |
|---|---|
| implemented outcome (parsed answer) | **0.275** |
| registered outcome (forced answer) | **1.000** (by criterion 1) |

Under the implemented outcome, a rule that stopped exactly where the answer
became recoverable would be scored unsafe on most problems. The registered
outcome cannot be recomputed for every rule from the stored artifacts: forced
answers exist only at boundaries the AST scan evaluated, and 554 boundaries on
the NLA problem set have none. Doing it needs forced answers at the 554 boundaries that lack one
(up to 48 tokens each, at most about 27k tokens), well under an hour on this
GPU. It was not run.

### 5.2 The matched-budget window is beyond what any rule can save safely

A rule that stopped only where stopping is safe, and abstained otherwise, can
save at most:

| outcome | mean over problems of the largest safe saving |
|---|---|
| implemented (parsed answer) | **0.174** (median 0.067) |
| registered (forced answer), lower bound | **≥ 0.314** (median 0.276) |

The head-to-heads were made at mean savings of **0.613** (against semantic
entropy) and **0.722** (against the probe). The window is fixed by the
intersection of the rules' reachable ranges (`budget_grid`), not by where the
question is meaningful. Every compared operating point therefore *requires*
unsafe stops on most problems, whatever the signal. The literature the project
cites works in a different regime: Mo et al. at about a third of tokens saved
[22], and Liu and Wang reporting savings "with limited accuracy loss" [5]
(`output/.../review1/04_literature_comparison.md`).

### 5.3 The safe rates are mostly abstention

A rule that does not fire on a problem is safe there by definition
(`evaluate_stop`). At the compared points:

| rule | budget | safe | from abstention | safe when it stopped | stops before the tail |
|---|---|---|---|---|---|
| semantic entropy | 0.613 | 39/120 = 0.325 | **38** | **1 of 82 (0.012)** | 0.988 |
| NLA readout | 0.614 | 28/120 = 0.233 | 10 | 18 of 110 (0.164) | 0.918 |
| probe | 0.722 | 16/120 = 0.133 | 0 | 16 of 120 (0.133) | 0.908 |
| NLA readout | 0.722 | 15/120 = 0.125 | 0 | 15 of 120 (0.125) | 0.983 |

Semantic entropy is constant within a problem, so it always stops at the first
boundary. That gives the largest saving per stop, so it reaches a given budget
with the fewest stops and the most abstentions. **Its higher safe rate is a
property of where it stops, not of what it knows.** When it does stop it is
almost never safe. The exploratory N = 240 extension (D54) shows the same
structure: semantic entropy's 0.325 there is 74 abstentions and 4 safe stops
out of 166.

### 5.4 An information-free rule beats every compared rule

A rule that stops at the first boundary on a uniformly random subset of
problems, sized to reach the same mean saving, uses no signal at all
(`random_first_boundary_null`, 20,000 draws):

| budget | information-free rule | semantic entropy | probe | NLA readout |
|---|---|---|---|---|
| 0.613 | **0.373** [0.350, 0.392] | 0.325 | — | 0.233 |
| 0.722 | **0.261** [0.242, 0.275] | — | 0.133 | 0.125 |

*The interval is the central 95% of the null distribution.*

Every compared rule sits **below** the rule that knows nothing. In this regime
the safe rate is not a measure of a signal's information about when an answer
is settled. A difference between two rules there reflects their firing
geometry. So:

- the registered O3 result, "no significant difference", stands as reported;
  it is **not evidence about incremental value in either direction**;
- the direction against semantic entropy, at N = 120 and at N = 240, should
  not be read as semantic entropy carrying more information than the readout.

**One exploratory hint, and its limit.** Holding the readout's possible firing
positions fixed and shuffling its scores (1,000 draws) gives 0.155 [0.125,
0.192] at 0.613. The readout's actual 0.233 lies above that range. So the
reconstruction score carries *some* information beyond where the verbaliser
was run. At 0.722 the two do not separate: 0.125 against [0.108, 0.125].
This is one post-hoc check, made in a regime the previous paragraph calls
uninformative, and it supports no claim.

### 5.5 Budget versus compute

PROJECT_PLAN's O3 row says "at matched compute". The implementation matches
the share of the trace's own tokens that are not generated. The cost of
computing the signal is not counted:

| signal | extra tokens generated per problem | relative to the trace (mean 325 tokens) |
|---|---|---|
| semantic entropy (8 samples) | 2,577 | **8.2×** |
| verbaliser (7B, all samples incl. the Gaussian control) | 2,577 | **9.6×** |
| probe | 0 (one linear map on activations already computed) | ~0 |
| answer convergence | 0 | 0 |

Under any accounting that includes what a signal costs, semantic entropy and
the verbalised readout both cost far more than they could save. Neither is a
practical stopping rule here. The comparison is a test of information content
at a matched *trace* budget, which is the most it can be.

### 5.6 What O3 can and cannot say

- **Can say:** at the budgets reachable by every rule on this corpus, no
  significant difference in the registered safe-stopping rate (as
  implemented) between the readout and either comparator.
- **Cannot say:** whether verbalising activations adds predictive value
  beyond cheaper signals. The design reached no regime where any signal beats
  chance. Equivalence is not established either.
- **What a valid test needs:** the outcome as registered (forced answers at
  every boundary, about an hour on this GPU); comparison at budgets *below*
  the oracle bound, e.g. over a range of savings in [0, 0.3] rather than at a
  single matched point; the information-free rule reported beside every
  comparison; and, for the readout, a scoring rule fixed in advance. A
  threshold-free alternative estimand is per-boundary discrimination of
  "safe to stop here" (AUC, grouped by problem). The readout can be scored
  only at its windows, so that comparison would be restricted to window
  boundaries.

## 6. F10, RQ2 and the multiplicity family: units of analysis

The project corrected pseudo-replication in O3 (D29) and RQ3 (D41). Two
registered tests still pool non-independent units: F10 compares about 4
samples per problem with Mann–Whitney, and RQ2 compares 6,007 claims from
120 problems with Wilcoxon. Re-expressed at the problem level, on the same
rows:

| test | registered p (unit) | problem-level p | problem-level effect |
|---|---|---|---|
| F10 tail − matched-position | 6.4e-8 (sample) | **0.012** | −0.0052 [−0.0119, 0.0016]; tail lower on 0.62 of problems |
| F10 tail − matched-length | 2.7e-4 (sample) | **0.053** | −0.0031 [−0.0089, 0.0028] |
| F10 tail − Gaussian | 1e-187 (sample) | 2e-21 | +0.479 [0.472, 0.486] |
| RQ2 deletion − paraphrase | 0 (claim) | 2e-21 | all 120 problems positive; mean 0.0324 [0.0301, 0.0347], cluster bootstrap |

The BH family recomputed with these substituted:

| convention | significant (registered) | significant (problem-level) |
|---|---|---|
| evaluable tests (m = 7), primary | 5 | **4**: RQ1, RQ2, F10 vs matched-position, F10 vs Gaussian |
| planned family (m = 9) | 5 | **3**: RQ1 is no longer significant (q > 0.05) |

So report §5.5/§6.7's "the convention decides nothing on this run" holds only
under the registered, pseudo-replicated units. RQ1's significance depends on
the convention once the units are corrected. F10's substantive reading,
*fidelity is not higher in the tail*, is unaffected. "Significantly worse
than both matched windows" is not supported at the problem level for the
matched-length window.

## 7. RQ2: what the claim-level result establishes

### 7.1 It survives clustering

The dependent rate is 0.750, problem-cluster 95% CI [0.734, 0.766]. The paired
deletion − paraphrase difference is 0.0324 [0.0301, 0.0347] against the
registered claim-level [0.0306, 0.0329]. Every one of the 120 problems has a
positive mean. The registered conclusion holds.

### 7.2 What it does and does not show

| property | status | evidence |
|---|---|---|
| Activation reconstruction succeeds | **Established** | cosine 0.875 against 0.393 for the Gaussian control; FVE 0.695 against this run's own baseline |
| Generated claims carry reconstruction weight | **Established**, against a weak null | 0.750 dependent. The null is surface edits pooled per explanation, and 29.2% of the paraphrase draws are *identical* to the original (effect exactly 0, e.g. no synonym matched). Deleting a claim removes 23% of the text on average. The length-adjusted rate (0.772) addresses size, not content |
| Claims are not verbaliser priors | **Supported, for this control** | 0.002 reproduced from matched-norm noise. This is lexical overlap at a Jaccard threshold, with a reported sensitivity |
| Claims are semantically accurate (true of the activation's content) | **Untested** | Amendment A1, the only content test, did not corroborate (D56–D57) and used unvalidated rule-derived labels |
| Claims are faithful to the model's computation | **Untested** | Nothing in the design measures what the computation uses |
| Claims identify a mechanism | **Untested** | Needs intervention on the claimed content |

**Exploratory observation.** The dependent rate is *lower* in tail windows
(0.659) than in matched-position (0.820) or matched-length (0.776) windows.
This is not tested and not interpreted.

**The strongest additional control that existing infrastructure allows** is a
*content-swap null*: replace a claim with a claim of similar length from a
verbalisation of a *different problem's* activation, and score
reconstruction. A claim is content-dependent if its deletion, and its swap,
degrade reconstruction more than swapping in a same-problem resampled claim
does (the resample arm already exists: mean effect 0.0040). This separates
"any text in this slot" from "this content". It needs the reconstructor on GPU
(one forward pass per claim, about 6,000 passes) and was not run. It would
still establish reconstruction dependence on content. Faithfulness would
remain untested.

## 8. RQ3

Verified: 2 of 274 traces carry a registered marker (pooled 0.0073, 95% CI
[0.0009, 0.0261]); the readout is zero in every arm at the compared dose. RQ3
stays **not tested** (D62). The audit adds three causes to the diagnosis:

- **The construct is near-absent, not just mismeasured.** A broader lexicon
  finds 12 of 250 traces, and 3 tails.
- **The random-direction control is about 19× smaller in edit size** than the
  candidate (1.1% against 21.8% of activation norm per position).
- **Every effect is confined to doses where output length roughly triples
  without controls** (64.6 → 183.6 tokens at dose 2.0).

It also corrects a sentence in D62 and the RQ3 brief that reads a
no-contrast observation as "controls change nothing". The full diagnosis, the
design for a follow-up (a new question, not RQ3), the power analysis and the
two-stage plan are in [`RQ3_FOLLOWUP_DESIGN.md`](RQ3_FOLLOWUP_DESIGN.md).

## 9. F8: what the partial replication establishes

On 24 problems: bf16 and NF4 *target* activations agree at cosine 0.990, and
the readout's per-window score shifts by −0.0014 [−0.0033, 0.0005] when bf16
activations are fed to the **NF4** verbaliser and reconstructor (D55).

- **Establishes:** quantising the target perturbs layer-20 activations little,
  and the NF4 autoencoder's score is insensitive to that perturbation.
- **Does not establish:** that the NF4 verbaliser and reconstructor behave
  like the released bf16 checkpoints; that bf16 *generation* gives the same
  traces, tails or baselines; or anything about O3 at bf16. 24 problems give
  no power for O3. This is a partial replication of F8's activation half
  only.

## 10. Literature positioning, using the existing review

No new literature search was made. The repository's review
(`output/nla_answer_stable_tail/`, `review1/05_research_gap.md`) rates the
project's novelty as "combination + evaluation", with medium confidence. The
frozen report is more conservative ("none of these is asserted to be novel").
Against the evidence:

- **The evaluation contribution needs a qualifier.** The review's strongest
  claim is a new composite evaluation, a matched-budget, safe-stopping
  head-to-head. §5 shows that, as executed, that evaluation cannot
  discriminate signals at the budgets it reaches. The defensible contribution
  is the protocol's controls and reporting discipline, *plus* a documented
  failure mode of the matched-mean-budget safe-rate design (abstention
  dominance beyond the oracle bound). That failure mode is a methodological
  lesson in its own right.
- **[5] and [22] already locate the useful regime.** Liu and Wang report
  savings at limited accuracy loss, and Mo et al. study rules saving about a
  third of tokens with a residual premature-stop rate. Both are below this
  study's oracle bounds (0.17–0.31) and far below its comparison window
  (0.61–0.74). The report cites both but does not use them to place its own
  window.
- **The causal gap was framed for reasoning-trained models.** The review's
  "causal gap" ("whether the putative verification tail is causally
  redundant") and RQ3's markers presuppose reasoning-model behaviour: the
  markers are the idiom of reasoning-trained traces such as those of [2] and
  the end-of-thinking models of [24]. The study's target is an
  instruction-tuned model, which the review does not flag as a mismatch.
- **[12] and [13] already bound the RQ2 reading.** Li et al.'s verbaliser-only
  concern is addressed for lexical reproduction. Lek et al.'s point that
  reconstruction can reward unsupported detail is the gap §7.2 leaves open.
  The report states both (§7.4).
- **One narrow gap.** Whether a safe-rate comparison at matched *mean* budget
  is the standard way to compare early-exit signals cannot be settled from the
  references in the review. The related literature on selective prediction
  and risk–coverage curves is not in it. Judged only by its internal logic,
  §5's critique does not depend on that literature.

## 11. Robustness checks, ranked by value

| rank | check | could change | data and infrastructure | cost | run here? |
|---|---|---|---|---|---|
| 1 | O3 rescored with **forced answers at every boundary**, compared over budgets below the oracle bound, with the information-free rule beside it | Whether *any* signal, the readout included, beats chance at a meaningful budget: the primary question | the existing pipeline's forcing code and NF4 target | under 1 GPU-hour (554 boundaries × ≤ 48 tokens) | **No.** Needs the GPU and a dated decision; specified in §5.6 |
| 2 | RQ3 Stage 1: **size-matched controls** at doses 1.5 and 2.0 | Whether the answer-change dose–response is direction-specific | the causal stage plus two new arms | 1–2.5 GPU-hours | **No.** `RQ3_FOLLOWUP_DESIGN.md` |
| 3 | AST **forcing-budget sweep** (24 / 48 / 96 tokens) on stored traces | How much of the tail location is the budget | the AST stage on existing traces | a few GPU-hours | No |
| 4 | RQ2 **content-swap null** | Whether dependence is content-specific | reconstructor | about 6,000 forward passes | No |
| 5 | F5 temperature and chunker sweeps | tail-definition sensitivity | implemented (`run_f5_sweep.py`) | 60–150 GPU-hours (D52) | No. Ranks 1 and 3 answer more of the same question for a fraction of the cost |
| 6 | Full bf16 F8 | quantisation threat to the autoencoder | needs about 41 GB bf16 and bf16 generation | not feasible on 6.4 GB VRAM | No |

**Run here, post hoc and CPU-only:** everything in §3–§9 marked with a number
above, through `audit_claims_posthoc.py`, plus the RQ3 power simulation in
the design document.

## 12. Validation

**Run on 2026-10-19:**

- `pytest -m "not slow"`: exit 0 before the change. After it: 975 passed, 3
  skipped (they need the autoencoder sidecars in the local cache), 0 failed.
  The 5 GPU tests were not run.
- `tests/test_posthoc_audit.py`: 9 new tests, all passing.
- `audit_claims_posthoc.py` re-invoked on an existing output: refuses, exit 1.
- `check_run_consistency.py --run runs/main`: 179 checks, 0 mismatched, 0
  warnings, before and after.
- `measure_rq3_readout_floor.py`: 2 of 274, unchanged.
- `assemble_paper.py --check`: exit 0.

**Software validation is not scientific validation.** The checks show that the
code reproduces the stored numbers. They do not show that the registered
outcome measures what the research question asks, and §5 argues that for O3 it
does not.

**Not run:** every GPU item in §11.

## 13. Remaining weaknesses

1. **The primary question is unanswered.** Not because the readout failed: the
   comparison could not discriminate any signal from chance.
2. **The implemented safety outcome is not the registered one.** No result
   has been recomputed under the registered outcome.
3. **One model, one seed, one layer, an NF4 autoencoder, 120 problems in the
   NLA arm.** No counting-and-probability problems in the NLA and causal arms
   (§8.11).
4. **The RQ2 null is weak** (29% identity paraphrases), and no test of claim
   content has been validated.
5. **RQ3 is unanswered.** Its registered construct barely occurs in this
   model's output.
6. **Two registered tests use non-independent units** (F10, RQ2), and the
   stored results are kept as registered for the audit trail.
