# RQ3 follow-up: a design, not a result (2026-10-19)

**Status: proposed. Nothing in this document has been run.** RQ3 stays
classified as *not tested* on Qwen2.5-7B-Instruct (D62). This document gives a
design that a separately preregistered study could run, and states what the
existing evidence does and does not tell it. It does not replace RQ3's
registered construct or change any stored verdict. The numbers it cites come
from `scripts/audit_claims_posthoc.py` (D68) and
`scripts/measure_rq3_readout_floor.py`. Both are read-only.

> **Revised 2026-10-19 (D74), after the design was checked against the stored
> rows and a CPU dry run.** Three changes:
> - The 13/60 answer-change rate this design was powered on is a truncation
>   artefact: all 13 continuations hit the 200-token cap.
> - Coefficient-rescaled "size matching" is a different edit; the controls
>   are now matched on what they remove at the same coefficient.
> - The candidate is largely a position direction (ρ = 0.81 with relative
>   position).
>
> §3's arms, §4's power and §5 are revised below. The pilot is implemented
> (`scripts/run_rq3_pilot.py`) and **not run**.

## 1. Why the registered test produced no outcome

RQ3 asks whether a tail direction's ablation "selectively changes
verification-like behaviour without changing final answers"
(`Project_Review_II.md` l.122). The question failed on this model for four
reasons, each measured.

| # | Cause | Evidence | What it implies |
|---|---|---|---|
| 1 | **The construct barely occurs in this model's output.** | The 20 registered markers occur in 2 of 274 traces (0.0073, 95% CI [0.0009, 0.0261]). A broader screen (verify/check/confirm/substitute back/plug in) finds 12 of 250 traces, some of them false positives such as "checks Buford writes", and 3 of 240 tails. On average only 0.117 of a trace's tokens come after the chunk where the answer is first stated (median 0.066), and that text is mostly restatement and `\boxed{}`. | The proxy did not fail on its own; the behaviour it was built to detect is almost absent. The markers ("wait", "hmm", "let me check") are how *reasoning-trained* models write self-verification. Qwen2.5-7B-Instruct is instruction-tuned and writes linear derivations. |
| 2 | **The tail is not verification on this model.** | The tail starts before the answer is stated on 0.988 of problems (strict extractor; registered summary). When the tail starts before even a permissive mention of the answer, the 48-token forced completion at the tail start does arithmetic on 0.595 of problems. | The premise of RQ3, that the tail is a verification phase, does not hold here. The tail is mostly the last steps of the computation plus a restatement. |
| 3 | **The random-direction control does not match the edit's size.** | Projecting out the candidate direction removes 21.8% of a layer-20 activation's norm per position. A random unit direction removes 1.1% (mean over 5 random directions), so the candidate's edit is about **19×** larger (19.3; eval-split boundary activations, used as a proxy for the edited positions). | Even with a readout that varied, "beats `random_dir`" would be confounded with edit size. The docstring's "matched norm" refers to the direction vector, which is unit length in both arms, not to the edit. |
| 4 | **The doses where something happened have no controls.** | Answer change in the candidate arm is 0, 0, 1, 4 and 13 of 60 problems at doses 0, 0.5, 1.0, 1.5 and 2.0. Controls were run only at dose 1.0, where the candidate changed 1 answer and every control changed 0. Mean generated tokens rise from 64.6 at dose 0 to 140.2 at 1.5 and 183.6 at 2.0. | Any effect is at doses where the edit also roughly triples output length, which is a sign of generic disruption, and nothing was measured to separate that from a direction-specific effect. |

**Correction to the record.** D62 and `docs/RQ3_DECISION_BRIEF.md` describe
the answer-change dose–response as rising "while random-direction and
position-matched controls change nothing". The controls ran only at dose 1.0,
where the candidate also changed 1 of 60 answers. That sentence states no
contrast, and it is not evidence that the effect is specific to the
direction. The report's §6.5 states this correctly ("No control was run at
those doses"). Both historical documents are kept as written; D68 records the
correction.

**What this does and does not show about RQ3.** It does not show that no tail
direction exists. It shows that, on this model, the registered question was
measured with an instrument that has no range, against a control that is not
matched in edit size. The question stays open.

## 2. Two routes, and which one this design takes

- **Route V (the registered construct).** Test verification-like behaviour
  where it occurs, in a reasoning-trained model whose traces contain it. The
  candidate direction is a difference in means of the target model's own
  activations; it does not need the autoencoder, so RQ3 can move to a model
  without a released NLA. This answers RQ3 as posed, but in a different
  population. It needs a new model (about 15 GB in bf16 before conversion to
  NF4) and a validated measure of verification behaviour. Without a second
  annotator the project cannot provide the validation (PROJECT_PLAN §8, D62).
- **Route R (a new, related question) — specified below.** Ask whether the
  computation in the Answer-Stable Tail is **causally redundant along the
  candidate direction**. This is distinction 1 of the project ("answer
  stability ≠ causal redundancy"), which the study enforces in its design but
  never tests. Its outcome, answer change, has range on this model. **It is a
  new question, not RQ3.** D62 records that answer change was screened on the
  60 causal-arm problems, so a test there is exploratory. A confirmatory test
  must use problems the causal arm never touched.

## 3. Route R, specified

**Causal hypothesis (H-R).** In Qwen2.5-7B-Instruct at layer 20, removing the
tail direction over the regenerated tail span changes the final answer more
often than removing control directions of the **same edit size**, at the same
dose and over the same span.

**Unit of analysis.** The problem. Every arm runs on every problem. Replicate
control directions are averaged within a problem first, as D41 does.

**Arms**, at doses {0, 1.0, 1.5, 2.0}; dose 0 is the identity check:

| arm | edit | controls for |
|---|---|---|
| `ablate_dir` | project out the training-split difference in means, tail − matched-position (as registered) | — |
| `variance_matched` *(revised, D74)* | 5 random directions drawn from the activations' second-moment geometry, orthogonal to the candidate, accepted only if their mean squared projection on train-split activations is within ±10% of the candidate's. Projected out at the **same coefficient**. *Replaces* coefficient rescaling, which at about 19× reflects a component instead of removing it | "any edit this large changes answers" |
| `permuted_label` *(revised, D74)* | difference in means after permuting tail/pre labels within each problem, orthogonalised to the candidate and size-filtered to ±10%; 5 directions. Raw permuted directions ranged from 0.33× to 1.20× in size, with cosines up to ±0.77 with the candidate | "any direction this estimator returns from these windows" |
| `position_direction` *(added, D74)* | the same estimator with late-versus-early labels on pre-tail boundaries; cosine 0.357 with the candidate. Secondary, not in the decision rule | "a direction encoding progress through the solution" |
| `matched_position` | `ablate_dir` at the pre-stabilisation window (as registered). *Dropped from the pilot*: it differs in kind (§8.3) and is not needed to measure the control rate | location |

**Primary outcome.** `answer_changed`: the final answer after the intervened
continuation is not equivalent, by `math_verify`, to the unintervened answer
from the same prefix under the same seed. Binary, per problem and arm.
*Revised (D74):* the continuation may run to 768 tokens. One that still hits
the cap is made to commit by the registered forcing call, and its answer is
labelled `forced_after_cap`. The answer is never parsed from a truncated
text, which is what produced every registered dose-2.0 change.

**Estimand.** At dose 2.0, the paired difference in answer-change rate, Δ =
mean over problems of [change(candidate) − mean change(size-matched
random)]. The same is computed against `permuted_label_dir`. The primary
claim needs both.

**Test.** For each control, a one-sided exact sign-flip permutation test on
the per-problem differences, α = 0.05. The claim needs **both** controls to be
beaten. This is an intersection–union test, so no multiplicity correction is
needed for that conjunction. A 95% interval for each Δ comes from a problem
bootstrap. A dose–response over {1.0, 1.5, 2.0} is reported as secondary and
is not used to rescue a failed primary test.

**Manipulation checks.** Each one must pass, or the run is reported as a
failed manipulation rather than a result:

1. Dose 0 reproduces the unintervened answer and token count on every problem.
2. `n_positions_edited > 0` for every problem with a continuation. Problems
   without one are counted and excluded, as in D47.
3. The **mean removed norm per position** is logged for every arm. The
   size-matched arm's must lie within ±10% of the candidate's on each problem.
   This check is new; the registered stage did not record edit size.
4. The two arms' edit spans are disjoint where the design requires it, as
   asserted at run time now.

**Secondary outcomes**, never substituted for the primary:

- generated-token ratio to dose 0;
- a degeneration flag: a repeated n-gram run, or a non-ASCII share above the
  integrity gate's threshold;
- correctness change.

These separate a direction that changes what the model *computes* from one
that only breaks generation.

**Principal confounds, and what addresses them:**

| confound | addressed by |
|---|---|
| edit size (the registered control's flaw) | `size_matched_random` and check 3 |
| anisotropy: the estimator returns high-variance directions whatever the labels | `permuted_label_dir` |
| generic disruption | the degeneration and length secondaries. A candidate effect that comes only with degeneration is reported as disruption |
| problem difficulty | pairing within problem; dataset stratified in the holdout |
| the direction fitted on the evaluation problems | the training-split fit, as registered; the holdout is eval-split |
| forking paths in the outcome | the outcome, dose and test fixed here, before any run. `answer_changed` was seen on the 60 used problems, which is why they are confined to the pilot stage below |

**Interpretation, fixed in advance:**

- **Positive:** both controls beaten at dose 2.0, all manipulation checks
  pass, and the effect is not confined to degenerate continuations. Read it as
  *"the tail computation is not causally redundant along this direction on this
  model"*. It is **not** evidence about verification, or about RQ3.
- **Negative:** Δ against the size-matched control has a 95% upper bound
  below 0.05. Read it as *"no direction-specific effect larger than 0.05
  beyond what an edit of the same size does"*. The margin of 0.05 is fixed
  here. A nonsignificant Δ whose interval includes 0.05 is **not** negative.
- **Inconclusive:** anything else, reported as such, including a test that
  is significant against one control and not the other.

## 4. Power, and why a cheap first stage is needed

> **Superseded basis (D74).** The candidate's 13/60 below was measured with a
> 200-token cap, and every one of the 13 was a truncated continuation. The
> table is kept as power *as a function of* the two rates. The candidate's
> real rate is unknown until Stage 0 measures it with completion allowed.

From the main run: the candidate changes 13 of 60 answers at dose 2.0 (0.217,
exact 95% CI [0.121, 0.342]). The answer-change rate of an edit-size-matched
control has never been measured, and power depends almost entirely on it.
Simulated power is below. It assumes one-sided α = 0.05, 5 control replicates
averaged within problem, independent problems, and 4,000 simulations per cell.

| N problems | control rate 0.00 | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|
| 60 | 1.00 | 0.82 | 0.41 | 0.13 |
| 83 (the D62 holdout) | 1.00 | 0.92 | 0.49 | 0.15 |
| 120 | 1.00 | 0.97 | 0.63 | 0.19 |
| 160 | 1.00 | 0.99 | 0.73 | 0.22 |

If the candidate's true rate is at the lower end of its interval (0.12),
power at N = 83 falls to 0.57 against a control rate of 0.03, and to 0.25
against 0.05.

**Hence two stages:**

1. **Stage 1, diagnostic, exploratory: size the control.** Run
   `size_matched_random` and `permuted_label_dir` at doses 1.5 and 2.0 on the
   **60 problems already used** by the causal arm. Rough cost: 60 problems × 2
   doses × 10 control directions × about 150 generated tokens is about 180k
   tokens. At the 20–45 tok/s measured on this GPU, that is about 1–2.5
   GPU-hours. It measures the control rate, which is the input to N. It
   cannot confirm anything: these problems' answer change has been seen.
2. **Stage 2, confirmatory.** Register the Stage-1 control rate, the N it
   implies for power ≥ 0.8, the outcome, the estimand and the decision rules
   above. Then run on the 83 held-out eval problems, plus fresh traces if N
   exceeds 83. If the Stage-1 control rate is ≥ 0.10, the design's power at
   any N this hardware can produce is low. Report that before running, rather
   than running an underpowered confirmatory test.

## 5. Status (revised, D74)

- **Code: implemented and tested on CPU.** `causal/controls.py`;
  removed-norm accounting in `projection_editor`; `scripts/run_rq3_pilot.py`
  with `--dry-run`, `--stage0` and `--summarise`; tests in
  `tests/test_rq3_pilot_controls.py`. The registered `random_dir` arm is
  unchanged, so the stored run still reproduces.
- **Dry run, done on CPU.**
  - The candidate reproduces (387/354 vectors, 97 problems).
  - Both control families remove 20–25% of the norm on held-out eval
    activations, against the candidate's 21.8%.
  - The candidate differs from its permutation null (p = 0.0005; held-out
    tail-versus-pre AUC 0.918), but tracks relative position (ρ = 0.81).
- **Hardware:** the NF4 target is on disk (`D:\nla_models\nf4`, 5.2 GB).
- **Cost, revised from "1–2.5 h":**
  - Stage 0: 120 generations, 0.2–1.3 GPU-hours.
  - Stage 1: 1,500 generations, 1.5–16 GPU-hours, depending on whether the
    controls also lengthen the output.
- **Go/no-go rules:** D74.
- **Not run.** Stage 0 needs a dated decision. Stage 2 needs a
  preregistration frozen by hash, as D58 did.
