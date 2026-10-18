# Implementation decisions and deviations

Every point where the implementation departs from the obvious reading of the
specification, or from an upstream reference, with the reason. Recorded here so
the execution report can point at it rather than re-argue it, and so a reviewer
can disagree with a specific choice rather than with the whole system.

---

## D1. The verbaliser runs through `transformers`, not SGLang

**Upstream** drives the activation verbaliser through an SGLang server using
its `input_embeds` API.

**Here** it runs locally through `transformers.generate(inputs_embeds=...)`.

**Why.** SGLang has no Windows support, and would not fit beside the model in
6.44 GB of VRAM.

**What is preserved.** The arithmetic, step for step. The chat template,
injection character and token ids, left/right neighbour validation, and the
injection scale all come from the checkpoint's own `nla_meta.yaml`. The two
correctness-critical functions — `normalize_activation` and
`inject_at_marked_positions` — are called from the *vendored upstream file*,
not reimplemented.

**What was verified.** `tests/test_inputs_embeds_contract.py` confirms against
a tiny Qwen2 that generating from `inputs_embeds` returns only the new tokens
(no prompt echo), that injection writes exactly the marked row and leaves every
other position untouched, and that two different activations produce different
output logits — i.e. that injection is not silently a no-op.

**Residual risk.** Sampling differences between the two backends are possible
and unmeasured. This affects the particular text sampled, not the mechanism.

---

## D2. All three checkpoints run in 4-bit NF4

**Released as** bf16. 41 GB across the three models, against ~22 GB of staging
disk and 6.44 GB of VRAM.

**Here** each is downloaded, converted to NF4 (double quantisation, bf16
compute, `lm_head` left in bf16), saved, and its bf16 source evicted before the
next one.

**Why.** There is no configuration in which the released precision fits.

**Why it might not matter.** The verbaliser normalises its input to a fixed L2
norm, so only the *direction* of an activation reaches it. And `embed_tokens`
is an `nn.Embedding`; bitsandbytes quantises only `nn.Linear`, so the injected
vector itself enters the network exactly — only the transformer weights are
4-bit.

**Why it is nonetheless measured.** Falsification test **F8** computes
layer-20 activations from both the 4-bit and the bf16 target on a subsample and
reports the cosine between them. The NLA stage additionally gates every
verbaliser sample on the off-distribution failure signature upstream documents
(CJK output instead of an English `<explanation>`) and compares round-trip
fidelity against the 0.752 in-distribution value on the checkpoint card.

**Ordering consequence.** F8 needs the bf16 target, which must be evicted to
make room for the verbaliser. The orchestrator therefore runs F8 between
`acts` and `nla`. This is enforced in `run_pipeline.py`, not left to memory.

---

## D3. Semantic entropy clusters by symbolic equivalence, not NLI

**Farquhar et al.** cluster sampled generations by bidirectional NLI
entailment.

**Here** clustering uses the same external verifier (`math_verify`) the rest of
the study uses, and the method is labelled `semantic_entropy_symbolic` in every
output.

**Why.** For mathematical answers the correct equivalence relation *is*
symbolic equality. `\frac{1}{2}`, `0.5` and `1/2` are one meaning and no
off-the-shelf NLI model reliably says so. Using NLI here would also mean the
entropy baseline and the tail detector disagreed about what "the same answer"
means, which would make any comparison between them uninterpretable.

**Secondary reason.** A DeBERTa-MNLI checkpoint is another ~1.6 GB of disk that
this machine does not have spare.

**Consequence.** This is not semantic entropy as published, and must not be
reported as though it were.

---

## D4. The backwards boundary scan is capped

Criterion 3 of the tail definition (persistence) means only a *suffix* of
boundaries can qualify, so the scan runs backwards from the end of the trace
and stops at the first boundary that fails. That part is an exact optimisation
of the definition.

The **cap** (`ast.max_boundaries_evaluated`, default 12) is not: a tail longer
than the cap is reported as exactly the cap length.

**Why.** Each boundary costs one forced-answer generation plus K resampled
continuations. An uncapped scan on a 40-chunk trace is 40 × (K+1) generations
for one problem.

**Direction of the bias.** It only ever *understates* tail length. Any claim of
the form "this much of the trace is redundant" is therefore conservative. Every
affected trace carries `boundary_scan_capped: true`.

---

## D5. Boundaries are evaluated in blocks

Within the backwards scan, boundaries are taken in blocks so their generations
can be batched — single-stream decoding on this GPU runs at ~20 tok/s against
~45 tok/s batched.

A block may evaluate a few boundaries past the first failure. This wastes a
little generation and **changes no result**: the `ast` stage decides from the
evidence, not from the order in which it was gathered.

---

## D6. Activations are read at the sampled token ids, not re-tokenised text

Chunk boundaries are character positions; activations are read at token
positions. The obvious implementation re-tokenises the decoded trace for the
forward pass.

**That is unsafe.** Byte-level BPE does not guarantee that re-tokenising
decoded text reproduces the sequence that produced it — merges can cross what
used to be a token boundary. A one-token shift would read every activation at
the wrong position, and nothing downstream would notice: the verbaliser would
simply describe the wrong state, confidently.

**Here** the token ids the model actually sampled are stored with the trace and
replayed verbatim, making the alignment exact by construction. The drift that
re-tokenisation *would* have caused is measured and recorded in the activation
store's alignment audit, so the size of the hazard is on record rather than
assumed to be zero.

---

## D7. The answer extractor comes in two strengths

`PERMISSIVE` falls back to the last number on the last line; `STRICT` refuses
to, requiring `\boxed{}` or an explicit "the answer is".

**Why both.** Intermediate answers at chunk boundaries are usually not phrased
as conclusions, so a strict parser would return `None` for most of a trace and
the convergence baseline would never fire. But a permissive parser can mark a
coincidental trailing number as an answer — exactly the weakness Caldarella et
al. identify in the first-correct-prefix construction. Falsification test
**F4** re-runs detection under the strict parser and reports the disagreement
rate, so the sensitivity is a measured quantity rather than a worry.

**Unit suffixes.** `"The answer is 180 minutes."` is extracted as `180`. This
was found in live pilot output: without it, a correct answer scores as
incorrect and the same value looks like two different answers across
boundaries, manufacturing spurious instability. The stripping is conservative —
it fires only when the whole candidate is one number plus alphabetic words, and
never when the trailing token is mathematical (`3 pi`, `2 squared`, `5 x`).

---

## D8. Safety is defined on verified outcome, not on answer identity

A stopping rule is *safe* when the correctness of the answer you walk away with
equals the correctness of the answer the full trace would have produced.

**Not** "the same string". That definition would count a rule that stops early
on a wrong answer the model later fixes as safe, and would count stopping on a
correct answer that the model later ruins as unsafe.

---

## D9. Rules are compared at matched token budget

Each signal is exposed as a per-boundary score, the threshold is swept, and
rules are read off their own safety/saving curves at the *same* mean token
saving.

**Why.** Comparing rules at their own natural operating points compares
different budgets, and a rule that simply stops later is trivially safer. The
budget grid is the intersection of the ranges every rule can actually reach, so
no rule is read at an operating point another cannot attain.

---

## D10. The NLA stopping score is reconstruction cosine

The autoencoder produces text, not a number, so a scalar has to be derived from
it to place it on the same curve as the other rules. The scalar used is mean
reconstruction cosine at the boundary, carried forward between scored windows.

**This is a choice and a limitation.** A different reduction of free text to a
scalar could give a different curve, and this study does not claim to have
found the best one. It is stated in the results rather than buried.

---

## D11. The NLA arm may run on fewer problems than the behavioural arms

The verbaliser emits hundreds of tokens per activation, making this arm an
order of magnitude more expensive than the others. `nla.max_problems` caps it;
the subset is a deterministic prefix by problem id, so a resumed or extended
run keeps the same subset rather than drifting.

The achieved N for the NLA arm is reported separately from the behavioural
arms', never merged into a single headline count.

---

## D12. Paraphrases are rule-based, not model-generated

The paraphrase arm of the faithfulness audit supplies the **null distribution**
against which the deletion effect is judged. It therefore has to be a *surface*
edit that preserves meaning.

Using a language model to paraphrase would inject that model's semantics into
the null — and if the paraphraser subtly changed meaning, the null would widen
and real deletion effects would vanish into it. The transformations here are
lexical substitution and clause reordering only.

---

## D13. The verbaliser-only control is run on everything

Li et al. showed that activation-verbalisation methods can score well on
standard benchmarks with no access to target-model internals at all. The
corresponding control here runs the verbaliser on a Gaussian vector at matched
L2 norm.

It is run on **every** sampled verbalisation, not a subset, and
`dependent_and_not_noise_fraction` is reported as a headline rather than a
footnote. Matching the norm matters: an off-norm vector would fail injection
for a trivial reason and the control would be vacuous.

---

## D14. A direction claim requires both controls and a dose–response

`causal.direction_claim_supported` returns true only if the candidate direction
beats a matched-random direction **and** the same direction applied at a
matched pre-stabilisation position, with a monotone dose–response, while
leaving final answers intact.

This implements Zhang & Nanda's recommendations as a gate in code rather than
as a caveat in prose. The candidate direction is a difference in means fitted
on the **train split only** — fitting it on the problems it is evaluated on
would be circular — and difference in means rather than a probe weight, so the
claim does not rest on a classifier's regularisation path.

---

## D15. Edge cases are kept, with status codes

`no_stable_point`, `stable_at_zero`, `unparseable_final`, `truncated`,
`too_short` all stay in the corpus and are counted in the report.

In particular **problems the model got wrong are kept**. The tail construct
does not presuppose correctness, and excluding them would bias every stopping
comparison toward problems the model already solved.

---

## D16. Unevaluable hypothesis tests stay in the family

Benjamini–Hochberg is applied across the pre-registered family. Tests that
could not be evaluated (too few samples, a single class, identical arms) are
carried through with a null q-value rather than dropped: shrinking *m* would
make the correction look kinder than it is.

---

## D17. The probe's stopping score is out-of-fold, not train-split-fitted

The obvious reading of "fit the probe on the train split" is: fit there, then
score every boundary with that model. But every problem is in the stopping
sweep, including the train-split ones — so their probe scores would be
in-sample, and the probe is the **baseline the primary research question is
measured against**. An inflated baseline makes the verbalised readout look
worse for a reason that has nothing to do with the readout.

The per-boundary score therefore comes from the grouped cross-validation's
out-of-fold predictions: every score is produced by a probe that never saw that
problem. Folds are hashed by problem id, so this is leak-free *and* uses the
whole corpus rather than the eval split alone. The train-split fit is still
made — the causal stage quotes its direction, and its eval-split AUC is the
figure comparable with earlier runs — but it does not feed the sweep.

---

## D18. F6 permutes labels *between* problems, not within

The published form of a probe leakage control permutes labels within group, so
per-group class balance survives and only the activation–label link is
destroyed.

**On this corpus that control is degenerate.** Once a trace states its answer,
the correctness label is the same at every later boundary, so permuting inside
the problem moves nothing: the "shuffled" probe scores exactly as the real one,
which reads as catastrophic leakage while in fact testing nothing. A test on a
corpus with one label per problem reproduces it — the shuffled AUC came out at
1.0.

The control reassigns **whole problems' label sequences between problems**.
Group structure, each problem's internal label pattern and the overall class
balance all survive; what is destroyed is which activations go with which
labels. Both scopes are reported, each with the number of labels it actually
moved, and the verdict is null rather than `false` when a shuffle moved none —
a shuffle that changes nothing is not evidence of anything.

---

## D19. The candidate direction and its controls are read at one dose

`ablate_dir` is swept over a projection-coefficient grid that includes `0.0`,
its own no-op control. `random_dir` and `matched_position` run at full strength
only. Comparing the pooled sweep against those compares a diluted candidate
against undiluted controls, and would let a real effect fail the gate because
its own no-op arm dragged the mean down.

`direction_claim_supported` therefore reads all three arms at the same
coefficient (1.0 by default) and reports the pooled figure separately, next to
the full dose–response curve.

---

## D20. F8 splits its bf16 weights across GPU and host memory

Truncated to the 21 layers it needs, the bf16 target is still about 11 GB,
against 15.65 GB of RAM of which 2–3 GB is typically free. Loaded CPU-only, F8
does not run on this machine — and F8 is the measurement that decides whether
the NLA arm is being fed in-distribution vectors.

F8 is sequenced between `acts` and `nla`, so no other model is resident and the
whole GPU is free. It measures free VRAM and free host memory at load time and
splits the weights across both, falling back to CPU-only when there is no usable
GPU slice. The placement, the measurements behind it and any failed attempt are
recorded in the result, so a slow or failed F8 is readable rather than
mysterious.

---

## D21. An unsampled semantic-entropy arm is absent, not confident

Normalised entropy is `H / log(n_samples)`, which is 0 — maximal confidence —
for `n_samples <= 1`. With **zero** samples that is the wrong answer: fed to
the sweep it is a rule that fires at the first boundary of every problem, and
it would appear on the O3 curve labelled semantic entropy. The larger trace
corpus runs with `semantic_entropy.n_samples: 0` by design, so this was
reachable.

No samples now yields `nan`, which the sweep already reads as "this rule cannot
fire here", and the stage records that the arm is unavailable. One sample still
scores 1.0: a single sample genuinely cannot disagree with itself.

---

## D22. Which runs pool into the supervised corpus is decided, not globbed

`common_data.build_table` fed the two root-level supervised scripts by globbing
`runs/*/traces/traces.jsonl` and keeping the first row per problem id. That
pooled the fixture run (traces from a randomly initialised 64-wide model), the
smoke runs (a different token cap), and — because sorted-path order puts
`mlcorpus` before `pilot` — traces written **before** the chunker and
answer-parser fixes, in preference to the corrected ones on every shared
problem. A stale chunk boundary is baked into the trace; no reparse repairs it.

Comparability is now decided by the settings that determine what a trace *is*
(target, decoding, chunker, AST parameters) **plus the code revision that
produced it**, with the most recently written run as the reference so current
traces are never outvoted by a larger older one. A dirty working tree yields a
revision unique to its run and so never pools automatically; `--runs` overrides
when a human has checked. Every result carries a `provenance.json` naming the
runs, the row count and the signature.

---

## D23. The reconstructor's backbone is placed directly on the GPU

Upstream's `NLACritic` loads its backbone with no `device_map` and then calls
`.to(device)`. That materialises the whole checkpoint in **host** memory first
— for the released AR, 21 quantised layers plus a bf16 embedding table and an
`lm_head` it discards on the next line, about 4.9 GB — and this machine
routinely has 2–3 GB free. The load fails before any reconstruction happens.

`src/nlaast/nla/reconstructor.py` therefore gives `from_pretrained` a
`device_map` for the duration of the construction only, then restores the
vendored module.

**What this changes:** where the weights are put. **What it does not change:**
the backbone, the final-LayerNorm removal, the trained value head, the
read-at-last-token convention and the `MSE = 2(1 − cos)` metric are all still
upstream's, untouched. The subsequent `.to(device)` becomes a no-op. It is a
`setdefault`, so a future upstream that passes its own placement wins; on CPU,
without CUDA, or against a vendored copy that imports differently, the original
behaviour is used unchanged.

Recorded here because it is a modification of upstream's behaviour, narrow as
it is. `tests/test_reconstructor_loading.py` pins that the patch is scoped,
reversible even when loading raises, and inert on CPU.

---

## D24. F8 offloads part of its bf16 arm to disk

The truncated bf16 target is ~12 GB against 6.44 GB of VRAM and, in practice,
2–3 GB of free host memory. The GPU/host split alone does not fit it, so
whatever is left over is written to a disk offload inside the run directory —
not onto the model-cache volume, which is the one the autoencoder download
needs free next — and cleared when F8 finishes either way.

Disk offload is slow. That is the right trade here: F8 runs a couple of dozen
forward passes on a subsample, and the alternative is leaving the quantisation
deviation unmeasured, which is the one thing the NLA arm's credibility rests
on.

**A guard comes with it.** A near-zero mean cosine is ambiguous: it is what
4-bit destroying the activations would look like, and equally what a bf16 arm
that is *not the same model* looks like after a silent load failure or a
position mismatch. The two are indistinguishable in the number, and reporting
the wrong one would condemn the NLA arm on a bug. F8 refuses to report a
near-zero cosine combined with an implausible nf4/bf16 norm ratio as a drift
measurement, and returns `failed` with the diagnosis instead.

F8 is also the only test whose input the pipeline deliberately destroys, so
when the bf16 weights are gone but an earlier pass succeeded, the earlier
result is carried forward labelled `reused_from_earlier_run` rather than
replaced by a `blocked` one.

---

## D25. FVE is reported against two baselines, and the empirical one is primary

`FVE = 1 − mean(MSE) / baseline`. A baseline of 2.0 assumes an uninformative
prediction is *orthogonal* to the target. A random direction in 3584
dimensions is — but a plausible guess is not, and layer-20 residual streams
are strongly anisotropic. Measured on the pilot's own vectors, two drawn at
random have a **mean cosine of 0.586**, so guessing a typical layer-20
activation already achieves MSE 0.829.

§5 of the plan asks for "the empirical variance baseline of our own activation
sample", and the difference is not cosmetic:

| | against 2.0 | against 0.829 |
|---|---|---|
| real windows | 0.866 | **0.677** |
| Gaussian control | 0.391 | **−0.470** |

The control is the point. Against 2.0 it looks like partial signal; against the
empirical baseline it reconstructs *worse than guessing*, which is what an
uninformative control should do. Both are reported, the empirical one named as
the figure to read. The checkpoint card's 0.752 uses an unstated baseline and
is therefore comparable to neither — the report says so rather than inviting
the comparison.

---

## D26. A budget that cannot be matched is not compared

`at_matched_budget` returned each rule's *nearest* operating point regardless
of distance. A safety/saving curve is a step function — the rule fires at a
boundary or it does not — and on a small corpus the steps are wide. Asking all
four pilot rules for a saving of 0.632 returned convergence at 0.407 and the
probe at 0.855: a 45-point spread, with their safe-stopping rates then
compared as though the budgets were equal. That is exactly what D9 exists to
prevent, and the table looked matched.

A point must now lie within **0.05** of the target to count. On the pilot that
leaves one evaluable head-to-head of three. The two that cannot be made stay
in the pre-registered family as unevaluable (D16) rather than disappearing
from it, and the comparison table carries the realised saving, the gap and a
`matched` flag per row.

Each head-to-head also records both arms' mean safe rates and the direction in
words, because a reader scanning q-values can read a significant result as a
win for the readout when here it is the opposite.

---

## D27. A constant outcome is reported as untested, not as refuted

`direction_claim_supported` compares the candidate direction against two
controls on the change in rechecking-marker count. On the pilot that count is
**identically zero**: not one of the twenty markers occurs in any of the 24
traces, because this model writes clean step-by-step derivations. With a
constant outcome `|0| > |0|` is false, every comparison fails, and the gate
returned "not supported" — which reads as evidence against a tail direction
when there is none either way.

The gate now checks whether the outcome varies before judging it, and returns
`supported: null` with the reason when it does not, with both control
comparisons null rather than false. An experiment with no power is a different
statement from a negative result.

The same principle already applied to F6 (`leak_suspected: null` when the
label shuffle moved nothing) and F5 (`not_applicable` for a `k` the recorded
continuations cannot simulate). It is now applied consistently.

---

## D28. Provenance is recorded per stage, not once per run

The manifest captured git state when the run directory was created and never
again. The pilot then spanned five days and some sixty commits, most of the
methodological fixes among them, so the single commit the report named
predated the code behind nearly every number in it.

`start_stage` now records the commit each stage ran under, and the report lists
the distinct ones — including, honestly, how many stages have no such record
because they predate the change.

---

## D29. The matched-budget comparison is tested over problems

The pilot's one significant primary finding was an artefact of a default.

O3 reads each rule off its curve at a shared budget and compared the two safe
rates with a Wilcoxon signed-rank test over the **nine requested budgets**.
All nine resolved to a single pair of operating points — safe rates 0.095 and
0.286 at every one of them — so the test saw nine copies of one comparison.
Its p-value was therefore a function of how many budgets were asked for:

| budgets | 3 | 5 | **9** | 15 | 25 |
|---|---|---|---|---|---|
| Wilcoxon p | 0.250 | 0.0625 | **0.0039** | 0.000108 | 0.000001 |

Nine is `budget_grid`'s default, and 0.0039 was reported as q = 0.0065.

Safety is a per-problem binary outcome and both rules are scored on the same
problems, so the unit of analysis is the **problem**. `paired_binary_test`
runs McNemar's exact test over problems and `paired_rate_difference_ci` gives
the matching conditional-exact interval — `rate_difference_ci` assumes the two
proportions are independent, which is precisely what a matched comparison is
not. Repeated operating points are also collapsed before testing.

On the same data the comparison against semantic entropy is 4 discordant
problems of 21, all four favouring semantic entropy: **p = 0.125, q = 0.156**.
The direction survives and the significance does not. The study's headline is
now that the readout is directionally worse than semantic entropy on a corpus
too small to establish it.

---

## D30. The threshold sweep enumerates the curve instead of sampling it

Two of three head-to-heads were reported unevaluable for the wrong reason.

Every rule was swept on 25 evenly spaced thresholds between its minimum and
maximum score. That samples the *score axis*, so the result depends on how the
scores happen to be distributed. The hidden-state probe's are logistic and
saturate near 0 and 1, so 25 linear thresholds found 5 operating points and
none within tolerance of a shared budget — and the comparison was recorded as
"not evaluable on this corpus", implying a sample-size limit.

A curve can only change at a threshold that crosses an observed score, so
`complete_threshold_grid` sweeps the observed values and enumerates every
achievable operating point exactly. The same 21 problems then give the probe
**86** operating points and a match at all nine budgets, where it turns out to
be *exactly tied* with the readout: zero discordant problems, identical safe
rates, the two rules preserving the outcome on the same problems.

`budget_grid` now also excludes operating points that save nothing. Every
complete curve has one — the threshold above which the rule never fires — and
matching two rules there compares nothing, since neither stops and both are
trivially safe.

---

## D31. An unmatched comparison says whether more data would help

With the thresholds exhausted, a comparison that still cannot be matched is
informative, and the two reasons it can happen call for opposite responses.

`granularity_bound` reports the most operating points a rule's curve can ever
have: a score taking `d` distinct values admits at most `d + 1` thresholds
that behave differently, **at any corpus size**. The agreement rule's score is
1.0 once the last `window` answers agree and 0.0 otherwise, so `d = 2` and its
curve has three points on 21 problems and three on 21,000. Its two savings,
0.407 and 0.885, straddle the budget window without entering it.

That comparison is therefore not waiting for more data; it needs a graded
score. The `baselines` stage records the bound per rule and warns when a score
is this coarse, and O3 reports it as the reason rather than as a shortage of
problems — which is what the pilot's resume notes had concluded.

**Superseded in part by D36.** The graded score now exists, and the diagnosis
above was correct about the mechanism but wrong about the remedy's effect. With
the run-length score the agreement rule has `d = 7` on the pilot and `d = 9` on
the main run so far, so `granularity_limited` is now false and the bound is no
longer what blocks the comparison. It is still unevaluable, for the narrower
reason D36 gives.

---

## D36. The agreement baseline is swept over its own window, not a derived flag

The pre-registration fixes the baseline as a *boundary* and names its parameter
as free:

> **Answer-convergence boundary** `b_conv(w)` — smallest `i` such that
> `a_i..a_{i+w-1}` are all non-null and pairwise equivalent. This is the Liu &
> Wang baseline, **`w` configurable**.

O3 compares rules on a safety/saving curve, which requires a swept threshold,
so the baseline was adapted by scoring 1.0 once `w` answers agree at one `w`
taken from config and 0.0 otherwise. That adapter fixes the rule's own
parameter and then sweeps a derived indicator of it — and `w` is precisely what
moves where the rule fires. Two distinct values give three operating points at
any corpus size (D31).

**The score is now the agreement run length**: how many consecutive boundaries
ending here carry one answer, 0 where the answer did not parse. The identity
that licenses this is

```
first i with run_length[i] >= w   ==   convergence_boundary(parsed, w)
```

because "the last `w` answers agree" and "the agreement run here is at least
`w` long" are the same statement. So every operating point on the curve *is*
the Liu & Wang rule at one `w`, and the sweep axis is the parameter the
pre-registration left free. This is a re-parameterisation, not a new rule, and
the claim is tested rather than argued: `tests/test_ast_detector.py` asserts
the identity over nine answer patterns and every `w` up to the trace length,
and it was checked against the canonical `convergence_boundary` on all 24 pilot
and 59 then-available main traces for `w = 1..8` — 83 traces, zero mismatches.

**It does not make the comparison evaluable, and that was verified before
implementing rather than hoped for afterwards.** On the pilot the achievable
mean savings go from three values to eight, and they stay clustered:

```
0.0, 0.024, 0.025, 0.059, 0.097, 0.407, 0.855, 0.885
```

The shared budget window of 0.632–0.675 falls in the gap between `w = 2`
(0.407) and `w = 1` (0.855). `w` is an integer and a threshold of 1.5 selects
the same boundaries as 2, so nothing is reachable in between.

What changed is the honesty of the diagnosis. O3 no longer reports the score as
too coarse; it reports that the thresholds are exhausted and the two curves do
not overlap near a shared budget. Whether a larger corpus helps is now an open
question rather than a settled no: the *gap* between adjacent `w` is structural,
but where the shared window sits depends on both curves on the actual corpus,
and the window is set by the NLA rule's own achievable range. D31's flat "no
corpus size will help" was over-stated and is corrected here.

Everything else about the pilot is unchanged by this: semantic entropy still
q = 0.156, the probe still exactly tied, RQ2 and the F10 family identical. The
full-corpus budget grid is also unchanged, so no other comparison moved.

## D32. The corpus is sized by discordant problems, not by configured N

Because O3 is now a McNemar test, its power comes from **discordant**
problems — those where one rule preserved the verified outcome and the other
did not. Concordant problems contribute nothing however many there are, so
sizing a run by its total N sizes it by the wrong quantity.

`scripts/size_o3_corpus.py` computes this from a completed run. Eight
discordant problems are the fewest that can clear q = 0.05 in a family of five
evaluable tests, and at the pilot's observed rate of 4/21 = 0.190 that is
about **42 problems in the NLA arm**. Simulated power at 80 problems is 0.99
if discordances keep falling the pilot's way and 0.71 if one in ten favours
the readout.

This is why `main.yaml` keeps 250 problems rather than being cut: the arms are
resumable and the order is now interleaved (D33), so the run can be stopped
once the primary comparison has the discordant problems it needs, and the
achieved N is a valid sample at whatever point that is.

---

## D33. Any prefix of a run is a proportional sample

`main.yaml` states that the achieved N, not the configured N, is what the
report states — which is only sound if stopping early leaves a usable corpus.
It did not. `load_problems` concatenated the datasets, so all 150 GSM8K
problems preceded all 100 MATH ones and a run halted at 120 would have held no
MATH at all: F2 would have had one arm, every pooled quantity would have been
biased towards the easier benchmark, and the output would have reported only
the achieved N, not that the whole shortfall fell on one dataset.

`interleave_by_dataset` spreads each dataset across the order in proportion,
so a prefix of 24 holds 14 GSM8K and 10 MATH and a prefix of 120 holds 72 and
48. The order depends only on the group sizes, so a resumed run keeps it.

The NLA arm's cap had the same defect and worse consequences. It took its
subset as a prefix of the sorted problem ids, and ids sort `gsm8k-...` before
`math-...`, so `nla.max_problems: 80` on 250 problems would have selected 80
GSM8K problems and no MATH — and the primary O3 head-to-head is computed on
whatever that arm covers. `stratified_cap` gives each dataset its proportional
share, filled with a deterministic prefix within the dataset so the subset
stays stable as the corpus grows.

---

## D34. The direction interventions are verified to have happened

Three of the five causal arms were no-ops, and nothing said so.

`projection_editor` restricted its edit to `[start, end)` **absolute token
positions**, but applied the span as an index into whatever tensor the forward
hook received. The arms edit the generated continuation, so the span starts at
the prompt length; cached decoding calls the hook once with the whole prompt
and then once per new token with a sequence length of one, and the span never
intersected either. The prompt call ends where the span begins and each decode
call is one position wide.

The pilot's own output is unambiguous once read for it: on all ten problems
`ablate_dir` returned identical answers and identical token counts at
coefficients 0.0, 1.0 and 2.0, and dose 0.0 is a no-op by definition.

The editor now carries a position counter, and — the part that matters more —
counts the positions it actually modified. The stage records that per result
and warns when a non-zero dose edits nothing, and the outcome screen reports
the arm set as a broken independent variable rather than as a negative result.
The defect survived a completed run because nothing asserted that the
intervention had occurred.

This revises what RQ3's failure consisted of. It was recorded as a missing
dependent variable, with the direction fittable and the marker readout empty.
Both were true; the independent variable was also not being applied.

---

## D35. A replacement readout for RQ3 is screened, not chosen

The pre-registered RQ3 readout is the change in rechecking-marker count and it
is identically zero on this model (D27). Swapping in whichever alternative
produces the nicest answer would be choosing the hypothesis after seeing the
data.

`screen_outcomes` instead reports every candidate readout against the one
property a dependent variable needs before it can test anything: that it
varies, and that it separates the candidate arm from its controls. The verdict
is still computed from the pre-registered readout. The screen says which
readout a *later* pre-registration could use.

`continuation_tokens` is the leading candidate — already recorded, continuous,
and a question about how much further the model writes rather than about what
the text means, which section 2.1.6 of `Project_Review_II.md` rules out.
Whether it separates the arms cannot be known until the arms do something,
which needs D34's fix and a GPU pass.

---

## D37. Two decisions the pre-registration does not settle

Recorded as **open**, with the implementation's current behaviour stated so
that no result is reported under an unstated convention. Neither was chosen to
favour an outcome; both were found by reading the implementation against
PROJECT_PLAN.md, and on the pilot neither changes a verdict.

### 37a. The Benjamini–Hochberg family: evaluable or pre-registered?

**What the pre-registration says.** F9, in full: *"Multiple testing | Pre-
registered family; Benjamini–Hochberg at q=0.05."* That fixes **which** tests
may be corrected over — the pre-registered set, not a post-hoc selection — and
is silent on a test that runs and yields no p-value.

**What the implementation does.** `benjamini_hochberg` takes
`m = len(usable)`, the tests with a finite p-value. On the pilot nine tests
were pre-registered and five were evaluable, so `m = 5`.

**Why the two readings differ by a factor of 1.8.** Reading "family" as the
pre-registered count gives `m = 9`. Every q-value scales with `m`, so the
pilot's RQ2 q of 8.96e-119 and F10's 0.0229 would become 1.61e-118 and 0.0413.
No pilot verdict flips — F10's tail-vs-matched-position result stays
significant at 0.0413 and the O3 comparison clears neither — but on a larger
corpus it could decide one.

**Three places had claimed the opposite of what the code does**, stating that
unevaluable tests remain in the family so the correction "is not flattered by
their absence": `benjamini_hochberg`'s docstring, a comment in
`aggregate_results.py`, and the execution report's own prose. All three are
corrected to describe `m = len(usable)`; the report now prints the factor by
which the other convention would change its q-values.

**The argument for the implementation's choice**, which is why it was left
alone rather than changed: an unevaluable test has no rejection region, so
including it in `m` tightens every other test's threshold on the strength of a
test that could never have rejected. That is not conservatism, it is a
penalty paid for nothing. Excluding it is also the standard reading of BH.

**The argument against**, which is why this is open rather than closed: the
pre-registered family is what guards against choosing the denominator after
seeing which tests worked, and "evaluable" is itself determined by the data.
A reader could reasonably hold that the family was fixed in advance at nine.

**Resolved by D38**, which found that one of the four exclusions was not
independent of the effect under test, fixed it, and settles the convention on
that basis. The paragraph below records what was in place before that.

`summarise_tests` reports `bh_family_size`, `n_planned_but_unevaluable` and
`bh_family_convention` explicitly, so every q-value is reproducible from the
artifact. `size_o3_corpus.py` kept `DEFAULT_FAMILY_SIZE = 5` — the pilot's
evaluable count — and warned when the run being sized corrected over a
different `m`.

### 37b. D31's "no corpus size will help" is withdrawn

D36 already corrected this in substance; D37 records where the stale claim
survived and what replaced it.

The claim was argued from the binary convergence score (`d = 2`, three
operating points at any N). D36 replaced that score with the agreement run
length, an **unbounded count**, so the resolution bound is computed from the
run lengths *observed* and rises with the corpus: 8 on the pilot's 24 traces,
10 on the first 86 traces of the main run. `granularity_limited` is now
`false`, and the O3 stage already reported the weaker, correct reason — *"the
two curves do not come within tolerance of a shared budget on this corpus
rather than the sweep having missed a point"*.

The stale claim survived in `SESSION_HANDOFF.md` §3 and §8, `docs/STATUS.md`,
`granularity_bound`'s docstring and summary line, `agreement_run_length`'s
closing paragraph, the `baselines` stage's warning, the O3 `reason` string and
the execution report's prose. All are corrected to scope the statement to the
corpus measured.

**What is observed**, measured with the project's own sweep functions over all
usable traces:

| corpus | `w = 2` | `w = 1` | gap |
|---|---|---|---|
| pilot, 24 traces | 0.464 | 0.861 | 0.398 |
| main, first 86 traces | 0.395 | 0.886 | 0.491 |
| main, first 100 traces | 0.419 | 0.882 | 0.464 |

The gap is wider on the main run than on the pilot at both checkpoints. It is
**not monotone** — 0.398, 0.491, 0.464 — so this is a noisy quantity observed
three times, not a trend, and the caution above was warranted. The mechanism
for it being wider at all is intelligible: in a longer trace the first parsed
answer arrives relatively earlier while the first *repeated* answer arrives
relatively later. But **100 traces is not 250, and no projection is made.**
Two quantities decide the full-corpus outcome and neither is known yet:

1. The achieved savings are **means over problems** and move with the corpus.
2. The window is the intersection of every rule's reachable range
   (`budget_grid`) and on the pilot was pinned by the NLA arm, whose band is
   narrow because it is scored only at sampled windows and carried forward.
   Where that band lands on `main` is unknown until the `nla` stage runs.

**No claim is made, in either direction, about whether the full corpus will
make this comparison evaluable.** The structural part of the finding — that no
operating point exists *between* adjacent integer `w` — stands at any corpus
size and is unaffected by the above.

---

## D38. Zero discordant pairs is a tested null, and that settles the BH family

D37a left the Benjamini–Hochberg denominator open: correct over the evaluable
tests (`m = 5` on the pilot) or over the pre-registered family (`m = 9`)? F9
says only "pre-registered family; Benjamini–Hochberg at q=0.05".

The question turned out to be answerable, because the right question is not
"how many tests were planned" but **"is each exclusion independent of the
effect being tested?"** That is what F9's "pre-registered family" protects:
not a particular integer, but the analyst's inability to pick the denominator
after seeing which tests came out well. BH controls `E[V/R]` over the
hypotheses actually tested, and a hypothesis that yields no p-value can be
neither a rejection nor a false one — so excluding it costs no error control,
*provided the exclusion rule does not correlate with the effect*.

### The four ways a planned test can yield no p-value

The pilot's nine-member family contains all four, and they are not alike:

| test | why no p-value | correlated with the effect? |
|---|---|---|
| RQ1 tail fraction, correct vs incorrect | fewer than 3 problems with an incorrect final answer | No — set by accuracy, not by the tail fractions compared |
| O3 vs convergence | no matched budget exists | No — set by the token-saving curves, not by safe rates |
| RQ3 marker change | the readout is identically zero, so Mann–Whitney has no variance | No — an instrument floor, not the arms' means |
| **O3 vs probe** | **zero discordant pairs** | **Yes** |

The last one is the defect. Zero discordance means both rules preserved the
verified outcome on *exactly* the same problems — which is the null holding
perfectly. Dropping that test shrank `m` **because a null was true**, and a
smaller `m` makes every other q-value smaller. The family was therefore
contracting in an effect-dependent way, in the liberal direction, under a
procedure whose entire purpose is to be conservative about multiplicity.

### The fix

`paired_binary_test` returned `nan` at `n_discordant == 0`. That is wrong as
arithmetic, not just as framing. McNemar's exact test is *conditional* on the
number of discordant pairs; conditioning on `d = 0` leaves a sample space with
one point, `b = 0`, so `P(as extreme or more) = 1` and the two-sided exact
p-value **is** 1.0. The test was conducted and not rejected.

It now returns `p = 1.0` with `notes['zero_discordant'] = True` and an
`interpretation` note saying the comparison is *uninformative* rather than
*unevaluable* — nothing could have rejected, because there was nothing to
assign. The rate-difference interval, which is the informative part, was
already being returned and still is.

**Effect on the pilot, replayed from its own p-values without re-running it:**

| | before | after |
|---|---|---|
| `m` | 5 | **6** |
| significant at q=0.05 | 3 | **3** |
| RQ2 | 8.96e-119 | 1.08e-118 |
| F10 tail vs matched_position | 0.0229 | **0.0275** (still significant) |
| F10 tail vs matched_length | 0.372 | 0.446 |
| F10 tail vs gaussian_control | 3.45e-34 | 4.14e-34 |
| O3 vs semantic entropy | 0.156 | 0.188 |
| O3 vs probe | *absent* | **q = 1.0, reported** |

Every q-value rose — the fix is strictly conservative and cannot manufacture
a finding — and **no verdict changed**. The comparison that used to vanish is
now on the table as a null.

### The convention, and what is still the author's call

With the effect-correlated exclusion removed, every remaining exclusion is
outcome-independent, so **correct over the evaluable tests** is the primary
convention. It is implemented, and the sizing script now reads `m` from the
run's own analysis instead of a constant, so the power target cannot drift
from the correction.

`summarise_tests` additionally reports `q_values_at_planned_family_size` — the
same p-values corrected over every pre-registered test — plus
`verdict_is_robust_to_family_convention`. On the pilot that flag is **true**:
three findings either way. So the choice changes no conclusion there, and both
sets of numbers are in the artifact regardless.

**What needs approval:** which of the two to foreground in the paper. The
recommendation is evaluable-as-primary with the pre-registered count as a
stated sensitivity, for the FDR argument above. If the main corpus returns
`verdict_is_robust_to_family_convention = false`, the choice decides a
finding and must be stated explicitly in the Results rather than assumed —
the flag exists so that case cannot pass unnoticed.

### A second thing this exposed, now reported

`notes['smallest_attainable_p']` is the p-value a comparison's discordant-pair
count would give if every pair fell the same way — the best it could possibly
have done. It separates *no difference was found* from *no difference could
have been found*, with no simulation. On the pilot's four discordant pairs it
is 0.125, against a corrected target near 0.01: **that comparison could not
have reached significance whichever way the pairs fell.** Eight discordant
pairs is the minimum that can, which is what `size_o3_corpus.py` independently
reports and is unchanged at `m = 6` (target 0.0083).

---

## D39. The two RQ3 controls are matched on different things, and that is reported

The concern raised was that the candidate's span (`tail_tokens + 64`) is
longer than the matched-position control's (`≈ tail_tokens`), confounding
*where* the edit happened with *how much* was edited. Measured against the
pilot's own recorded spans and generation lengths, the premise is wrong in two
ways and the real asymmetry runs the other way.

**The `+ 64` never binds.** The candidate's span is a median of 148 tokens,
but the regenerated continuation is a median of 35.5, and the editor can only
edit positions the forward pass actually produces. So the candidate edits a
median of **35.5** positions — the whole continuation — and the 64-token slack
is a non-binding safety margin, not an extent. Trimming it would change
nothing.

**The control edits about twice as many positions, not fewer.** The
matched-position window lives inside the *prompt* (the prefix is fed as
prompt, so the pre-stabilisation window is prompt context), which arrives in a
single forward pass, so the control edits its whole span: a median of **72**.
The window is the pre-registered matched-length one — the same number of
chunks as the tail — and the original tail is longer than its regeneration,
which is where the 2x comes from.

**The two controls therefore do different jobs, and only one is extent-matched
— by construction, not by accident:**

| contrast | holds constant | varies | matched on extent? |
|---|---|---|---|
| `ablate_dir` vs `random_dir` | span, dose, position | the direction | **Yes, exactly** — same span object |
| `ablate_dir` vs `matched_position` | direction, dose | the position | **No** — and it cannot be, see below |

The direction-specificity test, which is the one carrying "a tail direction
exists", is clean. The location test is not, and the difference is not only
extent: the candidate edits the representation of tokens *as they are
generated*, so its edit compounds into what gets written next, while the
control edits already-fixed context that influences generation only through
attention. Those are structurally different interventions. Matching the counts
would not make them the same experiment.

**The three options considered.**

- **A — match the edited-position counts.** Cannot be done by trimming
  `+ 64`, since it never binds. It would require setting the control window
  from the *realised* generation length, which is unknown until after
  generating: a two-pass design. Changes the control from the pre-registered
  matched-length window to a post-hoc matched-count window.
- **B — keep the candidate span and grow the control to match.** Feasible
  without contaminating the candidate region, since the control extends
  backwards into the prompt and the candidate starts at `prompt_tokens`. But
  it abandons the pre-registered "window of length L" definition, and matches
  the control to a *span* the candidate never actually fills.
- **C — keep both and report the asymmetry.** The spans stay faithful to §5:
  the candidate covers the generated tail, the control is the registered
  matched-length window.

**C is implemented**, because A and B both redefine a pre-registered control
to fix a confound that B does not remove (the online/static difference
remains) and that is now measurable instead of hidden. `screen_outcomes`
reports `edit_extent_by_arm`, `extent_matched_candidate_vs_random` and
`extent_ratio_matched_position_over_candidate`, so a reader comparing the two
arms sees the extent beside the means.

**The inferential limitation, stated for the write-up.** A difference between
`ablate_dir` and `matched_position` cannot be attributed to position alone:
the control also edits roughly twice as many positions, and edits context
rather than generation. Note the direction of that bias — the control is the
*larger* edit — so if the candidate nonetheless moves the readout further,
the extent difference works against the candidate and the conclusion is
conservative. If the control moves it further, extent is a live alternative
explanation and the comparison does not support the location claim. **This
asymmetry should be stated in the Results, and option A or B would need a
pre-registration amendment.**

---

## D40. RQ3 has no pre-registered dependent variable, and `marker_delta` was falsely flagged as one

**What the pre-registration fixes.** §2's table: "Causal sequence:
truncation, tail replacement with neutral filler, matched-position and
matched-random-direction comparisons, per Zhang & Nanda". §5: the five arms,
and "the claim 'a tail direction exists' requires `ablate_dir` to beat
**both** `random_dir` and `matched_position`, with a dose–response curve over
projection coefficient."

So the arms, both required comparisons, the conjunction and the dose-response
requirement are all registered. **"Beat" is not defined, and no dependent
variable is named anywhere** — not in PROJECT_PLAN.md, and not in
`Project_Review_I.md` or `Project_Review_II.md`, where the only occurrences of
"marker" are about end-of-thinking tokens in unrelated prior work.

**The defect.** `screen_outcomes` set `pre_registered: True` on
`marker_delta`, and its note said "the verdict is computed from the
pre-registered `marker_delta`". Both asserted a commitment that was never
made: the twenty-item lexical rechecking list is an implementation default.
Recording it as registered is exactly the error the screen exists to prevent,
pointed the other way — instead of choosing a readout after seeing the data,
it backdated the one chosen before it.

Fixed: no outcome is flagged registered, `preregistered_outcome` is `None`,
and the readout the gate reads is identified as `verdict_outcome` /
`is_verdict_outcome` — a choice, labelled as one. `VERDICT_OUTCOME` is now a
module constant so the gate and the screen cannot disagree about it.

**A second defect, in the gate itself.** `monotone_dose_response` was computed
from `marker_delta` but, unlike `beats_random_direction` and
`beats_matched_position`, was not gated on the readout varying. "Non-decreasing
in |delta|" is vacuously true of a constant, so an identically-zero readout
reported `monotone_dose_response: true` — and the pilot's `summary.json` says
exactly that, next to `outcome_varies: false`. One of the four conditions
appeared met when none was testable. It is now `None` when the readout does
not vary, and additionally requires the curve to move (`len(set(deltas)) > 1`)
rather than merely not to go down: a dose-response claim needs a response.
`answers_preserved` is deliberately *not* gated this way, because it is
measured on a different variable; its own vacuity under a no-op intervention
is what `interventions_were_no_ops` reports.

Verified that the gate can still be satisfied: on the positive fixture all
four conditions and the conjunction remain `True`, so only vacuous truths were
removed.

**What is still needed, and it is a scientific decision.** RQ3 cannot produce
a valid test until a dependent variable is registered. The amendment needs
four things, and the screen is built to supply the evidence for the first:

1. **Outcome.** `continuation_tokens` is the leading candidate — already
   recorded, continuous, and a question about how much further the model
   writes rather than about what its text means, which
   `Project_Review_II.md` §2.1.6 rules out. It is **not** designated here.
2. **Direction.** Which way ablation is predicted to move it, stated before
   the data are seen. Without this the "beat" test is two-tailed on an
   undirected quantity and `abs()` comparisons are uninterpretable.
3. **Comparisons.** Both are already registered: beat `random_dir` *and*
   `matched_position`, conjunctively, read at the controls' dose
   (`compare_at`), with D39's extent caveat on the second.
4. **Dose-response criterion.** Currently "non-decreasing in |outcome| over
   the dose sweep, with at least three doses and a curve that moves". That is
   the implementation's reading and should be registered explicitly, including
   whether it is required conjunctively or reported alongside.

Until that amendment exists, `screen_outcomes` should be run on the GPU
validation pass as a **screening** step — to find which candidate readouts
vary and separate the arms at all — and the registered choice made from that
*before* any confirmatory run. Picking the readout from the run that is then
reported is what the screen's own note forbids.

---

## D41. RQ3's construct *is* registered; only its operationalisation is not

This supersedes part of D40 and **reverses the direction D40 was leaning in.**
D40 recorded that RQ3 "names no dependent variable" and treated
`continuation_tokens` as the leading candidate to register. Re-reading the
contemporaneous materials shows the first half was too strong and the second
half was a construct error.

### What is actually registered

`Project_Review_II.md` line 122, in the proposal's own statement of the
research question:

> **RQ3.** Under matched-control interventions, do tail-window states carry a
> direction whose ablation **selectively changes verification-like behaviour**
> without changing final answers?

So the **construct is registered**: verification-like behaviour, changed
selectively, with final answers preserved. `PROJECT_PLAN.md` §5 then registers
the five arms, both control comparisons, their conjunction and the
dose-response requirement. What no document supplies is an **operationalisation**
— how verification-like behaviour is to be measured, and what numerical
relation counts as "beat".

That is a narrower gap than D40 claimed, and it constrains the remedy. An
amendment may supply a *measure of the registered construct*. It may not
substitute a different construct and keep RQ3's name.

### Why `continuation_tokens` is not the answer

`continuation_tokens` is how many tokens the model generates after the
intervention. That is **output length**, not verification. It kept being
proposed because it has variance where the marker count has none, and because
it needs no judgement about what the text means — both true, and neither makes
it a measure of verification-like behaviour. A model can write more because it
is rechecking, or because it is repeating itself, degenerating, or padding.
Adopting it would answer "does ablating this direction change how much the
model writes?" under the heading of a question about verification.

Choosing a readout because it is the one that moves is also precisely the
failure mode `screen_outcomes` exists to prevent, approached from the other
side: not picking the outcome that gives the nicest answer, but picking the one
that gives *any* answer.

`OUTCOME_CONSTRUCT` now records what each readout measures,
`ELIGIBLE_OUTCOMES` is the subset measuring the registered construct, and
`_assert_eligible` raises if the verdict is asked for an out-of-construct
readout. `continuation_tokens` stays in `OUTCOMES` as a screening candidate and
is reported, but cannot become the verdict outcome by editing a constant.

### The decision

**`marker_delta` is retained as the operationalisation**, and RQ3's
confirmatory claim is **left blocked on this model**. Reasoning:

- It is the face-valid measure of the registered construct: a count of
  rechecking markers, scored on the generated text, compared between arms. The
  code already documents that it is a behavioural readout and not a claim to
  read text as a log of the computation, which `Project_Review_II.md` §2.1.6
  rules out.
- Its defect is a **floor effect**, not a construct mismatch. That is a
  statement about measurement power, and the honest response is to report RQ3
  as unevaluable on this model rather than to swap the construct.
- Widening the marker lexicon until it moves would be tuning the instrument
  against the data, which is the same error in a different coat.

So: **no amendment to the dependent variable is made.** The gap is documented,
the construct is cited, and the claim stays blocked. This is the "strongest
valid alternative" rather than a manufactured answer.

### The floor effect, measured at scale rather than asserted

The pilot's 24 traces were too few to distinguish "this model does not
recheck" from "we have not seen enough traces". Re-measured across the
pilot **and the 116 main-run traces available on 2026-10-12** — 140 traces,
5.8x the pilot:

| corpus | traces with any marker | total marker occurrences | rate | 95% exact CI |
|---|---|---|---|---|
| pilot | 0 / 24 | 0 | 0.0000 | [0.0000, 0.1425] |
| main at 116 traces | 1 / 116 | 1 | 0.0086 | [0.0002, 0.0471] |
| pooled at 140 traces | 1 / 140 | 1 | 0.0071 | [0.0002, 0.0392] |
| **pooled at 200 traces** (2026-10-15) | **1 / 200** | **1** | **0.0050** | **[0.0001, 0.0275]** |

The single occurrence is the word "confirming", once, in one trace. Not one of
the other nineteen markers appears anywhere in 200 traces. Re-measure it with
`scripts/measure_rq3_readout_floor.py`, which is read-only and CPU-only.

**What that implies for the causal arm**, which runs on `causal.max_problems`
= 60 problems:

| measured over | expected marker-bearing problems in the arm | at the interval's upper bound |
|---|---|---|
| 140 traces | 0.43 | 2.35 |
| **200 traces** | **0.30** | **1.65** |

`MIN_PAIRS` is 3, and a paired test with any power needs far more.

**The projection has moved against the readout as the corpus has grown, not
for it.** The count stays at one while N rises, so the interval's upper bound
falls — 0.0392 to 0.0275 — and with it the most optimistic number of usable
problems: 2.35 to 1.65. At 200 traces, three marker-bearing problems are not
expected until the arm holds about **109** problems *even at the top of the
interval*, against a configured 60 and roughly 130 eval-split tailed problems
to draw from. This is the sense in which finishing the corpus cannot help:
each additional clean trace tightens the bound downwards.

This is the evidence that converts "blocked, pending more data" into
**"blocked, and more data will not help"**. It is a finding about Qwen2.5-7B-
Instruct's output — it writes clean step-by-step derivations without
rechecking language — and not a defect in the study design. Reported as such.

**It is also the one claim in this study that is already settled before the
run finishes.** Every other blocked question waits on the corpus; this one is
decided by a base rate the corpus is still lowering.

### What this costs, stated plainly

RQ3 cannot return a positive or negative result on this model with this
readout. Three independent reasons, all now reported rather than inferred:

1. **The dependent variable has no variance** (floor effect, above).
2. **The independent variable is unvalidated.** D34's fix has never executed on
   a GPU, so no run has demonstrated a non-zero edit. The RQ3 family test now
   refuses to report a p-value unless the candidate arm is *demonstrated* to
   have edited a position, so an identical-arms result under a no-op can no
   longer be mistaken for a tested null.
3. **The location contrast is confounded with extent** (D39), which the report
   now states along with the direction of the bias.

### Routes out, for a later study rather than this one

- **Register an in-construct measure with better range.** A validated
  verification-behaviour measure — for example a rubric scored by a judge
  model, with inter-rater agreement reported — would measure the registered
  construct without a lexical floor. It needs instrument validation that is out
  of scope here, and it must be registered *before* the confirmatory run.
- **Amend RQ3 itself.** If output length is the quantity of interest, say so in
  the question. That is a change to the research question, not to its
  measurement, and must be dated and justified as one. This decision record
  does not make it.

### Separation of roles, for the write-up

- **Originally registered:** the construct (Review II line 122), the five arms,
  both control comparisons, the conjunction, the dose-response requirement
  (PROJECT_PLAN.md §5).
- **Not registered, chosen in implementation:** the operationalisation
  (`marker_delta`), the "beat" relation, `alpha`, `MIN_PAIRS`, `MIN_DOSES`, and
  the 0.2 answer-change ceiling. All are now named constants in one module so
  they are visible as choices.
- **Exploratory:** everything `screen_outcomes` reports about which candidate
  readouts vary and separate the arms. Screening may inform a *future*
  registration; it may not be read as confirmatory evidence in the same run.
- **Still necessary and not performed:** a GPU run demonstrating a non-zero
  edit, and — only if an in-construct readout with range is registered first —
  a separate confirmatory run. Reusing a screening run as its own confirmation
  is the thing this record exists to prevent.

### "Beat" is now defined, which it was not

Independently of the outcome question, the relation was indefensible: the gate
compared two arm *means*, so a candidate at 0.30 against a control at 0.29
satisfied it on ten problems with no test. It is now a Wilcoxon signed-rank
test paired over the problems the arms share, requiring both the right
direction and `alpha = 0.05`, read at the controls' dose. This is the
correction D29 made to the primary O3 comparison; the two are now consistent.
Nothing about this weakens a gate — it replaces no test with a test.

---

## D42. Three RQ1 summary statistics did not mean what they were called

Found by recomputing `ast/summary.json` from `ast/ast.jsonl` rather than
reading the prose. Every other field in that summary reproduced exactly; these
three did not, and two of them are quoted in the README and in
[`STATUS.md`](STATUS.md).

**`median_tail_fraction` was not a median.** It was `sorted(fr)[len(fr) // 2]`,
the *upper* of the two central values at even `n`. On the pilot's 21 tailed
problems that coincides with the median, so the error was invisible; at 250
problems the tailed count is as likely to be even as odd. It is now
`statistics.median`, which is what every other median in the codebase already
used.

**`tail_token_share` has two denominators and only one was reported.** It
divides the tail's tokens by the tokens generated in traces that *have* a
tail. That is the right denominator for "how much of a tailed trace is tail",
and it is not the share of all generated tokens — which is how the report, the
README and `STATUS.md` all described it. The traces without a tail are not a
random sample: they are the ones that hit the generation cap, so they are the
longest. On the pilot the two are 1885/5943 = **0.317** and 1885/7479 =
**0.252**. Both are now computed, named and printed, so neither can be read as
the other. The relaxed-budget interval also now says when its two ends do not
share a denominator.

**`mean_gap_chunks` is signed, and was printed as a gap.** Earlier and later
cancel in it. On the pilot the agreement rule fires earlier than the AST on
57.1% of problems — by **5.5 chunks** where it does — and later on 38.1% by
1.88; the signed mean of 2.43 was the figure quoted as "mean gap", and it
understates the early case by more than half. The summary now carries
`mean_abs_gap_chunks`, `mean_gap_when_earlier` and `mean_gap_when_later`, and
the report names which one it is stating. This matters because the sentence it
appears in is the study's own quantification of Mo et al.'s concern about
consensus stopping, so understating the gap understates the concern.

None of the three changes a number already published for the pilot: the
median agrees at odd `n`, and the other two add fields rather than redefining
the ones that exist. What changes is that the main run cannot report them
under the wrong name.

---

## D43. A head-to-head must be run on the other rule's own scores

`decisions_at` decides where a rule fires by reading `row['scores']`, so the
matched-budget comparison needs the other rule's **per-problem scores**, not
merely its curve. The loop guarded on the curve and fell back to the NLA
readout's rows when a rule had no re-swept ones:

```python
other_dec = decisions_at(restricted_rows.get(other, []) or nla_rows, ...)
```

That applies the baseline's threshold to the readout's own cosines and reports
the result as a McNemar test between the two rules — a comparison of the
readout with itself, labelled as a head-to-head, with a real p-value and no
warning.

**It is reachable per rule.** `baseline_curves_on` omits a rule that has no
usable score on the problems the NLA arm covers while keeping the others, so a
baseline scored corpus-wide but not on that subset — semantic entropy needs
sampled continuations, and the probe needs activations — takes the fallback
while its neighbours do not. It is also taken for every rule when the re-sweep
returns nothing at all, which is the `full_corpus` basis.

**No published number came from it.** All three rules were re-swept on the
pilot (`rules_resswept` has three entries), and re-deriving the whole
comparison from `runs/pilot/` through the corrected code reproduces every
stored field exactly — safe rates, discordant counts, operating points, budget
gaps and the conditional-exact intervals. The one difference is the probe
comparison's p-value, and that is D38 rather than this (see D44).

The comparison is now refused when the rule's own rows are absent, and
recorded in `o3_unmatched` with `per_problem_scores_available: false` — an
absent comparison rather than a tested null, so it carries no p-value and does
not enter the BH denominator.

### Two things fixed alongside it

**The unevaluability diagnostics described a different corpus.** The reason
text and `granularity_limited` were read from `curve_resolution`, which
`score_baselines` writes over the **whole** corpus, while the comparison runs
on the problems the NLA arm covers. The scopes differ: on the pilot the probe
takes 272 distinct score values corpus-wide and 203 on those 21 problems, and
semantic entropy 6 against 5. For convergence — the one rule actually
unmatched on the pilot — they coincide at 7, which is why the stored record is
unchanged and D37b's reading of it still stands. The diagnostics are now
computed from the rows compared, tagged with `diagnostics_basis` and
`n_problems_compared`, and the corpus-wide figures are kept beside them under
`full_corpus_resolution`.

**Each arm's own sample count is now recorded.** The test runs on the
intersection of the two arms' problems, so a problem covered by only one arm
was dropped silently and showed up only as a smaller `n_problems`.
`n_problems_nla_arm`, `n_problems_<rule>_arm` and `n_problems_shared` make the
loss visible.

### Why the comparison was extracted from `main`

It had no test of its own — it was inline in the analysis stage, so exercising
it needed a run directory, and the only thing that ran it was the integration
test. It is now `o3_head_to_head`, a function of its inputs, with the branches
above pinned by unit tests: the refusal, the recorded sample counts, the
discordant accounting, the scope of the diagnostics, and that repeating a
budget does not multiply the evidence. The body is unchanged apart from the
fixes named here, and the pilot reproduction above is the evidence.

---

## D44. The pilot's analysis predated its own corrections, and q = 0.156 was stale

**The pilot's `analysis` and `report` stages have not been re-run since D38 and
D41 landed.** Both ran on 2026-10-11 at 12:56 UTC under commit `31a30e3`;
`0d6d2c8` (D38, zero discordance is a tested null) and `01efa2a` (D41, the RQ3
family test is paired) were committed the next day. So
`runs/pilot/analysis/results.json` and `runs/pilot/report/RESULTS.md` are the
output of the superseded code, and they are what `docs/STATUS.md`, the README
and the paper drafts quote.

**This is not a reproducibility failure.** Re-deriving the whole O3 comparison
from `runs/pilot/` through the current code reproduces every stored field
exactly — safe rates 0.095 and 0.286, four discordant problems of 21 all
favouring semantic entropy, seven distinct operating points, a budget gap of
0.00998, and the conditional-exact interval [−0.190, 0.039]. The artifacts are
sound. What has changed is what the code *makes of* them.

**One stored value differs, and it moves the family.** The probe comparison has
zero discordant problems. The old code returned `nan` there and marked itself
skipped; D38 established that this is the null holding exactly, so the exact
conditional p-value is 1.0 and the test belongs in the denominator. With it,
`m` goes from 5 to 6:

| test | p | q as stored (m = 5) | q under the current code (m = 6) |
|---|---|---|---|
| RQ2: deletion minus paraphrase | 1.8e-119 | 9.0e-119 | 1.1e-118 |
| F10: tail vs gaussian_control | 1.4e-34 | 3.4e-34 | 4.1e-34 |
| F10: tail vs matched_position | 0.0138 | **0.0229** | **0.0275** |
| **O3: readout vs semantic entropy** | 0.125 | **0.1562** | **0.1875** |
| F10: tail vs matched_length | 0.372 | 0.372 | 0.446 |
| O3: readout vs probe | — / 1.0 | *(absent)* | 1.000 |

**No conclusion changes.** Three tests are significant either way, the primary
comparison is non-significant either way, and its direction is unchanged. What
changes is the number: **the primary comparison's q is 0.1875, not 0.156**, and
F10's position control is 0.0275 rather than 0.0229. The RQ3 family test in the
stored file is also still the unpaired Mann-Whitney over 60 rows that D41
replaced, and `summarise_tests`'s newer fields — `bh_family_size`,
`n_planned_but_unevaluable`, `q_values_at_planned_family_size` — are absent, so
the D38 sensitivity cannot be read off the run at all.

**What has to happen.** Re-run the two stages. They are CPU-only, read only
saved artifacts, and take seconds:

```powershell
.venv\Scripts\python.exe scripts\aggregate_results.py --config pilot --force
.venv\Scripts\python.exe scripts\write_run_report.py --config pilot --force
.venv\Scripts\python.exe scripts\check_run_consistency.py --run runs\pilot
```

Then update the q-values in `docs/STATUS.md`, `README.md` and
`paper/`. **Until that is done, every published q from the pilot is the stale
one**, and this record is the authority on what the correct values are. They
are not edited into the documents here, because a number in a document should
be the one an artifact holds, not one computed in a decision record.

The `baselines`, `robustness` and `report` stages are flagged by the same check
but are **inconclusive rather than stale**: each ran from a dirty tree, so it
may have run with the code that later became the commits that touch it. Git
cannot distinguish that from having run with older code. Re-running them costs
nothing and settles it.

### Why this needed a tool

Nothing in the run said so. The manifest records the commit per stage and the
analysis records its p-values, so both halves of the evidence were on disk and
neither was being read. `scripts/check_run_consistency.py` now reads them: it
re-derives the summaries from the artifacts, recomputes the q-values from the
stored p-values, enforces D38's invariant directly, diffs the report's prose
against the analysis, and asks git whether each stage's code has moved since
the stage ran. On the pilot it reports exactly one numeric mismatch — the
probe comparison's p-value — which is the finding above, reached from the
artifacts rather than from the history.

The main run should be checked with it before any of its numbers are quoted.
Its `traces` stage began under `ea82fa845911b70f` on 2026-10-11, so the same
drift is possible there by the time it finishes.

### Resolved 2026-10-15

The pilot's `ast`, `baselines`, `robustness`, `analysis` and `report` stages
were re-run on CPU. Every value in the table above reproduced exactly: the
primary comparison is now stored at **q = 0.1875**, F10's position control at
**0.0275**, the probe comparison at p = 1.0, q = 1.0, and the RQ3 family test
is the paired contrast (unevaluable: no demonstrated edit). `ast.jsonl`,
`boundary_scores.json`, `probe.json` and every robustness scalar were
byte-identical or gained fields only, which settles the three stages the
staleness check had called inconclusive. `check_run_consistency.py --run
runs/pilot` now reports 153 checks and 0 mismatches.

F8 was carried forward from the original measurement rather than re-run. The
re-run also exposed that F8, which every config enables, asked
`snapshot_download` for the bf16 target whenever no earlier F8 result existed
in the run - which on `main` is always - and would have started the 15.3 GB
download during the post-trace `robustness` stage. It now resolves the
snapshot from disk only and reports `blocked` unless `--allow-download` is
given (D45).

---

## D45. F8 never downloads the bf16 target unless told to

F8 is in `enabled_tests` in `base.yaml`, so `main` runs it. It resolved the
bf16 target through `snapshot()`, which is `snapshot_download`. When no earlier
F8 result exists in the run's `robustness.json` there is nothing to carry
forward, so on `main` - always - the post-trace `robustness` stage would have
started a 15.3 GB download onto a cache volume with 12 GB free, against a
standing instruction not to. The documented manual stage order avoided
`run_pipeline.py`'s F8 insertion but not this, because it runs the battery
without `--only`.

F8 now asks `snapshot(..., local_files_only=True)` and reports
`status: blocked` with `download_attempted: false` when the weights are absent.
`--allow-download` restores the old behaviour deliberately. An earlier `ran`
result is still carried forward unchanged, which is what the pilot does. On
`main` F8 will therefore read `blocked`, and the measurement that stands is the
pilot's - cosine 0.987 over 82 vectors - which is a property of the checkpoint
rather than of the corpus. No configuration was changed.

## D46. Two arms that moved by the same non-zero amount: reading A kept

**Decided 2026-10-17: the current behaviour (reading A) stands.** The
researcher approved the five pending items of the 2026-10-17 report, the first
of which was this choice, framed with reading A as the current, test-pinned
behaviour; no change to it was requested, so none was made. It has no effect on
`main`, where every tie is at zero (D47). The record of the alternatives is
kept below. `paired_arm_contrast` returns `beats: False`
(p = 1.0) when every paired difference is zero, arguing that a candidate which
moved the readout by exactly as much as its control on every problem is the
strongest evidence available against direction-specificity. D27 says a
constant outcome is untested, not refuted. The two conflict, and which governs
a tie at a *non-zero* level is a methodological choice this project has not
made.

- **Reading A — determinate refusal (current behaviour).** Both arms changed
  behaviour, identically; the edit is real and not direction-specific. The
  contrast enters the RQ3 family as a tested null at p = 1.0 (D38).
- **Reading B — untested.** "Beat" is one-sided and a tie has no rejection
  region; report `beats: None` and keep the test out of the BH denominator.

Reading A counts against RQ3 on evidence of movement; B never lets a tie
count. Neither is chosen here. The behaviour is unchanged, and a test pins it
(`test_equal_non_zero_movement_is_unchanged_pending_d46`) so any change is
deliberate. The case does not arise on `main`, where every tie is at zero
(D47).

## D47. A readout that is zero in both arms is untested, and so is an empty continuation

Two defects decided RQ3 on `main` without evidence, in opposite directions.

**A zero readout read as a refutation.** At dose 1.0, where the candidate is
compared with its controls, `marker_delta` is 0 for every one of the 60
problems in every arm. Every pair tied at 0 vs 0, so `paired_arm_contrast`
returned `beats: False`, and because the markers move at doses 1.5 and 2.0,
`outcome_varies` was true and the D27 guard did not fire: the summary and
`RESULTS.md` said **"Direction claim supported: False"**. A contrast whose two
arms are both constant at zero now returns `beats: None` with
`readout_constant_zero: true`, and the verdict's `untestable_reason` carries
it. This is D27 applied where the controls are actually read, and the
researcher's standing instruction that a constant zero readout is never
evidence against the hypothesis. It is not D46: neither arm moved.

**One empty continuation read as a broken intervention.** The RQ3 family test
required every candidate row at dose 1.0 to have edited a position.
`math-algebra-test-00008` has a final-chunk tail and its continuation
generated 0 tokens, so no arm could edit it, and that one row declared the
whole arm a "broken independent variable" although 59 of 60 problems were
edited. Rows that generated nothing are now counted
(`candidate_rows_without_continuation`) and excluded from the requirement; a
zero count on a real continuation still voids the test.

**Effect on `main`, verified by diffing `results.json` against the
2026-10-17 archive:** the verdict is `supported: null` with both control
comparisons null; the family test is skipped for the zero readout instead of
for a broken edit; `intervention_demonstrated` is true. The BH family is
unchanged (7 evaluable, 5 significant) and no other number moved. The rows
were not regenerated: `run_interventions.py --summarise-only` rebuilt
`causal/summary.json` from `interventions.jsonl` on CPU, because the verdict
had been frozen at GPU time and copied forward by the analysis.

## D48. The manifest is merge-safe, and each stage records its configuration

**Lost updates.** Every stage process loaded `manifest.json` once and wrote its
whole copy back. On `main`, `ast` and both supervised analyses ran beside
`acts`, as the post-trace checklist allowed, and `acts`' final write erased
all three records; the report listed `ast` as "not run". Writes now take an
OS-level lock (`msvcrt` / `fcntl`, released if the process dies), re-read the
file and overlay only the entries this process changed. The three records
were restored by re-running the stages on CPU: a scratch copy showed all 38
output files byte-identical on re-run, and after the real re-run all 38 still
match the archive's SHA-256 sums. Their records therefore carry the re-run's
commit, not the original's.

**Which configuration produced a stage.** The run-level `config_hash` is
whichever process opened last. On `main`, `nla` ran under `--set
nla.max_problems=120` (`0bb33cf4c468a3f6`); `faithfulness` reopened under
`ea82fa845911b70f`, and the report's header attributed every stage to it.
Each stage now records `config_hash` and `config_overrides`. For stages that
predate this, `stage_config_hash` derives the hash from `config_hash_history`
by start time and labels it `derived`. `main`'s `nla` record was annotated
`recovered` after four checks agreed: recomputing the config with the override
gives `0bb33cf4c468a3f6` and without it `ea82fa845911b70f`; the history makes
`0bb33…` active from 18:28:35Z to 21:36:26Z, containing the stage's start at
18:28:39Z; and `run.log` shows that process opening with `0bb33…` and capping
the arm at 120. Only `nla` ran under `0bb33…`; `faithfulness` and `causal`
ran under `ea82…`, and `nla.max_problems` is read by no other stage.

## D49. An interval names the estimand it bounds

`paired_test` reports Cohen's d beside a bootstrap CI for the *raw* mean
paired difference; `paired_binary_test` reports the discordant odds ratio
beside a conditional-exact CI for the paired *rate difference*. Both are
correct, and both were printed under one effect column: `main`'s O3 row read
as an odds ratio of 0.833 with a CI of [−0.061, 0.049], and RQ2's as d = 0.712
with [0.031, 0.033]. No number changed. Intervals carry an `estimand`, and the
hypothesis-test table prints `ci_estimand` and `ci_estimate` (the estimand's
own point value) beside the bounds. Artifacts written before the label are
labelled from the effect name, which maps to exactly one kind of interval.

## D50. The causal stage's rows and derived summary are checked separately

`check_run_consistency.py` asked git whether any commit had touched
`run_interventions.py` or `src/nlaast/causal/` since the GPU run. Those files
hold both the row-generating code and the verdict code, so the D47 correction
and `--summarise-only` made `main`'s rows look stale (PW-23) when no
row-generating code had changed. The rows are now judged by the source of the
symbols that generate them (compared as ASTs, docstrings and comments
ignored), plus file-level history for generation-only dependencies. The
derived layer is judged by content: `summary.json` records the SHA-256 of the
rows it was built from, the verdict is recomputed from the rows, and the
analysis's RQ3 verdict and family-test basis and the report's verdict line are
held to the summary. A summary rebuilt over unchanged rows passes; changed
rows, a stale summary, and stale analysis or report each fail. **Limit:** the
symbol list is declared by hand; a test fails if a declared symbol
disappears, but a new generating function must be added to it.

## D51. The supervised analyses: held-out and cross-validated evidence

On `main` the held-out estimates and the training-split cross-validation
disagree: regression held-out R² 0.333 [0.19, 0.44] against CV R² −0.358
(label-shuffle p = 0.73); classification held-out AUROC 0.818 [0.73, 0.90]
against CV AUROC 0.747 (p = 0.08). Both stored CV values were reproduced
exactly and diagnosed on CPU (no new model):

- **Regression.** 97 training rows, 5 folds of ~19. One fold scores R² −2.58
  and drags the mean; over 20 fold assignments the mean-of-folds CV R² ranges
  −0.41 to +0.16 (median −0.12) and the pooled out-of-fold R² −0.34 to +0.15.
  Train and eval targets are alike (mean 0.39 vs 0.37, SD 0.15 vs 0.16).
- **Classification.** 6 incorrect answers in 101 training rows (24 in 149
  held out): 1–2 per CV fold. Fold AUROCs run 0.53–1.0.

**Reading, fixed by `SUPERVISED_ANALYSES.md` rather than chosen now:** the
held-out split is the pre-specified performance estimate (§6); the
label-shuffle null is a leakage check on the CV score (§4). The check shows
no leakage and has little power at this size, so it neither corroborates nor
contradicts the held-out estimates. The held-out claims stand on their own
intervals, as one estimate on one split. Threshold metrics and calibration
for the classifier are less secure than its AUROC, because the 6 training
negatives fix the threshold.

**Reporting corrected:** `write_supervised_report.py` called p = 0.0796
"separated" (its cut-off was p < 0.10, against the project's α = 0.05, with a
null maximum of 0.911 above the observed 0.747). It now tests at α = 0.05 and
states what the check is for; the conclusions carry the caveats above.

## D52. Deviations from §6 that this study does not close

Recorded so that none disappears from the record; none was run.

- **F5's sweeps over `T` and the chunker were never implemented.**
  `ast_sweep_temperature` is configured and read by no code, and no chunker
  sweep exists; F5 exercised `k` and unanimity only, and the robustness output
  listed nothing as not applicable. Existing data cannot close it:
  continuations were recorded at one temperature, and a different chunker
  needs forced answers and continuations at new boundaries. Both need GPU
  generation. F4's alternative segmentation (11.0 vs 15.7 chunks per trace)
  is a parser check, not this sweep.
- **RQ2's registered independent-probe control was not implemented.** §5 asks
  for a probe that scores whether a *claim's content* is decodable. The stored
  `independent_probe_score` is the correctness probe's output for the whole
  activation: one value per explanation (1,413 on `main`), none per claim. It
  cannot test claim content, and implementing the registered control needs a
  claim-content target that §5 does not define.
- **F8 is half done.** Q1 (layer-20 activation drift, NF4 vs bf16) was
  measured on the pilot: cosine 0.987 over 82 vectors. §6's bf16 replication
  of the headline comparison was never run; it needs the 15.3 GB bf16 target,
  which is not on disk, and no download was made.
- **The NLA arm is 120 problems, not more.** 240 problems have a tail. Growing
  the arm after seeing the O3 result would be a post-hoc increase in N, and is
  not done.

**Status update 2026-10-17 (evening), from artefacts; the entry above is kept
as written.** Each item verified against code, manifests and outputs:

| item | status | evidence |
|---|---|---|
| F5 temperature sweep (T = 0.6, 1.0) | **deferred by decision** (researcher, 2026-10-17) — implemented, not run | `scripts/run_f5_sweep.py` + 12 tests kept; conditions from the 2026-10-06 config; 2 of 250 problems generated for T = 0.6 as a timing check (`runs/main_f5/T_0p6/evidence.jsonl`, kept); estimated 60–150 GPU-hours for the three conditions (extrapolated from 2 problems, low confidence) |
| F5 chunker sweep (`newline`) | **deferred by decision** (same) — implemented, not run | same script; the 12-boundary scan cap binds on 164 of 250 traces under this chunker |
| F8 bf16 headline replication | **completed — partial by decision (D55)**; the full registered replication is not implementable on this hardware | 24 problems; bf16 target activations only, verbaliser/reconstructor/behavioural arms NF4; window-score shift −0.0014 [−0.0033, 0.0005]; O3 direction vs semantic entropy unchanged; no O3 equivalence shown (D55) |
| RQ2 independent-probe control | **completed — exploratory** (A1, D56–D57) | did not corroborate the reconstruction-dependence reading for either attribute; rule-derived labels, no human validation (A1 §12) |
| Larger NLA arm (N = 240) | **completed — exploratory (D54)** | 240 problems; original 120 reproduced exactly; O3 vs semantic entropy −0.138 [−0.191, −0.071], vs probe −0.029 [−0.059, 0.010] (D54) |

Correction to the entry above: the bf16 *target* is 15.23 GB (four shards);
15.3 GB is the verbaliser's size (PROJECT_PLAN §2).

## D53. Which working documents are version-controlled

`PROJECT_PLAN.md` and `docs/` were untracked on 2026-10-15 (1c26d0b) and are
tracked again from 2026-10-17, at the researcher's approval (D7). They hold
the pre-registration, the decision record and the stage-order checklist, and
a reproducibility record that exists on one disk is not one. Before
re-tracking, `PROJECT_PLAN.md` §5–§6 were confirmed byte-identical to the
2026-10-06 commit (5ebf6dd; 87 lines, same SHA-256), and all files were
scanned for credentials. `SESSION_HANDOFF.md` (session notes) and
`Project_Review_*` (review submissions) stay local. `EXECUTION_REPORT_SUPERVISED.md`
is tracked as the generated supervised report for `main`.

## D54. The larger NLA arm is a post-hoc, exploratory extension (recorded before it ran)

**Recorded 2026-10-17, before any extension output existed.** The researcher
authorised running the larger arm on 2026-10-17 (D5 had been left open, "not
recommended"). This entry fixes what is run and how it is read, so the
extension cannot be shaped by its own result.

**Why it is post-hoc.** No NLA-arm N was pre-registered. `main.yaml` set
`nla.max_problems: 80`; the arm ran at 120 under a `--set` override (stage
config hash `0bb33cf4c468a3f6`, D48). The O3 result at N = 120 has been seen
(vs semantic entropy: 37 discordant, 13 favour the readout, p = 0.099,
q = 0.115). Any larger N is chosen after that, so the extension is
**exploratory**. The N = 120 analysis stays the study's result for O3 and is
not overwritten.

**What runs.**

- N = every problem with a detected tail: 240 (the cap no longer binds, so
  `stratified_cap` selects nothing away). The original 120 are a subset by
  construction - the cap takes a deterministic per-dataset prefix.
- Separate run directory `runs/main_nla240/`. `data`, `traces`, `acts`, `ast`,
  `baselines` and the original `nla/verbalisations.jsonl` are copied from
  `runs/main`, byte-identical (763 files, `SOURCE_SHA256SUMS.txt`).
- `run_autoencoder.py --config main --run-id main_nla240 --set nla.max_problems=240`.
  The stage resumes: the 120 original problems' verbalisations are kept as they
  were generated, and only the new problems' windows are verbalised, under the
  stage's own seed derivation. The reconstruct phase re-scores every
  verbalisation, so the original 120 problems' cosines are re-computed and
  compared with `runs/main/nla/reconstructions.jsonl` as a determinism check.
- `faithfulness` is not run on the extension: O3's NLA score is reconstruction
  cosine from the `nla` stage alone, and RQ2 is not part of this extension.
  `causal` is not run either. Tests that need those stages are absent from the
  extension's analysis and must not be read from it.
- `aggregate_results.py` unchanged, on the new directory.

**How it is read.**

- The quantities reported are the three O3 head-to-heads (vs semantic entropy,
  probe, convergence) at matched budget: McNemar exact over problems, the
  paired rate difference with its conditional-exact interval, the discordant
  counts, and the operating points - each beside its N = 120 value.
- Multiplicity: the BH q-values the analysis computes over the extension's own
  evaluable family are reported, but the family differs from `main`'s (no RQ2,
  no RQ3), so the extension's q-values are not comparable with `main`'s and no
  confirmatory claim is made from them. The study's O3 conclusion remains the
  N = 120 result; the extension says whether the direction and size hold at
  twice the N.
- One extension only. No further increase in N whatever the result.
- Power, stated as optimistic: at the observed discordance (rate 0.308, 35%
  favouring the readout) `size_o3_corpus.py` gives roughly 0.6-0.7 power at
  q < 0.05 near N = 240; a power figure computed from the observed effect
  overstates the true power and is not a reason to expect significance.

**Found while preparing this: the N = 120 arm contains no MATH
counting-and-probability problem.** `stratified_cap` stratifies by `dataset`
(GSM8K / MATH), not by subject, and fills each stratum with a prefix of its
sorted ids; `math-algebra-…` sorts before `math-counting_and_probability-…`,
so the arm's 46 MATH problems are all algebra (74 GSM8K + 46 algebra). Every
`main` result computed on the NLA arm - O3, RQ2, F10 - therefore describes
GSM8K and MATH algebra, not the corpus's third subject. Nothing was wrong with
the cap's stated purpose (proportional by dataset), but the subject coverage
was not reported. The N = 240 extension has no cap, so it includes the
counting-and-probability problems.

**Result, 2026-10-17 23:12 — exploratory (N chosen after seeing N = 120).**
`run_autoencoder.py --config main --run-id main_nla240 --set
nla.max_problems=240` completed (stage hash `fd8ba06d29ea4f56`, 4.2 h GPU);
1,421 windows over 240 problems (148 GSM8K, 69 algebra, 23 counting &
probability); 4,263 reconstructions; new-sample integrity 99.4%. The 120
original problems reproduce exactly: 709 verbalisations byte-identical, 2,127
of 2,127 reconstruction cosines bit-identical. `aggregate_results.py` on the
new directory (same override): exit 0; `check_run_consistency.py`: 132
checks, 0 mismatched, 0 warnings (faithfulness, causal and report absent by
design).

| O3, NLA readout minus … | N = 120 (the study's result) | N = 240 (exploratory) |
|---|---|---|
| semantic entropy | 37 discordant (13 / 24); safe 0.233 vs 0.325; −0.092 [−0.184, 0.016]; p 0.099, q 0.115 | 65 discordant (16 / 49); safe 0.188 vs 0.325; −0.138 [−0.191, −0.071]; p 5.1 × 10⁻⁵ |
| probe | 11 discordant (5 / 6); −0.008 [−0.061, 0.049]; p 1.0 | 19 discordant (6 / 13); −0.029 [−0.059, 0.010]; p 0.167 |
| convergence | no shared budget | no shared budget |

Read as fixed above: the direction against semantic entropy holds and its
interval now excludes zero at twice the N; the comparison with the probe
stays inconclusive. The extension's BH family (6 tests: no RQ2, no RQ3) is
not `main`'s, so its q-values (0.0001, 0.167) are not comparable and no
confirmatory claim is made; the study's O3 conclusion remains the N = 120
result. No further increase in N.

## D55. F8's headline replication is run in part: bf16 target activations only

**Decided by the researcher 2026-10-17** ("run partial, labelled"), before the
download or any bf16 output.

§6 registers "bf16-CPU replication of the headline comparison on a
subsample". The headline comparison is O3: the NLA readout's safe-stopping
rate against semantic entropy and the probe at matched budget. A full bf16
replication needs all three checkpoints in bf16 (target 15.23 GB, verbaliser
15.3 GB, reconstructor 10.9 GB) and bf16 generation for the verbaliser and for
every forced answer and semantic-entropy sample. On a 6 GB GPU with 15.6 GB
of RAM that means disk-offloaded 7B decoding at seconds per token: on the
order of 40-200 h for the verbaliser alone on the subsample, more for the
behavioural arms. That is recorded as not implementable on this hardware.

**What is run** (`scripts/run_f8_bf16_headline.py`):

- Subsample: `robustness.quantisation_subsample` = 24 problems from the NLA
  arm, drawn by the arm's own `stratified_cap` (15 GSM8K, 9 MATH algebra -
  see D54 on subject coverage). Fixed before any bf16 output.
- bf16 target `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28` (the revision the NF4 checkpoint
  was converted from), downloaded to `D:\nla_models\hf_cache` through
  `NLAAST_MODEL_CACHE` because the configured cache volume (E:) has 12 GB
  free. Loaded truncated to layer 20, split GPU/CPU/disk by F8's existing
  placement rule. The same token sequences are replayed (`f8_replay_sequence`)
  and layer 20 is read at the run's own positions.
- Two sibling run directories, `runs/main_f8_bf16` and `runs/main_f8_nf4`,
  identical except for the activations' source. Both run the unchanged `nla`
  stage (NF4 verbaliser/reconstructor) on an identical task list - so batching
  and every sampling seed match - and the unchanged `analysis` stage on the
  same 24 problems.
- Read: Q1 drift on these 24 problems; the paired difference in per-window
  NLA score (bf16 minus NF4) against the sampling-noise floor (NF4 re-run
  minus the run's original NLA output on the same windows); integrity rates;
  and the O3 rows from both arms beside `main`'s N = 120 values.

**What it can and cannot show.** It tests the specific threat D2 named - that
off-distribution (4-bit) target activations drive the NLA readout - on the
quantity O3 actually uses. It cannot test whether 4-bit verbaliser and
reconstructor weights, or the 4-bit target's generations, drive the result.
With 24 problems the O3 comparison has no power to establish a difference; the
replication's question is whether the bf16 arm reproduces the NF4 arm's
scores and decisions, not whether either reaches significance. F8 is reported
as **partially replicated**, never as complete.

**Result, run 2026-10-17 23:13 – 2026-10-18 00:32** (protocol unchanged,
one GPU job at a time). bf16 activations: 3.5 min for 24 problems (placement
3 GiB GPU / 5 GiB host / rest disk, offload cleared). Each NLA arm: 141
windows, ~37 min. Analysis and comparison on CPU. Output
`runs/main_f8_comparison.json`.

- **Q1 on `main`'s subsample:** cosine(NF4, bf16) mean of problem means
  0.990 (range 0.979–0.994; lowest single vector 0.956), norm ratio
  0.983–0.997, 245 vectors, none skipped — consistent with the pilot's 0.987.
- **Integrity:** bf16 arm 98.9% (279/282), NF4 re-run 100%, original 99.6%.
- **NLA score per window** (the quantity O3 uses): bf16 − NF4 re-run
  −0.0014 [−0.0033, 0.0005], r 0.964, mean |diff| 0.0084; the sampling-noise
  floor (NF4 re-run − original NLA output, same windows) +0.0004
  [−0.0008, 0.0016], r 0.985, mean |diff| 0.0047. No mean shift is
  detectable; per-window disagreement is about 1.8× the noise floor
  (descriptive, not a pre-specified test).
- **O3 on the 24 problems** (no power; direction only): vs semantic entropy,
  bf16 −0.042 (5 discordant, 2/3), NF4 re-run −0.125 (5 discordant, 1/4),
  both "readout worse" as at N = 120; vs probe, the two arms matched at
  different budgets (0.81 vs 0.70), 0 vs 6 discordant, so not comparable.

**Reading.** On 24 problems, swapping the target's activations from NF4 to
bf16 does not detectably move the NLA readout's score or reverse the O3
direction against semantic entropy. This narrows, but does not close, the D2
threat: the verbaliser, reconstructor and behavioural arms stayed NF4, and 24
problems cannot show O3 equivalence.

## D56. RQ2's independent-probe control is pursued through amendment A1

**Decided by the researcher 2026-10-17:** the control is to be designed and
run, not left as a recorded deviation, with the claim-content target fixed by
a dated amendment before any evaluation.

§5 fixes the control's inputs (activations, never verbalised text), its split
policy (group-disjoint) and its leakage test, but not what a free-text claim's
"content" is as a probe target. Amendment A1 (`docs/RQ2_PROBE_AMENDMENT.md`,
**PROPOSED**, 2026-10-17) supplies it: content attributes with text-only
ground truth (`answer_stated`, `domain`; `final_word` described but not
scored), a deterministic claim→assertion rule validated against a blind
150-claim annotation (κ ≥ 0.6 per attribute, fixed in advance), probes fitted
on train-split boundary activations and scored only on eval-split claims, a
between-problem label-permutation leakage control, a mismatched-activation
specificity control, and a reading rule fixed before evaluation.

Done so far, none of it an outcome: the implementation
(`scripts/rq2_claim_probe.py`, 30 tests), the coverage counts (`describe`;
1,476 of 3,779 eval claims assert a scored or described attribute), and the
blind annotation sheet (`runs/main_rq2probe/annotation_sheet.csv`). No probe
has been fitted for this control and no claim scored. The evaluation is gated
in code on the amendment's status line and on the annotation agreement
existing.

Two side findings. The stored `independent_probe_score` is a per-activation
correctness score and must never be read as claim-content decodability; it is
used by no test. And 518 of the 1,413 audited explanations come from
train-split problems, whose activations that correctness probe was fitted on.

**Researcher decisions, 2026-10-17, before any evaluation** (A1 still
PROPOSED):

1. *Leakage control, option D.* `answer_stated` uses F6's registered
   within-group shuffle (prevalence exact; relative-position AUC 0.997 →
   0.453); `domain` keeps the across-problem swap (a within-group shuffle is
   the identity for it); `answer_stated` gets a position-only comparator
   (four position features, same model class), and evidence beyond position
   requires the lower bound of the problem-cluster interval for (probe −
   position-model accuracy) to exceed 0. That is not evidence of semantic
   understanding. The earlier across-problem positional resampling kept
   `answer_stated` aligned with position (AUC 0.996, prevalence 0.103) and
   would have read a position effect as leakage.
2. *No human annotation.* The claim→assertion rule is unvalidated; no κ is
   computed; outputs label assertions rule-derived and ground truth
   text-derived. One AI annotation pass (Claude, not independent of the
   rule) is stored separately and is not used for any statistic.
3. *`answer_stated` retained provisionally*, with its positional
   predictability reported (0.970 stated at final boundaries vs 0.003
   elsewhere); no claim beyond position unless the comparator is beaten.

Finding recorded, not changed: the project's own F6 code
(`baselines/probe.permute_labels`, across-groups arm) fits donor label
sequences with `np.resize`, the tiling/truncation pattern A1 removed; it
bears on F6's `main` result for the correctness probe. **Pending as a
separate audit** (researcher, 2026-10-17): F6 is not modified or rerun under
A1.

Confirmed final for A1's next revision (researcher, 2026-10-17): AI annotation
exploratory only, no κ; the four-feature position comparator is primary;
accuracy difference with lower bound > 0 is the primary criterion, Brier
secondary; criteria not to change after evaluation. Failing to beat the
comparator means predictive value beyond those features was not established,
not that the activations lack the information.

## D57. Amendment A1 approved for an exploratory, CPU-only evaluation

**Approved by the researcher 2026-10-17**, after the pre-approval audit (A1
§10) found the `answer_stated = 0` reading systematically wrong, the rule was
revised (option (a), A1 §3), and a fresh read-only audit (A1 §11) found no
blocking issue.

- Scope: the gated `rq2_claim_probe.py --phase evaluate` on `main`, CPU only.
  Code approved at SHA-256 `80a7bc55…53b`; criteria, features, shuffles and
  thresholds as in A1 §5–§7, not to be changed after results are seen.
- **Exploratory.** Claim assertions are rule-derived and **not
  independently human-validated** (Decision 2); `domain` is read as a
  lexical subject cue; `answer_stated` readings are close to one-class
  (533 / 28) and largely positional. No claim of semantic understanding or of
  validated ground truth is made from the results.
- Unchanged: the registered RQ2 primary analysis (deletion vs paraphrase);
  F6 (its `np.resize` audit remains separate and pending).

**Run 2026-10-17 22:08** (`CUDA_VISIBLE_DEVICES="" python scripts/rq2_claim_probe.py
--config main --phase evaluate`, exit 0; `runs/main_rq2probe/evaluation.json`,
SHA-256 `39e4b5e3…855`). CPU-only because the code path never imports torch
(the empty `CUDA_VISIBLE_DEVICES` does not hide the GPU on this machine; see
A1 §12). Read by A1 §7 as fixed; exploratory; rule-derived labels, not
human-validated:

- `answer_stated` (561 claims, 76 problems; eval truth is exactly "final
  boundary", AUC 1.0): the probe decodes it (probe − majority +0.095
  [0.087, 0.104]); no leakage (within-problem shuffled − majority −0.028
  [−0.042, −0.015]); the position model is perfect (1.000) and the probe does
  **not** beat it (−0.0013 [−0.0040, 0.000]). Reading: *decodable, not
  distinguishable from boundary position* — no predictive value beyond the
  four position features was established, which is not evidence the
  activations lack it. M3 −0.009 [−0.023, 0.000] (taken at the same relative
  position, so uninformative here, §10); M4 −0.103 [−0.174, −0.027]. Not
  corroborating.
- `domain`, read as a subject cue, two-class on eval (GSM8K 43 / algebra 33
  problems; 75 counting & probability claims excluded, probe false-positive
  rate on them 0.067 [0.000, 0.188]): decodes (+0.406 [0.292, 0.515]); no
  leakage (+0.047 [−0.030, 0.124]); M3 +0.368 [0.224, 0.522]; M4 −0.034
  [−0.089, 0.028]. Reading: the cued subject is specifically decodable from
  the claim's own activation, but unrelated to which claims reconstruction
  depends on — not support for RQ2's reconstruction-dependence reading.
- Coverage: 1,233 of 3,779 eval claims (32.6%) are scored in M2–M4 (561
  `answer_stated` + 695 `domain` − 23 scored for both). *Correction:* a
  session report gave "1,584 of 3,779"; that counts claims asserting any
  attribute including the unscored `final_word`. Full table: A1 §12.

## D58. F6's length-correct diagnostic: protocol approved, implementation not

**Recorded 2026-10-18, before any implementation or fit.** The researcher
approved the protocol `docs/F6_LENGTH_CORRECT_DIAGNOSTIC.md`, SHA-256
`097cbf8a60b381b17b7973c50684153c1af85eea55b838b43b264504baf66eb1`. It is
committed byte-identical to the approved text, so its header still reads
"PROPOSED, not approved"; this entry is the approval record. It is the
follow-up refit that `docs/F6_RESIZE_AUDIT.md` left to the researcher.

- **Approved:** the protocol text only.
- **Not approved:** writing `scripts/diag_f6_length_correct.py`, fitting any
  probe, or creating `runs/main_f6_diag/`. Each needs separate authorisation;
  the protocol's DECISIONS entry with the code SHA-256 is made at that point.
- **Decided within the protocol:** the positional oracle's donor pool is every
  problem except the receiver and its donor (no fold restriction); SD / sqrt(B)
  is kept as a supplementary Monte Carlo standard error.
- **Unchanged:** F6's registered result (real AUC 0.9460944231, across-groups
  shuffled 0.6142873504, within-group 0.5385), its threshold,
  `leak_suspected = false`, its interpretation and `permute_labels`;
  `runs/main/robustness/robustness.json` SHA-256 `45701a51…d5ed`. The
  diagnostic is exploratory and descriptive and gives no leakage verdict.

## D59. F6's length-correct diagnostic: implementation recorded, not run

**Recorded 2026-10-18.** The researcher approved the implementation of the
D58 protocol for commit, after a review against the protocol and four
corrections. `scripts/diag_f6_length_correct.py`, SHA-256 of its LF-normalised
text (`\r\n` folded to `\n`, the committed blob):
`45c7807b3b7c99b8c4d6e137e2693e723f4a148ede9754e5ab37734da7d2fdf9`. Its
preflight refuses to run unless this hash appears here and the script, the
protocol and this file are committed and unmodified. Tests:
`tests/test_diag_f6_length_correct.py`.

**Corrections approved before commit.** None changes a seed, donor
assignment, mapping, metric, estimand or sampling plan.

1. The derangement assumption is enforced: the data checks require F6's
   draw-0 permutation to have no fixed point, and the oracle raises an error
   for a receiver that is its own donor (D_r would otherwise drop it twice).
   No receiver is excluded or given another donor.
2. The preflight probe-settings check includes `max_iter = 2000`.
3. `summary.json` carries the caveats: the donor-draw 2.5–97.5 percentiles
   describe single draws, not an interval for the mean; the oracle is a
   position-derived reference benchmark, not a guaranteed upper bound, formal
   null distribution or definitive leakage test; probe AUC above it is a
   discrepancy to investigate, not proof of leakage; similar or lower probe
   AUC neither shows that position explains the predictability nor rules out
   leakage.
4. Every write after the output directory is created is inside the failure
   handler, so an early failure leaves `status: failed` and no `summary.json`.
   The directory is still created only after the seed-12 gate passes.

**Implementation readings (the protocol does not spell these out).**

1. Draw 0 of the subset and S2 uses the protocol's streams
   `[11, 0xF6, 1, 0]` and `[11, 0xF6, 2, 0]`: F6's draw-0 permutation is
   defined on the full sample only and is not a derangement of the subset.
2. The full sample and the subset each have their own bootstrap generator,
   both seeded with `analysis.seed` (2024). Arms within a sample share one
   set of resamples, so every reported difference, all of which are within
   one sample, stays paired.
3. The explicit-feature comparator P is fitted on the full-sample S0 and S1
   labellings only (Question B's scope): no subset or real-label comparator
   (confirmed by the researcher).
4. The predeclared wording table is applied to differences only (Delta_A, E,
   subset differences), not to AUC levels or the share-of-oracle ratio; both
   uncertainty summaries are still reported for every statistic.
5. Stage-0 label statistics are computed before any fit and written only
   after the seed-12 gate passes, since a gate failure writes nothing.
6. Problem IDs are in sorted order throughout (F6's convention, needed to
   reproduce its draw 0); S2's strata are deranged in ascending length, one
   generator per draw.

**Not done.** Tests were synthetic and CPU-only (a fake fit function; no
research data read). No research-data fit, preflight, seed-12 gate or
diagnostic run has been authorised or executed, and `runs/main_f6_diag/` does
not exist. Preflight, the gate and the full run each need separate
authorisation.

## D60. F6's length-correct diagnostic: run, result frozen, recorded as exploratory

**Recorded 2026-10-18.** The researcher authorised preflight, the seed-12
gate and the full run together; all three ran at `27c131a` under the D58
protocol and the D59 implementation, unmodified. Preflight passed every
check; the gate reproduced F6 exactly (718 labels changed, shuffled AUC
0.6142873504351077, real AUC 0.9460944231357181); the run completed all
B = 100 draws (no fallback: projected 4,920 s against 6 h), with no failure.
An independent validation (27 checks, 0 failed) regenerated every labelling
from the seed definitions and recomputed every AUC and summary from the stored
outputs. Results and their reading: `docs/F6_LENGTH_CORRECT_RESULTS.md`.

- **Δ_A = +0.240** (single draws 0.196–0.278, all positive; bootstrap
  0.233–0.247): with the same donors, the length-correct rule φ gives a
  shuffled AUC of 0.841 against `np.resize`'s 0.602. The registered 0.614
  depends on the length-fitting rule as well as on donor reassignment.
- **S1 shifts prevalence** from 0.200 to 0.228 (S0: 0.156), more than the
  protocol's "approximately preserved"; Δ_A carries that shift as well as the
  restored label–position structure (0.920 original, 0.919 S1, 0.683 S0).
- **Oracle:** the probe is below the position-derived benchmark in every draw
  (E = −0.092 for S0, −0.069 for S1). This is not evidence that position
  explains the probe's predictability and does not rule out leakage; the
  oracle is not an upper bound, formal null or leakage test (the linear
  position model exceeds it under S1, 0.916 against 0.910).
- **Subset (246 problems, the four longest traces excluded):** S2 − S1 =
  +0.031; read for the subset only.

**Frozen.** The outputs stay in `runs/main_f6_diag/` (not version-controlled,
marked read-only), with the validator, its log and the outputs' SHA-256 list
in `runs/main_f6_diag_validation/`. They are not in any existing backup.

**Unchanged.** F6's registered values (0.9460944231 / 0.6142873504 / 0.5385),
its threshold, `leak_suspected = false` and its interpretation;
`runs/main/robustness/robustness.json` SHA-256 `45701a51…d5ed` before, during
and after the run. No leakage verdict is made from the diagnostic.

**Next step, decided by the researcher:** no further experiment now. The
diagnostic has identified the methodological issue — the control depends on
how donor labels are fitted to receiver length. A stronger leakage claim would
need a new preregistered study fixing, before any result is seen, a control
with a defined null, a length-matching strategy, refit-inclusive uncertainty
and a decision threshold.

