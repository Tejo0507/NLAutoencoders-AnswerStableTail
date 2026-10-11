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

---

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
