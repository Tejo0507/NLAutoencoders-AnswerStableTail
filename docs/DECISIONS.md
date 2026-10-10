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
