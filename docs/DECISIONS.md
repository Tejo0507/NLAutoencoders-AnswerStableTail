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
