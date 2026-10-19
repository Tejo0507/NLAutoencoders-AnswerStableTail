# 3. Method

*Final. These are the pre-registered definitions, fixed before the main run;
they do not depend on its outcome and were not changed after it. Source: `PROJECT_PLAN.md` §5, with the
decision record in `docs/DECISIONS.md`.*

## 3.1 Chunking and intermediate answers

A trace is split at sentence boundaries into chunks `c_1..c_n`. For each
prefix `P_i = c_1..c_i` an intermediate answer `a_i` is parsed by the *same*
extractor used for the final answer; `a_i` may be null. Answer equivalence is
symbolic throughout, by a verifier rather than string match, so `\frac{1}{2}`
and `0.5` are one answer.

## 3.2 The answer-convergence baseline

`b_conv(w)` is the smallest `i` such that `a_i..a_{i+w-1}` are all non-null
and pairwise equivalent — the rule of [5], with `w` configurable.

For the matched-budget comparison this has to expose a swept score. We score
each boundary by its **agreement run length**: how many consecutive boundaries
ending here carry one answer, and 0 where the answer did not parse. The
identity

```
first i with run_length[i] >= w   ==   b_conv(parsed, w)
```

holds because "the last `w` answers agree" and "the agreement run here is at
least `w` long" are the same statement. So every operating point on the swept
curve *is* the baseline at one window width, and `w` — which the
pre-registration leaves free — sits on the sweep axis rather than being fixed
and a derived indicator swept in its place. The identity is asserted by test
over nine answer patterns and every `w` up to trace length, and was checked
against the canonical implementation on 83 traces for `w = 1..8` with zero
mismatches (`docs/DECISIONS.md` D36).

## 3.3 The Answer-Stable Tail

A boundary `i` is the AST start iff all three hold:

1. **Truncation equivalence.** Forcing an answer from prefix `P_i` yields an
   answer equivalent to the full trace's final answer.
2. **Resampled-continuation stability.** `K` independent continuations sampled
   from `P_i` at temperature `T` all yield answers equivalent to the full
   trace's final answer. This is the control [22] shows the agreement rule
   lacks, and it is required rather than optional.
3. **Persistence.** Conditions 1–2 hold for every boundary `j >= i`.

The AST is `c_{i*}..c_n` for the smallest such `i*`, and
`tail_fraction = (n - i* + 1)/n`.

**Edge cases are recorded, not dropped.** Each of `no_stable_point`,
`stable_at_zero`, `unparseable`, `incorrect_final` and `truncated_generation`
carries a status code into the corpus and is counted in the report.
`incorrect_final` is kept deliberately: the construct does not presuppose
correctness, and excluding wrong answers would bias the comparison toward
problems the model happens to solve.

**What the definition measures.** On every pilot problem with a tail, the tail
began *before* the trace stated its answer. The criteria therefore measure
when the answer became **determinate**, not when it was announced. That is
correct for the safe-stopping outcome in §3.4 — a rule that stops where the
answer is already fixed loses nothing — but it is not interchangeable with
"post-answer redundancy", and we do not use the two as synonyms
(`docs/DECISIONS.md`).

## 3.4 Matched controls for the tail

For every detected AST of length `L` starting at `i*`:

- **matched-position**: a window of length `L` ending at `i*-1` — before
  stabilisation, same trace;
- **matched-length**: a length-`L` window at a uniformly random valid start in
  the same trace.

The pair separates "this region is special" from "this region is late" and
from "windows of this size look like this".

## 3.5 Safe stopping — the primary outcome

A stopping rule fires at boundary `s`. It is **safe** iff the answer forced
from `P_s` is equivalent to the full-trace answer *and* the full-trace answer
is correct, or both are incorrect — that is, stopping did not change the
verified outcome. Reported jointly with
`tokens_saved = 1 - |P_s| / |full trace|`.

Defining safety on verified outcome rather than on answer identity matters in
both directions: a rule that stops early on a wrong answer the model would
later have fixed is *not* safe, and a lucky early stop on a problem the model
would later have ruined is *not* unsafe.

Rules are compared at **matched token budget**, never at their own operating
points (§5.1).

## 3.6 Reconstruction fidelity

Exactly the released convention: both vectors L2-normalised to
`mse_scale = sqrt(3584) = 59.8665`, `MSE = 2(1 - cos)`, and FVE against the
empirical variance baseline of our own activation sample. Reported as
**reconstruction**, never as faithfulness.

## 3.7 Claim-level audit of the verbalisation

A verbalisation is split into atomic claims. For each claim `c` we apply
`delete(c)`, `resample(c)` — regenerate that claim from the autoencoder at a
different seed — and `paraphrase(c)`, following [16].

A claim is **reconstruction-dependent** iff deleting it degrades
reconstruction cosine by more than the **95th percentile of the
paraphrase-null distribution**. The paraphrase arm *is* the null, which is
what makes the threshold non-arbitrary: paraphrasing changes surface form
without removing content, so it calibrates how much degradation mere
rewording causes.

Two controls:

- **Independent-probe control.** A probe trained on held-out activations,
  never on the verbalised text, scores whether the claim's content is
  decodable from the activation. Splits are group-disjoint and leakage is
  tested explicitly by the falsification battery, not assumed.
- **Verbaliser-only control** [12]. The autoencoder is run with the activation
  replaced by a Gaussian vector at the same L2 norm. Claims reproduced under
  this control are attributable to verbaliser priors, not to the target
  activation. This is the single most important control here and is run on
  **every** sampled verbalisation, not a subset.

**Stated limitation of the threshold.** The paraphrase null controls for
surface change but not for how much text a deletion removes, which is about a
quarter of an explanation. Both headline numbers are therefore reported with a
sensitivity analysis over the choice (§8).

## 3.8 Causal interventions

Interventions at the layer-20 residual stream, five arms:

| arm | what it does |
|---|---|
| `truncate` | stop generation at `i*` and force an answer |
| `filler` | replace the tail text with neutral filler of matched token length |
| `ablate_dir` | project out a candidate tail direction, swept over a dose coefficient |
| `random_dir` | project out a random direction of matched norm — the control of [19] |
| `matched_position` | apply `ablate_dir` at the pre-stabilisation window instead |

The candidate direction is the **training-split** difference-in-means between
tail and matched-position windows, so the direction is never fitted on the
problems it is evaluated on.

Following [19], the claim *a tail direction exists* requires `ablate_dir` to
beat **both** `random_dir` **and** `matched_position`, conjunctively, with a
dose-response curve over the projection coefficient, while leaving final
answers intact. §5.4 defines "beat" numerically and states what the gate
returns when a condition cannot be evaluated at all.

**The registered construct for the dependent variable** is *verification-like
behaviour*: the question as posed asks whether ablation "selectively changes
verification-like behaviour without changing final answers"
(`Project_Review_II.md` line 122). The pre-registration fixes that construct,
the five arms, both comparisons, the conjunction and the dose-response
requirement — but **no operationalisation of the construct**. §5.4 states the
measure used, why it is a documented choice rather than a registered one, and
why a readout measuring a different construct may not be substituted for it
(`docs/DECISIONS.md` D40, D41).

## 3.9 Span construction for the direction arms

The candidate's edit span begins at the prompt boundary, so it covers the
*generated* continuation — the regenerated tail — and nothing before it. The
matched-position span *ends* at that boundary, so it covers the
pre-stabilisation window, which lies inside the prompt because the prefix up
to the tail start is fed as prompt. The two meet without overlapping, and
disjointness is asserted at run time rather than assumed.

That the control edits prompt context while the candidate edits generated
tokens is the registered design, not an artefact: §3.4 defines
matched-position as a window ending at `i*-1`, which is necessarily already
written. It does mean the two arms differ in kind as well as in location —
the candidate's edit compounds into what gets generated next, the control's
acts only through attention — and §8 reports that as a limit on what the
location contrast can establish.
