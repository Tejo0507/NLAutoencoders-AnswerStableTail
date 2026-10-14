# Natural Language Autoencoders for Redundancy Diagnostics

**Project Review II: Method, Results and Discussion**

---

**Student name:** _______________________

**Registration number:** _______________________

**Programme / Department:** _______________________

**Supervisor:** _______________________

**Date of submission:** _______________________

---

## Table of Contents

- Abstract  *(not written)*
- 1. Introduction
- 2. Related work
- 3. Method
- 4. Experimental setup
- 5. Statistical analysis plan
- 6. Results  *(not written)*
- 7. Discussion  *(not written)*
- 8. Limitations and threats to validity

---

# Abstract

**Not written.** Waiting on the Results: an abstract states the outcome, and the outcome is what the main run has not produced.

This placeholder is deliberate. The section is part of the planned report and its absence is a state of the work, so it is recorded here rather than left out of the document: a reader of this draft should be able to see what is missing without consulting the repository. Nothing in it is a finding, and no number in it is provisional, because there is no number in it.

---

# 1. Introduction

Chain-of-thought prompting made it standard for a language model to produce a
long intermediate derivation before its answer [1], and reasoning-trained
models extend those derivations further still [2]. A practical consequence is
that much of a trace may not be doing work. If a model settles on its answer
early and then continues writing, the remaining tokens are paid for and
discarded, and the question of *when* a trace has effectively finished becomes
an operational one.

The cheapest way to ask that question is behavioural. Liu and Wang segment a
trace at sentence boundaries, parse an intermediate answer at each boundary,
and stop once consecutive intermediate answers agree, reporting substantial
token savings at little accuracy cost [5]. But Mo et al. stress-tested that
family of rules and found that agreement is necessary without being
sufficient: at a rule saving roughly a third of tokens, about one stop in nine
fired on an answer the trajectory later abandoned, and widening the agreement
window trades the problem away against the saving rather than solving it [22].
**[VERIFY]** — both attributions follow `Project_Review_II.md`'s reading.

So a stable answer is not the same as a finished computation, and that gap is
where internal state becomes interesting. Hidden-state probes show that
correctness information is linearly available in activations [6]. Availability,
however, is not use: Zhang and Nanda's account of activation patching makes
the point that what a probe can decode and what the model's computation
depends on are different claims, separated only by intervention [19].
Meanwhile semantic entropy, the reference uncertainty signal for free-form
generation [21], measures uncertainty about the *outcome* rather than the
character of the region being generated.

Natural-language autoencoders offer a third kind of signal. Rather than
reducing an activation to a scalar, they decode it into text, which is a
richer interface onto internal state than a probe score. The difficulty is
that verbalised output is an unreliable surface: Li et al. reproduced
prominent verbalisation methods, ablated access to target-model internals, and
found that benchmark performance is attainable without that access, with
output often reflecting the verbaliser's own parametric knowledge rather than
the target model's state [12]. Any claim resting on a verbalisation therefore
needs a verbaliser-only control, not just a plausibility check.

**This paper asks whether a verbalised readout of internal state tells us
anything about the end of a reasoning trace that cheaper signals do not.**
Answering it honestly requires three things that are easy to conflate and that
we keep apart by construction:

1. **A behavioural definition of the phenomenon that does not mention
   activations.** We define an *Answer-Stable Tail* (AST) from truncation
   equivalence, resampled-continuation stability and persistence (§3). The
   resampling criterion is the control the agreement rules of [5] lack, and we
   require it rather than offering it. Because the definition is
   activation-free, it can be — and is — computed before any autoencoder code
   runs.
2. **A comparison at matched compute.** A stopping rule that fires later is
   trivially safer, so comparing rules at their own operating points compares
   token budgets. Every rule here is reduced to a per-boundary score, its
   threshold swept, and the rules are read off their safety/saving curves at
   the *same* mean token saving (§5). The comparators are answer-convergence
   stopping [5], a hidden-state correctness probe [6], and semantic entropy
   adapted to symbolic answers [21].
3. **Controls strong enough that a negative result is informative.** The
   verbalisation is audited claim by claim against a paraphrase null and a
   verbaliser-only control [12], [16]; the causal arm follows Zhang and
   Nanda's recommendations with a matched-random direction, a matched-position
   window and a dose-response sweep [19]; and a nine-test falsification
   battery targets the alternative explanations directly.

We state in advance that the cheaper signals matching or beating the
verbalised readout is a scientifically useful outcome, and we report it with
the same weight a positive result would receive. That commitment is not
rhetorical: the analysis plan in §5 fixes the test, the unit of analysis and
the multiplicity correction before the comparison is made, and §8 records
every case where a planned comparison could not be made at all rather than
dropping it.

## Contributions

Stated as what the paper does. **[VERIFY]** — none of these is asserted to be
novel; establishing priority would need a literature search that this draft
has not performed.

- An activation-free, three-criterion definition of an answer-stable tail,
  including the resampled-continuation control, with every edge case carried
  into the corpus as a status code rather than dropped (§3).
- A matched-budget, problem-level comparison of a verbalised readout against
  three cheaper signals, with the unit of analysis, the test and the
  multiplicity correction fixed in advance (§5).
- A claim-level faithfulness audit of the verbalisations in which the
  threshold is set by a paraphrase null rather than chosen, and the
  verbaliser-only control is run on every sampled verbalisation rather than a
  subset (§3).
- A causal battery for the candidate tail direction that reports *why* it
  could not establish a claim when it cannot, distinguishing a broken
  independent variable and a measurement with no range from a negative result
  (§5, §8).
- A released corpus, harness and per-run provenance record, so that each
  reported quantity is traceable to the artifact that produced it (§4).

---

# 2. Related work

Organised by the four questions this paper has to borrow an answer to: what
the end of a reasoning trace looks like behaviourally, what internal state
makes available, what a verbalisation of that state can be trusted to mean,
and what an intervention has to satisfy before it counts as causal evidence.

## 2.1 Redundancy and early exit in reasoning traces

Chain-of-thought established the long intermediate derivation as the default
format [1], and reasoning-trained models lengthen it [2]. Work on
*overthinking* treats that length as a cost to be measured: Caldarella et al.
evaluate harmful overthinking [7], Zhai et al. quantify redundancy in
reasoning directly [8], and Long et al. describe a self-verification dilemma
in which checking behaviour is overused [9]. **[VERIFY]** — these three
attributions are the review's summaries, not re-derived.

The line this paper engages most closely converts the regularity into a
policy. Liu and Wang parse an intermediate answer at each chunk boundary and
stop when consecutive answers agree, across several open-weight models and
benchmarks, and also train a supervised stopping policy on internal
activations [5]. Mo et al. are the necessary counterweight: self-consensus is
not a safe early-exit signal, because a stable answer can still be abandoned
later, and the premature-stop rate falls only as the saving does [22]. Koh et
al. add that even an explicit end-of-thinking marker does not reliably halt
generation [24].

**Where this paper sits.** We take [5]'s rule as a baseline rather than as a
definition, and we take [22]'s objection as a design requirement: the AST
definition in §3 requires resampled continuations to agree, which is the
control that distinguishes "the answer has stopped changing" from "the
computation has finished". We also put [5]'s window parameter `w` on the sweep
axis rather than fixing it, so each operating point on its curve *is* the Liu
and Wang rule at one window width.

## 2.2 What internal state makes available

Zhang et al. probe hidden states and find that reasoning models carry
decodable self-verification information — correctness is linearly available
[6]. Farquhar et al. establish semantic entropy, clustering sampled answers by
meaning and taking entropy over clusters, as the reference uncertainty signal
for free-form generation [21].

Both are comparators here rather than targets, and both are included because
they are cheap and strong. Two caveats shape how we use them.

First, **availability is not use.** A probe's success shows information is
present, not that the computation depends on it; separating the two requires
intervention, which is the subject of §2.4 [19].

Second, **a probe can be right for an uninteresting reason.** Our own pilot
measured the probe's AUC at 0.996 against 0.962 for an activation-free
predictor built from token position alone — a margin of 0.034
(`runs/pilot/baselines/`). A readout that merely ties such a probe has not
been shown to read more than position. We therefore report the positional
floor alongside the probe rather than the probe alone, and §8 records this as
a limitation on what the primary comparison can mean.

**Adaptation, declared.** Semantic entropy as published clusters by
bidirectional NLI entailment [21]. For mathematical answers, semantic
equivalence *is* symbolic equivalence, so our implementation clusters with a
symbolic verifier instead. This is an adaptation and is reported as one, not
as a reproduction; §4 states the substitution and §8 lists it as a threat.

## 2.3 Verbalising activations, and whether to believe it

Natural-language autoencoders decode an activation into text. The critical
result for any downstream use is Li et al.'s: reproducing prominent
verbalisation methods and ablating access to target-model internals shows
benchmark performance is attainable *without* that access, and output often
reflects verbaliser priors rather than the target model's state [12]. Lek et
al. analyse the same tension from the training side, documenting that
explanations become more useful for predicting behaviour while introducing
more unsupported detail, and propose modelling the distribution of activations
compatible with an explanation rather than optimising single-point
reconstruction [13]. Zhao et al. attack hallucination in verbalisation with a
two-stage preference-optimisation framework [14], and Dingeto proposes
decodability supervision for verifiable explanations [11]. **[VERIFY]**.

Separately, the chain-of-thought faithfulness literature supplies the audit
template. Turpin et al. show models do not always say what they think [15];
Lanham et al. establish the deletion/paraphrase/resampling perturbation
pattern [16]; Chen et al. extend the reasoning-faithfulness analysis [17]; and
Huang et al. caution that intrinsic self-correction can degrade accuracy [18].
Fraser-Taliente et al. and Templeton et al. bear on interpretable feature
structure in the residual stream [10], [26].

**Where this paper sits.** We adopt [16]'s perturbation template at the claim
level and score each perturbation by its effect on reconstruction. Two things
follow from [12] and are load-bearing rather than decorative. The deletion
threshold is set by the *paraphrase null's* 95th percentile, so it is not a
number we chose. And the verbaliser-only control — the autoencoder run on a
Gaussian vector at matched norm — is run on every sampled verbalisation, not a
subsample, and is reported as a headline number. We call the resulting
quantity *reconstruction*, never *faithfulness*, because it measures whether
the text supports reconstructing the activation and not whether it describes
the computation.

## 2.4 Standards for a causal claim about internal state

Meng et al. set the pattern for localising associations by causal tracing
[25], and Goldowsky-Dill et al. introduce path patching to localise behaviour
to residual-stream paths [20]. The governing methodological result is Zhang
and Nanda's: choices of metric, corruption, restoration site and
normalisation materially change what an activation-patching experiment
appears to show, and they recommend matched-random-direction controls,
matched-position controls and dose-response curves before an intervention is
read as mechanistic evidence [19].

**Where this paper sits.** §3 implements that recommendation as a conjunctive
gate rather than a narrative: the candidate direction must beat *both* a
matched-random direction and the same direction applied at a matched
pre-stabilisation window, with a dose-response curve over the projection
coefficient, while leaving final answers intact. §5 defines what "beat" means
numerically — a paired test over problems, not a comparison of arm means — and
§8 reports the one place where our two controls are not matched on the same
quantity, along with the direction in which that asymmetry biases
interpretation.

One community analysis of NLA thought anchors informed early framing [23] and
is cited as the non-peer-reviewed source it is.

---

# 3. Method

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

---

# 4. Experimental setup

## 4.1 Model and site

| | |
|---|---|
| target | `Qwen/Qwen2.5-7B-Instruct` |
| precision | NF4 (4-bit) |
| `d_model` | 3584 |
| primary layer | **20** |
| sweep layers | 14 and 24 (falsification test F7) |
| max position embeddings | 8192 |
| autoencoder | `kitft/nla-qwen2.5-7b-L20-av` (verbaliser), `kitft/nla-qwen2.5-7b-L20-ar` (reconstructor), both NF4 |

The autoencoder is released for layer 20 only, so the layer sweep applies to
the probe and semantic-entropy arms; the verbalised readout is layer-20 only
by construction, and §8 records that this makes the layer comparison partial
rather than complete.

**Quantisation is a threat to validity, and it is measured rather than
assumed.** 4-bit layer-20 activations were compared against bf16 on a
subsample: cosine **0.987**, integrity gate passing on essentially every
sample (`runs/pilot/robustness/`). That is far from any threshold that would
change a conclusion. The measurement is fixed at the pilot's 8-problem
subsample and cannot be repeated without re-downloading 15.3 GB of bf16
weights, which §8 records.

**Hardware.** A single NVIDIA GeForce RTX 4050 Laptop GPU, 6.14 GiB VRAM,
with the target model occupying about 5.5 GiB. **[VERIFY]** — read from
`nvidia-smi`, not from a spec sheet. Every stage is single-stream and
batch-limited by that budget, which is the binding constraint on the study's
scale.

## 4.2 Data

| | |
|---|---|
| benchmarks | GSM8K [3], MATH [4] |
| MATH subjects | `algebra`, `counting_and_probability` |
| split | `test` |
| configured N | **150** GSM8K + **100** MATH = **250** |
| train / eval | 40% / 60% by problem, seed `20261005` |

Both benchmarks admit automatic verification of a parsed final answer, which
is what the behavioural outcomes require.

**The problem order is interleaved by dataset.** This is not cosmetic. The
loader previously concatenated the datasets, so every GSM8K problem preceded
every MATH problem and a run halted early would have been GSM8K-only — with
every pooled quantity biased toward the easier benchmark and nothing in the
output saying so. Interleaving makes any prefix of the run proportional, which
is what licenses reporting an achieved N below the configured one
(`docs/DECISIONS.md` D33). Arms that run on a subset are capped
**proportionally by dataset** for the same reason.

The train/eval split matters for two things: the candidate tail direction is
fitted on the train split only, and the probe's folds are group-disjoint by
problem.

## 4.3 Generation

| parameter | value |
|---|---|
| `max_new_tokens` | **768** |
| sampling | `temperature` 0.7, `top_p` 0.95 |
| canonical trace | greedy |
| batch size | 4 |
| `force_answer_max_new_tokens` | **48** |
| forcing suffix | `"\n\nTherefore, the answer is"` |

Two of these were raised from pilot values on measured evidence, and both
changes are declared because they move the tail numbers.

**The token cap went from 512 to 768.** At 512, three of ten pilot MATH traces
hit the cap — and those three were exactly the three with no detectable tail,
because a truncated trace has no settled answer for the criteria to be stable
about. The cap was appearing as a result and biasing the corpus toward
problems the model could finish in budget.

At 768, measured over the main run's first **128** traces:

| | n | truncated | rate | 95% exact CI | accuracy |
|---|---|---|---|---|---|
| GSM8K | 77 | 0 | 0.000 | [0.000, 0.047] | 0.909 |
| MATH | 51 | 4 | **0.078** | **[0.022, 0.189]** | 0.863 |

The interval's upper bound excludes the pilot's 3/10 = 0.300 at a 512-token
cap, so the change did what it was made for. We report the trajectory rather
than the best snapshot: at an earlier 53-problem checkpoint the MATH rate was
1/21 = 0.048 with accuracy 0.905, so the point estimate has **risen** and
accuracy fallen as harder MATH levels entered the corpus. The conclusion that
768 is no longer the binding constraint rests on the interval, not on the
point estimate, and the achieved rate is reported as a limit either way
(`docs/DECISIONS.md` D32).

**The forcing budget went from 24 to 48 tokens.** At 24, the forced completion
was cut off mid-derivation on 40.7% of evaluated boundaries, and on 24.1% the
cut was *blocking*: criterion 1 failed while every resampled continuation from
the same prefix reached the final answer. Counting those boundaries as
satisfying criterion 1 moved the mean tail fraction from 0.356 to 0.486, so
the pilot's measured tail is a **floor**, not an estimate. The relaxed budget
costs roughly 2% more generation and removes a 24% false-negative rate on the
stage's own criterion.

Both caps change the tail numbers, so **the main run's tail statistics are not
directly comparable with the pilot's**, deliberately.

## 4.4 Signals and arms

| stage | parameters |
|---|---|
| AST detection | `K = 5` continuations, `T = 0.8`, `max_new_tokens` 200, at most 12 boundaries evaluated per trace |
| convergence baseline | window on the sweep axis (§3.2); the config's `w = 2` is used only where a single operating point is needed |
| semantic entropy | 8 samples, `T = 1.0`, `top_p` 0.95, clustered by **symbolic equivalence** |
| probe | logistic, `C = 1.0`, standardised, 5 folds, group-disjoint by problem |
| verbaliser | 2 windows per problem, 2 samples per window, `max_new_tokens` 256, `T = 1.0`; controls on |
| NLA arm cap | 80 problems configured; see §4.6 |
| faithfulness | at most 8 claims per explanation, 3 paraphrases, null at the 95th percentile |
| causal | 5 arms, doses {0.0, 0.5, 1.0, 1.5, 2.0}, 5 random directions, cap 60 problems |

Dose 0.0 is retained as a within-arm no-op control: it must reproduce the
unintervened baseline exactly, and the stage checks that it does.

**Semantic entropy is adapted, not reproduced.** The published method clusters
by bidirectional NLI entailment [21]; this implementation clusters by symbolic
equivalence instead. For mathematical answers that is the right equivalence
relation — NLI would mis-handle `\frac{1}{2}` against `0.5` — and no NLI model
is downloaded. The substitution is reported as an adaptation in §8.

**Semantic entropy cannot say when to stop.** It is a per-problem quantity, so
its score is constant across boundaries; the sweep treats it accordingly. That
is a real property of the signal, not a modelling shortcut, and it bounds what
the comparison against it can show.

## 4.5 Seeds, provenance and reproducibility

Every stage is separately seeded (generation, AST, semantic entropy, probe,
verbaliser, faithfulness, causal, analysis). Bootstrap resampling uses 10,000
iterations at 95% confidence.

Each run directory carries the resolved config and its hash, the git commit,
seeds, model revisions and file hashes, dataset version, package versions,
timestamps and per-stage status. Nothing overwrites: a re-run with the same id
refuses unless forced. Every quantity reported in the paper is therefore
traceable to the artifact that produced it.

**The run is resumable per problem**, and the interleaved order means an
interruption costs at most one problem and leaves a proportional corpus.

## 4.6 Achieved N — outstanding

**The achieved N, not the configured N, is what the paper reports.** At the
time of this draft the main run is in progress and the number is not yet
final, so it is left as a blank to be filled rather than guessed.

Two arms run on a proportional subset for cost reasons and report their own
achieved N separately: the verbaliser arm, because the autoencoder emits
hundreds of tokens per activation and is an order of magnitude more expensive
than the behavioural arms; and the causal arm, at 60 problems. The primary
comparison is computed on exactly the problems the verbaliser arm covers, with
every baseline re-swept over those problems, so that the comparison is between
rules rather than between problem sets (§5.1).

Sizing was done against the primary question rather than by ambition: because
the primary comparison is a McNemar test over problems, its power comes from
**discordant** problems, and concordant problems contribute nothing however
many there are. Eight discordant problems is the fewest that can clear the
corrected threshold, which at the pilot's rate of 4/21 puts the requirement at
roughly 42 problems in the verbaliser arm (`docs/DECISIONS.md` D32).
`scripts/size_o3_corpus.py` recomputes this from any completed run.

---

# 5. Statistical analysis plan

## 5.1 The primary comparison: matched budget, problem-level

**Unit of analysis: the problem.** Safe stopping (§3.5) is a per-problem binary
outcome and every rule is scored on the same problems, so the comparison is
paired over problems.

This is stated because the alternative was tried and is wrong. An earlier
analysis paired over the nine requested *budget levels* and applied a Wilcoxon
signed-rank test. A rule's safety/saving curve is a step function, so a span
of requested budgets resolves to a handful of distinct operating points and
asking for more budgets copies one comparison rather than adding evidence. On
the pilot all nine budgets landed on a single operating-point pair, and the
resulting p-value was a function of the grid size — 0.250 at three budgets,
0.0039 at nine, 1e-6 at twenty-five. Nine was a default. The test is now
**McNemar's exact test over problems**, with a conditional-exact paired
interval on the rate difference (D29).

**Matched budget.** Each rule exposes a per-boundary score; the threshold is
swept; rules are read off their curves at the same mean token saving. A rule
that stops later is trivially safer, so comparing rules at their own operating
points would compare budgets rather than rules.

Three implementation commitments make that comparison honest:

- **The threshold grid is exhaustive, not fixed-size.** A curve changes only
  at a threshold that crosses an observed score, so sweeping the *observed
  values* enumerates every achievable operating point exactly. A fixed linear
  grid samples the score axis instead and therefore depends on how the scores
  happen to be distributed: the probe's scores saturate, so 25 evenly spaced
  thresholds found 5 operating points and none near a shared budget, and its
  comparison was reported as unevaluable. Swept at observed values the same
  problems give 86 operating points (D30).
- **A match is a match, within tolerance.** An operating point counts as
  matched only if it lies within **0.05** of the target in mean fraction of
  tokens saved; otherwise the comparison is not made at that budget. Returning
  the nearest point regardless would produce a table that looks matched and is
  not — on the pilot, asking all four rules for a budget of 0.632 returned one
  rule at 0.407 and another at 0.855.
- **The budget grid is the intersection of what every rule can reach**, with
  zero-saving points excluded. At a point where no rule stops, every rule is
  trivially safe and the difference is identically zero; including it would
  make the comparison vacuous rather than conservative.

**The comparison is computed on the problems the verbaliser arm covers**, with
every baseline re-swept over exactly those problems. Comparing curves built
from different problem sets would confound the rule with the problems.

## 5.2 Three outcomes a planned comparison can have

These are reported differently, and conflating them is the main way a
matched-budget design misleads.

| outcome | signature | what it licenses |
|---|---|---|
| **evaluable** | matched budget exists; ≥1 discordant problem | the McNemar result and its interval |
| **evaluable, uninformative** | matched; **zero** discordant problems | an observation of exact agreement over N problems — *not* an absence of evidence |
| **unevaluable** | no budget where both rules have a matched point | **nothing** about the rules' relative safety; the comparison was not made |

A test in the third category stays in the report rather than disappearing from
it, because a comparison that was planned and could not be made is itself a
result. We report the reason, and distinguish a rule whose score is too coarse
to place an operating point in the window from two curves that simply do not
overlap on this corpus (D31, D37b).

**Power is reported per comparison, not just overall.** For each McNemar
comparison we report `smallest_attainable_p`: the p-value that many discordant
pairs would give if every pair fell the same way — the best the comparison
could possibly have done. It separates *no difference was found* from *no
difference could have been found*, and needs no simulation. On the pilot's four
discordant pairs it is 0.125 against a corrected threshold near 0.01, so that
comparison could not have reached significance whichever way the pairs fell.

## 5.3 Multiplicity

Benjamini–Hochberg at **q = 0.05** over the pre-registered family.

**The denominator is the evaluable tests.** The justification is not
convenience: BH controls the false discovery rate over hypotheses actually
tested, and a hypothesis that produced no p-value can be neither a rejection
nor a false one, so including it buys no error control and only makes every
other test conservative. What "pre-registered family" guards against is
choosing the denominator *after seeing which tests came out well*, and that
guard holds provided every exclusion is for a reason independent of the
direction and size of the effect under test. On this plan they are: no matched
budget exists (set by the saving curves, not the safe rates); fewer than three
observations in an arm (set by accuracy, not by the quantity compared); and a
dependent variable with no variance (an instrument floor, not the arms' means).

**One exclusion was not independent of the effect, and was corrected.** A
McNemar comparison with zero discordant pairs previously returned no p-value
and dropped out of the denominator — but zero discordance is exactly the null
holding perfectly, so the family was contracting *because a null was true*,
which pulls every other q-value down. The exact conditional p-value there is
**1.0**: conditioning on zero discordant pairs leaves a single possible table.
That is arithmetic, not convention, and the test is now counted. Replayed on
the pilot's own p-values the correction raises every q-value, changes no
verdict, and puts a previously vanishing comparison on the table as a null
(D38).

**Both conventions are reported.** Every run also emits the same p-values
corrected over the *whole* pre-registered family, plus a flag recording
whether the set of significant findings differs between the two. The choice of
convention therefore changes which number is foregrounded, never which numbers
exist.

## 5.4 The causal gate

The claim *a tail direction exists* is a conjunction of four conditions
(§3.8). Two points matter numerically.

**"Beat" is a paired test, not a comparison of means.** It previously compared
two arm averages, which has no inferential content — a candidate at 0.30
against a control at 0.29 satisfied it, on ten problems, with no test, no
interval and no sensitivity to how many problems were behind either number.
Every arm runs on the same problems, so the contrast is paired over the
problems each arm shares, by Wilcoxon signed-rank at α = 0.05, read at the
dose the controls run at. The quantity per problem is
`|effect(candidate)| - |effect(control)|`: absolute, because the question asks
whether ablation *changes* behaviour rather than which way. Replicate control
directions are averaged within a problem first, so a problem counts once
however many replicates it has. This is the same correction §5.1 applies to
the primary comparison, and the two are now consistent (D41).

**Every condition is tri-state.** True, false, and *could not be compared* —
and the conjunction is undetermined if any input is. The three ways a
condition goes undetermined are reported separately: the readout does not vary
at all; fewer than three problems appear in both arms; or the two arms moved
the readout identically, in which case the answer is a determinate refusal
because "beat" is one-sided. Collapsing any of these into "false" would read
as evidence against a tail direction where there is none either way.

**The dose-response requirement** needs at least three *distinct doses*, a
curve that actually moves, and non-decreasing magnitude over the sweep. A
constant readout satisfies "non-decreasing" vacuously, so an unmoving curve
previously reported a satisfied dose-response beside a readout that never
changed; it is now undetermined in that case (D41).

**The intervention must be shown to have happened.** No p-value is reported
for the causal family test unless the candidate arm is *demonstrated* to have
edited at least one position at the compared dose. Without that condition, two
arms agreeing because nothing was done to either would read as a tested null.
An unrecorded edit count counts as not demonstrated.

**The dependent variable is a documented choice, not a registered one.** §3.8
records that the construct is registered and its operationalisation is not.
The measure used is a count of rechecking markers in the generated text,
compared between arms — the face-valid reading of the construct. A readout
measuring a *different* construct may not be substituted for it: output
length, for instance, varies where the marker count does not, but "how much
further the model writes" is not verification, and adopting it would answer a
different question under this one's name. Eligibility is enforced in code
against the registered construct (D41). §8 reports what this costs.

## 5.5 Outstanding reporting decision

**Which multiplicity convention to foreground.** §5.3's argument supports the
evaluable-test denominator as primary, with the pre-registered-family count as
a stated sensitivity, and that is how the code computes it. The
pre-registration's wording admits the stricter reading, and no data can settle
a reporting convention. Both number sets are in every artifact and the pilot's
verdict is identical under both; if the main run's robustness flag comes back
false, the choice decides a finding and must be stated explicitly in the
Results rather than assumed.

## 5.6 Falsification battery

Ten pre-registered alternative explanations, each with a test rather than an
argument: length alone (F1); difficulty, stratified by MATH level **and** by
gold-answer magnitude (F2); verbaliser priors, reported as a headline (F3);
answer-parser artefact, via a second stricter extractor (F4); sensitivity to
the AST definition itself, sweeping `K`, `T`, `w` and the chunker (F5); probe
leakage, by label shuffling within group — the probe must collapse to chance
(F6); layer specificity, at 14 and 24 (F7); quantisation (F8); multiplicity
(F9, §5.3); and token-position confounding in reconstruction, comparing tail
against matched-position and matched-length windows (F10).

F10 is the one most likely to constrain the headline: if tail reconstruction
differs from matched-position but *not* from matched-length, the effect is
positional rather than tail-specific, and §8 says so.

---

# 6. Results

**Not written.** Waiting on the main run; `runs/main/analysis/results.json` and `runs/main/report/RESULTS.md` are the sources it will be written from, and `scripts/check_run_consistency.py` is what certifies them first.

This placeholder is deliberate. The section is part of the planned report and its absence is a state of the work, so it is recorded here rather than left out of the document: a reader of this draft should be able to see what is missing without consulting the repository. Nothing in it is a finding, and no number in it is provisional, because there is no number in it.

---

# 7. Discussion

**Not written.** Waiting on the Results.

This placeholder is deliberate. The section is part of the planned report and its absence is a state of the work, so it is recorded here rather than left out of the document: a reader of this draft should be able to see what is missing without consulting the repository. Nothing in it is a finding, and no number in it is provisional, because there is no number in it.

---

# 8. Limitations and threats to validity

Ordered by how much each constrains what the paper can claim.

## 8.1 RQ3 cannot be answered on this model, and we can say why

The causal question is not reported as negative. It is reported as
**untestable here**, for three independent reasons, all measured rather than
inferred.

**The dependent variable has no range on this model.** The registered
construct is verification-like behaviour (§3.8), and its face-valid measure is
a count of rechecking markers in the generated text. Across **200 traces** —
the pilot's 24 plus the first 176 of the main run — there is exactly **one**
marker occurrence: the word "confirming", once, in one trace. None of the
other nineteen markers appears anywhere. Pooled rate 1/200 = 0.0050, 95% exact
interval [0.0001, 0.0275]. Projected onto the 60-problem causal arm that is
0.30 marker-bearing problems expected, or 1.65 at the interval's upper bound,
against a minimum of three paired problems before a contrast is attempted.

**More data would not fix this, and the evidence for that is that more data
has already made it worse.** The count stayed at one as the corpus grew from
140 traces to 200, so the interval's upper bound fell from 0.0392 to 0.0275
and the most optimistic yield from 2.35 marker-bearing problems to 1.65. Three
are not expected until the arm holds roughly 109 problems even at the top of
the interval, against a configured 60. The binding constraint is that this
model writes clean step-by-step derivations without rechecking language, so
the quantity the construct names barely occurs. That is a finding about
Qwen2.5-7B-Instruct's output, not a defect in the design, and it means no
corpus size reachable by this study would make the comparison possible.

**A readout that varies is not a substitute.** Output length — how much
further the model writes after the intervention — does vary, and was
considered for that reason. It measures a different construct: a model writes
more when rechecking, and also when repeating, degenerating or padding.
Adopting it because it moves would be choosing the dependent variable to get
an answer, and would answer a question about length under a question about
verification. We did not, and the eligibility constraint is enforced in code
rather than left to discipline.

**The independent variable is also unvalidated.** The direction edit was a
no-op for the entire pilot: all three direction arms edited token positions
the forward pass never contained, which is why doses 0.0, 1.0 and 2.0 returned
byte-identical answers on all ten problems. The defect is fixed and
unit-tested, including the cached-decoding case that caused it, but the stage
has not been re-executed on a GPU. Until it has, nothing may be claimed about
the tail direction in either direction, and the pilot's recorded RQ3 row must
not be read as a null — it was measured while the independent variable was not
being applied.

**What would unblock it:** registering a validated in-construct measure with
range — for example a judge-scored rechecking rubric with inter-rater
agreement reported — *before* a confirmatory run; or amending the research
question itself if length is the quantity of interest, dated and justified as
a change to the question rather than to its measurement. Neither is done here.

## 8.2 The probe baseline is largely positional, and the headline is scored against it

The hidden-state correctness probe reaches AUC 0.996, against **0.962** for an
activation-free predictor built from token position alone — a margin of 0.034.
Since the primary comparison is scored against that probe, a verbalised
readout that merely *ties* it has not been shown to read more than position.
Any claim of parity with the probe has to be read with this number beside it.

## 8.3 The location contrast is confounded with edit extent

The two causal controls are matched on different things, and only one is
matched exactly. `random_dir` is given the candidate's own span, so that
contrast isolates the direction. `matched_position` is the registered
matched-length window ending before the tail, which lies inside the prompt and
therefore arrives in a single forward pass and is edited in full, while the
candidate edits only as far as generation actually ran. On the pilot's spans
that is a median of 72 positions against 36 — the control is the **larger**
edit, by about a factor of two.

The arms also differ in kind: the candidate edits representations as tokens are
generated, so its edit compounds into what gets written next, while the
control edits already-fixed context acting only through attention. Matching the
counts would not make them the same experiment.

**Direction of the bias, stated so it cannot be read either way.** If the
candidate nonetheless moves the readout further, it does so against the
asymmetry and the conclusion is conservative. If the control moves it further,
extent is a live alternative explanation and the comparison does **not**
support the location claim. The per-arm extents and their ratio are reported
with the result rather than left in a log.

## 8.4 Scale, and what a wide interval means

One model, one seed, one primary layer, two benchmarks, two MATH subjects. The
pilot's conclusions rest on 24 problems; the main run changes the corpus size
and none of the other four.

A wide interval is reported as a wide interval. Where the primary comparison
cannot resolve a difference we say it is underpowered rather than claiming
equivalence, and §5.2's `smallest_attainable_p` makes the distinction
checkable per comparison rather than asserted.

**[N]** The achieved N, the realised discordance rate, and whether the primary
comparison had the power its sizing assumed.

## 8.5 One head-to-head may not be comparable at all

The answer-convergence rule's achievable operating points fall either side of
the shared budget window rather than in it, so on the corpora measured so far
that comparison cannot be made at matched budget (§5.2).

Because `w` is an integer, no operating point exists *between* adjacent window
widths, and that part is structural at any corpus size. But whether the
*window* falls in such a gap is not: the achieved savings are means over
problems and move with the corpus, and the window itself is the intersection
of every rule's reachable range, pinned in our pilot by the verbalised arm's
own narrow band. An earlier claim that no corpus size could resolve this was
withdrawn; it had been argued from a coarser score that has since been
replaced.

Observed gap between the bracketing operating points: 0.398 at 24 traces,
0.491 at 86, 0.464 at 100 — wider than the pilot throughout, and **not
monotone**. Three observations of a noisy quantity, not a trend, and no
projection is made.

**[N]** Whether the completed corpus makes this comparison evaluable.

## 8.6 Measurement conventions that are choices

- **Reconstruction is not faithfulness.** The quantity measured is whether a
  verbalisation supports reconstructing the activation, not whether it
  describes the computation. We use the former word throughout.
- **The verbalised readout's stopping score is one reduction of text to a
  scalar.** Reconstruction cosine at the boundary, carried forward from the
  sampled windows, is the cheapest defensible reading of "the readout says
  this state is well-characterised". A different reduction could give a
  different curve, and this study does not claim to have found the best one.
  The choice also bounds the arm's reachable savings band, which is what
  constrains the matched-budget window in §8.5.
- **The faithfulness thresholds** are calibrated by the paraphrase null, but
  that null controls for surface change and not for how much text a deletion
  removes. Both headline numbers are reported with a sensitivity over the
  choice, and the Jaccard cutoff used for the noise-reproduction number has no
  principled basis and is reported with its own sensitivity.
- **The tail definition measures determinacy, not statement** (§3.3). It is
  not interchangeable with post-answer redundancy.
- **The measured tail is a floor.** The forcing budget bounds it from below;
  the relaxed arm gives the ceiling (§4.3).

## 8.7 Adaptations and substitutions, declared

- **Semantic entropy is adapted**, clustering by symbolic equivalence rather
  than bidirectional NLI entailment. Defensible for mathematical answers and
  reported as an adaptation, not a reproduction. It also cannot say *when* to
  stop, being a per-problem quantity, which bounds what the comparison against
  it shows.
- **The verbaliser runs through a local inference path** rather than the
  upstream serving stack, which has no support on this platform and would not
  fit beside the target model in 6.14 GiB. The arithmetic is preserved step
  for step and the conventions are taken from the released model card rather
  than invented.
- **The quantisation check is fixed at the pilot's 8-problem subsample**
  (cosine 0.987) and cannot be re-measured without re-downloading 15.3 GB of
  bf16 weights. The cosine is a property of the checkpoint rather than of the
  corpus, so it carries forward, but the sample is small.
- **The layer sweep is partial.** The released autoencoder is layer-20 only,
  so layers 14 and 24 exercise the probe and entropy arms but not the
  verbalised readout.

## 8.8 Corpus composition

Truncated traces are a biased subset: in the pilot, the three truncated MATH
traces were exactly the three with no detectable tail. Raising the cap reduces
that bias without removing it, and the achieved truncation rate is reported
rather than assumed away.

At 128 main-run traces the MATH rate is 4/51 = 0.078, 95% interval
[0.022, 0.189]. That excludes the pilot's 0.300, but it is not zero and it has
risen from 0.048 at an earlier checkpoint as harder levels entered the corpus,
so some tail-free-because-truncated bias remains in the main corpus too.

The corpus deliberately retains problems the model answers incorrectly, since
the construct does not presuppose correctness; pooled quantities should be read
with that in mind.

**[N]** The achieved truncation rate and the status-code distribution.

## 8.9 Single rater

The corpus-release objective anticipated a blinded multi-rater annotation. A
second trained annotator was not obtained, so annotation is single-rater and
**no reliability coefficient is reported**. We state the absence rather than
omitting the limitation.

---

## References

*Numbering follows `Project_Review_II.md`, so every citation in this report points at an entry that already exists in the repository. Each entry still requires independent verification before submission.*

[1] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-thought prompting elicits reasoning in large language models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[2] DeepSeek-AI, "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," *Nature*, vol. 645, pp. 633-638, 2025, doi: 10.1038/s41586-025-09422-z.

[3] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "Training verifiers to solve math word problems," preprint, 2021, arXiv:2110.14168.

[4] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, "Measuring mathematical problem solving with the MATH dataset," in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2021.

[5] X. Liu and L. Wang, "Answer convergence as a signal for early stopping in reasoning," in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2025, pp. 17896-17907, arXiv:2506.02536.

[6] A. Zhang, Y. Chen, J. Pan, C. Zhao, A. Panda, J. Li, and H. He, "Reasoning models know when they're right: probing hidden states for self-verification," in *Proceedings of the Conference on Language Modeling (COLM)*, 2025, arXiv:2504.05419.

[7] S. Caldarella, D. Talon, R. Aljundi, E. Ricci, and M. Mancini, "Thinking past the answer: evaluating harmful overthinking in large reasoning models," preprint, 2026, arXiv:2606.02835.

[8] Z. Zhai, X. You, W. Yan, and X. Wang, "How much thinking is enough? Quantifying and understanding redundancy in LLM reasoning," preprint, 2026, arXiv:2605.23926.

[9] Q. Long, K. J. Jiang, J. Chen, X. Guo, L. Gan, and W. Wang, "Self-verification dilemma: experience-driven suppression of overused checking in LLM reasoning," preprint, 2026, arXiv:2602.03485.

[10] K. Fraser-Taliente, S. Kantamneni, E. Ong, D. Mossing, C. Lu, P. C. Bogdan, E. Ameisen, J. Chen, D. Kishylau, A. Pearce, J. Tarng, A. Wu, J. Wu, Y. Zhang, D. M. Ziegler, E. Hubinger, J. Batson, J. Lindsey, S. Zimmerman, and S. Marks, "Natural language autoencoders produce unsupervised explanations of LLM activations," *Transformer Circuits Thread*, project report, 7 May 2026. [Online]. Available: https://transformer-circuits.pub/2026/nla/

[11] H. Dingeto, "Train the model, not the reader: decodability supervision for verifiable activation explanations," preprint, 2026, arXiv:2607.20379.

[12] M. Li, A. M. Ceballos Arroyo, G. Rogers, N. Saphra, and B. C. Wallace, "Do activation verbalization methods convey privileged information?," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2026, arXiv:2509.13316.

[13] G. Lek, Z. Xia, P.-Y. Chen, and L. Y. Chen, "Understanding confabulation and rethinking reconstruction in activation explanations," preprint, 2026, arXiv:2609.33702.

[14] H. Zhao, Z. Hei, W. Shi, H. Deng, N. Zou, and M. Du, "Faithful activation verbalization: reducing hallucinations in LLM representation interpretation," preprint, 2026, arXiv:2609.34033.

[15] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023, arXiv:2305.04388.

[16] T. Lanham, A. Chen, A. Radhakrishnan, B. Steiner, C. Denison, D. Hernandez, D. Li, E. Durmus, E. Hubinger, J. Kernion, K. Lukosiute, K. Nguyen, N. Cheng, N. Joseph, N. Schiefer, O. Rausch, R. Larson, S. McCandlish, S. Kundu, S. Kadavath, S. Yang, T. Henighan, T. Maxwell, T. Telleen-Lawton, T. Hume, Z. Hatfield-Dodds, J. Kaplan, J. Brauner, S. R. Bowman, and E. Perez, "Measuring faithfulness in chain-of-thought reasoning," preprint, 2023, arXiv:2307.13702.

[17] Y. Chen, J. Benton, A. Radhakrishnan, J. Uesato, C. Denison, J. Schulman, A. Somani, P. Hase, M. Wagner, F. Roger, V. Mikulik, S. R. Bowman, J. Leike, J. Kaplan, and E. Perez, "Reasoning models don't always say what they think," preprint, 2025, arXiv:2505.05410.

[18] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou, "Large language models cannot self-correct reasoning yet," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2024.

[19] F. Zhang and N. Nanda, "Towards best practices of activation patching in language models: metrics and methods," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2024, arXiv:2309.16042.

[20] N. Goldowsky-Dill, C. MacLeod, L. Sato, and A. Arora, "Localizing model behavior with path patching," preprint, 2023, arXiv:2304.05969.

[21] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature*, vol. 630, pp. 625-630, 2024, doi: 10.1038/s41586-024-07421-0.

[22] Y. Mo, D. Zhao, and H. Geng, "Stable answers, unfinished reasoning: why self-consensus is not a safe early-exit signal," preprint, 2026, arXiv:2609.09989.

[23] Realmbird, "NLA thought anchors," online community analysis, 31 May 2026. [Online]. Available: https://www.lesswrong.com/posts/6HnnMHRoJLff46kgw/nla-thought-anchors

[24] S. Koh, S. Choi, M. Kwon, S. Baek, and J. Kim, "&lt;/think&gt; doesn't stop reasoning: analysis of spurious CoT termination," preprint, 2026, arXiv:2609.03633.

[25] K. Meng, D. Bau, A. Andonian, and Y. Belinkov, "Locating and editing factual associations in GPT," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[26] A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, B. Chen, A. Pearce, C. Citro, E. Ameisen, A. Jones, H. Cunningham, N. L. Turner, C. McDougall, M. MacDiarmid, C. D. Freeman, T. R. Sumers, E. Rees, J. Batson, A. S. Jermyn, S. Carter, C. Olah, and T. Henighan, "Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet," *Transformer Circuits Thread*, project report, 2024.

[27] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Welleck, B. P. Majumder, S. Gupta, A. Yazdanbakhsh, and P. Clark, "Self-refine: iterative refinement with self-feedback," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.  *(not yet cited)*
