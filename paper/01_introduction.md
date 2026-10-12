# 1. Introduction

*Draft. Does not depend on the main run's outcome.*

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
