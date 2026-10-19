# 2. Related work

*Final. Every `[n]` was checked against its source on 2026-10-19; see
[`README.md`](README.md).*

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
in which checking behaviour is overused [9]. A model can improve a first-pass
output by iterating on its own explicit feedback [27]. Without external
feedback, however, language models struggle to self-correct their reasoning,
and their performance sometimes degrades after the attempt [18]. The
post-answer region of a standard reasoning trace has no external signal and no
explicit critique-and-revise structure, which limits how much productive
computation it is likely to hold.

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
decodability supervision for verifiable explanations [11].

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
