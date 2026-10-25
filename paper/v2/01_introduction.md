# 1. Introduction

*Revised candidate, 2026-10-24. Evidence classes are marked as defined in
§3.8: [R] registered, [C] confirmatory, [S] pre-specified secondary or
sensitivity, [X] exploratory, [P] post hoc, [A] analytic. No claim is
stated more strongly than its row in `paper/CLAIM_EVIDENCE_MATRIX.md`
allows.*

Language models that reason step by step often keep generating after the
answer they will give has already settled [1, 2]. Stopping earlier saves
compute, but only if a signal can tell that the answer is settled without
changing it. Proposed signals include agreement between intermediate
answers [3, 4], the confidence of a trial answer [5], the entropy of the
next token after an end-of-thinking marker [6], probes on hidden states
or neurons [7, 8, 9], and calibrated or learned stopping rules [10, 11].
Natural-language autoencoders (NLAs) offer a different kind of signal [12].
An NLA verbaliser describes an activation in text, and a jointly trained
reconstructor rebuilds the activation from that text. If the description
of an activation says that the model is concluding rather than setting
up, it could indicate when to stop, and it would also say why.

We ask whether an NLA readout provides information about when a model's
answer has settled **beyond what cheaper signals already provide**. We
study Qwen2.5-7B-Instruct with its released layer-20 NLA on GSM8K and MATH
problems, first under a pre-registered plan and then in a confirmatory
replication on 400 problems that no earlier analysis had used.

**Findings.** On the registered outcome, a 48-token forced answer, the
replication supported both registered hypotheses: the readout beat the
registered correctness probe and a no-signal rule [C]. Two pre-specified
analyses qualify that result. A rule that sees only the length of the
prefix beat the readout, and once the readout's position trend was
removed it no longer beat the no-signal rule, so the advantage cannot be
attributed to what the explanations say [S]. When the forcing budget was
raised to 256 tokens, most pre-tail answers that had been cut off reached
the final answer, and neither registered hypothesis was supported under
that outcome [S]. The registered 48-token verdict stands as a result about
commitment within 48 tokens. The explanation text does carry predictive
information about the reasoning phase, concluding as against setting up
[X, S]. At positions chosen without reference to the outcome, however, it
adds no practically relevant predictive information beyond position and a
linear probe trained for the same label on the same activation [S]. In an
exploratory pilot on fresh problems, a stopping policy built on that probe
saved tokens, but its unsafe-stop rate could not be bounded below the
pre-specified target [X]. We do not test whether the explanations are
faithful to the computation.

**Contributions.** Prior work establishes that stopping signals should be
charged for their compute and compared at matched risk [10, 11], that
hidden-state probes predict the correctness of intermediate answers [7],
that evaluations of activation verbalisers need input-only and
same-activation probe controls [13], that truncating and forcing an answer
reveals when an answer commits [14, 15], and that agreement between
answers shows persistence rather than finished reasoning [16]. We claim
none of these. Within the focused review of §2, this paper contributes:

1. **A pre-registered, confirmatory test of an NLA readout as a stopping
   signal, reported with its pre-specified qualifications** (§4.1–§4.3).
2. **A measurement of the forced-answer instrument's dependence on its
   budget.** At 48 tokens, 54–56% of pre-tail forced answers are cut off on
   the earlier arms, and the registered conclusion changes when the budget
   is raised to 256 tokens (§4.3, §4.6).
3. **Two demonstrated ways in which an evaluation can manufacture a
   stopping advantage:** candidate points derived from the outcome, and a
   matched-mean-saving estimand that counts abstention as safe (§4.2,
   §4.6).
4. **An incremental-value test of NLA text against a same-label probe on
   the same activation at outcome-blind positions**, using the control of
   Li et al. [13] for a reasoning-state label (§4.4).
5. **An exploratory, prospectively calibrated pilot of a probe-based
   stopping policy on fresh problems.** It did not meet its safety
   criterion, and it supplies the out-of-sample risk, coverage and saving
   estimates that a confirmatory study would need (§4.9).

**Scope.** All claims concern one target model, one layer and one released
NLA. They concern prediction and incremental value, not faithfulness.
Selective-risk results are measured at outcome-selected candidate points
and are not evidence that any stopping policy is safe to deploy (§4.5,
§5.4). The one out-of-sample estimate, from the pilot, is not a safety
certification (§4.9). Cost-aware and risk-controlled evaluation are not
new [9, 10, 11].
The contribution is a carefully qualified negative result about NLAs as
stopping signals, together with measurements of the evaluation choices
that would otherwise have suggested a positive one.
