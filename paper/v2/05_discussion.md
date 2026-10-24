# 5. Discussion

*Revised candidate, 2026-10-24. Written from `paper/CLAIM_EVIDENCE_MATRIX.md`
§R, C, D and E; every claim keeps its row's evidence class.*

## 5.1 What the replication supports

The replication yields one confirmatory result, and it stands as
registered: on the registered 48-token outcome, the NLA readout's safe
rate at matched mean saving exceeded that of the registered correctness
probe and that of a no-signal rule [C]. No later analysis overwrites or
rewords that verdict. Two pre-specified analyses, however, determine how
it can be read.

First, the advantage cannot be attributed to explanation content [S]. A
rule that sees only the prefix length beat the readout, and the readout
with its position trend removed did not beat the no-signal rule. Because
the menu always contains the chunks on both sides of the answer change, a
preference for early positions is enough to look informative on it.
Second, the result depends on the forcing budget [S]. Most pre-tail
answers cut off at 48 tokens reach the final answer when given 256, so the
registered outcome mostly measures whether the model can commit to its
answer within 48 tokens; under the 256-token outcome, the registered
hypotheses are not supported, and effects of 0.05 or more are excluded.
The non-independent 240-problem arm shows the same pattern [X].

The accurate summary is therefore narrow. As registered, on the candidate
menu and when safety means committing within 48 tokens, the readout's safe
rate at matched mean saving exceeds that of the registered probe and of a
no-signal rule. This advantage is not shown to come from what the
explanations say, and it is not supported once forced answers may finish.
It is also not evidence that the readout's stops are mostly safe: at
matched coverage, most of them are unsafe (§4.5).

## 5.2 Prediction, incremental value and faithfulness

The results separate three questions that are easily run together.

**Prediction.** The explanation text carries information associated with
the stopping labels. It adds to position and the raw trace text on the
menu, adds to position at outcome-blind positions, and its most predictive
terms, which describe a concluding as against a setting-up phase, transfer
to unseen problems [S].

**Incremental value.** For these labels, the text adds no practically
relevant information once a linear probe trained for the same label on the
same layer-20 activation is included: the Stage 1 increment is equivalent
to zero within ±0.01 AUC, and the probe is better head to head [S]. The
readout's scalar score adds nothing beyond position and the registered
probe [S].

**Faithfulness.** Whether the explanations describe the computation that
produced the activation is not tested here (matrix E1), and neither of the
first two answers bears on it. Text can predict a label because it
describes the computation correctly or because it reflects correlates of
position and register, and a probe can match the text without the text
being wrong.

Two consequences follow. The absence of an increment does not show that
everything the text says is linearly available in the activation; it shows
that, for predicting these labels with TF-IDF features, nothing practically
relevant is left once a same-label linear probe is used. And the NLA's
value in this setting is interpretive access, a legible description of the
reasoning phase, rather than information that a probe lacks. This is
consistent with Li et al.'s argument for same-activation controls [13],
and it qualifies reported advantages of activation decoders over generic
linear probes [20]: against a probe trained for the same label, no
advantage appeared here.

The reconstruction-dependence rate (0.749 of 6,007 claims) belongs to the
same family of caveats. It shows that deleting most claims moves the
reconstruction more than the registered null does, not that the claims
are true. The exploratory rewording null addresses one weakness of that
null: under model-written rewordings that move the reconstruction about
three times as much, the dependent rate on a sample of claims changes
little (§4.8) [X]. Because the rewordings' meaning was not validated, this
qualifies the rate's robustness; it is not a test of faithfulness.

## 5.3 How evaluation choices manufacture a stopping advantage

The main run's results form a sequence of evaluation choices, each of
which moved the conclusion (§4.6). They suggest four practices for
evaluating stopping signals, none of them new in spirit [10, 11]:

1. **Choose candidate points without reference to the outcome.** A menu
   built around the answer change contains it, and a position rule
   exploits it [A, S]. Stage 1's outcome-blind draw is the remedy used
   here.
2. **Report risk and saving together rather than a single
   matched-mean-saving estimand.** Counting abstention as safe rewards
   rules that concentrate their savings in a few problems: an
   information-free falling rule outranked a probe-based rule with
   boundary AUC 0.861 on the registered estimand, and selective risk
   reverses the ranking [X].
3. **Report the forcing budget and the share of forced answers cut off.**
   Short forced answers are reproducible but often unfinished before the
   tail [X], and the registered verdict changes with the budget [S].
4. **Compare against a same-label probe on the same activation.** The
   registered correctness probe adds +0.026 AUC over position at every
   boundary of the main run, against +0.091 for a same-label probe [X]; a
   weak baseline makes any signal look useful.

## 5.4 What this does and does not say about stopping

The selective-risk results are **not deployment evidence**. They are
measured on the outcome-selected menu with 48-token labels; on it, the
registered probe stops rarely unsafely but saves little, the readout saves
more but mostly unsafely, and no rule dominates [S]. Cost rules out an
online NLA check in this setting: one check generates 0.89 of a mean
trace's tokens before reconstruction [A], so the NLA is at most an offline
research comparator.

**The practical question remains open.** A planning replay on stored
replication data suggests that a same-label probe could stop on far more
problems than a position-only rule at the same unsafe-stop rate. That
replay chose its thresholds on the same data it scored and walked only
four positions per problem; it is planning evidence, not evidence of
out-of-sample safety. A safety-constrained stopping-policy study, with
calibration and evaluation on fresh, disjoint problems, has been designed
but not run (`docs/STOPPING_POLICY_PROPOSAL.md`,
`docs/STOPPING_POLICY_PILOT_PROTOCOL.md`).

## 5.5 What the answer-stable tail is

The tail marks where the answer becomes *recoverable within a forcing
budget*, not where redundant text after the answer begins: tails begin
before the answer is written, and the measured budget dependence shows how
much of an apparent pre-tail failure is an unfinished derivation. This
agrees with Mo et al.'s reading of answer agreement as persistence under a
fixed probing procedure [16]. A tail defined under a longer budget could
start earlier; that recomputation was not done.

## 5.6 Conclusion

A natural-language readout of a language model's activations passes its
pre-registered stopping test on the registered 48-token outcome. The
advantage cannot be attributed to what the explanations say, and it is not
supported when forced answers are allowed to finish. The explanations
carry legible, transferable information about the reasoning phase, but for
predicting stopping labels they add no practically relevant information
beyond a linear probe on the same activation. Whether they are faithful
is untested, and whether any signal studied here supports a safe stopping
policy is a question for a separate, prospectively calibrated study. In
this study, the choice of candidate points, estimand, forcing budget and
baseline determined the answer more than the signal did; such choices
should be fixed without reference to the outcome and reported.
