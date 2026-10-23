# 5. Discussion (draft v1, 2026-10-23)

*Written from `paper/CLAIM_EVIDENCE_MATRIX.md` §R, C, D and E. Every claim
keeps the class its row gives it, and every number is the row's. Markers
as in §4.*

## 5.1 What the replication supports, and what it does not

The registered replication gives one confirmatory result, and it stands
as registered. On the registered 48-token forced-answer outcome, the NLA
readout beats the registered layer-20 correctness probe (H1, +0.047 [0.019,
0.075]) and no signal (H2, +0.057 [0.039, 0.076]) [C; R1]. No later
analysis overwrites or rewords that verdict. Two pre-specified analyses
qualify how it can be read:

- **It cannot be attributed to explanation content** [S; R2]. A rule that
  sees only the prefix length beats the readout by 0.025 [0.005, 0.044].
  With its log-position trend removed, the readout no longer beats no
  signal: +0.007 [−0.016, 0.029]. The menu always holds the chunks on both
  sides of the answer change (§3.4), so a preference for early positions is
  enough to look informative on it.
- **It depends on the forcing budget** [S; R4]. 86.7% of the pre-tail
  answers cut off at 48 tokens reach the final answer when given 256. So
  the registered outcome mostly measures whether the model can *commit
  within 48 tokens*. Under the 256-token outcome, the registered
  hypotheses are not supported: H1 −0.008 [−0.036, 0.024] and H2 −0.004
  [−0.024, 0.016], both equivalent within ±0.05. These intervals exclude
  effects of 0.05 or more, not smaller ones, and the 48-token estimates lie
  outside them.

The non-independent 240-problem arm shows the same pattern [X]. The
correct summary is therefore narrow: *as registered, the readout's safe
rate at matched mean saving on the candidate menu exceeds the registered
probe's and no signal's, when safety means committing within 48 tokens;
this advantage is not shown to come from what the explanations say, and it
is not supported once answers may finish.* It is not evidence that the
readout's stops are mostly safe. At matched coverage they are mostly unsafe
(§5.4; B6).

## 5.2 Prediction, incremental value and faithfulness are different questions

The results separate three questions that are easy to run together.

1. **Prediction.** Does the explanation text carry information associated
   with the stopping labels? **Yes.** On the menu, the text adds +0.077
   [0.063, 0.090] AUC beyond position and the raw trace text on the
   replication [S; R3]. At outcome-blind positions it adds +0.071 [0.052,
   0.089] beyond position [S; R6]. Its most predictive terms describe a
   reasoning *phase*, concluding against setting up, and a lexicon frozen
   on earlier data transfers to unseen problems (+0.091 [0.076, 0.106]) [S;
   R5].
2. **Incremental value.** Does it add anything beyond cheaper signals?
   **Not a practically relevant amount, for these labels.** At outcome-blind
   positions, the text's increment over position plus a linear probe on the
   same layer-20 activation, trained for the same label, is +0.00002
   [−0.004, 0.004]: equivalent within ±0.01 [S; R6]. Head to head, that
   probe is better than the text by 0.031 [0.015, 0.047]. The readout's
   scalar score adds nothing beyond position and the registered probe
   (+0.0005 [−0.0015, 0.0025]) [S; R3].
3. **Faithfulness.** Do the explanations describe the computation that
   produced the activation? **Not tested** [E1]. Neither predictive
   accuracy nor the absence of an increment bears on it. Text can predict a
   label because it describes the computation correctly, or because it
   reflects correlates of position and register. And a probe can match the
   text without the text being wrong.

Two consequences follow. First, the absence of an increment does **not**
show that everything the text says is linearly available in the
activation; it shows that, for predicting these labels with TF-IDF text
features, nothing practically relevant is left over once a same-label
linear probe is used [R6]. Second, the NLA's value here is *interpretive
access*, a legible description of a reasoning phase, rather than
information a probe lacks. That agrees with Li et al.'s call for
same-activation controls [12], and it qualifies reported advantages of
activation decoders over *generic* linear probes [32]: against a probe
trained for the same label, the advantage did not appear in this setting.

The reconstruction-dependence rate of RQ2 (0.749 of 6,007 claims; C6)
belongs to the same family of caveats. It says that deleting most claims
moves the reconstruction more than the registered null does. It does not
say the claims are true, and the registered null is mostly no edit or an
added discourse marker (D70).

## 5.3 How evaluation choices manufacture a stopping advantage

The main run's story is a sequence of evaluation choices, each of which
moved the conclusion (§4.6). They generalise as four recommendations for
evaluating any stopping signal, none of which is new in spirit [28, 29]:

1. **Choose candidate points blind to the outcome.** A menu built around
   the answer change contains the answer change, and a position rule
   exploits it [A; B5, R2]. Stage 1's outcome-blind draw is the remedy we
   used.
2. **Report risk and saving together, not one matched-mean-saving
   estimand.** Counting abstention as safe rewards concentrating savings
   in a few problems. An information-free falling rule then beats a
   same-label probe with boundary AUC 0.861 on the registered estimand [X;
   B7]. Selective risk at matched coverage reverses that ranking.
3. **Report the forcing budget and the truncation rate.** A 48-token
   forced answer is reproducible but often unfinished before the tail
   [X; A2, A3], and the registered verdict changes with the budget [S; R4].
4. **Compare against a same-label probe on the same activation.** The
   registered correctness probe adds +0.026 over position at every
   boundary, against +0.091 for a same-label probe [X; D3, D1]. A weak
   baseline makes any signal look useful.

## 5.4 What this does and does not say about stopping

**Selective risk on the menu is not deployment evidence** [S; R7]. On the
outcome-selected menu with 48-token labels, at 50% coverage, the
registered probe's stops are almost always safe but save little (risk
0.011, saving 0.028). The readout's stops save more, but most are unsafe
(0.632, 0.711), and no signal sits between them (0.472, 0.413). No rule
dominates, and these numbers do not rank policies for use.

**Cost rules out an online NLA check here** [A; D2]. One check costs 0.83
of a mean trace in generated tokens before reconstruction. The NLA is at
most an offline research comparator for stopping.

**The practical question is open.** A planning replay on stored rep1
Stage 1 data suggests that a same-label probe could stop far more often
than a position-only rule at the same in-sample unsafe-stop rate. But that
replay chose its thresholds in sample and walked only four positions per
problem. It is planning evidence, not evidence of out-of-sample safety. A
safety-constrained stopping-policy study with calibration and evaluation
on fresh, disjoint problems is proposed (`docs/STOPPING_POLICY_PROPOSAL.md`)
and has not been run.

## 5.5 What the answer-stable tail is

The tail marks where the answer becomes *recoverable within a forcing
budget*, not where redundant post-answer text begins. Tails begin before
the answer is written (A2), and the measured budget dependence (R4) says
how much of an apparent pre-tail failure is an unfinished derivation. This
agrees with Mo et al.'s reading of answer agreement as persistence under a
fixed probing procedure [22]. A tail defined under a longer budget could
start earlier. That recomputation was not done (A1).

## 5.6 Conclusion

A natural-language readout of a reasoning model's activations passes its
registered stopping test, on the registered 48-token outcome. But the
advantage cannot be attributed to what the explanations say, and it is not
supported when forced answers may finish. The explanations carry legible,
transferable information about the reasoning phase. For predicting
stopping labels, they add no practically relevant information beyond a
linear probe on the same activation. Whether they are faithful is untested,
and whether any of these signals supports a safe stopping policy at
deployment is a separate study. The broader lesson concerns evaluation: the
choice of candidate points, estimand, forcing budget and baseline decided
the answer here more than the signal did. These choices should be fixed
blind to the outcome and reported.
