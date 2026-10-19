# 6. Results

*Final. Every number is read from the main run's artifacts: `runs/main/analysis/results.json` and its tables, and the execution report `runs/main/report/RESULTS.md`, which `scripts/check_run_consistency.py --run runs/main` certifies against the analysis (179 checks, 0 mismatched). §6.8 reports the exploratory and partial analyses separately, each from its own recorded artifact and decision entry.*

Sections 6.1 to 6.7 are the registered analysis of the main run. Section 6.8
is not: it collects analyses run after the main result was seen, or run in
part, and none of them changes a registered conclusion.

## 6.1 Corpus and achieved N

All **250** configured problems were traced (150 GSM8K, 100 MATH: 74 algebra,
26 counting and probability), with no failed generation. Final-answer accuracy
is **0.880**: 0.900 on GSM8K and 0.850 on MATH.

The arms run on the following problems:

| arm | problems | composition |
|---|---|---|
| traces, AST, baselines, semantic entropy, probe | 250 | the whole corpus |
| verbalised readout (NLA) and the primary comparison | **120** | 74 GSM8K + 46 MATH algebra |
| claim-level audit (RQ2) | 120 | as the NLA arm: 6,007 claims over 1,413 explanations |
| causal interventions (RQ3) | 60 | 34 GSM8K + 26 MATH algebra |

The NLA arm ran at 120 problems through a recorded configuration override of
the configured 80 (stage hash `0bb33cf4c468a3f6`, `docs/DECISIONS.md` D48).
Both subset arms are capped proportionally by dataset, not by subject. Each
fills its MATH share with a sorted prefix of problem ids, so **neither contains
a counting-and-probability problem** (D54). Every result computed on the NLA
or causal arm therefore describes GSM8K and MATH algebra only (§8.11).

The generation cap of 768 tokens truncated **0 of 150** GSM8K traces and
**8 of 100** MATH traces (0.080, 95% exact interval [0.035, 0.152]): 5 of 74 in
algebra and 3 of 26 in counting and probability. The upper bound excludes the
pilot's 0.300 at a 512-token cap.

## 6.2 RQ1 — the Answer-Stable Tail

A tail was detected on **240 of 250** problems (0.960). On 3 of those it covers
only the final chunk, so the non-trivial tail rate is **0.948**. As §3.3
anticipates, this rate sits near its ceiling by construction: at the final
boundary the prefix is the whole trace and both criteria are close to
tautological. The informative quantity is how much of the trace the tail
covers:

- mean tail fraction **0.378** of chunks (median 0.341, range 0.04–1.00);
  mean tail length 107.7 tokens;
- tail tokens are **0.347** of the tokens generated in traces that have a
  tail, and 0.318 of all generated tokens.

**The detected tail is a lower bound.** The 48-token forcing budget cut the
forced answer off on 0.274 of evaluated boundaries. On 0.182 the cut was
*blocking*: criterion 1 failed while every resampled continuation reached the
final answer, on 162 problems in all. Counting those boundaries as satisfying
criterion 1 raises the mean tail fraction from 0.378 to **0.484**, and the
tail's token share from 0.347 to 0.447. The relaxed figure is not the
registered criterion and is reported only as the other end of the interval.

**The tail measures determinacy, not statement.** On 0.988 of problems the
tail begins before the trace states its answer, by a mean of 2.76 chunks (up
to 9). For the stopping comparison that is the right property, because a rule
stopping there loses nothing. It also means a tail window contains computation
that produces the answer, so it must not be read as post-answer
verification (§3.3, §8.6).

Detection status, with the edge cases kept in the corpus: `ok` 238 problems
(accuracy 0.912), `no_stable_point` 10 (accuracy 0.100) and `stable_at_zero`
2.

**Registered test.** The tail fraction is higher on correct final answers:
Cliff's δ **+0.295** over 240 problems, p = 0.026, **q = 0.036**, which is
significant after correction.

The answer-convergence rule fires earlier than the AST on 0.601 of problems
(by 6.05 chunks where it does) and later on 0.373 (by 2.20 chunks). This is
the concern that consensus stopping fires on answers that are not yet settled
[22], measured on this corpus.

## 6.3 The primary question — does the verbalised readout add value?

Every rule is read off its own safety/saving curve at matched token budget,
with each baseline re-swept over exactly the 120 problems the NLA arm covers
(§5.1). On those problems, the budget window shared by the rules runs from a
mean token saving of **0.613 to 0.738**. Within it, every matched rule's
safe-stopping rate lies between **0.117 and 0.325**.

| compared against | NLA safe rate | comparator | discordant problems | paired rate difference [95% CI] | p | q |
|---|---|---|---|---|---|---|
| semantic entropy | 0.233 | 0.325 | 37 (24 favour semantic entropy) | −0.092 [−0.184, 0.016] | 0.099 | 0.115 |
| hidden-state probe | 0.125 | 0.133 | 11 (6 favour the probe) | −0.008 [−0.061, 0.049] | 1.000 | 1.000 |
| answer convergence | — | — | — | not made | — | — |

*McNemar's exact test over problems. The interval is the conditional-exact
interval for the paired difference in safe-stopping rate (readout minus
comparator).*

**The verbalised readout does not beat the cheaper signals.** Against both
comparators it can be matched to, its safe-stopping rate is lower. Neither
difference is significant after correction, and both intervals include zero.
The study therefore establishes **no difference in either direction**. It
shows a readout that, at matched budget on this corpus, is no better than
semantic entropy or a correctness probe, and directionally worse than semantic
entropy. A significant result here would have been evidence *against* the
readout's incremental value; this one is not evidence either way about
equivalence.

**The comparison with answer convergence could not be made.** On the NLA
problem set the rule reaches 10 operating points, and its threshold sweep is
exhaustive. Its points fall either side of the shared window, at savings of
0.431 and 0.879, so no budget has both rules within the 0.05 tolerance. The
comparison is reported as planned and not made (§5.2). It produces no p-value,
and no claim is made about the two rules' relative safety.

**The probe the readout is scored against is largely positional.** Its
grouped cross-validated AUC is **0.946** over 2,681 boundaries. The same model
on activation-free positional features reaches **0.928**, a margin of 0.018.
The probe's row is therefore best read as a positional stopping rule with a
small activation-derived increment (§8.2). A readout that ties it has not
been shown to read more than position.

**How the readout was scored.** The NLA score is mean reconstruction cosine at
the boundary, carried forward. It averages 1,413 of the 2,127 reconstruction
samples: 709 are the Gaussian noise control, and 5 failed the verbaliser's
integrity gate (0.004 of activation-driven samples). This reduction is one
choice among several (§8.6).

## 6.4 RQ2 — claim-level faithfulness and reconstruction fidelity

**Claims.** 6,007 claims were audited over 1,413 explanations.

| quantity | rate |
|---|---|
| reconstruction-dependent (deletion effect above the paraphrase null) | **0.750** |
| reproduced from a matched-norm Gaussian vector (verbaliser-only control) | **0.002** |
| dependent **and** not reproduced from noise | **0.749** |

**Registered test.** The deletion effect minus the paraphrase effect, paired
per claim: Cohen's d **0.712**, mean paired difference **0.0317** [0.0306,
0.0329], q < 0.001.

Two sensitivity checks bear on this. **First, deletion size.** Deleting a claim
removes 23.3% of an explanation on average, and the deletion effect rises with
the fraction removed (r = +0.330). Removing that trend and re-applying the same
null threshold gives a length-adjusted dependence rate of **0.772**, so the
result does not rest on the length confound. **Second, the noise control.**
Its reproduction rate stays at or below 0.016 across Jaccard thresholds from
0.2 to 0.6, and the similarity distribution (99th percentile 0.222) has no
cliff near the chosen threshold.

**Verbaliser integrity.** 1,413 of 1,418 activation-driven samples (0.996)
were well-formed English, against 560 of 709 (0.790) for the Gaussian control.
This difference between real activations and matched-norm noise was not a
planned measurement.

**Reconstruction fidelity** is reported separately from faithfulness (§3.6).

| window | n | mean cosine |
|---|---|---|
| tail | 476 | 0.873 |
| matched position (pre-stabilisation) | 470 | 0.878 |
| matched length, elsewhere in the trace | 472 | 0.876 |
| Gaussian control | 709 | 0.393 |

The FVE against this run's own activations as the baseline is 0.695
(PROJECT_PLAN §5); against an orthogonal prediction it is 0.875.

**F10: fidelity is not specific to the tail.** Tail windows reconstruct
slightly *worse* than position-matched windows (Cliff's δ −0.203) and
length-matched ones (δ −0.137); both differences are significant. Against the
Gaussian control the separation is total (δ 1.000). The readout describes a
late mathematical-reasoning state about equally well wherever in the trace it
is sampled. So a fidelity result on the tail is a result about reading this
kind of trace, not about the tail.

These numbers say that the verbalisations carry information the
reconstructor uses, and that they are not reproduced from verbaliser priors.
They do not say that a claim is a true description of the computation:
reconstruction dependence is not faithfulness (§8.6). The registered
independent-probe control meant to bear on that question was implemented only
later, as an exploratory amendment (§6.8).

## 6.5 RQ3 — causal evidence: not tested

**This time the interventions were applied.** The candidate direction was
ablated on 59 of the 60 causal-arm problems at every non-zero dose; the 60th
generated no continuation to edit. Its median edit spans 90 positions; the
matched-position control spans 89 (0.99 times), and the random-direction arm
63.

The share of problems whose answer changed rose with dose: 0.000, 0.000,
0.017, 0.067 and 0.217 at doses 0, 0.5, 1.0, 1.5 and 2.0. The controls are run
at the compared dose, 1.0, where neither changed an answer: the five random
directions (300 runs) and the matched-position control (60 runs) both had an
answer-change rate of 0.000.

**The registered readout did not move where the test is made.** RQ3 registers
the construct "verification-like behaviour". Its operationalisation, a count of
rechecking markers, is a documented choice (§3.8, D40–D41). At the compared
dose, the marker count is zero in the candidate arm and in both controls on
every one of the 60 paired problems. No contrast could be formed, so neither
control comparison can be beaten or failed. The gate reports the direction
claim as **not tested** (D47): this is neither support nor refutation, and it
is not evidence against a tail direction.

At doses 1.5 and 2.0 the candidate arm did produce markers, on 4 and 5 of 60
problems. No control was run at those doses, so these counts admit no
contrast either. They are reported as an observation, not as a test.

The constraint is the model's prose, not the sample size. Across 274 traces
(the pilot's 24 and the main run's 250), **2** carry any of the twenty
registered markers: pooled rate 0.0073, 95% exact interval [0.0009, 0.0261].
Projected onto a 60-problem arm, that gives 0.44 marker-bearing problems
expected (1.57 at the interval's upper end), against the three paired problems
a contrast needs (`scripts/measure_rq3_readout_floor.py`). RQ3 is therefore
closed as not tested on this model (D62). The answer-change dose–response
above is an exploratory observation about a different quantity and is not an
RQ3 result (§8.1).

## 6.6 Falsification battery

| test | result |
|---|---|
| F1 length | corr(trace length, tail fraction) = −0.512 |
| F2 difficulty | mean tail fraction 0.391 on GSM8K (148), 0.359 on MATH (92) |
| F3 verbaliser only | reproduced from noise 0.002; reconstruction cosine 0.875 real vs 0.393 control |
| F4 parser | final-answer agreement 0.968 under a stricter extractor; 11.0 vs 15.7 chunks per trace under the alternative segmentation |
| F5 AST sensitivity | over the 3 settings run (`k` and unanimity), mean tail fraction varies by 0.010 and the non-trivial tail rate by 0.008; the temperature and chunker sweeps were **not run** (§8.7) |
| F6 probe leakage | real AUC 0.946; across-groups shuffle 0.614; within-group 0.539; leak suspected: **no** (registered threshold \|AUC − 0.5\| > 0.15) |
| F7 layer | probe AUC 0.940 at layer 14, 0.946 at 20, 0.944 at 24; layer 20 is not special for the probe |
| F8 quantisation | blocked in the main run by design (the bf16 target was not on disk); measured on the pilot (cosine 0.987) and replicated in part afterwards (§6.8) |
| F9 multiplicity | §6.7 |
| F10 position | §6.4: tail fidelity not above matched windows |

F6 passed under its registered threshold. Its shuffled AUC depends on how
donor label sequences are fitted to each receiver's length, so the control
supports no claim stronger than "no leakage suspected under this control"
(§8.10, §6.8).

## 6.7 The registered hypothesis family

There are nine tests in the pre-registered family. Seven are evaluable, and
**five are significant** after Benjamini–Hochberg at q = 0.05:

- RQ1 (tail fraction by correctness);
- RQ2 (deletion versus paraphrase);
- the three F10 contrasts.

Neither evaluable O3 comparison is significant. Two tests produced no p-value,
each for a reason independent of the effect under test (§5.3): O3 against
convergence (no matched budget) and RQ3 (no variance in the readout).
Correcting over all nine tests instead of the seven evaluable ones gives the
same five significant results. The reporting convention §5.5 left open
therefore decides nothing on this run.

## 6.8 Exploratory and partial analyses

Each analysis below was run after the main result was seen, or run in part
by decision, and is labelled as such in its decision entry. None of them
changes a registered verdict above.

**The NLA arm at N = 240** (exploratory, post hoc; D54). The arm was extended
to all 240 problems with a tail, including counting and probability. The
original 120 problems reproduced exactly: 2,127 of 2,127 reconstruction
cosines were bit-identical.

| readout minus | N = 120 (registered) | N = 240 (exploratory) |
|---|---|---|
| semantic entropy | −0.092 [−0.184, 0.016], p 0.099 | −0.138 [−0.191, −0.071], p 5.1 × 10⁻⁵ |
| probe | −0.008 [−0.061, 0.049], p 1.0 | −0.029 [−0.059, 0.010], p 0.167 |

At twice the N the direction against semantic entropy holds and its interval
excludes zero. Because that N was chosen after the N = 120 result was seen, and
the extension's correction family differs from the main run's, this is not a
confirmatory result. The study's O3 conclusion remains the N = 120 result.

**F8, the bf16 replication, in part** (D55). On 24 problems from the NLA arm,
the target's layer-20 activations were recomputed in bf16 and the unchanged
NF4 verbaliser and reconstructor were rerun on both arms. Results:

- NF4 and bf16 activations agree at mean cosine 0.990.
- The per-window NLA score shifts by −0.0014 [−0.0033, 0.0005].
- The direction against semantic entropy is unchanged.

The verbaliser, reconstructor and behavioural arms stayed 4-bit, and 24
problems cannot show O3 equivalence. The quantisation threat is narrowed, not
closed.

**RQ2's independent-probe control, through amendment A1** (exploratory;
D56–D57, `docs/RQ2_PROBE_AMENDMENT.md` §12). The control registered for RQ2
asks whether a claim's content is decodable from the activation. Labels came
from an unvalidated lexicon rule, with no human validation. **It did not
corroborate the reconstruction-dependence reading for either attribute
tested:**

- *`answer_stated`*: the activation probe did not beat a position model
  (−0.0013 [−0.0040, 0.000]).
- *Subject cue (GSM8K versus MATH algebra only)*: the cue was specifically
  decodable from the claim's own activation (+0.368 [0.224, 0.522]). It was
  unrelated to which claims reconstruction depends on (−0.034 [−0.089, 0.028]).

The registered RQ2 result in §6.4 is unchanged.

**F6's control under a length-correct mapping** (exploratory; D58–D60). The
probe was refitted over 100 donor draws. A mapping that fits donor labels
along relative position gives a mean shuffled AUC of 0.841, against 0.602
under F6's `np.resize` with the same donors (+0.240). It also shifts label
prevalence from 0.200 to 0.228. F6's registered result and its no-leak
reading are unchanged, but they cannot carry a stronger claim (§8.10).

**Supplementary supervised analyses** (D51). These are cheap-feature models on
the 250 traces, with no activations:

- **Tail fraction.** Predicted at held-out R² **0.333** [0.188, 0.437] (ridge,
  143 held-out problems).
- **Final-answer correctness.** Predicted at held-out AUROC **0.818** [0.727,
  0.896] (logistic, 149 held-out problems).

Cross-validation inside the small training split corroborates neither
estimate; it is unstable at 97 to 101 rows, and the classifier's training
split holds only 6 incorrect answers. Each is therefore one estimate on one
split. Both bear on the premise: surface features already account for part of
what a readout would have to add to.
