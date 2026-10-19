# 8. Limitations and threats to validity

*Final. Written before the main run and completed with its own numbers; where
an entry first quoted a partial count, the completed run's figure is now
given beside it.*

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

**The completed corpus confirms it.** Over all 274 traces (the pilot's 24 and
the main run's 250), 2 carry a marker ("confirming" and "to confirm"): pooled
rate 0.0073, 95% exact interval [0.0009, 0.0261]. That projects to 0.44
marker-bearing problems in a 60-problem arm, or 1.57 at the upper bound.
Three would need an arm of about 115 problems even at the upper bound
(`scripts/measure_rq3_readout_floor.py`). In the main run's causal arm the
readout was zero in every arm at the compared dose (§6.5).

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
unit-tested, including the cached-decoding case that caused it. The pilot's
recorded RQ3 row must not be read as a null, because it was measured while the
independent variable was not being applied. In the main run the fixed stage
did apply the edit, on 59 of 60 problems at every non-zero dose, and answer
change rose with dose (§6.5). So the independent variable is now demonstrated;
the dependent variable is what has no range.

**An effect on answers is not an effect on verification.** The dose–response
in answer change is a real observation, but it is about a different quantity
from the one RQ3 registers, measured without a registered contrast. It is
reported as exploratory and not substituted for the construct, for the same
reason output length is not.

**What would unblock it:** registering a validated in-construct measure with
range — for example a judge-scored rechecking rubric with inter-rater
agreement reported — *before* a confirmatory run; or amending the research
question itself if length is the quantity of interest, dated and justified as
a change to the question rather than to its measurement. Neither is done here.
RQ3 is closed as not tested on this model, and the 83 evaluation problems with
a tail that the causal arm never touched are held back for a separately
preregistered study (`docs/DECISIONS.md` D62).

## 8.2 The probe baseline is largely positional, and the headline is scored against it

On the pilot the hidden-state correctness probe reached AUC 0.996, against
0.962 for an activation-free predictor built from token position alone, a
margin of 0.034. On the main run the margin is smaller still: grouped-CV AUC
**0.946** against **0.928**, a margin of **0.018** (§6.3). Since the primary
comparison is scored against that probe, a verbalised readout that merely
*ties* it has not been shown to read more than position. Any claim of parity
with the probe has to be read with this number beside it.

## 8.3 The location contrast is confounded with edit extent

The two causal controls are matched on different things, and only one is
matched exactly. `random_dir` is given the candidate's own span, so that
contrast isolates the direction. `matched_position` is the registered
matched-length window ending before the tail, which lies inside the prompt and
therefore arrives in a single forward pass and is edited in full, while the
candidate edits only as far as generation actually ran. On the pilot's spans
that is a median of 72 positions against 36 — the control is the **larger**
edit, by about a factor of two. On the main run the medians are 89 for the
control and 90 for the candidate (0.99 times), so the counts were nearly
equal there. The difference in kind below remains.

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

**On the main run.** The achieved N is 250 problems, 120 of them in the
primary comparison. Against semantic entropy, 37 of 120 problems are
discordant (0.308), well above the eight the sizing required. The split was
13 to 24, which gives p = 0.099 and an interval of [−0.184, 0.016]. The sizing
guaranteed only that a difference *could* clear the corrected threshold, not
that one of this size would: the comparison had the discordant pairs it
needed and still could not resolve the effect. Against the probe, 11 of 120
are discordant, split 5 to 6. The exploratory extension to 240 problems shows
what twice the N does to the first comparison (§6.8). It was not planned, so
the main result is reported as underpowered for its observed effect, not as
equivalence.

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

**On the completed corpus it is still not evaluable.** On the 120 problems of
the primary comparison, the rule's operating points fall at savings of 0.431
and 0.879, either side of the shared window of 0.613 to 0.738. The sweep is
exhaustive, with 10 operating points, so the gap is not a matter of grid
resolution. The comparison is reported as planned and not made (§6.3).

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
  fit beside the target model in 6,141 MiB. The arithmetic is preserved step
  for step and the conventions are taken from the released model card rather
  than invented.
- **The quantisation check is partial.** The pilot measured cosine 0.987 on
  8 problems. The main run's F8 stage was blocked by design, because the bf16
  weights were not on disk. A later replication, partial by decision, then
  recomputed the target's activations in bf16 for 24 of the main run's
  problems (D55). It found cosine 0.990, no detectable shift in the readout's
  score (−0.0014 [−0.0033, 0.0005]) and no change in direction against
  semantic entropy. The verbaliser, the reconstructor and every behavioural
  arm stayed 4-bit, and the registered bf16 replication of the headline
  comparison was not run in full. It needs about 41 GB of bf16 weights and
  bf16 generation that this hardware cannot provide.
- **F5's sweeps over sampling temperature and the chunker were not run.**
  F5 exercised `k` and unanimity only. The temperature and chunker sweeps are
  implemented but were deferred by decision, at an estimated 60–150 GPU-hours,
  because existing generations cannot answer them (D52). The AST's
  sensitivity to those two choices is therefore unmeasured.
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

**On the completed corpus**, the cap truncated 0 of 150 GSM8K traces and 8 of
100 MATH traces: 0.080, 95% interval [0.035, 0.152]. Of the MATH traces, 5 of
74 were algebra and 3 of 26 counting and probability. Detection status is
`ok` on 238 problems, `no_stable_point` on 10 (accuracy 0.100) and
`stable_at_zero` on 2. All of them are kept in the corpus as status codes, so
the tail-free problems are visible rather than dropped.

## 8.9 Single rater

The corpus-release objective anticipated a blinded multi-rater annotation. A
second trained annotator was not obtained, so annotation is single-rater and
**no reliability coefficient is reported**. We state the absence rather than
omitting the limitation.

The same gap limits RQ2's exploratory independent-probe control (amendment
A1, §6.8). Its claim labels come from a lexicon rule that was not validated
against human judgement. The one completed pass over its annotation sheet was
made by an AI model and is not independent, because the same model designed
the rule. It was not an input to the evaluation, no agreement statistic is
computed from it, and it is released under a name that says so.

## 8.10 The probe-leakage control depends on how labels are fitted to length

Falsification test F6 controls for probe leakage by giving each problem
another problem's label sequence. Problems have different numbers of
boundaries, so the implementation fits each donor sequence to the receiver's
length with `np.resize`, which tiles a shorter sequence from its start and
truncates a longer one. That breaks most of the labels' association with
boundary position (relative-position AUC 0.920 → 0.686) and lowers their
prevalence (0.200 → 0.152) in F6's own shuffle.

An exploratory diagnostic, run after F6 under its own approved protocol,
refitted the probe over 100 donor draws. With the same donors, a mapping that
fits each donor sequence to the receiver's length along relative position
raises the shuffled AUC from 0.602 to 0.841 (+0.240; single draws +0.196 to
+0.278, problem-bootstrap interval +0.233 to +0.247, no refitting variability
included). That mapping also shifts prevalence, to 0.228, so it is a
sensitivity analysis rather than a definitive null. In every draw, the probe
scored below a position-derived benchmark that is neither an upper bound nor a
leakage test.

F6's registered result — real AUC 0.946, shuffled AUC 0.614, no leakage
suspected under its threshold of |AUC − 0.5| > 0.15 — is reported as
registered. Its shuffled AUC, and so its margin under the threshold, depends on
the length-fitting rule, and none of the alternative arms has a null centred at
0.5. The control therefore cannot support a stronger claim than "no leakage
suspected under this control"; ruling leakage out would need a new
preregistered design with a defined null, a fixed length-matching strategy and
uncertainty that includes refitting.

## 8.11 The verbaliser and causal arms omit one MATH subject

Both subset arms are capped proportionally by **dataset**, not by subject, and
fill their MATH share with a sorted prefix of problem ids. Algebra ids sort
before counting-and-probability ids, so the verbaliser arm's 46 MATH problems
and the causal arm's 26 are all algebra (`docs/DECISIONS.md` D54). The primary
comparison, RQ2, F10, the reconstruction-fidelity results and RQ3 are
therefore results on GSM8K and MATH algebra. They say nothing about counting
and probability, a quarter of the MATH problems in the corpus, on which the
model is also least accurate (0.769). The corpus-wide results (RQ1, the
falsification tests on the probe, the supervised analyses) cover all three.
The cap's stated purpose, proportionality by dataset, was met. Its effect on
subject coverage was found only after the main run, when the arm was
extended, and is reported here rather than corrected after the fact. The
exploratory 240-problem extension does include the missing subject (§6.8).
