# 7. Discussion

*Final. Interprets §6 only; no number here is new. Registered results, exploratory analyses and the question left untested are kept apart throughout.*

## 7.1 What the evidence supports

**A verbalised readout of layer-20 state did not improve safe stopping over
cheaper signals.** At matched token budget, on the 120 problems the arm
covers, the readout's safe-stopping rate was lower than semantic entropy's
(−0.092) and the probe's (−0.008). Neither difference was significant after
correction, and both intervals include zero (§6.3). This is the outcome the
study named as acceptable in advance, and it is reported at the weight a
positive result would have had.

What the result does **not** say is that the readout is equivalent to the
cheaper signals. The design can reject "no difference"; it cannot accept it,
and the intervals are wide. The exploratory extension to 240 problems found
the gap against semantic entropy in the same direction with an interval
excluding zero (§6.8). That makes the direction more credible but does not
convert it into a registered finding, because the N was chosen after the
first result was seen.

**The tail is real and large, and the registered criteria locate it before
the answer is stated.** Answer-stable tails were found on 0.948 of problems
non-trivially, covering a mean 0.378 of the trace on the detected (lower-bound)
criterion and up to 0.484 on the relaxed one (§6.2). Tail fraction differs
with correctness (q = 0.036). The criteria measure when the answer is
*determined*, not when it is *said*. That is what a stopping rule needs, and it
rules out reading the tail as a phase of post-answer checking.

**The verbalisations carry information the reconstructor uses, and are not
produced by the verbaliser's priors.** Three-quarters of claims are
reconstruction-dependent against a paraphrase null, almost none are reproduced
from matched-norm noise, and the result survives adjustment for deletion length
(§6.4). This is evidence about **reconstruction**, which §3.6 and §8.6
distinguish from faithfulness. The control registered to test claim content was
run only later, as exploratory amendment A1, and it did not corroborate the
dependence reading for either attribute it could test (§6.8).

## 7.2 What the evidence does not support

**Nothing is claimed about a causal tail direction.** The edit was applied and
had a dose-dependent effect on answers. But the registered readout,
verification-like behaviour measured by rechecking markers, has essentially no
range on this model (2 of 274 traces). At the compared dose it was zero in
every arm. RQ3 is therefore *not tested* (§6.5, D62). That is a statement about
the instrument on this model's prose, not evidence for or against a direction.
The answer-change dose–response is a different quantity, observed without a
registered contrast, and is not substituted for the construct (§8.1).

**Nothing tail-specific is claimed for the readout.** Reconstruction is no
better on tail windows than on position- or length-matched windows elsewhere
in the same trace (F10, §6.4). Whatever the readout describes, it describes it
about as well outside the tail as inside.

**Nothing is claimed about counting-and-probability problems** for the NLA
and causal arms, which contain none (§6.1, §8.11). Nothing is claimed about
other models, layers or precisions. The autoencoder is released for one layer
of one model, and the quantisation check is partial (§6.8, §8.7).

## 7.3 Reading the comparison

Two properties of the comparators shape how the primary result should be read.

**The probe is mostly position.** It exceeds an activation-free positional
predictor by 0.018 AUC (§6.3). So a readout that ties it has not shown that it
reads more than *where in the trace a boundary is*. That a readout fails to
beat even this mostly positional rule is the more informative half of the
result.

**Semantic entropy is a per-problem quantity, adapted to symbolic answers.**
It cannot tell where within a trace to stop (§4.4), yet at matched budget it
had, descriptively, the highest safe-stopping rates of the rules that could be
compared (0.175–0.325 in the window). Only its comparison with the readout was
tested, and that test was not significant. Its advantage therefore
comes from identifying which problems are safe to cut short at all, not from
locating the point within a trace.

Taken together with the supervised analyses, which predict tail fraction and
correctness from surface features alone with moderate accuracy (§6.8), the
picture is that a substantial part of what a stopping rule needs is available
before any activation is read. That does not show that internal state carries
nothing further. It shows that this readout, reduced to a reconstruction score,
did not demonstrate that it adds to what is already available.

## 7.4 Relation to prior work

The tail's size is consistent with prior reports of substantial redundancy in
reasoning traces [5], [8]. The convergence rule firing earlier than the AST on
most problems (§6.2) matches the concern that consensus stopping fires on
answers the trajectory has not finished with [22]. The positional character of
the probe qualifies, for this setting, how far correctness probes [6] can be
read as reading correctness rather than progress. The near-zero rate of
noise-reproduced claims addresses the verbaliser-prior concern of [12] for this
readout. It does so for reconstruction dependence, not for claim truth, which
is the gap [11] and [13] identify. None of these comparisons is a replication:
the models, benchmarks and definitions differ, and the statements above are
about consistency, not confirmation.

## 7.5 What would change the conclusions

- **A confirmatory test of the readout's direction against semantic entropy.**
  This needs a fresh, preregistered corpus at the size the N = 240 extension
  suggests, covering all three subjects, rather than any further increase of
  this one.
- **RQ3.** It needs a validated in-construct measure of verification-like
  behaviour, registered before a confirmatory run, or a model whose output
  contains such behaviour. The 83 evaluation problems the causal arm never
  touched are held back for that purpose (D62).
- **The quantisation threat.** Closing it needs the verbaliser, reconstructor
  and behavioural arms run at bf16, which this hardware cannot do (D55).
- **A different reduction of the verbalisation to a stopping score.** This
  study used one. A readout scored differently could produce a different curve,
  and this study does not claim to have found the best one (§8.6).
