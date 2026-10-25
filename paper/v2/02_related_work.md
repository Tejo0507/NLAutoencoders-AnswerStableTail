# 2. Related work

*Revised candidate, 2026-10-24. A focused review, not a systematic one:
`docs/NOVELTY_REVIEW_2026-10-20.md` (abstracts and five full texts) and
the 2026-10-23 sweep of training-free early-exit methods. Every reference
was checked against its primary record (`paper/v2/references.md`). Where
we say that prior work does not report something, we mean the papers
named. Statements about other papers are theirs; ours carry an evidence
class.*

**Overthinking and answer stability.** Models that reason step by step
often continue after their answer is settled [1, 2, 17]. Agreement between
successive intermediate answers has been proposed as a stopping signal
[3]. In a pre-registered sweep of consensus rules, Mo et al. [16] show
that such agreement establishes only that an answer persists under a
fixed probing procedure, not that reasoning has finished. Our
answer-stable tail is defined by the same kind of agreement, and we reach
the same reading: the tail marks where the answer becomes recoverable
within a forcing budget. We add a measurement of how the budget itself
changes the outcome (§4.3).

**Hidden-state probes and calibrated stopping.** Probes on hidden states
predict whether an intermediate answer is correct and can trigger an early
exit [7], and probes can also locate a commitment boundary after which
further reasoning steps are epiphenomenal [15]. Thought Calibration
controls the risk of terminating early with Learn-then-Test [11]. Dong et
al. [10] compare confidence, entropy, answer stability and learned
stoppers at a fixed rate of lost correct answers, and show that the
savings depend on how the cost of the stopping signal is counted. We
adopt both lessons, charging the signal's cost and comparing at matched
risk, and do not claim them. We differ from these papers in three
respects: our probe baseline is trained for the same label as the signal
under test; we report a failure mode of matched-mean-saving comparisons;
and we measure the validity of short forced answers of the kind that [10]
(48 tokens) and [15] (64 tokens) use to label stops. Neither paper reports
an analysis of forced answers that are cut off at the budget.

**Training-free early exit.** Several methods decide online when a
reasoning model may stop. DEER [5] treats transition words such as
"Wait" as candidate exit points, induces a trial answer there, and exits
when the answer's confidence exceeds a threshold (0.95 by default).
Dynasor [4] probes the chain of thought at a fixed token interval by
appending an answer prompt, and stops once recent probed answers agree.
ThinkBrake [18] compares, at sentence boundaries, the log-probability of
the end-of-thinking token with that of the most likely token. EAT [6]
thresholds the entropy of the next token after an appended
end-of-thinking token. NEAT [8] monitors the activations of neurons
associated with exiting. LYNX [9] trains a probe on hidden states at cue
words, with labels obtained from forced exits, and wraps its scores in
split conformal prediction to control premature exits.

These methods take their candidate points from the generated text or at
fixed intervals, not from a known change in the answer. The leak we
report (§4.2) concerns evaluation designs that place candidates around
that change, as our registered design did; we do not claim that these
methods are affected by it. DEER, Dynasor and LYNX use answers forced at a
candidate point, for the exit decision or for training labels. We did not
check the budget each uses, and make no claim about them, but our
measurement (§4.3) implies that such results should report the forcing
budget and the rate at which forced answers are cut off. ThinkBrake and
EAT require an end-of-thinking delimiter, which the instruction-tuned
model studied here does not have, so they are not baselines in this
study. LYNX's conformal control of premature exits is close in purpose to
the guaranteed-risk threshold calibration of our exploratory
stopping-policy pilot (§3.9).

**Early answering.** Truncating a chain of thought and forcing an answer
was introduced to test whether the stated reasoning is load-bearing [14].
Recent work on belief tracking uses the same instrument with 11-token
multiple-choice answers [19]. With free-form mathematical answers, a short
budget often cuts off the derivation that the forced completion begins.
We show that such answers are reproducible but not valid before the tail
[X], and we measure the consequence for a stopping comparison [S].

**Verbalising activations.** NLAs [12] train a verbaliser and a
reconstructor jointly, so that the text must carry enough information to
rebuild the activation. Other decoders answer questions about activations
and report gains over generic linear probes [20]. Li et al. [13] show
that verbaliser benchmarks can often be solved from the input text alone
and that verbalisations can reflect the verbaliser's own knowledge; in one
experiment they train a linear probe on the same activations for the same
labels and find it roughly on par with the verbaliser. Work on
confabulation and on faithful verbalisation questions what
reconstruction-trained explanations contain [21, 22]. We apply Li et al.'s
same-activation, same-label control to an NLA and to a reasoning-state
label: the NLA text outperforms the raw input text, but not that probe
(§4.4).

**Faithfulness of stated reasoning.** A model's stated reasoning can
diverge from what drives its answers [23, 24]. We make no faithfulness
claim about NLA explanations; our evidence is predictive and string-level
(§6).
