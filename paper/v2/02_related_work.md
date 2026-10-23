# 2. Related work (draft v3, 2026-10-23)

*Built from `docs/NOVELTY_REVIEW_2026-10-20.md` (abstracts and five full
texts) and the 2026-10-23 sweep of training-free early-exit methods
(DEER, Dynasor, ThinkBrake, EAT, NEAT, LYNX; references [33]–[38]). This
is a focused review, not a systematic one. Each sentence saying where our
work differs is phrased as a difference from the papers named, not as a
claim that nobody has done it. Findings attributed to other papers are
theirs; ours are marked with their evidence class.*

**Overthinking and answer stability.** Reasoning models often continue
after their answer is settled [7, 8, 9].
- Answer convergence, in which successive intermediate answers agree, has
  been proposed as a stopping signal [5].
- Mo et al. [22] show in a preregistered sweep that agreement establishes
  only that an answer *persists under a fixed probing procedure*, not that
  reasoning has finished.

Our answer-stable tail is defined by the same kind of agreement, and we
reach the same reading: the tail marks where the answer becomes recoverable
within a budget. We add a measurement of how that budget itself changes the
outcome (§4.3).

**Early exit from hidden states and calibrated stopping.**
- Probes on hidden states predict whether an intermediate answer is correct
  and can trigger an exit [6].
- Probes can also detect a commitment boundary after which the steps are
  epiphenomenal [31].
- Thought Calibration [29] controls the risk of early termination with
  Learn-then-Test.
- Dong et al. [28] compare confidence, entropy, stability and learned
  stoppers at a fixed lost-correct risk. They show that savings depend on
  how the cost of the stopping signal is counted.

We adopt their two lessons: charge the signal's cost, and compare at matched
risk. We do not claim them. Where we differ from these papers:
- our probe baseline is trained for the *same* label as the signal under
  test;
- we report a failure mode of matched-*mean*-saving comparisons;
- we measure the validity of the forced answers that both [28] (48 tokens)
  and [31] (64 tokens) use to label stops. Neither paper reports a
  truncation analysis.

**Training-free early-exit methods.** Several methods decide online when a
reasoning model may stop, each from a different signal:
- **DEER** [33] treats transition words such as "Wait" as candidate exit
  points. There it induces a trial answer and exits when the answer's
  confidence (token probabilities, threshold 0.95 by default) is high.
- **Dynasor** [34] probes the chain of thought at a fixed token interval by
  appending an answer prompt, and stops once recent probed answers agree.
- **ThinkBrake** [35] compares the log-probability of the end-of-thinking
  token with that of the top token at sentence boundaries.
- **EAT** [36] thresholds the entropy of the next token after an appended
  end-of-thinking token.
- **NEAT** [37] tracks the activations of neurons associated with exiting.
- **LYNX** [38] trains a probe on hidden states at cue words, with labels
  from forced exits, and wraps its scores in split conformal prediction to
  control premature exits.

Three points of contact with this paper:
- **Candidate points.** The methods above take candidate points from the
  generated text (transition words, sentence boundaries, cue words) or at
  fixed intervals, not from a known answer change. The menu leak we report
  (§4.2) concerns evaluation designs that place candidates around the
  answer change, as our registered design did. We do not claim that these
  methods are affected by it.
- **Forced answers.** DEER, Dynasor and LYNX use answers forced at a
  candidate point, for the exit decision or for training labels. We did not
  check the budget each uses for those answers, so we make no claim about
  them. Our measurement (§4.3) bears on any such design. With a short
  budget, many forced answers are cut off before they finish. So the budget
  and the truncation rate should be reported with the result.
- **Applicability here.** ThinkBrake and EAT need an end-of-thinking
  delimiter. The instruction-tuned model we study has none, so they are not
  baselines in this study. LYNX's conformal control of premature exits is
  close to the risk control that our proposed stopping-policy study would
  use (§5.4).

**Early answering as a faithfulness probe.** Truncate-and-answer was
introduced to test whether chain-of-thought reasoning is load-bearing [16].
It is used again in recent work on belief tracking [30], whose forced
answers are 11-token multiple-choice letters. With free-form maths answers,
a short forcing budget often cuts off the derivation the forced completion
starts. We show that such answers are reproducible but invalid before the
tail.

**Verbalising activations.**
- NLAs [10] train a verbaliser and a reconstructor jointly, so that the text
  must carry enough information to rebuild the activation.
- Other decoders answer questions about activations [32], and report gains
  over generic linear probes.
- Li et al. [12] show that verbalizer benchmarks can often be solved from
  the input text alone, and that verbalizations can reflect the verbalizer's
  own knowledge. They include a linear probe on the same activations for the
  same labels in one experiment, and find it roughly on par with the
  verbalizer.
- Work on confabulation and on faithful verbalization [13, 14] questions
  what reconstruction-trained explanations contain.

We apply Li et al.'s control, a same-activation, same-label probe, to NLAs
and to a reasoning-state label. The NLA text beats the raw input text, but
not that probe.

**Faithfulness of stated reasoning.** Models' stated reasoning can diverge
from what drives their answers [15, 17]. We make no faithfulness claim
about NLA explanations. Our evidence is predictive and string-level (§6).
