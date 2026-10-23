# 2. Related work (draft v2, 2026-10-21)

*Provisional.* Built from `docs/NOVELTY_REVIEW_2026-10-20.md`: abstracts and
five full texts. The remaining checks are listed in #31. Each sentence
saying where our work differs is phrased as a difference from the papers
named, not as a claim that nobody has done it.

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
outcome (§7).

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
