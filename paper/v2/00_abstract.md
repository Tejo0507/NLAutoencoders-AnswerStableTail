# Do verbalised activations know when reasoning is done? A pre-registered test of a natural-language autoencoder as an early-stopping signal

*Revised candidate, 2026-10-24; the stopping-policy pilot added
2026-10-25 (D102). Awaiting author review; not frozen. Every number is
bound to a stored output in `paper/v2/evidence/key_numbers.json` and
checked by `scripts/verify_manuscript_numbers.py`.*

## Abstract

Natural-language autoencoders (NLAs) translate a language model's internal
activations into text from which the activation can be reconstructed. We
asked whether such a readout signals when a model's answer has settled
during step-by-step reasoning, and whether it adds anything to cheaper
signals. We studied Qwen2.5-7B-Instruct and its released layer-20 NLA on
GSM8K and MATH problems. In a pre-registered confirmatory replication on
400 previously unused problems, both registered hypotheses were supported
on the registered outcome, a 48-token forced answer. At matched mean token
saving, the readout's safe rate (the share of problems on which the rule's
decision preserved the correctness of the final answer, with abstention
counted as safe) exceeded that of the registered correctness probe by
0.047 (95% CI 0.019 to 0.075) and that of a no-signal rule by 0.057 (0.039
to 0.076). Two pre-specified analyses qualify this
result. A rule that uses only the prefix length beat the readout by 0.025
(0.005 to 0.044), so the advantage cannot be attributed to the content of
the explanations. And 86.7% of the pre-tail forced answers that were cut
off at 48 tokens reached the final answer when allowed 256 tokens; under
that 256-token outcome, neither hypothesis was supported, and both
differences were equivalent to zero within ±0.05. The explanation text
does predict stopping-related labels. At 1,600 positions in 400 problems
chosen without reference to the outcome, however, its increment over
position and a linear probe trained for the same label on the same
activation was 0.00002 AUC (−0.004 to 0.004), equivalent within the
pre-specified ±0.01 margin. In an exploratory pilot, a stopping policy
built on that probe was calibrated on 300 fresh problems and evaluated on
150 others. It saved 5.5% of tokens with an observed unsafe-stop rate of
4.1%, but the one-sided 95% upper bound on that rate, 7.8%, exceeded the
5% target, so its pre-specified safety criterion was not met; a
position-only policy could not be calibrated at all. The findings about
the explanations concern prediction and incremental value; we did not test
whether the explanations are faithful to the computation, and no stopping
rule studied here is shown to be safe to deploy. We also show how outcome-derived candidate points, an estimand
that rewards abstention, and a short forcing budget can each manufacture a
stopping advantage.
