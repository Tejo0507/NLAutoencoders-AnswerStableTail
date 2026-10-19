# Abstract

*Final. Summarises §6 and §7; every number is from the main run's certified artifacts.*

Reasoning models often keep writing after their answer is settled. We ask
whether a natural-language autoencoder's verbalised readout of layer-20
activations tells us anything about that region that cheaper signals do not.

We first define an Answer-Stable Tail from behaviour alone: truncation
equivalence, agreement of resampled continuations, and persistence. We then
compare the readout with answer convergence, a hidden-state correctness probe
and semantic entropy, as stopping rules at matched token budget, using a
pre-registered problem-level test. The design adds a claim-level audit with a
paraphrase null and a verbaliser-only control, a causal battery with matched
controls, and ten falsification tests.

On Qwen2.5-7B-Instruct, run in 4-bit, over 250 GSM8K and MATH problems,
tails occur on 0.948 of problems non-trivially. They cover a mean 0.378 of
each trace (a lower bound) and begin before the answer is stated.

On the 120 problems the readout covers, its safe-stopping rate is lower than
semantic entropy's (−0.092, 95% CI [−0.184, 0.016], q = 0.115) and the
probe's (−0.008, q = 1.0). Neither difference is significant, and the
comparison with answer convergence could not be made at a shared budget. The
probe itself is largely positional.

Of 6,007 claims, 0.749 are reconstruction-dependent and not reproduced from
noise. An exploratory independent-probe control did not corroborate that
reading, and reconstruction fidelity is no higher in the tail than elsewhere.
The causal question is not tested: the registered rechecking readout has
essentially no range on this model (2 of 274 traces).

On this corpus the verbalised readout did not demonstrate value beyond the
cheaper signals.
