# Abstract (draft v1, 2026-10-23)

*Every number is from `paper/CLAIM_EVIDENCE_MATRIX.md` §R; classes as in
§4.*

Natural-language autoencoders (NLAs) describe a model's activations in
text. We ask whether such a readout tells when a reasoning model's answer is
settled, beyond cheaper signals, for Qwen2.5-7B-Instruct with its released
layer-20 NLA on GSM8K and MATH. In a pre-registered confirmatory
replication on 400 unseen problems, both registered hypotheses were
supported on the registered 48-token forced-answer outcome. The readout
beat the registered correctness probe by +0.047 [0.019, 0.075] and a
no-signal rule by +0.057 [0.039, 0.076] in safe-stop rate at matched
saving. Two pre-specified analyses qualify this. First, a rule that sees
only the prefix length beat the readout by 0.025 [0.005, 0.044], so the
advantage cannot be attributed to explanation content. Second, 86.7% of
pre-tail answers cut off at 48 tokens reached the final answer when given
256. Under that 256-token outcome the registered hypotheses were not
supported: both differences were equivalent to zero within ±0.05, with
smaller effects not excluded. The explanation text does predict
stopping-related labels, and a phase vocabulary transfers to unseen
problems. But at positions chosen without the outcome, its increment over
position plus a linear probe on the same activation, trained for the same
label, was +0.00002 [−0.004, 0.004] AUC, equivalent within ±0.01. These
are results about prediction and incremental value. Faithfulness of the
explanations was not tested. Selective-risk results on the
outcome-selected candidate menu are not deployment evidence. We show how
outcome-derived candidate points, a matched-mean-saving estimand and a
short forcing budget can each manufacture a stopping advantage, and
recommend outcome-blind candidates, risk and saving reported together,
reported forcing budgets and same-label probe baselines.
