# 3. Methods

*Revised candidate, 2026-10-24. Written from the code, the configurations
and the frozen plans.*

Decision numbers (D48, D69, …) refer to entries in the project's public
decision record (`docs/DECISIONS.md` in the code repository), which holds
every plan hash, deviation and output hash.

## 3.1 Model, data and traces [R]

The target model is Qwen2.5-7B-Instruct [25], quantised to 4-bit NF4 [26].
Problems come from GSM8K [27] and from the algebra and counting-and-
probability subjects of MATH [28]. For each problem the model generates one
greedy trace of at most 768 tokens, which is split into chunks at
reasoning-step boundaries; a boundary *s* defines the prefix *P_s*, the
trace up to that boundary. Table 1 gives the setting of each run.

Three corpora are analysed. The **main run** (`main`) has 250 problems
(150 GSM8K, 100 MATH); its NLA arm covers 120 of them (D48). An
exploratory **240-problem arm** (`main_nla240`, D54) extends the NLA arm
and contains the main run's 120 problems, so it is not an independent
sample. The **replication** (`rep1`, D75) uses 400 problems (240 GSM8K,
160 MATH) that no earlier run had used, selected in the registered hash
order after excluding every problem identifier used before.

## 3.2 The answer-stable tail and the safety outcome [R]

At each boundary that the tail scan evaluates, an answer is forced from
*P_s* by appending "Therefore, the answer is" and decoding greedily for 48
new tokens, and K = 5 continuations are sampled from *P_s* (temperature
0.8, 200 tokens). A boundary *i* starts the answer-stable tail if (1) the
forced answer is equivalent to the trace's final answer, (2) all five
continuations reach an answer equivalent to the final answer, and (3)
both conditions hold at every later boundary. The tail start *i\** is the
smallest such boundary, and the answer-stable tail is the set of
boundaries *i* ≥ *i\**. Traces without such a boundary are recorded as
having no stable point, not dropped.

The registered safety outcome, **O48**, labels a stop at *s* as *safe* if
the answer forced from *P_s* with 48 tokens has the same correctness as
the trace's final answer (D69). Forced answers at boundaries the scan did
not reach were completed with the same call. Before any outcome was used,
the forcing instrument had to reproduce stored forced answers on at least
95% of re-forced boundaries (on `rep1`: 200 boundaries, D80).

## 3.3 Signals and stopping rules

**NLA readout [R].** At each verbalised boundary, the NLA verbaliser
produces two explanations (at most 256 tokens each) of the layer-20
activation, and the reconstructor maps each back to an activation. The
registered readout score is the mean reconstruction cosine over samples
that pass an integrity gate (well-formed English). On the main run,
explanations of matched-norm Gaussian vectors serve as a noise control.

**Probes.** The *registered probe* [R] is a logistic regression on
layer-20 activations that predicts whether the intermediate parsed answer
at a boundary is correct, fitted out of fold. The *same-label probe* [P,
pre-specified for `rep1`] is a logistic regression on the same layer-20
activation, trained for the label being predicted, with its regularisation
chosen by inner problem-grouped cross-validation (D81 deviation 1). The
two probes answer different questions and are never interchanged.

**Position and no-signal rules.** Position rules score a boundary by its
prefix length, rising or falling [X, D79]. The no-signal rule assigns
independent uniform scores, with its expected curve computed in closed
form [R].

**Stopping rule.** Every rule stops at the first allowed boundary whose
score reaches a threshold *t*. Its *saving* on a problem is 1 − (prefix
tokens)/(trace tokens), and 0 if it never stops.

**Text features [X, D81].** Explanations, and as a control the last 600
characters or the whole of the raw prefix, are represented with TF-IDF
features over word unigrams and bigrams and modelled with logistic
regression.

## 3.4 Candidate points

**Design A** [P on the main run, D69; registered for the replication by
D75] allows every rule to stop only at the verbalised
chunks, the *menu*. The menu contains three kinds of window: the tail, a
window matched for position that ends at *i\** − 1, and a window matched
for length. Every menu therefore contains the chunks on both sides of the
change in outcome, and is derived from the outcome (Fig 1) [A].

Two designs remove this dependence. **Stage 0** [X, D83] uses every chunk
boundary of the main run as a candidate, which is possible only for
signals defined at every boundary. **Stage 1** [S; specified in D83,
approved as revised in D85] verbalises K = 4 boundaries in each of the
400 replication problems (1,600 units), drawn uniformly without
replacement by a seeded function (seed 20261020) that receives only the
problem identifier and the number of chunks. It never reads *i\**, a
forced answer or a continuation. Every Stage 1 unit is forced with 256
tokens.

## 3.5 Estimands and inference

**Registered matched-saving estimand** [R; its budget grid and margin
follow D69 and were fixed for the replication by D75]. Each rule's curve
of safe rate against mean saving is matched to
budgets of 0.1, 0.2, …, 0.9 times the oracle bound *B\**, the mean over
problems of the largest saving at a safe menu boundary (*B\** = 0.414 on
`rep1`), with a tolerance of 0.02. The safe rate counts a problem on which
the rule never stops as safe. The test statistic is the mean safe-rate gap
over matched budgets; its 95% interval and p-value come from a paired
problem bootstrap (2,000 replicates, seed 20261019), with Holm's
correction [29] over the two hypotheses: H1, readout minus registered
probe, and H2, readout minus no signal. Bootstrap p-values are two-sided;
the smallest attainable value is 0.0005. The plan's readings are
mechanical: *supported* if Holm-significant with a positive gap;
*reversed* if Holm-significant with a negative gap; *equivalent within
±0.05* if the 95% interval lies inside (−0.05, 0.05); *advantage of 0.05
or more excluded* if only the upper bound is below 0.05; otherwise
*inconclusive*. The same readings are applied to the 256-token
sensitivity outcomes (§3.6).

**Position audit [S; D79].** The same estimand compares the readout with
the falling-position rule and a position-residualised readout (its
linear-in-log-position trend removed). Its two families of comparisons
have fixed readings, specified before any `rep1` outcome. The comparisons
involving the falling-position rule matched 8 of the 9 budgets.

**Information [X on the earlier arms; S on `rep1`; D81].** Each signal's
out-of-fold AUC is estimated with 10-fold, problem-grouped,
outcome-stratified cross-validation. Contrasts are differences in AUC,
with problem-cluster bootstrap intervals (2,000 draws). Readings compare
the 95% interval with a margin of ±0.02 AUC. The primary Stage 1 contrast
[S] is **Δ₁ = AUC(position + same-label probe + text) − AUC(position +
same-label probe)** on the 256-token safety label, read against a margin
of ±0.01 AUC in the manner of two one-sided tests [30]. The 48-token label
and tail membership are secondary, and five further cross-validation seeds
check the primary contrast's stability.

**Selective risk [P on the earlier arms; S on `rep1`; D83 deviation 1].**
Among problems on which a rule stops before the end, selective risk is the
share of unsafe stops. It is reported at coverages of 25%, 50% and 75% of
problems, together with the mean saving among stops. Risk and saving are
reported side by side and not combined into a ranking.

## 3.6 The forcing-budget measurement [S on `rep1`; X on the 240-problem arm; D82]

Every menu boundary whose 48-token forced answer was cut off before it
finished was re-forced with the same call at 256 tokens. A stability gate
required 50 finished boundaries to reproduce their answers at 256 tokens
with agreement of at least 0.95. The statistic *f* is the share of
pre-tail cut-off boundaries whose 256-token answer equals the final
answer. The plan fixed its interpretation in advance: *f* ≥ 0.80 means
that the 48-token outcome mostly measured whether the model could commit
within 48 tokens. The sensitivity outcomes are O256 (the 256-token
answer, as parsed), O256-u (an answer still cut off at 256 tokens counts
as no answer) and the continuations' majority (diagnostic). The plan also
compared recorded forced-token counts with re-tokenised counts (±2
tokens); this check failed on both arms (maximum differences of 4 and 5
tokens), so net savings, which charge the forced tokens, are not used, and
every 256-token comparison uses registered savings. **O256 never replaces
O48**: the confirmatory verdict is computed on O48 alone.

## 3.7 A rewording null for reconstruction dependence [X; D76]

The registered RQ2 analysis calls a claim in an explanation
*reconstruction-dependent* when deleting it moves the reconstruction more
than the 95th percentile of a per-explanation paraphrase null. The
registered paraphrases are mostly unchanged text or an added discourse
marker (D70). D76, specified before any rewording was generated, tests a
stronger null on a seeded sample of 60 explanations from the main run (20
per window kind), with every claim of each. The target model rewrote each
claim three times at temperature 0.7, instructed to keep the meaning,
every number, every quoted string and every technical term. A rewording
was rejected if it was empty or multi-line, identical after normalisation,
different in its numbers, missing a quoted string, outside a word-count
ratio of 0.6 to 1.67, or changed on fewer than 25% of tokens. The
reconstructor then scored each explanation with one claim replaced by an
accepted rewording, after re-scoring every original explanation and
requiring agreement with the stored cosines within 0.01. These rules check
form, not meaning; no rewording was judged by a person.

## 3.8 Pre-registration, deviations and evidence classes

Every analysis reported here had a plan whose SHA-256 hash was recorded
before the analysis was computed; deviations are appended to each plan
with their dates, and Table 1(b) gives the timeline. The confirmatory
analysis refuses to run on any plan other than the frozen one. Results are
marked with one evidence class:

- **[R] registered:** in the original pre-registered plan, computed as
  planned;
- **[C] confirmatory:** fixed before the replication's data existed (D75)
  and computed as planned (D94);
- **[S] pre-specified secondary or sensitivity:** fixed before any
  replication outcome, outside the confirmatory verdict, never replacing
  it;
- **[X] exploratory:** specified before computation, on data whose other
  results were already known;
- **[P] post hoc:** an analysis added after related results were seen;
- **[A] analytic:** follows from the construction of the code or from
  measured costs.
