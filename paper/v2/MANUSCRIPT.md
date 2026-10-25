# Do verbalised activations know when reasoning is done? A pre-registered test of a natural-language autoencoder as an early-stopping signal

> **Status:** Revised candidate, awaiting author review. Not approved, not frozen.

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

# 1. Introduction

Language models that reason step by step often keep generating after the
answer they will give has already settled [1, 2]. Stopping earlier saves
compute, but only if a signal can tell that the answer is settled without
changing it. Proposed signals include agreement between intermediate
answers [3, 4], the confidence of a trial answer [5], the entropy of the
next token after an end-of-thinking marker [6], probes on hidden states
or neurons [7, 8, 9], and calibrated or learned stopping rules [10, 11].
Natural-language autoencoders (NLAs) offer a different kind of signal [12].
An NLA verbaliser describes an activation in text, and a jointly trained
reconstructor rebuilds the activation from that text. If the description
of an activation says that the model is concluding rather than setting
up, it could indicate when to stop, and it would also say why.

We ask whether an NLA readout provides information about when a model's
answer has settled **beyond what cheaper signals already provide**. We
study Qwen2.5-7B-Instruct with its released layer-20 NLA on GSM8K and MATH
problems, first under a pre-registered plan and then in a confirmatory
replication on 400 problems that no earlier analysis had used.

**Findings.** On the registered outcome, a 48-token forced answer, the
replication supported both registered hypotheses: the readout beat the
registered correctness probe and a no-signal rule [C]. Two pre-specified
analyses qualify that result. A rule that sees only the length of the
prefix beat the readout, and once the readout's position trend was
removed it no longer beat the no-signal rule, so the advantage cannot be
attributed to what the explanations say [S]. When the forcing budget was
raised to 256 tokens, most pre-tail answers that had been cut off reached
the final answer, and neither registered hypothesis was supported under
that outcome [S]. The registered 48-token verdict stands as a result about
commitment within 48 tokens. The explanation text does carry predictive
information about the reasoning phase, concluding as against setting up
[X, S]. At positions chosen without reference to the outcome, however, it
adds no practically relevant predictive information beyond position and a
linear probe trained for the same label on the same activation [S]. In an
exploratory pilot on fresh problems, a stopping policy built on that probe
saved tokens, but its unsafe-stop rate could not be bounded below the
pre-specified target [X]. We do not test whether the explanations are
faithful to the computation.

**Contributions.** Prior work establishes that stopping signals should be
charged for their compute and compared at matched risk [10, 11], that
hidden-state probes predict the correctness of intermediate answers [7],
that evaluations of activation verbalisers need input-only and
same-activation probe controls [13], that truncating and forcing an answer
reveals when an answer commits [14, 15], and that agreement between
answers shows persistence rather than finished reasoning [16]. We claim
none of these. Within the focused review of §2, this paper contributes:

1. **A pre-registered, confirmatory test of an NLA readout as a stopping
   signal, reported with its pre-specified qualifications** (§4.1–§4.3).
2. **A measurement of the forced-answer instrument's dependence on its
   budget.** At 48 tokens, 54–56% of pre-tail forced answers are cut off on
   the earlier arms, and the registered conclusion changes when the budget
   is raised to 256 tokens (§4.3, §4.6).
3. **Two demonstrated ways in which an evaluation can manufacture a
   stopping advantage:** candidate points derived from the outcome, and a
   matched-mean-saving estimand that counts abstention as safe (§4.2,
   §4.6).
4. **An incremental-value test of NLA text against a same-label probe on
   the same activation at outcome-blind positions**, using the control of
   Li et al. [13] for a reasoning-state label (§4.4).
5. **An exploratory, prospectively calibrated pilot of a probe-based
   stopping policy on fresh problems.** It did not meet its safety
   criterion, and it supplies the out-of-sample risk, coverage and saving
   estimates that a confirmatory study would need (§4.9).

**Scope.** All claims concern one target model, one layer and one released
NLA. They concern prediction and incremental value, not faithfulness.
Selective-risk results are measured at outcome-selected candidate points
and are not evidence that any stopping policy is safe to deploy (§4.5,
§5.4). The one out-of-sample estimate, from the pilot, is not a safety
certification (§4.9). Cost-aware and risk-controlled evaluation are not
new [9, 10, 11].
The contribution is a carefully qualified negative result about NLAs as
stopping signals, together with measurements of the evaluation choices
that would otherwise have suggested a positive one.

# 2. Related work

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

# 3. Methods

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
order after excluding every problem identifier used before. Its frozen plan
fixed N = 400 before any outcome existed. From the main run's standard
error, the plan expected a 95% interval half-width of about 0.028 for H1,
and power of about 0.45, 0.80 and 0.98 for true gaps of 0.027, 0.040 and
0.057. It also fixed that with fewer than 200 problems with a tail, the
result would be reported as an underpowered replication rather than a
confirmatory test.

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
- **[X] exploratory:** specified before computation, either on data whose
  other results were already known or, for the stopping-policy pilot
  (§3.9), on fresh data under a frozen protocol that designates the study
  exploratory;
- **[P] post hoc:** an analysis added after related results were seen;
- **[A] analytic:** follows from the construction of the code or from
  measured costs.

## 3.9 A stopping-policy pilot on fresh problems [X; D99–D102]

The analyses above score signals at stored candidate points, and none
estimates how a stopping policy behaves on problems it was not tuned on. A
separate pilot asked that question for cheap signals. Its protocol, with
every threshold rule, sample size, gate and decision criterion, was frozen
and hashed before any pilot problem was selected (D99). The protocol
designates the pilot exploratory: it estimates the quantities needed to
size a confirmatory study and was not designed to certify safety. The NLA
is not part of it, because one NLA check costs almost a whole trace
(§4.6).

**Policies.** Each policy walks every chunk boundary of a trace in order
and stops at the first boundary whose score reaches its threshold τ. P0
(position only) is a logistic regression on the log prefix length, the
chunk index and a dataset indicator; both fitted position coefficients are
positive, so its score rises along every trace. P1 adds the out-of-fold
logit of a same-label linear probe on the standardised layer-20 activation
(§3.3). Both were fitted on the replication's 1,600 Stage 1 units with
their 256-token safety labels, with the probe's regularisation chosen by
problem-grouped cross-validation, and frozen before calibration.

**Outcome and saving.** A stop at *s* is safe if the answer forced from
*P_s* with 256 tokens has the same correctness as the trace's final
answer. A forced answer still cut off at 256 tokens counts as unsafe (the
primary rule); the as-parsed and no-answer conventions of §3.6 are
pre-specified sensitivity analyses. The net saving of a stop is
max(0, 1 − (prefix tokens + forced-answer tokens)/(trace tokens)), with
token counts recorded at generation. The probe's own computation is not
charged, and a problem on which a policy never stops saves 0.

**Data and calibration.** The calibration set (300 problems: 180 GSM8K,
120 MATH) and the evaluation set (150: 90 and 60) were drawn in the
registered hash order from problems that no earlier run had used, so they
are disjoint from each other and from the training problems. Traces were
generated as in §3.1, and every boundary was forced at 256 tokens. Each
policy's threshold was chosen on the calibration set by Selection with
Guaranteed Risk [31]. This is a binary search in k = 8 steps over 200
quantiles of the policy's calibration scores. A threshold is accepted if
the one-sided Clopper–Pearson [32] upper bound on its selective risk, at
confidence 1 − 0.05/8, is at most 0.05, and τ is the most permissive
accepted threshold. A policy with no accepted threshold never stops. The
thresholds were frozen and hashed before any evaluation problem was
forced, and the code refuses to run the steps in any other order.

**Evaluation and decision criteria.** On the evaluation set we report each
policy's coverage, its selective risk with a two-sided 95%
Clopper–Pearson interval and a one-sided 95% upper bound, and its mean net
saving. The paired P1 − P0 difference in net saving has a problem
bootstrap interval (2,000 replicates, seed 20261024). The researcher set
the maximum acceptable unsafe-stop rate to α = 0.05 for this pilot only,
since no deployment use case had been defined, and the minimum worthwhile
saving to 0.02. The protocol fixed in advance that a confirmatory study is
warranted only if (1) the evaluation upper bound on P1's selective risk is
below α and (2) P1's coverage is materially above 0 and above P0's.
Before any outcome was used, re-forcing a seeded sample of 50 boundaries
had to reproduce their stored answers with agreement of at least 0.95, and
at least 75% of the planned boundaries had to be forced.

**Deviations.** One deviation was recorded before any threshold, safety
label or evaluation problem had been read (D100). The re-forcing check had
re-forced each sampled boundary alone, although the stored answers had
been forced in pairs, and it failed. It was redone with each boundary
re-forced in its original pair, the gate unchanged, and the failed result
is reported (§4.9). Two further departures were found after the run and
are recorded in D102. First, the protocol's list of instrument checks
included a 256-token stability check carried over from §3.6, where it
compares answers across two budgets; the pilot forces at a single budget,
so the check had no counterpart and was not run. For an answer that
finished within its budget, a larger budget cannot change the greedy tokens
generated before the end token in the same batch [A]. The remaining risk
that the check guarded against, a forced answer that changes between
calls, is what the re-forcing check measures. Second, the evaluation
code skipped the pre-specified paired saving because P0 had no threshold.
That estimand was computed separately with the same functions, the frozen
threshold and the protocol's seed.

# 4. Results

Three distinctions are kept throughout this section: *prediction*, whether
a signal carries information associated with an outcome; *incremental
value*, whether it adds anything beyond cheaper signals; and
*faithfulness*, whether an explanation describes the computation, which is
not tested. We first report the confirmatory replication (§4.1) and the two
pre-specified analyses that qualify it (§4.2, §4.3), then what the
explanation text carries (§4.4) and selective risk (§4.5). We then show,
on the earlier main run, how evaluation choices can manufacture a stopping
advantage (§4.6), report the 256-token pass on the 240-problem arm (§4.7)
and the rewording null (§4.8), report the exploratory stopping-policy
pilot on fresh problems (§4.9), and summarise (§4.10).

## 4.1 The registered replication [C]

The replication ran the registered design on 400 problems that no earlier
analysis had used (§3.1, Table 1). Of these, 380 had an answer-stable tail,
against the plan's minimum of 200. The forcing instrument reproduced 0.965
of 200 re-forced stored boundaries, above the registered threshold of
0.95, and the analysis verified the frozen plan's hash before running.

Both registered hypotheses were supported on the registered 48-token
outcome (Table 2a; Fig 2, left). The readout's mean safe-rate gap over the
registered probe (H1) was +0.047 [0.019, 0.075], Holm-adjusted p = 0.002,
and its gap over the no-signal rule (H2) was +0.057 [0.039, 0.076],
Holm-adjusted p = 0.001. The interval for H1 spans the plan's margin of
0.05: an advantage over the registered probe is shown, but an advantage of
0.05 or more is neither shown nor excluded.

## 4.2 First qualification: candidate position [S]

The position audit (Table 2c), specified before any replication outcome,
asks whether the readout's advantage could arise from where the candidate
points lie rather than from what the explanations say. A rule that never
sees an activation, and scores each menu candidate by the negative of its
prefix length so that it prefers early stops, beat the no-signal rule by
+0.089 [0.066, 0.108] and beat the readout by 0.025 [0.005, 0.044]. With
its linear trend in log-position removed, the readout no longer beat the
no-signal rule: +0.007 [−0.016, 0.029] (p = 0.54). By the plan's fixed
rules, both families of position readings are *unresolved*.

The registered advantage therefore cannot be attributed to the content of
the explanations: on this menu, a preference for early positions performs
as well. The menu allows this because it is built from the tail start and
always contains the chunks immediately before and at *i\** (§3.4, Fig 1)
[A]. The same falling rule had already matched the readout on the earlier
data [X]: it beat the no-signal rule by +0.082 on the main run and by
+0.100 on the 240-problem arm, and the readout minus the falling rule was
−0.024 [−0.070, 0.015] on the main run. This is why the audit was
pre-specified for the replication.

## 4.3 Second qualification: the forcing budget [S]

The 256-token pass (Table 2b; Fig 2, left; Fig 3, right) re-forced the 661
menu boundaries whose 48-token answers had been cut off. The instrument
was stable at 256 tokens (agreement 1.00 on 50 finished boundaries). Of
the 600 cut-off boundaries before the tail, 520 (*f* = 0.867) reached the
final answer when allowed 256 tokens; the other 61 cut-off boundaries lay
at or after the tail and do not enter *f*. By the plan's fixed language,
the 48-token outcome mostly measured whether the model could commit to its
answer within 48 tokens.

Under the 256-token outcome, both hypotheses read *equivalent within
±0.05*: H1 −0.008 [−0.036, 0.024] and H2 −0.004 [−0.024, 0.016]. These
intervals exclude effects of 0.05 or more, and the 48-token estimates
(0.047 and 0.057) lie outside them. They do not exclude smaller effects. The registered
hypotheses are therefore not supported under the 256-token outcome, while
the 48-token verdict stands as registered and is to be read as a result
about commitment within 48 tokens. Net savings were not used, because the
token-count check failed (maximum difference 4 tokens against a tolerance
of 2); every comparison uses registered savings.

## 4.4 What the explanation text carries [S]

**At the menu's points on the replication** (D81 and D95), the readout's
scalar score added nothing beyond position and the registered probe:
+0.0005 [−0.0015, 0.0025] AUC. The explanation *text*, however, added
+0.077 [0.063, 0.090] AUC beyond position and the raw trace text. Before
the trace wrote its final answer, the text mentioned that answer no more
often than another problem's answer (−0.002 [−0.016, 0.012]). A phase
lexicon frozen on the 240-problem arm (Table 3) transferred to the unseen
problems, adding +0.091 [0.076, 0.106] AUC beyond position for tail
membership. Its heaviest terms distinguish concluding ("final answer",
"therefore") from setting up ("first", "let's").

These results reproduce, out of sample, what was found on the
240-problem arm (Fig 4): there, the text added +0.073 [0.058, 0.087] AUC
beyond position and trace text [X], and nothing beyond a same-label probe
on the same activation, −0.001 [−0.004, 0.002] [P].

**At positions chosen without the outcome** (Stage 1; Table 2d; Fig 2,
right), 1,598 of the 1,600 units had a passing verbalisation (gate 75%),
and the first 48 tokens of the 256-token forced answers agreed with a
separate 48-token forcing on 0.98 of checked units (gate 0.95). The
primary contrast, the text's increment over position plus the same-label
probe on the 256-token safety label, was Δ₁ = +0.00002 [−0.004, 0.004]:
*equivalent within ±0.01*. Both signals carry information on their own:
the text added +0.071 [0.052, 0.089] beyond position and the same-label
probe +0.102 [0.075, 0.129]. Head to head, the probe was better than the
text by 0.031 [0.015, 0.047]. The same pattern held for the 48-token label
and for tail membership.

The supported conclusion is narrow. For predicting these stopping labels
under the tested design (TF-IDF text features, a linear same-label probe,
one model and one NLA), the explanation text adds no practically relevant
information beyond position and the same-label probe. This does not show
that everything the text says is linearly available in the activation,
and it is a statement about prediction and incremental value, not about
faithfulness.

## 4.5 Selective risk on the candidate menu [S, P]

Selective risk at matched coverage separates the safety of a rule's stops
from the saving they bring (§3.5). These results use the outcome-selected
candidate menu and 48-token labels and are **not evidence about deployed
stopping**. The probe here is the registered correctness probe; no
selective-risk result exists for the same-label probe on the replication.

At 50% coverage on the replication [S], the registered probe's stops were
almost always safe but saved little (selective risk 0.011, mean saving
among stops 0.028). The readout's stops saved more but were mostly unsafe
(0.632 and 0.711), and the no-signal rule lay between them (0.472 and
0.413). No rule dominates on both quantities, and we do not combine them
into a ranking. On the earlier data [P], the readout's stops were already
less safe than those of the no-signal rule: by +0.16 [0.04, 0.26] at 50%
coverage on the main run and by +0.14 [0.05, 0.21] on the 240-problem arm
(Fig 5, left, shows the 240-problem arm).

## 4.6 How evaluation choices can manufacture an advantage (main run)

The following results come from the main run and its 240-problem arm, and
they motivated the replication's design. Their evidence classes differ and
are marked.

1. **The registered comparison was uninformative** [R, read post hoc].
   The registered test compared rules at budgets (0.61 and 0.72) above the
   oracle bound of 0.17, where a no-signal rule beats every compared rule
   (D68). Rescored on the registered outcome [P], the readout beat the
   no-signal rule by +0.052 [0.022, 0.082] with 120 problems, while its
   comparison with the registered probe was unresolved, +0.027 [−0.026,
   0.076] (D69; Fig 6). The amended reanalysis plan's hash was not
   recorded before these data were seen (D91).
2. **Candidate points derived from the outcome leak it** [A]; see §4.2 and
   Fig 1.
3. **A matched-mean-saving estimand rewards concentrating savings** [X].
   At every boundary of the main run, with no menu (Stage 0; 250
   problems), an information-free falling rule beat a model of position
   plus a same-label activation probe whose boundary AUC was 0.861. The
   probe-based rule minus the falling rule was −0.032 [−0.083, 0.009] on
   registered savings and −0.093 [−0.147, −0.043] on net savings (Fig 7).
   Selective risk at matched coverage reverses this ranking (Fig 5,
   right).
4. **Short forcing budgets give answers that are reproducible but not
   valid before the tail** [X]. Before the tail, 54% (main run) and 56%
   (240-problem arm) of 48-token forced completions were cut off, and
   99% and 98% of those disagreed with the majority of the 200-token
   continuations (Fig 3, left and centre). The 256-token pass measured the
   consequence on the replication (§4.3).
5. **Cost** [A]. One NLA check generates two verbaliser samples of 144.9
   tokens on average, 0.89 of a mean 324.8-token trace, before
   reconstruction. Verbalising at each candidate point therefore cannot
   save compute in this setting.

## 4.7 The 256-token pass on the 240-problem arm [X]

The same pass on the 240-problem arm reproduces the replication's pattern.
It is exploratory, because that arm's 48-token results were already known,
and it is not independent, because it contains the main run's 120
problems. Here *f* = 0.881. The readout minus the registered probe went
from +0.056 [0.023, 0.091] at 48 tokens to +0.001 [−0.031, 0.040] at 256
tokens, and the readout minus no signal from +0.069 [0.049, 0.093] to
−0.011 [−0.034, 0.017]. In both cases the 48-token estimate lies outside
the 256-token interval. Under the 256-token outcome, the text again added
nothing beyond the same-label probe: −0.001 [−0.005, 0.002].

## 4.8 Reconstruction dependence under a rewording null [X]

The registered RQ2 result is that 0.749 of 6,007 explanation claims are
reconstruction-dependent under the registered paraphrase null. The
D76 rewording null (§3.7) was run as specified on 60 explanations from 49
problems, with 246 claims. Of 738 generated rewordings, 313 (42.4%) passed
the fixed form checks, most rejections being changed quoted strings; 117
of the 246 claims received at least one accepted rewording, and every
explanation had at least two. Re-scored original explanations matched the
stored cosines exactly.

Accepted rewordings moved the reconstruction about three times as much as
the registered paraphrases (mean absolute effect 0.0045 against 0.0015),
while deleting a claim moved it by 0.0267 on average. On the same 246
claims, the dependent rate was 0.797 [0.729, 0.856] under the registered
null and 0.764 [0.685, 0.834] under the rewording null (problem-cluster
bootstrap). For 0.872 of the 117 claims with a rewording, deletion moved
the reconstruction more than the claim's mean rewording did. If the
accepted rewordings preserve meaning, the reconstruction-dependence rate
is therefore not mainly an artefact of a null that barely edits. Whether
they do is unvalidated, fewer than half of the claims received a
rewording, and reconstruction dependence is not faithfulness.

## 4.9 A stopping-policy pilot on fresh problems [X]

The pilot (§3.9) generated a trace for every calibration and evaluation
problem and forced every planned boundary at 256 tokens: 3,554 boundaries
in calibration and 1,727 in evaluation. The re-forcing check in the
original pairs agreed on 50 of 50 sampled boundaries in each set. Re-forced
alone, the same calibration boundaries agreed on only 0.92 (46 of 50), and
all four disagreements were boundaries originally forced in a pair.
The 256-token forced answer, and with it the safety label, therefore
depends slightly on what it is batched with.

**Calibration.** No threshold was accepted for the position-only policy
P0. The lowest bound in its search, 0.158 against the target of 0.05, came
at a threshold that stopped on 201 problems, 19 of them unsafely, so P0
never stops. For P1 the accepted threshold was τ = 0.945, at which it
stopped on 292 of the 300 calibration problems with 5 unsafe stops (upper
bound 0.047).

**Evaluation.** On the 150 evaluation problems, P1 stopped on 148
(coverage 0.987 [0.953, 0.998]). Six of those stops were unsafe:
3 of 90 on GSM8K and 3 of 58 on MATH. P1's selective risk was
0.041 [0.015, 0.086], and its one-sided 95% upper bound was 0.078, above
α = 0.05. **The pilot's safety criterion was not met.** No stop fell at a
boundary whose 256-token forced answer was still cut off, so the three
truncation conventions give identical results. P1's mean net saving was 0.055 [0.046, 0.065],
above the pre-set minimum of 0.02; because P0 never stops, this is also the
paired P1 − P0 difference. The saving is small because the threshold is
high and P1 stops late [P]: at a mean of 0.861 of the trace's tokens
(median 0.924), with the forced answer adding a further 0.089 on average.
Coverage therefore overstates how often a stop is useful. 11 of the 148
stops saved nothing once the forced answer was charged, so 137 of the 150
problems (0.913) had a stop that saved tokens [P].

**Against the fixed criteria.** P1's coverage is above 0 and above P0's,
so criterion (2) is met. The upper bound is above α, so criterion (1) is
not, and by the protocol's rule a confirmatory study of this policy as
designed is not warranted. The failed criterion is not evidence that P1 is
unsafe: with 148 stops the bound falls below 0.05 only if at most 2 stops
are unsafe, and at the observed rate an evaluation would need roughly
1,500 to 2,000 stops before the bound fell below 0.05 [A]. Nor is it
evidence that P1 is safe: the interval includes risks up to 0.086, and the
evaluation risk (6 of 148) was higher than the calibration risk (5 of
292).

## 4.10 Summary

| question | answer | class |
|---|---|---|
| Does the readout beat the registered probe and no signal on the registered outcome? | Yes: H1 +0.047, H2 +0.057 | [C] |
| Is that advantage due to the explanations' content? | Not shown: a position-only rule does better | [S] |
| Does it hold under a 256-token forcing budget? | Not supported. Both gaps are equivalent to zero within ±0.05; smaller effects are not excluded | [S] |
| Does the text predict stopping labels? | Yes, beyond position and the trace text | [S] |
| Does it add predictive information beyond a same-label probe on the same activation? | No practically relevant amount: equivalent within ±0.01 at outcome-blind positions | [S] |
| Which signal stops more safely on the candidate menu (48-token labels; not deployment evidence)? | The registered probe, at a much smaller saving; no rule dominates | [S] |
| Are the explanations faithful to the computation? | Not tested | — |
| Is reconstruction dependence robust to a rewording null? | Little change (0.797 to 0.764 on 246 claims), but the rewordings' meaning is unvalidated | [X] |
| Does a calibrated probe-based stopping policy bound its unsafe-stop rate below 0.05 on fresh problems? | Not shown. In a 150-problem pilot: risk 0.041, upper bound 0.078; saving 0.055; a position-only policy could not be calibrated | [X] |
| Is there a safe and useful stopping policy? | Open (§5.4) | — |

# 5. Discussion

## 5.1 What the replication supports

The replication yields one confirmatory result, and it stands as
registered: on the registered 48-token outcome, the NLA readout's safe
rate at matched mean saving exceeded that of the registered correctness
probe and that of a no-signal rule [C]. No later analysis overwrites or
rewords that verdict. Two pre-specified analyses, however, determine how
it can be read.

First, the advantage cannot be attributed to explanation content [S]. A
rule that sees only the prefix length beat the readout, and the readout
with its position trend removed did not beat the no-signal rule. Because
the menu always contains the chunks on both sides of the answer change, a
preference for early positions is enough to look informative on it.
Second, the result depends on the forcing budget [S]. Most pre-tail
answers cut off at 48 tokens reach the final answer when given 256, so the
registered outcome mostly measures whether the model can commit to its
answer within 48 tokens; under the 256-token outcome, the registered
hypotheses are not supported, and effects of 0.05 or more are excluded.
The non-independent 240-problem arm shows the same pattern [X].

The accurate summary is therefore narrow. As registered, on the candidate
menu and when safety means committing within 48 tokens, the readout's safe
rate at matched mean saving exceeds that of the registered probe and of a
no-signal rule. This advantage is not shown to come from what the
explanations say, and it is not supported once forced answers may finish.
It is also not evidence that the readout's stops are mostly safe: at
matched coverage, most of them are unsafe (§4.5).

## 5.2 Prediction, incremental value and faithfulness

The results separate three questions that are easily run together.

**Prediction.** The explanation text carries information associated with
the stopping labels. It adds to position and the raw trace text on the
menu, adds to position at outcome-blind positions, and its most predictive
terms, which describe a concluding as against a setting-up phase, transfer
to unseen problems [S].

**Incremental value.** For these labels, the text adds no practically
relevant information once a linear probe trained for the same label on the
same layer-20 activation is included: the Stage 1 increment is equivalent
to zero within ±0.01 AUC, and the probe is better head to head [S]. The
readout's scalar score adds nothing beyond position and the registered
probe [S].

**Faithfulness.** Whether the explanations describe the computation that
produced the activation is not tested here (matrix E1), and neither of the
first two answers bears on it. Text can predict a label because it
describes the computation correctly or because it reflects correlates of
position and register, and a probe can match the text without the text
being wrong.

Two consequences follow. The absence of an increment does not show that
everything the text says is linearly available in the activation; it shows
that, for predicting these labels with TF-IDF features, nothing practically
relevant is left once a same-label linear probe is used. And the NLA's
value in this setting is interpretive access, a legible description of the
reasoning phase, rather than information that a probe lacks. This is
consistent with Li et al.'s argument for same-activation controls [13],
and it qualifies reported advantages of activation decoders over generic
linear probes [20]: against a probe trained for the same label, no
advantage appeared here.

The reconstruction-dependence rate (0.749 of 6,007 claims) belongs to the
same family of caveats. It shows that deleting most claims moves the
reconstruction more than the registered null does, not that the claims
are true. The exploratory rewording null addresses one weakness of that
null: under model-written rewordings that move the reconstruction about
three times as much, the dependent rate on a sample of claims changes
little (§4.8) [X]. Because the rewordings' meaning was not validated, this
qualifies the rate's robustness; it is not a test of faithfulness.

## 5.3 How evaluation choices manufacture a stopping advantage

The main run's results form a sequence of evaluation choices, each of
which moved the conclusion (§4.6). They suggest four practices for
evaluating stopping signals, none of them new in spirit [10, 11]:

1. **Choose candidate points without reference to the outcome.** A menu
   built around the answer change contains it, and a position rule
   exploits it [A, S]. Stage 1's outcome-blind draw is the remedy used
   here.
2. **Report risk and saving together rather than a single
   matched-mean-saving estimand.** Counting abstention as safe rewards
   rules that concentrate their savings in a few problems: an
   information-free falling rule outranked a probe-based rule with
   boundary AUC 0.861 on the registered estimand, and selective risk
   reverses the ranking [X].
3. **Report the forcing budget and the share of forced answers cut off.**
   Short forced answers are reproducible but often unfinished before the
   tail [X], and the registered verdict changes with the budget [S].
4. **Compare against a same-label probe on the same activation.** The
   registered correctness probe adds +0.026 AUC over position at every
   boundary of the main run, against +0.091 for a same-label probe [X]; a
   weak baseline makes any signal look useful.

## 5.4 What this does and does not say about stopping

The selective-risk results are **not deployment evidence**. They are
measured on the outcome-selected menu with 48-token labels; on it, the
registered probe stops rarely unsafely but saves little, the readout saves
more but mostly unsafely, and no rule dominates [S]. Cost rules out an
online NLA check in this setting: one check generates 0.89 of a mean
trace's tokens before reconstruction [A], so the NLA is at most an offline
research comparator.

**The practical question remains open.** A planning replay on stored
replication data suggested that a same-label probe could stop on far more
problems than a position-only rule at the same unsafe-stop rate, but it
chose its thresholds on the data it scored and walked only four positions
per problem. The exploratory pilot (§3.9, §4.9) removed both weaknesses,
calibrating at every boundary of 300 fresh problems and evaluating on 150
others. Its results separate four findings [X]:

1. **Calibration.** The probe-based policy can be calibrated where the
   position-only policy cannot: no position threshold met the calibration
   bound, so P0 never stops.
2. **Utility.** At its calibrated threshold, P1 stopped on almost every
   evaluation problem and saved 5.5% of tokens, above the pre-set minimum
   of 2%. The saving is modest because the stops come late, and 11 of the
   stops saved nothing once the forced answer was charged.
3. **Safety.** P1's observed unsafe-stop rate, 4.1%, is below the 5%
   target, but 148 stops cannot bound it there: the one-sided upper bound
   is 7.8%. The pilot's safety criterion was not met, and by the
   protocol's own rule a confirmatory study of this policy as designed is
   not warranted. The result is inconclusive about safety in both
   directions; it neither shows that P1 is unsafe nor that it is safe.
4. **The instrument.** The safety label itself depends slightly on batch
   composition, since a few 256-token forced answers changed when they
   were re-forced without their original batch partner.

The pilot is not a safety certification. It did not use the NLA and says
nothing about faithfulness. A follow-up would need its own frozen plan; an
α justified by a use case rather than chosen for a pilot; an evaluation set
disjoint from the pilot's, which is not reused; and either a stricter
calibration target, so that the true risk sits further below α, or an
evaluation large enough to bound a risk near 0.04 below 0.05.

## 5.5 What the answer-stable tail is

The tail marks where the answer becomes *recoverable within a forcing
budget*, not where redundant text after the answer begins: tails begin
before the answer is written, and the measured budget dependence shows how
much of an apparent pre-tail failure is an unfinished derivation. This
agrees with Mo et al.'s reading of answer agreement as persistence under a
fixed probing procedure [16]. A tail defined under a longer budget could
start earlier; that recomputation was not done.

## 5.6 Conclusion

A natural-language readout of a language model's activations passes its
pre-registered stopping test on the registered 48-token outcome. The
advantage cannot be attributed to what the explanations say, and it is not
supported when forced answers are allowed to finish. The explanations
carry legible, transferable information about the reasoning phase, but for
predicting stopping labels they add no practically relevant information
beyond a linear probe on the same activation. Whether they are faithful
is untested. In an exploratory pilot on fresh problems, a stopping policy
built on that probe saved 5.5% of tokens with an observed unsafe-stop rate
of 4.1%, but could not bound that rate below its 5% target, so whether any
signal studied here supports a safe stopping policy remains open. In
this study, the choice of candidate points, estimand, forcing budget and
baseline determined the answer more than the signal did; such choices
should be fixed without reference to the outcome and reported.

# 6. Limitations and threats to validity

**One model, one layer, one NLA.** Every result concerns
Qwen2.5-7B-Instruct in 4-bit NF4 and the released layer-20 NLA; we found
no NLA for another model or layer with which to test generality. The bf16
check covered only the target model's activations, not the verbaliser or
reconstructor. Models that verify in prose, or that end a thinking phase
with a delimiter, may behave differently; the model studied here rarely
rechecks in prose and has no end-of-thinking token, so delimiter-based
signals [6, 18] were not compared. Traces are greedy with a single
generation seed, and the problems come from two public mathematical
benchmarks, which may appear in the model's training data. That would
affect every compared signal alike, but answers might settle earlier than
on genuinely new problems.

**Multiplicity.** Multiplicity is controlled only within pre-specified
families: Holm over the two confirmatory hypotheses, and within the
position audit's and the information audit's families. The many secondary
and exploratory intervals reported alongside them are not adjusted
jointly, so an isolated interval that excludes zero should be read as
descriptive. The paper's conclusions rest on the confirmatory test and on
the pre-specified analyses read by their fixed rules.

**Outcome-selected candidate points and position.** Every NLA stopping
comparison, including the confirmatory one, was made on design A's menu,
which contains the chunks on both sides of the answer change. This is why
the replication's advantage cannot be attributed to content (§4.2).
Stage 1 removes the selection for the information question, but no NLA
stopping *policy* was evaluated at outcome-blind positions.

**Dependence on the outcome and its budget.** The registered outcome is a
48-token forced answer, which is cut off for most pre-tail positions. The
registered verdict uses it and is not replaced, but under a 256-token
outcome the registered hypotheses are not supported, and those intervals
exclude only effects of 0.05 or more. A 256-token outcome still depends on
a budget: 9.1% of the re-forced cut-offs on the replication were still
cut off at 256 tokens. The token-count check failed on both arms, so net
savings were not used under the 256-token outcome, and the tail itself was
not recomputed under the longer budget.

**Prediction is not explanation.** The text features are TF-IDF with a
linear model, and the activation probes are linear. "No practically
relevant increment" means none that these models can extract, within
±0.01 AUC, for these labels; a stronger text model could find more, at a
cost the comparison would have to charge. Predictive information in an
explanation does not show that the explanation describes the computation.

**Limited evidence about faithfulness.** The phase lexicon, the
answer-mention test and the context checks against the trace are
string-level. No person rated the explanations; a two-annotator protocol
exists but was not run, and its blinded annotation sheets are withheld.
The reconstruction-dependence rate rests on a registered paraphrase null
that is mostly unchanged text. The exploratory rewording null changed the
rate little on 60 explanations, but its rewordings were checked for form,
not meaning, and only 117 of 246 claims received one. Reconstruction
dependence is not faithfulness.

**No established safe stopping policy.** The selective-risk results use
the outcome-selected menu and 48-token labels, and none exists for the
same-label probe on the replication's menu. The planning replay chose its
thresholds in sample on four positions per problem. The only out-of-sample
estimate of unsafe-stop risk comes from the exploratory pilot (§4.9), and
it is narrow:
- it covers one probe-based policy, not the NLA;
- it has 150 evaluation problems, from one model under NF4 and two
  benchmarks;
- its α was set for the pilot without a deployment use case;
- its safety criterion was not met;
- its safety label inherits the forcing instrument's limits, a 256-token
  budget and a small batch-composition effect (4 of 36 single re-forcings
  changed);
- its net saving does not charge the probe's computation.

Two of its departures from the protocol were found only after the run
(§3.9, D102). Neither changes a reported estimate.

**Post-hoc analyses and provenance.** The falling-position rule, the
same-label probe, the multiplicity and whole-prefix checks, the unfitted
ensemble and selective risk were added after related results on the main
run were seen. Each was specified before any replication outcome existed,
and the replication is their test. The main run's corrected comparison
(D69) has a provenance gap: the amended plan's hash was not recorded
before the data (D91).

**Independence and power.** The 240-problem arm contains the main run's
120 problems, so its results are exploratory and do not replicate the main
run independently; only the replication's 400 problems are fresh. The
registered H1 had a power of about 0.45 for the gap estimated on the main
run. H1 was supported, but its interval spans the 0.05 margin. Stage 1's
power was audited on proxies from stored data before any Stage 1 data
existed, and its sample was raised to all 400 problems in response.

**Engineering.** Batch sizes for continuations and forcing differ between
the main run (4) and the replication (2). With NF4 weights, what shares a
batch can change a greedy forced answer: in the pilot, 4 of 36 boundaries
forced in a pair and re-forced alone gave a different answer. The forcing
instrument was checked for reproducibility before any outcome was used;
on the replication and in the pilot, the check re-forced boundaries in
batches of their original size (D80, D100).

**A focused review.** The related-work review read abstracts and selected
full texts (§2). Where we say that prior work does not report something,
we mean the papers named.

# 7. Reproducibility statement

**Plans before data.** Every confirmatory, secondary and sensitivity
analysis had a plan whose SHA-256 hash was recorded before the analysis
was computed; hashes, dates and deviations are in the decision record
(`docs/DECISIONS.md`, D66–D102), and Table 1(b) gives the timeline. The
confirmatory script refuses to run on a plan whose hash differs from
D75's.

**From outputs to text.** Every number in §4 is taken from a stored JSON
output. `paper/v2/evidence/key_numbers.json` binds each key number in the
manuscript to its output file and field, and
`scripts/verify_manuscript_numbers.py` recomputes and compares them.
Separately, `scripts/check_run_consistency.py --replication --complete`
re-checks the recorded input hashes, recomputes every reading from the
stored estimates, and checks shared numbers, internal arithmetic and
gates (169 checks). Neither tool re-runs the bootstraps.

**Figures and tables** are generated from the stored outputs by
`scripts/make_paper_figures_v2.py` and `scripts/make_paper_tables_v2.py`
(`--runs <path>` reads another checkout's outputs).

**Software and hardware.** The pipeline ran on one NVIDIA RTX 4050 Laptop
GPU (6 GB) with NF4 weights, under Python 3.14 with PyTorch 2.9.1 (CUDA
12.8 build), Transformers 5.18.0, bitsandbytes 0.50.2, scikit-learn 1.9.1,
NumPy 2.5.3 and SciPy 1.18.1 (`requirements.lock.txt`). The CPU test
suite runs on every pull request with the CPU build of the same PyTorch
release.

**Seeds.** Traces are greedy. The confirmatory bootstrap uses seed
20261019 with 2,000 replicates; Stage 1's positions, folds and bootstrap
use seed 20261020, with five further cross-validation seeds as a
robustness check; the rewording null uses seed 20261019; the
stopping-policy pilot uses seed 20261024 for its re-forcing sample, its
model fitting and its bootstrap.

**The stopping-policy pilot.** Its protocol (`configs/stopping_pilot_protocol.json`,
SHA-256 `79c51284…3dc01`) and its two run configurations are hashed in
D99, with line endings pinned so that the hashes hold on any checkout. It
ran from a clean clone of the recorded commit in a separate directory,
with hash-verified copies of its inputs; `scripts/stopping_pilot.py`
refuses to run on an unfrozen protocol or out of order. The hashes of its
outputs are listed in D101 and D102, and its supplementary analyses are
`scripts/complete_pilot_c2.py` and `scripts/describe_pilot_stops.py`.

**Run artifacts.** The main run's artifacts are published as the GitHub
release `data-runs-2026-10-19`. The replication's artifacts, its
follow-ups and the rewording null are prepared as a release candidate
(sections 7 and 8 of `docs/ARCHIVE_PLAN_REP1.md`); its identifier will be added here
when it is published. The pilot's outputs are not yet in that candidate;
whether they join it is an open release decision (D102). Blinded annotation sheets and their keys are
withheld until the human annotation is complete.

# Declarations

**Author.** [record] Tejo Sridhar M V S, the author named in
`CITATION.cff` and `LICENSE`. [author] Affiliation, ORCID, a corresponding
address, and confirmation that the author list is complete.

**Author contributions.** [author] To be supplied, for example in CRediT
terms.

**Funding.** [author] To be supplied. The project's records name no funder;
that is not a statement that the work was unfunded.

**Competing interests.** [author] To be supplied.

**Ethics and consent.** [record] The study uses two public benchmarks,
GSM8K [27] and MATH [28], and openly released models. No human participants
were recruited and no personal data were collected. The planned
two-annotator protocol was not run, and its blinded sheets are withheld.
The one annotation pass that exists (the RQ2 amendment A1, D56–D57) was made
by an AI system, not by people. [author] Whether an institutional ethics
review applies.

**Use of AI assistance.** [record] The project's session records, from
2026-10-16 onwards, show that an AI assistant (Anthropic's Claude, used
through Claude Code):
- wrote and tested analysis code;
- ran the experiments the researcher approved;
- checked numbers and citations against their sources;
- drafted and revised the text of this manuscript.

The researcher's decisions, including every plan freeze and approval, are
recorded in `docs/DECISIONS.md`. [author] The extent of AI use before
2026-10-16, the wording the venue requires, and whether to cite the
assistant's skill packages.

**Data availability.** [record]
- The main run's artifacts are published as the GitHub release
  `data-runs-2026-10-19` (CC BY 4.0) of the code repository.
- The artifacts of the replication, its follow-ups, the rewording null
  and the stopping-policy pilot are prepared as a verified release
  candidate and are not yet published (§7).
- The blinded annotation sheets and their key are withheld until the
  annotation is complete.
- Problem text comes from GSM8K and MATH (MIT). Traces, activations and
  verbalisations come from Qwen2.5-7B-Instruct and the released layer-20
  NLA (Apache 2.0). No model weights are redistributed.

**Code availability.** [record] All code, configurations, frozen plans
and the decision record are at
https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail under the MIT
licence. The vendored `third_party/nla_inference/` is Apache 2.0.

**Compute.** [record] Every run in the record used one local NVIDIA RTX
4050 Laptop GPU, and the records list no cloud compute. The rewording null
took about 101 GPU-minutes and the stopping-policy pilot 9.1 GPU-hours
(D97, D101).

# Tables

## Table 1. Setting and pre-registration timeline

**(a) Setting** (from `configs/base.yaml` merged with `configs/main.yaml` and `configs/replication.yaml`; the NLA arm size of `main` is the D48 run override).

| | `main` (registered study) | `rep1` (confirmatory replication) |
|---|---|---|
| target model | Qwen/Qwen2.5-7B-Instruct, NF4 | Qwen/Qwen2.5-7B-Instruct, NF4 |
| problems | 150 GSM8K + 100 MATH | 240 GSM8K + 160 MATH, fresh (D75) |
| MATH subjects | algebra, counting_and_probability | algebra, counting_and_probability |
| traces | greedy, ≤ 768 tokens | greedy, ≤ 768 tokens |
| answer-stable tail | K = 5 continuations at T = 0.8, 200 tokens | K = 5 at T = 0.8, 200 tokens |
| registered outcome | 48-token forced answer (O48) | 48-token forced answer (O48); 256-token sensitivity (O256, D82) |
| probe and NLA layer | 20 | 20 |
| NLA verbaliser / reconstructor | kitft/nla-qwen2.5-7b-L20-av / -ar, NF4 | kitft/nla-qwen2.5-7b-L20-av / -ar, NF4 |
| verbalisation | 2 samples × 256 tokens, 2 per window kind | 2 samples × 256 tokens, 2 per window kind |
| NLA arm | 120 problems (`--set nla.max_problems=120`, D48) | every problem with a tail (380 of 400) |
| batch size | 4 | 2 (D78, deviation 1) |

**(b) Timeline** (dates from each decision's *Recorded* line in `docs/DECISIONS.md`).

| decision | recorded | what it fixed | relative to the data |
|---|---|---|---|
| D66 | 2026-10-19 | registered report frozen | main run complete |
| D68 | 2026-10-19 | post-hoc audit of the frozen report | after main's results |
| D69 | 2026-10-19T14:01Z | O3 rescored on the registered outcome; plan fixed first (D91: the amended plan's hash was not recorded before the data) | post hoc on main |
| D75 | 2026-10-19T17:46Z | **confirmatory replication frozen** (plan, problems, config) | before any rep1 trace |
| D79 | 2026-10-19T21:16Z | position-only control and truncation audit specified | before any rep1 outcome |
| D78 | 2026-10-20 | deviation 1: batch size 2 | before any rep1 outcome |
| D80 | 2026-10-20 at 10:30 IST | deviation 2: 200 re-forced boundaries; forcing in blocks of 2 | before any rep1 outcome |
| D81 | 2026-10-20T05:00Z (10:30 IST) | information audit specified | before any rep1 outcome |
| D82 | 2026-10-20 | 256-token forcing pass specified | before any extended answer |
| D83 | 2026-10-20 | leakage-resistant design (Stage 0, Stage 1) | before any Stage 1 data |
| D85 | 2026-10-20 | Stage 1 approved as revised | before any Stage 1 data |
| D94 | 2026-10-22 | **D75 verdict computed**; D79 audit | outcomes |
| D95 | 2026-10-23 | secondaries, D82 on rep1 | outcomes |
| D96 | 2026-10-23 | Stage 1 and the final D82 pass | outcomes |

## Table 2. The registered verdict and what qualifies it (rep1)

N = 380 problems with a tail; re-forcing agreement 0.965 (gate 0.95); paired problem bootstrap, 2,000 replicates; Holm over {H1, H2}; margin 0.05.

**(a) Confirmatory, verbatim (D75, D94).** The registered outcome: 48-token forced answers.

| hypothesis | Δ | 95% CI | p | Holm p | reading |
|---|---|---|---|---|---|
| H1: readout − registered probe | +0.047 | [0.019, 0.075] | 0.002 | 0.002 | **supported** |
| H2: readout − no signal | +0.057 | [0.039, 0.076] | 0.0005 | 0.001 | **supported** |

**(b) Sensitivity, never replacing (a) (D82, D95).** f = 0.867: *the 48-token outcome mostly measured commitment within 48 tokens*.

| outcome | H1 Δ [95% CI] | H1 reading | H2 Δ [95% CI] | H2 reading |
|---|---|---|---|---|
| O48 | +0.047 [0.019, 0.075] | supported | +0.057 [0.039, 0.076] | supported |
| O256 | −0.008 [−0.036, 0.024] | equivalent within ±0.05 | −0.004 [−0.024, 0.016] | equivalent within ±0.05 |
| O256-u | −0.009 [−0.035, 0.023] | equivalent within ±0.05 | −0.007 [−0.025, 0.014] | equivalent within ±0.05 |
| Ocont | −0.016 [−0.078, 0.029] | advantage of 0.05 or more excluded | −0.008 [−0.044, 0.026] | equivalent within ±0.05 |

**(c) Pre-specified secondary: candidate position (D79, D94).** Not in the verdict.

| comparison | Δ [95% CI] | Holm p |
|---|---|---|
| readout − rising-position rule | +0.156 [0.125, 0.187] | 0.002 |
| rising-position rule − no signal | −0.099 [−0.119, −0.079] | 0.002 |
| registered probe − rising-position rule | +0.109 [0.081, 0.136] | 0.002 |
| position-residualised readout − no signal | +0.007 [−0.016, 0.029] | 0.54 |
| readout − falling-position rule | −0.025 [−0.044, −0.005] | 0.016 |
| falling-position rule − no signal | +0.089 [0.066, 0.108] | 0.001 |

Readings: family X *unresolved*; family Y *unresolved*.

**(d) Pre-specified, outcome-blind positions (Stage 1; D83, D85, D96).**

1,600 positions in 400 problems. Δ₁ = text over position + a same-label probe on the same layer-20 activation, 256-token label: +0.00002 [−0.004, 0.004], *equivalent within ±0.01*. A statement about predictive information, not about faithfulness.

## Table 3. The frozen phase lexicon and its transfer to unseen problems

Frozen 2026-10-20T05:27:42+00:00 from the 240-problem arm (D84; source SHA-256 `a72e01e8…`). Exploratory in sample; the transfer to rep1 is pre-specified (D81 deviation 3) and out of sample: 0 problems shared. It describes a reasoning *phase*; it is not a faithfulness claim.

| # | concluding term (weight) | setting-up term (weight) |
|---|---|---|
| 1 | the final (2.29) | first (−1.34) |
| 2 | answer (2.24) | the first (−0.92) |
| 3 | final answer (1.93) | example (−0.88) |
| 4 | answer is (1.72) | let (−0.83) |
| 5 | concluding (1.60) | worked (−0.81) |
| 6 | thus (1.52) | for the (−0.77) |
| 7 | therefore (1.22) | let s (−0.75) |
| 8 | the answer (1.21) | ratio (−0.73) |
| 9 | thus the (1.19) | the second (−0.68) |
| 10 | result (1.14) | second (−0.66) |
| 11 | closing (1.13) | expecting continuation (−0.65) |
| 12 | the concluding (1.06) | worked example (−0.61) |
| 13 | therefore the (1.03) | establishing (−0.60) |
| 14 | sentence thus (0.98) | immediately expecting (−0.60) |
| 15 | summary (0.97) | for (−0.59) |
| 16 | process (0.96) | list (−0.59) |
| 17 | conclusion (0.86) | formula (−0.59) |
| 18 | concludes (0.85) | initial (−0.58) |
| 19 | statement (0.84) | next (−0.58) |
| 20 | final step (0.84) | continuation like (−0.57) |

| rep1 label | units | lexicon AUC | position-only AUC | position + lexicon − position: ΔAUC [95% CI] |
|---|---|---|---|---|
| tail membership | 1927 | 0.897 | 0.794 | +0.091 [0.076, 0.106] |
| registered safety (O48) | 1927 | 0.821 | 0.741 | +0.079 [0.059, 0.099] |

# Figure and table captions

**Figure 1. Study design.** *Top:* one illustrative trace of 12 chunks
with tail start *i\** = 8. Design A's menu holds two picks from each of
three windows (tail, matched position, matched length), drawn from the
code's rules, so it always contains chunks *i\** − 1 and *i\**; Stage 1
instead draws four boundaries without reference to the tail. *Bottom:* the
outcome at a boundary *s* is obtained by appending "Therefore, the answer
is" to the prefix and decoding greedily for 48 tokens (registered outcome)
or 256 tokens (sensitivity outcome); the stop is safe if the forced
answer's correctness equals the final answer's. The NLA path maps the
layer-20 activation to text and back; the probes read the activation
directly. Positions are illustrative. File: `fig_v2_0_schematic.png`.

**Figure 2. The replication (400 problems).** *Left:* the two registered
hypotheses as mean safe-rate gaps over matched budgets (380 problems with
a tail), on the registered 48-token outcome (filled; confirmatory) and the
256-token outcome (open; sensitivity analysis, which never replaces the
confirmatory result). The shaded band is the plan's ±0.05 margin;
"equivalent within ±0.05" excludes effects of 0.05 or more, not smaller
ones. *Right:* Stage 1 increments in AUC on the 256-token safety label at
1,600 outcome-blind positions; the shaded band is the ±0.01 margin. The
comparator is the same-label probe, not the registered correctness probe.
The panels are on different scales. File: `fig_v2_6_replication.png`.

**Figure 3. Forced-answer truncation.** *Left and centre* (main run, NLA
arm of 120 problems): the share of 48-token forced completions cut off
before and at or after the tail start, and the share of forced answers
that disagree with the majority of five 200-token continuations, by
whether the completion was cut off and where it lies. *Right* (256-token
pass): among pre-tail completions cut off at 48 tokens, the share that
reach the final answer when allowed 256 tokens (*f*) and the share still
cut off, on the replication (n = 600; pre-specified) and the 240-problem
arm (n = 394; exploratory). Counts are given above the bars. File:
`fig_v2_3_truncation.png`.

**Figure 4. What each signal adds at design-A menu chunks (240-problem
arm; 1,206 chunks; exploratory).** Differences in out-of-fold AUC with
95% problem-cluster bootstrap intervals, for the registered 48-token
safety label (left) and tail membership (right). "A | B" is the gain from
adding A to a model that already contains B. The explanation text adds to
position and trace text, but not to a same-label activation probe. File:
`fig_v2_1_information_increments.png`.

**Figure 5. Selective risk at matched coverage (post hoc on the earlier
data).** The share of unsafe stops among problems on which each rule
stops, at coverage 0.25, 0.50 and 0.75; labels give the mean saving among
stops. *Left:* design A's menu, 240-problem arm. *Right:* every boundary,
no menu (Stage 0, main run, 250 problems), where the same-label activation
probe is available. These are outcome-selected comparisons, not
deployment evidence. File: `fig_v2_5_selective_risk.png`.

**Figure 6. The registered estimand on the main run (120 problems).**
*Left:* break rate (one minus the safe rate) against mean saving for each
rule on the registered 48-token outcome; the dotted line marks the oracle
bound *B\**. *Right:* at 0.5 *B\**, the number of problems on which each
rule's stop is unsafe (light) or safe (dark); the remaining problems are
abstentions, which the estimand counts as safe. File:
`fig_v2_2_registered_estimand.png`.

**Figure 7. Every boundary, no menu (Stage 0; main run, 250 problems;
exploratory).** Mean safe-rate gap to the no-signal rule over matched
budgets, on registered savings (*B\** = 0.475), with 95% intervals. An
information-free falling-position rule has the largest gap, above learned
rules that use a same-label activation probe. File:
`fig_v2_4_stage0_policies.png`.

**Table 1. Setting and pre-registration timeline.** (a) The main run and
the replication, from the merged configuration files; the main run's NLA
arm size is the recorded run override. (b) Each decision's recorded date
and where it fell relative to the data.

**Table 2. The registered verdict and the analyses that qualify it
(replication).** (a) Confirmatory H1 and H2 on the registered 48-token
outcome, as computed. (b) The same hypotheses under the 256-token
sensitivity outcomes, which never replace (a); "Ocont" is the
continuations' majority, a diagnostic. (c) The pre-specified position
audit, outside the verdict. (d) Stage 1's primary contrast at
outcome-blind positions.

**Table 3. The frozen phase lexicon and its transfer to unseen problems.**
The 20 heaviest concluding and setting-up terms, frozen on the 240-problem
arm (exploratory in sample), and their pre-specified, out-of-sample
transfer to the replication (no problems shared). The lexicon describes a
reasoning phase; it is not a faithfulness claim.

# References

*The v2 manuscript's single bibliography, numbered by first citation (generated once by `scripts/consolidate_references_v2.py` on 2026-10-24 from `paper/references.md` and `paper/v2/references_additions.md`, which keep their own numbering as historical records). Each entry notes its former number. Verification: entries formerly [1]-[27] were checked on 2026-10-19 (`paper/references.md`); [28]-[32] on 2026-10-21 and [33]-[42] on 2026-10-23 (`paper/v2/references_additions.md`); the two entries added for the stopping-policy pilot on 2026-10-25 (now [31] and [32]; they have no former number) on that date; each against its arXiv record, ACL Anthology or proceedings page, or DOI/journal record. These are machine and agent checks; no human has yet verified them for this manuscript.*

[1] S. Caldarella, D. Talon, R. Aljundi, E. Ricci, and M. Mancini, "Thinking past the answer: evaluating harmful overthinking in large reasoning models," preprint, 2026, arXiv:2606.02835. (Formerly [7].)

[2] Z. Zhai, X. You, W. Yan, and X. Wang, "How much thinking is enough? Quantifying and understanding redundancy in LLM reasoning," preprint, 2026, arXiv:2605.23926. (Formerly [8].)

[3] X. Liu and L. Wang, "Answer convergence as a signal for early stopping in reasoning," in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2025, pp. 17896-17907, arXiv:2506.02536. (Formerly [5].)

[4] Y. Fu, J. Chen, S. Zhu, Z. Fu, Z. Dai, Y. Zhuang, Y. Ma, A. Qiao, T. Rosing, I. Stoica, and H. Zhang, "Efficiently scaling LLM reasoning with Certaindex," preprint, 2025, arXiv:2412.20993. (Dynasor. v1, December 2024, was titled "Efficiently serving LLM reasoning programs with Certaindex"; v2 is dated 27 May 2025.) (Formerly [34].)

[5] C. Yang, Q. Si, Y. Duan, Z. Zhu, C. Zhu, Q. Li, M. Chen, Z. Lin, and W. Wang, "Dynamic early exit in reasoning models," preprint, 2025, arXiv:2504.15895. (DEER; v3, 28 September 2025. The arXiv page lists no venue.) (Formerly [33].)

[6] X. Wang, J. McInerney, L. Wang, and N. Kallus, "Entropy after ⟨/Think⟩ for reasoning model early exiting," preprint, 2025, arXiv:2509.26522. (EAT; v3, 8 April 2026. The arXiv page lists no venue.) (Formerly [36].)

[7] A. Zhang, Y. Chen, J. Pan, C. Zhao, A. Panda, J. Li, and H. He, "Reasoning models know when they're right: probing hidden states for self-verification," in *Proceedings of the Conference on Language Modeling (COLM)*, 2025, arXiv:2504.05419. (Formerly [6].)

[8] K. Liu, Y. Liu, X. Yang, P. Wang, W. Zhang, S. Feng, Y. Zhang, and D. Wang, "NEAT: neuron-based early exit for large reasoning models," in *Findings of the Association for Computational Linguistics: ACL 2026*, 2026, pp. 24616–24627, doi: 10.18653/v1/2026.findings-acl.1231, arXiv:2602.02010. (Formerly [37].)

[9] Ö. F. Akgül, Y. H. Kalaycı, R. Kannan, W. Neiswanger, and V. Prasanna, "LYNX: learning dynamic exits for confidence-controlled reasoning," preprint, 2025, arXiv:2512.05325. (Formerly [38].)

[10] Z. Dong, F. Qin, and M. Shah, "When does learning to stop help? A cost-aware study of early exits in reasoning models," preprint, 2026, arXiv:2606.30852. (Formerly [28].)

[11] M. Wu, C. Zhou, S. Bates, and T. Jaakkola, "Thought calibration: efficient and confident test-time scaling," in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2025, arXiv:2505.18404. (The venue is from the ACL Anthology listing 2025.emnlp-main.722. The arXiv page lists none.) (Formerly [29].)

[12] K. Fraser-Taliente, S. Kantamneni, E. Ong, D. Mossing, C. Lu, P. C. Bogdan, E. Ameisen, J. Chen, D. Kishylau, A. Pearce, J. Tarng, A. Wu, J. Wu, Y. Zhang, D. M. Ziegler, E. Hubinger, J. Batson, J. Lindsey, S. Zimmerman, and S. Marks, "Natural language autoencoders produce unsupervised explanations of LLM activations," *Transformer Circuits Thread*, project report, 7 May 2026. [Online]. Available: https://transformer-circuits.pub/2026/nla/ (Formerly [10].)

[13] M. Li, A. M. Ceballos Arroyo, G. Rogers, N. Saphra, and B. C. Wallace, "Do activation verbalization methods convey privileged information?," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2026, arXiv:2509.13316. (Formerly [12].)

[14] T. Lanham, A. Chen, A. Radhakrishnan, B. Steiner, C. Denison, D. Hernandez, D. Li, E. Durmus, E. Hubinger, J. Kernion, K. Lukosiute, K. Nguyen, N. Cheng, N. Joseph, N. Schiefer, O. Rausch, R. Larson, S. McCandlish, S. Kundu, S. Kadavath, S. Yang, T. Henighan, T. Maxwell, T. Telleen-Lawton, T. Hume, Z. Hatfield-Dodds, J. Kaplan, J. Brauner, S. R. Bowman, and E. Perez, "Measuring faithfulness in chain-of-thought reasoning," preprint, 2023, arXiv:2307.13702. (Formerly [16].)

[15] D. Scalena, S. Candussio, L. Bortolussi, E. Fersini, M. Nissim, and G. Sarti, "Beyond the commitment boundary: probing epiphenomenal chain-of-thought in large reasoning models," preprint, 2026, arXiv:2606.13603. (Formerly [31].)

[16] Y. Mo, D. Zhao, and H. Geng, "Stable answers, unfinished reasoning: why self-consensus is not a safe early-exit signal," preprint, 2026, arXiv:2609.09989. (Formerly [22].)

[17] Q. Long, K. J. Jiang, J. Chen, X. Guo, L. Gan, and W. Wang, "Self-verification dilemma: experience-driven suppression of overused checking in LLM reasoning," preprint, 2026, arXiv:2602.03485. (Formerly [9].)

[18] S. Song, M. Oh, S. Lee, S. Jo, and Y. Jo, "ThinkBrake: efficient reasoning via log-probability margin guided decoding," in *Findings of the Association for Computational Linguistics: ACL 2026*, 2026, pp. 21765–21790, doi: 10.18653/v1/2026.findings-acl.1095, arXiv:2510.00546. (Earlier arXiv versions were titled "ThinkBrake: mitigating overthinking in tool reasoning".) (Formerly [35].)

[19] S. Boppana, A. Ma, M. Loeffler, R. Sarfati, E. Bigelow, A. Geiger, O. Lewis, and J. Merullo, "Reasoning theater: disentangling model beliefs from chain-of-thought," preprint, 2026, arXiv:2603.05488. (Formerly [30].)

[20] A. Pan, L. Chen, and J. Steinhardt, "LatentQA: teaching LLMs to decode activations into natural language," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2026, arXiv:2412.08686. (The venue is from the arXiv comments field.) (Formerly [32].)

[21] G. Lek, Z. Xia, P.-Y. Chen, and L. Y. Chen, "Understanding confabulation and rethinking reconstruction in activation explanations," preprint, 2026, arXiv:2609.33702. (Formerly [13].)

[22] H. Zhao, Z. Hei, W. Shi, H. Deng, N. Zou, and M. Du, "Faithful activation verbalization: reducing hallucinations in LLM representation interpretation," preprint, 2026, arXiv:2609.34033. (Formerly [14].)

[23] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023, arXiv:2305.04388. (Formerly [15].)

[24] Y. Chen, J. Benton, A. Radhakrishnan, J. Uesato, C. Denison, J. Schulman, A. Somani, P. Hase, M. Wagner, F. Roger, V. Mikulik, S. R. Bowman, J. Leike, J. Kaplan, and E. Perez, "Reasoning models don't always say what they think," preprint, 2025, arXiv:2505.05410. (Formerly [17].)

[25] Qwen Team: A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, *et al.*, "Qwen2.5 technical report," preprint, 2024, arXiv:2412.15115. (Formerly [39].)

[26] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: efficient finetuning of quantized LLMs," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023, arXiv:2305.14314. (NF4 is defined here.) (Formerly [40].)

[27] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "Training verifiers to solve math word problems," preprint, 2021, arXiv:2110.14168. (Formerly [3].)

[28] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, "Measuring mathematical problem solving with the MATH dataset," in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2021. (Formerly [4].)

[29] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. (Formerly [41].)

[30] D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, no. 6, pp. 657–680, 1987, doi: 10.1007/BF01068419. (Formerly [42].)

[31] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017, arXiv:1705.08500. (Added 2026-10-25; checked against the NeurIPS proceedings page.)

[32] C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," *Biometrika*, vol. 26, no. 4, pp. 404–413, 1934, doi: 10.1093/biomet/26.4.404. (Added 2026-10-25; checked against its Crossref record.)
