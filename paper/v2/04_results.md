# 4. Results

*Revised candidate, 2026-10-24. Every number is copied from the stored
output named in `paper/v2/evidence/key_numbers.json` and checked against
it by `scripts/verify_manuscript_numbers.py`. Evidence classes as in
§3.8.*

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
