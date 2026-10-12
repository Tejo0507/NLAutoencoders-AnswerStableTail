# 5. Statistical analysis plan

*Draft. Fixed before the main run's analysis. One reporting decision is
outstanding (§5.5). Decision records: `docs/DECISIONS.md` D29, D30, D37–D41.*

## 5.1 The primary comparison: matched budget, problem-level

**Unit of analysis: the problem.** Safe stopping (§3.5) is a per-problem binary
outcome and every rule is scored on the same problems, so the comparison is
paired over problems.

This is stated because the alternative was tried and is wrong. An earlier
analysis paired over the nine requested *budget levels* and applied a Wilcoxon
signed-rank test. A rule's safety/saving curve is a step function, so a span
of requested budgets resolves to a handful of distinct operating points and
asking for more budgets copies one comparison rather than adding evidence. On
the pilot all nine budgets landed on a single operating-point pair, and the
resulting p-value was a function of the grid size — 0.250 at three budgets,
0.0039 at nine, 1e-6 at twenty-five. Nine was a default. The test is now
**McNemar's exact test over problems**, with a conditional-exact paired
interval on the rate difference (D29).

**Matched budget.** Each rule exposes a per-boundary score; the threshold is
swept; rules are read off their curves at the same mean token saving. A rule
that stops later is trivially safer, so comparing rules at their own operating
points would compare budgets rather than rules.

Three implementation commitments make that comparison honest:

- **The threshold grid is exhaustive, not fixed-size.** A curve changes only
  at a threshold that crosses an observed score, so sweeping the *observed
  values* enumerates every achievable operating point exactly. A fixed linear
  grid samples the score axis instead and therefore depends on how the scores
  happen to be distributed: the probe's scores saturate, so 25 evenly spaced
  thresholds found 5 operating points and none near a shared budget, and its
  comparison was reported as unevaluable. Swept at observed values the same
  problems give 86 operating points (D30).
- **A match is a match, within tolerance.** An operating point counts as
  matched only if it lies within **0.05** of the target in mean fraction of
  tokens saved; otherwise the comparison is not made at that budget. Returning
  the nearest point regardless would produce a table that looks matched and is
  not — on the pilot, asking all four rules for a budget of 0.632 returned one
  rule at 0.407 and another at 0.855.
- **The budget grid is the intersection of what every rule can reach**, with
  zero-saving points excluded. At a point where no rule stops, every rule is
  trivially safe and the difference is identically zero; including it would
  make the comparison vacuous rather than conservative.

**The comparison is computed on the problems the verbaliser arm covers**, with
every baseline re-swept over exactly those problems. Comparing curves built
from different problem sets would confound the rule with the problems.

## 5.2 Three outcomes a planned comparison can have

These are reported differently, and conflating them is the main way a
matched-budget design misleads.

| outcome | signature | what it licenses |
|---|---|---|
| **evaluable** | matched budget exists; ≥1 discordant problem | the McNemar result and its interval |
| **evaluable, uninformative** | matched; **zero** discordant problems | an observation of exact agreement over N problems — *not* an absence of evidence |
| **unevaluable** | no budget where both rules have a matched point | **nothing** about the rules' relative safety; the comparison was not made |

A test in the third category stays in the report rather than disappearing from
it, because a comparison that was planned and could not be made is itself a
result. We report the reason, and distinguish a rule whose score is too coarse
to place an operating point in the window from two curves that simply do not
overlap on this corpus (D31, D37b).

**Power is reported per comparison, not just overall.** For each McNemar
comparison we report `smallest_attainable_p`: the p-value that many discordant
pairs would give if every pair fell the same way — the best the comparison
could possibly have done. It separates *no difference was found* from *no
difference could have been found*, and needs no simulation. On the pilot's four
discordant pairs it is 0.125 against a corrected threshold near 0.01, so that
comparison could not have reached significance whichever way the pairs fell.

## 5.3 Multiplicity

Benjamini–Hochberg at **q = 0.05** over the pre-registered family.

**The denominator is the evaluable tests.** The justification is not
convenience: BH controls the false discovery rate over hypotheses actually
tested, and a hypothesis that produced no p-value can be neither a rejection
nor a false one, so including it buys no error control and only makes every
other test conservative. What "pre-registered family" guards against is
choosing the denominator *after seeing which tests came out well*, and that
guard holds provided every exclusion is for a reason independent of the
direction and size of the effect under test. On this plan they are: no matched
budget exists (set by the saving curves, not the safe rates); fewer than three
observations in an arm (set by accuracy, not by the quantity compared); and a
dependent variable with no variance (an instrument floor, not the arms' means).

**One exclusion was not independent of the effect, and was corrected.** A
McNemar comparison with zero discordant pairs previously returned no p-value
and dropped out of the denominator — but zero discordance is exactly the null
holding perfectly, so the family was contracting *because a null was true*,
which pulls every other q-value down. The exact conditional p-value there is
**1.0**: conditioning on zero discordant pairs leaves a single possible table.
That is arithmetic, not convention, and the test is now counted. Replayed on
the pilot's own p-values the correction raises every q-value, changes no
verdict, and puts a previously vanishing comparison on the table as a null
(D38).

**Both conventions are reported.** Every run also emits the same p-values
corrected over the *whole* pre-registered family, plus a flag recording
whether the set of significant findings differs between the two. The choice of
convention therefore changes which number is foregrounded, never which numbers
exist.

## 5.4 The causal gate

The claim *a tail direction exists* is a conjunction of four conditions
(§3.8). Two points matter numerically.

**"Beat" is a paired test, not a comparison of means.** It previously compared
two arm averages, which has no inferential content — a candidate at 0.30
against a control at 0.29 satisfied it, on ten problems, with no test, no
interval and no sensitivity to how many problems were behind either number.
Every arm runs on the same problems, so the contrast is paired over the
problems each arm shares, by Wilcoxon signed-rank at α = 0.05, read at the
dose the controls run at. The quantity per problem is
`|effect(candidate)| - |effect(control)|`: absolute, because the question asks
whether ablation *changes* behaviour rather than which way. Replicate control
directions are averaged within a problem first, so a problem counts once
however many replicates it has. This is the same correction §5.1 applies to
the primary comparison, and the two are now consistent (D41).

**Every condition is tri-state.** True, false, and *could not be compared* —
and the conjunction is undetermined if any input is. The three ways a
condition goes undetermined are reported separately: the readout does not vary
at all; fewer than three problems appear in both arms; or the two arms moved
the readout identically, in which case the answer is a determinate refusal
because "beat" is one-sided. Collapsing any of these into "false" would read
as evidence against a tail direction where there is none either way.

**The dose-response requirement** needs at least three *distinct doses*, a
curve that actually moves, and non-decreasing magnitude over the sweep. A
constant readout satisfies "non-decreasing" vacuously, so an unmoving curve
previously reported a satisfied dose-response beside a readout that never
changed; it is now undetermined in that case (D41).

**The intervention must be shown to have happened.** No p-value is reported
for the causal family test unless the candidate arm is *demonstrated* to have
edited at least one position at the compared dose. Without that condition, two
arms agreeing because nothing was done to either would read as a tested null.
An unrecorded edit count counts as not demonstrated.

**The dependent variable is a documented choice, not a registered one.** §3.8
records that the construct is registered and its operationalisation is not.
The measure used is a count of rechecking markers in the generated text,
compared between arms — the face-valid reading of the construct. A readout
measuring a *different* construct may not be substituted for it: output
length, for instance, varies where the marker count does not, but "how much
further the model writes" is not verification, and adopting it would answer a
different question under this one's name. Eligibility is enforced in code
against the registered construct (D41). §8 reports what this costs.

## 5.5 Outstanding reporting decision

**Which multiplicity convention to foreground.** §5.3's argument supports the
evaluable-test denominator as primary, with the pre-registered-family count as
a stated sensitivity, and that is how the code computes it. The
pre-registration's wording admits the stricter reading, and no data can settle
a reporting convention. Both number sets are in every artifact and the pilot's
verdict is identical under both; if the main run's robustness flag comes back
false, the choice decides a finding and must be stated explicitly in the
Results rather than assumed.

## 5.6 Falsification battery

Ten pre-registered alternative explanations, each with a test rather than an
argument: length alone (F1); difficulty, stratified by MATH level **and** by
gold-answer magnitude (F2); verbaliser priors, reported as a headline (F3);
answer-parser artefact, via a second stricter extractor (F4); sensitivity to
the AST definition itself, sweeping `K`, `T`, `w` and the chunker (F5); probe
leakage, by label shuffling within group — the probe must collapse to chance
(F6); layer specificity, at 14 and 24 (F7); quantisation (F8); multiplicity
(F9, §5.3); and token-position confounding in reconstruction, comparing tail
against matched-position and matched-length windows (F10).

F10 is the one most likely to constrain the headline: if tail reconstruction
differs from matched-position but *not* from matched-length, the effect is
positional rather than tail-specific, and §8 says so.
