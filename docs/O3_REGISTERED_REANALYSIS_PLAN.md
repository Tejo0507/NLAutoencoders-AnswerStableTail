# O3 under the registered outcome: analysis plan (fixed 2026-10-19, before the data)

**Status.** Written and hashed (D69) **before** any forced answer was
generated for the boundaries that lack one, and before any O3 number was
computed under the registered outcome. The parsed-outcome results (the
registered run, D68) *had* been seen, so this is a **corrected, post-hoc
analysis**, not a confirmatory one. It does not replace the registered O3 test,
which stays in the record as computed. Changes after the hash are recorded as
dated deviations at the end of this file; none is made silently.

## 1. Why the registered comparison is redone, and what must change

1. **Outcome.** PROJECT_PLAN §5 scores a stop at `s` on the answer *forced*
   from `P_s`; the run scored the *parsed* answer (D68). This analysis uses the
   forced answer.
2. **Budgets.** The registered comparison was made above the oracle bound,
   where a rule using no signal beats every rule (D68). Here every budget is
   **below the oracle bound for the registered outcome**.
3. **Position leakage, new under the registered outcome.** The readout can
   stop only at the chunks the verbaliser was run on. With two windows per
   kind those include exactly the tail start `i*` and `i*−1`
   (`run_autoencoder.select_windows`). By criterion 1 and persistence, a stop
   at or after `i*` is always safe under the registered outcome, so the
   readout's *menu of stopping positions* encodes the outcome. A comparison in
   which the readout chooses from that menu and the other rules choose from
   every boundary would credit the readout with the AST's own information.
   **The primary design therefore gives every rule the same menu.**

## 2. Population, unit, outcome

- **Population:** the 120 problems of the registered NLA arm
  (`results.json` → `nla_stopping.problem_ids`). The 240-problem arm
  (`main_nla240`, D54) is analysed the same way as an exploratory extension.
- **Unit of analysis:** the problem. Every rule is scored on the same problems
  and each comparison is paired.
- **Outcome (registered):** stop at `s` is safe iff the forced answer from
  `P_s` is equivalent to the final answer and the final answer is correct, or
  both are incorrect. A forced completion that parses to no answer is not
  correct. A rule that never stops is safe (as registered).
- **Instrument:** the stored forced answer wherever the traces stage generated
  one; elsewhere, the answer from `scripts/force_all_boundaries.py`, which
  repeats the traces stage's forcing call. That script also re-forces 200
  stored boundaries chosen at random (seed 20261019). **If stored and
  regenerated answers agree on fewer than 0.95 of them**, the instrument is
  not reproducible enough to be mixed, and the analysis is reported as
  blocked on that rather than run.

## 3. Rules

| rule | score | fires at |
|---|---|---|
| NLA readout | mean reconstruction cosine of the usable samples at a chunk, carried forward (as registered) | first boundary whose score ≥ threshold |
| semantic entropy | confidence, constant within a problem (as registered) | same |
| probe | out-of-fold probe probability (as registered) | same |
| answer convergence | agreement run length (as registered, D36) | same |
| **no signal** | i.i.d. Uniform(0, 1) scores; 2,000 Monte Carlo draws, safe rates averaged | same |

Thresholds are swept exhaustively over each rule's observed scores (D30).

## 4. Designs

- **Design A, primary: a shared menu.** On problem *p*, every rule may stop
  only at `S_p`, the chunks with at least one usable NLA reconstruction. A
  rule's score elsewhere is treated as missing. The readout's own behaviour is
  unchanged by this, because its carried-forward score first crosses a
  threshold at a chunk in `S_p`.
- **Design B, secondary: every rule at every boundary**, as registered. It is
  reported because it is what the plan describes. Its readout comparisons
  carry the leakage of §1.3 and are not used for the verdict.

## 5. Budgets

For each design, the **oracle bound** `B*` is the mean over problems of the
largest token saving at a boundary on that design's menu where stopping is
safe (0 where none is). It is computed from the outcome alone, before any
rule. The budget grid is `G = {0.1, 0.2, …, 0.9} × B*`, which lies strictly
below the bound. A rule is matched at budget *b* by its operating point with
mean saving closest to *b*, if that point is within **0.02**. The registered
0.05 tolerance is too coarse for budgets as small as these. This is a stated
deviation from the registered tolerance.

## 6. Estimands, inference, multiplicity

- **Primary estimand,** for each comparison of the readout against a
  comparator *X* in design A:
  Δ_X = mean over budgets in `G` where both are matched of
  [safe rate(readout) − safe rate(X)]. This is the average vertical gap
  between the safety/saving curves below the oracle bound. A comparison with
  fewer than 3 matched budgets is **not evaluable** and is reported as such.
- **Inference:** a paired problem bootstrap with 2,000 replicates. Each
  replicate resamples problems, rebuilds every curve and re-matches at the
  fixed budgets. The no-signal rule uses 100 draws per replicate. The interval
  is the 95% percentile interval, and the two-sided p-value is
  2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)), floored at 1/2000.
- **Family F1, the question:** the readout against semantic entropy, the
  probe and convergence. Holm at α = 0.05.
- **Family F2, is the comparison informative at all:** each of the four
  rules against the no-signal rule, with the same estimand. Holm at
  α = 0.05.
- **Secondary:** McNemar's exact test at the single grid budget 0.5·`B*`,
  for continuity with the registered test; and the decomposition of each
  safe rate into abstentions and safe stops at that budget.

## 7. Decision rules, fixed now

- **The comparison is informative** iff at least one rule beats the no-signal
  rule in F2 (Holm-adjusted p < 0.05 and Δ > 0). If none does, every F1
  result is reported as uninformative, whatever its p-value.
- **Incremental value against X is established** iff Δ_X > 0, the
  Holm-adjusted p < 0.05, **and** the readout beats the no-signal rule in F2.
- **A practically relevant advantage over X is excluded** iff the upper
  95% bound of Δ_X < 0.05, a margin of 0.05 in safe rate.
- **Equivalence with X within ±0.05** iff the whole interval lies inside
  (−0.05, 0.05). Nothing short of that is called equivalence.
- **Overall verdict:**
  - *established* if established against every evaluable comparator;
  - *refuted as a practically relevant advantage* if excluded against at least
    one comparator that itself beats the no-signal rule;
  - otherwise **unresolved**.

## 8. Reported regardless of outcome

For each rule and budget: the safe rate, the mean saving, the fire rate and
the abstention count. Also: the oracle bound under both outcomes; the number
of problems; the reproducibility agreement; how often a stop lands on a forced
completion that was cut off; the cost of each signal and of the forcing
itself; and the historical parsed-outcome comparison, reproduced exactly as a
check.

## Deviations after the hash

1. **2026-10-19, before any registered-outcome result was computed: the
   no-signal rule is computed in closed form.** With i.i.d. Uniform(0, 1)
   scores and threshold `t`, each menu boundary qualifies independently with
   probability `q = 1 − t`. The rule stops at the first qualifying boundary,
   so the expected saving and expected safe rate of each problem are
   polynomials in `q`. The no-signal rule at budget *b* is taken at the `q`
   whose expected mean saving equals *b*. That `q` is found on a grid of
   2,001 values and must lie within the same 0.02 tolerance as the other
   rules.
   - *Why:* re-sweeping a Monte Carlo rule 100 times inside each of 2,000
     bootstrap replicates is too slow here. The closed form is the
     many-draw limit of the planned rule, up to matching on the expected
     rather than the realised saving.
   - *Check:* the planned 2,000-draw Monte Carlo is still computed on the full
     sample and reported beside the closed form, with their difference.
