# Position-only control and forced-answer truncation: results on the stored runs (D79)

**Exploratory.** Plan: `docs/POSITION_TRUNCATION_PLAN.md`. It was frozen at
`8435272e…f57a` before any result. It was run as amended at `bffc9e08…c877`,
after deviations 1–3; only its deviations section differs from the frozen
text, which was checked by hashing.

- **Runs:** `main` (120 problems) and `main_nla240` (240; post hoc, D54).
  On both, the readout, probe and no-signal results had been seen before
  the plan.
- **Family Y** (the falling-position rule) was added after seeing `main`'s
  Part A, so on these runs it is **post hoc** (deviation 3). For `rep1` it is
  specified before any outcome.
- **Code:**
  - `scripts/position_truncation_audit.py` → `runs/<run>_position_audit/position_truncation.json`
    (SHA-256 `main`: `1c963492…`, `main_nla240`: `953375b9…`);
  - `src/nlaast/analysis/position.py`;
  - `tests/test_position_baseline.py`.
- **Checks passed before anything was reported:**
  - every stored design-A comparison of `o3_registered.json` reproduces
    exactly (gap, CI and p, to 1e-9), from the same seed;
  - so do B\* and D69's cut-off share at 0.5·B\*;
  - the rebuilt safety labels equal the registered ones on every boundary;
  - every input hash equals the one the registered analysis recorded.

## 1. Part A: does position explain the readout's design-A result?

Mean safe-rate gap over the matched budgets. Bootstrap 95% CI. p is the
two-sided bootstrap p-value, Holm-adjusted within its family.

| comparison | `main` (N = 120) | `main_nla240` (N = 240) |
|---|---|---|
| *registered, for reference:* readout − no-signal | +0.052 [0.022, 0.082] | +0.069 [0.049, 0.093] |
| *registered:* readout − probe | +0.027 [−0.026, 0.076] | +0.057 [0.023, 0.091] |
| **X1** readout − `position_tokens` (rising) | +0.117 [0.064, 0.169], p 0.002 | +0.141 [0.106, 0.180], p 0.002 |
| **X2** `position_tokens` − no-signal | **−0.065** [−0.102, −0.030], p 0.002 | **−0.072** [−0.099, −0.049], p 0.002 |
| **X3** probe − `position_tokens` | +0.090 [0.039, 0.144], p 0.002 | +0.084 [0.050, 0.123], p 0.002 |
| **X4** `readout_residualised` − no-signal | +0.014 [−0.022, 0.050], p 0.48 | +0.024 [−0.001, 0.048], p 0.064 |
| **Y1** readout − `position_tokens_falling` *(post hoc)* | −0.024 [−0.070, 0.015], p 0.21 | −0.025 [−0.046, 0.001], p 0.062 |
| **Y2** `position_tokens_falling` − no-signal *(post hoc)* | **+0.082** [0.054, 0.117], p 0.001 | **+0.100** [0.074, 0.126], p 0.001 |
| `relative_position` − no-signal (diagnostic) | −0.073 [−0.109, −0.031] | −0.058 [−0.084, −0.028] |
| `readout_residualised` − probe | −0.011 [−0.067, 0.043] | +0.012 [−0.024, 0.047] |

**Within-problem correlation with position** over menu chunks (Spearman,
outcome-free):

| | `main` | `main_nla240` |
|---|---|---|
| readout | median ρ −0.37 (25% of problems positive) | −0.30 (27%) |
| probe | median ρ **+0.70** (98% positive) | +0.70 (98%) |

**Readings by the plan's fixed rules:**
- **Family X: unresolved** on both runs.
  - X1 is positive, but X4 is not significant, so "carries stopping
    information beyond position" is not met.
  - X2 is *negative*, so the plan's position caveat on "beats no signal"
    does not trigger.
- **Family Y** (post hoc here): on both runs, *a falling-position rule
  reproduces the readout's advantage over no signal*. Y2 is Holm-significant
  and positive, and Y1's interval contains 0. Its point estimates exceed the
  readout's.

**What this means.**
1. **The hypothesis that motivated the plan is wrong in its direction.** A
   score that *rises* with position does worse than no signal. The probe's
   score rises with position, and the readout's falls.
2. **The readout's design-A advantage over no signal is reproduced, on these
   data, by a rule with no readout at all.** The rule is "stop at the first
   menu chunk if it comes within the first T tokens; otherwise never stop".
   With its log-position trend removed, the readout no longer beats no signal
   significantly (X4).
3. **Why: the metric, not the readout.** At 0.5·B\* on `main`:

   | rule | safe rate | stops | safe stops | safe share of its stops |
   |---|---|---|---|---|
   | `position_tokens_falling` | **0.842** | 23 | 4 | **0.17** |
   | readout | 0.833 | 32 | 12 | 0.38 |
   | probe | 0.792 | 120 | 95 | 0.79 |
   | `position_tokens` | 0.692 | 101 | 64 | 0.63 |

   (`main_nla240`: falling rule 47 / 15; readout 53 / 21; probe 239 / 188.)

   The registered estimand counts a rule that does not stop as safe and
   matches rules on *mean* saving. The cheapest way to score well is
   therefore to stop very early in a few problems, each saving most of its
   trace and mostly breaking the answer, and to abstain everywhere else. The
   falling rule does exactly that by construction, and the readout's score
   behaves similarly. The probe stops late almost everywhere and is
   penalised for it. The registered metric is "fewest answers broken per
   token saved on average". It is not "picks good stopping points". D69's
   wording, "the readout's score selects safer stops than chance"
   (`docs/REANALYSIS_2026-10-19.md`), is therefore inaccurate: most of the
   readout's stops are unsafe. That document is not edited here; this note is
   the correction.
4. **The menu is not available online.** It is built from the tail start,
   which is the outcome. Design A gives it to every rule. So the falling rule
   is a fair *within-design-A* control, since it uses no more than the readout
   is given, but it is not a deployable rule, and neither is the readout as
   evaluated.

**What is not shown.**
- That the readout carries *no* information. X4 on the 240 arm is +0.024
  [−0.001, 0.048].
- That the falling rule is better than the readout. Y1's interval includes 0.
- Anything about the verbalisation's text. The score is reconstruction
  cosine.

## 2. Part B: forced-answer truncation

### B.1–B.2 Measured

| | `main` arm | `main_nla240` arm |
|---|---|---|
| traces at the 768-token cap | 3.2% (GSM8K 0%, MATH 8%); final answer unparsed: 0 | same traces |
| forced completions cut off at 48 tokens (all arm boundaries) | 37% (1290) | 39% (2625) |
| … before `i*` / at or after `i*` | **54% / 5%** | **57% / 6%** |
| … on the menu: matched-position / tail windows | 64% / 6% | 60% / 5% |
| continuations (200 tokens) with no parsed answer | 0% | 0% |

| at stored boundaries with 5 continuations | cut off, before `i*` | finished, before `i*` | at or after `i*` |
|---|---|---|---|
| forced answer ≠ continuations' majority (`main`) | **99%** (176) | 65% (112) | 0% (448) |
| continuations' majority = final answer (`main`) | 98% | 93% | 100% |
| same, `main_nla240` | 98% ≠ / 96% = final (353) | 69% / 93% (222) | 0% / 100% |

Re-forcing a stored boundary reproduces its answer in **100%** of cut-off
cases (22 and 52), against 94–96% of finished ones. So the cut-off answers
are *reliable* (greedy and deterministic) but not *valid*. Before the tail,
the 48-token forcing usually starts a derivation, and the extractor takes
whatever number came last. Meanwhile the model's own 200-token
continuations reach the final answer 96–98% of the time.

### B.3 Where each rule stops

Share of stops on a cut-off completion, from 0.1 to 0.9 × B\* (`main`):
- readout: 0.10–0.38;
- falling-position rule: 0.00–0.33;
- probe: 0.02–0.24;
- rising position: 0.21–0.41.

The readout's excess over the probe (D69: 28% vs 15% at 0.5·B\*) holds
across budgets.

### B.4 Sensitivity: two readings of a cut-off completion

| comparison | registered | uncommitted (→ no answer) | committed to final |
|---|---|---|---|
| **`main`**, labels changed | — | 31 | **423** |
| B\* | 0.362 | 0.327 | 0.606 |
| readout − no-signal | +0.052 [0.022, 0.082] | +0.053 [0.024, 0.084] | −0.021 [−0.054, 0.013] |
| readout − probe | +0.027 [−0.026, 0.076] | +0.032 [−0.015, 0.083] | −0.005 [−0.057, 0.049] |
| falling − no-signal | +0.082 [0.054, 0.117] | +0.088 [0.057, 0.123] | −0.020 [−0.053, 0.014] |
| **`main_nla240`**, labels changed | — | 70 | **863** |
| readout − no-signal | +0.069 [0.049, 0.093] | +0.073 [0.053, 0.096] | −0.021 [−0.045, 0.004] |
| readout − probe | +0.057 [0.023, 0.091] | +0.062 [0.029, 0.095] | −0.004 [−0.043, 0.030] |

**Readings that change** (plan §B.4: *dependent on the forcing budget*):
- **under "committed to final", on both runs:**
  - D69's overall verdict becomes *uninformative: no rule beats the no-signal
    rule*;
  - readout vs probe becomes *advantage of 0.05 excluded* (`main`) or
    *equivalent within 0.05* (240 arm);
  - the falling-rule reading becomes *unresolved*;
- **under "uncommitted":** only convergence, which becomes *not evaluable*.

**What this means.** The registered labels already treat almost every
cut-off as unsafe: reading them as "no answer" moves only 31 labels. So the
registered outcome is, in effect, "can the model commit within 48 tokens from
here". Every advantage the readout has — over no signal and, on the 240 arm,
over the probe — exists only under that reading. If a cut-off completion is
read as the answer the model's own continuations reach, every rule is
indistinguishable from no signal. Neither extreme is the truth. The
"committed" reading credits the saving without the extra tokens that
finishing would cost. The registered outcome stays primary. But a claim that
the readout carries useful stopping information is, on these data,
**dependent on the forcing budget**.

## 3. What `rep1` needs, in order

Nothing extra is generated. After the D75 pipeline has produced
`runs/rep1_forced/`, the analysis stages and `runs/rep1_o3_registered/`:

```
python scripts/position_truncation_audit.py --config replication \
    --forced-dir runs/rep1_forced --confirmatory
```

The D75 verdict is computed first, by `reanalyse_o3_registered.py
--confirmatory`, and is reported as registered. This audit is secondary. It
reports family X, family Y (here pre-specified), the truncation measurements
and, for each truncation scenario, whether each D75 reading changes.
