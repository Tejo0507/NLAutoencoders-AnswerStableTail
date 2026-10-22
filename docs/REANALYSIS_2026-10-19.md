# Reanalysis after the audit (D69–D74), 2026-10-19

Results of the five stages that followed the scientific audit (D68).
**Everything here is corrected or exploratory and post hoc.** The frozen
report, the registered results and every file under `runs/main/` are
unchanged. Each section names the script that regenerates its numbers, and
each script refuses to overwrite its output.

## Stage 1: the primary comparison under the registered outcome (D69)

**Plan:** `docs/O3_REGISTERED_REANALYSIS_PLAN.md`, hashed before any
registered-outcome number existed (one dated pre-data deviation, recorded in
the plan).

**Run, in order:**
1. `scripts/force_all_boundaries.py` (GPU, about 1 h 40 min) →
   `runs/main_forced/`;
2. `scripts/reanalyse_o3_registered.py` → `runs/main_o3_registered/o3_registered.json`;
3. `scripts/verify_o3_registered.py` → `verification.json`.

### What was fixed, and what was checked first

- **Outcome.** Safety is now scored on the **forced** answer (PROJECT_PLAN §5)
  through `build_rows(outcome="forced")`. The historical parsed outcome stays
  the default and still reproduces: the registered p-values 0.0989 and 1.0
  were recomputed exactly before anything else was reported.
- **Missing measurements.** All 1,236 boundaries without a forced answer were
  forced with the traces stage's own call (greedy decoding, 48 tokens, the
  same prompt and suffix, blocks of 4). The instrument reproduced on **194 of
  200** re-forced stored boundaries (0.97, above the plan's 0.95 bar).
  Preferring the regenerated answers wherever one exists changes no
  conclusion.
- **Independent check.** The original code path (`decisions_at`,
  `sweep_threshold`, `at_matched_budget`, `oracle_safe_saving`) reproduces
  every per-budget safe rate, saving and oracle bound of the matrix engine:
  86 checks, maximum difference 2.2e-16.
- **Leakage handled.** The readout can stop only at its verbalised windows,
  which straddle the tail start, and a stop at or after the tail start is
  always safe under the registered outcome. The primary design therefore
  gives **every** rule the same menu of positions (design A). Restricting the
  readout to its own menu leaves its curve unchanged, as checked.

### Oracle bounds

The largest mean saving achievable without one unsafe stop:

| outcome | all boundaries | the shared menu |
|---|---|---|
| parsed (historical) | 0.174 | — |
| **forced (registered)** | **0.459** | **0.362** |

The registered comparison was made at 0.61–0.74, above even the
registered-outcome bound. Every budget here lies below it (0.1–0.9 × B\*).

### Results, design A (primary): 120 problems, budgets 0.036–0.325

Safe rate at each budget; every rule matched within 0.02 where shown.

| × B\* | budget | NLA readout | probe | convergence | no signal |
|---|---|---|---|---|---|
| 0.1 | 0.036 | 0.958 | 0.967 | 0.950 | 0.954 |
| 0.3 | 0.108 | 0.917 | 0.900 | — | 0.863 |
| 0.5 | 0.181 | 0.833 | 0.792 | 0.708 | 0.772 |
| 0.7 | 0.253 | 0.733 | 0.717 | — | 0.682 |
| 0.9 | 0.325 | 0.658 | 0.608 | — | 0.592 |

Mean safe-rate gap over the matched budgets, from a paired problem bootstrap
(2,000 replicates; Holm within each family):

| family | comparison | budgets | gap [95% CI] | Holm p |
|---|---|---|---|---|
| F2 | **NLA − no signal** | 9 | **+0.052 [0.022, 0.082]** | **0.006** |
| F2 | probe − no signal | 9 | +0.025 [−0.013, 0.063] | 0.26 |
| F2 | convergence − no signal | 3 | −0.029 [−0.063, 0.008] | 0.26 |
| F1 | NLA − probe | 9 | +0.027 [−0.026, 0.076] | 0.28 |
| F1 | NLA − convergence | 3 | +0.072 [0.017, 0.122] | 0.001 |
| F1 | NLA − semantic entropy | **0** | not evaluable | — |

The closed-form no-signal rule agrees with the planned 2,000-draw Monte Carlo
to within 0.0014.

At 0.5 × B\* (McNemar, secondary): against the probe, 37 discordant problems
(21 favour the readout), p = 0.51; against convergence, 41 discordant (28 to
13), p = 0.028.

### Why semantic entropy cannot be compared

82 of the 120 problems have semantic-entropy confidence exactly 1.0, because
all 8 samples agree. Any threshold that stops one of them stops all 82, at
the first allowed boundary. Its smallest positive saving is therefore 0.512
on the menu and 0.613 on all boundaries. Both lie above the oracle bound.
Semantic entropy, as adapted here, **has no operating point at any budget
where safe stopping is possible**. That is also why the registered window sat
at 0.61–0.74. It is a property of the signal on this model, not of the sweep.

### Pre-specified verdict (plan §7)

| item | reading |
|---|---|
| Is the comparison informative? | **Yes.** The readout beats the no-signal rule; the probe and convergence do not, significantly. |
| Against semantic entropy | not evaluable |
| Against the probe | **unresolved**: +0.027, CI [−0.026, 0.076]. Neither an advantage nor a 0.05 margin is excluded, and this is not equivalence |
| Against convergence | **established** by the rule, but fragile. Convergence does not beat chance here. It is matched at only 3 budgets, two of them sharing one operating point. And it was matchable at 3 budgets in only **234 of 2,000** bootstrap replicates, so its interval and p-value are conditional on that 12% |
| **Overall** | **unresolved** |

### Secondary and sensitivity

- **Design B (every rule at every boundary, readout menu leaking).** Here the
  *probe* beats no-signal (+0.045 [0.013, 0.078], Holm p 0.024) and the
  readout does not (+0.029 [−0.006, 0.064]). NLA − probe is −0.016
  [−0.069, 0.032]. The ordering of the readout and the probe flips between
  designs, which is itself evidence that neither is shown to beat the other.
- **Strata (design A, descriptive).** NLA − no-signal is +0.059
  [0.021, 0.096] on GSM8K (74 problems) and +0.050 [−0.020, 0.105] on MATH
  algebra (46).
- **Instrument.** Of the readout's stops at 0.5 × B\*, 28% land on a forced
  completion that the 48-token budget cut off, against 15% for the probe. That
  is a measurement asymmetry the registered forcing budget creates. Its
  direction is not determined.

### What Stage 1 establishes

- **Established, post hoc:** at budgets where safe stopping is possible, and
  given the same positions to stop at, the readout's reconstruction-cosine
  score selects safer stops than chance (+0.052 [0.022, 0.082]). This is the
  first evidence in the study that the readout carries stopping-relevant
  information.
- **Not established:** incremental value over the hidden-state probe. The
  gap is +0.027, CI [−0.026, 0.076], and the sign reverses in design B. The
  per-budget gaps are small and mostly positive (−0.008 to +0.050), but no
  interval excludes 0.
- **Cannot be assessed:** value over semantic entropy, which cannot operate
  below the oracle bound on this corpus.
- **Not established either:** value of the *verbalisation's text*. The
  score used is the registered reduction (reconstruction cosine). Stage 4
  shows the text carries next-token information, which a different, newly
  registered reduction could use.

## Stage 2: experimental coverage (D73)

`scripts/coverage_robustness.py` → `runs/main_coverage/`; the 240-problem
arm through `reanalyse_o3_registered.py --run-id main_nla240` →
`runs/main_nla240_o3_registered/`. The full tables and the replication
design are in [`COVERAGE_PLAN.md`](COVERAGE_PLAN.md).

- **Robust:** the readout's score beats chance under each verbaliser seed
  alone (+0.044, +0.051), on the 240-problem arm (+0.069 [0.049, 0.093]), in
  design B there (+0.039 [0.015, 0.068]), and in GSM8K and algebra
  separately.
- **Not robust:** the readout beats the probe only on the post-hoc
  240-problem arm in design A (+0.057 [0.023, 0.091]). The advantage falls
  to +0.006 [−0.045, 0.059] against a layer-24 probe, which itself beats
  chance (+0.046 [0.005, 0.082]), and it vanishes in design B (+0.009).
- **Uninformative:** counting and probability, 23 problems in the
  exploratory arm only (+0.037, CI ±0.09).
- **Untested:** another model, another autoencoder layer, fresh problems.
  The ranked replication design is in `COVERAGE_PLAN.md` §3–§4; the top
  item, about 200–900 problems at 20–90 GPU-hours, needs a decision.

## Stage 3: the RQ2 paraphrase null (D70)

`scripts/audit_paraphrase_null.py` → `runs/main_paraphrase_qc/paraphrase_qc.json`

**Defect, measured.** The paraphrases were never stored. They were
regenerated deterministically, and two checks passed before any number was
used:
- all 6,007 regenerated claims equal the stored ones;
- the 5,266 draws classified `identical` are exactly the 5,266 draws whose
  stored effect is 0.

| class (D70) | share of 18,021 draws | source | valid as a null draw? |
|---|---|---|---|
| identical | 0.292 | variant 0 when no synonym matches (88% of variant 0) | no |
| normalised identical | 0.009 | variant 2's reordering yields the same words | no |
| prefix only ("In other words, …") | 0.414 | variant 1; variant 2 without a comma | yes: surface addition |
| reorder around a comma | 0.244 | variant 2 | yes as a perturbation; meaning **unverified** (it splits "1,000") |
| synonym swap | 0.041 | variant 0 when a synonym matches | yes |
| content changed | 0.000 | none | no |

Mean token change is 0 for identical draws, 0.036 for swaps, 0.191 for
prefixes and 0.283 for reorders.

**Reanalysis under the nulls fixed before it was run** (problem-cluster
bootstrap CIs; no claim became non-evaluable):

| null | median threshold | dependent rate | dependent and not noise | deletion − paraphrase |
|---|---|---|---|---|
| registered (all draws) | 0.0044 | 0.750 [0.734, 0.766] | 0.749 [0.733, 0.765] | 0.0324 [0.0301, 0.0347] |
| N1: valid draws | 0.0047 | **0.730** [0.713, 0.748] | 0.729 [0.712, 0.747] | 0.0317 [0.0295, 0.0340] |
| N2: surface-verified draws | 0.0032 | **0.775** [0.759, 0.790] | 0.773 [0.757, 0.789] | 0.0318 [0.0295, 0.0340] |

Under every null, all 120 problems have a positive mean.

**Conclusion.**
- The registered reading, that about three quarters of claims carry
  reconstruction weight beyond surface edits, survives removing the invalid
  draws: 0.73–0.78 against 0.75.
- The null itself is narrow. 29% of it is no edit at all, and the valid draws
  are mostly an added discourse marker. It shows that deletion moves
  reconstruction more than *surface* edits do. It cannot show that the
  reconstructor ignores *meaning-preserving rewording*, because no draw is a
  real rewording.
- **Still needed** to test that: paraphrases that genuinely reword, judged
  meaning-preserving by people or a validated model, then about 18,000
  reconstructor passes on the GPU. Not done.

## Stage 4: what the claims say (D71)

`scripts/validate_claim_semantics.py` → `runs/main_claim_checks/claim_checks.json`;
the protocol is in `docs/CLAIM_VALIDATION_PROTOCOL.md`.

**Validated objectively:** three claim templates, checked by string
comparison against the trace, in all 1,413 explanations with no selection,
each against three controls. "Specific" means above all three controls, with
problem-cluster CIs excluding 0.
- **Predictions of the next text are content-specific and often right.** The
  first word of a stated expectation is the word the model actually wrote next
  in 52.2% of explanations. The controls give 10.1% (mismatched trace), 7.8%
  (shifted position) and 3.4% (Gaussian). Real − mismatched = +0.421
  [0.376, 0.465].
- **Final-token claims get the form right and the word wrong.** The last
  character matches in 73.3% of explanations (controls 51.0%, 55.8%, 3.2%).
  The quoted word ends the context in only 2.1% (controls 0.5%, 0.1%, 0.0%).
- **"Quoted" context is almost never verbatim:** 0.3% at ≥ 80% contiguous
  support, which is not above the controls. Mean contiguous support is only
  slightly above them (0.186 against 0.155–0.167).
- No explicit mechanistic assertion matched the lexicon screen. The screen is
  narrow, so this is not a validated absence.

**What this establishes.** For the first time in the study, some
verbalisation content is shown to be *accurate* about the activation's
position, at a level controls cannot explain. That is P3 in the protocol's
terms, in a behavioural sense: predictions agree with what the model then
wrote. It does not establish faithfulness to the computation (P4). It does
not cover interpretive claims ("signals a concluding summation"). And it
shows the verbaliser presenting invented quotations, which bears on how its
text can be read.

**Not validated (human protocol, not run).** Understandability, support for
interpretive content, and unjustified mechanism. The blinded sheets exist
(`runs/main_claim_annotation/`, 92 development and 219 held-out claims with
blind controls), but no labels exist, and none were produced by a model.

## Stage 5: F10 and RQ2 at the right unit (D72)

`scripts/corrected_f10_rq2.py` → `runs/main_corrected_tests/corrected_tests.json`

| contrast | registered (pooled) | problem-level mean [95% CI] | Wilcoxon, problem means | mixed model [95% CI], p |
|---|---|---|---|---|
| F10 tail − matched-position | δ −0.203, p 6e-8 (946 samples) | −0.0052 [−0.0118, 0.0016]; d_z −0.14 | 0.012 | −0.0050 [−0.0129, 0.0029], 0.21 |
| F10 tail − matched-length | δ −0.137, p 3e-4 (948) | −0.0031 [−0.0088, 0.0027]; d_z −0.10 | 0.053 | −0.0029 [−0.0106, 0.0048], 0.46 |
| F10 tail − Gaussian | δ 1.00 | +0.479 [0.472, 0.486] | 2e-21 | +0.480 [0.472, 0.487], ≈ 0 |
| RQ2 deletion − paraphrase | d 0.712, p ≈ 0 (6,007 claims) | claim-weighted 0.0324 [0.0301, 0.0347]; problem d_z 2.46 | 2e-21 (120/120 positive) | 0.0367 [0.0300, 0.0435], 2e-26 |

**Significant tests in the BH family**, under the evaluable and
planned-family conventions:

| version | evaluable | planned family |
|---|---|---|
| registered | 5 | 5 |
| problem-level Wilcoxon | 4 | 3 |
| mixed model | 2 | 2 |

**Impact.**
- RQ2 is unchanged in substance.
- F10's reading that *fidelity is not higher in the tail* holds.
- "Tail reconstructs significantly worse than matched windows" is not
  robust: the mean-difference intervals include 0, and the rank test
  disagrees for matched-position.
- RQ1 is significant under some corrections and not others, so it is **not
  robustly significant**.

The choice between the Wilcoxon test and the mixed model was not fixed
before the numbers were seen, so the disagreement is reported, not resolved.
