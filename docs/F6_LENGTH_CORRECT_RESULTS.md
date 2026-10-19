# F6 length-correct diagnostic: results (2026-10-18) — exploratory

**Exploratory and descriptive. Separate from F6.** F6's registered result
(real AUC 0.9460944231, across-groups shuffled AUC 0.6142873504, within-group
0.5385), its threshold (|shuffled AUC − 0.5| > 0.15), `leak_suspected = false`
and its interpretation are unchanged, and nothing here is a new leakage
verdict. No formal hypothesis test was run: every number below is a
descriptive summary.

| | |
|---|---|
| Protocol | [`F6_LENGTH_CORRECT_DIAGNOSTIC.md`](F6_LENGTH_CORRECT_DIAGNOSTIC.md), SHA-256 `097cbf8a…6eb1`, approved D58 |
| Implementation | `scripts/diag_f6_length_correct.py`, LF SHA-256 `45c7807b…fdf9` (D59), commit `e884630` |
| Run | `--phase run` at `27c131a`, 2026-10-17 21:07–22:05 UTC, exit 0, `status: complete` |
| Outputs | `runs/main_f6_diag/` (not version-controlled; read-only); hashes in `runs/main_f6_diag_validation/SHA256SUMS.txt` |
| Validation | `runs/main_f6_diag_validation/VALIDATION_LOG.txt`: 27 independent checks, 0 failed |
| Backup | snapshot `runs-2026-10-18-f6-diagnostic` in `D:\` and `E:\NLAutoencoders-backups\` (archive SHA-256 `661ad710…51a5`); and the complete `runs/` directory on Google Drive and at `E:\NLAutoencoders-backups\runs\` (researcher-confirmed 2026-10-18; E: copy verified identical here, Drive not verified from this machine) |
| Registered artifact | `runs/main/robustness/robustness.json` SHA-256 `45701a51…d5ed`, unchanged before, during and after |

## What ran

- **Preflight** passed every check: output directory absent, registered and
  protocol hashes, eight reused registered functions unchanged, script hash
  recorded in D59 and committed, probe settings (logistic, C = 1,
  `max_iter` 2000, standardised, 5 folds, seed 11), no torch. Data: 2,681
  rows, 250 problems; exact-length subset 246 problems, 2,577 rows (excluded:
  four problems of 20, 21, 29 and 34 rows — the four longest traces);
  F6's draw 0 reproduced label for label (718 changed, no fixed point).
- **Seed-12 gate** reproduced F6 exactly: 718 labels changed, shuffled AUC
  0.6142873504351077, real AUC 0.9460944231357181 (both equal to the stored
  values; tolerance 1e-9).
- **Draws:** B = 100. The gate fit took 7.9 s, projecting 4,920 s against the
  6 h limit, so the 50-draw fallback was not used. Every fit scored every row.
  58 minutes in all; no failure.

## Results

Each row: mean over the 100 donor draws; the **spread of single draws**
(2.5th–97.5th percentiles, *not* an interval for the mean); the **Monte Carlo
SE** of that mean (SD/√B, precision over draws only); the **problem-level
cluster bootstrap** (2,000 resamples, no refit) of the draw-averaged value.
The two intervals are separate summaries, not a joint confidence statement,
and neither includes model-refitting variability.

| Statistic | Mean | Single-draw spread | MC SE | Bootstrap 95% | Wording (protocol table) |
|---|---|---|---|---|---|
| Probe, S0 (`np.resize`) | 0.602 | 0.571 – 0.634 | 0.002 | 0.596 – 0.608 | |
| Probe, S1 (length-correct) | 0.841 | 0.818 – 0.861 | 0.001 | 0.834 – 0.849 | |
| **Δ_A = S1 − S0** | **+0.240** | 0.196 – 0.278 (100 % of draws > 0) | 0.002 | 0.233 – 0.247 | consistent in sign across draws; stable under problem resampling |
| Oracle, S0 / S1 | 0.694 / 0.910 | 0.686 – 0.702 / 0.901 – 0.919 | < 0.001 | 0.688 – 0.701 / 0.908 – 0.913 | |
| **E = probe − oracle, S0** | **−0.092** | −0.124 – −0.061 (100 % < 0) | 0.002 | −0.098 – −0.087 | as above |
| **E, S1** | **−0.069** | −0.090 – −0.050 (100 % < 0) | 0.001 | −0.076 – −0.061 | as above |
| Probe's share of the oracle's margin over 0.5, S0 / S1 | 0.52 / 0.83 | 0.37 – 0.68 / 0.78 – 0.88 | | 0.50 – 0.55 / 0.81 – 0.85 | |
| P (linear, four explicit position features), S0 / S1 | 0.681 / 0.916 | | | 0.673 – 0.690 / 0.914 – 0.919 | |
| Subset probe: S0 / S1 / S2 | 0.602 / 0.844 / 0.876 | | | | |
| Subset S2 − S1 | +0.031 | 0.005 – 0.058 (99 % > 0) | 0.001 | 0.023 – 0.039 | as above |
| Subset S1 − S0 | +0.242 | 0.199 – 0.286 | 0.002 | 0.235 – 0.248 | as above |
| Subset S2 − S0 | +0.273 | 0.227 – 0.320 | 0.002 | 0.264 – 0.282 | as above |

The registered 0.6143 is draw 0 of S0 and lies inside S0's single-draw spread.

**Label statistics** (stage 0, before any fit; means over the 100 draws):

| Labelling | Prevalence | Label–relative-position AUC | Monotone problems |
|---|---|---|---|
| Original labels | 0.200 | 0.920 | 0.876 |
| S0 (`np.resize`) | 0.156 | 0.683 | 0.517 |
| S1 (length-correct, φ) | **0.228** | 0.919 | 0.907 |
| S2 (exact length, subset; original subset 0.205) | 0.205 | 0.919 | 0.878 |

At draw 0 both S0 and S1 duplicate and drop 510 labels; S2 none.

## Reading (as fixed by the protocol)

- **Question A — sensitivity to the length-fitting rule.** Given the same
  donors, replacing `np.resize`'s rule with φ raises the shuffled AUC by
  **+0.240**. The registered 0.6143 therefore depends on the fitting rule as
  well as on donor reassignment. This is a property of the control, not a
  finding about leakage, and not a revision of F6. The label statistics show
  the mechanism: tiling and truncation break the label–position structure
  (0.920 → 0.683) that φ keeps (0.919).
- **S1 shifts prevalence.** φ raises prevalence from 0.200 to **0.228**
  (S0 lowers it to 0.156). The protocol describes prevalence as
  "approximately preserved"; at +0.028 it is not, and Δ_A reflects that
  shift as well as the restored position structure. S1 is one reasonable
  length-fitting rule — a sensitivity analysis — not the definitive null.
- **Question A′ — exact length (subset only).** Copying a same-length
  donor's labels verbatim gives +0.031 over S1 on the subset, and the subset's
  S1 − S0 (+0.242) matches the full sample's. The subset excludes the four
  longest traces, so its results describe the subset and cannot separate
  sample composition from the donor constraint; they are not generalised.
- **Question B — the positional oracle.** In every draw and both arms the
  probe scores **below** the oracle (E = −0.092 and −0.069), reaching about
  half (S0) and four fifths (S1) of its margin over 0.5. That is no
  discrepancy from this benchmark. It does **not** establish that position
  explains the probe's predictability, and it does **not** rule out leakage.
  The oracle is a position-derived reference benchmark, not a guaranteed
  upper bound, formal null distribution or leakage test — the linear
  explicit-feature model P scores slightly above it under S1 (0.916 against
  0.910), as the protocol anticipated. No inference is drawn from probe
  against P.
- **Question C — leakage.** No verdict. The registered 0.15 threshold is not
  applied to any diagnostic arm: none of them has an expected AUC of 0.5 in
  the absence of leakage, so values such as 0.84 are not evidence of leakage.

## What this establishes, and what it does not

**Establishes:** F6's across-groups shuffled AUC is highly sensitive to how a
donor's label sequence is fitted to the receiver's length — by +0.24 AUC
between two defensible rules with the same donors — so the registered 0.614
cannot be read as a property of donor reassignment alone. Under every rule
tested, the probe's shuffled AUC stays below the position-derived oracle.

**Relation to F6's registered pass.** `np.resize` removed most of the
label–position structure (0.920 → 0.683) that a length-correct shuffle keeps,
and the shuffle that keeps it scores 0.841. F6's 0.614, and its margin under
the 0.15 threshold, were obtained under the structure-breaking rule. This is
not evidence of leakage — the length-correct arm's expected AUC absent
leakage is not 0.5, and the probe stays below the oracle — but it means the
registered control cannot support a stronger claim than F6 already makes
(see [`F6_RESIZE_AUDIT.md`](F6_RESIZE_AUDIT.md), *Outcome*).

**Does not establish:** whether any leakage exists; whether the probe's
shuffled-label signal is entirely positional; what F6's verdict would be under
a different control. None of these can be settled after the fact, because
every arm here lacks a null centred where the registered threshold assumes.

## Limitations

- No model-refitting variability; fold assignment fixed (hash of problem id).
- One corpus, one layer (20), one probe.
- S1 shifts prevalence (+0.028) and changes run lengths and transitions; it is
  not the only defensible mapping.
- The subset is selected by length (the four longest traces are out).
- The oracle's ranking is not AUC-optimal for dependent within-problem labels;
  it is a benchmark, not a bound.

## What a stronger claim would need

A new, **preregistered** study, designed and recorded before any of its
results are seen — not a re-reading of this one. At minimum: a control whose
null distribution is defined (where its AUC should sit absent leakage); a
length-matching strategy fixed in advance; uncertainty that includes model
refitting (and fold assignment); and a decision threshold stated for that
null. Until then F6's registered result and interpretation stand as recorded,
with this diagnostic and [`F6_RESIZE_AUDIT.md`](F6_RESIZE_AUDIT.md) as
documented caveats. No such study has been launched.
