# Stopping-policy pilot: protocol draft (2026-10-23)

> **STATUS: DRAFT. NOT FROZEN. NOT APPROVED. NOT RUN.**
> - It prepares the pilot that `docs/STOPPING_POLICY_PROPOSAL.md` §6.8
>   suggests. It changes no registered plan and no stored output.
> - Everything in §2 is taken from the proposal's adopted choices. Every
>   item in §3 is a *recommendation with its justification*, not a choice.
>   §5 lists the decisions the researcher must make before anything is
>   frozen or run.
> - No evaluation data exist. Nothing here was chosen after looking at
>   outcomes.
> - This study is separate from Stage 0 and from D76. Neither is run as part
>   of it.

## 1. What the pilot is for, and what it cannot show

**Purpose (exploratory):** measure, on fresh problems with outcome-blind
candidate positions and the 256-token safety outcome, the *out-of-sample*
quantities that the proposal's sample-size tables assume:
- the realised unsafe-stop rate r of the same-label probe (P1) at a
  threshold calibrated on separate problems;
- its coverage and mean net saving at that threshold;
- the same for position-only (P0), and σ_d, the per-problem SD of the P1 − P0
  net-saving difference;
- the attrition and instrument rates of the full every-boundary walk.

**It cannot show:**
- **Safety.** A favourable pilot is not a safety certification. At pilot
  size, the safety gate has little power. With 96–150 stopped problems and
  a true risk of 0.025, the probability that the 95% Clopper–Pearson upper
  bound falls at or below 0.05 is only 0.20–0.30 (§4.2). A failed gate in
  the pilot is therefore not evidence of unsafety either.
- **Faithfulness** of any explanation. The NLA is not part of the pilot
  (§3.7).
- **Generality** beyond Qwen2.5-7B-Instruct, layer 20 and these benchmarks.

The pilot's output is a planning input for a later confirmatory study,
with intervals. It produces no confirmatory verdict on C1 or C2.

## 2. Fixed by the proposal (adopted choices, §0)

| item | value | source |
|---|---|---|
| primary comparison | **P1 same-label probe vs P0 position-only**, at matched risk | proposal §2, §4 (C2) |
| P1 | logistic probe on the layer-20 residual activation at each boundary, trained for the 256-token safety label | §2 |
| P0 | monotone in prefix tokens; direction chosen on training data only | §2 |
| candidate positions | **every** chunk boundary, walked in order; stop at the first crossing | §3, choice 0.5 |
| safety outcome | the 256-token forced answer has the same correctness as the trace's final answer (the registered definition under O256) | §3 |
| still cut off at 256 | **primary: unsafe.** Pre-specified sensitivity: as stored (O256) and as no answer (O256-u). The truncated share at each policy's stops is always reported | choice 0.7 |
| utility | net saving = max(0, 1 − (p + f + c_signal)/T), with f the **exact** generated count recorded at generation, never re-tokenised | §5, choice 0.6 |
| gates | safety and utility are read separately, safety first | §8, choice 0.8 |
| instrument checks | re-forcing agreement ≥ 0.95 on a seeded sample; 256-token stability ≥ 0.95 on finished boundaries | §3 |
| leakage controls | problem-disjoint training, calibration and evaluation, enforced by id and recorded by hash; prefix-only features; thresholds frozen before evaluation labels are computed | §3 |

## 3. Recommendations for the open items (each needs approval)

### 3.1 Training, calibration and evaluation data

**Recommendation:**
- **Training (P0 direction, P1 weights, regularisation by grouped CV):**
  rep1 Stage 1's 1,600 outcome-blind positions in 400 problems, with their
  stored 256-token labels.
- **Calibration:** a **fresh** set *C*, walked at every boundary with
  256-token labels.
- **Evaluation:** a second fresh set *E*, disjoint from *C* and from every
  earlier run.

**Why not calibrate on held-out rep1 Stage 1 data**, as the proposal
allows: Stage 1 has 4 random positions per problem. A policy that stops at
the *first* crossing over every boundary has a different selective risk
from one scored on 4 thinned positions. Calibrating on the thinned walk and
evaluating on the full walk would be a design mismatch, not just a smaller
calibration set. The in-sample replay (proposal §6.7) has the same
limitation, which is one reason its numbers are planning inputs only.

**Alternative (cheaper):** split rep1's 400 problems by a seeded hash into
training (Stage 1 positions) and calibration halves. Force every boundary
of the calibration half at 256 tokens on its stored traces, reusing
finished 48-token answers as D82 does. This avoids generating new traces
for calibration, but halves the probe's training data.

**Fresh pool, checked read-only on 2026-10-23:** after excluding every
problem id used by any run under `runs/` (678 distinct ids), the test
splits hold **909 GSM8K** and **1,401 MATH** (algebra, counting and
probability) problems. Sets *C* and *E* are drawn in the registered hash
order by `prepare_fresh_problems.py`. *C* is drawn first and *E* next, so
*E* excludes *C*.

### 3.2 Threshold selection

**Recommendation: fixed-sequence testing over a pre-registered grid**
(Learn-then-Test style, cf. [29]).
- **Grid:** 200 thresholds at the quantiles of P1's out-of-fold training
  scores (and likewise for P0), fixed and hashed before *C* is labelled.
- **Rule:** scan from the strictest threshold to the most permissive. Keep
  going while the one-sided 95% Clopper–Pearson upper bound on *C*'s
  selective risk is ≤ α_cal. τ\* is the last threshold that passed.
- **If no threshold passes**, the policy never stops. That is reported as
  coverage 0, not as a safety success.
- **τ\* is frozen and hashed before any evaluation label exists.** It is
  never revisited after *E* is labelled.

### 3.3 α and α_cal

- **α = 0.05 remains provisional.** The record has no justification for it
  from the cost of a wrong early answer in an intended use, and the
  proposal (§0.1) forbids choosing it for affordability. This is the
  researcher's decision (§5, D1).
- **α_cal:** 0.025 is the replay's candidate (in-sample coverage 0.81,
  r/α ≈ 0.49). Under the rule in §3.2, it is only *attainable* when the
  calibration set has enough stopped problems. Even with zero unsafe stops,
  the Clopper–Pearson bound reaches 0.025 only at **n ≥ 119** stopped
  problems. With 2–3 unsafe stops it needs about 200–300 (§4.2). So
  choosing α_cal also fixes the minimum size of *C*.

### 3.4 Evaluation sample and the uncertainty it targets

**Recommendation: N_E = 150 fresh problems** (90 GSM8K, 60 MATH, the
registered 0.6/0.4 ratio). At the replay's coverage of 0.81 that is about
120 stopped problems.
- **What it estimates:** r to within roughly ±0.03 (for example, 3 unsafe
  of 120 gives r̂ = 0.025 with a 95% interval of [0.005, 0.071]),
  coverage to within about ±0.07, and σ_d.
- **What it is for:** deciding whether the confirmatory study's N should be
  planned at r/α ≈ 0.5 (about 450–730 evaluation problems at α = 0.05,
  proposal §6.8) or higher. It is not enough to pass a safety gate (§1).

### 3.5 Calibration sample

**Recommendation: N_C = 300 fresh problems** if α_cal = 0.025. At coverage
about 0.8, that is about 240 stopped problems, so the bound can reach
0.025 with at most 1 unsafe stop (2 gives about 0.026). With α_cal =
0.0167, N_C would have to be about 400 or more. A larger N_C (about 400)
would allow 2–3 unsafe stops at α_cal = 0.025.

### 3.6 Compute budget (local GPU only; no cloud spending)

The per-problem cost is from rep1's measured rates (proposal §6.8): a
trace takes about 89 s; 256-token forcing at about 11 boundaries takes
about 6.5 s each; activations take about 1.5 s. That is about **2.7
GPU-minutes per problem**, or about 4.5 GPU-hours per 100 problems.

| design | problems generated | GPU-hours at rep1 rates |
|---|---|---|
| fresh *C* (300) + fresh *E* (150) | 450 | **about 20** |
| rep1 calibration half (forcing only, about 2,200 boundaries, some reused) + fresh *E* (150) | 150 traces | **about 9–11** |
| proposal §6.8's "pilot" (evaluation only, 100–150) | 100–150 | 4.5–7 (calibration not counted) |

- **Hardware:** one RTX 4050 Laptop GPU (6 GB), local. There is no cloud
  spending and no purchase.
- **Rates are not guaranteed.** When another application holds VRAM, the
  7B model spills into shared memory and runs several times slower (D89).
  On 2026-10-23 the D76 run drew about 37 W, with about 530 MiB spilled,
  and took about 101 minutes against a 30-minute estimate (D97). Part of
  that gap is the estimate itself.
- **Budget cap (recommended):** the approved GPU-hours × 1.5, then stop
  and report (§3.8).

### 3.7 P3 (NLA) scope

**Recommendation: none in the pilot.** One check costs 0.83 of a mean
trace (claim matrix D2), and Stage 1 already answered the information
question (R6). Including P3 would roughly double the compute without
informing the sizing question.

### 3.8 Abort criteria (fixed before running)

Stop, record, and report without reading outcomes if any of these occurs:
- **Data:** any id in *C* or *E* overlaps an earlier run, or *C* and *E*
  overlap (checked by `prepare_fresh_problems.py` and by hash);
- **Instrument:** re-forcing agreement < 0.95, or 256-token stability
  < 0.95;
- **Attrition:** fewer than 75% of planned boundaries have a valid signal
  and label;
- **Token accounting:** any forced-token count is not the generated count
  recorded at generation (the D82 failure mode);
- **Compute:** GPU-hours exceed the approved cap;
- **Protocol:** any change to the frozen plan, grid, thresholds or code
  after *C* is labelled, unless recorded as a dated deviation before *E* is
  labelled. A deviation after *E* is labelled makes the affected analysis
  post hoc, and it is labelled so;
- **Integrity:** the run would write into an existing run directory, or an
  input hash changes.

### 3.9 From pilot to confirmatory

The pilot's outputs (r̂, coverage, σ_d and their intervals) feed the
proposal's §6.4 and §6.8 tables. A confirmatory study is warranted only if:
- the pilot's 95% upper bound on r at τ\* is below α (otherwise
  out-of-sample risk is too close to α to size a study affordably); and
- P1's coverage at τ\* is materially above 0 and above P0's.

The confirmatory study then needs its own frozen plan, a fresh evaluation
set disjoint from the pilot's *C* and *E*, and N chosen at a sawtooth
peak (proposal §6.4). The pilot's *E* is never reused as confirmatory
evaluation data.

## 4. Planning numbers (computed 2026-10-23; no data)

### 4.1 Fresh pool

909 GSM8K + 1,401 MATH unused problems (§3.1).

### 4.2 Clopper–Pearson arithmetic

One-sided 95% upper bound on selective risk, for x unsafe of n stopped:

| n stopped | x = 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 96 | 0.031 | 0.048 | 0.064 | 0.079 | 0.093 | 0.106 |
| 120 | 0.025 | 0.039 | 0.052 | 0.063 | 0.075 | 0.086 |
| 150 | 0.020 | 0.031 | 0.041 | 0.051 | 0.060 | 0.069 |
| 200 | 0.015 | 0.023 | 0.031 | 0.038 | 0.045 | 0.052 |
| 300 | 0.010 | 0.016 | 0.021 | 0.026 | 0.030 | 0.035 |

Smallest n stopped at which the bound reaches a target with zero unsafe
stops: 59 for 0.05, 119 for 0.025, 178 for 0.0167, and 239 for 0.0125.

Probability that the bound is ≤ 0.05 (the power of a pilot-sized safety
gate):

| n stopped | r = 0.0125 | 0.025 | 0.0375 |
|---|---|---|---|
| 96 | 0.66 | 0.30 | 0.12 |
| 120 | 0.56 | 0.20 | 0.06 |
| 150 | 0.71 | 0.27 | 0.08 |

(Power is a sawtooth in n with an exact bound, as the proposal's §6.4
notes.)

## 5. Decisions required before freezing (the researcher's)

| # | decision | recommendation here | why it cannot be settled from the record |
|---|---|---|---|
| D1 | **α** and its justification | none; 0.05 remains provisional | It must rest on the cost of a wrong early answer in an intended use (proposal §0.1). The record names no use |
| D2 | **α_cal** | 0.025, if D3 funds N_C ≈ 300 | It fixes the minimum calibration size (§3.3) and trades coverage for safety margin |
| D3 | **Calibration source**: fresh *C*, or a rep1 calibration half | fresh *C* (cleaner); rep1 half if compute is tight | A scientific and budget trade-off (§3.1, §3.6) |
| D4 | **Threshold rule and grid** | fixed-sequence Clopper–Pearson over 200 training-score quantiles (§3.2) | The proposal leaves it open (§9.4) |
| D5 | **N_E and N_C** | 150 and 300 | They follow from D2 and D3 |
| D6 | **δ_save**, the practical margin for net saving | none; the replay's in-sample savings are about 0.05–0.08, so any margin must be judged against that size | It must be argued from the use case (proposal §9.6) |
| D7 | **GPU-hour budget and cap** | the table in §3.6, cap × 1.5 | It is an approval |
| D8 | **Pilot first, or go straight to confirmatory** | pilot first | The proposal leaves it open (§6.8) |
| D9 | **Run conditions**: quit OMEN Command Center before the run (D89) | yes | It is an action on the researcher's machine |

**Nothing is run until D1–D9 are recorded in `docs/DECISIONS.md`, and the
protocol, grid and analysis code are frozen and hashed.**
