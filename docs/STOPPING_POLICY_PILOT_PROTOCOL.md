# Stopping-policy pilot: protocol draft (2026-10-23)

> **STATUS: FROZEN AND APPROVED 2026-10-25 (D99).** The researcher made
> decisions D1–D9 (§5); their values are in
> `configs/stopping_pilot_protocol.json` (SHA-256 `79c51284…dc01`): α =
> α_cal = 0.05 for this exploratory pilot only, SGR with δ = 0.05, N_C =
> 300, N_E = 150, fresh calibration problems, δ_save = 0.02, a 9 GPU-hour
> budget with a 13.5-hour cap, pilot first, OMEN quit. The text below is the
> design as frozen; the "draft" wording in it is historical.
>
> Earlier status: draft, not frozen, not approved, not run. Revised 2026-10-24
> before any pilot data: SGR replaces fixed-sequence testing (§3.2), the
> compute estimate is re-derived from measured rates (§3.6), and the code
> and runbook exist (`scripts/stopping_pilot.py`,
> `docs/STOPPING_POLICY_PILOT_RUNBOOK.md`).
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

**Revised 2026-10-24 (before any pilot data): Selection with Guaranteed
Risk (SGR; Geifman and El-Yaniv, NeurIPS 2017).** The earlier draft
recommended fixed-sequence testing from the strictest threshold. That
fails by construction here: at strict thresholds few problems stop, and
the Clopper–Pearson bound on a handful of stops is wide (5 stops with no
unsafe one give an upper bound of 0.45), so the sequence would stop at its
first step and return no threshold almost regardless of the data.
- **Grid:** 200 quantiles of the policy's scores on *C*, as SGR searches
  over the calibration sample's own confidence values.
- **Rule:** a binary search over the grid with k = ⌈log₂ 200⌉ = 8 steps.
  At each step, compute the one-sided Clopper–Pearson upper bound on *C*'s
  selective risk at confidence 1 − δ/k. Accept the threshold if the bound
  is ≤ α_cal (then try a more permissive one), otherwise try a stricter
  one. τ\* is the most permissive accepted threshold. By the union bound,
  all k bounds hold together with probability ≥ 1 − δ, so the guarantee
  does not depend on risk being monotone in the threshold.
- **If no threshold is accepted**, the policy never stops. That is reported
  as coverage 0, not as a safety success.
- **τ\* is frozen and hashed before any evaluation label exists.**
  `stopping_pilot.py` refuses to force *E* before the calibration exists,
  and refuses to calibrate once *E* has been forced.
- **Decision (D4):** confirm SGR and δ (recommended δ = 0.05).

### 3.3 α and α_cal

- **α = 0.05 remains provisional.** The record has no justification for it
  from the cost of a wrong early answer in an intended use, and the
  proposal (§0.1) forbids choosing it for affordability. This is the
  researcher's decision (§5, D1).
- **α_cal (revised 2026-10-24 for SGR).** The draft's α_cal = 0.025 came
  from point-estimate calibration, where α_cal < α buys a safety margin.
  SGR's bound already gives P(selective risk on the calibration distribution
  > α_cal) ≤ δ, so that margin is built in. Its bounds are also stricter:
  each step uses confidence 1 − δ/8 = 0.99375, not 0.95 (§4.3). At that
  confidence, α_cal = 0.025 needs at least 201 stopped calibration problems
  even with no unsafe stop, and SGR would accept a threshold whose true risk
  is 0.005 only about 30% of the time with 240 stops: practically
  unattainable at N_C = 300. **Recommendation: α_cal = α = 0.05 with SGR at
  δ = 0.05.** With 240 stops it accepts a threshold whose true risk is 0.01
  about 78% of the time. A smaller α_cal would need N_C of about 600–1,000.

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

**Recommendation: N_C = 300 fresh problems, with α_cal = 0.05 under SGR**
(§3.3). At coverage about 0.8, that is about 240 stopped problems; under
SGR's per-step confidence the bound reaches 0.05 with up to 3 unsafe stops
at 210 or more stops (§4.3). If coverage at the accepted threshold is
lower, fewer problems stop, and the chance of accepting any threshold
falls; that is reported, not repaired.

### 3.6 Compute budget (local GPU only; no cloud spending)

**Revised 2026-10-24 from measured rates.** The earlier figure (about 2.7
GPU-minutes per problem) used rep1's trace time of about 89 s, which
included the answer-stable-tail scan and its continuations. The pilot needs
neither: with `ast.max_boundaries_evaluated: 0` the traces stage generates
the trace only. Measured or derived per problem:
- the trace: about 329 tokens at 15–25 tokens/s, 13–22 s;
- 256-token forcing at every boundary: 11.1 boundaries per rep1 trace;
  Stage 1's forcing log shows about 8 s per block of 2 at this budget, so
  about 45 s;
- activations: about 1.5 s.

That is **about 60–70 GPU-seconds per problem without VRAM spill**, so
about 1.7–2 GPU-hours per 100 problems.

| design | problems generated | GPU-hours, no spill |
|---|---|---|
| fresh *C* (300) + fresh *E* (150) | 450 | **about 8–9** |
| fresh *C* (200) + fresh *E* (150) | 350 | about 6–7 |
| rep1 calibration half + fresh *E* (150) | 150 traces | about 5 (not implemented; needs code) |

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

**Recommendation: none in the pilot.** One check generates 0.89 of a mean
trace's tokens (claim matrix D2, corrected in D98), and Stage 1 already answered the information
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

### 4.3 Under SGR (confidence 1 − 0.05/8 = 0.99375 per step; computed 2026-10-24)

Smallest number of stopped calibration problems at which the bound reaches
α_cal, with x unsafe stops:

| α_cal | x = 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0.025 | 201 | 284 | 357 | 424 |
| 0.05 | 99 | 141 | 177 | 210 |

Probability that SGR's bound at a given threshold is ≤ α_cal (so that the
threshold can be accepted), by stops n and true selective risk r:

| α_cal | n stopped | r = 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| 0.025 | 240 | 0.30 | 0.09 | 0.01 |
| 0.025 | 320 | 0.52 | 0.17 | 0.01 |
| 0.05 | 160 | 0.81 | 0.52 | 0.17 |
| 0.05 | 240 | 0.97 | 0.78 | 0.29 |
| 0.05 | 320 | 1.00 | 0.96 | 0.54 |

## 5. Decisions required before freezing (the researcher's)

| # | decision | recommendation here | why it cannot be settled from the record |
|---|---|---|---|
| D1 | **α** and its justification | none; 0.05 remains provisional | It must rest on the cost of a wrong early answer in an intended use (proposal §0.1). The record names no use |
| D2 | **α_cal** | α_cal = α = 0.05 under SGR (revised; 0.025 is practically unattainable at N_C = 300, §3.3, §4.3) | It fixes the minimum calibration size and trades coverage for safety margin |
| D3 | **Calibration source**: fresh *C*, or a rep1 calibration half | fresh *C* (cleaner; the only source implemented) | A scientific and budget trade-off (§3.1, §3.6) |
| D4 | **Threshold rule and grid** | SGR binary search over 200 quantiles of the calibration scores, δ = 0.05 (§3.2, revised) | The proposal leaves it open (§9.4) |
| D5 | **N_E and N_C** | 150 and 300 | They follow from D2 and D3 |
| D6 | **δ_save**, the practical margin for net saving | none; the replay's in-sample savings are about 0.05–0.08, so any margin must be judged against that size | It must be argued from the use case (proposal §9.6) |
| D7 | **GPU-hour budget and cap** | the table in §3.6, cap × 1.5 | It is an approval |
| D8 | **Pilot first, or go straight to confirmatory** | pilot first | The proposal leaves it open (§6.8) |
| D9 | **Run conditions**: quit OMEN Command Center before the run (D89) | yes | It is an action on the researcher's machine |

**Nothing is run until D1–D9 are recorded in `docs/DECISIONS.md`, and the
protocol, grid and analysis code are frozen and hashed.**
