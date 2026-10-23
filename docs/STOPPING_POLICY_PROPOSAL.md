# Proposal: a safety-constrained stopping-policy evaluation (draft, 2026-10-23)

> **STATUS: NOT APPROVED. NOT RUN. NOT FROZEN.**
> - This is a proposal for review. It authorises no generation, forcing,
>   training or analysis.
> - It changes no registered plan: not `docs/REPLICATION_PLAN.md`,
>   `docs/EXTENDED_FORCING_PLAN.md`, `docs/LEAKAGE_RESISTANT_DESIGN.md` or
>   any other.
> - Every item marked **DECISION** is the researcher's to make before it
>   could be frozen. Nothing here was chosen after looking at evaluation
>   data, because no evaluation data exist.

## 1. Why this study, and what it cannot show

The replication (D94–D96) left a practical question open.
- The NLA readout's registered advantage holds only on the 48-token outcome
  (D95).
- That advantage cannot be attributed to the explanations' content (D79).
- At outcome-blind positions, a same-label linear probe on the same
  activation carries every bit of the text's predictive information for
  stopping (Stage 1, D96).
- The only rule with low unsafe-stop rates so far is the registered
  correctness probe. Its result is on the outcome-selected menu, with O48
  labels, and it saves little (D95, D96). That is not deployment evidence.

**The question:** with outcome-blind candidate positions and the 256-token
safety outcome, can a stopping policy keep its unsafe-stop rate below a
fixed maximum on unseen problems? And among policies that do, which saves
the most generated tokens once the cost of obtaining its signal is counted?

**What it cannot show:**
- **Faithfulness.** A policy that stops well says nothing about whether any
  explanation describes the computation faithfully. The NLA arm here is a
  predictive comparator only.
- **Online value of the NLA.** One NLA check costs about 0.83 of a mean
  trace in generated tokens (claim matrix D2). The NLA arm is therefore an
  *offline research comparator*. It is never assumed to be a viable online
  policy, and its cost is reported, not waived.

## 2. Policies compared

At each candidate boundary, every policy sees only the trace prefix (and,
where stated, the activations the target model already computed while
writing it). It stops at the first candidate whose score crosses its
threshold τ, and emits the 256-token forced answer from that prefix.

| # | policy | signal | online cost of the signal |
|---|---|---|---|
| P0 | **position-only** | a monotone function of prefix tokens (both directions fitted on training data; the direction is chosen on training data only) | none |
| P1 | **same-label probe** | logistic probe on the layer-20 residual activation at the boundary, trained for the 256-token safety label | one dot product on an activation the forward pass already computes |
| P2 | **registered correctness probe** | the registered layer-20 probe, as stored | as P1 |
| P3 | **NLA readout (offline comparator)** | the readout score, and separately a text model of the verbalisation (as in Stage 1), at the same boundaries | verbalisation (2 × 256 tokens) plus reconstruction: reported, not waived |
| — | no signal | stop at a uniformly random candidate, calibrated to the same rate | none |

- **P2 is valid only where defined:** at boundaries where the registered
  probe has a score. Where it has none, P2 is reported on the subset
  together with that subset's size. It is never compared on a different
  set of boundaries from the others.
- **P0 must be beaten:** a probe that does not beat position-only at
  matched risk has no stopping value here.

## 3. Data, positions, outcome

- **Evaluation set: fresh problems.** They are selected with
  `scripts/prepare_fresh_problems.py`, which excludes every id used by
  `main`, the arms, `rep1` and everything else under `runs/`, and takes
  the next problems in the registered hash order.
  - **DECISION (N_eval):** the number of evaluation problems, and the
    GSM8K/MATH split. §6 gives the precision rationale. Whether the pool
    holds enough unused problems must be checked before freezing.
- **Training and calibration data, never evaluation data:**
  - **training (P0, P1 and P3's text model):** rep1 Stage 1's 1,600
    outcome-blind positions in 400 problems, which already carry 256-token
    labels (`runs/rep1_stage1/`);
  - **calibration (each τ):** a problem-disjoint split. **DECISION:** a
    held-out part of rep1 Stage 1 (which shrinks training), or a separate
    fresh calibration set (which costs more). Recommendation: a separate
    fresh set, so training, calibration and evaluation are disjoint at the
    problem level.
- **Candidate positions, outcome-blind:** **every** chunk boundary of each
  evaluation trace, in order, as an online policy would see them. This is
  not a menu, and K positions are not drawn: the policy walks the trace.
  No position is chosen with reference to the tail, the final answer or
  any outcome.
- **Outcome:** safety at a boundary means that the 256-token forced answer
  (D82, `force_answer_max_new_tokens = 256`) has the same correctness as
  the trace's final answer. This is the registered safety definition,
  under O256.
  - **DECISION:** a completion still cut off at 256 is either counted as
    unsafe (conservative) or has no answer (O256-u). Recommendation:
    unsafe.
- **Instrument checks, before any outcome is used:**
  - re-force a seeded sample of boundaries and require agreement ≥ 0.95
    (as D80);
  - check 256-token stability on finished boundaries (as D82, gate 0.95).
- **Leakage controls:**
  - problems are disjoint across training, calibration and evaluation,
    which is enforced by id and recorded by hash;
  - every feature is computed from the prefix only;
  - the probe's regularisation and P0's direction are chosen by grouped CV
    on training data;
  - no threshold is touched after calibration;
  - the evaluation labels are computed only after the analysis code and
    every threshold are frozen and hashed (the D75 practice).

## 4. Hypotheses and analyses

**Fixed quantities (DECISION, before freezing):**
- **α, the maximum unsafe-stop rate:** the share of *stopped* problems
  whose stop is unsafe (selective risk).
  - The record does not determine α. Possible values are 0.05 or 0.10, and
    the choice changes which policies can stop at all.
- **δ_conf:** the confidence level of the acceptance rule (for example
  95%).
- **Calibration rule:** τ is the most permissive threshold whose one-sided
  upper confidence bound on selective risk, computed on the calibration
  set, is ≤ α. The bound is either a Clopper–Pearson binomial bound or a
  Learn-then-Test style fixed-sequence test over a pre-specified τ grid
  (cf. [29]).
  - **DECISION:** the procedure and the τ grid.

**Confirmatory** (Holm over the family, α_test = 0.05):
- **C1, safety transfer for P1.** On the evaluation set, P1 at its
  calibrated τ has selective risk ≤ α.
  - Test: the one-sided 95% upper bound on evaluation selective risk is
    below α + **ε**, with a paired problem-cluster bootstrap.
  - **DECISION:** ε, the allowed calibration slack; it may be 0.
- **C2, utility at matched risk.** P1 saves more net tokens than P0 when
  both are calibrated to the same α.
  - Estimand: the mean over evaluation problems of net saving, defined in
    §5.
  - Paired problem bootstrap. Supported if Holm-significant with Δ > 0.
    Equivalence is read only from the interval, against the margin
    **δ_save** (DECISION).

**Secondary (pre-specified, not in the verdict):**
- C2 against no signal, and P2 against P0 and P1 on P2's defined subset;
- **matched coverage:** selective risk and saving at fixed coverage (25%,
  50% and 75% of problems stopped), as D83 deviation 1, but at
  outcome-blind positions on O256;
- **matched risk:** saving at each α in a pre-specified grid, for example
  {0.02, 0.05, 0.10}: the risk–saving frontier;
- the realised coverage at α for each policy (a policy that never stops is
  trivially safe, so coverage is always reported);
- strata by dataset.

**Offline comparator (secondary, labelled offline):**
- P3's readout score and text model, against P1 at matched risk on the
  same boundaries, as a re-test of Stage 1 at the policy level;
- P3's net saving with the NLA's cost charged, which is expected to be
  negative (claim matrix D2).

**Exploratory:** other layers, other probe families and ensembles. They
are reported as exploratory and never enter C1 or C2.

## 5. Compute accounting, end to end

For a problem with total trace length T tokens, stopped at a boundary with
prefix length p:

```
net saving = max(0, 1 − (p + f + c_signal) / T)
```

- **f:** the generated tokens of the forced answer.
- **c_signal:** the signal's generated-token cost up to the stop, summed
  over every candidate checked.
  - P0 and the no-signal rule: 0.
  - P1 and P2: 0 generated tokens. The probe's FLOPs are reported
    separately; at one dot product per boundary they are negligible.
  - P3: 2 × the verbaliser's generated tokens per boundary checked, plus
    the reconstruction forward pass, reported as token-equivalent compute.
- **A problem that never stops:** saving 0, and counted as not covered.

**Also reported:**
- wall-clock time per signal, measured on this machine. Rep1 provides
  reference rates: about 12.9 s per verbalised window with 2 samples and
  256 tokens; about 0.4 s per reconstruction; about 6.5 s per 256-token
  forced answer (D95, D96 logs);
- savings under the registered definition (1 − p/T), for comparison with
  D75.

D82's token-count check failed because re-tokenised counts were off by up
to 4–5 tokens (D95, D96). Here f must therefore be the *exact* generated
count, recorded at generation, never a re-tokenised one.

## 6. Precision and sample size

These are planning scenarios, computed 2026-10-23 from the binomial and
normal formulas below. They use no evaluation data, because none exists.
They are inputs to the DECISIONs, not choices.

### 6.1 C1, safety: stopped problems needed

**Method:**
- n_stopped is the smallest n for which the probability that the one-sided
  95% Clopper–Pearson upper bound on selective risk falls at or below α is
  at least 0.80, when the true selective risk is r. It is exact binomial,
  with ε = 0.
- N_eval = n_stopped / coverage.

| α | true risk r | r/α | n_stopped | N_eval at coverage 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|---|---|
| 0.02 | 0.005 | 0.25 | 386 | 1,544 | 772 | 515 |
| 0.02 | 0.010 | 0.50 | 969 | 3,876 | 1,938 | 1,292 |
| 0.02 | 0.015 | 0.75 | 4,318 | 17,272 | 8,636 | 5,758 |
| 0.05 | 0.0125 | 0.25 | 153 | 612 | 306 | 204 |
| 0.05 | 0.025 | 0.50 | 386 | 1,544 | 772 | 515 |
| 0.05 | 0.0375 | 0.75 | 1,679 | 6,716 | 3,358 | 2,239 |
| 0.10 | 0.025 | 0.25 | 61 | 244 | 122 | 82 |
| 0.10 | 0.050 | 0.50 | 179 | 716 | 358 | 239 |
| 0.10 | 0.075 | 0.75 | 803 | 3,212 | 1,606 | 1,071 |

**What the table implies:**
- **The ratio r/α dominates the cost.** If τ is calibrated to the *most
  permissive* threshold that meets α, the true evaluation risk will sit
  close to α (r/α → 1), and C1 becomes very expensive. The design choice
  that avoids this is to calibrate to an internal target α_cal below α
  (for example α/2), which buys the r/α ≈ 0.5 rows at the price of less
  coverage and saving. **DECISION: α_cal.**
- **α = 0.02 is impractical** at this machine's throughput (§7) unless
  coverage is high and r/α is small.
- **Holm is not included here.** If C1 enters a Holm family with C2, its
  first-step level is 0.025. That raises these numbers by roughly 10–20%;
  recompute when the family is fixed.

### 6.2 C2, utility: problems needed for the paired net-saving difference

**Method:** the normal approximation for a paired mean difference, where
σ_d is the per-problem SD of the difference in net saving between P1 and
P0.

| σ_d | δ_save | n for 80% power (two-sided 0.05, true difference = δ_save) | n for a 95% half-width ≤ δ_save / 2 |
|---|---|---|---|
| 0.15 | 0.02 | 442 | 865 |
| 0.15 | 0.05 | 71 | 139 |
| 0.15 | 0.10 | 18 | 35 |
| 0.25 | 0.02 | 1,227 | 2,401 |
| 0.25 | 0.05 | 197 | 385 |
| 0.25 | 0.10 | 50 | 97 |
| 0.35 | 0.02 | 2,404 | 4,706 |
| 0.35 | 0.05 | 385 | 753 |
| 0.35 | 0.10 | 97 | 189 |

**About σ_d:**
- **It is an assumption, not a measurement.** No stored output has
  per-problem net savings at outcome-blind positions on O256. Savings are
  fractions of the trace, so σ_d ≤ about 0.5. The table spans 0.15–0.35.
- **Before freezing,** estimate σ_d on *existing* data only, by replaying P0
  and a same-label probe policy over `main`'s Stage 0 boundaries and rep1
  Stage 1's positions. Then read N from this table. This is a CPU analysis
  of stored data, but it is still analysis work and **needs approval**.

### 6.3 Combining them

- N_eval must meet both C1 and C2. In most plausible cells (α = 0.05,
  coverage 0.5, r/α ≈ 0.5, σ_d about 0.25, δ_save = 0.05), **C1 drives N:
  about 770 evaluation problems.**
- **The calibration set** needs a similar number of stopped problems. That
  is a few hundred to about 800 more problems, unless calibration reuses
  rep1 Stage 1 (DECISION §3).
- **The fresh pool must hold N_eval plus the calibration problems.** This
  is unchecked (§3).

## 7. Cost estimate (planning only; nothing authorised)

The scale column uses N_eval = 600 for illustration only. §6.3's typical
cell needs about 770 evaluation problems plus the calibration set; scale
the GPU-hours linearly.

| step | per unit (rep1 rates) | scale at N_eval = 600, about 11 boundaries per trace |
|---|---|---|
| traces (greedy, 768 tokens) | about 89 s per trace | about 15 GPU-hours |
| 256-token forcing at every boundary | about 6.5 s per boundary (fewer if finished-at-48 boundaries reuse the 48-token answer, as in D82) | up to about 12 GPU-hours |
| activations (layer 20) | 598 s per 400 traces | about 0.3 GPU-hours |
| P3: verbalise and reconstruct every boundary | about 13 s per boundary | about 24 GPU-hours. **DECISION:** run P3 on all of the set, on a subset, or not at all |

The calibration set adds proportionally. Everything is local; there is no
cloud spending.

## 8. Go/no-go

**Before running:**
- every DECISION is resolved;
- the fresh pool is large enough;
- the protocol, the analysis code and the τ grid are frozen and hashed;
- the compute budget is approved.

**Before reading outcomes:**
- both instrument checks pass;
- attrition is ≥ 75% of planned boundaries with a valid signal (as Stage 1).
Otherwise, report "instrument failed" and stop.

**After the analysis:**
- a practical "probe-based early stopping is safe here" claim requires C1
  *and* C2;
- C1 without C2 means safe but not worth more than position;
- C2 without C1 means not safe enough at α;
- neither result bears on faithfulness.

## 9. Open decisions (the researcher's)

1. α (and its grid for the frontier), ε, δ_conf, and the internal
   calibration target α_cal (§6.1: it decides r/α, and with it the cost).
2. The calibration procedure (Clopper–Pearson or Learn-then-Test) and the
   τ grid.
3. N_eval, the calibration set size, the source of the calibration set,
   and the GSM8K/MATH split.
4. Whether a boundary still cut off at 256 counts as unsafe or as no
   answer.
5. δ_save, the practical margin for net saving.
6. P3's scope: everything, a subset, or none.
7. The compute budget.
8. Whether the study is confirmatory (a frozen plan with hashes, as D75) or
   an exploratory pilot first.
9. Whether to approve the CPU replay that estimates σ_d from existing data
   (§6.2) before N is fixed.
