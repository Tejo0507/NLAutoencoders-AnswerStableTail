# Experimental coverage: what was tested, and the replication that would matter (D73)

**Status.** §1–§2 were run, CPU only, from stored artifacts
(`scripts/coverage_robustness.py`, `scripts/reanalyse_o3_registered.py
--run-id main_nla240`). §3 is a design, **not run**.

## 1. What the study covers, and what can vary without new inference

| dimension | as run | variable without new inference? |
|---|---|---|
| model | Qwen2.5-7B-Instruct, NF4 | no: the autoencoder is released for this model and layer only. Whether other released checkpoints exist was not verified from this machine |
| autoencoder layer | 20 only | no, for the readout. The probe's layer, yes: activations are stored at 14, 20 and 24 |
| generation seed | one | **largely moot.** The traces are greedy (`canonical trace: greedy`), so the seed matters only for the stochastic parts: AST continuations, semantic-entropy samples and verbaliser samples |
| verbaliser sampling | 2 samples per window, different seeds | **yes**: one seed replicate each |
| benchmark composition | NLA arm: 74 GSM8K + 46 MATH algebra; no counting and probability | **partly**: the exploratory 240-problem arm (D54) includes 23 counting-and-probability problems |
| problems | 120 (registered) and 240 (exploratory, post hoc) | the paired problem bootstrap covers problem sampling |

## 2. Robustness results (exploratory, post hoc)

Design A of the registered-outcome reanalysis (shared menu, budgets below
the oracle bound); mean safe-rate gaps with 95% bootstrap CIs.

| variant | NLA − no signal | NLA − probe | probe − no signal |
|---|---|---|---|
| main, as analysed | +0.052 [0.021, 0.082] | +0.027 [−0.023, 0.078] | +0.025 [−0.013, 0.061] |
| verbaliser sample 0 only | +0.044 [0.015, 0.074] | +0.019 [−0.032, 0.069] | +0.025 |
| verbaliser sample 1 only | +0.051 [0.021, 0.083] | +0.026 [−0.022, 0.076] | +0.025 |
| probe refitted at layer 14 | +0.052 | +0.019 [−0.031, 0.075] | +0.033 [−0.009, 0.071] |
| probe refitted at layer 24 | +0.052 | **+0.006 [−0.045, 0.059]** | **+0.046 [0.005, 0.082]** |
| 240-problem arm (D54) | **+0.069 [0.049, 0.093]** | **+0.057 [0.023, 0.091]** | +0.012 [−0.014, 0.038] |
| 240-problem arm, design B | +0.039 [0.015, 0.068] | +0.009 [−0.021, 0.047] | +0.030 [0.004, 0.052] |

The 240-problem arm by subject (design A):

| subject | problems | NLA − no signal | NLA − probe |
|---|---|---|---|
| GSM8K | 148 | +0.073 [0.049, 0.100] | +0.061 [0.015, 0.105] |
| MATH algebra | 69 | +0.074 [0.028, 0.114] | +0.052 [−0.008, 0.124] |
| MATH counting and probability | 23 | +0.037 [−0.040, 0.132] | +0.043 [−0.080, 0.166] |

**Check.** Refitting the probe at layer 20 by the registered procedure
reproduces all 1,255 stored scores exactly.

**What this establishes.**
- **Robust:** the readout's score beats chance at budgets below the oracle
  bound. It holds under each verbaliser seed alone, in both designs on the
  240-problem arm, and in GSM8K and algebra separately.
- **Not robust:** the readout beating the probe. It depends on the probe's
  layer (+0.006 at layer 24, where the probe itself beats chance), on the
  design (it vanishes in design B), and on the problem set. The only
  significant result comes from the post-hoc 240-problem arm. It is a lead,
  not a finding.
- **Uninformative:** counting and probability (23 problems). The CI is
  ±0.09.
- **Untested:** another model, another autoencoder layer, and fresh problems.

## 3. Ranked next experiments

| rank | experiment | uncertainty it addresses | feasible here? | cost (RTX 4050, NF4) |
|---|---|---|---|---|
| 1 | **Confirmatory design-A comparison on fresh problems**, preregistered: readout vs probe vs no-signal at a shared menu, registered outcome, budgets below the oracle bound; probe layer fixed in advance (or all three with Holm); stratified to include counting and probability | whether the readout adds anything over the probe, the only open question with a live lead | yes | about 6 GPU-min per problem: traces with AST evidence ~3.5, verbaliser ~1.6, forcing every boundary ~0.9. N from §4: about 200–900 problems, **20–90 GPU-hours** |
| 2 | Same, with a **text-based reduction** of the verbalisation, registered in advance (e.g. stated expectations; D71 shows they predict the next text) | whether the verbalisation's *content*, not only its reconstruction cosine, carries stopping information | yes, on CPU from the stored verbalisations, but only as exploratory work on these problems; confirmatory only on rank 1's fresh problems | CPU; nothing extra on top of rank 1 |
| 3 | A second model with a released autoencoder | generality across models | **unknown**: needs checkpoints not verified to exist, and disk for a second set (about 15 GB NF4) | the full pipeline again, 50+ GPU-hours |
| 4 | Another generation seed | low: traces are greedy | yes | only the stochastic stages; not recommended |

## 4. Sizing rank 1

Precision observed on design A: the CI half-width for NLA − probe is about
0.050 at N = 120 (SE about 0.026) and about 0.034 at N = 240. Treating SE as
proportional to 1/√N, 80% power at two-sided α = 0.05 needs
SE ≤ effect / 2.8:

| assumed true gap | SE needed | N |
|---|---|---|
| 0.057 (the post-hoc 240-arm estimate, likely inflated) | 0.020 | ≈ 200 |
| 0.040 | 0.014 | ≈ 400 |
| 0.027 (the registered 120 estimate) | 0.0096 | ≈ 880 |

At 6 GPU-minutes per problem, N = 400 is about 40 GPU-hours on this machine.

**Unresolved before a run:**
- which probe layer is the comparator. It decides the answer at N = 120, so
  it must be fixed in advance;
- whether the margin of practical interest is 0.03 or 0.05;
- whether the readout's 48-token-cut-off asymmetry (28% of its stops, against
  15% for the probe) should be removed by forcing with a larger budget. That
  changes the registered instrument and would itself need a decision.
