# What the stored data could still answer: results of 2026-10-20 (D81, D83)

**All exploratory, all CPU, all from artifacts that already existed.** Every
plan was frozen and hashed before its first quantity was computed. Post-hoc
additions are labelled where they appear (D81 deviations 1–3, D83
deviation 1). Nothing here changes a registered result, D69, D79 or the D75
plan. On `rep1` every analysis below is **pre-specified** and runs after
D75's confirmatory analysis.

`main_nla240`'s 240 problems include `main`'s 120. The two are a consistency
check, not two replications.

## 0. The short version

| question | answer on stored data | strength |
|---|---|---|
| Does the registered readout *score* (reconstruction cosine) carry stopping information beyond position and the probe? | **No.** ΔAUC −0.001 [−0.002, −0.000] | tight interval; menu only |
| Does the explanation *text* carry information about the reasoning state? | **Yes**: +0.073 AUC beyond position and the raw trace text, and the content names a phase ("concluding" against "setting up") | robust to four checks; menu only |
| …beyond a linear probe on the same activation, trained for the same label? | **No.** −0.001 [−0.004, 0.002]; the probe is better head to head | post hoc; menu only |
| Do explanations name the answer before the trace writes it? | **No.** 3.1% against 4.0% for another problem's answer | string matching |
| Does the registered estimand measure stop quality? | **No.** At every boundary, an information-free rule beats an AUC-0.86 probe on it, and selective risk reverses the readout's ranking | Stage 0 has no menu |
| Can an NLA check pay for itself online? | **No.** One check costs 0.83 of a mean trace | analytic |

**What this changes for the paper.** The verbaliser *does* make the model's
state legible: it describes whether the model is concluding or still
setting up. But that information is already linearly readable from the
activation it verbalises. And the registered stopping estimand cannot tell
a good stopping signal from an abandonment strategy. That supports a
narrower, defensible empirical claim (C2–C4 in
`paper/CLAIM_EVIDENCE_MATRIX.md`) and a methodological one (B5–B7).

## 1. The information audit (D81)

**Plan:** `docs/SUPPLEMENTARY_INFORMATION_PLAN.md`, frozen at `acba5597…2aad`;
as amended (deviations 1–3): `5c351b65…6567`.

**Commands:**

```
python scripts/information_audit.py --config main --run-id main_nla240
python scripts/information_audit.py --config main
python scripts/information_audit.py --config main --run-id main_nla240 --post-hoc --acts-dir runs/main/acts
python scripts/information_audit.py --config main --post-hoc
python scripts/information_audit.py --config main --run-id main_nla240 --freeze-lexicon
```

**Outputs:**

| file | SHA-256 |
|---|---|
| `runs/main_nla240_information/information.json` | `726c3176…14f1` |
| `runs/main_nla240_information/information_posthoc.json` | `a72e01e8…a824` |
| `runs/main_nla240_information/lexicon.json` | `11674b0a…b079` |
| `runs/main_information/` (three files) | — |

**Checks passed before anything was reported:**
- every input is byte-identical to the registered analysis's;
- the rebuilt safety labels equal the registered ones;
- the regenerated post-hoc file reproduced every earlier value.

**Units:** 1,206 menu chunks in 240 problems (743 GSM8K, 463 MATH). The
registered probe is missing on 1.2% of them.

**Estimation:**
- out-of-fold predictions, 10 folds grouped by problem;
- a problem-cluster bootstrap, 2,000 replicates;
- Holm over the three primaries.

### 1.1 Primaries (`main_nla240`)

| | contrast | Δ | 95% CI | p (Holm) | reading |
|---|---|---|---|---|---|
| **Q1-P** | readout score over position + probe | −0.0012 AUC | [−0.0023, −0.0001] | 0.052 | no practically relevant increment |
| **Q2-P** | explanation text over position + trace text | **+0.073 AUC** | [0.058, 0.087] | 0.0015 | increment detected |
| **Q3-P** | own answer named, minus another problem's answer | −0.010 | [−0.022, 0.004] | 0.161 | unresolved; no positive evidence |

On `main` (120): −0.002 [−0.004, −0.001], +0.074 [0.055, 0.093] and −0.011
[−0.029, 0.008], with the same readings. `main`'s Q1-P is Holm-significant
*below* zero (p 0.01): adding the score slightly hurts out of sample. It is
still inside the ±0.02 margin. CV-seed range over five other
seeds: Q1-P −0.0015 to −0.0001; Q2-P +0.072 to +0.075.

### 1.2 What each signal holds (`main_nla240`, out-of-fold AUC)

| model | registered safety | tail membership |
|---|---|---|
| position | 0.772 | 0.807 |
| + registered probe | 0.818 | 0.889 |
| + readout score | 0.770 | 0.808 |
| + trace text, 600 characters | 0.780 | 0.836 |
| + whole prefix text | 0.775 | 0.832 |
| + explanation text | 0.860 | 0.917 |
| + noise-vector text (null) | 0.769 | 0.805 |
| **+ same-label activation probe** | **0.889** | **0.948** |
| + activation probe + explanation text | 0.888 | 0.947 |

Within-problem concordance:
- position: 0.948 (safety) and 1.000 (tail);
- the raw readout score: **0.436**, below chance. It *falls* with position,
  so it ranks earlier chunks as safer.

### 1.3 The explanation text under scrutiny (deviations 1–2, post hoc)

| threat | check | result (safety; tail) |
|---|---|---|
| chunks in two windows carry 4 samples, not 2 | one verbalisation row per unit (`E2`) over position + trace text | +0.073; +0.078 |
| 600 characters is a weak text control | the whole prefix in place of 600 characters | +0.078; +0.082 |
| the pipeline leaks through text form | noise-vector verbalisations over position | −0.003; −0.002 |
| **the cheap competitor is the activation itself** | a same-label linear probe on the layer-20 activation (nested-CV C) over position | **+0.118; +0.141** |
| | explanation text over that probe | **−0.001 [−0.004, 0.002]; −0.000 [−0.002, 0.001]** |
| | text against probe, head to head | −0.029 [−0.043, −0.016]; −0.031 [−0.040, −0.022] |
| one strong column could crowd out the text | an unfitted equal-weight ensemble against the probe | −0.005 [−0.010, 0.001]; −0.005 [−0.009, −0.002] |

**Q4 (exploratory subgroups):** GSM8K and MATH agree on all three
primaries. Every GSM8K − MATH difference has a CI containing 0. Text over
probe: −0.000 (GSM8K) and −0.001 (MATH).

**Q5 (reliability):** the two verbaliser samples' cosines correlate 0.94
(Spearman 0.92). 20% of the readout's variance lies between problems. The
score is not uninformative because it is noisy: it is a stable quantity that
carries no extra information.

### 1.4 What the text says: a reasoning phase

The heaviest terms of a model fitted on all units, for tail membership
(descriptive):
- **tail:** "the final", "answer", "final answer", "answer is",
  "concluding", "thus", "therefore", "result", "closing", "summary",
  "conclusion", "concludes", "final step";
- **before the tail:** "first", "let", "worked example", "ratio", "second",
  "expecting continuation", "establishing", "immediately expecting",
  "formula", "initial", "next".

**Frozen lexicon (deviation 3).** These 40 terms, unchanged, are
`runs/main_nla240_information/lexicon.json`. Positive terms present minus
negative terms present gives AUC 0.925 for tail membership on `main`. That
figure is **in sample**: the lexicon was learned on those problems, and the
file records the fact. **The test is `rep1`**, where nothing is fitted.

### 1.5 No latent answer (Q3)

- 684 menu chunks in 198 problems have a plain-number final answer that is
  not yet in the question or the prefix.
- Explanations name it in 3.1% of them, against 4.0% for another problem's
  answer. Before the tail, the gap is −0.013 [−0.025, 0.001].
- Noise-vector texts name it in 0.3%. That +0.028 gap reflects how many
  numbers real explanations contain, not knowledge of the answer.

## 2. Stage 0: cheap signals at every boundary, with no menu (D83)

**Plan:** `docs/LEAKAGE_RESISTANT_DESIGN.md` §2, frozen at `b26f902d…8a17`;
as amended (deviation 1): `7d9294db…e7ef`.

**Command:** `python scripts/offmenu_stage0.py --config main --forced-dir runs/main_forced`

**Output:** `runs/main_offmenu_stage0/stage0.json` (`176487fd…7efa`). It was
regenerated once to add selective risk, and reproduced every earlier number.

**Units:** all 2,756 chunk boundaries of 250 problems. `main_forced`
completed O48 everywhere, and no unit was chosen with `i*`.

### 2.1 Information

| contrast (out-of-fold ΔAUC) | registered safety | tail membership |
|---|---|---|
| **same-label activation probe over position** (primary) | **+0.091** [0.066, 0.115] (AUC 0.861) | +0.123 [0.102, 0.144] (AUC 0.975) |
| registered probe over position | +0.026 [0.010, 0.042] | +0.064 [0.049, 0.079] |
| raw trace text over position | −0.002 [−0.014, 0.011] | +0.021 [0.012, 0.030] |
| activation probe over registered probe | +0.065 [0.047, 0.081] | +0.059 [0.049, 0.069] |

### 2.2 The same signals as stopping rules, under the registered estimand

The machinery is D69's, over **all** boundaries. The oracle bound is
B\* = 0.475 with registered savings and 0.404 net.

| gap to no signal (mean over budgets) | registered savings | net savings |
|---|---|---|
| learned activation probe | +0.046 [0.017, 0.072] | +0.007 [−0.027, 0.039] |
| registered probe | +0.028 [0.002, 0.051] | +0.012 [−0.017, 0.040] |
| learned position only | −0.057 [−0.078, −0.033] | −0.080 [−0.106, −0.054] |
| rising position rule | −0.035 [−0.059, −0.014] | −0.055 [−0.081, −0.028] |
| **falling position rule (no information)** | **+0.073** [0.050, 0.105] | **+0.098** [0.070, 0.126] |
| activation probe − falling rule | −0.032 [−0.083, 0.009] | **−0.093** [−0.147, −0.043] |

**At 0.5·B\* (net savings):**
- the activation probe stops in 249 problems, 190 of them safely (safe rate
  0.764);
- the falling rule stops in 57, 16 of them safely (safe rate 0.836).

**Reading.** The artefact D79 found on the menu is a property of the
estimand. Matched on mean saving, with abstention counted safe, abandoning
a few problems at their first boundary beats stopping every problem at a
well-predicted point. That holds even for a signal with AUC 0.86.

### 2.3 Selective risk at matched coverage (D83 deviation 1, post hoc)

Among problems where a rule stops before the end, this is the share of
unsafe stops, with the saving among stops beside it. Stage 0, registered
savings:

| coverage 0.5 | selective risk | saving among stops | risk − no signal |
|---|---|---|---|
| no signal | 0.558 | 0.525 | — |
| learned activation probe | **0.016** | 0.026 | −0.542 [−0.581, −0.502] |
| registered probe | 0.008 | 0.051 | −0.550 [−0.588, −0.510] |
| rising position rule | 0.238 | 0.217 | −0.320 [−0.383, −0.236] |
| falling position rule | 0.721 | 0.934 | +0.163 [0.093, 0.241] |

On design A's menu, the same view (`runs/main_nla240_selective/`;
`runs/main_selective/` agrees):

| coverage 0.5 | selective risk | saving among stops | risk − no signal |
|---|---|---|---|
| no signal | 0.519 | 0.413 | — |
| **NLA readout** | **0.658** | 0.678 | **+0.140** [0.048, 0.210] |
| registered probe | 0.008 | 0.021 | −0.510 [−0.542, −0.475] |
| falling position rule | 0.760 | 0.870 | +0.242 [0.170, 0.309] |

**Reading.**
- The readout's apparent advantage under the registered estimand comes from
  stopping early. When it stops, it is *more* often unsafe than chance, at
  every coverage and on both runs.
- Informative state signals (both probes) are almost never wrong when they
  stop. But they stop so late that they save 2–5% of the trace, against an
  oracle bound of about 0.45. A boundary classifier with AUC 0.86 does not
  turn into early, safe stops: its confident-safe region is the end of the
  trace.
- **No single number measures stop quality here.** The paper should show the
  saving–risk trade-off, not rank rules by one estimand.

## 3. Cost: what an NLA check spends

| quantity | value |
|---|---|
| verbaliser sample length, `main_nla240` (2,842 samples) | mean 134 tokens, median 135, none at the 256 cap |
| mean trace | 325 tokens (median 286), 11.0 chunks |
| **one check (2 samples), as a share of a mean trace** | **0.83**, before reconstruction |

An online rule that verbalises at each candidate it passes spends close to a
whole trace per check. On this setup the readout's possible value is
diagnostic or interpretive, not efficiency.

## 4. Ready, waiting for compute or data

| item | state | what it settles |
|---|---|---|
| `rep1` stages after traces (`scripts/rep1_runbook.py`, running in the background, waiting on PID 27992) | gated: ast → acts → baselines → nla → forcing (blocks of 2, 200 re-forced) → D75 confirmatory → D79 audit | the registered H1/H2 |
| `rep1` information audit, post-hoc checks, lexicon transfer, selective risk | pre-specified; CPU; after D75 | whether C1–C5 and B6 replicate on unseen problems |
| D82 extended-budget forcing, `main_nla240` then `rep1` | plan frozen; code tested on an identity fixture; about 1–2 GPU-hours each | whether every O3 reading depends on the 48-token budget |
| D83 Stage 1 (NLA at outcome-blind positions) | designed; **needs review**; 4–6 GPU-hours | whether C4 holds off the menu |

## 5. Limitations that apply to everything above

- **One target model (Qwen2.5-7B-Instruct, NF4), one NLA pair, layer 20.**
- **Menu results are at outcome-selected points.** Only Stage 0 is free of
  that, and it has no NLA.
- **The text is featurised with TF-IDF and logistic regression.** A
  stronger text model might extract more, and would bring its own cost.
- **The probes are linear.** A Δ near 0 says the text adds nothing a linear
  read of the activation lacks. It does not say the verbaliser decodes
  nothing non-linear.
- **The safety label is the 48-token forced outcome.** D82 measures how much
  that matters.
- **Everything is exploratory** until `rep1`'s pre-specified runs and Stage 1
  report.
