# Results: the pre-registered replication (draft v2, 2026-10-23)

*Placement: this is the confirmatory core of Results. The outline's §4–§7
were written before rep1 existed and are to be reordered around it. The
subsection numbers here are local.*

*Status markers, as in §1, plus one:*
- **[C]** confirmatory;
- **[S]** pre-specified secondary or sensitivity analysis. It was fixed
  before any rep1 outcome existed, and it does not enter the verdict;
- **[X]** exploratory.

*Two different probes appear below:*
- the **registered probe** is the layer-20 correctness probe of the
  registered plan;
- the **same-label probe** is a linear probe on the same layer-20
  activation, trained for the label being predicted (Stage 1, D81).

*Every number was copied from the output named beside it and checked by
review against that output. The automated check
(`scripts/check_run_consistency.py --replication`) covers a listed subset
of these numbers as they appear in `docs/DECISIONS.md`, not this file.
D96 §4 says exactly what it covers.*

## 5.1 The registered verdict [C]

The replication ran the registered design on 400 problems that no earlier
run had used (§3.1).
- **Instrument and sample:** 380 have an answer-stable tail, against the
  plan's minimum of 200. The forcing instrument reproduced on 0.965 of 200
  re-forced stored boundaries, against the registered 0.95. The analysis
  checked the frozen plan's hash before it ran (D94).
- **Test:** a paired problem bootstrap with 2,000 replicates and Holm
  correction over the two hypotheses. Both are supported on the registered
  48-token forced-answer outcome
  (`runs/rep1_o3_registered/o3_registered.json`):

| hypothesis | Δ (mean safe-rate gap) | 95% CI | Holm p | reading |
|---|---|---|---|---|
| H1 readout − registered probe | +0.047 | [0.019, 0.075] | 0.002 | supported |
| H2 readout − no signal | +0.057 | [0.039, 0.076] | 0.001 | supported |

H1's interval spans the plan's 0.05 margin. The readout's advantage over
the registered probe is significant, but an advantage of 0.05 or more is
neither shown nor excluded.

## 5.2 What qualifies it

**Candidate position [S] (D79, `runs/rep1_position_audit/`).**
- A rule with no access to the activation scores each menu candidate by
  −prefix tokens, so it favours early stops. It beats no signal by +0.089
  [0.066, 0.108], and it beats the readout by 0.025 [0.005, 0.044].
- With its linear-in-log-position trend removed, the readout no longer
  beats no signal: +0.007 [−0.016, 0.029].
- Both families of position readings are *unresolved* by the plan's rules.
- So the registered advantage cannot be attributed to the content of the
  explanations: on this menu, a preference for early positions does as
  well.

**The forcing budget [S] (D82, `runs/rep1_extforced/`).**
- **Re-forcing:** the 661 boundaries cut off at 48 tokens were re-forced
  with 256. The instrument was stable: 1.00 on 50 finished boundaries.
- **f = 0.867.** f is computed over the 600 cut-off boundaries *before the
  tail*. 520 of them (86.7%) reach the final answer when given 256 tokens.
  The other 61 cut-off boundaries are at or after the tail and are not
  part of f.
- By the plan's fixed language, the 48-token outcome mostly measures
  whether the model can *commit within 48 tokens*.
- **Under the 256-token outcome,** both hypotheses read *equivalent within
  ±0.05*:
  - H1: −0.008 [−0.036, 0.024];
  - H2: −0.004 [−0.024, 0.016].
- **What those intervals show:**
  - they exclude effects of 0.05 or more, and the 48-token estimates (0.047
    and 0.057) lie outside them;
  - they do not rule out smaller effects.
- **So the registered hypotheses are not supported under the 256-token
  outcome.** The 48-token verdict stands as registered, but it depends on
  the forcing budget, and it is worded as a result about 48-token
  commitment.
- **One limitation:** the token-count check failed (maximum difference 4,
  against ±2). So net savings, which charge the forced tokens, are not
  used; every comparison uses registered savings.

## 5.3 What the explanation text carries [S]

**On the menu** (D81, `runs/rep1_information/`):
- the readout's *score* adds nothing beyond position and the registered
  probe: +0.0005 AUC [−0.0015, 0.0025];
- the explanation *text* adds +0.077 [0.063, 0.090] beyond position and
  the trace text;
- the text names the final answer no more often than another problem's
  answer: −0.002 [−0.016, 0.012];
- the phase lexicon frozen on the earlier arm transfers to unseen
  problems: +0.091 [0.076, 0.106] for tail membership.

**At positions chosen without the outcome** (Stage 1, D83; approved in D85;
`runs/rep1_stage1/`):
- **Setup:** 1,600 positions in 400 problems, with 1,598 passing
  verbalisations (attrition gate 0.75). The forcing check gave 0.98
  (gate 0.95).
- **Primary result:** the text's increment over position plus the
  same-label probe (trained for the 256-token safety label) is +0.00002
  [−0.004, 0.004]. That is **equivalent within ±0.01**.
- **Both signals carry information on their own:** the text adds +0.071
  [0.052, 0.089] over position, and the same-label probe +0.102 [0.075,
  0.129].
- **The same-label probe is better head to head**, by 0.031 [0.015, 0.047].
- **The same holds on the 48-token and tail labels.**

**Under the 256-token outcome, on the menu** (D95): text over the
same-label probe is +0.002 [−0.002, 0.005], and that probe is better head
to head by 0.035.

**The supported conclusion is narrow.** For predicting these stopping
labels, under the tested design (TF-IDF text features, a linear
same-label probe, one model and one NLA), the explanation text adds no
practically relevant information beyond position and the same-label probe.
- It does not show that everything the text says is available in the
  activation.
- It is a statement about predictive information only. It is not evidence
  about whether the explanations describe the computation faithfully (§9).

## 5.4 Selective risk on the candidate menu [S]

Selective risk at matched coverage (D83 deviation 1,
`runs/rep1_selective/`) separates risk from saving.
- **These results use the outcome-selected candidate menu (design A) and
  O48 labels. They are not deployment evidence.**
- The probe here is the **registered** probe. No selective-risk result
  exists for the same-label probe on rep1.

| at 50% coverage (menu, O48) | selective risk | saving among stops |
|---|---|---|
| registered probe | 0.011 | 0.028 |
| no signal | 0.472 | 0.413 |
| NLA readout | 0.632 | 0.711 |

- The registered probe's stops are almost always safe but save little.
- The readout's stops save more, but most are unsafe.
- No rule dominates, and the two numbers are not combined into one ranking.

## 5.5 The earlier 240-problem arm [X]

The same 256-token pass on `main_nla240` (D96) reproduces the pattern.
These results are exploratory because that arm's 48-token results were
known, and it is not independent: it contains `main`'s 120 problems.
- f = 0.881.
- Readout − registered probe goes from +0.056 [0.023, 0.091] at 48 tokens
  to +0.001 [−0.031, 0.040] at 256.
- Readout − no signal goes from +0.069 [0.049, 0.093] to −0.011
  [−0.034, 0.017].
- In both, the 48-token estimate lies outside the 256-token interval.
- Text over the same-label probe under O256: −0.001 [−0.005, 0.002].

## 5.6 Summary

| question | answer | class |
|---|---|---|
| Does the readout beat the registered probe and no signal on the registered outcome? | Yes: H1 +0.047, H2 +0.057 | [C] |
| Is that the explanations' content? | Not shown: a position-only rule does better | [S] |
| Does it hold under a 256-token forcing budget? | Not supported. Both read *equivalent within ±0.05*: effects of 0.05 or more are excluded, smaller ones are not | [S] |
| Does the text add predictive information for these labels beyond a same-label probe on the same activation? | No practically relevant amount: equivalent within ±0.01 at outcome-blind positions | [S] |
| Which signal stops more safely on the candidate menu (O48, not deployment)? | The registered probe, at a much smaller saving. No rule dominates | [S] |
| Are the explanations faithful? | Not tested | — |
