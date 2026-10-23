# 4. Results (draft v3, 2026-10-23)

*Order:*
1. the registered replication;
2. the two analyses that qualify it;
3. what the explanation text carries;
4. selective risk;
5. how evaluation choices can manufacture an advantage, shown on the main
   run.

The main run's own registered comparison appears in §4.6, as context, not
as the headline.

*Status markers:*
- **[R]** registered (main run);
- **[C]** confirmatory (rep1, D75);
- **[S]** pre-specified secondary or sensitivity analysis, fixed before any
  rep1 outcome and outside the verdict;
- **[X]** exploratory;
- **[P]** post hoc;
- **[A]** analytic.

*Two probes:*
- the **registered probe** is the layer-20 correctness probe of the
  registered plan;
- the **same-label probe** is a linear probe on the same layer-20
  activation, trained for the label being predicted.

*Three things are kept apart throughout:*
- **prediction:** the text carries information associated with an
  outcome;
- **incremental value:** whether that information adds anything beyond
  cheaper signals;
- **faithfulness:** whether the text describes the computation. It is
  **not tested here.**

*Every number was copied from the output named beside it and checked by
review. Tables 1–3 are generated from the outputs
(`scripts/make_paper_tables_v2.py`). `scripts/check_run_consistency.py
--replication` checks a listed subset of the numbers as they appear in the
decision record (D96 §4).*

**Figures and tables:** the design is in Fig 1 and Table 1. Fig 7 and
Table 3 carry §4.1–§4.3, with Fig 6's right panel for §4.3. Table 2 and
Fig 3 carry §4.4. Fig 5 carries §4.5, and Figs 2, 4 and 6 carry §4.6.

## 4.1 The registered replication [C]

The registered design ran on 400 problems that no earlier run had used
(§3, Table 1).
- **Instrument and sample:** 380 have an answer-stable tail, against the
  plan's minimum of 200. The forcing instrument reproduced on 0.965 of 200
  re-forced stored boundaries, against the registered 0.95. The analysis
  checked the frozen plan's hash before it ran (D94).
- **Test:** a paired problem bootstrap with 2,000 replicates and Holm
  correction over the two hypotheses. Both are **supported** on the
  registered 48-token forced-answer outcome (Table 3a; Fig 7, left):

| hypothesis | Δ (mean safe-rate gap) | 95% CI | Holm p | reading |
|---|---|---|---|---|
| H1 readout − registered probe | +0.047 | [0.019, 0.075] | 0.002 | supported |
| H2 readout − no signal | +0.057 | [0.039, 0.076] | 0.001 | supported |

H1's interval spans the plan's 0.05 margin. The advantage over the
registered probe is significant, but an advantage of 0.05 or more is
neither shown nor excluded.

## 4.2 First qualification: candidate position [S]

**The audit (D79; Table 3c):**
- **A rule with no access to the activation** scores each menu candidate by
  −prefix tokens, so it favours early stops. It beats no signal by +0.089
  [0.066, 0.108], and it beats the readout by 0.025 [0.005, 0.044].
- **With its linear-in-log-position trend removed,** the readout no longer
  beats no signal: +0.007 [−0.016, 0.029].
- **Both families of position readings are unresolved** by the plan's
  rules.

**So the registered advantage cannot be attributed to the content of the
explanations:** on this menu, a preference for early positions does as well.

**Why the menu allows this.** It is built from the tail start: it always
holds i* − 1 and i* (Fig 1) [A]. The same falling rule had already matched
the readout on the main run [X]:
- falling rule − no signal: +0.082 and +0.100;
- readout − falling: −0.024 [−0.070, 0.015].

That is why the audit was pre-specified for rep1.

## 4.3 Second qualification: the forcing budget [S]

The pass (D82; Table 3b; Fig 7, left; Fig 6, right):
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

## 4.4 What the explanation text carries [S]

**On the menu, rep1** (D81, D95):
- **Prediction without increment from the score:** the readout's *score*
  adds nothing beyond position and the registered probe: +0.0005 AUC
  [−0.0015, 0.0025].
- **Prediction from the text:** the explanation *text* adds +0.077 [0.063,
  0.090] beyond position and the trace text.
- **No latent answer:** the text names the final answer no more often than
  another problem's answer: −0.002 [−0.016, 0.012].
- **The phase lexicon frozen on the earlier arm transfers** to unseen
  problems: +0.091 [0.076, 0.106] for tail membership (Table 2). It
  describes a reasoning *phase*, concluding against setting up.

These replicate the main run's exploratory and post-hoc findings out of
sample (Fig 3):
- text over position and trace text: +0.073 [0.058, 0.087] [X];
- text over the same-label probe: −0.001 [−0.004, 0.002] [P].

**At positions chosen without the outcome** (Stage 1; D83, approved D85;
Table 3d; Fig 7, right):
- **Setup:** 1,600 positions in 400 problems, with 1,598 passing
  verbalisations (attrition gate 0.75). The forcing check gave 0.98
  (gate 0.95).
- **Primary result (incremental value):** the text's increment over
  position plus the same-label probe, on the 256-token safety label, is
  +0.00002 [−0.004, 0.004]. That is **equivalent within ±0.01**.
- **Prediction:** both signals carry information on their own. The text
  adds +0.071 [0.052, 0.089] over position, and the same-label probe +0.102
  [0.075, 0.129].
- **Head to head:** the same-label probe is better, by 0.031 [0.015, 0.047].
- **The same holds** on the 48-token and tail labels.

**The supported conclusion is narrow.** For predicting these stopping
labels, under the tested design (TF-IDF text features, a linear
same-label probe, one model and one NLA), the explanation text adds no
practically relevant information beyond position and the same-label probe.
- It does not show that everything the text says is available in the
  activation.
- **It is a statement about prediction and incremental value. It says
  nothing about faithfulness.**

## 4.5 Selective risk on the candidate menu [S, P]

Selective risk at matched coverage (D83 deviation 1) separates risk from
saving.
- **These results use the outcome-selected candidate menu and O48 labels.
  They are not deployment evidence.**
- The probe here is the **registered** probe. No selective-risk result
  exists for the same-label probe on rep1.

| rep1, at 50% coverage (menu, O48) [S] | selective risk | saving among stops |
|---|---|---|
| registered probe | 0.011 | 0.028 |
| no signal | 0.472 | 0.413 |
| NLA readout | 0.632 | 0.711 |

- The registered probe's stops are almost always safe but save little.
- The readout's stops save more, but most are unsafe.
- No rule dominates, and the two numbers are not combined into one ranking.

**On the main run** [P], the readout's stops were already *less* safe than
chance (Fig 5, left):
- +0.16 [0.04, 0.26] at 0.5 coverage on `main`;
- +0.14 [0.05, 0.21] on the 240-problem arm.

## 4.6 How evaluation choices can manufacture an advantage (main run)

These findings come from the main run and explain why the replication was
designed as it was. Their classes vary and are marked.

1. **The registered comparison was uninformative** [R, read post hoc].
   - Its budgets (0.61, 0.72) lie above the oracle bound's 0.17, where a
     no-signal rule beats every compared rule (D68).
   - Rescored on the registered outcome [P], the readout beat no signal by
     +0.052 [0.022, 0.082] at N = 120. Its comparison with the registered
     probe was unresolved: +0.027 [−0.026, 0.076] (D69; Fig 2).
   - **Provenance:** D91 records that the amended reanalysis plan's hash was
     not recorded before the data.
2. **Candidate points derived from the outcome leak it** [A]. See §4.2 and
   Fig 1.
3. **A matched-mean-saving estimand rewards concentrating savings** [X].
   - At every boundary of the main run, with no menu (Stage 0, D83), an
     information-free falling rule beats a same-label activation probe
     whose boundary AUC is 0.861.
   - Probe − falling: −0.032 [−0.083, 0.009] on registered savings, and
     −0.093 [−0.147, −0.043] on net savings (Fig 4).
   - Selective risk at matched coverage reverses the ranking (Fig 5,
     right).
4. **Short forcing budgets give answers that are reproducible but not
   valid** before the tail [X].
   - Before the tail, 54–57% of 48-token completions are cut off, and 98–99%
     of those disagree with the continuations' majority (Fig 6).
   - On rep1 the 256-token pass then measured the consequence (§4.3).
5. **Cost** [A]: one NLA check costs 0.83 of a mean trace in generated
   tokens, so verbalising at each candidate cannot save compute here.

## 4.7 The earlier 240-problem arm [X]

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

## 4.8 Summary

| question | answer | class |
|---|---|---|
| Does the readout beat the registered probe and no signal on the registered outcome? | Yes: H1 +0.047, H2 +0.057 | [C] |
| Is that the explanations' content? | Not shown: a position-only rule does better | [S] |
| Does it hold under a 256-token forcing budget? | Not supported. Both read *equivalent within ±0.05*: effects of 0.05 or more are excluded, smaller ones are not | [S] |
| Does the text predict stopping labels? | Yes, beyond position and the trace text | [S] |
| Does it add predictive information beyond a same-label probe on the same activation? | No practically relevant amount: equivalent within ±0.01 at outcome-blind positions | [S] |
| Which signal stops more safely on the candidate menu (O48, not deployment)? | The registered probe, at a much smaller saving. No rule dominates | [S] |
| Are the explanations faithful? | Not tested | — |
| Is there a safe and useful stopping policy? | Open. A planning replay motivates testing the same-label probe on fresh problems (`docs/STOPPING_POLICY_PROPOSAL.md`, not approved). Its in-sample numbers are not evidence | — |
