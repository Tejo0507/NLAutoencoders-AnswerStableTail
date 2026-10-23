# Stage 0 on rep1 and D76: should either come before the stopping-policy study? (2026-10-23)

> Decision note. **Neither study is authorised or run.** Their costs are
> estimates from rep1's measured rates.

## Stage 0 on rep1

**The uncertainty it resolves:**
- Stage 0 (D83) has run only on `main`'s 250 problems, on O48, at every
  boundary.
- It produced claim matrix B7 and D1:
  - **B7:** under the registered matched-mean-saving estimand, an
    information-free falling-position rule beats an activation probe whose
    boundary AUC is 0.861;
  - **D1:** a same-label probe carries substantial safety information at
    every boundary.
- Both are exploratory, because they rest on one run. Stage 0 on rep1 would
  re-test them on fresh problems.

**Could it change the stopping-policy design?** Only a little.
- The proposal already abandons the matched-mean-saving estimand B7
  criticises. It uses selective risk, matched risk and net savings.
- It already uses every boundary and the same-label probe.
- It labels on O256, where Stage 0 would use O48.
- A rep1 Stage 0 could still help the replay (proposal §6.6). It would add
  an every-boundary walk alongside Stage 1's thinned one. But O48 labels
  make it a weaker guide than Stage 1's O256 labels.

**Cost:**
- About 1,782 rep1 boundaries have no forced answer: 4,458 boundaries,
  2,077 stored and 599 forced for the menu.
- 48-token forcing at about 1.2 s per generation is **about 35–40
  GPU-minutes**, plus a few CPU minutes for the analysis.

**Recommendation:** **do not run it before the stopping-policy study.** Run
it only if the paper's evaluation-pitfalls section needs B7 replicated on
fresh data, and then as an exploratory analysis. The stopping-policy study
answers the practical question directly.

## D76, the rewording null

**The uncertainty it resolves:**
- RQ2 counts a claim as *reconstruction-dependent* when deleting it moves
  the reconstruction more than a null does (claim matrix C6: 0.749 of
  6,007 claims).
- D70 found the registered paraphrase null is mostly no edit, or an added
  discourse marker.
- D76 tests the missing case: does a meaning-preserving *rewording* move
  reconstruction as much as deleting the claim does? If so, the
  0.749 rate overstates claim-specific dependence.

**What it is not:**
- It is **not a faithfulness test.** Reconstruction dependence is not
  faithfulness, and D76 only validates the null behind a
  reconstruction-dependence rate.
- It says nothing about stopping.

**Could it change the stopping-policy design?** No.
- The stopping study treats the NLA as an offline predictive comparator.
- Whether claims carry claim-specific reconstruction weight does not enter
  any of its policies, labels or gates.

**Cost:** about **30 GPU-minutes**: 60 explanations, 3 rewordings per
claim, then reconstruction scoring.

**Recommendation:** **do not run it before the stopping-policy study.** Run
it only before the paper reports C6. It qualifies an RQ2 number that the
manuscript currently reports with a caveat.

## Summary

| | resolves | changes the stopping study? | cost | precede it? |
|---|---|---|---|---|
| Stage 0 on rep1 | whether B7 and D1 hold on fresh data (O48) | barely; the study already avoids the estimand and uses O256 | about 35–40 GPU-minutes plus CPU | **no**; only if the paper needs B7 replicated |
| D76 | whether C6's reconstruction-dependence rate survives a rewording null | no | about 30 GPU-minutes | **no**; only before the paper reports C6 |

**Priority:**
1. The stopping-policy study, after its decisions and the planning replay.
2. D76 and Stage 0 only if the manuscript needs them.
