# F6 diagnostic audit: the across-groups shuffle's `np.resize` (2026-10-17)

**Separate from F6 and from RQ2's amendment A1. F6's code, its registered
result on `main` and its interpretation are unchanged.** Authorised by the
researcher 2026-10-17 as a CPU-only diagnostic. Label statistics only: no
probe was fitted and F6 was not rerun.

Script: `scripts/audit_f6_resize.py --config main` (no torch import). Output:
`runs/main_f6_audit/f6_resize_audit.json`. Since 2026-10-18 (D61) the script
refuses to overwrite that file unless given `--force`; the recorded output was
not regenerated.

## What F6 does

`run_falsification_tests.f6_probe_leakage` labels every layer-20 boundary of
every problem (2,681 boundaries, 250 problems) with "the intermediate answer
parsed here equals the gold answer", fits the correctness probe in grouped
5-fold CV, and repeats the CV on shuffled labels. Its primary control,
`across_groups` (`baselines/probe.permute_labels`), gives each problem another
problem's label sequence and fits it to the receiver's length with
`np.resize`: a shorter donor is **tiled from its start**, a longer one is
**truncated at its end**. Leakage is suspected if |shuffled AUC − 0.5| > 0.15.

Registered result on `main` (`runs/main/robustness/robustness.json`): real
AUC 0.946; across-groups shuffled AUC **0.614**; within-group 0.539;
`leak_suspected: false` (margin 0.036 below the threshold).

## Reproduction

The audit rebuilds F6's labels and groups with F6's own assembly and calls
`permute_labels` with the seed `cross_validate` passes (`probe.seed + 1`). It
reproduces the registered shuffle exactly: **718 labels changed, as
recorded**; the donor of every receiver is recovered and the rebuilt labels
equal `permute_labels`' output element for element.

## Demonstrated (on F6's labels)

| property | value |
|---|---|
| donor assignment | bijection, **0** problems keep their own labels |
| receivers with a donor of a different length | **235 of 250** (117 tiled, 118 truncated, 15 equal) |
| labels repeated by tiling / dropped by truncation | 510 / 510 (boundaries per problem 4–34, median 10) |
| positive labels dropped by truncation | **194** (late boundaries, where the answer is right) |
| positive labels re-introduced by tiling restarts | 64 |
| prevalence, original → shuffled | **0.200 → 0.152** |
| mean / max per-problem prevalence shift vs the donor's own | 0.085 / 0.333 |
| relative-position AUC for the labels, original → shuffled | **0.920 → 0.686** |
| problems whose labels are monotone (once right, stays right), original → shuffled | 0.876 → 0.512 |

So the shuffle does not do what F6's docstring says it does ("each problem's
internal label pattern and the class balance all survive"): the class balance
falls by a quarter, half the problems lose their monotone pattern, and a
substantial association with boundary position remains.

## Possible, not demonstrated

- **The shuffled AUC of 0.614 may be partly positional rather than leakage.**
  The shuffled labels still track relative position (AUC 0.686), and
  layer-20 activations carry position information, so a probe trained on
  them can plausibly beat 0.5 on held-out problems through position alone.
  That would inflate the shuffled AUC — the direction that makes F6 *more*
  likely to flag leakage. F6 passed anyway, so this mechanism does not make
  the registered "no leak suspected" conclusion easier to reach. What it does
  is make the control's number hard to read: 0.614 cannot be attributed to
  leakage, position, or both from this audit.
- **The prevalence change** (0.200 → 0.152) does not bias AUC directly (AUC
  is rank-based) but changes the shuffled run's accuracy and calibration, so
  F6's shuffled *accuracy* (0.766) is not comparable to the real one.
- **Size of any effect on the AUC is not measured.** Measuring it needs the
  probe refitted under a length-correct shuffle (e.g. A1's within-problem or
  relative-position mapping), which this audit was not authorised to do and
  did not do.

## What this does not change

F6's registered result and verdict stand as recorded. Nothing here shows
leakage, and nothing shows its absence more strongly than F6 already did.

## Decision for the researcher

Whether to run a separate CPU-only *diagnostic refit* of the correctness
probe under a length-correct shuffle (minutes on CPU) to measure the effect
on the shuffled AUC — reported beside F6, not replacing it — or to leave F6
with this audit as its caveat.

## Outcome (2026-10-18; the sections above are kept as written)

The researcher chose the diagnostic refit. Its protocol is
[`F6_LENGTH_CORRECT_DIAGNOSTIC.md`](F6_LENGTH_CORRECT_DIAGNOSTIC.md) (D58), and
its result, exploratory, is
[`F6_LENGTH_CORRECT_RESULTS.md`](F6_LENGTH_CORRECT_RESULTS.md) (D60). It
measured what this audit could not: over 100 donor draws, with the same donors,
a length-correct mapping gives a shuffled AUC of 0.841 against 0.602 under
`np.resize` — **+0.240**. The length-correct mapping also shifts prevalence
(0.200 → 0.228), so it is a sensitivity analysis, not the definitive null.

That result bears on the first bullet under *Possible, not demonstrated*. The
bullet expected the residual position structure in `np.resize`'s labels to push
F6's shuffled AUC *up*. The other side is now measured: `np.resize` also
removed most of that structure (label–position AUC 0.920 → 0.686 in F6's
own draw, as in the table above; 0.683 averaged over the 100 draws), and a
shuffle that keeps it scores far higher. F6's 0.614, and its margin under the
0.15 threshold, were therefore obtained under a control that had partly
destroyed the pattern its docstring says it preserves. This is **not evidence
of leakage**: the length-correct arm's expected AUC without leakage is not 0.5
either, and the probe stays below the position-derived oracle in every draw.
It does mean the registered control cannot carry a stronger claim than it
already makes. F6's registered result, verdict and interpretation are
unchanged; a stronger claim would need a new preregistered design.
