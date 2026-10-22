# Current state and next steps

What exists, what has run, what it found, and the next command to type.
Last reviewed 2026-10-22.

> **Update 2026-10-22, about 20:55 IST: the D75 verdict exists (D94).**
> - Stages A–D (reconstruction, 48-token forcing, the confirmatory analysis,
>   the D79 audit) completed, and every runbook gate passed. Re-forcing
>   agreement was 0.965 on 200.
> - **H1 supported** (+0.047 [0.019, 0.075], Holm p 0.002). **H2
>   supported** (+0.057 [0.039, 0.076], Holm p 0.001).
> - D79: both families are **unresolved**. A falling-position rule beats the
>   readout by 0.025 [0.005, 0.044]. The position-residualised readout does
>   not beat no signal (+0.007, p 0.54).
> - Full record and limits: D94. The approved queue (F below) started at
>   20:52.
>
> **Update 2026-10-22, 20:19 IST: resumed (D93).** The pipeline was paused
> at 16:43 (D92) and relaunched at 20:19 with the unchanged
> `resume_rep1.py`. Verbalisation was skipped (2,227 windows, 0 remaining);
> reconstruction restarted from 0/2,227. Every recorded hash was re-verified
> first. Live checkpoint, parameters and hashes:
> [`CHECKPOINT_2026-10-22.md`](CHECKPOINT_2026-10-22.md). The table below is
> the state at the 16:43 pause, kept as written.
>
> **Update 2026-10-22 (paused at 16:43): the confirmatory replication (D75)
> is past traces and verbalisation. Its verdict is not known.** Everything below this
> block is the 2026-10-17 state and stays as written. Records: D75–D91 in
> `docs/DECISIONS.md`.
>
> | milestone | state (verified 2026-10-22) |
> |---|---|
> | **A. Traces** | **Complete:** 400 rows, 400 ok, 400 unique ids, exact frozen order. Plan `e45629a7…`, config `d31383f6…` and problems `e633885f…` hashes match. 17 hit the 768-token cap (all MATH, all with a parsed answer); AST gives 15 of them no stable point, and 2 keep a flagged tail. 380/400 have a tail, against the plan's 200 minimum |
> | ast, acts, baselines | **Complete**; the runbook gates pass (1,200 activation blocks, 0 clamped; probe present) |
> | **B. NLA stage** | **Verbalisation complete:** 2,227/2,227 windows, 4,439/4,454 samples passing integrity (99.7%). **Reconstruction not complete:** the pipeline was paused at 16:43 on 2026-10-22 at 2,000/2,227 reconstructions (D92). Reconstruction writes its output only at the end, so it reruns in full (about 1.3 h) on resume; verbalisation is not repeated |
> | **C. Menu forcing** | Pending: the registered **48-token** instrument, blocks of 2, 200 re-forced boundaries, agreement ≥ 0.95, else blocked (D80) |
> | **D. D75 confirmatory analysis** | Pending: `reanalyse_o3_registered.py --confirmatory docs/REPLICATION_PLAN.md --plan-sha256 e45629a7…0761`. H1 readout − layer-20 probe; H2 readout − no signal; mean safe-rate gap over 0.1–0.9 × B\*, paired problem bootstrap (2,000, seed 20261019), Holm over {H1, H2}, margin 0.05, mechanical readings. **The registered 48-token outcome decides it** |
> | **E. D79 position/leakage audit** | Pending; runs right after D. It is secondary and does not enter the verdict, but it says whether a positive H2 reflects content or the menu's position structure (on stored data a rule with no readout matched the readout) |
> | **F. Follow-ups**, in the D87 order | Pending: rep1 secondaries (#29) → D82 256-token forcing on rep1 (#28) → Stage 1 (#30) → D82 on `main_nla240`. All are pre-specified or exploratory; none replaces the registered outcome |
> | **G. Manuscript** | Drafts merged (#37), with provisional novelty (#31). No faithfulness claim. Results sections wait for D–F |
>
> **To continue after any interruption** (quit OMEN Command Center first;
> D89):
> `.venv\Scripts\python.exe scripts\resume_rep1.py`. To pause:
> `scripts\safe_stop.ps1`. See `docs/RESUME_2026-10-20.md`.
>
> **Repository:** the research stack #33–#37 is merged into `main` (D91
> records the review). The working checkout that runs the pipeline stays on
> its branch until the pipeline finishes.

> **Update 2026-10-17 — the main run is complete.** 250 traces (0 failed,
> 250 unique ids), then every post-trace stage: `ast`, `acts` (750 blocks, 0
> clamped), the supervised analyses, `baselines`, `nla` (120 problems under
> `--set nla.max_problems=120`, hash `0bb33cf4c468a3f6`), `faithfulness`,
> `causal`, `analysis`, `robustness` (F8 blocked by design), `report`.
> `check_run_consistency.py --run runs/main`: 159 checks, 0 mismatched.
> The execution report is `runs/main/report/RESULTS.md`; the README's *Main
> run* table summarises it.
>
> A post-run audit found the RQ3 verdict reported as "supported: False" from a
> readout that was zero in every arm, a manifest race that had erased three
> stage records, intervals printed beside effects on another scale, the NLA
> override recorded nowhere, and a supervised report that called p = 0.08
> "separated". All are fixed and merged (#23, #24, then D50–D53) with no new
> experiment and no measured number changed; RQ3 now reads *not tested*.
> `check_run_consistency.py`: `main` 179 checks, 0 mismatched, 0 warnings;
> pilot 173, 0 mismatched, 1 historical warning. **What is still open, and
> which items need the researcher's decision, is tracked in
> [`AUDIT_2026-10-17.md`](AUDIT_2026-10-17.md)**; RQ3's options are in
> [`RQ3_DECISION_BRIEF.md`](RQ3_DECISION_BRIEF.md). Those files supersede the
> "what has not run" list in §5 below, which predates the main run.
>
> Run artifacts are archived in `D:\NLAutoencoders-backups\`, each snapshot a
> `.tar.gz` with `SHA256SUMS.txt` and `VERIFICATION.json` and mirrored to
> `E:\NLAutoencoders-backups\`: `runs-2026-10-17`, `runs-2026-10-17-pre-experiments`,
> `runs-2026-10-18-post-experiments` and `runs-2026-10-18-f6-diagnostic`
> (all four re-verified on both volumes 2026-10-18). C:, D: and E: are
> partitions of **one** NVMe disk, so these are not independent physical
> copies. **Off-machine copy: confirmed by the researcher, 2026-10-18** — the
> complete `runs/` directory (including `main_f6_diag` and
> `main_f6_diag_validation`) is backed up to Google Drive and to
> `E:\NLAutoencoders-backups\runs\`. The E: copy was checked file by file
> against the live tree (2,041 of 2,041 identical); the Google Drive copy is
> researcher-confirmed and was not verified from this machine (audit,
> "Backup").
>
> **Update 2026-10-18 — F6's control is sensitive to its length mapping
> (exploratory).** A separate diagnostic (protocol D58, implementation D59,
> result D60, [`F6_LENGTH_CORRECT_RESULTS.md`](F6_LENGTH_CORRECT_RESULTS.md))
> refitted the probe over 100 donor draws. With the same donors, a
> length-correct mapping gives a shuffled AUC of 0.841 against 0.602 under
> F6's `np.resize` (**+0.240**), and shifts prevalence 0.200 → 0.228. The
> probe stays below a position-derived oracle in every draw; that is not a
> leakage test and rules nothing out. F6's registered result (0.946 / 0.614 /
> 0.539), threshold, `leak_suspected = false` and interpretation are
> unchanged, but the registered control cannot support a stronger claim; that
> would need a new preregistered design. No further experiment is planned.
> The other post-run analyses (F5 deferred, F8 partial, RQ2 amendment A1, the
> N = 240 extension) are tracked in `DECISIONS.md` D52's status table.
>
> **Update 2026-10-18 — engineering pass, no new experiment.** Three defects
> in the post-hoc scripts fixed with regression tests (D61: ignored
> `--run-id`, a half-prepared F8 arm, an overwritable F6 audit output). Every
> recorded check was re-run read-only and still passes, and every file under
> `runs/` is unchanged apart from the test suite's own fixture run
> `runs/tiny/`; the audit register is reconciled with D52–D60 (PW-24,
> PW-25). Read-only validation commands: README, *Checking the recorded
> results*. RQ3 is closed as untested (D62; issue #11 closed). The run-artifact
> release is **prepared and verified as a draft, not published** (D64, D65):
> local path roots are sanitised in the release copy only. One path is
> retained in the F6 validator, as the researcher approved on 2026-10-19. The
> release waits only for the write-up freeze. Steps and gates:
> `release/PUBLISHING.md`.
>
> **Update 2026-10-19 — the write-up is complete and frozen** at `32b5539`
> (D66). `paper/REPORT.md` holds all nine sections, and
> `assemble_paper.py --check` exits 0. Its Results are read from the main
> run's certified artifacts; §6.8 keeps the exploratory and partial analyses
> apart. The run-artifact release is built from the commit recording D66.
>
> **Run artifacts published on GitHub, 2026-10-19** (D67): release
> `data-runs-2026-10-19`, archive SHA-256 `b50c9320…b2ff`, verified by an
> anonymous download. The Zenodo deposit is prepared but not made (no
> credentials here), so there is no DOI yet.
>
> Everything below this box describes the state on 2026-10-16 and the pilot.

**Main run in progress: 231 of 250 traces** as of 18:53 on 2026-10-16, 0
failures, 0 unparseable records, 0 duplicates (139 GSM8K / 92 MATH). Three
reboots in two days (14:10 and 20:59 on 15 Oct, 15:57 on 16 Oct) each ended the
process; the run was relaunched at 18:52 and resumed from problem 232 under an
unchanged config hash. 19 problems remain (11 GSM8K, 8 MATH). The time left is
genuinely uncertain, **2 to 10 hours**, because the most recent launch ran far
slower than the first.

Throughput has varied sharply across launches — GSM8K medians of 5.1, 20.6,
9.5, 11.4 and 20.6 minutes — with no cause established beyond the machine's
state; the code and config did not change. The launch history, including an
unexplained 10½-hour gap on 16 Oct, is in `SESSION_HANDOFF.md` at the top
(a local session log, not part of the public repository). Everything downstream
of `traces` is blocked on it.

MATH truncation at the 768-token cap is **8 of 92 (0.087, 95% CI [0.038,
0.164])**, GSM8K 0 of 139, against the pilot's 3 of 10 at 512 tokens. The upper
bound still excludes the pilot's 0.300, so the cap stands.

**One-line status:** the pilot has run **to completion** — all thirteen
stages on all 24 configured problems, including the released autoencoder,
with the full nine-test falsification battery. The results below are real and
come from `runs/pilot/`, whose execution report is
`runs/pilot/report/RESULTS.md`. The sample is 24 problems, so every interval
is wide and nothing here is a claim about language models in general.

**The headline is negative for the verbalised readout, and not significantly
so.** At matched budget the readout is *worse* than semantic entropy — 4
discordant problems of 21, all four favouring semantic entropy — but on 21
problems that is p = 0.125, q = 0.1875. It is **exactly tied** with the
hidden-state probe, preserving the outcome on the same problems. The review
named a negative result as acceptable in advance and it is reported with the
weight a positive result would have had.

> **This supersedes an earlier reading of the same run.** The comparison was
> previously reported as significant at q = 0.0065. That test paired the two
> rules over nine *budget levels*, all nine of which resolved to one pair of
> operating points, so it saw nine copies of one comparison and its p-value
> was set by the size of the budget grid — 0.0625 at five budgets, 1e-6 at
> twenty-five. Tested over problems, which is the unit safety is measured on,
> the same data give q = 0.156 over the five tests then evaluable, and
> **0.1875** once D38 counts the probe comparison's tested null (six tests).
> See [`DECISIONS.md`](DECISIONS.md) D29 and D44. The
> direction of every claim below is unchanged; the strength is not.

> **Regenerated on 2026-10-15; the q-values below are current.** The pilot's
> `ast`, `baselines`, `robustness`, `analysis` and `report` stages were re-run
> on CPU under the corrected code (D38, D41, D42). Every per-problem artifact
> was byte-identical or gained fields only; what moved is the BH family, from
> five evaluable tests to six, so the primary comparison is **q = 0.1875**
> (was 0.156) and F10's position control **q = 0.0275** (was 0.0229). No
> conclusion changed: three tests significant, the primary comparison not.
> `scripts/check_run_consistency.py --run runs/pilot`: **153 checks, 0
> mismatched**; the only warnings are that the GPU stages `nla` and `causal`
> predate per-stage commit recording. D44 has the before/after table.

---

**After `traces` finishes, follow [`POST_TRACE_CHECKLIST.md`](POST_TRACE_CHECKLIST.md).**
Every CPU stage in it was rehearsed on a copy of the first 216 main traces on
2026-10-15; the two bugs that rehearsal found (D45 and the supervised-analysis
crash) are fixed.

---

## 1. What ran, with its result

| stage | status | headline |
|---|---|---|
| `env` | complete | RTX 4050, 6.44 GB VRAM, 16.8 GB (15.65 GiB) RAM; the bf16-staging disk check fails by design and is worked around by eviction |
| `data` | complete | 24 problems: 14 GSM8K, 10 MATH; 11 train / 13 eval; gold recovery 1.000 |
| `models` | complete | all three checkpoints converted to NF4; AV/AR/target sidecars mutually consistent; injection site verified at token 111 |
| `traces` | complete | 24/24, no failures, 310 min; accuracy 0.792 (GSM8K 0.857, MATH 0.700); 3 traces hit the 512-token cap |
| `acts` | complete | 72 blocks, 834 vectors at layers 14/20/24; **0 positions clamped, 0 re-tokenisation drift** |
| `ast` | complete | tail on 21/24; mean tail fraction 0.356; see §2 |
| `baselines` | complete | probe grouped-CV AUC 0.996 — but see §2 for what that number is |
| `nla` | complete | 126 windows, 252 activation samples + 126 controls; integrity **0.976**; FVE **0.677** against the empirical baseline |
| `faithfulness` | complete | 967 claims over 246 explanations; 0.759 reconstruction-dependent, 0.003 reproduced from noise |
| `causal` | complete | truncating the tail changes **no answer on 10/10** problems; the direction claim is **untested** — the three direction arms were no-ops (D34), now fixed but not re-run |
| `analysis` | complete | 9 tests in the family, 6 evaluable, 3 significant at q = 0.05 (regenerated 2026-10-15) |
| `robustness` | complete | all nine tests ran; F8 cos 0.987, F6 no leak, F7 layer 20 not special |
| `report` | complete | `runs/pilot/report/RESULTS.md`, 5 figures, 7 tables |

---

## 2. What the pilot found

Five results, in descending order of how much they change the reading of the
study. Each is computed by the stage named, and `runs/pilot/report/RESULTS.md`
carries the full tables.

### The 4-bit deviation is small (F8)

Layer-20 activations from the NF4 target agree with the bf16 ones at **cosine
0.987** on average (median 0.991, 5th percentile 0.973, minimum 0.937, norm
ratio 0.990) across 82 vectors from 8 problems, with no problem skipped for
misalignment. The verbaliser normalises its input to a fixed L2 norm, so only
direction reaches it. Together with the integrity gate passing on every sample
so far, this closes the largest threat to the NLA arm that
[`DECISIONS.md`](DECISIONS.md) D2 identified: the autoencoder is being fed
vectors it recognises.

### `tail_rate` is near its ceiling by construction

A tail was detected on 21 of 24 problems, and the **final boundary qualifies on
21 of 24** — which is why. At the last boundary the prefix is the whole trace,
so forcing an answer from it reproduces the final answer and continuations from
it restate the same answer: both criteria are close to tautological there. The
quantity that carries information is how *much* of the trace the tail covers:
mean tail fraction **0.356** (median 0.308, range 0.15–0.62), and tail tokens
are **31.7%** of the tokens generated in traces that *have* a tail — **25.2%**
of all generated tokens, since the three traces without a tail are the three
that hit the generation cap and so are the longest. No tail is trivial.

The three problems without a tail are exactly the three traces that hit the
512-token cap, where the "final" answer is not final. That is the token-cap
decision in §4 showing up as a result rather than as a worry.

### The tail begins before the answer is stated — on every problem

The criteria ask whether the answer is *determined* from a prefix, not whether
the trace has **said** it. On all 21 problems with a tail, the tail begins
before the answer is stated, by a mean of **2.19 chunks** and up to 5. Forcing
elicits the answer from the prefix; the model finishes the arithmetic inside
the forced completion.

For the stopping comparison that is exactly right — a rule stopping there loses
nothing. For reading the window as post-answer behaviour it is not: such a
window contains the computation that produces the answer rather than redundant
verification of it, so a verbalisation sampled from early in a tail must not be
read as a model checking its work.

### The forcing budget makes the tail a lower bound

Criterion 1 forces an answer under a 24-token budget. On **40.7%** of evaluated
boundaries that completion was cut off mid-derivation, and on **24.1%** the cut
was *blocking*: criterion 1 failed while every resampled continuation from the
same prefix reached the final answer. 19 of 24 problems are affected.

So the detected tail is a floor. Counting those boundaries as satisfying
criterion 1 gives the other end: mean tail fraction **0.356 → 0.486**, tail
token share **31.7% → 45.0%** (both over the same 21 tailed traces). The
relaxed figure is not the pre-registered criterion and is reported only as the
upper end of the interval.

### The hidden-state probe is largely reading position

The layer-20 correctness probe reaches a grouped-CV AUC of **0.996** on 272
boundaries. Fitting the same model family on the same grouped folds using only
activation-free positional features — chunk index, relative position, prefix
tokens — reaches **0.962**. The probe is 0.034 AUC above a predictor that never
sees an activation.

Its label, *the intermediate answer at this boundary is already correct*,
becomes true once the trace has worked the answer out and stays true, so it is
strongly ordered by position. The probe's row in the stopping comparison should
be read as a positional rule with a small activation-derived increment, not as
a correctness readout. This matters because the probe is one of the baselines
the primary research question is measured against.

### The primary question: the readout does not beat the cheap signals

Two of the three planned head-to-heads can now be made. The outcome is binary
per problem — stopping there either preserved the verified outcome or it did
not — and both rules are scored on the same problems, so each comparison is
McNemar's exact test over problems. Only *discordant* problems carry
information.

| compared against | NLA safe | its safe | discordant | favouring NLA | q |
|---|---|---|---|---|---|
| semantic entropy | **0.095** | **0.286** | 4 of 21 | 0 | 0.1875 |
| hidden-state probe | 0.095 | 0.095 | **0** of 21 | — | tied (p = 1.0, q = 1.0) |
| convergence | — | — | — | — | not evaluable on this corpus |

Neither comparison favours the readout, and neither is significant. Against
semantic entropy all four discordant problems fall the same way, which is
suggestive on a corpus this size and no more: eight such problems would be
needed to clear q = 0.05, and `scripts/size_o3_corpus.py` turns that into a
corpus size.

Against the probe the result is a genuine tie — the two rules preserved the
outcome on *exactly the same* problems, so there is nothing to test rather
than nothing to see. Read with the positional-floor finding above, a readout
that matches a predictor largely reading position is not obviously reading
more than position either.

The convergence comparison is **unevaluable on this corpus**: its operating
points straddle the shared budget window without entering it. Swept over its
own window `w` as the pre-registration leaves free (D36), the rule reaches
eight distinct savings on the pilot, of which the two bracketing the window of
0.632–0.675 are 0.407 at `w = 2` and 0.855 at `w = 1`. `w` is an integer, so
nothing is reachable between them.

Whether a **larger** corpus would make the comparison evaluable is an open
question, not a settled no. The earlier claim that no corpus size could help
was argued from a binary score that D36 replaced; `granularity_limited` is now
`false`. Both quantities that decide it move with the data: the achieved
savings are means over problems, and the window is the intersection of every
rule's reachable range, pinned on the pilot by the NLA arm. See
[`DECISIONS.md`](DECISIONS.md) D31 (superseded in part), D36 and D37.

Both of the previously "unevaluable" comparisons were partly an artefact of
sweeping 25 evenly spaced thresholds, which samples the score axis rather than
the rule. The probe's scores saturate, so the linear grid found 5 operating
points; swept at the observed values the same problems give 86 and match at
every budget (D30). The unevaluable test stays in the **report** rather than
being dropped, so a comparison that was planned and could not be made stays
visible. It produces no p-value, so it does not enter the Benjamini–Hochberg
denominator: that correction is over the six evaluable tests, not the nine
pre-registered ones (correcting over all nine also gives three significant),
and the pre-registration does not settle which of the two it should be (D37,
D38).

### RQ2: the claims are reconstruction-dependent and not noise

967 claims across 246 explanations. **0.759** are reconstruction-dependent —
deleting them degrades reconstruction more than the 95th percentile of the
paraphrase null — and only **0.003** are reproduced by a matched-norm Gaussian
vector, giving **0.756** dependent and not noise.

Both numbers survive the thresholds they rest on:

- The paraphrase null does not control for *how much text* a deletion removes,
  and a deletion removes 24.6% of an explanation. The effect does rise with
  deleted fraction (r = +0.31, monotone across quintiles) but that is only
  9.6% of the variance; removing the trend gives a **length-adjusted 0.781**,
  slightly higher than registered. Within one explanation, where deletions are
  comparable in size, the spread of effects is 0.031 against a mean of 0.041.
- The noise control's Jaccard cut-off of 0.5 is arbitrary, but the rate is flat
  across it: 0.018 at 0.2 down to 0.001 at 0.6, against a similarity
  distribution with median 0.059 and 99th percentile 0.222.

**Reconstruction fidelity is not tail-specific**, though. Tail windows
reconstruct at cosine 0.870, a matched pre-stabilisation window at 0.869, and
a matched-length window elsewhere at 0.859 — F10's answer is that the readout
describes a late mathematical-reasoning state about equally well wherever it
is sampled. The Gaussian control at 0.391 is what shows it is doing anything
at all.

### RQ3: untested, for two reasons rather than one

**The direction edit never reached the computation.** All three direction arms
were no-ops. The edit is restricted to absolute token positions covering the
generated continuation, but the span was applied as an index into whichever
tensor the forward hook received — and cached decoding passes the whole prompt
once and then one token at a time, so the span intersected neither. The run's
own output shows it: on all ten problems `ablate_dir` returned identical
answers and identical token counts at coefficients 0.0, 1.0 and 2.0, and dose
0.0 is a no-op by definition.

Fixed, unit-tested including the cached-decode case, and now *checked* — the
editor counts the positions it modifies, the stage warns when a non-zero dose
edits nothing, and the screen reports a broken independent variable rather
than a negative result. **Not yet validated on GPU.** The numbers below are
from the pre-fix run and describe arms that did nothing.

**The behavioural readout also never moved, and now we know it will not.**
Measured across the pilot's 24 traces *and* the 176 main-run traces available
on 2026-10-15 — **200 traces in all** — there is exactly **one** marker
occurrence: the word "confirming", once, in one trace. Not one of the other
nineteen markers appears anywhere. The pooled rate is 1/200 = **0.0050**, 95%
exact CI **[0.0001, 0.0275]**.

Projected onto the 60-problem causal arm, that is **0.30** marker-bearing
problems expected at the point rate and **1.65** at the interval's upper
bound, against a `MIN_PAIRS` of 3. So the dependent variable cannot support
the comparison **at any corpus size this study can reach**: the binding
constraint is that Qwen2.5-7B-Instruct writes clean step-by-step derivations
without rechecking language, not that N is too small. This is a finding about
the model, not a defect in the design.

**Growing the corpus has moved this against the readout, not for it.** The
count stays at one while N rises, so the interval's upper bound falls — it was
[0.0002, 0.0392] at 140 traces — and with it the most optimistic number of
usable problems, from 2.35 to 1.65. Three marker-bearing problems are not
expected until the arm holds about **109**, even at the top of the interval,
against a configured 60. Re-measure with

```powershell
.venv\Scripts\python.exe scripts\measure_rq3_readout_floor.py
```

which is read-only and CPU-only, so it is safe to run mid-generation. This is
the one blocked claim in the study that is already settled before the run
finishes: every other one waits on the corpus, and this one is decided by a
base rate the corpus is still lowering.

With a constant outcome every comparison in the Zhang & Nanda gate fails
trivially, so the gate returns `supported: null` with the reason rather than
`false`, and both control comparisons as null. That is no evidence either way.

So RQ3 had neither a working independent variable nor a varying dependent one,
and the second is now established rather than suspected. The construct
*is* registered — `Project_Review_II.md` line 122 asks whether ablation
"selectively changes verification-like behaviour" — so a readout may be
amended only if it measures that construct. `continuation_tokens` does not; it
measures output length, and was rejected for that reason rather than adopted
because it happens to vary. Candidate readouts are screened for variation and
arm separation on every run, no readout is flagged pre-registered, and the
verdict's readout is named as a choice ([`DECISIONS.md`](DECISIONS.md) D34,
D35, D40, **D41**).

What the behavioural arms *do* show, after fixing an off-by-one that had them
truncating one chunk too early: **stopping at the tail start changes no answer
on 10 of 10 eval problems**, and replacing the tail with neutral filler of
matched token length changes none either. Accuracy is 0.800 in both arms,
identical to the unintervened baseline. The tail contributes nothing to the
verified outcome.

### Also measured

- **The cheap agreement rule fires earlier than the AST on 57.1% of problems**,
  by **5.5 chunks** where it does, and later on 38.1% by 1.88. The net signed
  gap is 2.43 chunks, in which the two directions cancel — it is the figure
  previously quoted here as "mean gap" and it understates the early case by
  more than half. Mo et al.'s concern, quantified on
  this run's own traces. Sweeping the agreement window shows how much that
  depends on its parameter: at window 1 it fires earlier on *every* problem by
  6.43 chunks; at window 3 it fires on only 14 of 24 and later on 73%.
- **F5**: perturbing the criteria themselves moves the mean tail fraction by
  **0.004** — the construct is stable under its own parameters. `k = 5` could
  not be simulated (only 3 continuations per boundary were generated) and is
  reported as not applicable rather than as stability.
- **F6**: no leakage. Real AUC 0.996 against 0.594 with labels permuted
  between problems and 0.472 within them; both shuffles moved labels, so both
  are real controls. (Later finding, 2026-10-18: the between-problem arm fits
  donor labels to length with `np.resize`, which the shuffled AUC is sensitive
  to; see [`F6_LENGTH_CORRECT_RESULTS.md`](F6_LENGTH_CORRECT_RESULTS.md).)
- **F7**: layer 20 is **not special** for decodability — AUC 0.995 at layer 14,
  0.996 at 20, 0.994 at 24. With the positional floor at 0.962, the probe's
  near-perfect score is a property of the task and the position, not of the
  layer the autoencoder happens to read.
- **F4**: the alternative chunker yields 16.0 chunks against 11.6, a real
  sensitivity of a chunk-indexed construct. Strict and permissive extractors
  agree on the final answer of 0.875 of traces.
- **F2**: no detectable difference in tail fraction by benchmark (GSM8K 0.369,
  MATH 0.331, p = 0.50).
- **F1**: correlation between trace length and tail *fraction* is −0.33, so the
  construct is not a length statistic in disguise.
- **Verbaliser integrity**: 0.976 of activation-driven samples are well-formed
  English against 0.786 of noise-driven ones, and the documented CJK failure
  signature appears in 2.4% against 21.4%. Unplanned, and a difference between
  real activations and matched-norm noise that needs no judgement of content.

---

## 3. What the first real verbalisations look like

Worth reading before the aggregate faithfulness numbers, because they shape how
those numbers should be interpreted. On `gsm8k-test-00197`, whose answer is
6277, the verbaliser produced well-formed English that identified the register
and discourse position of the window correctly:

> Structured numbered problem format with formal math text explaining
> calculations, presenting sequential steps using "Thus" and "Final Answer"
> sections to derive a conclusion.

and then invented the specifics — one sample asserted the quantity was 122, the
other 53. Two samples of the same activation inventing different numbers is
itself evidence that neither is reading the value. The matched-norm Gaussian
control produced equally confident text about Excel-style procedure formats and
maths forum posts, so the real activation does carry the maths-trace register
and the concluding-position signal that noise does not.

The claim-level audit exists to separate those two kinds of claim, and the run
report prints examples verbatim beside their controls so the characterisation
is checkable.

---

## 4. Deviations this run actually needed

All recorded in [`DECISIONS.md`](DECISIONS.md); these are the ones the run
itself forced.

| | |
|---|---|
| **D20/D24** | F8's bf16 arm is ~12 GB against 6.44 GB of VRAM and 2–3 GB of free RAM. It is split across GPU and host by measured free memory with the remainder offloaded to disk, and refuses to report a near-zero cosine with an implausible norm ratio as drift rather than as a load failure. |
| **D23** | Upstream's `NLACritic` stages the whole reconstructor in host memory before moving it to the GPU. That does not fit here, so the backbone is placed directly on the GPU for the duration of the construction. The arithmetic is untouched. |
| **conversion** | `save_pretrained` defaults to a 50 GB shard, so a 7B model is written as one preallocated file; that failed with `os error 112` against ~10 GB free. Conversions now write 2 GB shards after a pre-flight space check, and the bf16 source is evicted *before* the save — without which the reconstructor's output had nowhere to go. |
| **token cap** | MATH traces hit `generation.max_new_tokens: 512` on 3 of 10 problems, and those are precisely the three with no detectable tail. Raising it is the first thing a larger run should change. |

---

## 5. What has not run

- **The main configuration** is **running** as of 2026-10-11. 250 problems,
  interleaved by dataset so the achieved N is a proportional sample of both
  benchmarks at whatever point it stops (D33). Measured throughput on an
  otherwise idle machine is **3.2 min/problem** for GSM8K under main's
  settings, against the 12.9 min/problem the pilot's manifest records — the
  pilot generated the same traces twice, once at about 2 min/problem and once
  at 12.9, with byte-identical outputs, so the slow figure was a degraded
  machine state (1.3 GB free RAM) and not an intrinsic cost. Expect roughly
  20 hours for all 250, with MATH about twice GSM8K per problem.
- **The two supervised analyses** (`scripts/predict_tail_fraction.py`,
  `scripts/predict_correctness.py`) refuse below 30 train / 20 eval rows and so
  cannot run on 24 problems. They are correct to refuse. The root-level
  `regression.py` / `classification.py` do run at this size and write a
  `provenance.json` recording the corpus they used.
- **`runs/mlcorpus`** holds 33 traces from a run that predates the chunker and
  answer-parser fixes. They are not poolable with the pilot's — the chunk
  boundaries baked into them differ — and `common_data.py` now refuses to pool
  them automatically (D22).
- **A second annotator** for O6 inter-rater reliability. Single-rater, as the
  review anticipated.
- **A corpus large enough for the O3 comparison to resolve anything.** The
  binding quantity is **discordant** problems, not total ones: 8 are needed to
  clear q = 0.05 and the pilot's rate of 4/21 puts that at about 42 problems in
  the NLA arm (D32). `scripts/size_o3_corpus.py` recomputes it from any run.
  Note that this is no longer what blocks *two* of three head-to-heads — one
  was a threshold-grid artefact and is now evaluable (D30), and one is
  unevaluable on the corpora measured, for a reason that is a property of
  where the rule fires rather than of N (D31 superseded in part, D36, D37b).
- **A GPU validation of the fixed direction interventions.** They were no-ops
  for the whole pilot (D34). The fix is unit-tested but the causal stage has
  not been re-run, because the traces stage holds the GPU. The RQ3 family test
  now refuses a p-value unless a non-zero edit is demonstrated, so an
  identical-arms result under a no-op can no longer read as a tested null.
- **An in-construct behavioural readout for RQ3 with range on this model.**
  This is now a *measured* dead end rather than an open search: 1 marker
  occurrence in 200 traces, projecting to 0.30 marker-bearing problems in the
  60-problem causal arm, and 1.65 at the interval's upper bound against a
  `MIN_PAIRS` of 3 (§RQ3 above). `continuation_tokens` varies but
  measures output length, not verification-like behaviour, and the registered
  construct is the latter — so it is **not** the candidate it was previously
  recorded as (D41). Routes out are to register a validated in-construct
  measure, e.g. a judge-scored rubric with inter-rater agreement, before any
  confirmatory run; or to amend RQ3's question itself and say so. The
  direction *is* fittable (38 tail / 32 pre-stabilisation vectors from 11
  train problems), so the independent variable is not the obstacle.

---

## 6. How to reproduce or extend this

```powershell
$env:HF_HOME = "<volume>/hf_cache"      # bf16 staging
$env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints

.venv\Scripts\python.exe -m pytest                       # 513 tests, no GPU
.venv\Scripts\python.exe -m pytest -m slow               # adds the 4-bit GPU paths
.venv\Scripts\python.exe scripts\run_pipeline.py --config pilot --dry-run
```

The pilot's stages are all complete, so `run_pipeline.py --config pilot` is a
no-op. To re-run from a stage, pass `--from <stage> --force`.

**The larger run is in progress.** `env`, `data` and `models` are complete and
`traces` is running; the token caps are already raised in `main.yaml`. To
continue it after an interruption, the same command resumes per problem:

```powershell
.venv\Scripts\python.exe scripts\generate_traces.py --config main
```

Then the rest, in the order this hardware requires. **F8 is the exception to
the documented ordering:** it is placed before the autoencoders so it can run
while the bf16 target still exists, but all bf16 sources were evicted during
the pilot, so it cannot run at all without re-downloading 15.3 GB. The
download cache has 12.0 GB free, which is why the `env` stage's
`disk_for_bf16_staging` check fails; `D:` has over 50 GB, so repointing
`HF_HOME` there would make it possible. F8 measures a property of the
checkpoint rather than of the corpus, so the pilot's cosine of 0.987 carries
forward and re-measuring it is optional.

```powershell
.venv\Scripts\python.exe scripts\extract_activations.py --config main --layers 14 20 24
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --from ast
```

If the autoencoder stages need the three checkpoints and they cannot coexist,
the preparation splits in two:

```powershell
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --evict-bf16 target
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --skip target --force
```

All three NF4 checkpoints are currently present (14.8 GB: target on `D:`, AV
and AR reached through junctions to `E:`), so `models` is already a no-op.

---

## 7. What to watch on a larger run

**Run the consistency check first, then read this list.** Most of what follows
is a field to look up; `scripts/check_run_consistency.py --run runs/main` looks
up the mechanical half of it for you — it re-derives each summary from the
artifacts it came from, recomputes every q-value from the stored p-values,
enforces the invariants D38 and D43 argue for, diffs the report's prose against
the analysis, and asks git whether any stage's code has moved since the stage
ran. It is read-only and CPU-only, so it is safe to run mid-generation. Exit
code 1 means something below needs reading. What it cannot judge is in this
list.

- **Verbaliser integrity.** The `nla` stage logs `ok=N/M well-formed English`.
  It held at 0.976 here; below 50% it logs an error and the arm should be
  treated as compromised (cross-check F8).
- **Reconstruction FVE.** Read `fve_empirical_baseline`, not `fve`. The
  checkpoint card's 0.752 uses an unstated baseline and is not comparable to
  either.
- **`tail_rate` at 0 or 1.** Read `nontrivial_tail_rate` and
  `mean_tail_fraction` instead; §2 explains why.
- **The probe's margin over its positional floor.** Under 0.05 the stage warns,
  and the probe should not be described as a correctness readout. It was 0.034
  here.
- **F6's shuffle arms.** `leak_suspected: null` means the shuffle moved no
  labels and tested nothing, not that the probe is clean. The across-groups
  arm's AUC also depends on how donor labels are fitted to the receiver's
  length (`np.resize`; +0.240 under a length-correct mapping on `main`,
  [`F6_LENGTH_CORRECT_RESULTS.md`](F6_LENGTH_CORRECT_RESULTS.md)), so read it
  as a property of that rule as well as of the probe.
- **`total_clamped_positions`** in `acts/summary.json`. It was 0 here; anything
  else means an activation was read where its chunk boundary did not say.
- **`matched` in the stopping comparison table, and `o3_unmatched` in
  `analysis/results.json`.** A head-to-head that could not be matched is not a
  null result; it is an absent one. Read `granularity_limited` alongside it:
  true means the rule's score is too coarse to ever be matched and more
  problems will not help.
- **`n_discordant` in each O3 test.** This, not `n_problems`, is the
  comparison's effective sample size. Zero means the two rules preserved the
  outcome on the same problems — a tie, with nothing to test. Fewer than 8
  cannot clear q = 0.05 however lopsided they are.
- **`n_distinct_operating_points` against `n_budgets_requested`.** If the
  second greatly exceeds the first, the requested budgets are landing on the
  same points, which is the condition that used to inflate the p-value.
- **`n_positions_edited` in `causal/interventions.jsonl`, and
  `outcome_screen.interventions_were_no_ops`.** Zero on a non-zero dose means
  the direction edit never reached the forward pass and the arm is not
  evidence about anything. This was true of the entire pilot.
- **`outcome_screen.usable_outcomes`.** Empty means no candidate RQ3 readout
  varied *and* separated the arms, so the stage had no power regardless of
  what the verdict says.
- **`verdict.supported: null` in `causal/summary.json`.** Means the readout
  did not vary, so nothing was tested. Distinct from `false`.
- **`phases_present` in `nla`'s metrics.** Both `verbalise` and `reconstruct`
  must appear. A stage recorded as `partial` has one phase outstanding; only
  `complete` is skipped on a later pass.
