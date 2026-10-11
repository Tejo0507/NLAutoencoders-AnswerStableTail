# Session handoff — 2026-10-11

Where the project stands, what this session did, and what to do next.

For *what the study found*, read [`docs/STATUS.md`](docs/STATUS.md) and
`runs/pilot/report/RESULTS.md`. This file is about the state of the work, not
the science.

---

## 1. One-line state

The pilot's committed history is **pushed**. Re-reading the pilot's analysis
against its own statistics found that **the study's one significant primary
finding was an artefact of a default parameter**, and that three of five causal
arms had never done anything. Both are fixed, with tests. The **main run is
under way**: `env`, `data` and `models` are complete and `traces` is
generating, resumable per problem.

---

## 2. Git

| | |
|---|---|
| branch | `main` |
| working tree | clean |
| `origin/main` | synchronized — see below |
| feature branches | `o3-complete-threshold-grid`, merged `--no-ff` and deleted |

**The push asked for was performed and verified.** Before anything else, the
14 local commits were pushed to
`https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail.git`:

```
02d722d..a527c45  main -> main
```

The handoff's count of 13 was stale — the handoff commit itself made 14. The
fetch beforehand showed `0` commits on the remote that were not local, so this
was a fast-forward; no force, no rewrite, no reset. `git rev-list --count`
afterwards was `0 0`.

This session then added **9 commits** on `main` (eight plus the merge),
including this handoff, and pushed them to the same remote. The eight before
this file, oldest first:

```
6c48ce3 Test the matched-budget comparison over problems, not over budget levels
8ebb7ff Order the problem set so that any prefix of a run is representative
6989fec Cap the NLA arm proportionally, not by sorted problem id
bc6b064 Raise the main run's token caps off the two limits the pilot hit
d799285 Make the direction interventions reach the computation they edit
c2cec38 Correct the record: the primary finding was significant by default, not by data
1a46598 State the primary result in the report and README as the data supports it
836391c Merge o3-complete-threshold-grid
```

---

## 3. What this session found

Four real defects, all in code that had already produced a completed run. The
first changes a headline.

### 1. The primary O3 test's p-value was set by a grid size, not by the data

O3 reads each rule off its safety/saving curve at a shared token budget and
compared the two safe rates with a **Wilcoxon signed-rank test over the nine
requested budgets**. All nine resolved to a *single* pair of operating points —
safe rates 0.095 and 0.286 at every one of them — so the test saw nine copies
of one comparison:

| budgets | 3 | 5 | **9** | 15 | 25 |
|---|---|---|---|---|---|
| Wilcoxon p | 0.250 | 0.0625 | **0.0039** | 0.000108 | 0.000001 |

Nine is `budget_grid`'s default and 0.0039 was reported as **q = 0.0065**.

Safety is a per-problem binary outcome and both rules are scored on the same
problems, so the unit of analysis is the **problem**. McNemar's exact test over
problems gives **4 discordant problems of 21, all four favouring semantic
entropy: p = 0.125, q = 0.156**. The direction survives; the significance does
not. (D29)

### 2. Two of three head-to-heads were unevaluable because of the threshold grid

Each curve was swept on 25 evenly spaced thresholds, which samples the *score
axis* rather than the rule. The probe's scores are logistic and saturate, so
25 linear thresholds found 5 operating points and none near a shared budget —
reported as "not evaluable on this corpus", implying a sample-size limit.

Sweeping the **observed score values** enumerates every achievable operating
point exactly. The same 21 problems then give the probe **86** operating points
and a match at all nine budgets, where it is **exactly tied** with the readout:
zero discordant problems, the two rules preserving the outcome on the same
problems. (D30)

The convergence comparison is genuinely unevaluable, and **no corpus size will
change that**: its score is 1.0 once the last `window` answers agree and 0.0
otherwise, so `d = 2` and its curve has three operating points at any N, with
savings of 0.407 and 0.885 straddling the budget window. It needs a graded
score, not more data. `granularity_bound` now reports this per rule. (D31)

### 3. A partial run would have been all GSM8K and no MATH

`main.yaml` states that the achieved N is what the report states, which only
holds if stopping early leaves a balanced corpus. `load_problems` concatenated
the datasets, so all 150 GSM8K problems preceded all 100 MATH ones. A run
halted at 120 would have had no MATH: F2 with one arm, every pooled quantity
biased toward the easier benchmark, and nothing in the output saying so.

Worse, the **NLA arm's cap** took its subset as a prefix of the *sorted problem
ids*, and ids sort `gsm8k-...` before `math-...`. `nla.max_problems: 80` on 250
problems would have made that arm 80 GSM8K problems and no MATH — and the
primary O3 head-to-head is computed on exactly the problems that arm covers.
Both are now proportional. (D33)

### 4. Three of the five causal arms were no-ops, for the whole pilot

`projection_editor` restricts its edit to `[start, end)` **absolute token
positions**, but applied the span as an index into whichever tensor the forward
hook received. The arms edit the *generated* continuation, so the span starts
at the prompt length; cached decoding passes the whole prompt once and then one
token at a time, and the span intersected neither.

The pilot's own output says so: on **all ten problems**, `ablate_dir` returned
identical answers and identical token counts at coefficients 0.0, 1.0 and 2.0 —
and dose 0.0 is a no-op by definition.

This revises what RQ3's failure was. It was recorded as a missing dependent
variable, with the direction fittable and the marker list empty. Both true; the
*independent* variable was also not being applied. The editor now counts the
positions it modifies, the stage warns when a non-zero dose edits nothing, and
the screen reports a broken independent variable rather than a negative result.
The defect survived a completed run because nothing asserted the intervention
had happened. (D34)

---

## 4. Verification actually performed

| | |
|---|---|
| `pytest -m "not slow"` | **513 passed**, 3 skipped, 5 deselected |
| at session start | 466 passed |
| new tests this session | **+47** |

New or extended: `tests/test_stopping.py` (+17: the complete grid, the
granularity bound, the no-saving budget), `tests/test_stats.py` (+12: McNemar
over problems, the conditional-exact interval, and explicitly that repeating
one comparison must not manufacture significance), `tests/test_data.py` (+14:
interleaving and the stratified cap, including a test asserting that the old
sorted-id prefix would have dropped a dataset), `tests/test_causal.py` (+12:
the cached-decode span, no-op detection, the outcome screen).

**`pytest -m slow` (GPU) was not re-run** — the GPU is held by the main run.
It passed 5/5 at the end of the previous session and nothing this session
touched the 4-bit loading paths.

**Not verified:**

- **The causal fix has not run on a GPU.** It is unit-tested, including the
  cached-decode case that was failing, but no intervention has actually been
  re-executed. Until it has, RQ3 remains untested and `continuation_tokens`
  remains an untested *candidate* readout, not a validated one.
- Nothing in the pilot's scientific conclusions is replicated: n = 24, one
  seed, one model, one layer.
- F8 has not been re-measured (see §6).

---

## 5. The main run

**Status: `traces` running.** Launched 10:58 on 2026-10-11, detached, logging
to `runs/main/traces_stage.out`. `env`, `data` and `models` are complete.

```powershell
# resumes per problem; safe to run after any interruption
.venv\Scripts\python.exe scripts\generate_traces.py --config main
```

**It is safe to stop at any point.** The problem order is interleaved, so any
prefix is proportional: 24 problems is 14 GSM8K / 10 MATH, 120 is 72 / 48.

**Throughput, measured rather than assumed.** The handoff's 12.9 min/problem
made 250 problems "over 50 hours". That figure is a degraded machine state, not
a cost: the pilot generated the same traces *twice*, byte-identical, once at
about 2 min/problem and once at 12.9, the slow pass with 1.3 GB of free RAM. A
3-problem calibration under `main`'s own settings measured **3.2 min/problem**
for GSM8K. MATH is substantially heavier — longer traces and `k = 5`
continuations of 200 tokens at up to 12 boundaries — and the first MATH problem
of the real run took over 15 minutes. Budget roughly **35–40 hours for all
250**, dominated by MATH. The stage's own ETA line is the number to trust as it
accumulates.

**Config changes made** (D32, and `configs/main.yaml` carries the reasoning):

| key | was | now | why |
|---|---|---|---|
| `generation.max_new_tokens` | 640 | **768** | 3 of 10 pilot MATH traces hit the cap, and those 3 were exactly the 3 with no detectable tail |
| `generation.force_answer_max_new_tokens` | 24 | **48** | the 24-token budget blocked criterion 1 on 24.1% of boundaries while every continuation reached the answer; ~2% more generation to remove a 24% false-negative rate |
| `ast.k_continuations` | 5 | 5 | already correct; F5's `k = 5` arm will be simulable from this run |

**250 problems was deliberately *not* cut.** `scripts/size_o3_corpus.py` sizes
the run against the primary question: power comes from **discordant** problems,
8 are needed to clear q = 0.05 in a family of five evaluable tests, and the
pilot's rate of 4/21 = 0.190 puts that at **about 42 problems in the NLA arm**.
Simulated power at 80 problems is 0.99 if discordances keep falling the pilot's
way and 0.71 if one in ten favours the readout; at 120 it is 1.00 and 0.93.
Since the run is resumable and the order is interleaved, the better plan is to
let it run and stop when the comparison has what it needs, rather than commit
to an N now.

---

## 6. Environment and constraints

Nothing else is running. `D:` 41 GB free, `E:` 12 GB free. All three NF4
checkpoints present and verified mutually consistent, injection site confirmed
at token 111: target 5.18 GB on `D:`, AV 5.18 GB and AR 4.42 GB reached through
junctions to `E:`. The target model occupies 5.55 GB of 6.14 GB VRAM.

**The documented "run F8 while the bf16 target still exists" ordering is moot:
all bf16 sources were already evicted during the pilot.** F8 cannot run without
re-downloading 15.3 GB, and `HF_HOME` points at `E:` with 12.0 GB free — which
is exactly why the `env` stage's `disk_for_bf16_staging` check fails, as
designed. `D:` has over 40 GB, so repointing `HF_HOME` there would make it
possible.

F8 was **deliberately deferred**, not forgotten: the quantisation cosine is a
property of the checkpoint, not of the corpus, so the pilot's 0.987 over 82
vectors carries forward. Re-measuring it on more problems buys precision on a
number that is already far from the threshold that would matter.

Note the machine has only ~2.7–3.8 GB of free RAM with the user's applications
open, and the pilot's 7x slowdown is attributable to RAM pressure. Avoid
running the test suite or other GPU work alongside the traces stage; doing so
roughly doubled the first problem's wall time this session (601s against the
calibration's 316s on the same problem).

---

## 7. Where to resume, in priority order

### 1. Let `traces` finish, then continue the pipeline

```powershell
.venv\Scripts\python.exe scripts\extract_activations.py --config main --layers 14 20 24
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --from ast
```

Skip `robustness:F8` or expect it to fail on disk (§6). If `run_pipeline` is
used instead of the stage scripts, it will attempt F8 after `acts`.

**Before the `nla` stage, decide `nla.max_problems`.** It is 80, which the
power table puts at 0.99/0.71 depending on how discordances fall; 120 gives
1.00/0.93 for 50% more verbaliser cost. The arm is now a proportional sample
either way, so this is purely a power-versus-cost choice.

### 2. Validate the causal fix on a GPU — the one thing this session could not

Once the GPU is free:

```powershell
.venv\Scripts\python.exe scripts\run_interventions.py --config pilot --force
```

Then check, in order:

1. `n_positions_edited` in `runs/pilot/causal/interventions.jsonl` is non-zero
   for every non-zero dose. If it is zero the arms are still no-ops and
   nothing else in the stage means anything.
2. `outcome_screen.usable_outcomes` in `causal/summary.json`. Empty means no
   candidate readout both varies and separates the arms, so the stage still
   has no power — which is a result about the design, not about the tail.
3. Whether `ablate_dir` now differs from dose 0.0 at all. The pilot's
   identical-across-doses output is the baseline to beat.

Only if a readout separates the arms is there a dependent variable to
pre-register for RQ3. `continuation_tokens` is the leading candidate — already
recorded, continuous, and a question about how much further the model writes
rather than about what the text means. **Do not pick the readout that gives
the nicest answer**; that is what `screen_outcomes` exists to prevent. (D35)

### 3. Re-run the pilot analysis if anything upstream changes

`score_baselines.py`, `aggregate_results.py` and `write_run_report.py` were all
re-run this session and `runs/pilot/` reflects the corrected analysis. They are
CPU-only and cheap.

### Lower priority

- The two `scripts/predict_*.py` analyses still refuse below 30 train / 20 eval
  rows. The main run unblocks them; they are correct to refuse at 24.
- `runs/mlcorpus` holds 33 traces predating the chunker and parser fixes. Not
  poolable, refused automatically (D22). Deleting it would be tidy.
- `runs/calib` is a 3-problem throughput calibration from this session. Safe to
  delete.
- The `env` stage's `disk_for_bf16_staging` check could be taught about
  eviction rather than reported as a failure.
- The agreement rule needs a **graded** score if its head-to-head is ever to be
  evaluable (D31) — for instance the fraction of the window that agrees, rather
  than unanimity as a boolean.

---

## 8. Honest limits

- **The primary comparison is now significant nowhere.** The readout is
  directionally worse than semantic entropy on 4 discordant problems of 21, and
  exactly tied with the probe. Read with the probe's positional floor — 0.996
  AUC against 0.962 from position alone — a readout that ties a largely
  positional predictor is not obviously reading more than position.
- **One of three head-to-heads cannot be made at any corpus size.**
- **RQ3 produced no evidence in either direction, and now for two known
  reasons.** The fix to the second is unit-tested and unvalidated.
- **n = 24, one seed, one model, one layer** for everything in `docs/STATUS.md`.
  The main run changes the first of those and none of the others.
- **The three truncated MATH traces** were the three with no tail. The main run
  raises the cap, which should remove that bias — and will change the tail
  numbers, so they are not directly comparable with the pilot's. The forcing
  budget change does the same, deliberately (§5).
- **F8 is fixed at the pilot's 8-problem subsample** and cannot be re-measured
  without 15 GB of downloads.
- **A second annotator** for O6 inter-rater reliability was never obtained.
  Single-rater, as the review anticipated.
