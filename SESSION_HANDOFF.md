# Session handoff — 2026-10-10

Where the project stands, what this session did, and what to do next.

For *what the study found*, read [`docs/STATUS.md`](docs/STATUS.md) and
`runs/pilot/report/RESULTS.md`. This file is about the state of the work, not
the science.

---

## 1. One-line state

The pilot is **complete**: all thirteen stages, all 24 configured problems, the
released autoencoder, and the full nine-test falsification battery. The
execution report exists. **Thirteen local commits are unpushed** — pushing was
not authorised for this session.

---

## 2. Git

| | |
|---|---|
| branch | `main` |
| working tree | clean |
| local commits ahead of `origin/main` | **13** |
| pushed this session | **nothing — no push, fetch-write or remote upload was performed** |
| feature branches | all merged into local `main` with `--no-ff` and deleted |

The thirteen commits, oldest first:

```
84daa3a Measure FVE against the empirical baseline the plan specifies
400cafc Refuse to call a budget matched when it is not, and state O3's direction
a38da63 Merge empirical-fve-and-budget-match
f2517d4 Record the commit each stage ran under, not one per run
ddc90fc Merge per-stage-provenance
f3e5f57 Put both FVE baselines in the window table, and state F10's answer
4bea6d4 Merge f10-and-fve-in-the-table
aa10e8a Keep F5's measurement bound out of its definitional-stability range
ba65e75 Merge f5-separate-bound-from-criteria
2829882 Record the pilot's complete results and the decisions the run forced
12b68e4 Merge status-and-decisions-final
de03945 Lead the README with the answer to the central question
c7b42df Merge readme-final-findings
```

`origin/main` is at `02d722d`, which was pushed in the previous session.

**To publish, when authorised:** `git push origin main`. Nothing else is
needed; the history is linear-with-merges and no rewriting has occurred.

---

## 3. What this session did

The pilot's remaining stages had just finished, so the work was **validating
what they produced** rather than producing more. Five real defects came out of
reading the results against the plan.

| # | Defect | Consequence | Fixed in |
|---|---|---|---|
| 1 | `truncate`/`filler` forced an answer from `P[tail_start − 1]`, one chunk before the tail | Guaranteed to lose the answer. Reported accuracy 0.000 against a 0.800 baseline, which reads as "the tail is essential". After the fix: **0.800, zero answers changed** — the opposite conclusion | `02d722d` (prev. session), re-run here |
| 2 | The causal gate judged a readout that is **identically zero** on this corpus | `|0| > |0|` is false, so the gate said "not supported" when nothing had been tested | `02d722d`, verified here |
| 3 | FVE used a theoretical baseline of 2.0, not the empirical one §5 specifies | 0.866 instead of 0.677; the Gaussian control looked like 0.391 instead of **−0.470** | `84daa3a` |
| 4 | `at_matched_budget` returned the *nearest* point at any distance | Convergence at 0.407 tokens saved compared against the probe at 0.855 as though matched — what D9 exists to prevent | `400cafc` |
| 5 | The manifest recorded git state once per run | The report attributed a five-day, ~60-commit run to the commit its directory was created under | `f2517d4` |

Two reporting problems also fixed: F5 pooled a measurement bound into its
definitional-stability range (0.130 instead of **0.004**), and the fidelity
table's single `fve` column sat directly above the text explaining why that
baseline is wrong.

### Re-runs performed

All with `--force`, in dependency order, after each fix:

- `run_autoencoder.py --phase reconstruct` — the phase a stage-completeness bug
  had caused to be skipped entirely (fixed `d38239a`, prev. session)
- `run_interventions.py` — with the corrected truncation prefix
- `audit_faithfulness.py` — to regenerate the summary with the sensitivity
  fields (the audit itself resumes, so this cost one AR load)
- `run_falsification_tests.py --only F2 F5 F7` — completing the battery; these
  need no new compute and F7 turned out to matter
- `aggregate_results.py`, `write_run_report.py` — several times as fixes landed

---

## 4. Verification actually performed

| | |
|---|---|
| `pytest -m "not slow"` | **466 passed**, 3 skipped, 5 deselected |
| `pytest -m slow` (GPU) | **5 passed** |
| total | **471 passing**, up from 331 at the start of the previous session |

New test files this session: `tests/test_fve_baseline.py` (7 tests).
Extended: `tests/test_stopping.py` (+5 for the match tolerance),
`tests/test_causal.py` (+4 for the constant-outcome path).

The end-to-end fixture test (`tests/test_end_to_end_tiny.py`, 25 tests) runs
every stage script against 64-dimensional random stand-ins in about a minute,
and is what lets the autoencoder stages be exercised without 26 GB of
checkpoints.

**Not verified:** nothing in the pilot's *scientific* conclusions is replicated
— n = 24, one seed, one model. Two of three O3 head-to-heads could not be
evaluated at all. RQ3 was not tested.

---

## 5. Files changed this session

| file | change |
|---|---|
| `src/nlaast/nla/reconstructor.py` | `empirical_baseline_mse` |
| `src/nlaast/baselines/stopping.py` | `MATCH_TOLERANCE`, tolerance in `at_matched_budget` |
| `src/nlaast/analysis/tables.py` | both FVE columns, `matched`/`budget_gap` per row |
| `src/nlaast/provenance.py` | per-stage git commit |
| `scripts/run_autoencoder.py` | empirical baseline wired into the reconstruct metrics |
| `scripts/aggregate_results.py` | matched-budget filtering, unevaluable tests kept in the family, direction recorded |
| `scripts/run_falsification_tests.py` | F5 separates bound arms from criteria arms |
| `scripts/write_run_report.py` | O3 verdict block, FVE baseline block, F10 one-liner, per-stage commits |
| `docs/STATUS.md` | rewritten against the completed run |
| `docs/DECISIONS.md` | D25–D28 |
| `README.md` | central answer first |
| `tests/test_fve_baseline.py` | new |

---

## 6. Experiment and process status

Nothing is running. The GPU is free. No background jobs, no partial writes.

`runs/pilot/` holds the complete run and is **untracked** by design — it is
reproducible from the committed config, and `.gitignore` excludes `runs/`
because a partial run's tables read like findings.

Disk, for whoever runs this next: `D:` ~50 GB free, `E:` ~12 GB free. All three
NF4 checkpoints are present (target 5.2 GB on `D:`, AV 5.2 GB and AR 4.5 GB on
`E:`, reached through junctions from `D:\nla_models\nf4`). **All bf16 sources
have been evicted** — re-running F8 would need a fresh 15 GB download, which is
why F8 carries its earlier measurement forward rather than overwriting it.

---

## 7. Where to resume, in priority order

### 1. A larger corpus — the one thing that would change the conclusions

Two of three O3 comparisons could not be made at matched budget, and the third
rests on nine operating points over 21 problems. This is the binding
constraint on every quantitative claim.

`configs/main.yaml` is 250 problems; at the pilot's measured 12.9 min/problem
for traces alone that is over 50 hours on this machine. Before starting:

- **raise `generation.max_new_tokens` to 768.** Three of ten MATH traces hit
  the 512 cap, and those three are exactly the three with no detectable tail.
- **raise `ast.k_continuations` above 3** if F5's `k = 5` arm is wanted; it
  cannot be simulated upwards from what was recorded.
- consider `generation.force_answer_max_new_tokens` above 24. It blocked 24%
  of boundaries, which is why the measured tail is a lower bound.

The order this hardware requires, because the three checkpoints cannot coexist:

```powershell
$env:HF_HOME = "<volume>/hf_cache"; $env:NLAAST_QUANT_DIR = "<volume>/nf4"
# 1. traces + activations, then F8 while the bf16 target still exists
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --stages env data traces acts
.venv\Scripts\python.exe scripts\run_falsification_tests.py --config main --only F8_quantisation --force
# 2. the autoencoders, after evicting the bf16 target
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --evict-bf16 target
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --skip target --force
# 3. the rest
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --from ast
```

### 2. A behavioural readout for RQ3 that varies on this model

The causal direction *is* fittable — 38 tail and 32 pre-stabilisation vectors
from 11 train problems. What is missing is a dependent variable: the
rechecking-marker list is empty on every one of the 24 traces. Candidates:
entropy of the forced-answer distribution under intervention, or the
continuation's length, or a model that actually verbalises rechecking. Until
one exists, `causal/summary.json` will keep returning `supported: null`, which
is correct but uninformative.

### 3. Push, once authorised

`git push origin main`.

### Lower priority

- The two `scripts/predict_*.py` analyses still refuse below 30 train / 20 eval
  rows. A larger corpus unblocks them; they are correct to refuse at 24.
- `runs/mlcorpus` holds 33 traces from before the chunker and parser fixes.
  Not poolable, refused automatically (D22). Deleting it would be tidy.
- The `env` stage's `disk_for_bf16_staging` check fails by design on this
  machine — the eviction strategy is what makes the run possible. It could
  reasonably be taught about eviction rather than reported as a failure.

---

## 8. Honest limits

- **n = 24, one seed, one model, one layer.** Every interval is wide. The
  study's own report says so in several places and those caveats are load-
  bearing, not boilerplate.
- **The primary comparison is 1 of 3 evaluable.** Reporting it as "the readout
  loses" is accurate for the comparison that could be made and says nothing
  about the two that could not.
- **RQ3 produced no evidence in either direction.**
- **The three truncated MATH traces** are the three with no tail. That is the
  token cap showing up as a result, and it biases the corpus toward problems
  the model could finish in 512 tokens.
- **F8 cannot be re-measured** without re-downloading 15 GB, so the
  quantisation figure is fixed at what the 8-problem subsample gave.
