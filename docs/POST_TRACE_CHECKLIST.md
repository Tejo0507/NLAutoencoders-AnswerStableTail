# Post-trace checklist — `main`

What to run once `traces` finishes, in order, with the gate each step must pass.
Written 2026-10-15 against `main` at the commit that added it; every command and
field name below was checked against the scripts, and every CPU stage was
rehearsed end to end on a copy of the first 216 main traces
(`runs/rehearsal_main`, not `runs/main`).

**Run the stages by hand, in this order. Do not use `run_pipeline.py --from
ast`.** The orchestrator inserts `robustness:F8` after `acts`. Since D45 that
step no longer downloads anything — it reports `blocked` — so it is harmless,
but the manual order below is the documented one and skips it.

**One GPU job at a time.** The card has 6.14 GB; every GPU stage loads a ~5.5 GB
model. CPU stages may overlap a GPU stage, but free RAM has been ~2 GB with the
desktop open, and memory pressure is what slowed the traces stage 4x (handoff
§5). Overlap only the light ones marked below.

> **Corrected 2026-10-17, after the run this checklist was written for.** Four
> things in it were wrong, and are fixed in place below: (1) overlapping stages
> erased each other's manifest records until D48 — safe only with the
> `integrity-fixes` code; (2) step 5 cannot run before `nla` and `analysis`;
> (3) running `aggregate` before `nla` leaves `analysis` marked complete and
> stale, so it must be re-run with `--force`; (4) `--force` on the causal stage
> replayed the old rows until PR #23. On `main` all four were hit and are
> repaired; see `docs/AUDIT_2026-10-17.md`.

---

## 0. Gate: traces are actually finished

```powershell
(Get-Content runs\main\traces\traces.jsonl).Count          # expect 250
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId,CommandLine
```

Pass: 250 lines, **no** `generate_traces.py` process. Then check the corpus:
250 unique `problem_id`s, 0 `ok=false`. Report the achieved MATH truncation
rate; it was 7/81 = 0.086 at 203 traces.

## 1. `ast` — CPU, ~10 s

```powershell
.venv\Scripts\python.exe scripts\detect_answer_stable_tail.py --config main --sweep
```

- in: `traces/traces.jsonl` · out: `ast/ast.jsonl`, `ast/summary.json`, `ast/sweep_F5.json`
- pass: exit 0; `summary.json` has `median_tail_fraction`, `tail_token_share_all_traces`, `convergence_vs_ast.mean_gap_when_earlier` (D42); `sweep_F5.json` has `k=5` with `status: ran` (the main run records five continuations; the pilot could not run this arm)
- may run **while step 2 runs** on the GPU

## 2. `acts` — **GPU**

```powershell
.venv\Scripts\python.exe scripts\extract_activations.py --config main --layers 14 20 24
```

- in: `traces/` · out: `acts/blocks/*.npz`, `acts/index.jsonl`, `acts/summary.json`
- pass: `acts/summary.json` → `total_clamped_positions == 0`. Anything else means an activation was read where its chunk boundary did not say; stop.
- the three layers are required: F7 sweeps them

## 3. Supervised analyses — CPU, ~2 min each; may overlap step 2 or 5

```powershell
.venv\Scripts\python.exe scripts\predict_tail_fraction.py --config main
.venv\Scripts\python.exe scripts\predict_correctness.py --config main
```

- in: `traces/`, `ast/`, `data/problems.jsonl` (no activations)
- out: `regression_tail/`, `classify_correct/`
- pass: exit 0. On the rehearsal corpus the unpenalised logit does not converge and the odds table falls back to L2 with no intervals — that used to crash this stage and is fixed (62bd0e7). Report the fallback as such.

## 4. `baselines` — CPU, ~1 min

```powershell
.venv\Scripts\python.exe scripts\score_baselines.py --config main
```

- in: `traces/`, `ast/`, **`acts/`** · out: `baselines/curves.json`, `boundary_scores.json`, `probe.json`, `semantic_entropy.jsonl`
- **must run after step 2.** Without activations it logs `only 0 probe examples - skipping the probe`, drops the probe rule, and still records the stage `complete` — after which a plain re-run is skipped. If that happens, re-run with `--force`.
- pass: `probe.json` present; `rules` contains `probe`; read `positional_floor.probe_auc_above_floor` — under 0.05 the probe must not be described as a correctness readout (pilot: 0.034)

## 5. Size the NLA arm — decide from the standing recommendation

**`size_o3_corpus.py` cannot run here** — it reads `analysis/results.json` and
the O3 McNemar tests, which exist only after `nla` and step 9. Decide
`nla.max_problems` from the standing recommendation, **120** (power 0.71 →
0.93 at φ = 0.1 for ~1.3 h more GPU; 80 is the configured value), and re-run
the sizing after step 9 as a check:

```powershell
.venv\Scripts\python.exe scripts\size_o3_corpus.py --run runs\main
```

On `main` it reported 11 discordant problems of 120 against the probe (power
0.40 at φ = 0.1) — far fewer than the pilot's rate predicted.

## 6. `nla` — **GPU**, roughly 2 min/problem (estimate)

```powershell
.venv\Scripts\python.exe scripts\run_autoencoder.py --config main --set nla.max_problems=120
```

- in: `acts/`, `ast/` · out: `nla/verbalisations.jsonl`, `nla/reconstructions.jsonl`, `nla/summary.json`
- `--set` changes the resolved config hash (`ea82fa845911b70f` → `0bb33cf4c468a3f6`); the manifest records it, which is correct. **Do not edit `configs/main.yaml`.**
- pass: manifest `stages.nla.metrics.phases_present` contains both `verbalise` and `reconstruct` (a `partial` stage is not skipped and must be resumed); log `ok=N/M well-formed English` at or near the pilot's 0.976 — under 50% treat the arm as compromised

## 7. `faithfulness` — **GPU**

```powershell
.venv\Scripts\python.exe scripts\audit_faithfulness.py --config main
```

- in: `nla/`, `acts/`, `traces/` · out: `faithfulness/audits.jsonl`, `faithfulness/summary.json`

## 8. `causal` — **GPU**

Validate the D34 fix on the pilot first (#8; the pilot arm is 10 problems):

```powershell
.venv\Scripts\python.exe scripts\run_interventions.py --config pilot --force
```

Pass: every non-zero dose in `runs\pilot\causal\interventions.jsonl` has `notes.n_positions_edited > 0`, and `ablate_dir` now differs from dose 0.0. If the edits are still zero, **stop**: the main causal arm would be no-ops too.

**`--force` needs PR #23.** Without it the writer resumes by row id and replays the previous rows in seconds — the first attempt on 2026-10-17 did exactly that. With it the old rows are moved to `interventions.superseded-<mtime>.jsonl`. Done on 2026-10-17: 10/10 problems edited, `ablate_dir` differed from dose 0 on 16 of 20 rows.

To re-derive the verdict from existing rows after a change to the gate, **without the GPU**: `run_interventions.py --config <run> --summarise-only` (D47), then `aggregate_results.py --force` and `write_run_report.py --force`. The summary records the SHA-256 of the rows it was built from, and `check_run_consistency.py` fails if the rows, the summary, the analysis or the report disagree (D50). Then:

```powershell
.venv\Scripts\python.exe scripts\run_interventions.py --config main
```

- in: `traces/`, `ast/`, `acts/` · out: `causal/interventions.jsonl`, `causal/summary.json`
- pass: non-zero `n_positions_edited` on non-zero doses; read `outcome_screen.usable_outcomes`. RQ3's registered readout is already known to have no range on this model (1 marker in 200 traces), so expect `verdict.supported: null` — that is "not tested", not "refuted"

Steps 6–8 are independent of each other except 7 needs 6; 8 may run before 6.

## 9. Analysis, battery, reports — CPU, a few minutes in all

Only after steps 6–8. An `aggregate` run earlier (for example to try step 5)
records `analysis` as complete, and a plain re-run is then skipped: add
`--force` to `aggregate_results.py` and `write_run_report.py`.

```powershell
.venv\Scripts\python.exe scripts\aggregate_results.py --config main
.venv\Scripts\python.exe scripts\run_falsification_tests.py --config main
.venv\Scripts\python.exe scripts\write_run_report.py --config main
.venv\Scripts\python.exe scripts\check_run_consistency.py --run runs\main
.venv\Scripts\python.exe scripts\write_supervised_report.py --config main
.venv\Scripts\python.exe scripts\measure_rq3_readout_floor.py
.venv\Scripts\python.exe scripts\assemble_paper.py --check
```

- `run_falsification_tests.py` is CPU-only now: F8 reports `blocked` with `download_attempted: false` (D45). The pilot's 0.987 is the F8 measurement that stands; **do not pass `--allow-download`** without the user's say-so (15.3 GB, cache volume has 12 GB).
- pass for `check_run_consistency.py`: exit 0. Exit 1 means a number disagrees with its artifact; fix before quoting anything.
- read in `analysis/results.json`: `o3_unmatched` (absent ⇒ every head-to-head was made), each O3 test's `n_discordant` (under 8 cannot clear q = 0.05), `tests.bh_family_size`, and `verdict_is_robust_to_family_convention`
- `write_supervised_report.py` writes `EXECUTION_REPORT_SUPERVISED.md` at the repository root unless `--out` is given

## 10. Write up

Results, Discussion and Abstract go into `paper/06_results.md`,
`paper/07_discussion.md` and the Abstract, from `runs/main/report/RESULTS.md`
and `analysis/results.json` only. Declare each file in
`scripts/assemble_paper.py`'s `SECTIONS` and fill the `[N]` entries in §4 and
§8. `assemble_paper.py` exits 0 only when nothing is pending.

---

## What is reused and what reruns

| | status | why |
|---|---|---|
| `env`, `data`, `models` | **reuse** | complete; all three NF4 checkpoints present |
| `traces` | **reuse** | the thing being waited on |
| pilot `runs/pilot/*` | **reuse** | regenerated 2026-10-15; `check_run_consistency` 153 checks, 0 mismatched |
| pilot F8 (cos 0.987) | **reuse** | property of the checkpoint; bf16 source evicted |
| everything in steps 1–9 for `main` | **run** | nothing downstream of `traces` exists for `main` |
| pilot `causal` | **rerun** (step 8) | predates the D34 fix; the arms were no-ops |

## GPU vs CPU

GPU: `acts`, `nla`, `faithfulness`, `causal` (and the pilot causal validation).
CPU: everything else — `ast`, `baselines`, the supervised analyses,
`analysis`, `robustness`, `report`, and the four auditors.
