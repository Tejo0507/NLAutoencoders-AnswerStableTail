# `outputs/` — pilot-scale supervised analyses (not main-run results)

This directory holds the tables and figures written by the two root-level
scripts, `classification.py` and `regression.py`, by **repeated
cross-validation over the pilot's 24 traces**. They are an early, supplementary
look at how much cheap trace-surface features predict. They are **not** results
of the registered main run, and no conclusion of the report rests on them.

| Subdirectory | Script | Target | Rows |
|---|---|---|---|
| `classification/` | `classification.py` | final-answer correctness | 24 (pilot) |
| `regression/` | `regression.py` | tail fraction | 21 (pilot; 3 traces without a usable tail dropped) |

**Provenance.** Each subdirectory's `provenance.json` names:
- the runs pooled (here `pilot` only);
- the runs excluded or disqualified, and why;
- the row count;
- the pooling signature the corpus was selected on.

The files were last committed on 2026-10-10, before the main run existed.

**Where the main-run versions are.** The supervised analyses on the main run's
250 traces are separate pipeline stages, `scripts/predict_tail_fraction.py` and
`scripts/predict_correctness.py`. They have a held-out split and are reported
in `EXECUTION_REPORT_SUPERVISED.md` and in §6.8 of the report
(`paper/REPORT.md`) as supplementary analyses. Their methodology is in
`docs/SUPERVISED_ANALYSES.md`.

**Regenerating.** Running either script with its defaults rewrites the files
here, and it selects its corpus from whatever runs are present under `runs/`.
On a machine that holds the main run, that would no longer be the pilot. To
rebuild from the pilot only, pass `--runs pilot`. The code has changed since
these files were written, so a rebuild may not reproduce them byte for byte.
Pass `--out` to another directory to avoid overwriting the recorded files.
