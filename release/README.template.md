# NLA Answer-Stable Tail — run artifacts

{STATUS_BANNER}

The stored outputs of the study *Natural Language Autoencoders for
Redundancy Diagnostics: the Answer-Stable Tail*. The code that produced them,
the pre-registration (`PROJECT_PLAN.md` §5–§6) and the record of every
deviation (`docs/DECISIONS.md`) are in the code repository:

- Repository: https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail (MIT)
- Source commit: `{SOURCE_COMMIT}`

Nothing here was regenerated for the release. Every file is a byte-for-byte
copy of the run directory it comes from; only this README, `RUNS.json`,
`NOTICE.md`, `LICENSE`, `licences/` and `SHA256SUMS.txt` were added.

## What is in it

`RUNS.json` lists every directory with its classification, decision record,
file count and size. In short:

| Directory | Classification | What it is |
|---|---|---|
| `runs/main/` | **registered** | The study's main run: 250 problems (150 GSM8K, 100 MATH), every stage. The NLA arm covers 120 problems (`--set nla.max_problems=120`, recorded per stage, D48). |
| `runs/pilot/` | **registered (pilot)** | The 24-problem pilot, all stages. A scale check, not the confirmatory corpus. |
| `runs/main_f8_bf16/`, `runs/main_f8_nf4/`, `runs/main_f8_comparison.json` | **registered test, partial** | F8's bf16 replication run *in part* by decision (D55): bf16 target activations on 24 problems; verbaliser, reconstructor and behavioural arms stayed NF4. Not the full registered replication. |
| `runs/main_nla240/` | **exploratory, post hoc** | The NLA arm extended to 240 problems after the N = 120 result was seen (D54). Its inputs are copied from `main` (`SOURCE_SHA256SUMS.txt`, checked from inside `runs/main/`). |
| `runs/main_rq2probe/` | **exploratory** | RQ2 amendment A1 (D56–D57): an independent-probe control on rule-derived claim labels without human validation. Includes `annotation_sheet_v3.ai_claude_not_independent.jsonl`, an AI-produced annotation sheet that is **not** an independent human annotation, and two superseded blank sheets. |
| `runs/main_f6_audit/` | **exploratory diagnostic** | Label statistics of F6's `np.resize` shuffle (`docs/F6_RESIZE_AUDIT.md`). |
| `runs/main_f6_diag/`, `runs/main_f6_diag_validation/` | **exploratory diagnostic** | F6's length-correct diagnostic (D58–D60), 100 draws, and its independent validation (27 checks, 0 failed). |

**Exploratory and partial results are not registered confirmatory
results** and must not be read as such. The registered conclusions are those
of `runs/main/`; F6's registered result is `runs/main/robustness/robustness.json`
and the diagnostic does not change it. RQ3 is reported as *not tested* on
this model (D62).

**Left out**, as not results: `calib`, `mlcorpus` (traces from before the
chunker and parser fixes), `smoke`, `tiny` (synthetic test fixture) and
`main_f5` (a 2-problem timing check for the deferred F5 sweeps), plus the
empty `manifest.json.lock` files every run directory holds while a stage
writes.

## Layout of a run directory

`config.resolved.yaml` is the configuration it ran under; `manifest.json`
records each stage's status, configuration hash, overrides, git commit, model
revisions and metrics. One subdirectory per stage (`data`, `traces`, `acts`,
`ast`, `baselines`, `nla`, `faithfulness`, `causal`, `analysis`, `robustness`,
`report`, …); `docs/PIPELINE.md` in the repository documents each one.
Activations are `acts/blocks/<problem>__L<layer>.npz` with a content digest in
`acts/index.jsonl`. `*.out`, `*.err` and `run.log` are the stages' own logs.

## Verifying it

From the directory holding this README:

```bash
sha256sum -c SHA256SUMS.txt           # every file, this README included
```

The archive's own SHA-256 is published beside it. The repository's
`scripts/check_run_consistency.py --run <dir>` re-derives the analysis
checks from a run directory placed under the repository's `runs/`.

## Known limitations

- Reproducing the runs needs the three Qwen-7B-class checkpoints (~41 GB in
  bf16), converted to 4-bit NF4, and a CUDA GPU; this study ran on an RTX
  4050 Laptop (6.44 GB VRAM). See the repository README.
- The files record **local absolute paths** of the machine they ran on
  (repository, interpreter, model and cache directories) in `env/`, the
  manifests, `models/` and the logs. They are kept unedited because they are
  part of each stage's provenance; they contain no user name and no
  credential.
- The pilot's manifest holds a stale `classify_correct: running` record from
  an attempt the stage refused at 24 problems (audit PW-21).

## Licence

The artifacts are released under **Creative Commons Attribution 4.0
International** (`LICENSE`). Problem text from GSM8K and MATH (MIT) and
outputs of Apache-2.0 models are included; their notices are in `NOTICE.md`
and `licences/`.
