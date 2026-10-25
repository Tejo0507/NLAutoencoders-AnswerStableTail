# NLA Answer-Stable Tail — run artifacts

{STATUS_BANNER}

The stored outputs of the study *Natural Language Autoencoders for
Redundancy Diagnostics: the Answer-Stable Tail*. The code that produced them,
the pre-registration (`PROJECT_PLAN.md` §5–§6) and the record of every
deviation (`docs/DECISIONS.md`) are in the code repository:

- Repository: https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail (MIT)
- Source commit: `{SOURCE_COMMIT}`

Nothing here was regenerated for the release, and no number was changed.
Every file under `runs/` is a copy of the run directory it comes from.
**{N_SANITISED} of them are release-transformed**: the roots of local
absolute paths were replaced (see *Sanitised paths* below), and
`TRANSFORMS.json` lists each one. Every other file is byte-for-byte
identical to the original. Added for the release: this README, `RUNS.json`,
`TRANSFORMS.json`, `NOTICE.md`, `LICENSE`, `licences/` and `SHA256SUMS.txt`.

## What is in it

`RUNS.json` lists every directory with its classification, decision record,
file count, number of sanitised files and original size. In short:

| Directory | Classification | What it is |
|---|---|---|
| `runs/main/` | **registered** | The study's main run: 250 problems (150 GSM8K, 100 MATH), every stage. The NLA arm covers 120 problems (`--set nla.max_problems=120`, recorded per stage, D48). |
| `runs/pilot/` | **registered (pilot)** | The 24-problem pilot, all stages. A scale check, not the confirmatory corpus. |
| `runs/main_f8_bf16/`, `runs/main_f8_nf4/`, `runs/main_f8_comparison.json` | **registered test, partial** | F8's bf16 replication run *in part* by decision (D55): bf16 target activations on 24 problems; verbaliser, reconstructor and behavioural arms stayed NF4. Not the full registered replication. |
| `runs/main_nla240/` | **exploratory, post hoc** | The NLA arm extended to 240 problems after the N = 120 result was seen (D54). Its inputs are copied from `main` (`SOURCE_SHA256SUMS.txt`, checked from inside `runs/main/`). |
| `runs/main_rq2probe/` | **exploratory** | RQ2 amendment A1 (D56–D57): an independent-probe control on rule-derived claim labels without human validation. Includes `annotation_sheet_v3.ai_claude_not_independent.jsonl`, an AI-produced annotation sheet that is **not** an independent human annotation, and two superseded blank sheets. |
| `runs/main_f6_audit/` | **exploratory diagnostic** | Label statistics of F6's `np.resize` shuffle (`docs/F6_RESIZE_AUDIT.md`). |
| `runs/main_f6_diag/`, `runs/main_f6_diag_validation/` | **exploratory diagnostic** | F6's length-correct diagnostic (D58–D60), 100 draws, and its independent validation (27 checks, 0 failed). |
| `runs/rep1/`, `runs/rep1_forced/`, `runs/rep1_o3_registered/` | **confirmatory** | The pre-registered replication on 400 fresh problems (D75–D80). `rep1_o3_registered/o3_registered.json` holds the D75 verdict (D94). |
| `runs/rep1_position_audit/`, `runs/rep1_information/`, `runs/rep1_selective/` | **pre-specified secondary** | The position/truncation audit (D79), information audit and lexicon transfer (D81), and selective risk (D83 deviation 1), on rep1 (D94–D95). Not part of the verdict. |
| `runs/rep1_extforced/` | **pre-specified sensitivity** | The 256-token forcing pass (D82, D95). It qualifies the verdict and never replaces it. |
| `runs/rep1_stage1/` | **pre-specified** | Stage 1: NLA text against a same-label probe at outcome-blind positions (D83, D85, D96). |
| `runs/rep1_runbook/`, `runs/post_rep1_queue/` | **logs** | The replication pipeline's runbook, resume and queue logs. |
| `runs/main_forced/`, `runs/main_o3_registered/`, `runs/main_nla240_o3_registered/`, `runs/main_claims_audit/`, `runs/main_claim_checks/`, `runs/main_paraphrase_qc/`, `runs/main_corrected_tests/` | **post hoc / corrected** | The post-hoc reanalysis of `main` (D68–D72, `docs/REANALYSIS_2026-10-19.md`). D91 records a provenance gap in the O3 reanalysis plan. |
| `runs/main_coverage/`, `runs/main_position_audit/`, `runs/main_nla240_position_audit/`, `runs/main_information/`, `runs/main_nla240_information/`, `runs/main_offmenu_stage0/`, `runs/main_nla240_extforced/` | **exploratory** | Analyses on already-seen data (D73, D79, D81, D83–D84, D96). |
| `runs/main_selective/`, `runs/main_nla240_selective/` | **post hoc** | Selective risk on the candidate menu (D83 deviation 1). Not deployment evidence. |
| `runs/main_rq3_pilot/`, `runs/main_rq3_pilot_stage0/` | **exploratory pilot** | RQ3 follow-up pilot; Stage 0 was a no-go (D74, D77). |
| `runs/stage1_power_audit/` | **planning** | Stage 1's power audit on stored data (D85). |
| `runs/bench_*` | **engineering benchmark** | Batch-size benchmarks behind D77 and D78. |
| `runs/main_paraphrase_llm/`, `runs/main_paraphrase_llm_logs/` | **exploratory** | The D76 rewording null for RQ2 (60 explanations of `main`, 246 claims), run in an isolated tree (D97), with its phase logs and input and output SHA-256 lists. The rewordings were checked for form, not meaning; `rewording_validation_sheet.csv` is the blank sheet for human judgement. Not a faithfulness test. |
| `runs/pilot_cal/`, `runs/pilot_eval/`, `runs/pilot_models/`, `runs/pilot_calibration/`, `runs/pilot_evaluation/`, `runs/pilot_evaluation_supplement/`, `runs/pilot_logs/` | **exploratory pilot** | The stopping-policy pilot (D99–D101), run in an isolated tree under a protocol frozen before any data: thresholds calibrated on 300 fresh problems, one evaluation pass on 150 others. P1's selective risk was 0.041 with a one-sided 95% upper bound of 0.078, above α = 0.05, so **its safety criterion was not met**; it is not a safety certification. `pilot_models/models.pkl` is a Python pickle: check its SHA-256 against `models.json` before loading it. |
| `runs/pilot_posthoc/` | **post hoc** | Where P1 stops on the evaluation set, computed after the results were seen (D102). |
| `runs/rep1_checkout_state_manifest/` | **provenance** | The SHA-256 list, `git status`, `HEAD` and branch of the uncommitted code state that every rep1 output records (`git.dirty: true` at `b17c131`). The files themselves are kept offline because they hold local paths; any copy can be checked against this list. |

**Only `runs/main/` (the registered study) and the rep1 confirmatory
directories carry registered or confirmatory results.**
- Everything else is pre-specified secondary, sensitivity, exploratory,
  post hoc, planning or engineering, and must not be read as confirmatory.
- F6's registered result is `runs/main/robustness/robustness.json`, and the
  diagnostic does not change it.
- RQ3 is reported as *not tested* on this model (D62).
- No directory here establishes that the NLA explanations are faithful.

**Left out:**
- as not results: `calib`, `mlcorpus` (traces from before the chunker and
  parser fixes), `smoke`, `tiny` (synthetic test fixture), `main_f5` (a
  2-problem timing check for the deferred F5 sweeps) and
  `main_rq3_pilot_smoke`;
- the empty `manifest.json.lock` files every run directory holds while a
  stage writes;
- **withheld:** `main_claim_annotation`, the blinded human-annotation sheets
  and their unblinding key, until the human annotation is complete.

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

A checksum list *inside* a run refers to the original files. Where it names a
sanitised file, compare its entry with that file's `original_sha256` in
`TRANSFORMS.json`, not with the released file. This applies to two entries of
`runs/main_nla240/SOURCE_SHA256SUMS.txt` (`acts/summary.json` and
`data/validation.json`, checked from inside `runs/main/`).

## Sanitised paths

The runs recorded the absolute paths of the machine they ran on: stage output
directories, data source files, the Python interpreter, model checkpoint and
cache directories, and file paths in one logged traceback. These are kept,
with their roots replaced:

| Original root | Released as | Example |
|---|---|---|
| the repository's directory | `<repo>` | `<repo>\runs\main\traces` |
| any other local directory | `<local>`, minus the drive and the first directory below it | `<local>\nf4\<model directory>` |

What each path points at, relative to the repository or to the model store,
is unchanged; which disk and which local folder held it is not published.
Model identity does not depend on these paths: repository ids and revisions
are recorded in the manifests separately. `TRANSFORMS.json` gives, for each
sanitised file, its original and released SHA-256, the JSON fields or log
lines changed and the number of replacements. The original files are
unchanged in the study's own copies.

## Known limitations

- Reproducing the runs needs the three Qwen-7B-class checkpoints (~41 GB in
  bf16), converted to 4-bit NF4, and a CUDA GPU; this study ran on an RTX
  4050 Laptop (6.44 GB VRAM). See the repository README.
- Local absolute paths are sanitised as described above, so a path in a
  manifest or log cannot be opened as written. Resolve `<repo>` to a checkout
  of the repository.
- The pilot's manifest holds a stale `classify_correct: running` record from
  an attempt the stage refused at 24 problems (audit PW-21).

## Licence

The artifacts are released under **Creative Commons Attribution 4.0
International** (`LICENSE`). Problem text from GSM8K and MATH (MIT) and
outputs of Apache-2.0 models are included; their notices are in `NOTICE.md`
and `licences/`.
