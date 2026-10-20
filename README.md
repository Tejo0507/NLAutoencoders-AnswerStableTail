# NLA Answer-Stable Tail

**Natural Language Autoencoders for Redundancy Diagnostics: the Answer-Stable Tail**

*Does a verbalised readout of residual-stream activations tell us anything
useful about the part of a reasoning trace that comes after the answer has
already settled — beyond what much cheaper signals already say?*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-complete-brightgreen)]()

## Overview

Reasoning models often keep writing after their answer is settled. This
project calls that region the **Answer-Stable Tail (AST)**, defines it from
behaviour alone, and asks whether a **natural-language autoencoder** (NLA)
helps tell where it begins. An NLA is a released pair of models: one
*verbalises* an activation as text, the other *reconstructs* the activation
from that text. The readout is compared with cheaper stopping signals:
answer convergence, a hidden-state correctness probe and semantic entropy.
The comparison is made at matched token budget, under a pre-registered
analysis plan, on Qwen2.5-7B-Instruct run in 4-bit.

**What the study found.** These are the frozen report's results, read with
the post-hoc audit of 2026-10-19 ([addendum](paper/ADDENDUM_2026-10-19.md),
[audit](docs/SCIENTIFIC_AUDIT_2026-10-19.md), D68):

- **Whether the readout adds value beyond cheaper signals is unanswered.** The
  registered comparison found no significant difference against semantic
  entropy (−0.092, 95% CI [−0.184, 0.016], q = 0.115) or the probe (−0.008,
  q = 1.0). The audit found the comparison was made at token budgets (0.61 and
  0.72) where even a perfect rule must stop unsafely: an oracle can save only
  0.17 safely. In that regime a rule using **no signal at all** beats every
  compared rule (0.373 against semantic entropy's 0.325 and the readout's
  0.233). The result therefore says nothing about incremental value in either
  direction. Semantic entropy's apparent edge comes from abstaining, not from
  information.
  **Re-scored on the registered (forced-answer) outcome**
  ([D69](docs/REANALYSIS_2026-10-19.md); corrected, post hoc), every rule
  was given the same stopping positions and budgets lay below the oracle
  bound:
  - the readout's score **beats a no-signal rule**: +0.052 [0.022, 0.082];
  - **against the probe it is unresolved**: +0.027 [−0.026, 0.076]. This
    reading does not change with the verbaliser seed, and the gap shrinks
    to +0.006 against a layer-24 probe;
  - **semantic entropy cannot be compared**: it has no operating point
    where safe stopping is possible.

  So the readout carries stopping-relevant information. Its value beyond the
  probe is not established.
- **Tails are common and long, and they are not post-answer text.** They are
  found non-trivially on 0.948 of 250 problems and cover a mean 0.378 of the
  chunks, a floor that depends on the 48-token forcing budget. They usually
  begin before the answer is written, and the forced completion often finishes
  the arithmetic itself. The tail marks where the answer becomes *recoverable
  within that budget*, not redundant computation.
- **Most claims carry reconstruction weight, and some of their content checks
  out.** 0.749 of 6,007 claims are reconstruction-dependent and not
  reproduced from noise. This survives clustering by problem and the removal
  of invalid paraphrase-null draws (0.73–0.78). Checked objectively against
  the trace (D71):
  - the stated next word is right on 52% of explanations, against 3–10% under
    the controls;
  - "quoted" context is almost never verbatim;
  - the named final token's word is rarely right.

  Interpretive content, faithfulness to the computation and mechanism remain
  untested.
- **RQ3, the causal question, was not tested on this model.** Its registered
  readout, a count of rechecking phrases, occurs in 2 of 274 traces. This
  instruction-tuned model barely verifies in its prose at all. The
  exploratory answer-change dose–response turned out to be truncation: every
  dose-2.0 change was a continuation cut off at 200 tokens (D74). A corrected
  pilot is implemented and staged
  ([`docs/RQ3_FOLLOWUP_DESIGN.md`](docs/RQ3_FOLLOWUP_DESIGN.md)); it has not
  been run.

**Status: complete (2026-10-19).** The report is frozen at `32b5539`, and the
run artifacts are published as a GitHub release. The Zenodo deposit is
prepared but not made, so there is no DOI yet.

| Looking for | Where |
|---|---|
| The report (Abstract to Limitations, frozen) | [`paper/REPORT.md`](paper/REPORT.md), assembled from the sections in [`paper/`](paper/README.md) |
| Corrections to how the report reads (post hoc, D68) | [`paper/ADDENDUM_2026-10-19.md`](paper/ADDENDUM_2026-10-19.md); evidence in [`docs/SCIENTIFIC_AUDIT_2026-10-19.md`](docs/SCIENTIFIC_AUDIT_2026-10-19.md) |
| The research questions, pre-registered definitions and falsification battery | [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §1, §5 and §6 |
| Every deviation and decision, with reasons (D1–D67) | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Open and closed work items, and the release's verification evidence | [`docs/AUDIT_2026-10-17.md`](docs/AUDIT_2026-10-17.md), a living register opened on that date |
| Stage-by-stage manual | [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| State of the run, and the pilot's detail | [`docs/STATUS.md`](docs/STATUS.md) |
| The released run artifacts | [Released run artifacts](#released-run-artifacts) below |
| How to cite | [`CITATION.cff`](CITATION.cff) |

The review submitted at the proposal stage (`Project_Review_II.md`) is the
original scientific specification. It is a local document and is not part of
this repository. Its research questions are restated in `PROJECT_PLAN.md`, and
its bibliography is in [`paper/references.md`](paper/references.md).

---

## Main-run results at a glance

250 problems (150 GSM8K, 100 MATH), 250 traces with no failures, final-answer
accuracy 0.88. Every number below is read from `runs/main/analysis/results.json`
and `runs/main/report/RESULTS.md`. `scripts/check_run_consistency.py --run
runs/main` reports 179 checks, 0 mismatched, 0 warnings. The report's §6 gives
the full results with their sources.

| | |
|---|---|
| **RQ1 — the tail** | Detected on 240 of 250 (non-trivial on 0.948); mean tail fraction 0.378. Tail fraction is higher on correct final answers (Cliff's δ +0.295, q = 0.036). *Audit:* not separable from trace length (p = 0.68 with log tokens as covariate). |
| **Primary question (O3)** | 120 problems in the NLA arm (`--set nla.max_problems=120`, config hash `0bb33cf4c468a3f6`). Against the probe: 11 discordant problems, 6 favour the probe, q = 1.0. Against semantic entropy: 37 discordant, 24 favour semantic entropy, q = 0.115. Against the agreement rule: no shared budget, so not compared. *Audit:* compared at budgets (0.613, 0.722) above the oracle's 0.174, where an information-free rule scores 0.373 and 0.261, above every compared rule; safety was scored on the *parsed* answer, not the registered *forced* one. |
| **RQ2 — claim faithfulness** | 6,007 claims: 0.750 reconstruction-dependent, 0.002 reproduced from noise. *Audit:* survives clustering by problem, [0.734, 0.766]. This is reconstruction dependence, not faithfulness. |
| **F10 — position** | Tail windows reconstruct slightly *worse* than position-matched ones (cosine 0.873 vs 0.878, Cliff's δ −0.203). *Audit:* the registered tests are sample-level. Paired by problem, p = 0.012 against matched-position and 0.053 against matched-length. |
| **RQ3 — causal** | **Not tested on this model.** The edits were applied (59 of 60 problems; the 60th generated nothing to edit), but the registered rechecking-marker readout is zero in every arm at the compared dose, so no contrast could be made. This is not evidence against a tail direction (D62). |
| **Probe baseline** | Beats an activation-free positional predictor by only 0.018 AUC; it must not be read as a correctness readout. |
| **F8 — quantisation** | Partly replicated (D55): on 24 problems, bf16 target activations match NF4 at cosine 0.990 and leave the NLA readout's score unchanged within noise (−0.0014 [−0.0033, 0.0005]); the verbaliser, reconstructor and behavioural arms stayed NF4, so this is not the full registered replication. |

The NLA and causal arms contain no MATH counting-and-probability problem
(report §8.11), so their results describe GSM8K and MATH algebra only.

**Analyses outside the registered result** are labelled exploratory, partial
or deferred wherever they appear (report §6.8; `docs/DECISIONS.md` D51–D60):

| Analysis | Status | Outcome |
|---|---|---|
| NLA arm at 240 problems | **exploratory, post hoc** (D54) | The direction against semantic entropy holds, and its interval excludes zero. N was chosen after the first result was seen, so this is not confirmatory. *Audit:* semantic entropy's safe rate there is 74 abstentions plus 4 safe stops out of 166, so the direction reflects where the rules stop, not what they know. |
| Claim–evidence audit | **post hoc** (D68) | [`docs/SCIENTIFIC_AUDIT_2026-10-19.md`](docs/SCIENTIFIC_AUDIT_2026-10-19.md); regenerate with `scripts/audit_claims_posthoc.py --config main` (CPU, roughly 20–30 min). |
| O3 on the registered outcome | **corrected, post hoc** (D69; plan hashed before the data) | Readout − no signal +0.052 [0.022, 0.082]; readout − probe +0.027 [−0.026, 0.076]; semantic entropy not comparable. Overall *unresolved*. [`docs/REANALYSIS_2026-10-19.md`](docs/REANALYSIS_2026-10-19.md) |
| Coverage (seed, layer, subject) | **exploratory** (D73) | Readout beats chance under every variant; its edge over the probe is not robust. [`docs/COVERAGE_PLAN.md`](docs/COVERAGE_PLAN.md) |
| RQ2 paraphrase null, claim checks, F10/RQ2 units | **post hoc** (D70–D72) | 29% identity draws; the conclusion holds (0.73–0.78). The next word is predicted on 52% of explanations. With corrected units RQ1 is not robustly significant. |
| RQ2 independent-probe control (amendment A1) | **exploratory** (D56–D57) | Did not corroborate the reconstruction-dependence reading. Its labels are rule-derived and were not validated by people. |
| F8 bf16 replication | **partial** (D55) | Covered above. |
| F6 control under a length-correct mapping | **exploratory diagnostic** (D58–D60) | Shuffled AUC 0.841 against 0.602 under `np.resize` (+0.240). F6's registered no-leak result is unchanged, but it cannot carry a stronger claim ([`docs/F6_LENGTH_CORRECT_RESULTS.md`](docs/F6_LENGTH_CORRECT_RESULTS.md)). |
| Supervised analyses on cheap features | **supplementary** (D51) | Held-out R² 0.333 [0.19, 0.44] for tail fraction; AUROC 0.818 [0.73, 0.90] for correctness ([`EXECUTION_REPORT_SUPERVISED.md`](EXECUTION_REPORT_SUPERVISED.md)). |
| F5 temperature and chunker sweeps | **deferred** (D52) | Implemented (`scripts/run_f5_sweep.py`) but not run; an estimated 60–150 GPU-hours. |

---

## What this investigates

Language models often keep generating after a problem is functionally solved.
This project calls that region the **Answer-Stable Tail (AST)** and asks what,
if anything, is going on inside the model while it happens.

The tempting answer is "it is verifying its work". The research question is
whether that claim can be *supported*:
- by a readout that is faithful at the level of individual claims;
- that beats cheaper signals at the practical task of deciding when to stop;
- and that survives a causal test with proper controls.

A plausible outcome is that the cheap signals do just as well. The protocol was
built to report that with the same weight as a positive result. The frozen
report read its result that way. The post-hoc audit (D68) found the comparison
could not distinguish any signal from chance at the budgets it reached, so the
question remains open rather than answered negatively.

### Three distinctions the code enforces

These come from §2.1.10 of the review and are structural, not editorial:

| | |
|---|---|
| **Answer stability ≠ causal redundancy** | AST detection never reads an activation. Stage `ast` has no dependency on stage `acts`; the independence is an absent edge in the DAG, not a comment. |
| **Decodability ≠ causal use** | The hidden-state probe and the intervention experiments are separate stages reported in separate tables. A probe result is never described as mechanistic. |
| **Reconstruction fidelity ≠ claim-level faithfulness** | Computed by different modules (`nla/reconstructor.py` vs `faithfulness/audit.py`) and reported in different tables. High fidelity with unfaithful claims is the documented failure mode, not an edge case. |

### On RECAP

**RECAP** (Dingeto, 2026) is a *training* intervention: it co-trains the target
model with linear decodability heads. Reproducing it means retraining
Qwen2.5-7B, which is out of scope.

This project runs a **RECAP-inspired** evaluation. It adopts the principle of
independent verification (independent probes, verbaliser-only controls) against
released checkpoints. It does not reproduce the RECAP procedure, and nothing
here is labelled as though it did.

### Method as implemented

| Objective | What the code does |
|---|---|
| **Tail definition** (O1) | Chunk-level answer equivalence under truncation, plus K independently resampled continuations checked against an external verifier, plus persistence. Matched-position and matched-length controls. |
| **Readout** (O2) | The released Qwen2.5-7B-L20 autoencoder, run through its own `nla_meta.yaml` conventions, at tail and control windows. |
| **Comparison** (O3) | Against answer-convergence stopping, a hidden-state correctness probe, and semantic entropy — on the safety/saving curve **at matched token budget**. |
| **Faithfulness** (O4) | Claim-level delete / resample / paraphrase, with the paraphrase arm as the null, plus an independent-probe control and a verbaliser-only control. |
| **Causal** (O5) | Truncation, filler, direction ablation with dose–response, against matched-random-direction and matched-position controls. |
| **Falsification** | Ten pre-registered tests (F1–F10), each designed to show the interpretation is wrong. |

Two supplementary supervised analyses measure what cheap trace features
already predict, because a readout can only be credited with *incremental*
value. One regresses `tail_fraction`, the other classifies `final_correct`.
Each compares an interpretable linear model with gradient boosting, and reports
the stronger one only if a paired bootstrap on held-out error excludes zero.
Methodology: [`docs/SUPERVISED_ANALYSES.md`](docs/SUPERVISED_ANALYSES.md).

---

## Released run artifacts

The stored outputs of the runs are published as the GitHub release
[**`data-runs-2026-10-19`**](https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail/releases/tag/data-runs-2026-10-19).
The release page carries the release notes.

| | |
|---|---|
| Archive | `nlaast-runs-2026-10-19.tar.gz`, 145,402,631 bytes |
| SHA-256 | `b50c9320d4aefcfdc3e7c1989ff40ea013fdb3bbf86aa12a0f309e978b10b2ff` |
| Source commit | `4713607`, which records the freeze; its `paper/` is identical to the frozen `32b5539` |
| Licence | CC-BY-4.0 for the data (the code is MIT); third-party notices inside the archive |

The archive holds the following runs, each classified in its `RUNS.json`:
- **Registered:** `main` and `pilot`.
- **Registered test, partial:** F8's two arms and their comparison.
- **Exploratory:** the N = 240 NLA arm, RQ2 amendment A1 and the F6
  diagnostics. A1's material includes an AI-produced annotation sheet named as
  not independent.

Local path roots in 41 files were replaced in the release copy only.
`TRANSFORMS.json` records each one with its original hash; every other file is
byte-identical to the run. To verify a download:

```bash
sha256sum -c nlaast-runs-2026-10-19.tar.gz.sha256
tar -xzf nlaast-runs-2026-10-19.tar.gz
cd nlaast-runs && sha256sum -c SHA256SUMS.txt
```

How the release was built and checked is described in
[`release/PUBLISHING.md`](release/PUBLISHING.md) and `docs/DECISIONS.md`
D63–D67. The Zenodo deposit is prepared but not yet made, so no DOI exists.

---

## Getting started

### Hardware and what can be reproduced

The study ran on an RTX 4050 Laptop GPU with **6.44 GB of VRAM** (6,141 MiB),
**16.8 GB (15.65 GiB) of RAM** and about 22 GB of spare disk. It needs three
Qwen-7B-shaped checkpoints, totalling **41 GB in bf16**, converted to 4-bit NF4
one at a time.

- **Without a GPU or the checkpoints:**
  - the test suite runs;
  - the read-only checks below run against any run directory you have;
  - the report can be assembled.
- **With them:** the pipeline runs, but the `main` configuration takes many
  GPU-hours on this class of hardware.

`smoke` and `tiny` check the wiring only. They reproduce no result.

### Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

The script installs the CUDA build of torch from the PyTorch index (it is not
on PyPI), then the project from `pyproject.toml`. It then verifies torch, CUDA,
bitsandbytes and `math_verify`. `requirements.lock.txt` pins the 87 packages
this study actually ran on.

Point the two caches at volumes with room. They default to sitting beside the
repository, which on most machines is not where the space is:

```powershell
$env:HF_HOME = "<volume>/hf_cache"      # bf16 download staging, ~22 GB free
$env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints, ~15 GB
```

### Configurations

Overlays in [`configs/`](configs/) are deep-merged over `base.yaml`, and
`--set key=value` overrides are merged last. The resolved configuration is
hashed into every run's manifest.

| Config | Role |
|---|---|
| `base` | Defaults for every run; mirrors `src/nlaast/config.py` |
| `main` | The main experiment: 150 GSM8K + 100 MATH. It produced the reported results. |
| `pilot` | 24 problems (14 GSM8K, 10 MATH): small enough to surface faults, large enough to read |
| `smoke` | 4 problems, minimal budgets; proves the pipeline runs end to end, and nothing scientific |
| `tiny` | Not an experiment: randomly initialised 2-layer stand-ins built by `tests/tiny_fixtures.py`, driven by `tests/test_end_to_end_tiny.py` |
| `mlcorpus` | A larger trace corpus for the supervised analyses, with generation and AST settings identical to `pilot` |

### Running the pipeline

```powershell
.venv/Scripts/python.exe scripts/run_pipeline.py --config smoke              # 4 problems, wiring only
.venv/Scripts/python.exe scripts/run_pipeline.py --config pilot              # 24 problems
.venv/Scripts/python.exe scripts/run_pipeline.py --config main --dry-run     # show the stage plan
.venv/Scripts/python.exe scripts/run_pipeline.py --config main --from traces # resume from a stage
```

`run_pipeline.py` holds the stage order, and `docs/PIPELINE.md` documents each
stage. Every stage is resumable and checkpoints per problem. Results land in
`runs/<run_id>/`, which is not tracked by git. Each run directory holds:
- the resolved config and its hash;
- the git commit;
- model revisions and activation digests;
- per-stage metrics, in `manifest.json`.

**On this hardware the three checkpoints cannot coexist.** Falsification test
F8 compares 4-bit activations with bf16 ones, so it has to run while the bf16
target still exists. The orchestrator orders the stages for that, and
`docs/STATUS.md` §6 gives the sequence.

### Tests

```powershell
.venv/Scripts/python.exe -m pytest                 # whole suite; GPU tests skip without CUDA
.venv/Scripts/python.exe -m pytest -m "not slow"   # CPU only: deselects the 5 GPU tests
.venv/Scripts/python.exe -m pytest -m slow         # only the 4-bit GPU paths
```

On 2026-10-19 the whole suite gave 971 passed, 3 skipped on the RTX 4050,
including the 5 GPU tests. The 3 skips need the released autoencoder sidecars
in the local Hugging Face cache.

The end-to-end test runs every stage script against tiny stand-ins in about a
minute on CPU; it proves wiring, not science. It deletes and rebuilds
`runs/tiny/`, a synthetic run the supervised analyses refuse to pool; no other
test writes under `runs/`.

### Checking the recorded results (read-only)

None of these writes to `runs/`. Each re-derives a check from the stored
artifacts, so a changed or missing file shows up as a failure.

```powershell
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/main          # 179 checks, 0 warnings
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/pilot         # 173 checks, 1 historical warning
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/main_nla240   # 132 (exploratory, D54)
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/main_f8_bf16  # 133 (partial F8, D55)
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/main_f8_nf4   # 132
.venv/Scripts/python.exe scripts/measure_rq3_readout_floor.py                      # RQ3's readout floor: 2 of 274 traces
```

```bash
# the F6 length-correct diagnostic's outputs (D60), from the repository root
sha256sum -c runs/main_f6_diag_validation/SHA256SUMS.txt
# F6's registered artifact: 45701a51c98d5ebd936ed1fc6a463f7fb01f1e2743378094ee7bd6cac640d5ed
sha256sum runs/main/robustness/robustness.json
# the N = 240 extension's inputs, copied from main (D54): run inside runs/main
(cd runs/main && sha256sum -c ../main_nla240/SOURCE_SHA256SUMS.txt)
```

The expected counts are recorded in `docs/AUDIT_2026-10-17.md`. The diagnostic
and post-hoc scripts (`audit_f6_resize.py`, `run_f8_bf16_headline.py --phase
prepare`, `diag_f6_length_correct.py`, `audit_claims_posthoc.py`, and the
D69–D74 scripts listed in `docs/REANALYSIS_2026-10-19.md`) refuse to
overwrite an existing output directory. The last one regenerates every number
in the D68 audit, on CPU in roughly 20–30 minutes:

```powershell
.venv/Scripts/python.exe scripts/audit_claims_posthoc.py --config main   # writes runs/main_claims_audit/
```

### The report

```powershell
.venv/Scripts/python.exe scripts/assemble_paper.py --check   # state only; exits 0 when complete
```

`paper/REPORT.md` is generated from the section files. The report is frozen
(D66): its sections are not edited without a new, dated decision.

---

## Layout

```
CITATION.cff          citation metadata for this repository
PROJECT_PLAN.md       the plan and pre-registration (§5–§6 are the registered definitions)
EXECUTION_REPORT_SUPERVISED.md   generated report of the supervised analyses on the main run
classification.py,    root-level supervised analyses by repeated cross-validation;
regression.py,          they write to outputs/ (common_data.py selects which runs pool)
common_data.py
configs/              base.yaml and the main / pilot / smoke / tiny / mlcorpus overlays
docs/                 decision record, audit register, pipeline manual, status, protocols
src/nlaast/           library code; no I/O at import
  data/  trace/  models/  activations/  ast_detect/  nla/
  baselines/  faithfulness/  causal/  analysis/  supervised/  viz/
scripts/              run_pipeline.py and one entry point per stage; read-only checks
                      (check_run_consistency.py, measure_rq3_readout_floor.py);
                      the post-hoc analyses; the report assembler; the release builder
                      (build_results_release.py) and its scanner (scan_release.py)
tests/                unit tests and the fixture-backed end-to-end test
paper/                the frozen report: section files, generated REPORT.md, references.md
release/              templates, notices and licence texts for the run-artifact release,
                      and PUBLISHING.md (gates and procedure)
outputs/              pilot-scale supervised cross-validation tables and figures; not
                      main-run results (see outputs/README.md)
output/               literature-review working notes from the proposal stage
figures/              figures drawn for the project review documents (make_review_figures.py)
third_party/          vendored Apache-2.0 nla_inference, with LICENSE and NOTICE
runs/<id>/            per-run config, manifest, logs and stage outputs (not tracked)
data/                 staged benchmark files (not tracked)
```

`src/nlaast/` modules: `data` loads benchmarks and verifies answers; `trace`
chunks traces with exact token alignment; `models` loads, quantises and hooks
the models; `activations` is a content-addressed store; `ast_detect` finds the
tail without reading activations; `nla` parses sidecars and runs the
verbaliser and reconstructor; `baselines` holds convergence, the probe,
semantic entropy and stopping; `faithfulness` splits and perturbs claims;
`causal` runs interventions and the Zhang & Nanda gate; `analysis` computes
bootstraps, effect sizes, BH correction and tables; `supervised` builds the
cheap-feature models; `viz` draws figures.

---

## Versioning

Three things carry version-like names here, and they are independent:

- **The Python package**, `nlaast` 0.1.0 in `pyproject.toml`. This is the code's
  version. No code release has been published, and the number does not
  describe the data.
- **Run artifacts**, released as date-stamped snapshots. The tag
  `data-runs-YYYY-MM-DD` is on the commit the release was built from, and the
  archive is named `nlaast-runs-YYYY-MM-DD.tar.gz`. A published snapshot is
  never replaced: a later one gets a new date.
- **The report**, frozen at a recorded commit (`32b5539`, D66) rather than
  versioned.

---

## Pilot

The pilot ran to completion: 24 problems (14 GSM8K, 10 MATH), 24 traces with no
failures, activations at layers 14/20/24 with **zero positional drift and zero
clamped positions**. The tail was detected, all three baselines scored, and all
three Qwen-7B-class checkpoints downloaded, converted to 4-bit NF4 and run. It
predates the main run and is reported here as history. Detail and full tables:
[`docs/STATUS.md`](docs/STATUS.md).

| | |
|---|---|
| **The answer to the primary question is no, and the corpus is too small to say it firmly.** | Two of three head-to-head comparisons at matched token budget can be made on 21 problems, and neither favours the readout. Against semantic entropy its safe-stopping rate is **0.095** against **0.286**, but that rests on **4 discordant problems of 21** — all four falling the same way — which is q = 0.1875, not significant. Against the hidden-state probe it is **exactly tied**: zero discordant problems, the same outcome preserved on the same problems. The third, against the agreement rule, is unevaluable **on this corpus**: with its graded score (D36) the rule reaches 8 distinct savings, but they fall either side of the shared budget window rather than in it (D37). |
| **But the claims it makes are reconstruction-dependent.** | 967 claims over 246 explanations: **0.759** degrade reconstruction more when deleted than a meaning-preserving paraphrase does, only **0.003** are reproduced from matched-norm noise, giving **0.756** dependent-and-not-noise. Length-adjusted the rate is 0.781, and the noise rate is flat across thresholds. |
| **Fidelity is not tail-specific.** | Tail windows reconstruct at cosine **0.870**, a matched *pre-stabilisation* window at **0.869**, a matched-length window elsewhere at 0.859. The Gaussian control at 0.391 is what shows the readout is doing anything. |
| **Removing the tail changes no answer.** | Truncating at the tail start, and replacing the tail with matched-length filler, both leave accuracy at **0.800** — identical to the unintervened baseline — on 10 of 10 eval problems. |
| **RQ3 was not tested.** | The marker readout occurred once in 200 traces (24 pilot + 176 main). The pilot's original direction arms were also no-ops, editing positions the forward pass never contained. That was fixed and the pilot's interventions re-run: RQ3 is *not tested* there too, for the same zero readout. |
| **The 4-bit deviation is small.** | Layer-20 activations from the NF4 target match the bf16 ones at cosine **0.987** (median 0.991, min 0.937). The verbaliser's integrity gate passed on 0.976 of samples against 0.786 for noise. |
| **Determinacy is not statement.** | On every problem with a tail, the tail begins *before* the trace states the answer, a mean 2.19 chunks earlier. Right for a stopping rule; wrong for reading the window as post-answer verification. |
| **The measured tail is a lower bound.** | The 24-token forcing budget cut the forced answer off on 40.7% of evaluated boundaries. Relaxing it moves the tail fraction 0.356 → 0.486, which is why the main run uses 48 tokens. |
| **The hidden-state probe is largely reading position.** | Grouped-CV AUC **0.996**, but the same model on activation-free positional features alone reaches **0.962**. |

---

## Resources

| | |
|---|---|
| **Target model** | `Qwen/Qwen2.5-7B-Instruct`, layer 20 (output of decoder block 20) |
| **Verbaliser** | `kitft/nla-qwen2.5-7b-L20-av` |
| **Reconstructor** | `kitft/nla-qwen2.5-7b-L20-ar` |
| **Benchmarks** | GSM8K, MATH (algebra, counting and probability) |
| **Verifier** | `math_verify`, with a deterministic fallback |

The autoencoder is bound to one model at one layer and does not transfer. The
pipeline cross-checks both sidecars and the live tokeniser before running, and
refuses to continue on a mismatch.

Upstream's standalone `nla_inference.py` is vendored under
[`third_party/nla_inference/`](third_party/nla_inference/) (Apache-2.0, with
`NOTICE.md`). Its injection arithmetic and reconstructor are used as shipped.
Only the generation backend is replaced, because upstream drives the verbaliser
through an SGLang server that will not run on Windows or fit in 6.44 GB of VRAM.
Quantisation to 4-bit is a deviation from the released checkpoints' precision.
It is measured rather than assumed away (F8), and the verbaliser's output is
gated on every sample for the off-distribution failure signature upstream
documents.

---

## Citing

To cite this repository and its report, use [`CITATION.cff`](CITATION.cff)
(GitHub shows it as "Cite this repository"). The run artifacts are a separate
dataset release, linked above.

The autoencoder this study evaluates:

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and et al.},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

Other key references:

- Dingeto, H. (2026). Train the Model, Not the Reader: Decodability Supervision
  for Verifiable Activation Explanations. Preprint,
  [arXiv:2607.20379](https://arxiv.org/abs/2607.20379).
- Li, M., Ceballos Arroyo, A. M., Rogers, G., Saphra, N., & Wallace, B. C.
  (2026). Do Activation Verbalization Methods Convey Privileged Information?
  *ICML 2026*, [arXiv:2509.13316](https://arxiv.org/abs/2509.13316).

The full, checked reference list is [`paper/references.md`](paper/references.md).

## License

The code is MIT, except `third_party/nla_inference/`, which is Apache-2.0 (see
its [`NOTICE.md`](third_party/nla_inference/NOTICE.md)). The released run
artifacts are CC-BY-4.0.
