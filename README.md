# NLA-AnswerStableTail

**Natural Language Autoencoders for Redundancy Diagnostics**

*Does a verbalised readout of residual-stream activations tell us anything
useful about the part of a reasoning trace that comes after the answer has
already settled — beyond what much cheaper signals already say?*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20development-orange)]()

> **Under development.** The pipeline is written and tested end to end, but the
> experiment has not been run to completion and **this repository contains no
> scientific result.** Section [Current state](#current-state) says exactly
> what has and has not run. Nothing in `runs/` is committed, because a partial
> run's tables would read like findings.

---

## What this investigates

Language models often keep generating after a problem is functionally solved.
This project calls that region the **Answer-Stable Tail (AST)** and asks what,
if anything, is going on inside the model while it happens.

The tempting answer is "it is verifying its work". The research question is
whether that claim can be *supported* — by a readout that is faithful at the
level of individual claims, that beats cheaper signals at the practical task of
deciding when to stop, and that survives a causal test with proper controls.

A plausible outcome is that the cheap signals do just as well. The protocol is
built to report that with the same weight as a positive result.

| | |
|---|---|
| Scientific specification | [`Project_Review_II.md`](Project_Review_II.md) (Introduction and Literature Review) |
| Implementation plan | [`PROJECT_PLAN.md`](PROJECT_PLAN.md) |
| Stage-by-stage manual | [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| Every deviation, with reasons | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Current state and next steps | [`docs/STATUS.md`](docs/STATUS.md) |

---

## Three distinctions the code enforces

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

This project runs a **RECAP-inspired** evaluation — it adopts the principle of
independent verification (independent probes, verbaliser-only controls) against
released checkpoints. It does not reproduce the RECAP procedure, and nothing
here is labelled as though it did.

---

## Method as implemented

| Objective | What the code does |
|---|---|
| **Tail definition** (O1) | Chunk-level answer equivalence under truncation, plus K independently resampled continuations checked against an external verifier, plus persistence. Matched-position and matched-length controls. |
| **Readout** (O2) | The released Qwen2.5-7B-L20 autoencoder, run through its own `nla_meta.yaml` conventions, at tail and control windows. |
| **Comparison** (O3) | Against answer-convergence stopping, a hidden-state correctness probe, and semantic entropy — on the safety/saving curve **at matched token budget**. |
| **Faithfulness** (O4) | Claim-level delete / resample / paraphrase, with the paraphrase arm as the null, plus an independent-probe control and a verbaliser-only control. |
| **Causal** (O5) | Truncation, filler, direction ablation with dose–response, against matched-random-direction and matched-position controls. |
| **Falsification** | Ten pre-registered tests that each try to show the interpretation is wrong. |

### Two supplementary supervised analyses

Before an expensive readout can be credited with telling us something, the
cheap signals have to be measured — otherwise "incremental value" has no
denominator. Two side analyses do that over the trace corpus, on CPU, reading
no activations:

| | Target | Why it is the right target |
|---|---|---|
| **Regression** | `tail_fraction` — the share of a trace that is post-answer redundancy | the study's own construct (§5 of the plan), and the quantity `tokens_saved` is denominated in |
| **Classification** | `final_correct` — verified answer correctness | safe stopping is *defined* in terms of it, and it is already the target of the layer-20 probe |

Each compares an interpretable linear model against gradient boosting, with the
stronger model reported **only** if a paired bootstrap on held-out error
excludes zero. Both are falsification tests F1 (length) and F2 (difficulty)
restated as estimates with intervals, and both are constructed so a strong
cheap-tier result is bad news for this project's central claim.

Methodology: [`docs/SUPERVISED_ANALYSES.md`](docs/SUPERVISED_ANALYSES.md).

---

## Current state

**Complete and verified**

- All 13 pipeline stages, the 3 supplementary analysis stages, the
  orchestrator, and the library under `src/nlaast/`. `python -m pytest` passes,
  including cases taken verbatim from live pilot output.
- The real NLA interface, recovered from the released checkpoints rather than
  guessed: injection token and its required neighbours, injection scale, MSE
  scale, both prompt templates, the reconstructor's architecture, and the
  layer-20 extraction convention. See [`docs/STATUS.md`](docs/STATUS.md).
- The target model, downloaded and converted to 4-bit NF4. Measured: 5.56 GB
  resident, 5.77 GB peak at batch 4, ~45 tok/s batched.
- GSM8K and MATH (algebra, counting and probability) staged and validated; the
  permissive parser recovers the gold answer from the gold solution on every
  record.

**Not done**

- **The autoencoder checkpoints have never been run.** ~26 GB still to
  download, and the bf16 target has to be evicted first to make room. This is
  the largest remaining technical risk — see
  [`docs/DECISIONS.md`](docs/DECISIONS.md) D2.
- **No stage downstream of `ast` has completed even once.** The pilot's
  trace generation was interrupted partway; several real bugs were found and
  fixed during it, so its traces are stale and it needs re-running.
- No faithfulness audit, no causal arm, no analysis, no execution report.
- The supervised analyses have been exercised on the partial pilot corpus only.
  At that size one split came out single-class and the held-out R² is negative:
  that is a sample-size artefact, not a finding, and no number from it is
  reportable.

[`docs/STATUS.md`](docs/STATUS.md) holds the detail, including the bugs the
pilot surfaced and the exact next command.

---

## Resources

| | |
|---|---|
| **Target model** | `Qwen/Qwen2.5-7B-Instruct`, layer 20 (output of decoder block 20) |
| **Verbaliser** | `kitft/nla-qwen2.5-7b-L20-av` |
| **Reconstructor** | `kitft/nla-qwen2.5-7b-L20-ar` |
| **Benchmarks** | GSM8K, MATH (algebra, counting and probability) |
| **Verifier** | `math_verify`, with a deterministic fallback |

The autoencoder is bound to one model at one layer and does not transfer; the
pipeline cross-checks both sidecars and the live tokeniser before running, and
refuses to continue on a mismatch.

Upstream's standalone `nla_inference.py` is vendored under
[`third_party/nla_inference/`](third_party/nla_inference/) (Apache-2.0, with
`NOTICE.md`). Its injection arithmetic and reconstructor are used as shipped;
only the generation backend is replaced, because upstream drives the verbaliser
through an SGLang server that will not run on Windows or fit in 6.44 GB.

---

## Hardware reality

This was built and run on an RTX 4050 Laptop with **6.44 GB of VRAM**, 15.65 GB
of RAM and ~22 GB of spare disk — against three Qwen-7B-shaped checkpoints
totalling **41 GB in bf16**.

Consequences, all documented rather than hidden:

- Each model is downloaded, converted to 4-bit NF4, saved, and its bf16 source
  evicted before the next one. Stages load exactly one model at a time.
- Quantisation is a deviation from the released checkpoints' precision. It is
  **measured**, not assumed away: falsification test **F8** compares 4-bit
  layer-20 activations against bf16 ones directly, and the verbaliser's output
  is gated on every sample for the off-distribution failure signature upstream
  documents.

---

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

The script installs the CUDA build of torch from the PyTorch index (it is not
on PyPI), then the project from `pyproject.toml`, then verifies torch, CUDA,
bitsandbytes and `math_verify`. `requirements.lock.txt` pins the 87 packages
this study actually ran on.

Point the two caches at volumes with room — they default to sitting beside the
repository, which on most machines is not where the space is:

```powershell
$env:HF_HOME = "<volume>/hf_cache"      # bf16 download staging, ~22 GB free
$env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints, ~15 GB
```

## Running what currently works

```powershell
.venv/Scripts/python.exe -m pytest                                  # no GPU needed
.venv/Scripts/python.exe scripts/run_pipeline.py --config smoke     # 4 problems
.venv/Scripts/python.exe scripts/run_pipeline.py --config pilot     # 24 problems
```

`smoke` proves the pipeline runs end to end and nothing scientific. `pilot` is
the smallest configuration meant to be read. `main` is the full run and has not
been attempted.

Everything up to and including `ast`, plus the two supervised analyses, runs
today. Stages `nla`, `faithfulness` and `causal` need the autoencoder
checkpoints and have never executed.

Every stage is resumable and checkpoints per problem; interrupting loses at
most the item in flight. Results land in `runs/<run_id>/` (untracked), with the
resolved config, its hash, the git commit, model revisions, activation digests
and per-stage metrics in `manifest.json`.

---

## Layout

```
configs/            base.yaml + pilot / main / smoke / mlcorpus overlays
src/nlaast/         library code — no I/O at import
  data/             benchmark loading, answer extraction and verification
  trace/            sentence chunking with exact token alignment
  models/           loading, NF4 conversion, activation hooks, interventions
  activations/      content-addressed activation store
  ast_detect/       the Answer-Stable Tail — no activation dependency
  nla/              sidecar parsing, verbaliser, reconstructor
  baselines/        convergence, correctness probe, semantic entropy, stopping
  faithfulness/     claim splitting, perturbation, the audit
  causal/           interventions and the Zhang & Nanda gate
  analysis/         paired bootstrap, effect sizes, BH correction, tables
  supervised/       features and evaluation for the two supervised analyses
  viz/              figures
scripts/            run_pipeline.py + one entry point per stage
tests/              unit tests, including cases taken verbatim from live output
third_party/        vendored Apache-2.0 nla_inference + LICENSE + NOTICE
runs/<id>/          config snapshot, manifest, logs, per-stage artefacts (untracked)
```

Stage scripts are named for what they do; `run_pipeline.py` holds the
dependency order and [`docs/PIPELINE.md`](docs/PIPELINE.md) documents it.

---

## Citation

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and et al.},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## References

Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html). *Transformer Circuits Thread.*

Dingeto, H. (2026). Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations. Preprint, [arXiv:2607.20379](https://arxiv.org/abs/2607.20379).

Li, M., Ceballos Arroyo, A. M., Rogers, G., Saphra, N., & Wallace, B. C. (2026). Do Activation Verbalization Methods Convey Privileged Information? *ICML 2026*, [arXiv:2509.13316](https://arxiv.org/abs/2509.13316).

The full reference list, with publication status marked for every entry, is in
the review document.

## License

MIT, except `third_party/nla_inference/` which is Apache-2.0 — see its
[`NOTICE.md`](third_party/nla_inference/NOTICE.md).
