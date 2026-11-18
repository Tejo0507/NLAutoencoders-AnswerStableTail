# NLA Answer-Stable Tail

[![cpu-tests](https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail/actions/workflows/cpu-tests.yml/badge.svg?branch=main)](https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail/actions/workflows/cpu-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for a pre-registered study of the **Answer-Stable Tail (AST)**: the part
of a reasoning trace that a model keeps writing after its answer has settled.
The study asks whether a **natural-language autoencoder (NLA)** readout of the
model's activations tells you where that tail begins, and so when it is safe to
stop early, better than cheaper signals do.

An NLA is a released pair of models. The verbaliser turns an activation into
text, and the reconstructor turns that text back into an activation. This
repository runs the NLA for Qwen2.5-7B-Instruct at layer 20 and compares its
readout with answer convergence, a hidden-state correctness probe, and
semantic entropy, at matched token budget.

## Result

The readout does not beat cheaper signals in any way that holds up.

- On 400 fresh GSM8K and MATH problems, the readout beat the correctness probe
  by 0.047 (95% CI 0.019 to 0.075) in safe-stopping rate, on the registered
  48-token forced-answer outcome.
- A rule that uses only the prefix length beat the readout by 0.025.
- Under a 256-token outcome, neither registered hypothesis was supported.
- Beyond position and a same-label probe on the same activation, the
  explanation text adds 0.00002 AUC.
- An exploratory stopping-policy pilot saved 5.5% of tokens with 4.1% unsafe
  stops. Its 95% upper bound, 7.8%, is above the 5% target, so the policy
  failed its safety criterion.

The study does not test whether the explanations are faithful to the model's
computation.

## Requirements

- Python 3.11 or later, and Windows PowerShell for the setup script.
- For the full pipeline: a CUDA GPU and about 41 GB of disk for the three
  checkpoints in bf16. The study ran on an RTX 4050 Laptop GPU with 6.44 GB of
  VRAM and 16.8 GB of RAM, converting each checkpoint to 4-bit NF4 in turn.
- Without a GPU, you can still run the CPU test suite and the read-only checks.

## Set up

1. Create the environment:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
   ```

   The script installs the CUDA build of torch from the PyTorch index, then
   this package from `pyproject.toml`. It then checks torch, CUDA,
   bitsandbytes, and `math_verify`. `requirements.lock.txt` pins the exact
   versions the study ran on.

2. Point the two caches at a volume with free space:

   ```powershell
   $env:HF_HOME = "<volume>/hf_cache"      # bf16 download staging, about 22 GB
   $env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints, about 15 GB
   ```

## Run the pipeline

```powershell
.venv/Scripts/python.exe scripts/run_pipeline.py --config smoke              # 4 problems, wiring only
.venv/Scripts/python.exe scripts/run_pipeline.py --config pilot              # 24 problems
.venv/Scripts/python.exe scripts/run_pipeline.py --config main --dry-run     # print the stage plan
.venv/Scripts/python.exe scripts/run_pipeline.py --config main --from traces # resume from a stage
```

`run_pipeline.py` holds the stage order. Every stage checkpoints per problem,
so you can stop and resume a run. Output goes to `runs/<run_id>/`, which git
does not track. Each run directory records the resolved config and its hash,
the git commit, the model revisions, the activation digests, and per-stage
metrics in `manifest.json`.

The three checkpoints do not fit on the study's hardware at the same time.
Falsification test F8 compares 4-bit activations with bf16 ones, so it must run
while the bf16 target is still on disk. The pipeline orders the stages for
this.

### Configurations

Each file in [`configs/`](configs/) is deep-merged over `base.yaml`, and
`--set key=value` overrides are merged last. The resolved config is hashed
into the run's manifest.

| Config | Use |
|---|---|
| `base` | Defaults for every run. Mirrors `src/nlaast/config.py`. |
| `main` | The main experiment: 150 GSM8K and 100 MATH problems. |
| `replication` | The confirmatory replication on 400 fresh problems. |
| `pilot` | 24 problems (14 GSM8K, 10 MATH). |
| `smoke` | 4 problems with minimal budgets. Checks that the pipeline runs end to end. |
| `tiny` | Randomly initialised 2-layer stand-ins, used by the end-to-end test. |
| `mlcorpus` | A larger trace corpus for the supervised analyses. |

`smoke` and `tiny` check wiring only. They reproduce no result.

## Test

```powershell
.venv/Scripts/python.exe -m pytest -m "not slow"   # CPU only
.venv/Scripts/python.exe -m pytest -m slow         # the 4-bit GPU paths only
.venv/Scripts/python.exe -m pytest                 # everything; GPU tests skip without CUDA
```

Some tests skip when the staged benchmarks, the cached NLA sidecars, or the
stored runs are missing. Each skip names what it needs.

CI ([`.github/workflows/cpu-tests.yml`](.github/workflows/cpu-tests.yml)) runs
the CPU suite and `ruff check .` on every pull request and every push to
`main`. The end-to-end test runs every stage against tiny stand-ins in about a
minute on CPU. It rebuilds `runs/tiny/`; no other test writes under `runs/`.

## Check a stored run

These scripts read a run directory and write nothing to `runs/`. A changed or
missing file shows up as a failed check.

```powershell
.venv/Scripts/python.exe scripts/check_run_consistency.py --run runs/main   # 179 checks on the main run
.venv/Scripts/python.exe scripts/measure_rq3_readout_floor.py              # RQ3's readout floor
```

The post-hoc audit regenerates its numbers from the stored main run, on CPU,
in about 20 to 30 minutes. It refuses to overwrite an existing output
directory.

```powershell
.venv/Scripts/python.exe scripts/audit_claims_posthoc.py --config main     # writes runs/main_claims_audit/
```

## Run artifacts

The stored outputs of the runs are published as the GitHub release
[`data-runs-2026-10-19`](https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail/releases/tag/data-runs-2026-10-19),
under CC-BY-4.0.

| | |
|---|---|
| Archive | `nlaast-runs-2026-10-19.tar.gz`, 145,402,631 bytes |
| SHA-256 | `b50c9320d4aefcfdc3e7c1989ff40ea013fdb3bbf86aa12a0f309e978b10b2ff` |

To verify a download:

```bash
sha256sum -c nlaast-runs-2026-10-19.tar.gz.sha256
tar -xzf nlaast-runs-2026-10-19.tar.gz
cd nlaast-runs && sha256sum -c SHA256SUMS.txt
```

[`release/PUBLISHING.md`](release/PUBLISHING.md) describes how the release was
built and checked. `scripts/build_results_release.py` builds it, and
`scripts/scan_release.py` scans it before publication.

## Models and data

| | |
|---|---|
| Target model | `Qwen/Qwen2.5-7B-Instruct`, layer 20 (output of decoder block 20) |
| Verbaliser | `kitft/nla-qwen2.5-7b-L20-av` |
| Reconstructor | `kitft/nla-qwen2.5-7b-L20-ar` |
| Benchmarks | GSM8K; MATH (algebra, counting and probability) |
| Answer checking | `math_verify`, with a deterministic fallback |

The NLA is bound to one model at one layer. Before a run, the pipeline checks
both sidecars against the live tokeniser and stops on a mismatch.

Running the models in 4-bit NF4 is a deviation from the released precision.
Test F8 measures its effect, and the pipeline screens every verbaliser output
for the failure signature that upstream documents.

## Layout

```
src/nlaast/        the library; no I/O at import
  data/            loads benchmarks and checks answers
  trace/           chunks traces with exact token alignment
  models/          loads, quantises and hooks the models
  activations/     content-addressed activation store
  ast_detect/      finds the tail without reading activations
  nla/             parses sidecars; runs the verbaliser and reconstructor
  baselines/       answer convergence, the probe, semantic entropy, stopping rules
  faithfulness/    splits and perturbs claims
  causal/          interventions and their controls
  analysis/        bootstraps, effect sizes, BH correction, tables
  supervised/      cheap-feature models
  viz/             figures
scripts/           run_pipeline.py, one entry point per stage, read-only checks,
                   post-hoc analyses, and the release builder
tests/             unit tests and the fixture-backed end-to-end test
configs/           base.yaml and its overlays
third_party/       vendored nla_inference (Apache-2.0)
release/           notices, licence texts and procedure for the artifact release
CITATION.cff       citation metadata
```

## Upstream code

[`third_party/nla_inference/`](third_party/nla_inference/) vendors upstream's
`nla_inference.py` from
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders).
Its injection arithmetic and reconstructor are used as shipped. This repository
replaces only the generation backend, because upstream's SGLang server does not
run on Windows or fit in 6.44 GB of VRAM.

## Licence

The code is MIT ([`LICENSE`](LICENSE)). `third_party/nla_inference/` is
Apache-2.0, with its own [`LICENSE`](third_party/nla_inference/LICENSE) and
[`NOTICE.md`](third_party/nla_inference/NOTICE.md). The released run artifacts
are CC-BY-4.0. To cite this repository, use [`CITATION.cff`](CITATION.cff).
