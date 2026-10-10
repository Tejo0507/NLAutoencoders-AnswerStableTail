# Pipeline reference

What each stage does, what it reads, what it writes, and why it is ordered
where it is. The scientific specification is `Project_Review_II.md`; the
implementation plan is `PROJECT_PLAN.md`; this file is the operational manual.

Every stage below is written, tested, and has **run** — at pilot scale, 24
problems, including the three that need the released autoencoder.
[`STATUS.md`](STATUS.md) reports what each one produced. The test suite also
runs every stage script against tiny randomly initialised stand-ins
(`configs/tiny.yaml`, `tests/test_end_to_end_tiny.py`), which is how the
wiring is checked without 26 GB of checkpoints.

---

## Running it

```powershell
# one-time
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
$env:HF_HOME = "<volume>/hf_cache"      # bf16 staging, needs ~22 GB free
$env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints, ~15 GB

# whole study
.venv/Scripts/python.exe scripts/run_pipeline.py --config pilot
.venv/Scripts/python.exe scripts/run_pipeline.py --config main

# one stage, or a resumed run
.venv/Scripts/python.exe scripts/run_pipeline.py --config main --from nla
.venv/Scripts/python.exe scripts/generate_traces.py --config main
```

Every stage is resumable: it reads back its own JSONL output, skips completed
units of work and continues. Killing a run loses at most the item in flight.

---

## Stage DAG

```
env ──┐
data ─┼─> traces ─┬─> acts ──┬─> [robustness:F8] ─> nla ─> faithfulness ─┐
models┘           │          │                                           │
                  ├─> ast ───┴─> baselines ─────────────────────────> analysis ─> report
                  └─> causal ──────────────────────────────────────────┘
                                   robustness ──────────────────────────┘
```

Two orderings are load-bearing rather than incidental:

**`ast` does not depend on `acts`.** The Answer-Stable Tail must be defined
without reference to any activation description (limitation L1, objective O1).
Expressing that as a DAG edge that is *absent* is stronger than a comment:
the `ast` stage cannot read an activation because nothing hands it one.

**`robustness:F8` runs between `acts` and `nla`.** F8 compares 4-bit
activations against bf16 ones, and the bf16 target weights are evicted to make
room for the verbaliser. Running F8 later would cost a fresh 15 GB download.
The orchestrator inserts this automatically; `--no-f8-ordering` disables it.

---

## Stages

### `env` — `check_environment.py`

Captures hardware, library versions and disk headroom, and runs six
feasibility checks (CUDA, VRAM for a 4-bit 7B, disk for the quantised and bf16
copies, RAM, bitsandbytes, math_verify). Writes `env/environment.json`. A
failing check is recorded, not fatal — the orchestrator decides.

### `data` — `prepare_problems.py`

Stages GSM8K and the configured MATH subjects under `data/raw/`, validates that
every record yields a gold answer, and emits the problem set with its
train/eval assignment.

The split is by hash of the problem id, not by shuffling, so it is stable as
the problem set grows: a probe fitted during the pilot is still evaluated on
genuinely held-out problems in the main run.

Also reports two parser diagnostics used later by F4: how often the permissive
extractor recovers the gold answer from the gold solution (an upper bound on
answer-extraction accuracy), and how often the strict and permissive extractors
agree. **Writes** `data/problems.jsonl`, `data/validation.json`.

### `models` — `prepare_checkpoints.py`

Downloads each checkpoint, converts it to 4-bit NF4, saves it, and evicts the
bf16 source before the next one. Three Qwen-7B-shaped checkpoints total 41 GB
in bf16 against ~22 GB of staging disk, so this is the only order that fits.

Then performs the compatibility checks nothing downstream would catch: AV and
AR sidecars against each other and against the target configuration, and the
live tokeniser against the pinned injection-token neighbours. A mismatch here
produces confident nonsense rather than an error, so the stage refuses to
continue. **Writes** `models/checks.json`, `models/resolved.json`.

### `traces` — `generate_traces.py`

The expensive stage. With the target model resident it collects, per problem:
the greedy canonical trace with exact token offsets; its chunking and the
intermediate answer at each boundary; a forced answer from each truncated
prefix; K resampled continuations at each boundary; and the semantic-entropy
samples.

Everything needing the target model is gathered here rather than spread over
several stages, because loading the model costs more than any single item of
work.

Boundaries are evaluated **from the end backwards** and the scan stops at the
first failure. Criterion 3 means only a suffix of boundaries can qualify, so
earlier ones cannot start the tail regardless of what they would have shown.
This is an exact optimisation of the definition, not an approximation.
**Writes** `traces/traces.jsonl`.

### `acts` — `extract_activations.py`

One layer-K residual-stream vector per chunk boundary, at the boundary's last
generated token. Layer K is the **output of decoder block K**
(`model.model.layers[K]`), equal to HF's `hidden_states[K+1]` — the convention
the released autoencoder was trained under. An off-by-one would feed the
verbaliser a different layer with no error raised.

The orchestrator passes `--layers 14 20 24` so F7 has something to sweep.
**Writes** `acts/index.jsonl`, `acts/blocks/*.npz`, content-addressed with
SHA-256 (objective O2 requires activation-cache hashes).

### `ast` — `detect_answer_stable_tail.py`

Applies the pre-registered criteria to the evidence the `traces` stage collected. Reads
no activations. `--sweep` additionally re-runs detection under alternative K,
agreement window and unanimity settings for F5, which costs nothing because no
generation is involved.

Edge cases (`no_stable_point`, `stable_at_zero`, `unparseable_final`,
`truncated`, `too_short`) get a status code and stay in the corpus.
**Writes** `ast/ast.jsonl`, `ast/summary.json`, `ast/sweep_F5.json`.

### `baselines` — `score_baselines.py`

Turns each comparison signal into a per-boundary score and sweeps a threshold
to produce a safety/saving curve. Convergence and semantic entropy are computed
from the `traces` stage's output; the probe is fitted here on the train split with folds
grouped by problem.

Semantic entropy is constant across boundaries within a problem. That is a real
property of the signal, not a shortcut: it says how confident the answer is,
not when to stop. **Writes** `baselines/curves.json`,
`baselines/boundary_scores.json`, `baselines/probe.json`,
`baselines/semantic_entropy.jsonl`.

### `nla` — `run_autoencoder.py`

Two passes, because the verbaliser and the reconstructor are each 7B-class and
only one fits in 6.44 GB:

1. **Verbalise** — AV resident. Descriptions for every selected window, plus
   the verbaliser-only control (a Gaussian vector at matched L2 norm). Every
   sample is gated for the documented off-distribution failure (CJK output
   instead of an English `<explanation>`) and the failure rate is reported.
2. **Reconstruct** — AR resident. Every description mapped back to a vector and
   scored against the original, in the released convention: both normalised to
   `mse_scale = sqrt(3584)`, `MSE = 2(1 − cos)`.

`--phase verbalise` / `--phase reconstruct` run one at a time.
**Writes** `nla/verbalisations.jsonl`, `nla/reconstructions.jsonl`,
`nla/summary.json`.

### `faithfulness` — `audit_faithfulness.py`

Claim-level audit with the AR resident. Delete, paraphrase (the null),
resample, independent-probe control, verbaliser-only control. A claim is
reconstruction-dependent only if its deletion effect exceeds the 95th
percentile of the paraphrase null for the same explanation.

RECAP-**inspired**, not RECAP: it adopts the principle of independent
verification but does not co-train the target model, which is what RECAP
requires and what cannot be done retrospectively to a released checkpoint.
**Writes** `faithfulness/audits.jsonl`, `faithfulness/summary.json`.

### `causal` — `run_interventions.py`

Five arms: `truncate`, `filler`, `ablate_dir` (swept over a dose coefficient),
`random_dir`, `matched_position`. The candidate direction is a difference in
means between tail and matched-position activations, fitted on the **train
split only**; interventions run on the eval split.

`direction_claim_supported` returns true only if the candidate beats both
controls with a monotone dose-response and leaves answers intact. The review
scopes this stage as an extension contingent on compute, and a partial arm set
is an expected outcome. **Writes** `causal/interventions.jsonl`,
`causal/summary.json`.

### `analysis` — `aggregate_results.py`

Aggregates everything, builds the matched-budget comparison table, and runs the
pre-registered test family with Benjamini–Hochberg correction at q = 0.05.
Tests that could not be evaluated stay in the family rather than being dropped.
**Writes** `analysis/results.json`, `analysis/tables/*.{csv,json}`.

### `robustness` — `run_falsification_tests.py`

The falsification battery (F1–F10). `--only F8_quantisation` runs the one test
with an ordering constraint. **Writes** `robustness/robustness.json`.

### `report` — `write_run_report.py`

Figures and the run report, from saved outputs only — reruns in seconds without
touching a model. **Writes** `report/RESULTS.md`, `report/figures/*.png`,
`report/tables/*`.

---

## Supplementary stages — the supervised analyses

Deliberately **not** members of `stages:` in any config, and not reachable from
`run_pipeline.py`: they are side analyses over the corpus, not steps the main
experiment depends on. Each needs only `traces` and `ast`, loads no model, and
reruns in seconds on CPU.

Methodology, target justification and leakage controls:
[`SUPERVISED_ANALYSES.md`](SUPERVISED_ANALYSES.md).

### `regression_tail` — `predict_tail_fraction.py`

Regression on `tail_fraction` — how much of a trace is post-answer redundancy —
from question-level and trace-surface features only. This is falsification
tests F1 (length) and F2 (difficulty) stated as an estimate with an interval
rather than a verdict: it measures the cheap-tier floor that the verbalised
readout has to clear.

Ridge against gradient boosting, with the stronger model reported only if a
paired bootstrap on held-out MAE excludes zero; OLS alongside for coefficients
and residual diagnostics. **Writes** `regression_tail/summary.json`,
`regression_tail/*.{csv,json}`, `regression_tail/fig_*.png`.

### `classify_correct` — `predict_correctness.py`

Classification of `final_correct` from the same features. Safe stopping is
defined in terms of verified correctness, and `probe.kind: logistic` is already
a correctness probe on layer-20 activations — so this stage fits the same label
with the same model family on no activations at all, and its AUROC is the floor
that probe has to clear.

Logistic regression against gradient boosting under the same justification
rule, with confusion and per-stratum error analysis at a threshold chosen on
the training split. **Writes** `classify_correct/summary.json`,
`classify_correct/*.{csv,json}`, `classify_correct/fig_*.png`.

### `supervised_report` — `write_supervised_report.py`

Assembles both into one markdown report. Recomputes nothing — every number
comes from the two `summary.json` files — and draws its verdicts from
thresholds fixed in the script, so a result the author would have liked and one
they would not are read the same way. **Writes** the path given by `--out`,
`EXECUTION_REPORT_SUPERVISED.md` at the repository root by default. That file
is not in the repository: these two stages refuse below 30 train / 20 eval
rows, and the pilot has 24 problems, so they have nothing to report (see
[`STATUS.md`](STATUS.md) §5). The root-level `regression.py` and
`classification.py` do run at this size — `outputs/` holds their tables with a
`provenance.json` naming the corpus.

```powershell
.venv\Scripts\python.exe scripts\predict_tail_fraction.py    --config mlcorpus
.venv\Scripts\python.exe scripts\predict_correctness.py      --config mlcorpus
.venv\Scripts\python.exe scripts\write_supervised_report.py  --config mlcorpus
```

---

## Run directory

```
runs/<run_id>/
  manifest.json            environment, git commit, per-stage status, metrics,
                           model revisions and digests, config hash history
  config.resolved.yaml     the exact settings used
  run.log
  <stage>/                 that stage's artefacts
```

A re-run with the same `run_id` resumes; it does not overwrite. Changing the
config under an existing `run_id` records the previous hash in
`config_hash_history` rather than silently replacing it.

---

## Configuration

`configs/base.yaml` holds the defaults, restated from `src/nlaast/config.py` so
the whole experiment is readable in one file. A test asserts the two cannot
drift. Overlays (`pilot`, `main`, `smoke`, `mlcorpus`) are deep-merged over the base, and
`--set key=value` is merged last:

```powershell
scripts/run_pipeline.py --config main --set nla.n_samples=4 --set data.n_math=150
```

Unknown keys are rejected rather than ignored — silently dropping a typo'd
setting is how a run ends up not being the run you thought you made.
