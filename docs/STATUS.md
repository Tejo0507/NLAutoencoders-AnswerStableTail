# Current state and next steps

What exists, what has run, what it found, and the next command to type.
Last reviewed 2026-10-10.

**One-line status:** the pilot has run. All 24 configured problems produced
traces, activations were extracted at three layers with exact alignment, the
Answer-Stable Tail was detected, the baselines were scored, F8 measured the
quantisation deviation, and **the released autoencoder has been downloaded,
converted and run for the first time**. The results below are real and come
from `runs/pilot/`; the sample is 24 problems, so every interval is wide and
nothing here is a claim about language models in general.

---

## 1. What has run, with its result

| stage | status | headline |
|---|---|---|
| `env` | complete | RTX 4050, 6.44 GB VRAM, 15.65 GB RAM; two disk checks failed as expected and were worked around |
| `data` | complete | 24 problems: 14 GSM8K, 10 MATH; 11 train / 13 eval; gold recovery 1.000 |
| `models` | complete | all three checkpoints converted to NF4; AV/AR/target sidecars mutually consistent; injection site verified at token 111 |
| `traces` | complete | 24/24, no failures, 309 min; accuracy 0.792 (GSM8K 0.857, MATH 0.700); 3 traces hit the 512-token cap |
| `acts` | complete | 72 blocks, 834 vectors at layers 14/20/24; **0 positions clamped, 0 re-tokenisation drift** |
| `ast` | complete | tail on 21/24; mean tail fraction 0.356; see §2 |
| `baselines` | complete | probe grouped-CV AUC 0.996 — but see §2 for what that number is |
| `robustness:F8` | complete | **cos(nf4, bf16) = 0.987** (median 0.991, min 0.937) over 82 vectors |
| `nla` | running at time of writing | verbaliser integrity **100%** well-formed English so far |

The remaining stages (`faithfulness`, `causal`, `analysis`, `robustness`,
`report`) run from the autoencoder's output and are sequenced after it.

---

## 2. What the pilot found

Five results, in descending order of how much they change the reading of the
study. Each is computed by the stage named, and `runs/pilot/report/RESULTS.md`
carries the full tables.

### The 4-bit deviation is small (F8)

Layer-20 activations from the NF4 target agree with the bf16 ones at **cosine
0.987** on average (median 0.991, 5th percentile 0.973, minimum 0.937, norm
ratio 0.990) across 82 vectors from 8 problems, with no problem skipped for
misalignment. The verbaliser normalises its input to a fixed L2 norm, so only
direction reaches it. Together with the integrity gate passing on every sample
so far, this closes the largest threat to the NLA arm that
[`DECISIONS.md`](DECISIONS.md) D2 identified: the autoencoder is being fed
vectors it recognises.

### `tail_rate` is near its ceiling by construction

A tail was detected on 21 of 24 problems, and the **final boundary qualifies on
21 of 24** — which is why. At the last boundary the prefix is the whole trace,
so forcing an answer from it reproduces the final answer and continuations from
it restate the same answer: both criteria are close to tautological there. The
quantity that carries information is how *much* of the trace the tail covers:
mean tail fraction **0.356** (median 0.308, range 0.15–0.62), and tail tokens
are **31.7%** of all generated tokens. No tail is trivial.

The three problems without a tail are exactly the three traces that hit the
512-token cap, where the "final" answer is not final. That is the token-cap
decision in §4 showing up as a result rather than as a worry.

### The tail begins before the answer is stated — on every problem

The criteria ask whether the answer is *determined* from a prefix, not whether
the trace has **said** it. On all 21 problems with a tail, the tail begins
before the answer is stated, by a mean of **2.19 chunks** and up to 5. Forcing
elicits the answer from the prefix; the model finishes the arithmetic inside
the forced completion.

For the stopping comparison that is exactly right — a rule stopping there loses
nothing. For reading the window as post-answer behaviour it is not: such a
window contains the computation that produces the answer rather than redundant
verification of it, so a verbalisation sampled from early in a tail must not be
read as a model checking its work.

### The forcing budget makes the tail a lower bound

Criterion 1 forces an answer under a 24-token budget. On **40.7%** of evaluated
boundaries that completion was cut off mid-derivation, and on **24.1%** the cut
was *blocking*: criterion 1 failed while every resampled continuation from the
same prefix reached the final answer. 19 of 24 problems are affected.

So the detected tail is a floor. Counting those boundaries as satisfying
criterion 1 gives the other end: mean tail fraction **0.356 → 0.486**, tail
token share **31.7% → 46.3%**. The relaxed figure is not the pre-registered
criterion and is reported only as the upper end of the interval.

### The hidden-state probe is largely reading position

The layer-20 correctness probe reaches a grouped-CV AUC of **0.996** on 272
boundaries. Fitting the same model family on the same grouped folds using only
activation-free positional features — chunk index, relative position, prefix
tokens — reaches **0.962**. The probe is 0.034 AUC above a predictor that never
sees an activation.

Its label, *the intermediate answer at this boundary is already correct*,
becomes true once the trace has worked the answer out and stays true, so it is
strongly ordered by position. The probe's row in the stopping comparison should
be read as a positional rule with a small activation-derived increment, not as
a correctness readout. This matters because the probe is one of the baselines
the primary research question is measured against.

### Also measured

- **The cheap agreement rule fires earlier than the AST on 57.1% of problems**
  (mean gap 2.43 chunks), later on 38.1%. Mo et al.'s concern, quantified on
  this run's own traces.
- **F5**: over the settings that could be exercised, mean tail fraction moves
  by 0.006. `k=5` could not be simulated — only 3 continuations per boundary
  were generated — and is reported as not applicable rather than as stability.
- **F4**: strict and permissive extractors agree on the final answer of every
  trace (1.000) and disagree at no boundary; the alternative chunker yields
  13.5 chunks against 9.6, which is a real sensitivity of the chunk-indexed
  tail.
- **F1**: correlation between trace length and tail *fraction* is −0.46, so the
  construct is not a length statistic in disguise.

---

## 3. What the first real verbalisations look like

Worth reading before the aggregate faithfulness numbers, because they shape how
those numbers should be interpreted. On `gsm8k-test-00197`, whose answer is
6277, the verbaliser produced well-formed English that identified the register
and discourse position of the window correctly:

> Structured numbered problem format with formal math text explaining
> calculations, presenting sequential steps using "Thus" and "Final Answer"
> sections to derive a conclusion.

and then invented the specifics — one sample asserted the quantity was 122, the
other 53. Two samples of the same activation inventing different numbers is
itself evidence that neither is reading the value. The matched-norm Gaussian
control produced equally confident text about Excel-style procedure formats and
maths forum posts, so the real activation does carry the maths-trace register
and the concluding-position signal that noise does not.

The claim-level audit exists to separate those two kinds of claim, and the run
report prints examples verbatim beside their controls so the characterisation
is checkable.

---

## 4. Deviations this run actually needed

All recorded in [`DECISIONS.md`](DECISIONS.md); these are the ones the run
itself forced.

| | |
|---|---|
| **D20/D24** | F8's bf16 arm is ~12 GB against 6.44 GB of VRAM and 2–3 GB of free RAM. It is split across GPU and host by measured free memory with the remainder offloaded to disk, and refuses to report a near-zero cosine with an implausible norm ratio as drift rather than as a load failure. |
| **D23** | Upstream's `NLACritic` stages the whole reconstructor in host memory before moving it to the GPU. That does not fit here, so the backbone is placed directly on the GPU for the duration of the construction. The arithmetic is untouched. |
| **conversion** | `save_pretrained` defaults to a 50 GB shard, so a 7B model is written as one preallocated file; that failed with `os error 112` against ~10 GB free. Conversions now write 2 GB shards after a pre-flight space check, and the bf16 source is evicted *before* the save — without which the reconstructor's output had nowhere to go. |
| **token cap** | MATH traces hit `generation.max_new_tokens: 512` on 3 of 10 problems, and those are precisely the three with no detectable tail. Raising it is the first thing a larger run should change. |

---

## 5. What has not run

- **The main configuration.** `configs/main.yaml` is 250 problems. At the
  pilot's measured 12.9 min/problem for traces alone that is over 50 hours on
  this machine, so it has not been attempted. The pilot is the achieved scale.
- **The two supervised analyses** (`scripts/predict_tail_fraction.py`,
  `scripts/predict_correctness.py`) refuse below 30 train / 20 eval rows and so
  cannot run on 24 problems. They are correct to refuse. The root-level
  `regression.py` / `classification.py` do run at this size and write a
  `provenance.json` recording the corpus they used.
- **`runs/mlcorpus`** holds 33 traces from a run that predates the chunker and
  answer-parser fixes. They are not poolable with the pilot's — the chunk
  boundaries baked into them differ — and `common_data.py` now refuses to pool
  them automatically (D22).
- **A second annotator** for O6 inter-rater reliability. Single-rater, as the
  review anticipated.

---

## 6. How to reproduce or extend this

```powershell
$env:HF_HOME = "<volume>/hf_cache"      # bf16 staging
$env:NLAAST_QUANT_DIR = "<volume>/nf4"  # 4-bit checkpoints

.venv\Scripts\python.exe -m pytest                       # 430 tests, no GPU
.venv\Scripts\python.exe -m pytest -m slow               # adds the 4-bit GPU paths
.venv\Scripts\python.exe scripts\run_pipeline.py --config pilot --dry-run
```

The pilot's stages are all complete, so `run_pipeline.py --config pilot` is a
no-op. To re-run from a stage, pass `--from <stage> --force`.

**For a larger run,** the order that fits on this hardware is the one the
orchestrator already encodes, with the checkpoint preparation split in two
because the three models cannot coexist:

```powershell
# 1. traces, activations, and F8 while the bf16 target still exists
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --stages env data traces acts
.venv\Scripts\python.exe scripts\run_falsification_tests.py --config main --only F8_quantisation --force

# 2. the autoencoders, after the bf16 target is evicted
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --evict-bf16 target
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config main --skip target --force

# 3. the rest
.venv\Scripts\python.exe scripts\run_pipeline.py --config main --from ast
```

Raise `generation.max_new_tokens` to 768 for MATH first — see §4.

---

## 7. What to watch on a larger run

- **Verbaliser integrity.** The `nla` stage logs `ok=N/M well-formed English`.
  It held at 100% here; below 50% it logs an error and the arm should be
  treated as compromised (cross-check F8).
- **Reconstruction FVE** against the checkpoint card's 0.752. Below ~0.38 the
  stage sets `fidelity_warning`.
- **`tail_rate` at 0 or 1.** Read `nontrivial_tail_rate` and
  `mean_tail_fraction` instead; §2 explains why.
- **The probe's margin over its positional floor.** Under 0.05 the stage warns,
  and the probe should not be described as a correctness readout.
- **F6's shuffle arms.** `leak_suspected: null` means the shuffle moved no
  labels and tested nothing, not that the probe is clean.
- **`total_clamped_positions`** in `acts/summary.json`. It was 0 here; anything
  else means an activation was read where its chunk boundary did not say.
