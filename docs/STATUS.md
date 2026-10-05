# Current state and next steps

What exists, what does not, where the work stopped, and the next command to
type. Last reviewed 2026-10-06.

**One-line status:** the codebase is complete and tested; the target model is
downloaded and quantised; the pilot's trace generation was interrupted partway
through its first run. Real bugs were found and fixed during that run, so the
pilot needs re-running before anything downstream is meaningful. **No
scientific result has been produced yet.**

---

## 1. What is complete

### Environment (verified, not assumed)

| | |
|---|---|
| venv | `.venv/`, Python 3.14.0 |
| torch | 2.9.1+cu128, CUDA available |
| GPU | RTX 4050 Laptop, 6.44 GB, sm_89 |
| transformers / bitsandbytes | 5.18.0 / 0.50.2 |
| `math_verify` | working (needs `timeout_seconds=None` on Windows) |
| lockfile | `requirements.lock.txt` (87 packages) |

**Measured feasibility** (`scripts/probe_vram.py`): 4-bit Qwen2.5-7B loads in
**5.56 GB**, peaks at **5.77 GB** at batch 4, runs at **20.5 tok/s** single and
**45 tok/s** batched. Layer-20 hook verified: `(1, 88, 3584)`, float32.

### Codebase

All 13 pipeline stages, the 3 supplementary analysis stages and the
orchestrator are written. `python -m pytest` passes, with no GPU required.
[`PIPELINE.md`](PIPELINE.md) says what each stage does.

### Research groundwork (the part that is hard to redo)

The **real** NLA interface was recovered from the released checkpoints rather
than guessed:

```
injection_token_id        149705   (char ㈎)
injection neighbours      29 / 522  (validated, or injection silently misfires)
injection_scale           150.0
mse_scale                 59.8665 = sqrt(3584)
AV prompt                 chat template with <concept>{injection_char}</concept>
AR prompt                 "Summary of the following text: <text>{explanation}</text> <summary>"
AR architecture           21-layer truncated Qwen2 + Linear(3584,3584) value head
extraction convention     output of decoder block 20 == hidden_states[21]
reconstruction metric     both vectors L2-normalised to sqrt(d); MSE = 2(1-cos)
```

### Models

| | state |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | bf16 downloaded **and** NF4 built (15 GB bf16, 5.2 GB NF4) |
| `kitft/nla-qwen2.5-7b-L20-av` | **not downloaded** — needs 15.3 GB of staging disk |
| `kitft/nla-qwen2.5-7b-L20-ar` | **not downloaded** — needs 10.9 GB of staging disk |

### Data

GSM8K (1319 test) and MATH algebra + counting_and_probability staged and
validated. Gold recovery by the permissive parser: **1.000**.

---

## 2. Where it stopped

`scripts/generate_traces.py --config pilot` was interrupted at 21 of 24
problems. `scripts/detect_answer_stable_tail.py` then ran on those 21, and the
two supervised analyses were exercised on the result.

**None of these numbers are reportable** and none are committed. They are
recorded here only as evidence that the stages execute:

- 21 traces, 16 with a correct final answer, 3 truncated at the token cap
- a tail was detected on 18 of 21; 3 came back `no_stable_point`
- the cheap convergence rule fired earlier than the AST on half of them
- the regression ran on 11 train / 7 held-out rows and produced a negative R²;
  the classification could not run at all, because at this size the training
  split came out single-class

A held-out R² from seven rows is a sample-size artefact. The synthetic-frame
tests in `tests/test_supervised_scripts.py` exist precisely because the real
corpus was too small to exercise those branches.

`runs/mlcorpus` holds 33 traces from an interrupted attempt at the larger
corpus.

---

## 3. Bugs found and fixed during the pilot

The pilot did its job. Each of these is fixed and covered by a test, but the
traces generated *before* the fix are stale — which is why the pilot has to be
re-run rather than resumed.

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | The chunker treated `$` as a LaTeX delimiter, but GSM8K writes money as `$2.` | Chunks merged — 7 instead of 12 on the first real trace, moving every activation position | `_math_spans()` only treats `$…$` as maths when the content looks mathematical |
| 2 | `_strip` removed leading `\` and trailing `}` | `\frac{1}{2}` became `frac{1}{2` — every LaTeX answer silently corrupted | Balanced, LaTeX-aware peeling |
| 3 | The `\frac` regex was `\\d?frac` (literal backslash plus optional "d") | Fraction→float conversion never fired | `\\[dt]?frac` |
| 4 | One regex captured through a repeated "answer is" | `"the answer is The answer is 6"` → `"The answer is 6"` | Anchor on the **last** marker, then re-anchor |
| 5 | Unit suffixes not stripped | `"180 minutes"` vs gold `"180"` scored **incorrect** | Conservative `_strip_units`, never fires on `3 pi`, `2 squared`, `5 x` |
| 6 | `np.savez_compressed` appends `.npz`, so the atomic-rename temp file never existed | Activation store was **completely broken** | Write through a file handle |
| 7 | Activations read from **re-tokenised** text | Byte-level BPE can re-tokenise differently → every activation read at the wrong position, undetectably | Replay the sampled `token_ids` verbatim; measure and record the drift re-tokenisation *would* have caused |

Also added during the pilot, not bug fixes but needed: continuation and forced
text retention (so a later parser fix can be applied without regenerating),
batched boundary evaluation, a backwards-scan cap, batched AV generation, an
`</explanation>` stop string, `nla.max_problems`, and `--evict-bf16`.

---

## 4. What is pending

### P1 — Re-run the pilot end to end *(next action)*

Nothing downstream of `ast` has run even once. The first full pass will likely
surface more problems; that is expected and is the point of a pilot.

### P2 — Decide the MATH token cap

MATH traces hit the 512-token cap repeatedly. Options: raise
`generation.max_new_tokens` to 768 for `main` (costs time), or accept
truncation (already flagged as `TailStatus.TRUNCATED`, but the "final" answer
is then not final). **Recommendation:** raise to 768 and reduce `data.n_math`
to keep the budget.

### P3 — An ordering constraint that must not be forgotten

F8 (quantisation drift) compares 4-bit against **bf16** activations. The bf16
target must be evicted to make room for the verbaliser. So:

```
acts  →  F8  →  evict bf16  →  download AV/AR  →  nla
```

`run_pipeline.py` inserts `robustness:F8` automatically. If stages are run by
hand, do not skip it — recovering costs a fresh 15 GB download.

### P4 — AV/AR not yet downloaded or exercised

~26 GB of download, and the eviction in P3 is a hard prerequisite rather than
an optimisation.

The verbaliser has never been run. `tests/test_inputs_embeds_contract.py`
verifies the mechanism on a tiny model, but the real checkpoint under 4-bit is
untested — this is the largest remaining technical risk (see
[`DECISIONS.md`](DECISIONS.md) D2).

### P5 — The main run

Several hours for traces alone at current throughput. Resumable; report the
achieved N, not the configured N.

### P6 — The execution report

To be written from real results at the end. `scripts/write_run_report.py`
generates the per-run `RESULTS.md`; the repository-level execution report does
not exist yet because there is nothing to report.

---

## 5. How to start again

```powershell
$env:HF_HOME = "<volume>/hf_cache"
$env:NLAAST_QUANT_DIR = "<volume>/nf4"

.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\run_pipeline.py --config pilot --dry-run
```

### The actual next command

```powershell
# 1. re-run pilot traces with the fixes (~20 min, faster now that it batches)
Remove-Item -Recurse -Force runs\pilot\traces
.venv\Scripts\python.exe scripts\generate_traces.py --config pilot --force
```

```powershell
# 2. everything that does not need the autoencoder (~20 min)
.venv\Scripts\python.exe scripts\extract_activations.py --config pilot --layers 14 20 24
.venv\Scripts\python.exe scripts\run_falsification_tests.py --config pilot --only F8_quantisation --force
.venv\Scripts\python.exe scripts\detect_answer_stable_tail.py --config pilot --sweep
.venv\Scripts\python.exe scripts\score_baselines.py --config pilot
```

**Inspect before continuing.** Check `runs/pilot/ast/summary.json` — a tail
rate of 0 or 1 means the criteria need re-examining, not that the result is
interesting.

```powershell
# 3. free the staging volume and fetch the autoencoder (~50 min, mostly download)
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config pilot --evict-bf16 target
.venv\Scripts\python.exe scripts\prepare_checkpoints.py --config pilot --skip target --force
```

```powershell
# 4. the NLA arm, then the rest
.venv\Scripts\python.exe scripts\run_autoencoder.py --config pilot
.venv\Scripts\python.exe scripts\audit_faithfulness.py --config pilot
.venv\Scripts\python.exe scripts\run_interventions.py --config pilot
.venv\Scripts\python.exe scripts\aggregate_results.py --config pilot
.venv\Scripts\python.exe scripts\run_falsification_tests.py --config pilot --force
.venv\Scripts\python.exe scripts\write_run_report.py --config pilot
```

Or, once step 3 has been done once, simply:

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --config pilot
```

which skips completed stages and respects the F8 ordering. The main run is the
same command with `--config main`, and is resumable the same way.

---

## 6. Things to watch for on the first full pass

- **Verbaliser integrity.** The `nla` stage logs `ok=N/M well-formed English`.
  Below 50 % it logs an error — that is the documented off-distribution
  signature and would mean the 4-bit activations are the likely cause
  (cross-check F8).
- **Reconstruction FVE** against the checkpoint card's 0.752. Below ~0.38 the
  stage sets `fidelity_warning` and the NLA arm should be treated as
  compromised.
- **Tail rate** at 0.0 or 1.0 — the criteria are mis-set, not the model.
- **Probe AUC under shuffled labels** (F6) away from 0.5 — leakage.
- **Alignment audit** in `runs/<id>/acts/summary.json`:
  `total_clamped_positions` should be ~0.
