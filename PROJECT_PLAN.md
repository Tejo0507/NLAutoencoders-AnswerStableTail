# Implementation Plan

**Project:** Natural Language Autoencoders for Redundancy Diagnostics — the Answer-Stable Tail
**Specification source:** `Project_Review_II.md` (Chapters 1–2). The literature review is the
scientific specification; this document translates it into an executable plan. It does not
restate the literature, it states what gets built and run.

Written 2026-10-05. This file is the intent. The record of what actually ran is
`docs/STATUS.md` plus the per-run manifests under `runs/`; the final execution report
will be written from a completed run and does not exist yet.

---

## 1. What the specification requires

From `Project_Review_II.md` §1.3, the study has one primary research question and three
supporting ones, mapped to six objectives O1–O6 against four limitations L1–L4.

| ID | Requirement | Implementation consequence |
|---|---|---|
| **RQ1 / O1 / L1** | Define the Answer-Stable Tail (AST) without reference to any activation description: chunk-level answer equivalence under truncation, K independently sampled continuations checked against an external verifier, matched-position and matched-length controls | `src/nlaast/ast_detect/` + `scripts/detect_answer_stable_tail.py`. Must run *before* and independently of any NLA code. |
| **O2 / L2** | Reproducible pipeline loading the released Qwen2.5-7B-Instruct L20 autoencoder, emitting verbalisations + reconstructed vectors at tail windows, with versioned prompts, fixed decoding, verifier integration, activation-cache hashes | `src/nlaast/nla/` + `scripts/run_autoencoder.py`. Conventions taken from the released `nla_meta.yaml`, not invented. |
| **RQ-primary / O3 / L2** | Incremental value of the verbalised readout over semantic entropy, a hidden-state correctness probe, and answer-convergence stopping, on paired safe-stopping accuracy and tokens saved at matched compute, with problem-level paired bootstrap CIs | `src/nlaast/baselines/` + `src/nlaast/analysis/` |
| **RQ2 / O4 / L3** | Claim-level audit of tail verbalisations using the deletion / resampling / paraphrase template of Lanham et al., scored by change in reconstruction, cross-checked against an independent-probe control and a verbaliser-only control. RECAP-**inspired**, not RECAP. | `src/nlaast/faithfulness/` |
| **RQ3 / O5 / L4** | Causal sequence: truncation, tail replacement with neutral filler, matched-position and matched-random-direction comparisons, per Zhang & Nanda | `src/nlaast/causal/` |
| **O6** | Release labelled tail corpus + harness, machine-readable results | `runs/<id>/` layout, `scripts/write_run_report.py` |

Three distinctions from §2.1.10 are load-bearing and are enforced structurally in the code,
not merely stated:

1. **Answer stability ≠ causal redundancy.** AST detection never sees an activation.
2. **Decodability ≠ causal use.** The probe stage and the causal stage are separate, and the
   probe's output is never described as mechanistic.
3. **Reconstruction fidelity ≠ claim-level faithfulness.** The reconstruction metric and the
   faithfulness metrics are computed by different modules and reported in different tables.

---

## 2. Hardware and the scale it forces

Measured on this machine, not assumed:

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **6.44 GB VRAM**, sm_89 (Ada) |
| RAM | 15.65 GB total, ~1.5–6 GB typically free |
| Disk | D: ~18 GB free (code + venv + runs), E: ~22 GB free (model cache) |
| OS / Python | Windows 11 26300, CPython 3.14.0 |
| Stack | torch 2.9.1+cu128, transformers 5.18.0, bitsandbytes 0.50.2 |
| Network | ~8.9 MB/s to HuggingFace |

Three models are needed, all Qwen2.5-7B-shaped:

| Model | bf16 size | Role |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 15.2 GB | target model; generation + layer-20 activations |
| `kitft/nla-qwen2.5-7b-L20-av` | 15.3 GB | activation verbaliser |
| `kitft/nla-qwen2.5-7b-L20-ar` | 10.9 GB | activation reconstructor (21-layer truncated backbone + Linear(3584,3584) head) |

41.4 GB of bf16 weights against 22 GB of usable cache and 6.44 GB of VRAM. Neither bf16
residency nor concurrent residency is possible.

**Resolution.** Each model is downloaded once, quantised to 4-bit NF4 (double quantisation,
bf16 compute), saved, and the bf16 source deleted. Quantised footprint ≈ 5.5 + 5.5 + 4.0 GB.
Stages load exactly one model at a time and the orchestrator serialises them.

**This is a deviation from the released checkpoints' native precision and it is treated as a
threat to validity, not as a detail.** Two things are quantified rather than assumed:

- **Q1 — activation drift.** For a subsample, layer-20 activations are computed both from the
  4-bit target and from the bf16 target (CPU, sequential, slow but exact) and compared by
  cosine similarity. If drift is large the NLA is being fed off-distribution vectors.
- **Q2 — verbaliser integrity.** The NLA authors document a specific failure signature when the
  injected vector is off-distribution: the verbaliser emits CJK text instead of an English
  `<explanation>`. A gate checks output language and tag well-formedness on every batch, and
  round-trip fidelity (FVE) is compared against the 0.752 in-distribution value on the
  checkpoint card. Note that `embed_tokens` is an `nn.Embedding` and is **not** quantised by
  bitsandbytes, so the injected vector itself enters the network exactly; only the transformer
  weights are 4-bit.

If Q1 or Q2 fails, that is reported as a failed experiment with diagnosis, not papered over.

**Scale.** The study is sized by token budget, not by ambition. Single-stream 7B NF4 on this GPU
runs at roughly 15–25 tok/s, batched meaningfully faster. Every stage writes per-problem
checkpoints and resumes, so the achieved N is whatever completed when time ran out, and the
achieved N is what gets reported.

---

## 3. Component strategy — reuse vs. write

Per the brief's §2 taxonomy.

| Component | Strategy | Detail |
|---|---|---|
| NLA conventions (injection, normalisation, prompts, scales) | **(A) reuse** | `nla_meta.yaml` shipped with each checkpoint is the single source of truth. Never hardcoded. |
| `nla_inference.py` (NLAClient / NLACritic) | **(A) vendor, Apache-2.0** | Vendored to `third_party/nla_inference/` with LICENSE + NOTICE. `NLACritic` (the reconstructor) is used essentially as shipped. |
| AV generation backend | **(E) wrapper/adaptation** | Upstream drives the verbaliser through an SGLang server with the `input_embeds` API. SGLang does not run on Windows and will not fit 6.44 GB alongside the model. Replaced with a local `transformers` path that reproduces the *same* arithmetic: chat template → embedding lookup → `normalize_activation(v, injection_scale)` → `inject_at_marked_positions` with neighbour validation → `generate(inputs_embeds=...)`. The injection math is the vendored function, unmodified. |
| Target model, AV, AR checkpoints | **(B) download** | `scripts/prepare_checkpoints.py`, resumable, hash-verified, sequential-with-eviction. |
| GSM8K, MATH | **(C) download** | Already partly staged in `data/raw/`; the loader validates and extends. |
| Answer verification | **(A) reuse** | `math_verify` (HF) for symbolic equivalence; a deterministic fallback extractor for GSM8K's `#### n` format. Windows note: `math_verify`'s timeout uses `multiprocessing` and fails on Windows — calls pass `timeout_seconds=None` and are wrapped. |
| torch / transformers / bitsandbytes / sklearn / statsmodels | **(D) dependency** | pinned in `requirements.lock.txt`. |
| Trace chunking, intermediate-answer parsing | **(F) new** | No reusable implementation of Liu & Wang's chunking exists; written here. |
| AST detector | **(F) new** | The study's own construct. |
| Hidden-state correctness probe | **(F) new** | Reimplements Zhang et al.'s linear probe on our activations; trivial with sklearn, but the leakage-safe grouped splitting is the part that matters. |
| Semantic entropy | **(F) new, adapted** | Farquhar et al. cluster free-form answers by bidirectional NLI entailment. For mathematical answers, semantic equivalence *is* symbolic equivalence, so clustering uses `math_verify`. **This is an adaptation and is documented as one** — no NLI model is downloaded (no disk budget, and it would be the wrong equivalence relation for `\frac{1}{2}` vs `0.5`). |
| Faithfulness audit | **(F) new** | Claim splitting + the three Lanham perturbations + two controls. |
| Causal interventions | **(F) new** | Truncation, filler replacement, matched-position, matched-random-direction. |
| Statistics | **(F) new over (D)** | Problem-level paired bootstrap, BCa where it matters, Benjamini–Hochberg across the pre-registered test family. |
| Orchestrator | **(G) new** | `scripts/run_pipeline.py`, stage DAG, resume, per-stage manifests. |

---

## 4. Stage DAG

```
env           environment + hardware report, written to the run dir
data          GSM8K + MATH -> validated, deterministically split problem set
models        download -> quantise(NF4) -> evict bf16; record revisions + hashes
traces        target model generates reasoning traces (+ resampled continuations)
acts          forward hook on model.model.layers[20]; chunk-boundary positions
ast           Answer-Stable Tail detection  [NO activation input]
baselines     answer-convergence stopping | hidden-state probe | semantic entropy
nla           AV verbalisation at tail windows -> AR reconstruction -> FVE
faithfulness  claim-level audit: delete / resample / paraphrase + 2 controls
causal        truncation | filler | matched-position | matched-random-direction
analysis      metrics, paired bootstrap, effect sizes, BH correction
robustness    falsification battery (section 6 below)
report        tables, figures, machine-readable results, execution report
```

Dependencies: `ast` needs `traces` only. The probe inside `baselines` needs `acts`+`ast`, as
does `nla`. `faithfulness` needs `nla`. `causal` needs `traces`+`ast`. `analysis` needs
everything from `ast` onwards. `ast` deliberately does not depend on `acts`, which is how
L1-independence is enforced in the DAG rather than by assertion.

---

## 5. Definitions fixed in advance

Pre-registered here so the analysis cannot drift to fit the result.

**Chunking.** A trace is split at sentence boundaries into chunks `c_1..c_n`. For each prefix
`P_i = c_1..c_i` an intermediate answer `a_i` is parsed by the same extractor used for the final
answer. `a_i` may be `None`.

**Answer-convergence boundary** `b_conv(w)` — smallest `i` such that `a_i..a_{i+w-1}` are all
non-null and pairwise equivalent. This is the Liu & Wang baseline, `w` configurable.

**Answer-Stable Tail.** A boundary `i` is the AST start iff all of:

1. *Truncation equivalence.* Forcing an answer from prefix `P_i` yields an answer equivalent to
   the full trace's final answer.
2. *Resampled-continuation stability.* `K` independent continuations sampled from `P_i` at
   temperature `T` all yield answers equivalent to the full trace's final answer. This is the
   control Mo et al. show the agreement rule lacks, and it is required, not optional.
3. *Persistence.* Conditions 1–2 hold for every boundary `j >= i` in the trace.

The AST is `c_{i*}..c_n` for the smallest such `i*`. `tail_fraction = (n - i* + 1)/n`.

**Edge cases are recorded, not dropped.** `no_stable_point` (no `i` qualifies),
`stable_at_zero` (`i* = 1`, answer stable before any reasoning), `unparseable`,
`incorrect_final` (final answer wrong — kept; the AST construct does not presuppose correctness
and excluding these would bias the comparison), `truncated_generation`. Each carries a status
code into the corpus and each is counted in the report.

**Matched controls for the tail.** For every detected AST of length `L` starting at `i*`:
*matched-position* = a window of length `L` ending at `i*-1` (pre-stabilisation, same trace);
*matched-length* = a length-`L` window at a uniformly random valid start in the same trace.

**Safe stopping (the O3 outcome).** A stopping rule fires at boundary `s`. It is *safe* iff the
answer forced from `P_s` is equivalent to the full-trace answer **and** the full-trace answer is
correct, or both are incorrect — i.e. stopping did not change verified outcome. Reported jointly
with `tokens_saved = 1 - |P_s| / |full trace|`. Rules are compared on the safety/saving curve at
**matched token budget**, not at each rule's own operating point.

**Reconstruction fidelity.** Exactly the released convention: both vectors L2-normalised to
`mse_scale = sqrt(3584) = 59.8665`, `MSE = 2(1 - cos)`, and `FVE` against the empirical variance
baseline of our own activation sample. Reported as reconstruction, never as faithfulness.

**Claim-level faithfulness (RQ2).** A verbalisation is split into atomic claims. For each claim
`c`: `delete(c)`, `resample(c)` (regenerate that claim from the AV at a different seed) and
`paraphrase(c)`. A claim is **reconstruction-dependent** iff deleting it degrades reconstruction
cosine by more than the paraphrase-null distribution's 95th percentile — the paraphrase arm is
the null, which is what makes the threshold non-arbitrary. Two controls:

- *Independent-probe control.* A probe trained on held-out activations, never on the verbalised
  text, scores whether the claim's content is decodable from the activation. Group-disjoint
  splits; leakage is tested explicitly by the falsification battery.
- *Verbaliser-only control* (Li et al.). The AV is run with the activation replaced by a
  Gaussian vector at the same L2 norm. Claims reproduced under this control are attributable to
  verbaliser priors, not to the target activation. This is the single most important control in
  the study and it is run on every sampled verbalisation, not a subset.

**Causal (RQ3).** Interventions at the layer-20 residual stream over the tail token span:
`truncate` (stop generation at `i*`), `filler` (replace tail text with neutral filler of matched
token length), `ablate_dir` (project out a candidate tail direction obtained from the
*training-split* difference-in-means between tail and matched-position windows),
`random_dir` (project out a random direction of matched norm — the Zhang & Nanda control), and
`matched_position` (apply `ablate_dir` at the pre-stabilisation window instead). The claim
"a tail direction exists" requires `ablate_dir` to beat **both** `random_dir` and
`matched_position`, with a dose–response curve over projection coefficient.

---

## 6. Falsification battery (stage `robustness`)

Run because the brief's §16–17 require it, and because the review itself (§1.3) states that the
cheaper signals matching or beating the NLA readout is a scientifically acceptable outcome.

| # | Alternative explanation | Test |
|---|---|---|
| F1 | NLA adds nothing over length | Add trace-length-only predictor to the baseline set |
| F2 | NLA adds nothing over difficulty | Stratify by MATH level and by gold-answer magnitude |
| F3 | The verbalisation reflects verbaliser priors | Verbaliser-only control (above), reported as a headline number |
| F4 | AST is an artefact of the answer parser | Re-run detection with a second, stricter extractor; report agreement |
| F5 | AST is sensitive to its own definition | Sweep `K`, `T`, `w`, chunker; report rank stability of conclusions |
| F6 | Probe leaks | Shuffle labels within group; probe must collapse to chance |
| F7 | Layer 20 is special only by construction | Extract at layers 14 and 24; the NLA is L20-only, so only probe/SE arms sweep |
| F8 | 4-bit quantisation drives the result | Q1/Q2 above, plus bf16-CPU replication of the headline comparison on a subsample |
| F9 | Multiple testing | Pre-registered family; Benjamini–Hochberg at q=0.05 |
| F10 | Token-position confound in reconstruction | Compare FVE at tail vs matched-position vs matched-length windows |

---

## 7. Deliverable layout

```
configs/            base.yaml + pilot / main / smoke / mlcorpus overlays
src/nlaast/         library code (no scripts, no I/O side effects at import)
scripts/            run_pipeline.py orchestrator + one entry point per stage
third_party/        vendored Apache-2.0 nla_inference + LICENSE + NOTICE
tests/              pytest; unit + a tiny end-to-end
runs/<run_id>/      config snapshot, manifest, env, logs, per-stage artefacts
docs/               stage documentation
docs/STATUS.md      what has run, what has not, and the next command
```

Every run directory carries: resolved config + its hash, git commit, seeds, model revisions and
file hashes, dataset version, package versions, timestamps, and per-stage status. Nothing
overwrites: a re-run with the same id refuses unless `--force`.

---

## 8. Known risks, carried openly

| Risk | Mitigation | If unmitigated |
|---|---|---|
| 4-bit shifts activations off the NLA's training distribution | Q1/Q2 gates, F8 | Report NLA arm as compromised; behavioural arms stand |
| 6.44 GB VRAM too small even at 4-bit with long contexts | cap context, batch 1–4, CPU offload of KV | reduce max trace length, document |
| AV costs several hundred tokens per activation | windows per problem capped and configurable | fewer windows, same analysis |
| Achieved N too small for the paired bootstrap to resolve a difference | report CIs honestly; a wide CI is a result | state as underpowered, do not claim |
| Second annotator for O6 inter-rater reliability unavailable | the review already anticipates this | report single-rater, state the absence |
| Causal sweep (O5) exceeds budget | O5 already scoped as an extension in the review | report the subset that ran |

---

## 9. Status

See `docs/STATUS.md`. This plan is the intent; that file is the record of what has
actually run. No stage downstream of `ast` has completed, so there is no execution
report yet.
