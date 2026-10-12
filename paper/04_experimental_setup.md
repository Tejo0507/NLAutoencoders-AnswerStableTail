# 4. Experimental setup

*Draft. Every value below is read from `configs/base.yaml` and
`configs/main.yaml` as resolved for the main run. The achieved N is the one
quantity still outstanding — see §4.6.*

## 4.1 Model and site

| | |
|---|---|
| target | `Qwen/Qwen2.5-7B-Instruct` |
| precision | NF4 (4-bit) |
| `d_model` | 3584 |
| primary layer | **20** |
| sweep layers | 14 and 24 (falsification test F7) |
| max position embeddings | 8192 |
| autoencoder | `kitft/nla-qwen2.5-7b-L20-av` (verbaliser), `kitft/nla-qwen2.5-7b-L20-ar` (reconstructor), both NF4 |

The autoencoder is released for layer 20 only, so the layer sweep applies to
the probe and semantic-entropy arms; the verbalised readout is layer-20 only
by construction, and §8 records that this makes the layer comparison partial
rather than complete.

**Quantisation is a threat to validity, and it is measured rather than
assumed.** 4-bit layer-20 activations were compared against bf16 on a
subsample: cosine **0.987**, integrity gate passing on essentially every
sample (`runs/pilot/robustness/`). That is far from any threshold that would
change a conclusion. The measurement is fixed at the pilot's 8-problem
subsample and cannot be repeated without re-downloading 15.3 GB of bf16
weights, which §8 records.

**Hardware.** A single NVIDIA GeForce RTX 4050 Laptop GPU, 6.14 GiB VRAM,
with the target model occupying about 5.5 GiB. **[VERIFY]** — read from
`nvidia-smi`, not from a spec sheet. Every stage is single-stream and
batch-limited by that budget, which is the binding constraint on the study's
scale.

## 4.2 Data

| | |
|---|---|
| benchmarks | GSM8K [3], MATH [4] |
| MATH subjects | `algebra`, `counting_and_probability` |
| split | `test` |
| configured N | **150** GSM8K + **100** MATH = **250** |
| train / eval | 40% / 60% by problem, seed `20261005` |

Both benchmarks admit automatic verification of a parsed final answer, which
is what the behavioural outcomes require.

**The problem order is interleaved by dataset.** This is not cosmetic. The
loader previously concatenated the datasets, so every GSM8K problem preceded
every MATH problem and a run halted early would have been GSM8K-only — with
every pooled quantity biased toward the easier benchmark and nothing in the
output saying so. Interleaving makes any prefix of the run proportional, which
is what licenses reporting an achieved N below the configured one
(`docs/DECISIONS.md` D33). Arms that run on a subset are capped
**proportionally by dataset** for the same reason.

The train/eval split matters for two things: the candidate tail direction is
fitted on the train split only, and the probe's folds are group-disjoint by
problem.

## 4.3 Generation

| parameter | value |
|---|---|
| `max_new_tokens` | **768** |
| sampling | `temperature` 0.7, `top_p` 0.95 |
| canonical trace | greedy |
| batch size | 4 |
| `force_answer_max_new_tokens` | **48** |
| forcing suffix | `"\n\nTherefore, the answer is"` |

Two of these were raised from pilot values on measured evidence, and both
changes are declared because they move the tail numbers.

**The token cap went from 512 to 768.** At 512, three of ten pilot MATH traces
hit the cap — and those three were exactly the three with no detectable tail,
because a truncated trace has no settled answer for the criteria to be stable
about. The cap was appearing as a result and biasing the corpus toward
problems the model could finish in budget.

At 768, measured over the main run's first **128** traces:

| | n | truncated | rate | 95% exact CI | accuracy |
|---|---|---|---|---|---|
| GSM8K | 77 | 0 | 0.000 | [0.000, 0.047] | 0.909 |
| MATH | 51 | 4 | **0.078** | **[0.022, 0.189]** | 0.863 |

The interval's upper bound excludes the pilot's 3/10 = 0.300 at a 512-token
cap, so the change did what it was made for. We report the trajectory rather
than the best snapshot: at an earlier 53-problem checkpoint the MATH rate was
1/21 = 0.048 with accuracy 0.905, so the point estimate has **risen** and
accuracy fallen as harder MATH levels entered the corpus. The conclusion that
768 is no longer the binding constraint rests on the interval, not on the
point estimate, and the achieved rate is reported as a limit either way
(`docs/DECISIONS.md` D32).

**The forcing budget went from 24 to 48 tokens.** At 24, the forced completion
was cut off mid-derivation on 40.7% of evaluated boundaries, and on 24.1% the
cut was *blocking*: criterion 1 failed while every resampled continuation from
the same prefix reached the final answer. Counting those boundaries as
satisfying criterion 1 moved the mean tail fraction from 0.356 to 0.486, so
the pilot's measured tail is a **floor**, not an estimate. The relaxed budget
costs roughly 2% more generation and removes a 24% false-negative rate on the
stage's own criterion.

Both caps change the tail numbers, so **the main run's tail statistics are not
directly comparable with the pilot's**, deliberately.

## 4.4 Signals and arms

| stage | parameters |
|---|---|
| AST detection | `K = 5` continuations, `T = 0.8`, `max_new_tokens` 200, at most 12 boundaries evaluated per trace |
| convergence baseline | window on the sweep axis (§3.2); the config's `w = 2` is used only where a single operating point is needed |
| semantic entropy | 8 samples, `T = 1.0`, `top_p` 0.95, clustered by **symbolic equivalence** |
| probe | logistic, `C = 1.0`, standardised, 5 folds, group-disjoint by problem |
| verbaliser | 2 windows per problem, 2 samples per window, `max_new_tokens` 256, `T = 1.0`; controls on |
| NLA arm cap | 80 problems configured; see §4.6 |
| faithfulness | at most 8 claims per explanation, 3 paraphrases, null at the 95th percentile |
| causal | 5 arms, doses {0.0, 0.5, 1.0, 1.5, 2.0}, 5 random directions, cap 60 problems |

Dose 0.0 is retained as a within-arm no-op control: it must reproduce the
unintervened baseline exactly, and the stage checks that it does.

**Semantic entropy is adapted, not reproduced.** The published method clusters
by bidirectional NLI entailment [21]; this implementation clusters by symbolic
equivalence instead. For mathematical answers that is the right equivalence
relation — NLI would mis-handle `\frac{1}{2}` against `0.5` — and no NLI model
is downloaded. The substitution is reported as an adaptation in §8.

**Semantic entropy cannot say when to stop.** It is a per-problem quantity, so
its score is constant across boundaries; the sweep treats it accordingly. That
is a real property of the signal, not a modelling shortcut, and it bounds what
the comparison against it can show.

## 4.5 Seeds, provenance and reproducibility

Every stage is separately seeded (generation, AST, semantic entropy, probe,
verbaliser, faithfulness, causal, analysis). Bootstrap resampling uses 10,000
iterations at 95% confidence.

Each run directory carries the resolved config and its hash, the git commit,
seeds, model revisions and file hashes, dataset version, package versions,
timestamps and per-stage status. Nothing overwrites: a re-run with the same id
refuses unless forced. Every quantity reported in the paper is therefore
traceable to the artifact that produced it.

**The run is resumable per problem**, and the interleaved order means an
interruption costs at most one problem and leaves a proportional corpus.

## 4.6 Achieved N — outstanding

**The achieved N, not the configured N, is what the paper reports.** At the
time of this draft the main run is in progress and the number is not yet
final, so it is left as a blank to be filled rather than guessed.

Two arms run on a proportional subset for cost reasons and report their own
achieved N separately: the verbaliser arm, because the autoencoder emits
hundreds of tokens per activation and is an order of magnitude more expensive
than the behavioural arms; and the causal arm, at 60 problems. The primary
comparison is computed on exactly the problems the verbaliser arm covers, with
every baseline re-swept over those problems, so that the comparison is between
rules rather than between problem sets (§5.1).

Sizing was done against the primary question rather than by ambition: because
the primary comparison is a McNemar test over problems, its power comes from
**discordant** problems, and concordant problems contribute nothing however
many there are. Eight discordant problems is the fewest that can clear the
corrected threshold, which at the pilot's rate of 4/21 puts the requirement at
roughly 42 problems in the verbaliser arm (`docs/DECISIONS.md` D32).
`scripts/size_o3_corpus.py` recomputes this from any completed run.
