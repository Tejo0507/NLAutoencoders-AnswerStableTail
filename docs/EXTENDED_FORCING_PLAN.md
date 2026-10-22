# What a cut-off forced answer means: an extended-budget forcing pass (D82)

**Frozen before any extended-budget answer was generated.** D82 records this
file's SHA-256. Later changes are appended under *Deviations*, dated.

**What had been seen when this was written.**
- On `main` and `main_nla240`: every D69, D79 and D81 result. That includes
  the two extreme readings of a cut-off completion (D79 §B.4) and the
  information audit (D81).
- On `rep1`: no outcome. No forcing, safety label or analysis existed.

## 1. The question

The registered outcome (PROJECT_PLAN §5) forces an answer from the prefix
`P_s` with a 48-token budget. Before the tail start, 54–57% of those
completions are cut off. The extractor then takes whatever number came
last. 98–99% of these disagree with the answer the model's own 200-token
continuations reach (D79).

Read a cut-off as *the answer the model would commit to*, and every readout
advantage disappears. Read it as *no answer*, and almost nothing changes.
Neither reading is a measurement.

This pass measures what the forced completion says when it is allowed to
finish. It **counts the tokens that finishing costs**.

## 2. What is generated

**Boundaries, on each run:**
- every design-A menu boundary whose registered forced completion was cut
  off at 48 tokens (`forced_finished` false, stored or completed);
- plus a seeded random sample of **50** menu boundaries whose completion
  finished, as the instrument check (seed 20261020).

**Instrument:** identical to the registered forcing call, except for the
budget:
- the same chat prompt, prefix and suffix (`"\n\nTherefore, the answer is"`);
- greedy decoding (temperature 0, top-p 1, no sampling);
- per-problem blocks of `block_size(batch_size)` (D80): 4 on `main`, 2 on
  `rep1`;
- the same seed derivation.

**Budget: E = 256 new tokens.**
- The model's own 200-token continuations reach a parseable answer 100% of
  the time (D79 B.1), and 256 leaves headroom.
- A completion still cut off at 256 is flagged. It is read both ways (§3).

**Recorded per boundary:**
- the answer (`extract_answer`, PERMISSIVE, on suffix + text, as
  registered);
- whether generation finished;
- the exact number of generated tokens;
- whether the first 48 generated tokens equal the stored 48-token
  completion's.

## 3. Outcomes

For every menu boundary `s`, safety is defined as registered: the answer's
correctness equals the final answer's. Only the answer differs between
outcomes.

| outcome | answer at `s` | role |
|---|---|---|
| **O48** | the registered 48-token forced answer | **registered**; D75's verdict is computed on it alone |
| **O256** | finished at 48: unchanged; cut off at 48: the 256-token answer | sensitivity (this plan's primary *reading*) |
| O256-u | as O256, but a completion still cut off at 256 has no answer | sensitivity |
| Ocont | the majority answer of the boundary's 5 sampled continuations (ties: no majority, not known) | diagnostic; only problems whose whole menu has continuations |

**Savings.** A stop must emit its answer, and a 200-token forced completion
is more reasoning, not an answer. Two definitions are therefore used for
every outcome:
- **registered:** `1 − prefix_tokens / total_tokens`. Forcing tokens are
  not counted.
- **net:** `max(0, 1 − (prefix_tokens + forced_tokens) / total_tokens)`, where
  `forced_tokens` is the generated count. It is exact for the new
  generations. For 48-token completions it is the re-tokenised count of the
  stored text, checked against the exact count on the 50-boundary sample.

  For Ocont the net saving uses the forced tokens of O256, because the
  continuations are a diagnostic and not a stopping procedure.

**Primary reading of this plan: O256 with net savings.** It is the only
combination that both lets the answer finish and charges for it.

## 4. Analyses

On each outcome × saving cell, design A only, with the same machinery and
seed as D69/D79 (`position_truncation_audit.build_design`):
- the menu, the rules' scores and the budget fractions are unchanged;
- B\* is recomputed per cell, from that cell's own labels and savings;
- the matching tolerance is 0.02, with at least 3 matched budgets;
- the paired problem bootstrap uses 2,000 replicates, seed 20261019.

**Comparisons:**
- **D69/D75:** readout − probe (L20), readout − no-signal, and every rule −
  no-signal;
- **D79:** families X and Y, with the falling-position rule and the
  residualised readout.

**Readings:**
- D69's `verdict` and family X/Y's readings are applied unchanged in every
  cell;
- on `rep1`, D75's `confirmatory_verdict` is also applied. It is labelled
  sensitivity and never replaces the registered verdict.

**Information audit (D81) under O256:**
- Q1-P (readout over position and probe);
- probe-on-activation over position;
- text over probe-on-activation.

**Instrument checks**, reported before any comparison:
1. **Stability:** on the 50 sampled finished boundaries, the 256-token
   answer equals the stored 48-token answer.
   - **Gate:** ≥ 0.95. Below it, O256 is reported as *instrument unstable*
     and no comparison is interpreted.
2. The re-tokenised 48-token count equals the exact count on those 50
   boundaries. Agreement within ±2 tokens is required for net savings to
   be used.
3. The share of cut-off boundaries whose first 48 tokens reproduce the
   stored completion.
4. The share still cut off at 256.
5. Agreement of the 256-token answer with the continuations' majority, and
   with the final answer, split by position relative to `i*`.

## 5. Decision language (fixed now)

**What cut-off answers mean.** Let `f` be the share of cut-off-at-48
boundaries *before* `i*` whose 256-token answer equals the final answer.

| `f` | reading |
|---|---|
| ≥ 0.80 | **the 48-token outcome mostly measured "can the model commit within 48 tokens"**: a cut-off answer is usually not the model's answer |
| ≤ 0.20 | **the registered reading is supported**: finishing usually does not reach the final answer |
| otherwise | **mixed** |

**Whether a conclusion is budget-dependent.** For each D69, D75 and D79
reading, report O48 → O256-net. A reading that changes is reported as
*dependent on the forcing budget*. One that holds is reported as *robust to
it*.

**Nothing here changes the registered result.** The D75 verdict is the one
computed on O48. This pass decides how that verdict is worded: as being
about "commitment within 48 tokens" or about "the answer reached".

## 6. Order, cost and status

| run | when | status |
|---|---|---|
| `main_nla240` (its menu contains `main`'s) | first, as soon as the GPU is free of `rep1` | exploratory: the O48 results were seen |
| `rep1` | after D75's confirmatory analysis exists | pre-specified sensitivity |

**Cost:**
- at most 256 tokens per cut-off menu boundary, plus 50 checks;
- about 400–500 boundaries on `main_nla240` and an estimated 600–800 on
  `rep1`;
- at the measured 1.35 s per 48-token answer, roughly 1–2 GPU-hours per run.

**Code:**
- `scripts/force_extended.py` writes `runs/<run>_extforced/`. It is
  resumable and never touches `runs/<run>/` or the registered completion.
- `scripts/extended_forcing_analysis.py` writes `runs/<run>_extforced/analysis.json`.

## Deviations

1. **2026-10-20, before any extended-budget answer existed: implementation
   clarifications from a CPU audit of the code against this plan.**
   - **§2, "the first 48 generated tokens equal the stored completion".**
     Implemented as: the new text begins with the stored completion's decoded
     text, with a trailing incomplete UTF-8 character removed. Only the text
     is stored, so token-level comparison is not possible.
   - **§4.5.** The agreement split by position relative to `i*` was missing
     its "at or after" half. It is now implemented for both halves.
     Agreement with the final answer is recomputed from the answers, not read
     from the stored `matches_final` flag. The fixture below showed that the
     flag is a single point of failure.
   - **§4, information audit under O256.** This was not implemented. Now:
     - `extended_forcing_analysis.py` writes `labels_O256.jsonl`;
     - `information_audit.py --extra-label` runs Q1 and the same-label probe
       and text contrasts (D81 deviation 1) under it, writing
       `information_O256.json`. Q2 is not part of §4.
   - **Verification, all on `main`, with fixtures in the scratchpad only:**

     | fixture | requirement | result |
     |---|---|---|
     | identity: extended = registered | O256 = O48 on both savings definitions, and O48 equals D79's registered analysis | all exact |
     | final answer: every cut-off answer = the final answer | O256 equals D79's independently computed *committed-to-final* scenario | B\* 0.6058 and every gap and CI exact; `f` = 1.0, the "commitment" reading |
     | identity labels through `--extra-label` | equals `main`'s stored D81 Q1 and post-hoc contrasts | all 12 exact |

     - Unit tests cover the net-saving formula, the frozen `f` boundaries
       (0.80 and 0.20), and the position split with deliberately wrong
       stored flags (`tests/test_extended_forcing.py`, 14 tests).
     - The forcing call was checked line by line against the traces stage's:
       the same prompt, suffix, greedy settings and seed derivation, with no
       stop strings.
     - The savings bases (`prefix_tokens`, `n_tokens`) are the registered
       ones.

