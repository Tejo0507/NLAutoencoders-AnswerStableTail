# Position-only control and forced-answer truncation: analysis plan (D79)

**Written 2026-10-20, before any position-rule result was computed on any
run, and before any `rep1` outcome exists.** D79 records this file's SHA-256.
`scripts/position_truncation_audit.py` records the hash it ran under. Later
changes are appended below as dated deviations and are never applied silently.

**Status of what this plan produces.**
- On `main` (120 problems) and `main_nla240` (240): **exploratory**. The
  readout's, the probe's and the no-signal rule's design-A results there have
  been seen (D69, D73). The position rules' results have not.
- On `rep1`: a **secondary analysis specified after the D75 freeze and before
  any outcome**. It does not enter, change or qualify the confirmatory verdict
  (`confirmatory_verdict`, `docs/REPLICATION_PLAN.md`), which is computed
  first and reported as registered.

## Part A. Does position alone explain the readout's design-A result?

### A.1 Why

Design A gives every rule the same menu `S_p`: the verbalised chunks. By
construction (`detector.matched_windows`, `run_autoencoder.select_windows`)
that menu is `{i*−L, i*−1}` (matched position), `{i*, n−1}` (tail) and two
matched-length chunks, with `L = n − i*`. The structural part is symmetric
about the tail start `i*`. A stop at or after `i*` is safe under the
registered outcome by criterion 1 and persistence wherever the AST scan
evaluated it (elsewhere the answer was forced afterwards, which reproduces at
0.97). So **within the menu, later is nearly always safer**, and any score that
rises with position can select safer stops.

The no-signal rule cannot exploit this. At qualification probability `q` it
stops at the `k`-th menu chunk with probability `q(1−q)^k`, which favours
early (unsafe) chunks. "Beats the no-signal rule" in design A is therefore a
weak bar. The readout's +0.052 over it (D69) could be a position effect if
reconstruction cosine rises with position. This part tests that.

### A.2 Rules

Every rule is the registered form: stop at the first menu chunk whose score
reaches the threshold, with the threshold swept over all observed scores. The
menu, outcome, budgets, tolerance and bootstrap are D69 design A's, unchanged.

| rule | score at menu chunk `i` | information it uses | status |
|---|---|---|---|
| `position_tokens` | tokens generated up to the boundary (`prefix_tokens`) | what an online rule has at `i`, plus the shared menu | **the control**: a fair cheap-signal control under design A |
| `position_index` | chunk index `i` | same | secondary |
| `relative_position` | `prefix_tokens / total_tokens` | the trace's final length, which no online rule has | **diagnostic only**: measures how much the menu's geometry gives away; never a fair control |
| `readout_residualised` | readout score minus its prediction from `[1, log prefix_tokens]`, by OLS over the menu chunks of all *other* problems (leave one problem out) | the readout, with its linear-in-log-position trend removed; no outcome is used | tests content beyond position |

The readout, probe and no-signal rules are computed exactly as in D69. Their
three registered design-A comparisons must reproduce the stored values (gap,
interval and p-value to 1e-9) from the same seed before anything new is
reported. All rules share the same bootstrap replicates (paired).

### A.3 Comparisons

**Family X**, Holm at α = 0.05, same estimand (mean safe-rate gap over the
matched budgets, at least 3 of them):

| | comparison | question |
|---|---|---|
| X1 | readout − `position_tokens` | does the readout beat position? |
| X2 | `position_tokens` − no-signal | does position alone beat the no-signal bar? |
| X3 | probe − `position_tokens` | does the probe beat position? |
| X4 | `readout_residualised` − no-signal | is there stopping information in the readout beyond its position trend? |

**Descriptive only, no decision:**
- readout − `relative_position`;
- `relative_position` − no-signal;
- `position_index` − no-signal;
- readout − `position_index`;
- `readout_residualised` − `position_tokens`;
- `readout_residualised` − probe;
- within-problem Spearman correlation of the readout score, and of the probe
  score, with `prefix_tokens` over menu chunks (outcome-free).

### A.4 Readings, fixed now

- **The readout carries stopping information beyond position** iff X1 and X4
  are both Holm-significant with Δ > 0.
- **An advantage of 0.05 or more over position is excluded** iff X1's upper
  95% bound is below 0.05.
- **"Beats no-signal" is not evidence of content beyond position** if X2 is
  Holm-significant with Δ > 0. This qualifies D69's F2 result. It does not
  replace it.
- Otherwise: **unresolved**. A non-significant X1 is not equivalence. A
  significant X1 says nothing about which part of the readout (text,
  reconstruction, or something else) carries the signal.

## Part B. Forced-answer truncation

The registered outcome is the answer forced from `P_s` within 48 tokens.
When the model starts a derivation instead of committing, the budget cuts it
off and the permissive extractor takes the last number written (D68,
`forced_was_cut_off`). At 0.5·B\* on `main`, 28% of the readout's stops land
on such completions, against 15% for the probe (D69). The direction of that
asymmetry was not determined.

### B.1 Measured, descriptive

On every run analysed:
- traces that reach the 768-token cap; final answers that do not parse;
- forced completions that are cut off (`forced_finished` false) or parse to
  no answer, by dataset, by source (stored or completed), by position
  relative to `i*` (before / at or after) and by window kind;
- AST continuations (200-token cap) whose answer does not parse.

### B.2 Measured disagreement

At stored, evaluated boundaries with K continuations:
- the rate at which the forced answer disagrees with the continuations'
  strict majority (≥ 3 of 5 equivalent), split by forced cut-off vs finished;
- the rate at which the forced answer disagrees with the parsed answer, by
  the same split.

Continuations are also capped (200 tokens), so the majority is a
comparison, not ground truth.

### B.3 Where each rule stops

For every rule of Parts A and D69, at every budget: the share of its stops
on a cut-off forced completion. This extends D69's single-budget figure.

### B.4 Sensitivity: two extreme readings of a cut-off completion

The registered outcome stays primary whatever these give. Each scenario
relabels *only* boundaries whose forced completion was cut off. It then
recomputes B\*, the budget grid and every Part A / D69 comparison:

- **uncommitted:** the forced answer is taken to be none, so the rule walks
  away with no answer;
- **committed to the final answer:** the forced answer is taken to be the
  trace's final answer.

These are extremes, not bounds: a gap between two rules is not monotone in
the labels. Bootstrap: 2,000 replicates, same seed.

**Reported:** for each scenario, each comparison's gap and interval, and
whether any D69 §7 reading, X-family reading or (on `rep1`) D75 reading
differs from the registered-outcome one. A conclusion that flips under
either scenario is reported as **dependent on the forcing budget**.

## What `rep1` needs that does not exist yet

All of it comes from the D75 pipeline. Nothing extra is generated:
- the menu-only forced completion;
- the probe scores;
- the reconstructions.

The script refuses to run on a run whose forced-completion file does not
match its traces (the same check as D69).

## Deviations after the hash

1. **2026-10-20, during implementation, before any result was computed.**
   One descriptive addition and one convention:
   - **Added to B.2:** the instrument's own reproducibility check (stored
     boundaries re-forced by `force_all_boundaries.py`), split by whether the
     stored completion was cut off. It measures whether cut-off answers
     reproduce less well.
   - **Convention for B.3:** a stop whose cut-off status was not recorded is
     counted separately and excluded from the share. D69's figure put it in
     the denominator. The script recomputes D69's figure under D69's
     convention and refuses to continue unless it reproduces. Where every
     status is recorded, the two conventions coincide.
2. **2026-10-20, from a synthetic test, before any result on real data.** The
   rationale in §A.1 overstates the case. It says any score that rises with
   position can beat the no-signal rule. It need not: the no-signal rule's
   abstentions are safe, and a single position threshold stops late-tail
   problems before their tail. In a synthetic world with varying tail starts,
   position loses to no signal
   (`test_position_need_not_beat_no_signal_when_tail_starts_vary`). Whether
   it beats no signal on real data is exactly what X2 tests. No rule,
   comparison or reading changes.
3. **2026-10-20, after the plan's analysis had run on `main`, before any
   `rep1` outcome. Post hoc on `main` and `main_nla240`; specified in advance
   for `rep1`.**
   - **What `main` showed:**
     - the readout's score *falls* with position within a problem (median
       ρ = −0.37);
     - its advantage over no signal does not survive removing its
       log-position trend (X4).
   - **So family Y is added**, Holm within Y at α = 0.05. Family X and its
     readings are unchanged. The new rule, `position_tokens_falling`, has
     score −`prefix_tokens`, the same information as `position_tokens`.

     | | comparison |
     |---|---|
     | Y1 | readout − `position_tokens_falling` |
     | Y2 | `position_tokens_falling` − no-signal |

   - **Readings:**
     - *a falling-position rule reproduces the readout's advantage over no
       signal* iff Y2 is Holm-significant with Δ > 0 and Y1's 95% interval
       contains 0;
     - *the readout beats the falling-position rule* iff Y1 is
       Holm-significant with Δ > 0;
     - otherwise *unresolved*.
   - **Two descriptive additions to B.2:**
     - disagreement with the continuations' majority, split by cut-off status
       *and* by position relative to `i*`, because both cut-off and
       disagreement concentrate before the tail;
     - how often the continuations' majority equals the final answer, by
       cut-off status. That indicates which extreme reading of B.4 is the
       more plausible.
