# Does verbalised content add stopping information at outcome-blind positions? (D83)

**Design for review.**
- **Stage 0** uses stored data only. It was frozen here before it was
  computed, and it runs on the CPU (exploratory).
- **Stage 1** is a new GPU experiment. **It is not run until this design has
  been reviewed.**

Changes are appended under *Deviations*, dated.

## 1. Why a new design is needed

Every NLA stopping result so far is measured on design A's *menu*: the
chunks the verbaliser was run on.
- The menu is built from the answer-stable tail start `i*`. The
  matched-position window ends at `i* − 1`, and the tail window starts at
  `i*` (`ast_detect.matched_windows`). So every problem's menu contains the
  last unsafe-looking chunk and the first safe one.
- Any rule evaluated there is handed the outcome's location (D79 §1.4).

What the stored data already say (D79, D81, exploratory):

| finding | estimate |
|---|---|
| the registered readout (reconstruction cosine) adds nothing beyond position and the probe | ΔAUC −0.001 [−0.002, −0.000] |
| the explanation **text** adds information beyond position and the raw trace text | +0.073 [0.058, 0.087] |
| … but none beyond a linear probe on the same activation trained for the same label | −0.001 [−0.004, 0.002] |
| a rule with no readout matches the readout under the registered estimand | Y1: −0.024 [−0.070, 0.015] |
| one NLA check (2 samples × 134 tokens) costs, against the mean 325-token trace | **0.83** of it |

These results leave two questions open:
1. **Do the menu results generalise to positions chosen without the tail?**
2. **What can cheap signals achieve there?** That is the ceiling any NLA
   rule must beat.

Stage 0 answers the second from stored data. Stage 1 answers the first.

**Cost alone settles one claim.** Each check spends 0.83 of a mean trace, so
an online rule that verbalises at every candidate point cannot save compute.
On this setup the readout's possible value is diagnostic or interpretive,
not efficiency. Stage 1 therefore asks an **information** question. It
reports compute-adjusted savings only to make that limit explicit.

## 2. Stage 0: cheap signals at every boundary (stored data; CPU)

**Population:** `main`'s 250 traces. Every chunk boundary is a unit, with no
menu and no use of `i*` in choosing units. `main_forced` completed the
registered forced answer at all 1,236 boundaries the scan skipped (D69), so
every boundary has its O48 label.

**Labels:**
- `Y_reg` (O48 safety, registered definition): primary;
- `Y_tail` (`i ≥ i*`): secondary; problems without a tail are excluded for
  this label only.

**Information models** (D81's estimation, unchanged):
- 10-fold `StratifiedGroupKFold` by problem, seed 20261020;
- a problem-cluster bootstrap, 2,000 replicates.

Models:
- `pos`: log prefix tokens, chunk index, MATH flag;
- `pos+probe`: the registered correctness probe (out-of-fold);
- `pos+A`: a layer-20 activation probe for the same label, nested-CV C
  (D81 deviation 1);
- `pos+R`: the last 600 characters of the prefix;
- `pos+A+R`.

**Primary Stage 0 contrast:** AUC(`pos+A`) − AUC(`pos`) on `Y_reg`. How much
safety information does the cheapest learned representation of the state
carry at outcome-blind positions?

**Stopping policies** (descriptive):
- each model's out-of-fold `P(safe)` becomes a score in the registered rule
  machinery (stop at the first boundary whose score reaches the threshold),
  over **all** boundaries (`menu = None`);
- compared with no signal, the rising and falling position rules, and the
  oracle bound B\*, at the D69 budget fractions of B\*;
- the gaps against no signal get the paired problem bootstrap (2,000,
  seed 20261019);
- the problem set is all 250. Registered and net savings are both reported
  (net charges the forced tokens, as in D82).

**What Stage 0 can show:**
- whether *any* cheap learned signal beats no signal without the menu;
- how far from the oracle it stays.

**What it cannot show:** anything about the NLA, which exists only on the
menu.

## 3. Stage 1: the NLA at outcome-blind positions (new GPU work; not run)

### 3.1 Hypothesis and estimand

**H₁ (primary):** at positions chosen without reference to the tail, the
verbalised text adds information about stopping safety beyond a linear probe
on the same layer-20 activation, trained for the same label, plus position.

**Δ₁ = AUC(`pos+A+E`) − AUC(`pos+A`)** on `Y_ext`:
- `Y_ext` is safety under the 256-token forced answer of D82, with the
  registered safety definition;
- the statistical unit is the problem;
- uncertainty comes from out-of-fold predictions and a problem-cluster
  bootstrap.

**Secondary, unadjusted:**
- E over `pos`;
- A over `pos`;
- readout cosine over `pos+A`;
- `pos+E` against `pos+A` head to head;
- all of the above on O48 and on `Y_tail`;
- the stopping policies of §3.6.

### 3.2 Population and candidate positions

**Problems:** the first **200** problems of `rep1`'s frozen interleaved order
with `ok` traces, whether or not they have a tail. Their traces, activations
and AST already exist. Using `rep1` costs no new traces, and its problems
are unseen by every model fitted so far.

**Positions:**
- for each problem, **K = 4** chunk boundaries drawn uniformly without
  replacement from `{0, …, n_chunks − 1}`, or all of them if there are fewer;
- the generator is seeded with `derive(20261020, "offmenu", problem_id)`;
- the draw reads only `n_chunks`. It never reads `i*`, a forced answer or a
  continuation. A test enforces this: the selection function receives only
  the problem id and the number of chunks.
- Positions are drawn before anything is generated, and their list is
  hashed into the decision record.

That is about 800 units.

### 3.3 Measurements at each unit

**Verbalisation:** the registered verbaliser settings (2 samples,
256-token cap, T = 1.0, top-p 0.95, the NLA seed derivation), then
reconstruction to give the cosine score. Integrity gate as registered.

**Forced answer:** the D82 instrument at 256 tokens, greedy, blocks of 2.
- O256 comes from the full completion.
- O48 comes from the same completion's first 48 generated tokens,
  re-extracted. Greedy decoding makes that the 48-token answer. This is
  checked on 50 units against a separate 48-token forcing, gate ≥ 0.95.
- The exact generated tokens are recorded.

**Activation:** layer 20 at the boundary, from `rep1`'s acts store. The
verbaliser reads the same row.

### 3.4 Outcome and handling of incomplete data

| case | handling |
|---|---|
| forced completion still cut off at 256 tokens | answer as extracted (registered rule); sensitivity: no answer (O256-u) |
| a verbalisation sample fails the integrity gate | the unit uses its remaining samples |
| a unit with no passing sample | `E` is empty and the unit is kept (its information is missing, not excluded); the count is reported |
| a problem whose trace failed | excluded before positions are drawn (none expected; `rep1` had 0 failures at 166 traces) |
| a unit with no forced answer | not possible: every unit is forced; a failure stops the run |

### 3.5 Models and decision rule

The classifiers are D81's: standardised L2 logistic, TF-IDF per text block,
everything fitted inside the fold. `A` is the nested-CV probe.

| reading of Δ₁ | rule |
|---|---|
| **increment** | two-sided bootstrap p < 0.05 and Δ₁ > 0 |
| **no practically relevant increment** | the 95% CI lies inside (−0.01, +0.01) |
| **probe better** | p < 0.05 and Δ₁ < 0 |
| **inconclusive** | otherwise |

The margin of 0.01 AUC is half of D81's, because the expected precision
allows it.

### 3.6 Stopping policies and compute

Each model's out-of-fold `P(safe)` is thresholded at the 4 candidate units,
taken in trace order: stop at the first unit whose score reaches the
threshold.

**Per problem, the policy records:**
- whether it stopped;
- whether the stop was safe;
- its net saving: `1 − (prefix + forced tokens) / total`.

For an NLA policy, the net saving also subtracts the **verbaliser tokens
generated at every candidate it visited**, in target-token units. The
verbaliser is the same 7B architecture. Reconstruction cost is reported
separately as prefill tokens.

**Reported:**
- break rate against mean net saving;
- the same with verbalisation charged;
- the gap to no signal at the D69 budget fractions.

These are descriptive. The decision in §3.5 is about information, not
deployment.

### 3.7 Power and sample size

The only variance estimate comes from D81 on the menu. There, Δ₁'s
bootstrap SE was about 0.0006 at 1,206 units.
- **Assumption:** off the menu, at 800 units and a different label balance,
  the SE is up to **5×** larger, about 0.003.
- **Then** the 95% CI half-width is about 0.006, inside the 0.01 margin, and
  a true Δ₁ of 0 reads *no practically relevant increment* with high
  probability.
- **Power** to detect a true Δ₁ of 0.01 at two-sided α = 0.05 is
  `Φ(0.01/0.003 − 1.96) ≈ 0.91`.

If the observed SE exceeds 0.005, the CI may straddle the margin, and the
result is reported as underpowered rather than extended post hoc.

### 3.8 Cost

| item | count | tokens |
|---|---|---|
| verbalisation | 800 units × 2 samples × ~134 | ~215k generated |
| reconstruction | 1,600 explanations | ~215k prefill |
| forcing at 256 | 800 units (most finish early) | at most 205k |

That is roughly 4–6 GPU-hours on this card at batch 2 (D78). Nothing is
downloaded.

### 3.9 When Stage 1 stays inconclusive

Stage 1 is inconclusive when:
- the CI straddles ±0.01 (underpowered);
- or the stability gate fails (instrument);
- or fewer than 600 units have both a passing verbalisation and a known
  outcome (attrition).

Then the result is reported as such, and N is not extended after seeing it.

### 3.10 Limitations, fixed now

- One model, one layer, one verbaliser.
- The text is featurised by TF-IDF. A stronger text model could extract more,
  but would add a second large model whose cost the comparison must then
  carry.
- The probe is linear. The verbaliser could in principle decode non-linear
  structure, which Δ₁ > 0 would detect.
- Positions are uniform over chunks, not over tokens.
- `Y_ext` still depends on a forcing budget, now 256 tokens.

## 4. Go / no-go

| | decision | the uncertainty it resolves |
|---|---|---|
| **Stage 0** | **go** (CPU, existing data) | how much a cheap learned signal can do without the menu |
| **Stage 1** | **conditional go**, after review | whether D81's central finding (text subsumed by the probe) holds off the menu, at a precision that can call it equivalent |

**Conditions for Stage 1:**
1. D75's confirmatory analysis and D82's `rep1` pass are complete.
2. The `rep1` information audit has run (D81, pre-specified), so that the
   menu result has its own replication first.
3. The researcher has reviewed this design.

**Not proposed:** a larger or multi-layer NLA run aimed at efficiency. §1's
cost fact means no outcome of it could show a compute saving on this
setup.

## Deviations

1. **2026-10-20, after Stage 0's policy results were seen: selective risk at
   matched coverage (post hoc).**

   **What Stage 0 showed:**
   - Under the registered estimand, the information-free falling-position
     rule beats every learned policy. That includes a same-label activation
     probe with a boundary AUC of 0.861.
   - With registered savings: falling − no-signal +0.073; probe − falling
     −0.032 [−0.083, 0.009].
   - With net savings: probe − falling −0.093 [−0.147, −0.043].
   - So D79's metric artefact is a property of the estimand, not of the
     menu. Matching on *mean* saving, with abstention counted safe, rewards
     concentrating savings in a few problems.

   **Added, as a complement that asks whether a rule picks good stopping
   points:**
   - **Coverage** is the share of problems a rule stops in before the end.
   - **Selective risk** is the share of those stops that are unsafe.
   - **Mean saving among stops** is reported beside it.
   - At coverage c ∈ {0.25, 0.50, 0.75}, each rule is taken at the threshold
     whose coverage is closest to c, within 0.03. No signal uses its closed
     form at the q whose expected coverage is c.
   - **Reported:** each rule's selective risk and its difference from no
     signal, with the paired problem bootstrap (2,000, seed 20261019).
   - **Run:** on Stage 0 (every boundary), and on design A's menu for `main`
     and `main_nla240` (D79's rules).
   - **Status:** post hoc everywhere it has been computed. It is specified
     here, before any outcome, for `rep1` and Stage 1, as secondary.

   **What it is not.** Selective risk ignores how much a stop saves. A rule
   can have low selective risk by stopping only at the very end. The two
   numbers are reported together and never ranked as one.
