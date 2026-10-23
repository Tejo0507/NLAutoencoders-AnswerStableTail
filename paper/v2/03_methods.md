# 3. Methods (draft v3, 2026-10-23)

*Written from the code and the frozen plans. Each paragraph names its
source so that it can be checked. Registered methods are described as
registered; later additions carry their decision number.*

## 3.1 Model, data and traces [R]

- **Target model:** Qwen2.5-7B-Instruct [39], 4-bit NF4 [40]
  (`configs/base.yaml`), with greedy traces of up to 768 tokens.
- **Problems:** GSM8K [3] and MATH [4] (algebra; counting and probability).
  - `main`: 250 problems (150 GSM8K, 100 MATH).
  - The NLA arm: 120 problems, and an exploratory 240-problem arm (D54).
  - The replication (`rep1`): 400 problems no earlier run used (240 GSM8K,
    160 MATH), chosen in the registered hash order
    (`scripts/prepare_fresh_problems.py`, D75).
- **Chunking:** traces are split into chunks at reasoning-step boundaries.
  A boundary `s` gives the prefix `P_s`.

## 3.2 The answer-stable tail and the safety outcome [R]

- **Evidence at boundaries:** at each boundary the AST scan evaluates, K = 5
  continuations (T = 0.8, 200 tokens) are sampled, and an answer is forced
  from `P_s` with the suffix "Therefore, the answer is". The forcing is
  greedy, with **48 new tokens** (`configs/main.yaml`).
- **The tail start `i*`:** the first boundary from which the answer is
  stable (PROJECT_PLAN §5; `ast_detect/detector.py`).
- **The registered outcome (O48):** a stop at `s` is *safe* if the forced
  answer's correctness equals the final answer's (D69,
  `reanalyse_o3_registered.problems_for`).
- **Completion:** forced answers at boundaries the scan never reached were
  completed with the same call (`force_all_boundaries.py`, D69).
- **Instrument check:** re-forced stored boundaries must agree at ≥ 0.95.

## 3.3 Signals

- **NLA readout [R]:** at each verbalised boundary, 2 explanations (256-token
  cap) of the layer-20 activation. The registered score is the mean
  reconstruction cosine of the samples that pass the integrity gate.
  Explanations of Gaussian vectors serve as a noise control (`main` only).
- **Probes:**
  - *Registered probe [R]:* logistic regression on layer-20 states,
    predicting whether the intermediate parsed answer is correct, out of
    fold.
  - *Same-label activation probe [P, D81 dev. 1]:* a linear probe on the
    *same* layer-20 activation, trained for the label being predicted. C is
    chosen by inner grouped CV, and the probe is stacked as one feature.
- **Position rules [X, D79]:** rising and falling in prefix tokens.
- **No signal [R]:** i.i.d. uniform scores, with expectations in closed form.
- **Text features [X, D81]:** TF-IDF (1–2-grams) of the explanations, or of
  the last 600 characters (or all) of the raw prefix.

## 3.4 Candidate points and the leak

**Design A** (D69) lets every rule stop only at the verbalised chunks, the
*menu*. The menu holds windows of three kinds: tail, matched-position and
matched-length. The matched-position window ends at `i* − 1`, and the tail
window starts at `i*` (`ast_detect.matched_windows`,
`run_autoencoder.select_windows`). **So every menu contains the chunks on
both sides of the outcome change.**

**Removing the leak:**
- **Stage 0 [X, D83]** uses every chunk boundary as a candidate. That is
  possible only for signals that exist everywhere.
- **Stage 1 [S, D83, approved as revised in D85; run D96]** verbalises
  K = 4 boundaries in each of rep1's 400 problems (1,600 units). They are
  drawn uniformly without replacement by a seeded function that receives
  only the problem id and the number of chunks. It never reads `i*`, a
  forced answer or a continuation. Every unit is forced with 256 tokens.

## 3.5 Estimands

1. **Registered matched-saving estimand [R; D69].**
   - Rules: "stop at the first allowed boundary whose score reaches t".
   - Comparison: safe rate at matched *mean* saving, `1 − prefix/total`.
   - Budgets: {0.1, …, 0.9} × the oracle bound B\*; tolerance 0.02.
   - Uncertainty: a paired problem bootstrap (2,000), Holm [41].
   - The D75 replication applies it, with a margin of 0.05 and fixed
     readings.
   - A rule that never stops counts as safe.
2. **Information [X; D81].**
   - Out-of-fold AUC from 10-fold, problem-grouped, outcome-stratified CV.
   - Contrasts are AUC(a) − AUC(b), with a problem-cluster bootstrap
     (exact weighted AUC).
   - Readings use a ±0.02 margin (±0.01 in Stage 1, with readings in the
     style of two one-sided tests [42]).
   - **Stage 1's primary contrast [S]** is Δ₁ = AUC(position + same-label
     probe + text) − AUC(position + same-label probe), on `Y_ext`: safety
     under the 256-token forced answer, with the registered definition.
     Its reading comes from the 95% interval against ±0.01. `Y_48` and
     `Y_tail` (tail membership) are secondary.
3. **Selective risk at matched coverage [P; D83 dev. 1].** Among problems
   where a rule stops before the end, the unsafe share of stops, at
   coverage 0.25 / 0.5 / 0.75, reported with the saving among stops.
4. **Net saving [D82].** `1 − (prefix + forced tokens)/total`, which
   charges the answer emission.

## 3.6 The forcing-budget measurement [S on rep1, X on the 240 arm; D82, run D95–D96]

- **Re-forcing:** every cut-off menu boundary is re-forced with the same
  call at 256 tokens, and the exact tokens are counted. A stability gate
  uses 50 finished boundaries.
- **Outcomes:** O48 (authoritative), O256, O256-u (still cut off means no
  answer) and the continuation majority (diagnostic).
- **The `f` statistic:** the share of pre-tail cut-offs whose 256-token
  answer equals the final answer. It decides the wording: ≥ 0.80
  "commitment within 48 tokens"; ≤ 0.20 "registered reading supported";
  otherwise "mixed".
- **Token counts:** a check compares the recorded forced-token counts with
  re-tokenised ones (±2). Where it fails, as it did on both arms (D95,
  D96), net savings are not used and every comparison uses registered
  savings.
- **O256 never replaces O48.** The D75 verdict is computed on O48 alone.

## 3.7 Pre-registration and deviations

- Every analysis has a plan hashed before it was computed. The hashes are
  in `docs/DECISIONS.md` (D69–D88); `.gitattributes` keeps the plans' bytes
  stable on checkout.
- Deviations are appended to each plan with their date.
- Post-hoc additions are marked [P] and are pre-specified for the
  replication.
