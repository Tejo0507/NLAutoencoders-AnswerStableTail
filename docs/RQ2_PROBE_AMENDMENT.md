# Amendment A1 — RQ2's independent-probe control, operationalised

**Status: APPROVED 2026-10-17 by the researcher** — for the exploratory, CPU-only evaluation, subject to the limitations in §4, §8, §10 and §11. Labels are rule-derived and not independently human-validated. Approved code: `scripts/rq2_claim_probe.py` SHA-256 `80a7bc5537c8a9a6960e3fb2ad222a4dd09e0798ba12ac0c72050a6fbba2253b`. Decision record: D57.

Approval is recorded by changing the status line above to
`**Status: APPROVED <date> by the researcher**`; `scripts/rq2_claim_probe.py
--phase evaluate` refuses to run until it reads that line. Decision record: D56.

Researcher decisions incorporated 2026-10-17, before any evaluation:
**Decision 1** — leakage control option D (within-problem shuffle for
`answer_stated`, across-problem swap for `domain`, position-only comparator
for `answer_stated`; §5–§7, §9). **Decision 2** — no human annotation; the
claim→assertion rule is unvalidated (§4). **Decision 3** — `answer_stated`
retained provisionally, its positional predictability reported, no claim
beyond position unless the comparator is beaten (§6–§8). The registered RQ2
primary analysis (deletion vs paraphrase) is unchanged.

**Final for this revision** (researcher, 2026-10-17, before any evaluation):
the AI annotation is exploratory only and yields no κ (§4); the four-feature
position comparator is primary (§5, §7); the accuracy difference with a
strictly-positive lower bound of its 95% problem-level paired bootstrap
interval is the primary criterion, Brier secondary (§6–§7); F6's `np.resize`
concern is a separate pending audit (§5). None of these may be changed after
evaluation.

## 1. What is registered, and what is missing

`PROJECT_PLAN.md` §5 (unchanged since 5ebf6dd, 2026-10-06), claim-level
faithfulness:

> *Independent-probe control.* A probe trained on held-out activations, never
> on the verbalised text, scores whether the claim's content is decodable from
> the activation. Group-disjoint splits; leakage is tested explicitly by the
> falsification battery.

That fixes the **inputs** (activations; never verbalised text), the **split
policy** (group-disjoint) and the **leakage test**. It does not fix the
**target**: what a free-text claim's "content" is, as something a probe can
predict. Claims look like *"Final token "answer." closes the concluding
summary"* or *"Structured math problem … involving a chemical formula"*. Some
mapping from claim to a decodable label is needed, and §5 leaves it undefined.

What was implemented instead, and why it cannot stand in: `audit_faithfulness.py`
stores `independent_probe_score`, the **correctness probe's** output on the
whole activation, one value per explanation (1,413 on `main`). Every claim in
an explanation gets the same number, and it says whether the model is likely
right, not whether the claim's content is in the activation. It is not used by
any test. (A side finding: 518 of the 1,413 explanations are from train-split
problems, whose activations that probe was fitted on.)

## 2. Content attributes

An attribute is admissible only if (a) its ground truth is computable from the
trace text or the problem record **without the verbaliser**, and (b) claims
assert it often enough to evaluate. Chosen on 2026-10-17 from claim frequency
(`describe` phase), before any probe was fitted or any claim was scored:

| attribute | ground truth (text/metadata only) | eval claims asserting it |
|---|---|---|
| `answer_stated` | 1 iff the strict extractor recovers the final answer from some prefix up to and including the window's chunk | 561 under the revised rule (533 "stated", 28 "not yet"); was 427 (333 / 94) before the §10 revision |
| `domain` | the problem's subject from its id: GSM8K, MATH algebra, MATH counting & probability | 770 (463 / 232 / 75) — read as a *subject cue*, see §3 |
| `final_word` | last word of the prefix, numbers pooled | 436 — **described, not scored** |

`final_word` is dropped from scoring: the words claims quote as the "final
token" ("value", "answer", "equation", …) almost never end a training
boundary, so no probe has a class for them — 6 claims are in-vocabulary at the
top-20 vocabulary, 27 even at every word seen three times. That is a coverage
finding and is reported as one.

Claims asserting none of the attributes (2,195 of 3,779 eval claims under the
revised rule; 2,303 before it)
are outside the control and are counted, not dropped silently. **The control
speaks only to claims of these two kinds.**

## 3. The claim → assertion rule

Deterministic, in `rq2_claim_probe.py`. **`answer_stated` was revised on
2026-10-17 after the §10 audit** (option (a)); `domain` is unchanged.

**`answer_stated`** (`answer_stated_evidence`, `quote_scope_map`). The claim
text is never altered. Its position in the explanation is recovered with the
RQ2 splitter itself (`claim_start`; 6,007 of 6,007 claims re-split exactly),
and quote and expectation state are read from the **whole explanation**,
because the splitter cuts inside quotations and 603 of 3,779 eval claims start
with the closing quote or parenthesis of a quotation opened in the previous
claim. An *expectation scope* opens at an unquoted prediction phrase
("expecting", "anticipating", "predicting", "about to", "likely followed by",
"likely completing", "requiring", "next tokens", …) and closes at the next
unquoted sentence end. Each character of the claim is then one of:

| text | kind | evidence |
|---|---|---|
| unquoted, outside a scope | description | "closes the concluding …", "concluding summary/statement/…", "has stated the answer" → **stated** |
| quoted, outside a scope | already-written context | an answer statement ("final answer", "answer is", `\boxed`, "final result", "final value is") → **stated** |
| quoted, inside a scope | predicted alternative | an answer statement that is not a closing phrase → **not yet**; a closing/confirming phrase only ("This concludes the answer", "end of solution", "… confirmed") → **neutral**; both in one phrase → ambiguous |
| unquoted, inside a scope | expectation text | "final answer / final result / the answer" not next to a closing word → **not yet**; next to one → neutral |

Value: `1` with stated evidence and no not-yet evidence; `0` only when the
predicted alternatives bearing on the answer are all answer statements and
nothing says it was stated; **abstain** when stated and not-yet both appear,
when answer-statement and closing alternatives are mixed ("expecting "The
final answer is" or "This concludes the problem""), on an ambiguous predicted
phrase, on a closing-only expectation, with no answer cue, or when the
explanation's quotes do not balance (19 explanations). Each claim's reason is
recorded (`answer_stated_reason`). A closing phrase is never on its own read
as either value.

**`domain`** (`_DOMAIN_CUES`, unchanged): one subject's cues only — algebra:
"algebra", "polynomial", "quadratic", "inequality", "system of equations",
"solve for x"; counting & probability: "probability", "combinatorics",
"permutation", "counting", "dice", "coin", "arrangements"; GSM8K: "word
problem", "dollars", "$<n>", "price", "cost", "shopping", "everyday",
"real-world". Cues for two subjects → abstain (26 claims). **What it
measures:** 418 of the 463 GSM8K readings fire on money/price wording alone
and 45 on a problem-type cue; algebra readings fire on named algebraic
objects (algebra 90, quadratic 55, inequality 42, polynomial 28, …);
counting & probability mostly on "probab" (64 of 75). It is therefore read as
**the subject cue a claim names, mapped to a benchmark subject** — "this
claim talks about money/prices" counts toward GSM8K — and **not** as a
validated classification of what problem domain the claim asserts. Every
`domain` result is reported under that interpretation
(`ATTRIBUTE_INTERPRETATION`).

## 4. Validating the rule — not performed (Decision 2)

**Researcher decision 2026-10-17 (Decision 2): no human annotation.** The
rule in §3 is therefore **not validated**. Consequences, fixed here:

- The evaluation does not depend on, and does not compute, Cohen's κ. The
  κ ≥ 0.6 gate below is **not applied**; no attribute is kept or dropped on
  agreement grounds. `evaluate` scores both attributes and records
  `rule_validation.human_validation = "not performed"` in its output.
- Every claim-level quantity (M1–M4) is about **the rule's reading** of each
  claim. Outputs label claim assertions `rule-derived (lexicon), not
  human-validated` and ground truth `text-derived`; the two are never called
  validated labels. A rule error and a verbaliser error are indistinguishable
  in M1, and a rule error can move M2–M4.
- **One AI annotation pass was made** (researcher instruction, 2026-10-17),
  kept as a clearly labelled **exploratory artifact** only:
  Claude annotated the v3 sheet once, from the sheet and guide only, without
  viewing the rule's readings. It is stored as
  `runs/main_rq2probe/annotation_sheet_v3.ai_claude_not_independent.jsonl`
  with its provenance. It is **not human validation and not independent** —
  the same model designed the rule — so no κ or agreement statistic is
  computed from it, and it is not an input to `agreement` or `evaluate` (the
  agreement phase reads only a CSV sheet in the v3 schema). Its label counts
  (`answer_stated`: 46 × `1`, 6 × `0`, 98 × `NA`; `domain`: 38 GSM8K, 16
  algebra, 4 counting & probability, 92 `NA`) are descriptive.
- The sheet, guide and agreement code below are kept so that independent
  human labels can be added later. If they are, κ ≥ 0.6 per attribute is the
  threshold, and evaluation results must not have influenced the decision to
  annotate or the threshold.

The procedure as designed (not executed):

- **Sample design** (`draw_annotation_sample`, seed
  `derive(seed, "rq2_annotation_v2")`): 150 eval-split claims drawn without
  replacement from three strata defined by the rule's reading, 50 each:

  | stratum | population (eval claims) | drawn |
  |---|---|---|
  | rule asserts `answer_stated` | 427 | 50 |
  | rule asserts `domain`, not `answer_stated` | 745 | 50 |
  | rule asserts neither scored attribute | 2,607 | 50 |

  The third stratum is the only place a claim the rule *missed* can appear.
  A stratum smaller than its quota is taken whole and the shortfall recorded,
  never refilled from another. The strata over-represent asserting claims
  relative to the eval population; κ is computed on the sample as drawn,
  unweighted, and the design is stored with the sheet
  (`annotation_sheet.key.json`).
- **Rows the rule reads as asserting each attribute, in this sample:**
  `answer_stated` 50, `domain` 52 (2 claims in the first stratum also assert
  a domain). How many the annotator reads as asserting is known only after
  annotation; `--phase agreement` reports, per attribute, rows, rule-asserting,
  annotator-asserting, either, both, and both-with-the-same-value
  (`annotation_counts`), so a κ that rests on few asserted rows is visible.
- **Sheet v3** (`runs/main_rq2probe/annotation_sheet_v3.csv`, guide
  `docs/RQ2_ANNOTATION_GUIDE.md`): columns `row_id`, `claim_text` (verbatim,
  the unit labelled), `explanation_context` (the full explanation with the
  claim marked `[[ ]]`), `answer_stated`, `domain`, `notes`. It holds neither
  the rule's reading, the stratum, the window kind, nor the ground truth.
  The researcher (or a person the researcher names) records, per claim, what
  **the marked claim** asserts: `answer_stated` ∈ {`1`, `0`, `NA`}, `domain` ∈
  {`gsm8k`, `math-algebra`, `math-counting_and_probability`, `NA`}. `NA` is
  the explicit code for "asserts nothing / cannot tell"; an empty cell is
  "not annotated" and `read_annotations` rejects the sheet until every cell is
  filled.
- **Why the context column.** 71 of the 150 sampled claims are fragments cut
  inside a quotation by the claim splitter (they start with `")`, have
  unbalanced quotes, or are under 25 characters), so from the fragment alone
  an annotator often cannot tell whether a quote is context text or a
  predicted continuation. The guide restricts the context to resolving that;
  assertions made only by unmarked parts of the explanation are not carried
  over. The rule sees only `claim_text`, so κ measures the rule against a
  reading of the claim in its explanation — the stricter target, and the one
  the control needs.
- Rows of the sampled 150 the rule reads as asserting: `answer_stated` 50 (42
  as `1`, 8 as `0`), `domain` 52 (31 GSM8K, 14 algebra, 7 counting &
  probability). The `0` category of `answer_stated` rests on few rows; the
  per-attribute counts in the agreement output make that visible.
- `--phase agreement` computes Cohen's κ, rule vs annotation, per attribute,
  with "no assertion" as its own category. **An attribute with κ < 0.6 is
  dropped from the evaluation and reported as failing validation.** The
  threshold is fixed here, before the sheet is filled in, and is not revised
  after it.
- Two earlier sheets were replaced before any cell was filled in, and are
  kept: v1 (18:58, strata "any assertion" vs none, which counted the unscored
  `final_word` as an assertion; `annotation_sheet.v1_superseded_blank.csv`) and
  v2 (19:12, the current strata, but claim text only and blank meaning both
  "not annotated" and "no assertion"; `annotation_sheet.v2_superseded_blank.csv`,
  byte-identical). v3 has the same rows in the same order as v2.

## 5. Probe, data and splits

- **Existing data suffice; nothing is re-extracted.** Layer-20 activations at
  every chunk boundary are already stored (`runs/main/acts`).
- **Training:** every boundary activation of every **train-split** problem
  (1,132 vectors, 101 problems), labelled from the text as in §2. Never a
  verbalisation.
- **Evaluation:** claims from **eval-split** NLA windows only (3,779 claims,
  76 problems). `check_disjoint` asserts that no problem appears on both
  sides; the run fails if one does.
- **Probe:** logistic regression with the registered probe settings
  (`probe.C = 1.0`, `max_iter = 2000`, standardised), one per attribute;
  multinomial for `domain`.
- **Leakage test, per attribute (Decision 1, option D).** What is registered
  and what the project already does:
  - `PROJECT_PLAN.md` §6, F6: "Shuffle labels within group; probe must
    collapse to chance."
  - The project's F6 code (`run_falsification_tests.f6_probe_leakage`,
    `baselines/probe.permute_labels`) runs both scopes and treats
    *across_groups* as primary, because for its correctness labels the
    within-group shuffle is often degenerate (labels constant after the
    answer is stated). Its across-groups arm fits donor labels with
    `np.resize` (tiling/truncation); that affects F6 on `main`, not this
    control. It is a **separate pending audit**: F6 is not modified or rerun
    under this amendment.

  This control uses the registered within-group form where it is informative,
  and the across-problem swap where it is not:

  | attribute | shuffle | why |
  |---|---|---|
  | `answer_stated` | **within problem** (`within_problem_shuffle`) | F6's registered form. Each problem keeps exactly its own labels (prevalence exact, 0.089), but which boundary carries the `1` is randomised, breaking the label–activation and label–position links (relative-position AUC 0.997 → 0.453 on the training labels). Not degenerate here: 98 of 101 training problems have mixed labels; 190 labels move. |
  | `domain` | **across problems** (`permute_across_problems`) | Constant within a problem, so a within-problem shuffle is the identity. Each problem receives one other problem's value exactly; shares move by < 0.3 points. |

  Both report `n_labels_changed`, `fraction_changed` and the problems that had
  anything to move, as F6 does. The across-problem mechanics, kept for
  `domain` (`permute_across_problems`):
  - Donors are a **derangement** (`derangement`): a seeded single cycle through
    a shuffled order of the training problems, so it is a bijection — every
    problem donates to exactly one problem and receives from exactly one — and
    no problem keeps its own labels.
  - Problems have different numbers of boundaries (here 101 problems, 1,132
    boundaries). A donor's label sequence is fitted to the receiver's length
    by **relative position** (`resample_by_position`: receiver boundary *j* of
    *n* takes donor boundary round-half-up(*j*(*m*−1)/(*n*−1)) of *m*). That
    keeps `answer_stated`'s 0…0 1…1 structure and the donor's final label,
    and keeps class shares up to rounding; `domain`, one value per problem,
    transfers exactly. Because lengths are re-matched, a donor's labels can
    be repeated (longer receiver) or thinned (shorter receiver) — evenly along
    the trace, never truncated at the end or cycled back to the start. The
    class shares before and after permutation are reported
    (`train_label_shares`, `permuted_label_shares`), so any drift is visible.
  - An earlier draft copied donor labels by index and cycled them, and its
    self-map fix was not a bijection. That truncated long donors — dropping
    exactly the late boundaries where `answer_stated` = 1 — and would have
    biased the shuffled labels toward 0. Corrected before any evaluation.
  - Applied to `answer_stated`, this swap kept labels aligned with the
    receiver's position (AUC 0.996) and raised prevalence to 0.103 (§9);
    that is why `answer_stated` now uses the within-problem shuffle.
  - Criterion in §7.
- **Position-only comparator (`answer_stated` only; Decision 1).** A logistic
  model with the same settings as the probe, fitted on the same training rows
  and labels, whose only inputs are four position features of the boundary
  (`POSITION_FEATURES`): relative position in the trace, a final-boundary
  indicator, the boundary index, and log prefix tokens. No activation enters
  it. It is generous to position by design (it knows the trace length through
  the final-boundary indicator, which an activation at a non-final boundary
  cannot), so beating it is a conservative test. Fixed before evaluation.

## 6. Metrics

Per scored attribute. **Every interval resamples whole problems**
(`analysis.bootstrap_iterations` = 10,000, 95% percentile, `analysis.seed`):
claims are nested in explanations, explanations in samples and windows, and
windows in problems, and none of those levels is independent of the others,
so the problem is the resampling unit (`cluster_bootstrap_mean`,
`cluster_bootstrap_paired_diff`, `cluster_bootstrap_rate_diff`). Estimands are
row-weighted (per boundary for M0, per claim for M1–M4); a problem with more
claims weighs more in the estimate, and the interval reflects that. Each
output reports `n_rows` and `n_problems`.

- **M0 probe validity**, on every eval boundary (not only audited windows):
  probe accuracy, majority-class accuracy (majority class taken from the
  *training* labels), shuffled-label accuracy, and the two **paired**
  differences on the same boundaries — probe minus majority, shuffled minus
  majority — each with its problem-cluster interval.
- **M0-P position comparison** (`answer_stated`): position-model accuracy on
  the same eval boundaries, and the **paired** difference (activation-probe
  accuracy − position-model accuracy), problem-cluster interval (primary,
  decides §7). Secondary, descriptive only: the paired Brier-score difference
  (position − probe), and the eval boundaries' positional predictability
  (rank AUC of relative position; rate at final vs other boundaries). At the
  claim level, descriptive only: confirmation by the position model at each
  claim's window, and probe − position confirmation, paired, problems
  resampled.
- **Positional predictability, reported regardless of outcome (Decision 3).**
  On the training labels (label statistics, no model): prevalence 0.089;
  relative-position AUC 0.997; rate 0.970 at final boundaries vs 0.003
  elsewhere (`describe.json`). The eval-boundary figures are computed in
  `evaluate`.
- **M1 rule reading vs text truth** (`m1_rule_reading_matches_text_truth`):
  the rule-derived asserted value equals the text-derived ground truth. Not
  a validated claim-truth rate (§4); reported as context, not as the control.
- **M2 confirmation rate** (the control's quantity): the probe's predicted
  value at the claim's own activation equals the claim's asserted value; and
  the same at the mismatched activation.
- **M3 specificity**: per claim, confirmed at own activation minus confirmed
  at a **mismatched** activation (a different eval problem, seeded partner,
  same relative position); paired over claims, problems resampled. The
  mismatched activation belongs to another problem, so an M3 row depends on
  two problems; the interval resamples the claim's own problem and treats
  the partner as fixed, which is stated rather than modelled.
- **M4 relation to RQ2's classification**: M2 among reconstruction-dependent
  claims minus M2 among the rest, problems resampled.
- **Class coverage**, per attribute: training classes, eval classes, classes
  absent from eval, eval boundaries and problems per class.
- **Claims asserting a class absent from the evaluation** are scored apart
  from M2–M4: their count, and the rate at which the probe nevertheless
  predicts that class at their activations (a false-positive rate, with its
  interval).

## 7. How it is read (fixed before evaluation)

For each scored attribute (both are scored; §4 validation was not performed):

- **M0 criterion** (`m0_verdict`): the probe *decodes* an attribute only if
  the lower bound of the 95% problem-cluster interval for (probe accuracy −
  majority-class accuracy) is **strictly above 0**. Otherwise the control is
  **inconclusive** for that attribute and M2–M4 are reported but not read.
- **Leakage criterion**: if the lower bound of the same interval for
  (shuffled-label accuracy − majority-class accuracy) is above 0, the
  shuffled probe beats chance, leakage is flagged, and the attribute's control
  is **invalid** whatever M2–M4 show. (For `answer_stated` the shuffle is
  within-problem, so this criterion is no longer confounded with position.)
- **Position criterion, `answer_stated` only** (`position_verdict`): the
  probe carries information **beyond the position features** only if the
  lower bound of the 95% problem-level **paired** bootstrap interval for
  (probe accuracy − position-model accuracy), on the same eval boundaries, is
  **strictly above 0**. Accuracy difference is the primary metric; the Brier
  comparison (§6) is secondary and does not enter the criterion. The
  comparator is the four-feature model of §5 — relative position,
  final-boundary flag, boundary index, log prefix tokens — and is not replaced
  by a weaker one (e.g. relative position alone). The reading is then one
  of four, fixed in `answer_stated_reading`:
  1. leakage flagged → *invalid*;
  2. M0 not met → *inconclusive*;
  3. M0 met, position criterion not met → *decodable, but not
     distinguishable from boundary position*: the experiment **did not
     establish predictive value beyond these four features**. This does
     **not** show that the activations contain no such information — only
     that this test, against this comparator, did not detect it. No claim of
     answer-statement content beyond position may be made, and M2–M4 for
     `answer_stated` are read as possibly positional;
  4. both met → *decodable beyond the four position features used*.
  Even reading 4 is evidence about linear decodability beyond these
  features. **It does not establish semantic understanding** of whether an
  answer was stated, and must not be described as such.
- The criteria and features were fixed on 2026-10-17 before any probe or
  position model was fitted; no evaluation output may be used to change them.
- **`domain`** is evaluated on two classes, not three. The eval split's NLA
  windows contain GSM8K and MATH algebra only (D54); counting & probability has
  no eval instance. So M0 for `domain` measures two-class discrimination by a
  three-class probe, and nothing in this control is evidence that counting &
  probability content is decodable. Claims asserting counting & probability
  (75 in the eval set) are excluded from M2–M4; their probe "confirmations"
  are reported only as a false-positive rate.
- The control **corroborates** RQ2's reconstruction-dependence reading for an
  attribute only if M3's interval excludes 0 (positive) **and** M4's
  interval excludes 0 (positive).
- M3 positive with M4 not: the content is decodable but unrelated to which
  claims reconstruction depends on — reported as that, not as support.
- M3 not positive: the asserted content is not specifically decodable from
  the claim's activation — reported as the control **not** supporting the
  claims.

The control is reported beside RQ2, not added to the main BH family; within
it, the M3/M4 intervals for the scored attributes are read jointly and the
number of attributes is stated. Nothing here changes the registered RQ2 test
(deletion vs paraphrase).

## 8. Limits stated in advance

- Covers two content types out of what claims say. Under the revised rule,
  1,306 of 3,779 eval claims assert `answer_stated` or `domain` and 1,233
  (32.6%) are scored in M2–M4 (the 75 counting & probability `domain` claims
  are excluded, §7); 1,584 is the count asserting *any* attribute including
  the unscored `final_word`, and is not a coverage figure for the control.
- The eval split's NLA windows contain no counting-and-probability problem
  (D54), so `domain` is a two-class test on this corpus (§7); claims asserting
  counting & probability are outside M2–M4. Closing that gap needs RQ2 audits
  on counting-and-probability problems (the N = 240 NLA extension has the
  verbalisations, but `faithfulness` has not been run on it).
- `answer_stated` is rare at training boundaries (8.9%) and almost
  determined by position (§6); it is **retained provisionally** (Decision 3),
  not removed for being predictable from position, with M0 guarding against
  the base rate and M0-P against position.
- The rule is a lexicon and **unvalidated** (§4, Decision 2).

## 9. The shuffled-label control: audit and decision (2026-10-17)

**Resolved by Decision 1 (option D): implemented as described in §5–§7.** The
audit that led to it is kept below. Label statistics on the 1,132 training
boundaries (no probe fitted):

| | `answer_stated` prevalence | AUC, relative position → label |
|---|---|---|
| original labels | 0.089 | 0.997 |
| current control (derangement + positional resampling) | 0.103 | 0.996 |
| within-problem shuffle (illustrative seed) | 0.089 | 0.48 |
| row-level shuffle | 0.089 | 0.49 |
| **within-problem shuffle as implemented** (`analysis.seed`) | **0.089** | **0.453** |

`answer_stated` is almost exactly "this is the final boundary": 97.0% of final
boundaries are stated, 0.3% of the others, and 95 of 101 problems have exactly
one stated boundary.

**What the current control tests.** It moves labels between problems but
keeps them aligned with the receiver's relative position. For `domain` (one
value per problem) that is the intended null: labels independent of the
problem's activations. For `answer_stated` it is not: the shuffled labels
still say "the last boundary is stated", so a probe trained on them can learn
the end of the trace from position information in the activations and beat
the majority rate on held-out problems. Under §7 that would be read as
*leakage* and void the attribute, when it would be a position effect. The
8.9% → 10.3% drift has the same origin: a short donor's single final `1` is
copied onto several late boundaries of a longer receiver.

**The deeper issue it exposes.** No shuffle fixes the confound in the
attribute itself: a probe that "decodes `answer_stated`" may be decoding
"end of trace". Distinguishing the two needs a position comparator, not a
different permutation.

**Options.**

- *A. Keep the current control.* Interpretable only as a position baseline
  for `answer_stated`; mislabels a position effect as leakage under §7.
- *B. Attribute-specific permutation.* Within-problem shuffle for
  `answer_stated` — the form F6 registers ("shuffle labels within group"),
  exact prevalence per problem, position association removed — and the
  current between-problem derangement for `domain`, where a within-problem
  shuffle is the identity. Tests the intended null for both.
- *C. Row-level shuffle for both.* Simplest permutation null, exact
  prevalence; departs from F6's within-group form and mixes `domain` values
  inside a problem.
- *D. B plus a position-only comparator for `answer_stated`.* A logistic model
  on position features only (relative boundary index, final-boundary
  indicator), fitted on the same training rows; M0 for `answer_stated` would
  then require the activation probe to beat the position model (paired,
  problem-cluster interval, lower bound > 0) as well as the majority rate.
  This changes M0's criterion and must be fixed before evaluation.

**Recommendation at audit: D. Decided: D** (researcher, 2026-10-17). The
choice did not depend on any probe result (none existed). The comparator's
features were widened from the two listed under D to four (adding boundary
index and log prefix tokens) when it was specified, again before any fit.

## 10. Pre-approval validity audit (2026-10-17) — approval withheld

Read-only. No probe or position model fitted, no claim scored against truth,
no agreement statistic computed. Evidence is the rule's code and its behaviour
on the claim text.

**What M1–M4 compare (traced in `evaluate`).** M1: rule-derived assertion vs
text-derived truth. M2: activation-probe prediction vs **rule-derived
assertion** (not truth). M3: M2 at own minus mismatched activation. M4: M2
among reconstruction-dependent claims minus the rest. M0/M0-P: probe (and
position model) vs text-derived truth on eval boundaries — the only metrics
that do not depend on the rule.

**BLOCKING — the `answer_stated = 0` reading is systematically wrong.** Of the
94 eval claims the rule reads as "answer not yet stated", **71** reach that
reading because an expectation verb is followed by a *closing or confirming*
phrase containing an answer cue — e.g. `strongly expecting "This concludes
the answer"` — which implies the answer has already been concluded, the
opposite of `0`; in 28 an answer statement appears before the expectation
verb. A contributing fault: 603 of 3,779 eval claims are fragments starting
with a stray `"` or `)`, which shifts `_QUOTE`'s pairing so already-written
quoted text ("Thus, our final answer is …") is no longer recognised as
quoted context (59 of the 333 `answer_stated = 1` readings are such
fragments). Because M1–M4 are computed against the rule's reading, these
misreads would appear as verbaliser errors (M1) and as non-decodable content
(M2–M4) — rule errors indistinguishable from the findings the control exists
to make.

**Not blocking, recorded as interpretation limits.**

- `domain`: 376 of 463 GSM8K readings fire on "cost"/"price" alone. The rule
  operationalises "the claim names money or price content", which is
  correlated with, but not the same as, asserting the problem is a GSM8K word
  problem. Results must be described as agreement with lexical domain cues.
- M3 for `answer_stated`: the mismatched activation is taken at the same
  relative position, and `answer_stated` is almost determined by position, so
  M3 there tests specificity *beyond relative position*; a null M3 must not be
  read as "content not decodable".
- M2 for a correctly read, true claim is close to probe accuracy; M2 is
  agreement between probe and rule reading, not a claim-truth measure.

**Resolved 2026-10-17 by option (a)** (researcher decision): the rule was
revised as described in §3 and re-audited in §11. The paragraph below is the
audit's original wording.

**Smallest correction (needs a decision before approval).** Either (a) amend
the rule: an expectation of a closing/confirming phrase is not an expectation
of the answer (abstain, or read as `1` when the claim says the context closes
the concluding statement), and strip an unmatched leading quote/parenthesis
before quote pairing; re-run `describe` and this audit; or (b) restrict the
evaluation to `domain`, reporting `answer_stated` as not evaluable under the
current rule; or (c) score `answer_stated` only on claims read as `1`, with
M2–M4 reported as one-class. Each is decided on claim text and rule
mechanics only, before any outcome exists.

## 11. Fresh audit after the rule revision (2026-10-17) — no blocking issue; awaiting approval

Read-only; no probe or position model fitted, no claim compared with truth,
no agreement statistic computed. Labels remain **rule-derived and not
independently human-validated**.

**Effect of the revision on `answer_stated` readings** (3,779 eval claims):

| | `1` | `0` | abstain |
|---|---|---|---|
| before | 333 | 94 | 3,352 |
| after | 533 (76 problems) | 28 (21 problems) | 3,218 |

- The 94 former `0`s: 72 now abstain, 10 read `1`, 12 remain `0`. Of the 71
  driven by a closing/confirming expectation: 61 abstain, 9 read `1`, 1 stays
  `0`. Of the 28 with an answer statement before the expectation verb: 24
  abstain, 4 read `1`.
- The 59 former `1`s that were fragments: 52 stay `1`, 5 abstain, 2 read `0`.
  All 603 fragment claims: 508 abstain, 91 read `1`, 4 read `0`.
- Abstention reasons: no answer cue 2,961; mixed answer/closing alternatives
  99; unbalanced quotes 68; ambiguous predicted phrase 50; closing-only 27;
  stated and expected both 13.
- Every remaining `0` is a claim whose predicted alternatives are all answer
  statements ("expecting "Calculate the final result"", "…"Therefore, the
  final answer is""). Whether each reading is correct cannot be verified
  without independent labels.

**Positional predictability, rechecked.** The training labels are text-derived
and unaffected by the rule: prevalence 0.089, relative-position AUC 0.997,
0.970 stated at final boundaries vs 0.003 elsewhere. The revision changes the
claim side: claims read `1` sit at the final boundary in 85.2% of cases,
claims read `0` in 7.1%, abstentions in 12.4%. So rule-derived `answer_stated`
claims are themselves largely positional, and the attribute is close to
one-class (533 vs 28). Consequences for interpretation: M2–M4 for
`answer_stated` mostly measure whether the probe recognises final-boundary
activations at windows whose claims the rule reads as "stated"; only M0-P
can separate content from position, and M2–M4 for the `0` class rest on 28
claims from 21 problems.

**Fresh checks.**

| check | result | evidence |
|---|---|---|
| A. evaluation gate; no prior evaluation | PASS | `amendment_approved()` False, no gate-matching line; no evaluation output in `runs/main_rq2probe/` |
| B. train/eval separation | PASS | 101 train vs 76 eval problems, overlap 0; splits consistent with `problems.jsonl` |
| B. preprocessing / model selection | PASS | probe, scaler, majority class and position model fitted on training rows only; eval labels used only for scoring; no tuning |
| B. uncovered leakage risk | UNRESOLVED (minor) | M3 partner activation from another eval problem; interval treats it as fixed |
| C. per-attribute shuffles | PASS | `SHUFFLE_SCOPE`: within-problem for `answer_stated`, across-problem for `domain` |
| D. paired problem-level bootstrap, accuracy criterion, lower bound > 0 | PASS | `cluster_bootstrap_paired_diff`, `position_verdict` |
| D. Brier secondary | PASS | verdict reads only the accuracy interval (tested) |
| M1–M4 dependence on rule-derived labels | PASS (documented) / UNRESOLVED (unverifiable) | M1 rule vs text truth; M2–M4 probe vs rule reading; correctness of readings unverifiable without human labels |
| registered RQ2 primary analysis unchanged | PASS | no diff to `audit_faithfulness.py`, `faithfulness/`, `aggregate_results.py`, `analysis/` |
| F6 unchanged, audit separate | PASS | no diff to `run_falsification_tests.py`, `baselines/probe.py` |
| CPU-only | PASS | no torch/CUDA reference; numpy, scipy, sklearn |

**Remaining limitations (not blocking):** unvalidated rule (Decision 2);
`domain` read as a subject cue (§3); `answer_stated` close to one-class and
positional (above); M3 for `answer_stated` taken at the same relative position
(§10); the v3 annotation sheet's strata were drawn under the pre-revision
rule, which matters only if human annotation is added later.

**Status: still PROPOSED.** No blocking issue was found; approval is the
researcher's. *(Approved later the same day: see the status line and D57.)*

## 12. Result (exploratory) — 2026-10-17

Run once, after approval, with the audited code (SHA-256 `80a7bc55…253b`):
`CUDA_VISIBLE_DEVICES="" python scripts/rq2_claim_probe.py --config main
--phase evaluate`, exit 0, CPU only. (The empty `CUDA_VISIBLE_DEVICES` does
**not** hide the GPU in this machine's Git Bash — torch still reports CUDA
available; `-1` would. CPU-only rests instead on the code path: the
evaluation never imports torch, checked by loading, assembling and fitting
along it and inspecting `sys.modules`.) Output `runs/main_rq2probe/evaluation.json`
(SHA-256 `39e4b5e3…855`). Read strictly by §7; no criterion changed.
Intervals are 95% problem-cluster percentile bootstraps (10,000 resamples).

**What this result rests on.** Claim assertions come from an **unvalidated
lexicon rule; no independent human validation was performed** (Decision 2).
A rule error cannot be told apart from a verbaliser error in M1–M4.
Decodability, where found, is linear decodability by a probe; it establishes
neither reconstruction dependence nor semantic understanding.

| | `answer_stated` | `domain` (subject cue) |
|---|---|---|
| eval boundaries (M0) | 788 (76 problems; 712 × `0`, 76 × `1`) | 788 (GSM8K 419 / algebra 369; 43 / 33 problems) |
| classes absent from eval | none | counting & probability |
| claims scored, M2–M4 | 561 (533 × `1`, 28 × `0`; 76 problems) | 695 (71 problems); 75 asserting the absent class kept apart |
| probe − majority accuracy (M0) | +0.095 [0.087, 0.104] → decodes | +0.406 [0.292, 0.515] → decodes |
| shuffled − majority (leakage) | −0.028 [−0.042, −0.015] (within-problem) → none | +0.047 [−0.030, 0.124] (across-problem) → none flagged |
| position model accuracy | 1.000 [1.000, 1.000] | — |
| **probe − position model (M0-P)** | **−0.0013 [−0.0040, 0.000] → not beaten** | — |
| Brier, position − probe (secondary) | −0.0008 [−0.0037, 0.0008] | — |
| M1 rule reading = text truth | 0.856 [0.814, 0.898] | 0.909 [0.842, 0.960] |
| M2 own / mismatched | 0.856 / 0.865 | 0.901 [0.832, 0.953] / 0.532 [0.386, 0.673] |
| **M3 own − mismatched** | −0.009 [−0.023, 0.000] | **+0.368 [0.224, 0.522]** |
| **M4 dependent − not** | **−0.103 [−0.174, −0.027]** | −0.034 [−0.089, 0.028] |
| absent-class false-positive rate | — | 0.067 [0.000, 0.188] (75 claims, 24 problems) |

**Conclusion.** The independent-probe control **did not corroborate** the
reconstruction-dependence reading of RQ2 for either attribute (§7 requires
M3 > 0 and M4 > 0; neither attribute meets both).

- `answer_stated` is highly positional: on the eval boundaries its truth is
  exactly "final boundary" (AUC 1.0), the four-feature position model is
  perfect, and the activation probe did **not** demonstrate predictive value
  beyond it. That does not show the activations lack such information. M3
  is uninformative for this attribute because the mismatched activation is
  taken at the same relative position (§10). M4 is negative: claims that
  reconstruction depends on are confirmed less often — observed, but on
  unvalidated, near-one-class (28 `0` claims, 21 problems), positional labels,
  and not a finding about content.
- `domain` is a rule-derived **subject cue** (418 of 463 GSM8K readings are
  money/price wording) tested as **GSM8K vs MATH algebra only**; nothing is
  shown for counting & probability, and it is not a validated domain
  classifier. The cued subject is specifically decodable from the claim's
  own activation (M3), but unrelated to which claims reconstruction depends on
  (M4 interval includes 0).

**Limits that remain.** Coverage: 1,233 of 3,779 eval claims (32.6%) scored
in M2–M4. M3's mismatched partner is treated as fixed in its interval. The
`domain` shuffled-label point estimate is positive (+0.047) though its
interval includes 0. The registered RQ2 primary analysis (deletion vs
paraphrase: 6,007 claims, Cohen's d 0.712, mean paired difference 0.0317
[0.0306, 0.0329]) is unchanged — its artefacts are byte-identical to the
pre-experiment backup.
