# Claim validation: what is checked objectively, and a protocol for the rest (D71)

**Status.**
- **Part A** has been run: objective checks of three claim types against the
  stored trace (`scripts/validate_claim_semantics.py`, results below).
- **Part B** is a protocol plus tooling for human annotation. It has **not**
  been run: no independent annotator exists (PROJECT_PLAN §8). Nothing in this
  document is a human judgement, and no LLM output is used as ground truth.

## The five properties, kept apart

| property | question | how it can be tested here | status |
|---|---|---|---|
| P1 understandable | can a reader say what the claim asserts? | human rating (B) | protocol only |
| P2 supported by the evidence | is what it says about the text *there*? | final-token and quote checks against the trace (A); human rating for paraphrased content (B) | **A done** for named tokens and quotes |
| P3 accurate about the activation | does it say something true about what this activation encodes? | expectations against the model's actual next output (A), as one observable consequence of what the activation encodes; probes (A1, D56) | **A done** for predictions |
| P4 faithful to the model's behaviour or computation | does the model's processing actually depend on what the claim says? | needs intervention on the claimed content; nothing here does that | **untested** |
| P5 unjustified mechanism | does it assert mechanism the evidence cannot support? | a lexicon screen (A); human rating (B) | screen done: 0 of 1,413 hits, a narrow lexicon |

Reconstruction dependence (RQ2) bears on none of P1–P5 directly. It shows
that the reconstructor uses a claim's text.

## Part A: objective checks (run 2026-10-19)

All 1,413 activation-driven explanations of the 120-problem NLA arm that
passed the integrity gate were checked, with no selection. Each check is
paired against three controls: a mismatched trace, a shifted position, and
the Gaussian-control explanation. Intervals are problem-cluster bootstraps of
real − control. "Specific" means the real rate exceeds all three controls
(D71).

| claim | real | mismatched trace | shifted position | Gaussian | specific? |
|---|---|---|---|---|---|
| final token: the quoted word ends the context (`exact_suffix`) | **0.021** | 0.005 | 0.001 | 0.000 | yes, but rare |
| final token: right last character | 0.733 | 0.510 | 0.558 | 0.032 | yes |
| context quote (≥ 4 words) found verbatim, support ≥ 0.8 | **0.003** | 0.002 | 0.005 | 0.000 | **no** |
| context quote: mean contiguous support | 0.186 | 0.155 | 0.167 | 0.116 | yes, weakly |
| expectation: first word = model's actual next word | **0.522** | 0.101 | 0.078 | 0.034 | **yes** |
| expectation: first word among the next 10 words | 0.718 | 0.350 | 0.321 | 0.316 | yes |

Pairs below are real − control, with 95% CIs.

- `expect_first_word`: +0.421 [0.376, 0.465] against the mismatched trace,
  +0.445 [0.389, 0.501] against the shifted position, and +0.516 [0.464,
  0.565] against the Gaussian control.
- `final_exact_suffix`: +0.016 [0.009, 0.025], +0.020 [0.011, 0.030] and
  +0.020 [0.010, 0.030] respectively.

What these show:

- **Claims about what comes next are accurate well above chance.** On 52% of
  explanations, the first word of a stated expectation is the word the model
  actually wrote next. Controls give 3–10%. This is the first validated
  content-level evidence in the study that the verbalisations carry
  position-specific information about the activation (P3, behavioural). The
  rate is higher in tail windows (0.618) than in matched-position (0.433) or
  matched-length (0.568) windows. That is descriptive and untested.
- **Claims that name the final token get its form right and its word wrong.**
  The last character, usually punctuation or a newline, is right 73% of the
  time. The word is right 2% of the time. Typical: the context ends
  `dollars} \).\n\n`, and the claim says `"method).\n\n"`.
- **Quoted "context" is essentially never verbatim** (0.3%). Text presented
  as a quotation in these explanations should be read as a description, not
  as evidence of what the trace says.
- **No explicit mechanistic assertion** matched the screen. The lexicon is
  narrow, so this is not a validated absence.

Limits:
- Part A covers three claim templates only.
- A matching quote or word says nothing about the interpretation attached to
  it.
- Predictions agreeing with output is behavioural agreement, not faithfulness
  of the computation.
- The window positions come from the AST, so all three window kinds sit late
  in the trace.

## Part B: human annotation protocol (not run)

**Units.** Claims, as split by `faithfulness/claims.split_claims`, shown with
the problem, the trace up to the activation's token, and the 200 characters
after it.

**Sampling.**
- Draw 60 explanations by seeded random sampling, stratified by window kind
  (tail, matched-position, matched-length) and dataset.
- Split by **problem** into development (20 explanations, used to train the
  annotators and refine the guide) and held-out (40, scored once, after the
  guide is frozen by hash).
- Mix in, blind, 10 Gaussian-control explanations and 10 explanations shown
  against a mismatched trace. These are the controls: a scheme that rates them
  as supported is not measuring support.
- Never select examples because they read well.

**Labels per claim.** Each label is one of the listed values, or `unsure`
with a note.

| field | values | guidance |
|---|---|---|
| P1 understandable | yes / partly / no | could you write down a test of it? |
| P2 supported | supported / contradicted / unsupported / not about the text | judged against the shown context only |
| P3 about what comes next or here | accurate / inaccurate / not checkable | only for claims about the next text or the current position |
| P5 mechanism | none / asserted, unsupported / asserted, supported | "signals", "implies" and "the model is checking" count as asserted |

Examples for the guide, all drawn from development items only:
- **Valid:** "Final token ').\n\n'" when the context ends `\).\n\n`.
- **Invalid:** a quotation that does not occur in the context.
- **Unsupported:** "signals a concluding summation" where nothing concludes.
- **Ambiguous:** "structured mathematics problem with steps": true of nearly
  every item, and it should be labelled as such through the generic-claim
  controls.

**Uncertainty.** Annotators label `unsure` instead of guessing. `unsure`
counts as its own category for agreement and is reported. Disagreements on
held-out items are **not** adjudicated into the score. Both raters' labels
are kept, and agreement is reported first.

**Raters and reliability.**
- Two independent annotators who did not design the claim splitter or this
  guide, and who see no reconstruction or model scores.
- Report Cohen's κ per field on the held-out set, with a problem-cluster
  bootstrap CI (`claim_checks.cohens_kappa`).
- A field with κ < 0.6 is reported as unreliable, and no rate from it is
  interpreted.
- An LLM judge may be added **only** as a third rater, validated against the
  two humans on the held-out items and reported as κ against each human. It
  is never ground truth.

**Report.** For each field, the held-out rate with its CI, κ, the control
items' rates, the share of `unsure`, and the frozen guide's hash.

**Tooling.** `scripts/make_claim_annotation_sheet.py`:
- builds the blinded development and held-out sheets and a separate key;
- with `--score A.csv B.csv`, computes κ per field;
- refuses to produce scores from a sheet with blank labels.

**Data still to collect.** Two annotators, about 6 hours each for roughly 250
held-out claims plus development.
