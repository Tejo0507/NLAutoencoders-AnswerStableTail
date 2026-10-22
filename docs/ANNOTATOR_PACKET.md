# Annotator packet: claim semantics (D71) and rewording validity (D76)

**For the researcher.** Everything here is ready to hand to two independent
annotators. Recruiting them, and their consent and payment, need you; no
assistant can do it. Annotators must not have designed the claim splitter,
this guide or the paraphrase rules, and must not see any reconstruction score
or model output beyond what the sheet shows. **No label in any sheet has been
filled.** Labels produced by a model would not be human judgements and must
never be entered.

## What to send each annotator

1. This file, from "Instructions for annotators" onwards.
2. `runs/main_claim_annotation/dev.csv`: 92 claims, for practice and
   calibration.
3. `runs/main_claim_annotation/heldout.csv`: 219 claims, to be scored once.
4. `runs/main_paraphrase_llm/rewording_validation_sheet.csv`: the accepted
   rewordings, once D76's run has produced them.

Keep `key.json` back. It reveals which items are the blind controls.

**Estimated time:** about 6 hours per annotator, roughly 1 minute per claim
plus about 30 seconds per rewording.

## Instructions for annotators

You will read short claims that an automatic system wrote about a moment in a
model's step-by-step solution to a maths problem. Each row shows:
- **question:** the problem;
- **context_end:** the last part of the solution written so far, which ends
  exactly at the moment described;
- **continuation:** the next 200 characters the model actually wrote;
- **claim_text:** one claim to judge.

Judge each column independently. If you cannot decide, write `unsure` and add
a note; never guess. Do not discuss items with the other annotator.

| column | allowed values | question to ask |
|---|---|---|
| `P1_understandable` | yes / partly / no / unsure | Could you write down what would make this claim true or false? |
| `P2_supported` | supported / contradicted / unsupported / not about the text / unsure | Judging only from *question* and *context_end*: is what the claim says about the text there? A claim that "quotes" text counts as supported only if that text, or text with exactly that meaning, appears. |
| `P3_next_or_here` | accurate / inaccurate / not checkable / unsure | Only for claims about the token at the end of *context_end*, or about what comes next. Is the claim right, judged against the end of *context_end* and the *continuation*? Otherwise, `not checkable`. |
| `P5_mechanism` | none / asserted, unsupported / asserted, supported / unsure | Does the claim say what the model is internally doing (checking, planning, being confident, "signals that …")? If it does, does the visible text support that? |

Examples, all from the practice set:
- **Final token `").\n\n"`, when *context_end* ends with `\).` and a blank
  line:** P3 is `accurate`.
- **"The paragraph 'Calculate the number of apples remaining…'", when that
  sentence is not in the context:** P2 is `contradicted`.
- **"Strongly expecting 'Next, we add…'", when the continuation begins
  "Next, let's calculate…":** P3 is `accurate` if the claim is only that the
  next sentence starts with "Next". Use `inaccurate` if the claim's specific
  wording is what it asserts and that wording is wrong.
- **"Structured math problem with steps":** P1 is `yes`, P2 `supported`, P5
  `none`. Generic claims are expected; the sheet includes controls for them.

**Rewording sheet.** For each row, is `rewording` exactly the same claim as
`original`, with no change in meaning, nothing added and nothing lost? Write
`same_meaning` as yes / no / unsure.

Return each sheet as CSV with the same rows in the same order.

## After the sheets come back

```powershell
.venv\Scripts\python.exe scripts\make_claim_annotation_sheet.py --config main --score a_heldout.csv b_heldout.csv
```

This prints Cohen's κ per field and refuses a sheet with any blank label.

**Use `--report` for the protocol's numbers (D79).** `--score` pools real
claims with the blind controls and gives a point κ only. It returns
κ = 1.0 when both raters use one label throughout, which is no evidence of
reliability.

```powershell
.venv\Scripts\python.exe scripts\make_claim_annotation_sheet.py --config main --report a_heldout.csv b_heldout.csv
```

It reports, per field:
- κ on held-out **real** claims, with a problem-cluster bootstrap 95% CI.
  κ is *undefined* when only one label is used;
- each rater's label shares for real claims and for each kind of blind
  control;
- `reliable` (κ defined and ≥ 0.6).

**Known, minor leak.** One problem, `gsm8k-test-00029`, appears in both
sheets:
- in the held-out sheet, as the mismatched-trace control `item005`;
- in the dev sheet, as the real item `item051`.

An annotator who remembers the practice item could recognise `item005` as a
control. Report the held-out results with and without `item005`. The sheets
are left as built, because their key and order are already recorded.
- A field with κ < 0.6 is reported as unreliable.
- Rates are reported only for fields that are reliable, on held-out items,
  with the blind controls' rates beside them.
- Both raters' labels are kept. Disagreements are not adjudicated into the
  score.
