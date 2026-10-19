# RQ2 claim annotation guide (amendment A1 §4)

For the sheet `runs/main_rq2probe/annotation_sheet_v3.csv`, 150 rows. The
annotation validates a lexicon rule, so you are recording **what each claim
asserts**, not what is true about the problem. You will not see the rule's
reading, the problem, its trace or the right answer, and you should not look
them up.

## 1. What you are labelling

Each row is one *claim*: a segment of a description that an automatic
verbaliser wrote about one moment inside a model's step-by-step solution to a
maths problem. The descriptions talk about "the context" (the solution text
written so far), the "final token" (the last bit of text at that moment) and
what the model is "expecting" to write next.

- `claim_text` — the claim, verbatim. **This is the only text you label.**
- `explanation_context` — the whole description the claim came from, with the
  claim marked `[[ like this ]]`. Read it only to understand the marked claim
  (§2). Do not label what other parts of the description say.
- `answer_stated`, `domain` — fill in one code each (§3, §4). Every cell must
  be filled; there is no blank.
- `notes` — optional, free text (e.g. "fragment, unclear what the quote is").

## 2. Using the context

Claims are often cut in the middle of a quotation, so a row can start with
`")` or end inside a quote. Use the surrounding description only to answer:
*what is this quoted text, and is the claim saying it was already written or
that it is expected next?* Never carry an assertion over from another part of
the description: if only an unmarked part says "this is a probability
problem", the marked claim does not assert a domain.

## 3. `answer_stated` — does the claim say the final answer has already been written?

| code | use when the marked claim … |
|---|---|
| `1` | says the solution text **has already stated** its final answer at this point: e.g. it quotes already-written text containing the answer ("…the context ends with "Therefore, the answer is 55.""), or says the text closes / concludes the solution, gives the final result, or ends on the answer. |
| `0` | says the final answer is **not yet stated** but is expected or about to come: e.g. "strongly expecting "The final answer is"", "about to compute the total", "leads into the final step". |
| `NA` | makes no claim either way, or you cannot tell (including fragments too short to judge, and claims that both look back at a stated answer and forward to a later one). |

Rules:

- Quoted text counts as *already written* only if the claim presents it as
  part of the context (closes, ends, the sentence "…", the final token "…").
  Quoted text after "expecting", "anticipating", "predicting", "next", "or"
  in a list of continuations is *expected*, not written.
- An intermediate result ("so far the total is 12") is not the final answer.
  Use `1` only for the final answer or a concluding statement of the result.
- When unsure, use `NA`. Do not guess.

## 4. `domain` — what kind of problem does the claim say this is?

| code | the marked claim describes the problem or its subject as … |
|---|---|
| `gsm8k` | a grade-school arithmetic **word problem**: everyday situations (shopping, money, prices, time, ages, distances, recipes, counts of objects) solved with arithmetic. |
| `math-algebra` | **symbolic algebra**: solving equations for a variable as the task itself, simplifying or factoring expressions, polynomials, quadratics, functions, inequalities, systems of equations. |
| `math-counting_and_probability` | **counting or probability**: arrangements, permutations, combinations, "how many ways", dice, coins, cards, computing a probability. |
| `NA` | the claim says nothing about the kind of problem, or its cues fit more than one category, or you are unsure. |

Rules:

- Label from what the claim *says*, not from what you guess the problem was.
  A claim that only mentions "an equation" or "a variable" used to solve an
  everyday word problem is `gsm8k` if it describes the everyday situation,
  `NA` if it does not; it is not `math-algebra` because an equation appears.
- Generic descriptions ("structured math problem", "step-by-step solution",
  "numbered steps") are `NA` unless they also name a situation or subject from
  the table.
- Numbers alone (e.g. "$5", "3 weeks") describing an everyday quantity count as
  `gsm8k` cues only if the claim presents them as part of the problem's
  situation.
- When cues point two ways, or you are unsure, use `NA`.

## 5. Procedure

1. Read §1–§4 once; do the first 10 rows, then re-read §3–§4.
2. Work in row order. Do not reorder, delete or edit `row_id`, `claim_text`
   or `explanation_context`; the sheet is checked against its key.
3. Fill every `answer_stated` and `domain` cell with exactly one code from the
   tables (case-sensitive, no spaces).
4. Return the sheet. `scripts/rq2_claim_probe.py --phase agreement
   --annotation <sheet>` rejects any empty or invalid cell and lists it.

The agreement threshold (Cohen's κ ≥ 0.6 per attribute) was fixed before
annotation and will not be changed after it.
