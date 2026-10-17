# RQ3 — decision brief (2026-10-17)

**Decision needed from the researcher. Nothing below has been started.**

## Where RQ3 stands

RQ3 asks whether tail-window states carry a direction whose ablation
*selectively changes verification-like behaviour* without changing final
answers (`Project_Review_II.md` l.122). The construct is registered; no
dependent variable is (D40, D41). The implementation's readout,
`marker_delta`, counts twenty lexical rechecking markers.

On `main` the **independent variable worked and the dependent variable did
not**:

- The edit was applied: 59 of 60 problems edited at every non-zero dose (the
  60th generated no continuation); answer-change rises with dose, 0 → 21.7%
  from dose 0.5 to 2.0, while random-direction and position-matched controls
  change nothing (exploratory, not a registered outcome).
- The readout has no range: markers occur in 2 of 274 traces (pooled 0.0073,
  95% CI [0.0009, 0.0261]); at the compared dose it is zero in every arm on
  every problem. `measure_rq3_readout_floor.py`: **unevaluable** — 3
  marker-bearing problems need an arm of ~115 even at the interval's top.

The verdict is therefore **not tested** (D47). It is neither support nor
refutation.

## The options

| | A. Close as untested | B. Amend, then confirm | C. Re-pose the question |
|---|---|---|---|
| What | Report RQ3 as untested on Qwen2.5-7B-Instruct: the registered construct has no measurable range on this model's prose. | Register, *before* any new run, a validated in-construct measure of verification-like behaviour, and test it on held-out problems. | Replace RQ3 with a question about a quantity that moves (answer change, continuation length) and say it is a changed question. |
| Cost | None. Documentation only. | Design + validation (e.g. a judge-scored rechecking rubric with inter-rater agreement on a calibration set), an amendment, then ~1 h GPU per 60 problems. **83 eval-split problems with a tail are untouched by the causal arm** and can serve as the holdout. | An amendment and a re-analysis; GPU only if a new run is wanted. |
| Validity | Fully consistent with the pre-registration. | Valid only if the measure, analysis plan, direction of effect and holdout are fixed before the run (D35, #11). Choosing the measure *because* the screen shows it moves is the forking path D35 forbids. | Honest only if labelled a new question. The existing screen (`usable_outcomes`: `continuation_tokens`, `answer_changed`, `correct`) was seen; any test on the same 60 problems is exploratory. |
| What it can establish | That the registered test could not be made on this model. | A confirmatory answer to the registered construct. | An answer to a different question. |
| Risk | Leaves RQ3 unanswered. | The validated measure may also lack range on this model. | Reads as moving the goalposts unless framed carefully. |

## What not to do

- Do not report `marker_delta`'s ties as evidence against a tail direction.
- Do not promote `continuation_tokens` into RQ3's verdict by changing a
  constant: it measures output length (D41) and `_assert_eligible` refuses it.
- Do not run a confirmatory pass before the measure, its validation, the
  analysis plan and the holdout list are written down and dated.

Related: issue #11, `DECISIONS.md` D34–D35, D38–D41, D46–D47.
