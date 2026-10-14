# Paper drafts

Sections that can be written before the main run finishes, and what each one
is still waiting on.

## Assembling it

[`REPORT.md`](REPORT.md) is the whole report in one file, **generated** by
`scripts/assemble_paper.py` — do not edit it; edit a section and re-run:

```powershell
.venv\Scripts\python.exe scripts\assemble_paper.py
.venv\Scripts\python.exe scripts\assemble_paper.py --check   # state only
```

The review was submitted as a single Word document rendered from a single
Markdown file, so the report needs the same shape. The assembler adds the title
page, the table of contents and the bibliography, lifts each section's drafting
note out of the submission text, and writes **a named placeholder for every
section that does not exist yet** rather than omitting it — a reader of the
draft should be able to see what is missing without consulting this file. It
exits 2 while anything is pending and 0 when every declared section is present,
so it is also the check for whether the report is submission-ready. It is not:
the Abstract, Results and Discussion are absent by design, and §4 and §8 are
drafted but still need the main run's numbers.

The bibliography is read from `../Project_Review_II.md` §References rather than
copied, so the two documents cannot drift. Two things that surfaced on first
assembly: it holds **27** entries rather than the 26 stated below, and `[27]`
(Madaan et al., *Self-refine*) is **not cited by any drafted section** — it is
marked as such in the generated references rather than dropped, and needs
either a citation in the Results or Discussion or removal before submission.

**Nothing here reports a result.** No section states an outcome of the main
run, and the Results and Abstract are deliberately absent rather than
stubbed — writing them now would require assuming numbers that do not exist.
The pilot's numbers are quoted only where they are explicitly labelled as the
pilot's and used to motivate a design choice, never as findings.

## Status

| § | File | State | Waiting on |
|---|---|---|---|
| 1 | [`01_introduction.md`](01_introduction.md) | **draft complete** | nothing — the motivation does not depend on the outcome |
| 2 | [`02_related_work.md`](02_related_work.md) | **draft complete**, citations traceable | independent verification of every reference (see below) |
| 3 | [`03_method.md`](03_method.md) | **draft complete** | nothing — these are the pre-registered definitions |
| 4 | [`04_experimental_setup.md`](04_experimental_setup.md) | **draft complete** except achieved N | the achieved N, which is what gets reported |
| 5 | [`05_analysis_plan.md`](05_analysis_plan.md) | **draft complete** | one reporting decision (BH convention, D38) |
| 6 | Results | **not written** | the main run |
| 7 | Discussion | **not written** | the Results |
| 8 | [`08_limitations.md`](08_limitations.md) | **draft complete** for everything already established | three entries need the main run's own numbers |
| — | Abstract | **not written** | the Results |

## Citation keys

Numbered `[n]` citations refer to the bibliography in
[`../Project_Review_II.md`](../Project_Review_II.md) §References, which has 26
entries. Using that numbering keeps every citation in these drafts traceable
to an entry that already exists in the repository rather than to one invented
here.

**Every reference still needs independent verification before submission.**
The bibliography was assembled during the proposal stage; author lists, venues,
years and DOIs have not been re-checked against the publishers in preparing
these drafts, and at least four entries are marked in the review as preprints
whose status may since have changed. Treat `[n]` as a pointer, not as a
verified citation.

## Claims flagged for verification

Marked inline as **[VERIFY]**. They fall into three kinds:

- **Novelty claims.** Nothing here asserts that a construct or method is new
  without a marker, because establishing novelty needs a literature search
  that has not been performed for this draft.
- **Attributions.** Where a draft says a specific prior work showed a specific
  thing, the attribution follows `Project_Review_II.md`'s own reading of it
  and has not been re-derived from the source.
- **Quantities from outside this repository.** Hardware figures, model card
  values and anything not computed by this codebase.

Quantities *computed by this codebase* are not flagged; they are reproducible
from `runs/<id>/` and are cited to the artifact that holds them.
