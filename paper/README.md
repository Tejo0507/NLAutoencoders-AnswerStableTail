# The report

The final report's sections, one file each, and how they become one document.
**All nine sections are written** and the report is frozen at the commit
recorded in `docs/DECISIONS.md` D66.

**Read it with [`ADDENDUM_2026-10-19.md`](ADDENDUM_2026-10-19.md).** A
post-hoc audit after the freeze (D68) changed no registered number, but it
narrows how several results should be read. The most consequential: the
primary comparison was made at budgets where a rule using no signal beats
every compared rule, so it is not evidence about the readout's incremental
value either way. The addendum is not a section and is not assembled into
`REPORT.md`.

## Assembling it

[`REPORT.md`](REPORT.md) is the whole report in one file, **generated** by
`scripts/assemble_paper.py`. Do not edit it; edit a section and re-run:

```powershell
.venv\Scripts\python.exe scripts\assemble_paper.py
.venv\Scripts\python.exe scripts\assemble_paper.py --check   # state only
```

The review was submitted as a single Word document rendered from a single
Markdown file, so the report has the same shape. The assembler adds:

- the title page;
- the table of contents;
- the bibliography;
- **a named placeholder for any section that does not exist**, rather than
  omitting it.

It also lifts each section's italic note out of the submission text.

It exits 2 while any section is missing or still carries an unresolved
`**[VERIFY]**` or `**[N]**` marker, and 0 when every section is present and
none does. On the frozen report it exits 0.

## Where the numbers come from

Every number in §6 is read from the main run's artifacts:
`runs/main/analysis/results.json` and its tables, and
`runs/main/report/RESULTS.md`. `scripts/check_run_consistency.py --run
runs/main` certifies those against each other (179 checks, 0 mismatched).

The exploratory and partial analyses in §6.8 are each quoted from their own
artifact and decision entry (`docs/DECISIONS.md` D51, D54–D60). They are
labelled as exploratory or partial wherever they appear, and none is used for
a registered conclusion.

## Status

| § | File | State | Waiting on |
|---|---|---|---|
| 1 | [`01_introduction.md`](01_introduction.md) | **final** | nothing |
| 2 | [`02_related_work.md`](02_related_work.md) | **final**, citations checked | nothing |
| 3 | [`03_method.md`](03_method.md) | **final** (pre-registered definitions, unchanged) | nothing |
| 4 | [`04_experimental_setup.md`](04_experimental_setup.md) | **final**, achieved N filled | nothing |
| 5 | [`05_analysis_plan.md`](05_analysis_plan.md) | **final**, §5.5 settled by the run | nothing |
| 6 | [`06_results.md`](06_results.md) | **final** | nothing |
| 7 | [`07_discussion.md`](07_discussion.md) | **final** | nothing |
| 8 | [`08_limitations.md`](08_limitations.md) | **final**, main-run numbers filled | nothing |
| — | [`00_abstract.md`](00_abstract.md) | **final** | nothing |

## Citations

Numbered `[n]` citations refer to [`references.md`](references.md), the
version-controlled copy of `../Project_Review_II.md` §References. The numbering
is unchanged, so every citation still points at the entry the review used.

On 2026-10-19 every entry was checked against its source:

- **Titles and authors:** checked against arXiv, Crossref or the publishing
  page.
- **Flagged attributions:** every in-text attribution the drafts had flagged
  was checked against the cited paper.

Three corrections came out of that check:

- **[26] and [27] author lists.** Each was corrected in `references.md`; the
  review's own copy is left as submitted.
- **§2.1's use of [27].** It had cited Self-Refine as a framework depending on
  *external* feedback. Self-Refine uses the model's own feedback, so the
  sentence now says that. Its point, that intrinsic self-correction does not
  reliably help, now rests on [18], whose abstract states it.

Venue names and page ranges were not re-checked against the proceedings,
except for the two DOI entries.

No statement is asserted to be novel. The contributions list in §1 says so
explicitly.
