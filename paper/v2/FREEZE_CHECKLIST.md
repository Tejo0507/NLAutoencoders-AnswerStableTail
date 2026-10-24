# Manuscript freeze checklist (prepared 2026-10-24)

> **Status: revised candidate, awaiting author review. Not approved, not
> frozen.** This checklist prepares a freeze; it does not perform one. Only
> the authors can approve the text and record the freeze.

## A. Checks that pass on the candidate (re-run before freezing)

| check | command | result on 2026-10-24 |
|---|---|---|
| Every key number equals its stored output | `python scripts/verify_manuscript_numbers.py --runs <runs> --d76 <d76 runs>` | 84 of 84 facts, 0 failures |
| Citations resolve, are all cited, and are numbered by first citation; figures and tables captioned and cited in order; every § reference exists | `python scripts/assemble_manuscript_v2.py --check` | 0 problems |
| The replication's readings, hashes, gates and listed numbers | `python scripts/check_run_consistency.py --replication --complete --runs-root <runs>` | 169 checks, 0 failed |
| Tables 1–3 regenerate from configs, record and outputs | `python scripts/make_paper_tables_v2.py --runs <runs>` | regenerated; no manual edits |
| Figures 1–7 regenerate from outputs | `python scripts/make_paper_figures_v2.py --runs <runs>` | all seven regenerate byte-identically to the committed files |
| CPU test suite | `python -m pytest -m "not slow"` | 1,182 passed, 31 skipped |
| Bibliography entries match their verified sources | `scripts/consolidate_references_v2.py` provenance notes; external metadata re-check (review §4) | see `review/REVIEW_2026-10-24.md` |
| scientific-writing linter (text) | `lint_manuscript.py MANUSCRIPT.md --manifest evidence/manuscript_manifest.json` | no text issues; 2 manifest errors that need a human (confidentiality and policy review) |

## B. Decisions and statements only the authors can supply

- [ ] **Approve the text** of `paper/v2/` (all sections, captions, tables,
      references), or return changes.
- [ ] **Author list, affiliations and contributions** (CRediT roles).
- [ ] **Funding** statement and sponsor role.
- [ ] **Competing interests** statement.
- [ ] **AI-use disclosure.** This revision was drafted and checked with an
      AI assistant using installed research-writing skills; the venue's
      policy decides the wording. The skills used ask to be cited
      (Kassis et al., *Scientific Agent Skills*, arXiv:2609.00065); whether
      to cite them is the authors' decision and is not done here.
- [ ] **Ethics and consent**: confirm "not applicable (no human or animal
      participants; public benchmarks; the human annotation protocol was
      not run)".
- [ ] **Data and code availability**: confirm the wording once the archive
      is published (its identifier goes into §7).
- [ ] **Human verification of the references**: open each cited source and
      confirm the bibliographic details and the cited proposition.
- [ ] **Target venue** (sets the template, length and reference style).

## C. Freeze procedure (after B)

1. Re-run every check in A on the final text; all must pass.
2. Regenerate `paper/v2/MANUSCRIPT.md` with the approved status:
   `python scripts/assemble_manuscript_v2.py --status "Approved by the authors on <date>; frozen at <commit>."`
3. Commit, merge through a pull request (required checks: `pytest`,
   GitGuardian), and record the freeze commit in `docs/DECISIONS.md` as a
   new decision (as D66 did for the registered report).
4. Build the release at that commit (`docs/ARCHIVE_PLAN_REP1.md` §8), then
   add the published identifier to §7 in a follow-up commit.
