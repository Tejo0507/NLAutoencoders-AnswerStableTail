# Manuscript freeze checklist (prepared 2026-10-24; re-run 2026-10-25 after the pilot was added, D102, and after the verification pass, D103)

> **Status: revised candidate, awaiting author review. Not approved, not
> frozen.** This checklist prepares a freeze; it does not perform one. Only
> the author can approve the text and record the freeze.

## A. Checks that pass on the candidate (re-run before freezing)

| check | command | result on 2026-10-25 |
|---|---|---|
| Every key number equals its stored output | `python scripts/verify_manuscript_numbers.py --runs <runs> --d76 <d76 runs> --pilot <pilot runs>` | 115 of 115 facts, 0 failures |
| The pilot's results, recomputed from the raw per-boundary files without the pilot's analysis module | D103 §1 (scratch script) | labels 0 mismatches; τ, counts, intervals, bound, saving and bootstrap CI identical; the calibration and evaluation sets are disjoint from each other and from all 678 earlier ids |
| Citations resolve, are all cited, and are numbered by first citation; figures and tables captioned and cited in order; every § reference exists | `python scripts/assemble_manuscript_v2.py --check` | 0 problems |
| The replication's readings, hashes, gates and listed numbers | `python scripts/check_run_consistency.py --replication --complete --runs-root <runs>` | 169 checks, 0 failed |
| Tables 1–3 regenerate from configs, record and outputs | `python scripts/make_paper_tables_v2.py --runs <runs>` | regenerated (one stale heading fixed, D102) |
| Figures 1–7 regenerate from outputs | `python scripts/make_paper_figures_v2.py --runs <runs>` | all seven regenerate byte-identically to the committed files |
| CPU test suite | `python -m pytest -m "not slow"` | 1,237 collected: 1,206 passed, 31 skipped, 0 failed, 0 errors |
| Bibliography metadata | arXiv API (27 entries), Crossref, proceedings pages, OpenAlex (D103 §4) | 32 of 32 match their external records; no correction needed |
| scientific-writing manifest | `validate_manifest.py evidence/manuscript_manifest.json` | pass, 0 errors; 5 statements `draft`, 3 `missing` |
| scientific-writing linter (text) | `lint_manuscript.py MANUSCRIPT.md --manifest evidence/manuscript_manifest.json` | 2 manifest items need a person (confidentiality and policy review). 2 warnings flag the corresponding email, which is intended. 2 errors ask for the skill's inline `[claim:…][evidence:…]` markers on the conflict and consent lines; this project traces evidence through `key_numbers.json` and the decision record instead, and these statements rest on the author's confirmation (D104). No other text issue |

## B. Items for the author, with what is already settled

Settled by D103, with the evidence there:
- [x] **The stopping-policy pilot's place in the paper.** It stays in the
      main text as written, because it is the only out-of-sample evidence
      on the paper's open practical question. Its numbers and wording
      follow claim-matrix §S.
- [x] **The pilot's outputs in the release:** included (D98's rule). A
      verification build with them passes every check (D102).
- [x] **Reference metadata:** 32 of 32 match their external records.

Confirmed by the author on 2026-10-25 and written into
`09_declarations.md` and the author block (D104):
- [x] **Author:** Tejo Sridhar M V S, sole author. The ORCID
      0009-0007-5023-1092 was checked against the public ORCID record,
      whose name matches. The corresponding address is the institutional
      one. The personal email is kept out of the public files.
- [x] **Funding:** "This research received no funding."
- [x] **Competing interests:** "The author declares no competing financial
      or professional interests."
- [x] **Ethics and consent:** no human participants, surveys, interviews or
      private personal data; public benchmarks; computational experiments
      only. No IRB review was sought, and no approval or exemption is
      claimed. IEEE requires an oversight statement only for research with
      human subjects or animals; the statement explains why none applies.
- [x] **Target publisher:** IEEE. Its AI policy (IEEE Author Center,
      checked 2026-10-25) requires disclosure in the Acknowledgments,
      naming the AI system, the sections affected and the level of use. The
      AI statement in `09_declarations.md` follows that structure.

Drafted from the record. The author approves each wording:
- [ ] **Affiliation wording.** SRMIST's own documents place the department
      in the School of Computing, Faculty of Engineering and Technology, at
      Kattankulathur, Chennai 603203. The draft reads "Department of
      Computational Intelligence, School of Computing, SRM Institute of
      Science and Technology, Kattankulathur 603203, Tamil Nadu, India".
      Confirm that form, or the form your institution asks for.
- [ ] **Author contributions** (CRediT): all roles, as sole author.
- [ ] **AI-use statement.** The session records (2026-10-03 onwards) name
      the Claude models and the four levels of use. Confirm that the
      statement is complete, including any AI use before 2026-10-03 (the
      repository starts on 2026-08-27), and that its wording is acceptable.
      The skills used ask to be cited (Kassis et al., *Scientific Agent
      Skills*, arXiv:2609.00065); whether to cite them is your decision.
- [ ] **Data and code availability:** the replication archive's identifier
      goes into §7 once it is published.

Still open:
- [ ] **The specific IEEE venue.** It fixes the template (IEEEtran), the
      length, where funding and conflict statements go, and any
      venue-specific rules.
- [ ] **Approve the text** of `paper/v2/` (all sections, captions, tables,
      references, declarations), or return changes. Recommended before
      approving: read §3.9, §4.9 and §5.4 against claim-matrix §S, and read
      the cited propositions of the references you rely on.

## C. Freeze procedure (after B)

1. Re-run every check in A on the final text; all must pass.
2. Regenerate `paper/v2/MANUSCRIPT.md` with the approved status:
   `python scripts/assemble_manuscript_v2.py --status "Approved by the authors on <date>; frozen at <commit>."`
3. Commit, merge through a pull request (required checks: `pytest`,
   GitGuardian), and record the freeze commit in `docs/DECISIONS.md` as a
   new decision (as D66 did for the registered report).
4. Build the release at that commit (`docs/ARCHIVE_PLAN_REP1.md` §8), then
   add the published identifier to §7 in a follow-up commit.
