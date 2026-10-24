# Zenodo deposit: follow-up task (not started)

> **No Zenodo account is available** (researcher, 2026-10-24). Nothing here
> has been done, and no credentials exist on this machine. The GitHub release
> does not depend on it.

When an account exists:

1. **Create the account** at zenodo.org (ORCID or GitHub sign-in) and decide
   whether to deposit manually or through Zenodo's GitHub integration.
   - The integration is triggered by new GitHub releases after it is
     enabled; releases published before that are expected to need a manual
     deposit. Check Zenodo's current documentation before relying on this
     (not verified here).
2. **Manual deposit, one per archive** (`data-runs-2026-10-19` and the rep1
   release):
   - upload type *Dataset*; upload the `.tar.gz` and its `.sha256` directly;
   - title, authors and description from `CITATION.cff` and the release
     notes;
   - licence CC BY 4.0;
   - related identifiers: "is supplement to" the GitHub repository at the
     release commit, and "is identical to" the GitHub release URL.
3. **Publish**, and record the DOI only once Zenodo shows the record as
   published.
4. **Record** the DOI in `docs/DECISIONS.md`, `CITATION.cff` (an
   `identifiers` entry, or `doi`) and §7 of the manuscript, through a pull
   request.
5. **Verify** by downloading from Zenodo and checking the SHA-256 against
   the GitHub release.
