# Publishing the run artifacts (D63, D64)

The package is built from `runs/` by `scripts/build_results_release.py`,
using this directory's README template, `NOTICE.md` and `licences/`. It is
published only when every gate below holds. A draft was built and verified on
2026-10-18 (`docs/AUDIT_2026-10-17.md`, PW-27); it is **not** published,
because the write-up is not frozen.

## Gates, all required

1. The write-up is frozen, and the freeze is recorded with its commit (in
   `docs/DECISIONS.md` or as a tag).
2. The working tree is clean at that commit.
3. `build_results_release.py --status final` succeeds: every entry under
   `runs/` is classified, and every released run is present.
4. Inside the package, `sha256sum -c SHA256SUMS.txt` passes, and the
   archive's `.sha256` verifies.
5. `scan_release.py` on the package directory **and** on the archive shows
   no finding other than the provenance paths the package README discloses
   and the repository URL in the release documents.
6. The researcher accepts that disclosure: the local absolute paths kept in
   `env/`, the manifests, `models/` and the logs, one of which names an
   unrelated local project folder (`main/env/environment.json`, the model
   cache path).
7. The release notes match the archive actually built.

## Commands

```powershell
$date = "<freeze date>"            # e.g. 2026-11-30
$out  = "D:/NLAutoencoders-release/data-runs-$date"
.venv/Scripts/python.exe scripts/build_results_release.py --out $out --status final --date $date
.venv/Scripts/python.exe scripts/scan_release.py "$out/nlaast-runs"
.venv/Scripts/python.exe scripts/scan_release.py "$out/nlaast-runs-$date.tar.gz"
```

```bash
(cd "$out/nlaast-runs" && sha256sum -c SHA256SUMS.txt) && (cd "$out" && sha256sum -c *.tar.gz.sha256)
```

Then, by hand:

1. Push the freeze commit, then `git tag data-runs-<date> <commit>` and
   `git push origin data-runs-<date>`.
2. `gh release create data-runs-<date> --title "Run artifacts: NLA Answer-Stable Tail (main, pilot, exploratory)" --notes-file <notes> <archive> <archive>.sha256`
3. Create a Zenodo record of type *Dataset* with the archive and its
   `.sha256` uploaded directly, licence CC-BY-4.0, related to the
   repository at the freeze commit. Publish it, and record the DOI only once
   Zenodo shows it as published.
4. Record the tag, release URL, Zenodo record, DOI, archive SHA-256 and date
   in `docs/DECISIONS.md` and the audit register.
