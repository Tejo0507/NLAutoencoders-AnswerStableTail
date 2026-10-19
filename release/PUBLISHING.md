# Publishing the run artifacts (D63, D64)

The package is built from `runs/` by `scripts/build_results_release.py`,
using this directory's README template, `NOTICE.md` and `licences/`. The
builder sanitises local path roots in the staged copy only and refuses to
archive anything its scan still finds (D65). The package is published only
when every gate below holds. The current draft,
`D:\NLAutoencoders-release\data-runs-DRAFT-2026-10-18-sanitised\`, was built
and verified on 2026-10-18 (`docs/AUDIT_2026-10-17.md`, PW-28). It is
**not** published, because the write-up is not frozen. The earlier
unsanitised draft was deleted on 2026-10-19 after the sanitised one was
re-verified (D65).

## Gates, all required

1. The write-up is frozen, and the freeze is recorded with its commit (in
   `docs/DECISIONS.md` or as a tag).
2. The working tree is clean at that commit.
3. `build_results_release.py --status final` succeeds: every entry under
   `runs/` is classified, and every released run is present.
4. Inside the package, `sha256sum -c SHA256SUMS.txt` passes, and the
   archive's `.sha256` verifies.
5. `scan_release.py` on the package directory **and** on the archive
   reports exactly one finding: `abs_windows_root` in
   `runs/main_f6_diag_validation/validate_f6_diag.py`, the retained validator
   (the builder pins it to its SHA-256). Any other finding blocks publication.
6. The retained path is accepted. **Satisfied 2026-10-19:** the researcher
   approved retaining the F6 validator's repository-root constant, `R` (a
   drive letter and the public repository name), unsanitised, under its
   SHA-256 pin (D65). At the freeze, confirm that the pin still matches; do
   not sanitise the file or relax the scan.
7. `TRANSFORMS.json` lists 41 sanitised files (or the researcher has
   reviewed a different count). Every untransformed run file is identical
   to `runs/`.
8. The release notes match the archive actually built.

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
