# Archive plan for the replication (#32), 2026-10-23

> **Plan only. Nothing has been archived, moved or removed.** The pipeline's
> checkout (`D:\NLAutoencoders-AnswerStableTail`) stays exactly as it is
> (Option A). Nothing is deleted anywhere until the archive has been
> independently verified (§5), and even then the original `runs/` stays.

## 1. What is archived

**A. Run outputs** (read from `runs/`, never written there):

| directory | size | content |
|---|---|---|
| `rep1/` | 114 MB | problems, traces, ast, acts (89 MB), baselines, nla (15 MB), manifest |
| `rep1_forced/` | 0.3 MB | 48-token menu completions and the 200 re-forced boundaries |
| `rep1_o3_registered/` | 24 KB | **the D75 confirmatory analysis** |
| `rep1_position_audit/` | 64 KB | D79 |
| `rep1_information/` | 52 KB | D81 on O48 and O256, and the lexicon transfer |
| `rep1_selective/` | 8 KB | selective risk |
| `rep1_extforced/` | 0.7 MB | D82 on rep1 |
| `rep1_stage1/` | 12 MB | Stage 1 |
| `main_nla240_extforced/` | 0.5 MB | D82 on `main_nla240` (exploratory) |
| `main_nla240_information/` | 52 KB | updated with `information_O256.json` |
| `rep1_runbook/`, `post_rep1_queue/` | 0.4 MB | stage logs and console output |
| `stage1_power_audit/` | small | `power.json`, the D85 power audit |

**B. The checkout state at the run**, from the pipeline checkout, read
only:
- the 8 modified tracked files and 82 untracked files (the comparison is in
  the reconciliation plan);
- `git status --porcelain`, `git rev-parse HEAD` (`b17c131`) and the branch
  name;
- a SHA-256 manifest of all 90 files.

This is the "dirty" code state that every rep1 output records
(`git.dirty: true` at `b17c131`).

## 2. Tooling

`scripts/build_results_release.py`, the D63–D67 builder, is reused.
- **How it works:** it copies the agreed directories into a new output
  directory and sanitises absolute local paths **in the copy only**. It
  writes `SHA256SUMS.txt`, `TRANSFORMS.json` (the original and released
  hash of every changed file) and `RUNS.json`, and builds a deterministic
  `.tar.gz`.
- **Gate:** `scan_release.py` must find no paths, identities or secrets
  before an archive exists.
- **Needed before it can run:** every new directory above must be added to
  its `RELEASED` table (or to `EXCLUDED`, with a reason). The builder
  refuses unclassified entries by design. That is a small code PR and needs
  approval.

## 3. Procedure (each step needs approval; none is destructive)

1. **PR:** classify the directories in §1A in `build_results_release.py`.
2. **Snapshot hashes of the originals:**
   `sha256sum` over every file in §1A, run against the pipeline checkout's
   `runs/` read-only, saved as `SOURCE_SHA256SUMS.txt` outside the
   repository.
3. **Consistency, before archiving:**
   `check_run_consistency.py --replication --complete --runs-root <runs>`
   must pass. It does today: 169 checks, 0 failed.
4. **Build a draft:**
   `python scripts/build_results_release.py --runs <runs> --out D:/NLAutoencoders-release/rep1-<date>`
   This is a new directory outside both checkouts.
5. **Bundle B:** copy the 90 files and their manifest into
   `<out>/checkout_state_rep1/` (a copy, not a move).
6. **Scan:** `scan_release.py` on the package and on the `.tar.gz`.
7. **Final build:** with `--status final` from a clean `main` checkout,
   after review.
8. **Publishing** (a GitHub release, and Zenodo if chosen, D67) is a
   separate manual step for the researcher.

## 4. The local `stage1_power_audit.py`

D85 recorded the version that produced `power.json`. D91 later removed an
unused import on `main`. The checkout keeps the D85 version.

| file | SHA-256 | matches |
|---|---|---|
| checkout `scripts/stage1_power_audit.py` | `64eaa58e595335e19a6222b29267bfeb21ec7d1a236ce606ff56407966305716` | D85 (`64eaa58e…5716`) |
| `main` `scripts/stage1_power_audit.py` | `5195be2ebdbd4d840501d56766474328557227683b45df4525ed8a0560846a72` | D91's edit |
| `runs/stage1_power_audit/power.json` | `85300f39ca052c421feb0b6383bb219a775cd19073e68ac4e4fd25f787d118a1` | D85 (`85300f39…18a1`) |

**Verification:**
1. Hash the checkout file. It must equal D85's `64eaa58e…5716`. Verified
   2026-10-23.
2. Diff it against `main` with line endings normalised. The only
   difference must be one added line, `import json`. Verified 2026-10-23.
3. Archive the checkout file in bundle B with its hash.
4. **Optional, needs approval:** reproduce `power.json` with the D85 version
   and compare hashes.
   - **Never run it in place.** It writes to a hard-coded
     `runs/stage1_power_audit/power.json` beside the script, and would
     overwrite the stored output.
   - Run it only from an isolated copy: the script, `src/` and copies of its
     inputs, in a scratch tree outside both checkouts. Then compare the new
     `power.json` hash with `85300f39…18a1`.
   - A mismatch would show non-determinism, not invalidate D85. Report it
     either way.

## 5. Independent verification before anything is retired

1. Extract the `.tar.gz` into a new directory.
2. Recompute `SHA256SUMS.txt` there. Every line must match.
3. For every file, the released hash equals the source hash, or
   `TRANSFORMS.json` maps it to the source hash and lists the
   sanitisation.
4. Spot-check: `o3_registered.json` in the archive carries the H1/H2
   numbers of D94; `rep1_stage1/analysis.json` carries D96's Δ₁.
5. Re-hash the originals in the pipeline checkout and compare them with
   `SOURCE_SHA256SUMS.txt` from step 3.2. This proves the archive process
   did not touch them.

**A caveat for step 3:** the replication checker cannot be run on the
*released* copy as is. Path sanitisation changes some JSON files, and the
hashes recorded inside the outputs refer to the originals. The checker is
run on the originals (step 3.3), and `TRANSFORMS.json` links each released
file to its original hash.

## 6. After verification

- The pipeline checkout can be retired (Option A): kept read-only as the
  run's provenance, with new work done in a worktree on `main`.
- Removing anything from it (Option B) remains a separate decision for the
  researcher, and is not required by this archive.

## 7. Release candidate, built and verified 2026-10-23 (not published)

Built by steps 3.2–3.6 and §5, from a fresh clean clone of `main` at
`56f66e5`, reading the pipeline checkout's `runs/` read-only. The output is
`D:\NLAutoencoders-release\rc-2026-10-23\`, a new directory outside both
checkouts. The earlier draft (`rep1-2026-10-23-draft`) was not touched.

**Package (`--status final`, date 2026-10-23):**
- 3,273 staged files, of which 3,264 are run files; 95 sanitised; archive
  `nlaast-runs-2026-10-23.tar.gz`, 240,109,648 bytes, SHA-256
  `e1f31aec259e1d54b7bc8e221940d323da59eb9928033521c18fa569e6a28df7`.

**Checks, all on the actual files:**

| check | result |
|---|---|
| originals unchanged (step 3.2/5.5) | the 3,415 files under the pipeline checkout's `runs/`, `configs/` and `docs/` hash identically to a manifest taken before any work that day; its `HEAD`, `git status` and tracked diff are unchanged |
| replication consistency (step 3.3) | `check_run_consistency.py --replication --complete`: 169 checks, 0 failed |
| scan, package and archive (step 3.6) | exactly one finding, `abs_windows_root` in the pinned F6 validator (the D65 retained file) |
| extraction (§5.1–5.2) | extracted into a new directory; `sha256sum -c SHA256SUMS.txt`: 3,272 of 3,272 entries OK; the archive's `.sha256` verifies |
| provenance (§5.3) | of the 3,264 run files, 3,169 are byte-identical to `runs/`; the other 95 are each listed in `TRANSFORMS.json` with an original hash equal to the source file's and a released hash equal to the extracted file's. 0 mismatches; 0 source files missing from a released run; no unclassified entry |
| exclusions | the 7 `EXCLUDED` entries are absent, including `main_claim_annotation` (blinded sheets and `key.json`) |
| spot checks (§5.4) | `rep1_o3_registered`: H1 +0.0471 [0.0190, 0.0754], Holm p 0.002; H2 +0.0571 [0.0388, 0.0763], Holm p 0.001, both "supported" (D94). `rep1_stage1`: Δ₁ +0.0000233 [−0.0041, 0.0041], p 0.982, "equivalent within ±0.01" (D96) |
| bundle B (step 5) | `checkout_state_rep1/`: the 90 changed or untracked files at `b17c131` on `research/d69-d75-followup`, with `git status`, `HEAD`, branch and a SHA-256 manifest; every copy equals its source. `scripts/stage1_power_audit.py` is `64eaa58e…5716`, D85's version (§4) |

**Blinding.** `main_rq2probe/` ships 7 annotation-sheet files, 3 of them
`key.json` unblinding keys for amendment A1 (D56–D57). All 7 are
byte-identical to the files already published in `data-runs-2026-10-19`,
so this package adds no new exposure. Those sheets can no longer serve a
blind annotation, because their keys are public. The D71 human-annotation
sheets stay withheld.

**Not published, because three decisions are open:**
1. **The write-up freeze** (PUBLISHING.md gate 1). The manuscript's
   Discussion, Limitations, abstract and contribution statement were
   drafted on 2026-10-23 (#52). The researcher has not recorded a freeze.
   The README pins the release to the commit it is built from, so the
   final build is repeated at the freeze commit.
2. **Bundle B's local paths.** The scan finds 15 absolute local paths
   (backup, release and model directories) in three of the copied
   documents (`docs/AUDIT_2026-10-17.md`, `docs/DECISIONS.md`,
   `docs/RQ3_FOLLOWUP_DESIGN.md`). Bundle B exists to be byte-identical to
   the run's code state, so sanitising it would defeat its purpose. The
   options are to retain these files pinned by hash (as D65 did for the F6
   validator), to sanitise them with a `TRANSFORMS.json`, or to leave
   bundle B out of the public release.
3. **The destination.** The repository's practice is a GitHub release
   (D67), then Zenodo for a DOI. Zenodo needs the researcher's account,
   which this machine does not have.
4. **D76's outputs** (D97) are in
   `D:\NLAutoencoders-experiments\d76-2026-10-23\`, not under the
   pipeline's `runs/`. They are not in this candidate. Including them, for
   example as a separate release asset with their hash list, is a
   decision.

**Also to update at release:** `CITATION.cff` still describes v0.1.0 (the
frozen main-run report at `4713607`). Its abstract, version, commit and
date, and a data reference for the rep1 archive, must be updated together
with the release. GitHub reports the MIT `LICENSE` as `NOASSERTION`
because of its appended third-party note. The text is correct.
