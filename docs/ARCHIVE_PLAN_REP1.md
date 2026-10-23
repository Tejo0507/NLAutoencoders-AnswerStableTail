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
