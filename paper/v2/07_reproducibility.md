# 7. Reproducibility statement (draft v1, 2026-10-23)

- **Plans before data.** Every confirmatory, secondary and sensitivity
  analysis has a plan hashed before it was computed. The hashes, dates and
  deviations are in `docs/DECISIONS.md` (D66–D96), and Table 1(b) gives the
  timeline. `reanalyse_o3_registered.py --confirmatory` refuses to run on a
  plan whose hash differs from D75's.
- **Numbers from outputs.** Every number in §4 comes from a JSON output
  named in D94–D96. `scripts/check_run_consistency.py --replication
  --complete` re-checks the recorded input hashes, recomputes every reading
  from the stored estimates, and checks shared numbers, internal arithmetic,
  gates and 11 listed numbers in the decision record (169 checks). It does
  not recompute bootstrap estimates, and it does not read this manuscript.
  The manuscript's numbers were copied from the outputs by hand and checked
  by review against the claim matrix.
- **Figures and tables** are generated from the stored outputs:
  `scripts/make_paper_figures_v2.py` and `scripts/make_paper_tables_v2.py`
  (`--runs <path>` reads another checkout's outputs).
- **Code and environment.** The pipeline ran on one RTX 4050 Laptop GPU
  (6 GB) with NF4 weights. Package versions are pinned in
  `requirements.lock.txt`, and the CPU test suite runs on every pull request.
- **Run artifacts.** The main run's artifacts are published as the GitHub
  release `data-runs-2026-10-19` (D67). The replication's artifacts
  (`rep1` and its follow-ups) are described in `docs/ARCHIVE_PLAN_REP1.md`;
  their release identifier is recorded here once it is published and
  verified. Blinded annotation sheets are withheld until annotation is
  complete.
