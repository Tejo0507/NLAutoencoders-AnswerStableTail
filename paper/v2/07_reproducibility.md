# 7. Reproducibility statement

*Revised candidate, 2026-10-24.*

**Plans before data.** Every confirmatory, secondary and sensitivity
analysis had a plan whose SHA-256 hash was recorded before the analysis
was computed; hashes, dates and deviations are in the decision record
(`docs/DECISIONS.md`, D66–D102), and Table 1(b) gives the timeline. The
confirmatory script refuses to run on a plan whose hash differs from
D75's.

**From outputs to text.** Every number in §4 is taken from a stored JSON
output. `paper/v2/evidence/key_numbers.json` binds each key number in the
manuscript to its output file and field, and
`scripts/verify_manuscript_numbers.py` recomputes and compares them.
Separately, `scripts/check_run_consistency.py --replication --complete`
re-checks the recorded input hashes, recomputes every reading from the
stored estimates, and checks shared numbers, internal arithmetic and
gates (169 checks). Neither tool re-runs the bootstraps.

**Figures and tables** are generated from the stored outputs by
`scripts/make_paper_figures_v2.py` and `scripts/make_paper_tables_v2.py`
(`--runs <path>` reads another checkout's outputs).

**Software and hardware.** The pipeline ran on one NVIDIA RTX 4050 Laptop
GPU (6 GB) with NF4 weights, under Python 3.14 with PyTorch 2.9.1 (CUDA
12.8 build), Transformers 5.18.0, bitsandbytes 0.50.2, scikit-learn 1.9.1,
NumPy 2.5.3 and SciPy 1.18.1 (`requirements.lock.txt`). The CPU test
suite runs on every pull request with the CPU build of the same PyTorch
release.

**Seeds.** Traces are greedy. The confirmatory bootstrap uses seed
20261019 with 2,000 replicates; Stage 1's positions, folds and bootstrap
use seed 20261020, with five further cross-validation seeds as a
robustness check; the rewording null uses seed 20261019; the
stopping-policy pilot uses seed 20261024 for its re-forcing sample, its
model fitting and its bootstrap.

**The stopping-policy pilot.** Its protocol (`configs/stopping_pilot_protocol.json`,
SHA-256 `79c51284…3dc01`) and its two run configurations are hashed in
D99, with line endings pinned so that the hashes hold on any checkout. It
ran from a clean clone of the recorded commit in a separate directory,
with hash-verified copies of its inputs; `scripts/stopping_pilot.py`
refuses to run on an unfrozen protocol or out of order. The hashes of its
outputs are listed in D101 and D102, and its supplementary analyses are
`scripts/complete_pilot_c2.py` and `scripts/describe_pilot_stops.py`.

**Run artifacts.** The main run's artifacts are published as the GitHub
release `data-runs-2026-10-19`. The replication's artifacts, its
follow-ups and the rewording null are prepared as a release candidate
(sections 7 and 8 of `docs/ARCHIVE_PLAN_REP1.md`); its identifier will be added here
when it is published. The pilot's outputs are not yet in that candidate;
whether they join it is an open release decision (D102). Blinded annotation sheets and their keys are
withheld until the human annotation is complete.
