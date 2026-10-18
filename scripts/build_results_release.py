"""Build the run-artifact release (D63): stage, checksum and archive.

Reads ``runs/`` and never writes there. Copies the agreed directories
byte-for-byte into a fresh output directory, adds ``release/``'s README,
notices and licence texts, writes ``SHA256SUMS.txt`` over every staged file,
and builds a deterministic ``.tar.gz`` (sorted members, fixed mtime, owner and
mode, regular files only), so the same commit and the same ``runs/`` give the
same archive SHA-256.

Every entry under ``runs/`` must be classified below, as released or as
excluded; a new run directory stops the build until it is. The output
directory must not exist.

    python scripts/build_results_release.py --out D:/NLAutoencoders-release/<name>            # draft
    python scripts/build_results_release.py --out <dir> --status final                         # after the freeze

``--status final`` refuses a dirty working tree, because the README pins the
release to ``HEAD``. Publishing is a separate, manual step (D63, D64).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"

#: name -> (classification, description, decision record)
RELEASED = {
    "main": ("registered", "The study's main run (NLA arm at 120 problems, D48)",
             "PROJECT_PLAN.md §5-§6; D46-D53"),
    "pilot": ("registered-pilot", "24-problem pilot, all stages", "PROJECT_PLAN.md §9; D44"),
    "main_f8_bf16": ("registered-test-partial", "F8 bf16 arm: bf16 target activations only", "D55"),
    "main_f8_nf4": ("registered-test-partial", "F8 NF4 re-run arm on the same 24 problems", "D55"),
    "main_f8_comparison.json": ("registered-test-partial", "F8 comparison of the two arms", "D55"),
    "main_nla240": ("exploratory-post-hoc", "NLA arm extended to 240 problems", "D54"),
    "main_rq2probe": ("exploratory",
                      "RQ2 amendment A1; its AI-produced annotation sheet is not independent",
                      "D56-D57"),
    "main_f6_audit": ("exploratory-diagnostic", "F6 np.resize label audit", "docs/F6_RESIZE_AUDIT.md"),
    "main_f6_diag": ("exploratory-diagnostic", "F6 length-correct diagnostic, 100 draws", "D58-D60"),
    "main_f6_diag_validation": ("exploratory-diagnostic", "Independent validation of main_f6_diag", "D60"),
}
EXCLUDED = {
    "calib": "calibration scratch run",
    "mlcorpus": "traces from before the chunker and parser fixes; not poolable",
    "smoke": "smoke test",
    "tiny": "synthetic test fixture, rebuilt by the test suite",
    "main_f5": "2-problem timing check for the deferred F5 sweeps",
}
#: Empty lock files every run holds while a stage writes its manifest.
SKIP_NAMES = {"manifest.json.lock"}
LICENCES = {  # release/licences/<file> -> path in the package
    "CC-BY-4.0_legalcode.txt": "LICENSE",
    "Qwen2.5-7B-Instruct_LICENSE.txt": "licences/Qwen2.5-7B-Instruct_LICENSE.txt",
    "GSM8K_MIT_LICENSE.txt": "licences/GSM8K_MIT_LICENSE.txt",
    "MATH_MIT_LICENSE.txt": "licences/MATH_MIT_LICENSE.txt",
}
TOP = "nlaast-runs"
FIXED_MTIME = 1760745600  # 2025-10-18T00:00:00Z; any fixed value makes the archive reproducible

BANNERS = {
    "draft": ("> **DRAFT — NOT FOR PUBLICATION.** Prepared {date} under decision D63 of\n"
              "> the code repository. Publication is blocked until the write-up is frozen;\n"
              "> the package is then rebuilt at the freeze commit with `--status final`\n"
              "> and re-verified. Do not cite, upload or share this draft."),
    "final": ("> Released under decision D63 of the code repository, built {date} at the\n"
              "> source commit below, which the frozen write-up reports."),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def classify(runs: Path) -> None:
    present = {p.name for p in runs.iterdir()}
    unknown = present - set(RELEASED) - set(EXCLUDED)
    if unknown:
        raise SystemExit(f"unclassified entries under {runs}: {sorted(unknown)}")
    missing = set(RELEASED) - present
    if missing:
        raise SystemExit(f"released entries missing from {runs}: {sorted(missing)}")


def stage_runs(runs: Path, stage: Path) -> list[dict]:
    table = []
    for name, (cls, what, rec) in RELEASED.items():
        src = runs / name
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
        links = [p for p in ([src] if src.is_file() else src.rglob("*")) if p.is_symlink()]
        if links:
            raise SystemExit(f"symbolic link in a released run: {links[0]}")
        kept = [p for p in files if p.name not in SKIP_NAMES]
        for p in kept:
            dst = stage / "runs" / p.relative_to(runs)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            if sha256(p) != sha256(dst):
                raise SystemExit(f"copy differs from its source: {p}")
        table.append({"path": f"runs/{name}", "classification": cls, "description": what,
                      "decision_record": rec, "n_files": len(kept),
                      "n_bytes": sum(p.stat().st_size for p in kept),
                      "skipped": sorted(p.relative_to(runs).as_posix()
                                        for p in files if p.name in SKIP_NAMES)})
    return table


def write_archive(stage: Path, archive: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(p for p in stage.rglob("*") if p.is_file()):
            ti = tarfile.TarInfo(f"{TOP}/{p.relative_to(stage).as_posix()}")
            ti.size = p.stat().st_size
            ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = FIXED_MTIME, 0o644, 0, 0, "", ""
            with open(p, "rb") as fh:
                tf.addfile(ti, fh)
    with open(archive, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
            gz.write(raw.getvalue())


def build(runs: Path, out: Path, status: str, commit: str, date: str, archive_name: str) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    classify(runs)
    stage = out / TOP
    (stage / "runs").mkdir(parents=True)
    table = stage_runs(runs, stage)

    readme = (RELEASE / "README.template.md").read_text(encoding="utf-8")
    readme = readme.replace("{STATUS_BANNER}", BANNERS[status].format(date=date))
    readme = readme.replace("{SOURCE_COMMIT}", commit)
    (stage / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    shutil.copy2(RELEASE / "NOTICE.md", stage / "NOTICE.md")
    for src, dst in LICENCES.items():
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RELEASE / "licences" / src, stage / dst)
    (stage / "RUNS.json").write_text(json.dumps({
        "status": status if status == "final"
        else "DRAFT - NOT FOR PUBLICATION (write-up not frozen)",
        "source_repository": "https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail",
        "source_commit": commit, "built": date, "licence": "CC-BY-4.0",
        "decision_record": "D63", "runs": table, "excluded": EXCLUDED,
        "excluded_files": sorted(SKIP_NAMES),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files)
    (stage / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")

    archive = out / archive_name
    write_archive(stage, archive)
    digest = sha256(archive)
    (out / f"{archive_name}.sha256").write_text(f"{digest}  {archive_name}\n",
                                                encoding="utf-8", newline="\n")
    return {"status": status, "source_commit": commit, "staged_files": len(files) + 1,
            "run_files": sum(r["n_files"] for r in table), "archive": str(archive),
            "archive_bytes": archive.stat().st_size, "archive_sha256": digest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="new directory for the package")
    ap.add_argument("--status", choices=sorted(BANNERS), default="draft")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="build date written into the README (default: today)")
    args = ap.parse_args()

    commit = git("rev-parse", "HEAD")
    if args.status == "final" and git("status", "--porcelain"):
        print("--status final needs a clean working tree: the release is pinned to HEAD",
              file=sys.stderr)
        return 1
    name = f"nlaast-runs-{'DRAFT-' if args.status == 'draft' else ''}{args.date}.tar.gz"
    print(json.dumps(build(args.runs, args.out, args.status, commit, args.date, name), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
