"""Build the run-artifact release (D63): stage, sanitise, checksum and archive.

Reads ``runs/`` and never writes there. Copies the agreed directories into a
fresh output directory, adds ``release/``'s README, notices and licence
texts, writes ``SHA256SUMS.txt`` over every staged file, and builds a
deterministic ``.tar.gz`` (sorted members, fixed mtime, owner and mode,
regular files only), so the same commit and the same ``runs/`` give the same
archive SHA-256.

**Release-only path sanitisation (D65).** The runs record the absolute paths
of the machine they ran on (stage outputs, data sources, the interpreter,
model and cache directories). In the *staged copy only*, the root of each
local absolute path is replaced and the rest is kept:

* a path inside the repository -> ``<repo>`` + its repository-relative part
  (``<repo>\\runs\\main\\traces``);
* any other local path -> ``<local>`` + everything below the first directory
  under the drive (``<local>\\nf4\\<model dir>``); below ``Users``/``home``
  the user's directory is removed too.

A bare root (a drive and one directory) becomes ``<repo>`` or ``<local>``
where it is a whole value. What a path points at stays readable; which disk
and which local folder it was on does not. Files in ``RETAINED`` are
released unmodified, each pinned to its exact SHA-256 and justified there. Every file that is changed is listed in ``TRANSFORMS.json``
with its original and released SHA-256, the JSON fields or log lines
changed and the number of replacements, so a released file can be matched to
its original (and to any original checksum list such as
``main_nla240/SOURCE_SHA256SUMS.txt``). Every other file is byte-identical
to its source. Files in ``runs/`` are never modified.

**Gate.** After staging, ``scan_release.scan`` runs over the package; any
remaining local path, home directory, local identity, e-mail address or
token stops the build (exit 1) before an archive exists. Content the
transformation cannot reach, such as metadata inside ``.npz`` files, is
caught by that scan rather than shipped.

Every entry under ``runs/`` must be classified below, as released or as
excluded; a new run directory stops the build until it is. The output
directory must not exist.

    python scripts/build_results_release.py --out D:/NLAutoencoders-release/<name>            # draft
    python scripts/build_results_release.py --out <dir> --status final                         # after the freeze

``--status final`` refuses a dirty working tree, because the README pins the
release to ``HEAD``. Publishing is a separate, manual step (D63-D65).

A classified entry that lives outside ``--runs`` (an experiment run in an
isolated tree, such as D76's) is added with ``--extra-run NAME=PATH``; it is
staged under ``runs/NAME``, sanitised and scanned like every other run, and
its source is only read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
sys.path.insert(0, str(ROOT / "scripts"))

import scan_release  # noqa: E402

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
    # --- after the D63 release: post-hoc reanalysis of main (D68-D74) ---
    "main_forced": ("post-hoc", "48-token forced answers completed at every boundary of main",
                    "D69"),
    "main_o3_registered": ("post-hoc-corrected", "O3 rescored on the registered outcome, N = 120",
                           "D69; D91 (plan provenance gap)"),
    "main_nla240_o3_registered": ("post-hoc-corrected",
                                  "O3 rescored on the registered outcome, 240-problem arm",
                                  "D69; D91 (plan provenance gap)"),
    "main_claims_audit": ("post-hoc", "Audit of RQ2's claim-level results", "D68"),
    "main_claim_checks": ("post-hoc", "String checks of explanation content against the trace",
                          "D71; docs/REANALYSIS_2026-10-19.md"),
    "main_paraphrase_qc": ("post-hoc", "Quality check of the registered paraphrase null",
                           "D70; docs/REANALYSIS_2026-10-19.md"),
    "main_corrected_tests": ("post-hoc-corrected", "F10 and RQ2 recomputed at the right unit",
                             "D72; docs/REANALYSIS_2026-10-19.md"),
    "main_coverage": ("exploratory", "Robustness of O3 to seed, layer and subject", "D73"),
    "main_rq3_pilot": ("exploratory-pilot", "RQ3 follow-up pilot", "D74; D77 (no-go)"),
    "main_rq3_pilot_stage0": ("exploratory-pilot", "RQ3 pilot Stage 0 (no-go)", "D77"),
    # --- position audit, information audit, leakage-resistant design ---
    "main_position_audit": ("exploratory", "Position-only control and truncation audit, main",
                            "D79"),
    "main_nla240_position_audit": ("exploratory",
                                   "Position-only control and truncation audit, 240 arm", "D79"),
    "main_information": ("exploratory", "Information audit of the readout, main", "D81"),
    "main_nla240_information": ("exploratory",
                                "Information audit, 240 arm; lexicon frozen for rep1; O256 audit",
                                "D81; D84; D96"),
    "main_selective": ("post-hoc", "Selective risk at matched coverage, main menu",
                       "D83 deviation 1"),
    "main_nla240_selective": ("post-hoc", "Selective risk at matched coverage, 240 arm",
                              "D83 deviation 1"),
    "main_offmenu_stage0": ("exploratory", "Stage 0: cheap signals at every boundary of main",
                            "D83; D84"),
    "main_nla240_extforced": ("exploratory", "256-token forcing pass on the 240 arm (O48 seen)",
                              "D82; D96"),
    "stage1_power_audit": ("planning", "Stage 1 power audit on stored data", "D85"),
    "bench_force_b0": ("engineering-benchmark", "Forcing batch benchmark, no batching", "D77"),
    "bench_force_b8": ("engineering-benchmark", "Forcing batch benchmark, batch 8", "D77"),
    "bench_force_b16": ("engineering-benchmark", "Forcing batch benchmark, batch 16", "D77"),
    "bench_decode_batch": ("engineering-benchmark", "Decode throughput by batch size", "D78"),
    # --- the confirmatory replication (D75) ---
    "rep1": ("confirmatory", "The replication run: 400 fresh problems, every stage", "D75-D80"),
    "rep1_forced": ("confirmatory", "Registered 48-token menu forcing and 200 re-forced checks",
                    "D80"),
    "rep1_o3_registered": ("confirmatory", "The D75 confirmatory analysis (H1, H2)", "D75; D94"),
    "rep1_position_audit": ("pre-specified-secondary", "D79 position and truncation audit, rep1",
                            "D79; D94"),
    "rep1_information": ("pre-specified-secondary",
                         "Information audit, lexicon transfer and O256 audit, rep1",
                         "D81; D95"),
    "rep1_selective": ("pre-specified-secondary", "Selective risk at matched coverage, rep1",
                       "D83 deviation 1; D95"),
    "rep1_extforced": ("pre-specified-sensitivity", "256-token forcing pass on rep1", "D82; D95"),
    "rep1_stage1": ("pre-specified", "Stage 1: NLA text against a same-label probe off the menu",
                    "D83; D85; D96"),
    "rep1_runbook": ("logs", "Runbook and resume logs of the rep1 pipeline", "D80; D86-D96"),
    "post_rep1_queue": ("logs", "Logs of the post-rep1 queue", "D85; D87; D95-D96"),
    # --- after the replication: run outside the pipeline checkout (--extra-run) ---
    "main_paraphrase_llm": ("exploratory",
                            "D76 rewording null for RQ2: 60 explanations, 246 claims; the "
                            "rewordings' meaning is unvalidated (the human sheet is blank)",
                            "D76; D97"),
    "main_paraphrase_llm_logs": ("logs", "D76 phase logs and its input and output SHA-256 lists",
                                 "D97"),
    "pilot_cal": ("exploratory-pilot",
                  "Stopping-policy pilot calibration set: 300 fresh problems, traces, layer-20 "
                  "activations, 256-token forced answers at every boundary, policy scores",
                  "D99; D100; D101"),
    "pilot_eval": ("exploratory-pilot",
                   "Stopping-policy pilot evaluation set: 150 fresh problems, as pilot_cal",
                   "D99; D100; D101"),
    "pilot_models": ("exploratory-pilot",
                     "P0 and P1 as fitted on rep1 Stage 1 and frozen (a pickle; its SHA-256 is "
                     "in models.json)", "D99; D101"),
    "pilot_calibration": ("exploratory-pilot", "SGR thresholds, frozen and hashed before evaluation",
                          "D99; D101"),
    "pilot_evaluation": ("exploratory-pilot",
                         "The pilot's one evaluation pass; not a safety certification", "D101"),
    "pilot_evaluation_supplement": ("exploratory-pilot",
                                    "The paired P1 - P0 saving that evaluate skipped",
                                    "D101"),
    "pilot_posthoc": ("post-hoc", "Where P1 stops on the evaluation set (descriptive)", "D102"),
    "pilot_logs": ("logs", "The pilot's orchestrator and step logs, and its input and output "
                   "SHA-256 lists", "D99-D102"),
    "rep1_checkout_state_manifest": ("provenance",
                                     "SHA-256 list, git status, HEAD and branch of the code "
                                     "state every rep1 output records; the files themselves "
                                     "are kept offline (they hold local paths)",
                                     "D94; docs/ARCHIVE_PLAN_REP1.md"),
}
EXCLUDED = {
    "calib": "calibration scratch run",
    "mlcorpus": "traces from before the chunker and parser fixes; not poolable",
    "smoke": "smoke test",
    "tiny": "synthetic test fixture, rebuilt by the test suite",
    "main_f5": "2-problem timing check for the deferred F5 sweeps",
    "main_rq3_pilot_smoke": "smoke test of the RQ3 pilot",
    "main_claim_annotation": ("blinded human-annotation sheets and their unblinding key "
                              "(key.json); withheld until the human annotation "
                              "(docs/CLAIM_VALIDATION_PROTOCOL.md) is complete, so that "
                              "annotators stay blind"),
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

# --- sanitisation -----------------------------------------------------------

#: Names the repository's directory had when the runs were made (the public
#: repository name) and has now; a path through one of them maps to <repo>.
REPO_DIR_NAMES = {"NLAutoencoders-AnswerStableTail", ROOT.name}
#: A directory whose child is a user's own folder, which goes too.
HOME_DIRS = {"users", "home"}
#: Files released unmodified although they hold a local path, each pinned to
#: its exact SHA-256 so that any other content fails the gate (D65).
RETAINED = {
    "runs/main_f6_diag_validation/validate_f6_diag.py": {
        "sha256": "7a9562bf682803b0157653edc58f3068275514d4cdcbc3f22baae2aeeef78115",
        "reason": "The independent F6 validator's source, listed by hash in that run's own "
                  "SHA256SUMS.txt. Its only local path is the repository root it ran from "
                  "(drive letter and the public repository name), in the constant R. "
                  "Rewriting it would publish validator code that never ran.",
    },
}
#: Files sanitised as text; anything else is only scanned.
TEXT_EXT = {".json", ".jsonl", ".log", ".out", ".err", ".txt", ".yaml", ".yml", ".md",
            ".csv", ".py"}
_COMP = r"[^\\/\"'\s<>|:*?]+"
#: In JSON text a real backslash is written ``\\``; a lone ``\`` starts an
#: escape such as ``\n``, which must never be read as a separator (model text
#: contains "s:\n" thousands of times). Elsewhere one or two backslashes.
_SEP = {"json": r"(?:\\\\|/)", "text": r"(?:\\\\|\\|/)"}


def _path_rx(kind: str) -> re.Pattern:
    sep = _SEP[kind]
    return re.compile(rf"(?<![A-Za-z0-9])(?P<drive>[A-Za-z]):(?P<s1>{sep})(?P<c1>{_COMP})"
                      rf"(?P<s2>{sep})(?P<c2>{_COMP})")


_RX = {k: _path_rx(k) for k in _SEP}
#: A bare root - a drive and one directory, e.g. a disk-usage record of the
#: repository itself. Matched only where it is a whole value (a complete JSON
#: string, or a token delimited by quotes, whitespace or brackets), because a
#: drive-like fragment inside model text ("s:\\frac") must stay untouched.
_ROOT_RX = {
    "json": re.compile(rf'(?<=")(?P<drive>[A-Za-z]):{_SEP["json"]}(?P<c1>{_COMP})(?=")'),
    "text": re.compile(rf"(?<=[\"'\s=(])(?P<drive>[A-Za-z]):{_SEP['text']}(?P<c1>{_COMP})"
                       rf"(?=[\"'\s),]|$)", re.M),
}


def _replace(m: re.Match) -> str:
    """The root of one local path, replaced; the rest of the path is kept."""
    c1, s2, c2 = m["c1"], m["s2"], m["c2"]
    if c1 in REPO_DIR_NAMES:
        return f"<repo>{s2}{c2}"
    if c1.lower() in HOME_DIRS:
        return "<local>"            # drops the user's own folder (c2) as well
    return f"<local>{s2}{c2}"


def _replace_root(m: re.Match) -> str:
    return "<repo>" if m["c1"] in REPO_DIR_NAMES else "<local>"


def sanitise_text(text: str, kind: str) -> tuple[str, int]:
    text, n = _RX[kind].subn(_replace, text)
    text, k = _ROOT_RX[kind].subn(_replace_root, text)
    return text, n + k


def _string_fields(obj, key: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_fields(v, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _string_fields(v, f"{key}[{i}]")
    elif isinstance(obj, str):
        yield key, obj


def changed_fields(before: str, after: str, jsonl: bool) -> list[str]:
    """The JSON fields whose value the sanitisation changed (row index first for JSONL)."""
    if jsonl:
        out = []
        for i, (a, b) in enumerate(zip(before.splitlines(), after.splitlines())):
            if a != b:
                out += [f"[row {i}].{f}" for f in changed_fields(a, b, False)]
        return out
    a = dict(_string_fields(json.loads(before)))
    b = dict(_string_fields(json.loads(after)))
    if a.keys() != b.keys():
        raise SystemExit("sanitisation changed a JSON document's structure")
    return [k for k in a if a[k] != b[k]]


def sanitise_file(path: Path) -> dict | None:
    """Sanitise one staged file in place; a record of the change, or None."""
    if path.suffix.lower() not in TEXT_EXT:
        return None
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    kind = "json" if path.suffix.lower() in {".json", ".jsonl"} else "text"
    new, n = sanitise_text(text, kind)
    if not n:
        return None
    rec = {"original_sha256": hashlib.sha256(raw).hexdigest(), "replacements": n}
    if kind == "json":
        rec["fields"] = changed_fields(text, new, path.suffix.lower() == ".jsonl")
        if path.suffix.lower() == ".json":
            json.loads(new)
        else:
            for line in new.splitlines():
                if line.strip():
                    json.loads(line)
    else:
        rec["lines"] = [i + 1 for i, (a, b) in enumerate(zip(text.splitlines(), new.splitlines()))
                        if a != b]
    path.write_bytes(new.encode("utf-8"))
    rec["released_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rec


# --- build ------------------------------------------------------------------

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


def classify(runs: Path, extra: dict[str, Path] | None = None) -> None:
    """Every entry under `runs`, and every extra run, must be classified.

    An extra run is a released entry whose files live outside `runs` (for
    example an experiment run in an isolated tree). It must be in RELEASED,
    must exist, and must not also be present under `runs`.
    """
    extra = extra or {}
    present = {p.name for p in runs.iterdir()}
    unknown = present - set(RELEASED) - set(EXCLUDED)
    if unknown:
        raise SystemExit(f"unclassified entries under {runs}: {sorted(unknown)}")
    for name, path in extra.items():
        if name not in RELEASED:
            raise SystemExit(f"unclassified extra run: {name}")
        if name in present:
            raise SystemExit(f"{name} is both under {runs} and given as an extra run")
        if not path.exists():
            raise SystemExit(f"extra run {name} missing: {path}")
    missing = set(RELEASED) - present - set(extra)
    if missing:
        raise SystemExit(f"released entries missing from {runs}: {sorted(missing)}")


def stage_runs(runs: Path, stage: Path, extra: dict[str, Path] | None = None
               ) -> tuple[list[dict], dict[str, dict]]:
    extra = extra or {}
    table, transforms = [], {}
    for name, (cls, what, rec) in RELEASED.items():
        src = extra.get(name, runs / name)
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
        links = [p for p in ([src] if src.is_file() else src.rglob("*")) if p.is_symlink()]
        if links:
            raise SystemExit(f"symbolic link in a released run: {links[0]}")
        kept = [p for p in files if p.name not in SKIP_NAMES]
        n_changed = 0

        def released_path(p: Path, src: Path = src, name: str = name) -> str:
            return name if src.is_file() else f"{name}/{p.relative_to(src).as_posix()}"

        for p in kept:
            rel = released_path(p)
            dst = stage / "runs" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            if sha256(p) != sha256(dst):
                raise SystemExit(f"copy differs from its source: {p}")
            if f"runs/{rel}" in RETAINED:
                continue
            change = sanitise_file(dst)
            if change:
                if change["original_sha256"] != sha256(p):
                    raise SystemExit(f"sanitised a file that is not its source: {p}")
                transforms[f"runs/{rel}"] = change
                n_changed += 1
        table.append({"path": f"runs/{name}", "classification": cls, "description": what,
                      "decision_record": rec, "n_files": len(kept),
                      "n_files_sanitised": n_changed,
                      "n_bytes_original": sum(p.stat().st_size for p in kept),
                      "skipped": sorted(released_path(p)
                                        for p in files if p.name in SKIP_NAMES)})
    return table, transforms


#: Scan categories a RETAINED file may show; anything else in it still fails.
_PATH_CATEGORIES = {"abs_windows_path", "abs_windows_root"}


def gate(stage: Path, identities: set[str]) -> None:
    """Refuse to archive a package in which the scan still finds anything.

    A RETAINED file passes only with its pinned content and only for local
    path findings.
    """
    _, findings = scan_release.scan(stage, identities)
    for name, rule in RETAINED.items():
        if name in findings and set(findings[name]) <= _PATH_CATEGORIES \
                and sha256(stage / name) == rule["sha256"]:
            del findings[name]
    if findings:
        lines = [f"  {name}: {', '.join(sorted(cats))}" for name, cats in sorted(findings.items())]
        raise SystemExit("sensitive-data scan failed after sanitisation:\n" + "\n".join(lines))


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


def build(runs: Path, out: Path, status: str, commit: str, date: str, archive_name: str,
          identities: set[str] | None = None, extra: dict[str, Path] | None = None) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    classify(runs, extra)
    stage = out / TOP
    (stage / "runs").mkdir(parents=True)
    table, transforms = stage_runs(runs, stage, extra)

    readme = (RELEASE / "README.template.md").read_text(encoding="utf-8")
    readme = readme.replace("{STATUS_BANNER}", BANNERS[status].format(date=date))
    readme = readme.replace("{SOURCE_COMMIT}", commit)
    readme = readme.replace("{N_SANITISED}", str(len(transforms)))
    (stage / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    shutil.copy2(RELEASE / "NOTICE.md", stage / "NOTICE.md")
    for src, dst in LICENCES.items():
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RELEASE / "licences" / src, stage / dst)
    (stage / "TRANSFORMS.json").write_text(json.dumps({
        "what": "Release-only path sanitisation (D65). Each listed file differs from the "
                "original run file only in the roots of local absolute paths: a path inside "
                "the repository starts '<repo>', any other local path '<local>'. "
                "original_sha256 is the run file as it is in runs/ and in the backups.",
        "placeholders": {
            "<repo>": "the repository's root directory on the machine that ran the stage",
            "<local>": "a machine-local directory outside the repository: the drive and "
                       "the first directory below it (and, under Users or home, the user's "
                       "folder) are removed; the rest of the path is kept",
        },
        "n_files": len(transforms),
        "files": transforms,
        "retained_unmodified": {k: v for k, v in RETAINED.items() if (stage / k).exists()},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    (stage / "RUNS.json").write_text(json.dumps({
        "status": status if status == "final"
        else "DRAFT - NOT FOR PUBLICATION (write-up not frozen)",
        "source_repository": "https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail",
        "source_commit": commit, "built": date, "licence": "CC-BY-4.0",
        "decision_record": "D63-D65", "runs": table, "excluded": EXCLUDED,
        "excluded_files": sorted(SKIP_NAMES),
        "sanitised_files": "TRANSFORMS.json",
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    gate(stage, scan_release.local_identities([]) if identities is None else identities)

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files)
    (stage / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")

    archive = out / archive_name
    write_archive(stage, archive)
    digest = sha256(archive)
    (out / f"{archive_name}.sha256").write_text(f"{digest}  {archive_name}\n",
                                                encoding="utf-8", newline="\n")
    return {"status": status, "source_commit": commit, "staged_files": len(files) + 1,
            "run_files": sum(r["n_files"] for r in table), "sanitised_files": len(transforms),
            "archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": digest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="new directory for the package")
    ap.add_argument("--status", choices=sorted(BANNERS), default="draft")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="build date written into the README (default: today)")
    ap.add_argument("--extra-run", action="append", default=[], metavar="NAME=PATH",
                    help="a classified released entry whose files live outside --runs "
                         "(repeatable); read only")
    args = ap.parse_args()
    extra = {}
    for item in args.extra_run:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            print(f"--extra-run expects NAME=PATH, got {item!r}", file=sys.stderr)
            return 1
        extra[name] = Path(path)

    commit = git("rev-parse", "HEAD")
    if args.status == "final" and git("status", "--porcelain"):
        print("--status final needs a clean working tree: the release is pinned to HEAD",
              file=sys.stderr)
        return 1
    name = f"nlaast-runs-{'DRAFT-' if args.status == 'draft' else ''}{args.date}.tar.gz"
    print(json.dumps(build(args.runs, args.out, args.status, commit, args.date, name,
                           extra=extra), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
