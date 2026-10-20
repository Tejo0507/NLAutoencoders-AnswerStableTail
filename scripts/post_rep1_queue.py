"""GPU work after `rep1`, in the documented order, gated (D85). GPU and CPU.

Waits for ``rep1_runbook.py`` to finish and checks that the D75 analysis and
the D79 audit exist and were not blocked. Then:

1. D82 on ``main_nla240``: ``force_extended.py`` (GPU), then its analysis
   and the D81 audit under O256 (CPU);
2. the pre-specified ``rep1`` secondaries (CPU): the D81 audit, post-hoc
   checks, lexicon transfer and selective risk;
3. D82 on ``rep1``: forcing (GPU), analysis, and the audit under O256 (CPU);
4. with ``--stage1`` only: D83 Stage 1, positions → verbalise → reconstruct →
   force → analyse.

It stops at the first failed step. Every step is resumable or refuses to
overwrite, so a re-run continues where it stopped. Logs go to
``runs/post_rep1_queue/``.

    python scripts/post_rep1_queue.py --dry-run
    python scripts/post_rep1_queue.py --wait-pid <runbook pid> --stage1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs" / "post_rep1_queue"
R = ROOT / "runs"


def _ok_json(path: Path, forbid: str | None = None):
    if not path.exists():
        return f"{path} missing"
    if forbid and forbid in json.loads(path.read_text(encoding="utf-8")):
        return f"{path} reports '{forbid}'"
    return None


def _forcing_complete(d: Path):
    """``provenance.json`` is written at the start; only its end-of-run fields
    and a full row count mean the pass finished."""
    p = d / "provenance.json"
    if not p.exists():
        return f"{p} missing"
    prov = json.loads(p.read_text(encoding="utf-8"))
    if "forced_ext_sha256" not in prov:
        return f"{d} did not finish"
    if prov.get("n_rows") != prov.get("n_cut_off", 0) + prov.get("n_check", 0):
        return f"{d}: {prov.get('n_rows')} rows of {prov.get('n_cut_off', 0) + prov.get('n_check', 0)}"
    return None


def steps(stage1: bool):
    py = [sys.executable]
    s = lambda name: str(ROOT / "scripts" / name)  # noqa: E731
    main240 = ["--config", "main", "--run-id", "main_nla240"]
    rep = ["--config", "replication", "--forced-dir", str(R / "rep1_forced")]
    out = [
        ("ext_force_main_nla240", py + [s("force_extended.py"), *main240, "--forced-dir", str(R / "main_forced")],
         lambda: _forcing_complete(R / "main_nla240_extforced")),
        ("ext_analysis_main_nla240", py + [s("extended_forcing_analysis.py"), *main240,
                                           "--forced-dir", str(R / "main_forced")],
         lambda: _ok_json(R / "main_nla240_extforced" / "analysis.json", forbid="blocked")),
        ("info_O256_main_nla240", py + [s("information_audit.py"), *main240, "--acts-dir", str(R / "main" / "acts"),
                                        "--extra-label", str(R / "main_nla240_extforced" / "labels_O256.jsonl")],
         lambda: _ok_json(R / "main_nla240_information" / "information_O256.json")),
        ("rep1_information", py + [s("information_audit.py"), *rep],
         lambda: _ok_json(R / "rep1_information" / "information.json")),
        ("rep1_information_posthoc", py + [s("information_audit.py"), *rep, "--post-hoc"],
         lambda: _ok_json(R / "rep1_information" / "information_posthoc.json")),
        ("rep1_lexicon", py + [s("information_audit.py"), *rep, "--lexicon",
                               str(R / "main_nla240_information" / "lexicon.json")],
         lambda: _ok_json(R / "rep1_information" / "information_lexicon.json")),
        ("rep1_selective", py + [s("selective_risk_menu.py"), *rep],
         lambda: _ok_json(R / "rep1_selective" / "selective_menu.json")),
        ("ext_force_rep1", py + [s("force_extended.py"), *rep],
         lambda: _forcing_complete(R / "rep1_extforced")),
        ("ext_analysis_rep1", py + [s("extended_forcing_analysis.py"), *rep, "--confirmatory"],
         lambda: _ok_json(R / "rep1_extforced" / "analysis.json", forbid="blocked")),
        ("info_O256_rep1", py + [s("information_audit.py"), *rep, "--extra-label",
                                 str(R / "rep1_extforced" / "labels_O256.jsonl")],
         lambda: _ok_json(R / "rep1_information" / "information_O256.json")),
    ]
    if stage1:
        for sub in ("positions", "verbalise", "reconstruct", "force", "analyse"):
            out.append((f"stage1_{sub}", py + [s("stage1_offmenu.py"), sub],
                        (lambda sub=sub: _ok_json(R / "rep1_stage1" / "analysis.json")
                         if sub == "analyse" else None)))
    # D87 (approved 2026-10-20): fresh-data work first; the exploratory D82
    # pass on already-seen main_nla240 runs last. No method or rule changes.
    first = [x for x in out if not x[0].endswith("_main_nla240")]
    last = [x for x in out if x[0].endswith("_main_nla240")]
    return first + last


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                         capture_output=True, text=True).stdout
    return f'"{pid}"' in out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=None, help="the runbook's process")
    p.add_argument("--poll", type=int, default=300)
    p.add_argument("--stage1", action="store_true", help="include D83 Stage 1 (approved in D85)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    plan = steps(args.stage1)
    if args.dry_run:
        for n, cmd, _ in plan:
            print(f"{n}: {' '.join(cmd[1:])}")
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        while pid_alive(args.wait_pid):
            time.sleep(args.poll)
    for need, forbid in ((R / "rep1_o3_registered" / "o3_registered.json", "blocked"),
                         (R / "rep1_position_audit" / "position_truncation.json", None)):
        why = _ok_json(need, forbid)
        if why:
            print(f"rep1 is not finished or was blocked: {why}; nothing run", file=sys.stderr)
            return 1
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2"}
    for name, cmd, gate in plan:
        done = gate() is None if name not in ("stage1_positions",) else (R / "rep1_stage1" / "positions.jsonl").exists()
        if done and name not in ("stage1_verbalise", "stage1_reconstruct", "stage1_force"):
            print(f"{name}: already done")
            continue
        print(f"{time.strftime('%H:%M:%S')} {name}", flush=True)
        t0 = time.monotonic()
        with open(LOGS / f"{name}.log", "a", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"  exit {rc} after {time.monotonic() - t0:.0f}s", flush=True)
        why = gate()
        if rc != 0 or why:
            print(f"{name} failed (exit {rc}; {why}); stopping", file=sys.stderr)
            return 1
    print(f"{time.strftime('%H:%M:%S')} queue done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
