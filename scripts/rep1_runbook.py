"""The D75 replication after its traces stage, in order, gated (D80).

Runs the commands D80 lists for ``rep1``. Before each step it checks the gate
the previous step must pass (``docs/POST_TRACE_CHECKLIST.md``), and it stops
at the first failure rather than carrying on. Every stage is resumable or
refuses to overwrite, so re-running the runbook continues where it stopped.

Nothing here decides anything. The registered analysis applies its own
instrument gate and plan-hash check, and the runbook only reports them. The
extended-budget forcing pass (``docs/EXTENDED_FORCING_PLAN.md``) is **not**
run here: it is a separate experiment and starts only after these outputs
have been checked.

    python scripts/rep1_runbook.py --dry-run
    python scripts/rep1_runbook.py --wait-pid 27992
    python scripts/rep1_runbook.py --through nla

Logs go to ``runs/rep1_runbook/``. Nothing under ``runs/rep1/`` is written by
this script itself; the stages write there as they always do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "rep1"
LOGS = ROOT / "runs" / "rep1_runbook"
FORCED = ROOT / "runs" / "rep1_forced"
PLAN = ROOT / "docs" / "REPLICATION_PLAN.md"
#: The plan as amended by deviations 1-2 (D80).
PLAN_SHA256 = "e45629a7914fb4c8eefc95758014a72ab7af53c363e670c862f441c886a60761"
N_PROBLEMS = 400
N_REPRO = 200


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _stage(name: str) -> dict:
    return _json(RUN / "manifest.json").get("stages", {}).get(name, {})


def pid_alive(pid: int) -> bool:
    """Read-only check through ``tasklist``. (``os.kill(pid, 0)`` would
    terminate the process on Windows.)"""
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                         capture_output=True, text=True).stdout
    return f'"{pid}"' in out


def traces_running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                          "| ForEach-Object { $_.CommandLine }"],
                         capture_output=True, text=True).stdout
    return any("generate_traces.py" in ln and "replication" in ln for ln in out.splitlines())


# ---- gates: each returns None if passed, else the reason it failed ----


def gate_traces():
    if traces_running():
        return "generate_traces.py --config replication is still running"
    st = _stage("traces")
    if st.get("status") != "complete":
        return f"traces stage is {st.get('status')!r}, not complete"
    rows = _lines(RUN / "traces" / "traces.jsonl")
    ids = [r["problem_id"] for r in rows]
    if len(ids) != len(set(ids)):
        return "duplicate problem ids in traces.jsonl"
    if len(rows) != N_PROBLEMS:
        return f"{len(rows)} traces, expected {N_PROBLEMS}"
    print(f"  traces: {len(rows)} rows, {sum(bool(r.get('ok')) for r in rows)} ok, "
          f"{sum(bool(r.get('truncated')) for r in rows)} at the token cap")
    return None


def gate_ast():
    return None if (RUN / "ast" / "summary.json").exists() else "ast/summary.json missing"


def gate_acts():
    p = RUN / "acts" / "summary.json"
    if not p.exists():
        return "acts/summary.json missing"
    s = _json(p)
    clamped = (s.get("alignment") or {}).get("total_clamped_positions")
    if s.get("n_failed"):
        return f"{s['n_failed']} activation blocks failed"
    return None if clamped == 0 else f"alignment.total_clamped_positions = {clamped}, expected 0"


def gate_baselines():
    # probe.json is written even when the probe is skipped; the manifest's
    # rule list is where a skipped probe shows (POST_TRACE_CHECKLIST step 4)
    rules = (_stage("baselines").get("metrics") or {}).get("rules") or []
    if "probe" not in rules:
        return f"baselines rules {rules} lack the probe: re-run score_baselines.py with --force"
    bs = _json(RUN / "baselines" / "boundary_scores.json")
    if not any(s is not None for v in bs.values() for s in v.get("probe", [])):
        return "boundary_scores.json has no probe score"
    return None


def gate_nla():
    m = _stage("nla")
    phases = set((m.get("metrics") or {}).get("phases_present") or [])
    if m.get("status") != "complete" or not {"verbalise", "reconstruct"} <= phases:
        return f"nla stage {m.get('status')!r}, phases {sorted(phases)}"
    rec = [r for r in _lines(RUN / "nla" / "reconstructions.jsonl")
           if r["window_kind"] != "gaussian_control"]
    ok = sum(bool(r.get("integrity_ok")) for r in rec)
    rate = ok / len(rec) if rec else 0.0
    print(f"  nla: {len(rec)} real samples, integrity {rate:.3f}")
    return None if rate >= 0.5 else f"integrity rate {rate:.3f} < 0.5: the arm is compromised"


def gate_force():
    p = FORCED / "provenance.json"
    if not p.exists():
        return "rep1_forced/provenance.json missing"
    prov = _json(p)
    if prov.get("block_size") != 2:
        return f"forced with block size {prov.get('block_size')}, expected 2 (D80)"
    if prov.get("n_repro_written") != prov.get("n_repro"):
        return f"{prov.get('n_repro_written')} of {prov.get('n_repro')} re-forced rows written"
    print(f"  forcing: {prov.get('n_missing_written')} completed, re-forcing agreement "
          f"{prov.get('repro_agreement')} on {prov.get('n_repro_written')}")
    return None


def gate_confirmatory():
    p = ROOT / "runs" / "rep1_o3_registered" / "o3_registered.json"
    if not p.exists():
        return "rep1_o3_registered/o3_registered.json missing"
    out = _json(p)
    if "blocked" in out:
        return f"the registered analysis blocked itself: {out['blocked']}"
    return None


def gate_position():
    p = ROOT / "runs" / "rep1_position_audit" / "position_truncation.json"
    return None if p.exists() else "rep1_position_audit/position_truncation.json missing"


def steps():
    py = [sys.executable]
    s = ROOT / "scripts"
    return [
        ("ast", py + [str(s / "detect_answer_stable_tail.py"), "--config", "replication", "--sweep"],
         gate_ast, None),
        ("acts", py + [str(s / "extract_activations.py"), "--config", "replication",
                       "--layers", "14", "20", "24"], gate_acts, None),
        ("baselines", py + [str(s / "score_baselines.py"), "--config", "replication"],
         gate_baselines, None),
        ("nla", py + [str(s / "run_autoencoder.py"), "--config", "replication"], gate_nla, None),
        ("force", py + [str(s / "force_all_boundaries.py"), "--config", "replication",
                        "--menu-only", "--n-repro", str(N_REPRO)], gate_force, None),
        ("confirmatory", py + [str(s / "reanalyse_o3_registered.py"), "--config", "replication",
                               "--forced-dir", str(FORCED), "--confirmatory", str(PLAN),
                               "--plan-sha256", PLAN_SHA256], gate_confirmatory,
         ROOT / "runs" / "rep1_o3_registered" / "o3_registered.json"),
        ("position_audit", py + [str(s / "position_truncation_audit.py"), "--config", "replication",
                                 "--forced-dir", str(FORCED), "--confirmatory"], gate_position,
         ROOT / "runs" / "rep1_position_audit" / "position_truncation.json"),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=None,
                   help="wait for this process (the traces stage) to exit first")
    p.add_argument("--poll", type=int, default=300, help="seconds between checks")
    p.add_argument("--through", default=None, help="stop after this step")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    plan = steps()
    names = [n for n, *_ in plan]
    if args.through and args.through not in names:
        print(f"--through must be one of {names}", file=sys.stderr)
        return 1
    if args.dry_run:
        for n, cmd, *_ in plan:
            print(f"{n}: {' '.join(cmd)}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        while pid_alive(args.wait_pid):
            time.sleep(args.poll)
        print(f"{time.strftime('%H:%M:%S')} PID {args.wait_pid} has exited")
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA256:
        print(f"{PLAN} no longer hashes to the D80 value; stopping", file=sys.stderr)
        return 1
    why = gate_traces()
    if why:
        print(f"gate before ast failed: {why}", file=sys.stderr)
        return 1

    env = {**os.environ, "PYTHONUTF8": "1"}
    for name, cmd, gate, output in plan:
        if output is not None and output.exists():
            print(f"{name}: output exists, not re-run")
        else:
            print(f"{time.strftime('%H:%M:%S')} {name}: {' '.join(cmd[1:])}", flush=True)
            t0 = time.monotonic()
            with open(LOGS / f"{name}.log", "a", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"  exit {rc} after {time.monotonic() - t0:.0f}s", flush=True)
            if rc not in (0, 2) or (rc == 2 and name != "confirmatory"):
                print(f"{name} failed (exit {rc}); see {LOGS / (name + '.log')}", file=sys.stderr)
                return 1
        why = gate()
        if why:
            print(f"gate after {name} failed: {why}", file=sys.stderr)
            return 1
        if name == args.through:
            break
    print(f"{time.strftime('%H:%M:%S')} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
