"""Resume the whole post-pause pipeline with one command (D86). GPU and CPU.

1. The rep1 traces stage, if the manifest does not record it complete. It
   regenerates only problems missing from ``traces.jsonl``; per-problem seeds
   make a resumed problem identical to an uninterrupted one.
2. ``rep1_runbook.py``: ast → acts → baselines → nla → forcing → D75 → D79,
   gated. Complete stages are skipped and partial ones resume.
3. ``post_rep1_queue.py --stage1``: D82 on main_nla240, the rep1
   secondaries, D82 on rep1, then Stage 1. Finished steps are skipped.

It stops at the first failure. **Do not let the machine sleep while it
runs.** A CUDA error after a wake is written by the traces stage as a
permanent ``ok: false`` row. To pause safely, run
``scripts/safe_stop.ps1``, which waits for the next finished trace and then
stops.

    python scripts/resume_rep1.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "rep1"


def run(cmd, log_name):
    log = ROOT / "runs" / "rep1_runbook" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%H:%M:%S')} {' '.join(cmd[1:])}  (log: {log})", flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs. A per-process
    request: no power setting changes, and it ends when the process exits.
    It does not prevent sleep from closing the lid or choosing Sleep."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> int:
    keep_awake()
    py = sys.executable
    st = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["stages"]["traces"]
    if st.get("status") != "complete":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        rc = run([py, str(ROOT / "scripts" / "generate_traces.py"), "--config", "replication",
                  "--no-entropy"], f"traces_resume_{stamp}.out")
        if rc != 0:
            print(f"traces stage exited {rc}; stopping", file=sys.stderr)
            return 1
    if run([py, "-u", str(ROOT / "scripts" / "rep1_runbook.py")], "runbook_resume.out") != 0:
        print("runbook stopped at a failed gate; see runs/rep1_runbook/", file=sys.stderr)
        return 1
    if run([py, "-u", str(ROOT / "scripts" / "post_rep1_queue.py"), "--stage1"],
           "queue_resume.out") != 0:
        print("queue stopped at a failed step; see runs/post_rep1_queue/", file=sys.stderr)
        return 1
    print(f"{time.strftime('%H:%M:%S')} everything done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
