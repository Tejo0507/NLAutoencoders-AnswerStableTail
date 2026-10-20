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


def other_gpu_memory() -> list[tuple[str, int]]:
    """Processes other than this one holding over 100 MiB of dedicated GPU
    memory (Windows performance counters). On this 6.1 GB card the 5.6 GB
    model spills to shared memory if anything else holds VRAM, which made
    traces 10-20x slower (D89). Read-only: nothing is stopped."""
    if sys.platform != "win32":
        return []
    cmd = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | "
           "Where-Object { $_.CookedValue -gt 100MB } | "
           "ForEach-Object { $_.InstanceName + ' ' + [math]::Round($_.CookedValue/1MB) }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout
    found = []
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0].startswith("pid_"):
            pid = parts[0].split("_")[1]
            name = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName"],
                                  capture_output=True, text=True).stdout.strip()
            found.append((f"{name or '?'} (PID {pid})", int(parts[1])))
    return found


#: D90: return cached GPU blocks before the 6.1 GB card fills. Without it the
#: model's KV caches pushed it into shared system memory (memory-bound at
#: ~29 W, 1.2-1.5 s/token); with it, 66 W and ~0.2 s/token. The allocator
#: option changes no kernel, seed or result.
ALLOC_CONF = "garbage_collection_threshold:0.8"


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    keep_awake()
    others = other_gpu_memory()
    if others:
        print("WARNING: other processes hold GPU memory, so the model will spill to "
              "shared memory and run 10-20x slower (D89): "
              + ", ".join(f"{n} {m} MiB" for n, m in others)
              + ". Close them (for example OMEN Command Center) and re-run.", flush=True)
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
