"""Run provenance: environment capture, manifests, resumable stage state.

Every run preserves its config, seed, model identifiers, dataset version, code
version, software environment, timestamps, output location and metrics. This
module is the only place that writes that record, so there is one format rather
than thirteen.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

MANIFEST_NAME = "manifest.json"
_STATUS = ("pending", "running", "complete", "failed", "skipped")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=paths.ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def environment() -> dict[str, Any]:
    """Hardware and software snapshot. Imports torch lazily - the `data` stage should
    not need CUDA just to parse a dataset."""
    info: dict[str, Any] = {
        "timestamp": utcnow(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception as exc:  # pragma: no cover - psutil is a hard dep but be safe
        info["ram_error"] = repr(exc)

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        info["torch_error"] = repr(exc)

    for mod in ("transformers", "bitsandbytes", "numpy", "scipy", "sklearn", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


@dataclass
class Manifest:
    """Per-run record. Written on every stage transition so a crash still
    leaves a readable state on disk."""

    path: Path
    data: dict[str, Any]

    @classmethod
    def open(cls, run_dir: Path, cfg_dict: dict[str, Any] | None = None,
             cfg_hash: str | None = None) -> "Manifest":
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / MANIFEST_NAME
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = {
                "created": utcnow(),
                "run_dir": str(run_dir),
                "git": git_commit(),
                "environment": environment(),
                "stages": {},
                "resources": {},
            }
        if cfg_dict is not None:
            data["config"] = cfg_dict
        if cfg_hash is not None:
            # A changed config hash on an existing run is a real hazard: stage
            # outputs on disk were produced under different settings. Record
            # the history rather than silently overwriting.
            prev = data.get("config_hash")
            if prev and prev != cfg_hash:
                data.setdefault("config_hash_history", []).append(
                    {"hash": prev, "replaced": utcnow()}
                )
            data["config_hash"] = cfg_hash
        data["updated"] = utcnow()
        m = cls(path=p, data=data)
        m.flush()
        return m

    def flush(self) -> None:
        self.data["updated"] = utcnow()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    # -- stage bookkeeping ------------------------------------------------

    def stage_status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "pending")

    def is_complete(self, stage: str) -> bool:
        return self.stage_status(stage) == "complete"

    def start_stage(self, stage: str, **meta: Any) -> None:
        self.data["stages"][stage] = {
            "status": "running",
            "started": utcnow(),
            "monotonic_start": time.monotonic(),
            **meta,
        }
        self.flush()

    def finish_stage(self, stage: str, status: str = "complete", **meta: Any) -> None:
        if status not in _STATUS:
            raise ValueError(f"status must be one of {_STATUS}, got {status!r}")
        entry = self.data["stages"].setdefault(stage, {})
        start = entry.pop("monotonic_start", None)
        entry.update(
            {
                "status": status,
                "finished": utcnow(),
                "elapsed_s": round(time.monotonic() - start, 1) if start else None,
                **meta,
            }
        )
        self.flush()

    def record_resource(self, name: str, **meta: Any) -> None:
        """Model/dataset provenance: repo id, revision, local path, digests."""
        self.data.setdefault("resources", {})[name] = {"recorded": utcnow(), **meta}
        self.flush()
