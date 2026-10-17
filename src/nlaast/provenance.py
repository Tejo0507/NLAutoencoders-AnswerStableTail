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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import paths

MANIFEST_NAME = "manifest.json"
#: ``partial`` is for a stage that finished what it was asked to do without
#: finishing the stage. The ``nla`` stage is the case: the verbaliser and the
#: reconstructor are each 7B-class and only one fits in VRAM, so it is run one
#: phase at a time, and a phase that succeeds must not record the stage as
#: complete - a later pass would skip the half that never ran. Only
#: ``complete`` is skippable (see ``is_complete``), so ``partial`` re-runs.
_STATUS = ("pending", "running", "complete", "partial", "failed", "skipped")


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


def stage_config_hash(data: dict[str, Any], stage: str) -> tuple[str | None, str]:
    """The config hash a stage ran under, and whether it was recorded or derived.

    Stages started since per-stage recording carry ``config_hash``. For older
    ones it is derived from ``config_hash_history``: each entry is a hash and
    the moment a later open replaced it, so the hash active when the stage
    started is the first one replaced after that moment, or the current hash
    if none was. Every stage opens the manifest immediately before starting,
    so this is exact unless two processes with *different* configurations ran
    at once - which is why it is labelled ``derived``.
    """
    entry = (data.get("stages") or {}).get(stage) or {}
    if entry.get("config_hash"):
        return entry["config_hash"], entry.get("config_hash_source", "recorded")
    started = entry.get("started")
    if not started:
        return None, "unknown"
    for h in data.get("config_hash_history") or []:
        if h.get("replaced") and h["replaced"] > started:
            return h.get("hash"), "derived"
    return data.get("config_hash"), "derived"


@contextmanager
def _file_lock(path: Path, timeout: float = 300.0) -> Iterator[None]:
    """An exclusive OS-level lock on ``path``, released if the process dies.

    ``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere - both are tied to
    the open handle, so a crashed stage cannot leave a stale lock behind.
    """
    fh = open(path, "a+b")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {path} within {timeout:.0f}s")
                time.sleep(0.05)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


@dataclass
class Manifest:
    """Per-run record. Written on every stage transition so a crash still
    leaves a readable state on disk.

    **Several stage processes share one manifest**, and the post-trace
    checklist runs CPU stages alongside a GPU one. Each process loads the file
    once, so writing back its whole in-memory copy erased every entry another
    process had added since: on `main`, `acts` overwrote the `ast`,
    `regression_tail` and `classify_correct` records. Every write therefore
    happens under a file lock, re-reads the file, and overlays only the stage
    and resource entries this process changed.
    """

    path: Path
    data: dict[str, Any]
    _touched_stages: set[str] = field(default_factory=set, repr=False)
    _touched_resources: set[str] = field(default_factory=set, repr=False)

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    #: Recorded into every stage this process starts. The run-level
    #: ``config_hash`` is whatever the most recent process opened with, so on
    #: its own it cannot say which configuration produced a given stage - on
    #: `main`, `nla` ran under a ``--set`` override that the next stage's open
    #: replaced, and the report then showed only the later hash.
    _stage_context: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, run_dir: Path, cfg_dict: dict[str, Any] | None = None,
             cfg_hash: str | None = None,
             overrides: list[str] | tuple[str, ...] | None = None) -> "Manifest":
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / MANIFEST_NAME
        with _file_lock(p.with_name(p.name + ".lock")):
            m = cls._open_locked(p, run_dir, cfg_dict, cfg_hash)
        if cfg_hash is not None:
            m._stage_context["config_hash"] = cfg_hash
            m._stage_context["config_overrides"] = list(overrides or [])
        return m

    @classmethod
    def _open_locked(cls, p: Path, run_dir: Path, cfg_dict: dict[str, Any] | None,
                     cfg_hash: str | None) -> "Manifest":
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
        m._write(data)
        return m

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def flush(self) -> None:
        with _file_lock(self._lock_path):
            on_disk = (json.loads(self.path.read_text(encoding="utf-8"))
                       if self.path.exists() else {})
            merged = on_disk or dict(self.data)
            stages = merged.setdefault("stages", {})
            for s in self._touched_stages:
                stages[s] = self.data["stages"][s]
            resources = merged.setdefault("resources", {})
            for r in self._touched_resources:
                resources[r] = self.data["resources"][r]
            merged["updated"] = utcnow()
            self._write(merged)
            self.data = merged

    # -- stage bookkeeping ------------------------------------------------

    def stage_status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "pending")

    def is_complete(self, stage: str) -> bool:
        return self.stage_status(stage) == "complete"

    def start_stage(self, stage: str, **meta: Any) -> None:
        # The commit is recorded per stage, not once per run.
        #
        # The run-level ``git`` block is captured when the manifest is first
        # created and never again, so on a long run - the pilot spanned five
        # days and dozens of commits - it names the code the *directory* was
        # made under, not the code that produced any particular result. The
        # report was attributing the whole run to a commit from before most of
        # the stages existed. Which code produced a number is the one thing a
        # reproducibility record has to get right.
        self.data["stages"][stage] = {
            "status": "running",
            "started": utcnow(),
            "monotonic_start": time.monotonic(),
            "git": git_commit(),
            **self._stage_context,
            **meta,
        }
        self._touched_stages.add(stage)
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
        self._touched_stages.add(stage)
        self.flush()

    def update_stage(self, stage: str, **fields: Any) -> None:
        """Add or replace fields on a stage's record without restarting it.

        For a derived artifact rebuilt after the fact - the stage's own
        ``started``, ``finished``, ``elapsed_s`` and ``git`` describe the run
        that produced its data and must survive.
        """
        if stage not in self.data["stages"]:
            raise KeyError(f"stage {stage!r} has no record to update")
        self.data["stages"][stage].update(fields)
        self._touched_stages.add(stage)
        self.flush()

    def record_resource(self, name: str, **meta: Any) -> None:
        """Model/dataset provenance: repo id, revision, local path, digests."""
        self.data.setdefault("resources", {})[name] = {"recorded": utcnow(), **meta}
        self._touched_resources.add(name)
        self.flush()
