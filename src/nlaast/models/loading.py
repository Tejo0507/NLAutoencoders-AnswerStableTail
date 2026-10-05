"""Checkpoint acquisition and 4-bit conversion.

Three Qwen-7B-shaped checkpoints totalling 41 GB of bf16 weights have to pass
through 22 GB of disk and 6.44 GB of VRAM (PROJECT_PLAN.md section 2). The
strategy implemented here is download -> quantise to NF4 -> save -> evict the
bf16 source, one model at a time.

Quantisation is a deviation from the released checkpoints' native precision and
is treated as a threat to validity: ``scripts/run_falsification_tests.py`` measures the
activation drift it causes (F8) and the NLA stage gates on verbaliser output
integrity (Q2).
"""

from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from ..logging_utils import get

log = get(__name__)

#: Which modules bitsandbytes must leave alone.
#:
#: ``lm_head`` is skipped by default and we keep it that way for the target
#: model: quantising the output projection changes the token distribution the
#: study is measuring. For the verbaliser it is also kept, because the AV's
#: output text *is* the measurement.
SKIP_QUANT_MODULES = ["lm_head"]


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    local_path: Path
    revision: str | None
    precision: str

    def provenance(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "local_path": str(self.local_path),
            "precision": self.precision,
        }


def snapshot(repo_id: str, revision: str | None = None,
             allow_patterns: list[str] | None = None) -> Path:
    """Download (or reuse) a HuggingFace snapshot into the model cache."""
    from huggingface_hub import snapshot_download

    cache = paths.model_cache()
    cache.mkdir(parents=True, exist_ok=True)
    log.info("resolving %s (revision=%s)", repo_id, revision or "main")
    local = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=str(cache / "hub"),
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    return Path(local)


def resolve_revision(repo_id: str, revision: str | None = None) -> str | None:
    """Pin the commit sha so the manifest records what was actually used."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id, revision=revision).sha
    except Exception as exc:
        log.warning("could not resolve revision for %s: %r", repo_id, exc)
        return revision


def nf4_config(skip: list[str] | None = None):
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=skip if skip is not None else SKIP_QUANT_MODULES,
    )


def _quant_dir(repo_id: str) -> Path:
    return paths.quantised_models() / repo_id.replace("/", "__")


def ensure_nf4(
    repo_id: str,
    revision: str | None = None,
    evict_source: bool = True,
    skip_modules: list[str] | None = None,
) -> ResolvedModel:
    """Return a local NF4 checkpoint directory, building it if needed.

    ``evict_source`` deletes the bf16 snapshot afterwards. On this machine that
    is not an optimisation, it is the only way the next model fits.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = _quant_dir(repo_id)
    if (out / "config.json").exists() and any(out.glob("*.safetensors")):
        log.info("NF4 checkpoint already present: %s", out)
        return ResolvedModel(repo_id, out, revision, "nf4")

    sha = resolve_revision(repo_id, revision)
    src = snapshot(repo_id, revision)
    log.info("quantising %s -> NF4 (this loads the bf16 weights once)", repo_id)

    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=nf4_config(skip_modules),
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(src)
    tok.save_pretrained(out)

    # Carry the NLA sidecar across - it is the authoritative source for the
    # injection convention and the quantised copy is useless without it.
    for extra in ("nla_meta.yaml", "value_head.safetensors"):
        p = Path(src) / extra
        if p.exists():
            shutil.copy2(p, out / extra)
            log.info("copied %s alongside the NF4 checkpoint", extra)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if evict_source:
        _evict_snapshot(Path(src), repo_id)
    return ResolvedModel(repo_id, out, sha, "nf4")


def _evict_snapshot(snapshot_dir: Path, repo_id: str) -> None:
    """Delete the bf16 blobs for ``repo_id`` from the hub cache.

    The snapshot directory is a tree of symlinks into ``blobs/``; removing the
    snapshot alone frees nothing, so the repo's whole cache entry goes.
    """
    repo_root = snapshot_dir
    while repo_root.parent != repo_root and not repo_root.name.startswith("models--"):
        repo_root = repo_root.parent
    if not repo_root.name.startswith("models--"):
        log.warning("could not locate hub cache root for %s; not evicting", repo_id)
        return
    size = sum(f.stat().st_size for f in repo_root.rglob("*") if f.is_file())
    shutil.rmtree(repo_root, ignore_errors=True)
    log.info("evicted bf16 cache for %s (%.1f GB freed)", repo_id, size / 1e9)


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def cuda_memory() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    except Exception:
        return {}
