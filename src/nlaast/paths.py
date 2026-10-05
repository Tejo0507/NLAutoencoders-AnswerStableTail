"""Filesystem layout.

One place that knows where things live. Everything else asks here, so moving the
model cache to another drive is a single edit (or a single environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root - this file is ``<root>/src/nlaast/paths.py``.
ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
THIRD_PARTY = ROOT / "third_party"
DOCS = ROOT / "docs"


def model_cache() -> Path:
    """Where bf16 HuggingFace weights are staged.

    The three Qwen-7B-shaped checkpoints total 41 GB in bf16, so this normally
    wants to point at a different volume from the repository. Set ``HF_HOME``,
    or ``NLAAST_MODEL_CACHE`` to override it independently of the rest of the
    HuggingFace tooling; the fallback keeps everything beside the repository so
    a fresh clone runs without configuration.
    """
    env = os.environ.get("NLAAST_MODEL_CACHE")
    if env:
        return Path(env)
    env = os.environ.get("HF_HOME")
    return Path(env) if env else ROOT.parent / "nla_models" / "hf_cache"


def quantised_models() -> Path:
    """Where 4-bit NF4 conversions are written (PROJECT_PLAN.md section 2).

    Deliberately *not* inside the download cache. Conversion needs the bf16
    source and the NF4 output on disk at the same time (15.3 GB + 5.5 GB for
    the largest checkpoint), so keeping the output on another volume removes
    that peak entirely. ``NLAAST_QUANT_DIR`` overrides.
    """
    env = os.environ.get("NLAAST_QUANT_DIR")
    if env:
        return Path(env)
    return ROOT.parent / "nla_models" / "nf4"


def run_dir(run_id: str) -> Path:
    return RUNS / run_id
