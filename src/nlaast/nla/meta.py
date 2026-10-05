"""Access to the vendored upstream implementation and to checkpoint sidecars.

Every NLA constant in this study - the injection token id and its required
neighbours, the injection scale, the MSE scale, both prompt templates - is read
from the checkpoint's own ``nla_meta.yaml``. None is hardcoded. Upstream's
docstrings are explicit that drift in any of them produces silent nonsense
rather than an error, so the sidecar is treated as the single source of truth.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .. import paths

VENDOR_DIR = paths.THIRD_PARTY / "nla_inference"
VENDOR_FILE = VENDOR_DIR / "nla_inference.py"


@lru_cache(maxsize=1)
def upstream():
    """Import the vendored ``nla_inference`` module.

    Loaded by path rather than added to ``sys.path`` so a stray
    ``nla_inference`` elsewhere on the path cannot shadow the vendored copy
    whose provenance is recorded in ``third_party/nla_inference/NOTICE.md``.
    """
    if not VENDOR_FILE.exists():
        raise FileNotFoundError(
            f"vendored upstream missing at {VENDOR_FILE}. Run "
            f"`python scripts/prepare_checkpoints.py --vendor-only` to restore it."
        )
    spec = importlib.util.spec_from_file_location("_nlaast_vendored_nla", VENDOR_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {VENDOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_nlaast_vendored_nla"] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class NLAMeta:
    """Parsed ``nla_meta.yaml``."""

    role: str
    d_model: int
    extraction_layer_index: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    injection_scale: float | None
    mse_scale: float | None
    av_template: str
    ar_template: str
    critic_suffix_ids: tuple[int, ...] | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["critic_suffix_ids"] = list(self.critic_suffix_ids or ())
        return d


def load_meta(checkpoint_dir: Path | str) -> NLAMeta:
    path = Path(checkpoint_dir) / "nla_meta.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no nla_meta.yaml in {checkpoint_dir}. This sidecar carries the "
            f"injection convention and the checkpoint cannot be used without it."
        )
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokens = meta["tokens"]
    extraction = meta.get("extraction", {})
    templates = meta.get("prompt_templates", {})
    d_model = int(meta["d_model"])

    suffix = tokens.get("critic_suffix_ids")
    return NLAMeta(
        role=meta["role"],
        d_model=d_model,
        extraction_layer_index=int(meta["extraction_layer_index"]),
        injection_char=tokens["injection_char"],
        injection_token_id=int(tokens["injection_token_id"]),
        injection_left_neighbor_id=int(tokens["injection_left_neighbor_id"]),
        injection_right_neighbor_id=int(tokens["injection_right_neighbor_id"]),
        injection_scale=(
            None if extraction.get("injection_scale") is None
            else float(extraction["injection_scale"])
        ),
        mse_scale=(
            None if extraction.get("mse_scale") is None
            else float(extraction["mse_scale"])
        ),
        av_template=templates.get("av") or templates.get("actor", ""),
        ar_template=templates.get("ar") or templates.get("critic", ""),
        critic_suffix_ids=tuple(int(t) for t in suffix) if suffix else None,
        source=str(path),
    )


def check_consistency(av: NLAMeta, ar: NLAMeta, target_layer: int, target_d_model: int) -> dict:
    """Verify the AV, the AR and the target model actually belong together.

    The autoencoders are trained against one model at one layer and are not
    transferable (Project_Review_II.md section 2.1.8). A mismatch here would not
    raise anywhere downstream; it would just produce confident, meaningless
    verbalisations, which is the worst possible failure for this study.
    """
    problems: list[str] = []
    if av.d_model != ar.d_model:
        problems.append(f"AV d_model {av.d_model} != AR d_model {ar.d_model}")
    if av.d_model != target_d_model:
        problems.append(f"AV d_model {av.d_model} != target d_model {target_d_model}")
    if av.extraction_layer_index != target_layer:
        problems.append(
            f"checkpoint was trained on layer {av.extraction_layer_index} but "
            f"the config extracts from layer {target_layer}"
        )
    if av.extraction_layer_index != ar.extraction_layer_index:
        problems.append("AV and AR disagree on the extraction layer")
    if av.injection_token_id != ar.injection_token_id:
        problems.append("AV and AR disagree on the injection token")
    if av.injection_scale is None:
        problems.append("AV sidecar has no injection_scale")
    if ar.mse_scale is None:
        problems.append("AR sidecar has no mse_scale (raw-MSE mode is unsupported here)")

    expected_mse = math.sqrt(av.d_model)
    note = None
    if ar.mse_scale is not None and abs(ar.mse_scale - expected_mse) > 1e-3:
        note = (
            f"mse_scale {ar.mse_scale} is not sqrt(d_model)={expected_mse:.4f}; "
            f"this is permitted but unusual, reconstruction MSE is still 2(1-cos)"
        )
    return {
        "ok": not problems,
        "problems": problems,
        "note": note,
        "av": av.to_dict(),
        "ar": ar.to_dict(),
    }


def verify_tokenizer(tokenizer, meta: NLAMeta) -> dict:
    """Confirm the live tokeniser reproduces the pinned injection token ids.

    Upstream's ``load_nla_config`` does this against the real prompt; this is
    the same check, kept here so the NF4 copy of the checkpoint (whose tokenizer
    files were re-saved) is validated before any inference runs.
    """
    content = meta.av_template.format(injection_char=meta.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )
    positions = [i for i, t in enumerate(ids) if t == meta.injection_token_id]
    report: dict[str, Any] = {
        "n_injection_sites": len(positions),
        "expected_token_id": meta.injection_token_id,
        "prompt_tokens": len(ids),
    }
    if len(positions) != 1:
        report["ok"] = False
        report["error"] = (
            f"expected exactly one injection site, found {len(positions)} - "
            f"tokenizer drift against the sidecar"
        )
        return report
    p = positions[0]
    left_ok = ids[p - 1] == meta.injection_left_neighbor_id
    right_ok = ids[p + 1] == meta.injection_right_neighbor_id
    report.update(
        position=p,
        left=ids[p - 1],
        right=ids[p + 1],
        left_ok=left_ok,
        right_ok=right_ok,
        ok=left_ok and right_ok,
    )
    if not report["ok"]:
        report["error"] = (
            f"neighbour drift: got ({ids[p-1]}, {ids[p+1]}), sidecar says "
            f"({meta.injection_left_neighbor_id}, {meta.injection_right_neighbor_id})"
        )
    return report
