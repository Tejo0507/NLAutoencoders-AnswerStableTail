"""Activation Reconstructor - text back to vector.

Thin wrapper over the vendored ``NLACritic``, which is used essentially as
shipped: 21-layer truncated Qwen2 backbone, final LayerNorm and ``lm_head``
replaced with identities, a ``Linear(3584, 3584)`` value head read at the last
token of ``"Summary of the following text: <text>{explanation}</text> <summary>"``.

The one thing this file adds is batching and the fidelity bookkeeping the study
needs. The metric convention is upstream's and is not reinterpreted: both
vectors are L2-normalised to ``mse_scale = sqrt(d_model)``, so

    MSE = 2 (1 - cos),  range [0, 4],  orthogonal = 2.

**This is reconstruction fidelity, not faithfulness.** The review's third
distinction (Project_Review_II.md section 2.1.10) is that a description can
preserve most of an activation's variance while individual claims inside it are
unsupported. Nothing in this module speaks to claim-level faithfulness; that is
``faithfulness/audit.py``, and the two are reported in separate tables.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..logging_utils import get
from ..models import loading
from .meta import NLAMeta, load_meta, upstream

log = get(__name__)


@dataclass
class Reconstruction:
    mse: float
    cosine: float
    pred_norm: float
    gold_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivationReconstructor:
    def __init__(self, cfg: Config, checkpoint_dir: Path | str):
        import torch

        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.meta: NLAMeta = load_meta(self.checkpoint_dir)
        if self.meta.role not in ("ar", "critic"):
            raise ValueError(
                f"{checkpoint_dir} has role={self.meta.role!r}; point this at the "
                f"AR (reconstructor) checkpoint, not the AV"
            )
        self._up = upstream()
        self.device = cfg.nla.device if torch.cuda.is_available() else "cpu"
        log.info("loading AR from %s", self.checkpoint_dir)
        self.critic = self._up.NLACritic(
            self.checkpoint_dir, device=self.device, dtype=torch.bfloat16
        )
        self.mse_scale = float(self.critic.mse_scale)
        self.d_model = self.critic.backbone.config.hidden_size
        log.info("AR ready: mse_scale=%.4f d_model=%d | %s",
                 self.mse_scale, self.d_model, loading.cuda_memory())

    def reconstruct(self, explanation: str) -> np.ndarray:
        return self.critic.reconstruct(explanation).numpy()

    def score(self, explanation: str, activation: np.ndarray) -> Reconstruction:
        pred = self.reconstruct(explanation)
        return self.compare(pred, activation)

    def compare(self, pred: np.ndarray, gold: np.ndarray) -> Reconstruction:
        pred = np.asarray(pred, dtype=np.float64).ravel()
        gold = np.asarray(gold, dtype=np.float64).ravel()
        pn = np.linalg.norm(pred)
        gn = np.linalg.norm(gold)
        p = pred / max(pn, 1e-12) * self.mse_scale
        g = gold / max(gn, 1e-12) * self.mse_scale
        mse = float(np.mean((p - g) ** 2))
        cos = float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-12))
        return Reconstruction(mse=mse, cosine=cos, pred_norm=float(pn), gold_norm=float(gn))

    def close(self) -> None:
        try:
            del self.critic
        except AttributeError:
            pass
        gc.collect()
        loading.free_cuda()


def fraction_variance_explained(
    mses: Sequence[float], baseline_mse: float | None = None
) -> dict[str, float]:
    """FVE for a batch of reconstructions.

    Upstream's training-time ``fve_nrm`` is ``1 - mean(MSE) / baseline``, where
    the baseline is the direction-MSE of an uninformative prediction. Under the
    normalise-to-``sqrt(d)`` convention a random direction in high dimensions is
    almost surely orthogonal to the target, giving MSE = 2, so 2.0 is the
    default baseline. It is kept explicit rather than folded into a constant
    because the number only means something alongside the baseline it used.
    """
    arr = np.asarray([m for m in mses if np.isfinite(m)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean_mse": float("nan"), "fve": float("nan"),
                "baseline_mse": baseline_mse or 2.0}
    base = 2.0 if baseline_mse is None else float(baseline_mse)
    mean_mse = float(arr.mean())
    return {
        "n": int(arr.size),
        "mean_mse": mean_mse,
        "median_mse": float(np.median(arr)),
        "mean_cosine": float(1.0 - mean_mse / 2.0),
        "fve": float(1.0 - mean_mse / base),
        "baseline_mse": base,
    }
