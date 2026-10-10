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

import contextlib
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


@contextlib.contextmanager
def _load_straight_to(up, device: str, log):
    """Make the vendored loader place the backbone directly on ``device``.

    ``NLACritic`` loads its backbone with no ``device_map`` and then calls
    ``.to(device)``. That materialises the whole checkpoint in host memory
    first - for the released AR, 21 quantised layers plus a bf16 embedding
    table and an ``lm_head`` it immediately discards, about 4.9 GB - and this
    machine routinely has 2-3 GB free. The load fails before any
    reconstruction happens.

    So ``from_pretrained`` is given ``device_map`` for the duration of the
    construction only. This changes *where the weights are put*, not what is
    computed: the backbone, the final-LayerNorm removal, the trained value
    head and the MSE convention are all still upstream's, and the subsequent
    ``.to(device)`` becomes a no-op. On CPU, or if the patch cannot be
    applied, the original behaviour is used unchanged.

    Recorded as a deviation in docs/DECISIONS.md (D23) because it is a
    modification of upstream's behaviour, narrow as it is.
    """
    import torch

    original = getattr(up, "AutoModelForCausalLM", None)
    if original is None or device == "cpu" or not torch.cuda.is_available():
        yield
        return

    class _Placed:
        """Stands in for the vendored module's ``AutoModelForCausalLM``."""

        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("device_map", {"": 0})
            kwargs.setdefault("low_cpu_mem_usage", True)
            return original.from_pretrained(*args, **kwargs)

    up.AutoModelForCausalLM = _Placed
    log.info("AR backbone will be placed directly on %s rather than staged in "
             "host memory first", device)
    try:
        yield
    finally:
        up.AutoModelForCausalLM = original


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
        with _load_straight_to(self._up, self.device, log):
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


def empirical_baseline_mse(
    vectors: np.ndarray, n_pairs: int = 200_000, seed: int = 0
) -> dict[str, float]:
    """What an *uninformative* prediction scores on this activation sample.

    PROJECT_PLAN.md §5 specifies FVE "against the empirical variance baseline
    of our own activation sample", and that is not 2.0. A baseline of 2.0
    assumes an uninformative prediction is orthogonal to the target, which
    holds for a random direction in high dimensions - but not for a *plausible*
    prediction, and residual-stream activations at one layer are strongly
    anisotropic. On the pilot, two layer-20 vectors drawn at random from
    different problems have a mean cosine of 0.58, so simply guessing a typical
    layer-20 activation already achieves MSE 0.83.

    Measured against 2.0 the reconstruction looks far better than it is: the
    same mean MSE gives FVE 0.87 against 2.0 and 0.68 against the empirical
    baseline. Both are reported, because the number means nothing without the
    baseline it used, and the comparison with the checkpoint card's 0.752 is
    only meaningful against whatever baseline *that* used.

    Direction is all that matters here - both vectors are normalised before
    the MSE - so the vectors are unit-normalised and random distinct pairs are
    sampled.
    """
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    n = v.shape[0]
    if n < 2:
        return {"available": False, "n_vectors": int(n)}
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(norms, 1e-12)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    cos = np.einsum("ij,ij->i", u[i[keep]], u[j[keep]])
    return {
        "available": True,
        "n_vectors": int(n),
        "n_pairs": int(keep.sum()),
        "mean_pairwise_cosine": float(cos.mean()),
        "median_pairwise_cosine": float(np.median(cos)),
        "baseline_mse": float(np.mean(2.0 * (1.0 - cos))),
        "note": ("MSE a prediction drawn from this same activation "
                 "distribution achieves; 2.0 would assume an orthogonal "
                 "prediction, which these activations are not"),
    }


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
