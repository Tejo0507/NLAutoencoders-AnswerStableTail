"""Hidden-state correctness probe (Zhang et al.).

A linear classifier predicting whether the intermediate answer at a chunk
boundary is correct, from the layer-K residual stream at that boundary. Used
here in two roles:

* as a **stopping baseline** for O3 - a scalar confidence per boundary;
* as the **independent-probe control** for the faithfulness audit (O4), where
  it is trained only on activations and never sees verbalised text, which is
  what makes it independent.

Two things are enforced rather than hoped for. Splits are grouped by problem, so
no boundary from a problem in training appears in evaluation - boundaries within
one trace are strongly correlated and an ungrouped split would report an
inflated accuracy. And a label-shuffle control is available (F6), because the
only convincing evidence that a probe is not leaking is that it collapses to
chance when the labels are destroyed.

Decodability is not causal use. Nothing here licenses a mechanistic claim; that
is the causal stage's job (Project_Review_II.md section 2.1.5).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

import numpy as np

from ..config import ProbeConfig
from ..logging_utils import get

log = get(__name__)


@dataclass
class ProbeReport:
    n_train: int
    n_eval: int
    n_features: int
    accuracy: float
    auc: float
    brier: float
    positive_rate: float
    #: Calibration: mean predicted probability vs observed rate, per decile.
    calibration: list[dict[str, float]] = field(default_factory=list)
    folds: list[dict[str, float]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

    out = {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(np.mean(y)),
    }
    # AUC is undefined on a single-class split, which happens on small pilots.
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if sel.sum() == 0:
            continue
        out.append(
            {
                "bin_low": float(edges[i]),
                "bin_high": float(edges[i + 1]),
                "n": int(sel.sum()),
                "mean_predicted": float(p[sel].mean()),
                "observed_rate": float(y[sel].mean()),
            }
        )
    return out


class CorrectnessProbe:
    """Logistic probe with standardisation, fitted on grouped splits."""

    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.model = None
        self.scaler = None

    def _new_estimator(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            solver="lbfgs",
            random_state=self.cfg.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrectnessProbe":
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if self.cfg.standardise:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.model = self._new_estimator().fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("probe is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    @property
    def direction(self) -> np.ndarray:
        """Unit weight vector in activation space.

        Standardisation is undone so the direction lives in the same space as
        the raw activations, which is what the causal stage needs in order to
        ablate it.
        """
        if self.model is None:
            raise RuntimeError("probe is not fitted")
        w = self.model.coef_.ravel().astype(np.float64)
        if self.scaler is not None:
            w = w / np.maximum(self.scaler.scale_, 1e-12)
        n = np.linalg.norm(w)
        return w / n if n > 0 else w


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int) -> list[np.ndarray]:
    """Assign each row to a fold by hashing its group.

    Hashing rather than shuffling means a problem lands in the same fold no
    matter how many other problems were processed, so a probe fitted during a
    pilot and one fitted during the full run are comparable.
    """
    import hashlib

    fold = np.empty(len(groups), dtype=int)
    for i, g in enumerate(groups):
        h = hashlib.sha256(f"{seed}:{g}".encode("utf-8")).digest()
        fold[i] = int.from_bytes(h[:4], "big") % n_folds
    return [np.where(fold == k)[0] for k in range(n_folds)]


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    cfg: ProbeConfig,
    shuffle_labels: bool = False,
) -> ProbeReport:
    """Grouped cross-validation. ``shuffle_labels`` runs the F6 leakage check.

    Labels are permuted *within* group so the class balance per problem is
    preserved and only the activation-label link is destroyed. A probe that
    still scores above chance here is reading something it should not.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = list(groups)

    if shuffle_labels:
        rng = np.random.default_rng(cfg.seed + 1)
        y = y.copy()
        for g in set(groups):
            idx = np.array([i for i, gg in enumerate(groups) if gg == g])
            y[idx] = rng.permutation(y[idx])

    folds = grouped_folds(groups, cfg.n_folds, cfg.seed)
    oof = np.full(len(y), np.nan)
    fold_metrics: list[dict[str, float]] = []

    for k, test_idx in enumerate(folds):
        if len(test_idx) == 0:
            continue
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) < 10 or len(np.unique(y[train_idx])) < 2:
            log.warning("fold %d unusable (n_train=%d, classes=%d)",
                        k, len(train_idx), len(np.unique(y[train_idx])))
            continue
        probe = CorrectnessProbe(cfg).fit(X[train_idx], y[train_idx])
        p = probe.predict_proba(X[test_idx])
        oof[test_idx] = p
        m = _metrics(y[test_idx], p)
        m["fold"] = k
        m["n"] = int(len(test_idx))
        fold_metrics.append(m)

    valid = ~np.isnan(oof)
    if valid.sum() == 0:
        return ProbeReport(
            n_train=0, n_eval=0, n_features=X.shape[1] if X.ndim == 2 else 0,
            accuracy=float("nan"), auc=float("nan"), brier=float("nan"),
            positive_rate=float(np.mean(y)) if len(y) else float("nan"),
            notes={"error": "no usable folds - too few samples or one class only"},
        )
    overall = _metrics(y[valid], oof[valid])
    return ProbeReport(
        n_train=int(valid.sum()),
        n_eval=int(valid.sum()),
        n_features=int(X.shape[1]),
        accuracy=overall["accuracy"],
        auc=overall["auc"],
        brier=overall["brier"],
        positive_rate=overall["positive_rate"],
        calibration=_calibration(y[valid], oof[valid]),
        folds=fold_metrics,
        notes={"shuffled_labels": shuffle_labels, "n_groups": len(set(groups))},
    )


def difference_in_means_direction(
    pos: np.ndarray, neg: np.ndarray
) -> np.ndarray:
    """Unit vector from the ``neg`` mean to the ``pos`` mean.

    Used by the causal stage as the candidate "tail direction". Difference in
    means rather than a probe weight because it is the less model-dependent
    estimator of the two, and the causal claim should not rest on a classifier's
    regularisation path. Must be fitted on the *training* split only.
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    neg = np.atleast_2d(np.asarray(neg, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both groups must be non-empty")
    d = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d
