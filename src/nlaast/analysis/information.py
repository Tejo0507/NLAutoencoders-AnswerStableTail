"""What the readout knows, beside the stopping metric (D81). Pure; no I/O.

Implements the estimation in ``docs/SUPPLEMENTARY_INFORMATION_PLAN.md``:

* out-of-fold predictions from a problem-grouped, outcome-stratified K-fold;
* one classifier everywhere: a standardised logistic regression (C = 1, L2),
  with each text block passed through its own TF-IDF and stacked beside the
  numeric block, every transform fitted inside the training fold;
* a problem-cluster bootstrap of the out-of-fold predictions for any
  difference in AUC or log loss;
* the number matching behind Q3.

**Why the bootstrap weights are exact.** Resampling problems with replacement
and recomputing AUC on the resampled units is the same as weighting each unit
by how many times its problem was drawn. ``weighted_auc`` computes that
weighted Mann-Whitney statistic directly, ties counted half, so a replicate
costs one sort instead of materialising the resample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- metrics


def weighted_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mann-Whitney AUC of score ``s`` for binary ``y`` with unit weights ``w``.

    Ties between a positive and a negative count one half. ``nan`` when either
    class has zero weight.
    """
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    wp, wn = w[y].sum(), w[~y].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    # group equal scores
    uniq, start = np.unique(s_sorted, return_index=True)
    neg_w = np.where(~y_sorted, w_sorted, 0.0)
    pos_w = np.where(y_sorted, w_sorted, 0.0)
    neg_tie = np.add.reduceat(neg_w, start)
    pos_tie = np.add.reduceat(pos_w, start)
    neg_below = np.concatenate([[0.0], np.cumsum(neg_tie)[:-1]])
    num = float(np.sum(pos_tie * (neg_below + 0.5 * neg_tie)))
    return num / (wp * wn)


def weighted_log_loss(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None,
                      eps: float = 1e-6) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(w * ll) / np.sum(w))


def within_problem_concordance(y: Sequence[bool], s: Sequence[float],
                               groups: Sequence[str]) -> float:
    """Share of within-problem (positive, negative) pairs the score orders
    correctly, ties one half, pooled over problems. ``nan`` if no problem
    has both classes."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    g = np.asarray(groups)
    good = total = 0.0
    for key in np.unique(g):
        m = g == key
        pos, neg = s[m & y], s[m & ~y]
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        good += float((d > 0).sum() + 0.5 * (d == 0).sum())
        total += d.size
    return good / total if total else float("nan")


# ---------------------------------------------------------------- models


@dataclass
class Design:
    """The inputs of one model: a numeric block and zero or more text blocks."""

    numeric: np.ndarray                 # (n, d), d may be 0
    texts: tuple[Sequence[str], ...] = ()


def fit_predict(design: Design, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                C: float = 1.0, min_df: int = 3,
                extra: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Fit on ``train`` and return ``P(y = 1)`` on ``test``; every transform is
    fitted on ``train`` only. ``extra`` adds fold-specific numeric columns
    ``(train rows, test rows)``, such as a stacked probe score."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    blocks_tr, blocks_te = [], []
    num_tr, num_te = design.numeric[train], design.numeric[test]
    if extra is not None:
        num_tr = np.c_[num_tr, extra[0]]
        num_te = np.c_[num_te, extra[1]]
    if num_tr.shape[1]:
        sc = StandardScaler().fit(num_tr)
        blocks_tr.append(sparse.csr_matrix(sc.transform(num_tr)))
        blocks_te.append(sparse.csr_matrix(sc.transform(num_te)))
    for block in design.texts:
        docs = np.asarray(block, dtype=object)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                              lowercase=True, token_pattern=r"(?u)\b\w+\b")
        try:
            blocks_tr.append(vec.fit_transform(docs[train]))
        except ValueError:          # no term reaches min_df in this fold
            continue
        blocks_te.append(vec.transform(docs[test]))
    ytr = y[train]
    if not blocks_tr or len(np.unique(ytr)) < 2:
        return np.full(len(test), float(ytr.mean()) if len(ytr) else 0.5)
    Xtr, Xte = sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()
    clf = LogisticRegression(C=C, max_iter=5000, solver="liblinear")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def grouped_folds(y: np.ndarray, groups: Sequence[str], n_splits: int, seed: int
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Problem-grouped folds, stratified on ``y``, deterministic in ``seed``."""
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, np.asarray(groups))]


def out_of_fold(design: Design, y: np.ndarray, folds,
                extras: Sequence[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    """Out-of-fold ``P(y = 1)``; ``extras[k]`` are fold ``k``'s stacked columns."""
    oof = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(folds):
        oof[te] = fit_predict(design, y, tr, te, extra=None if extras is None else extras[k])
    return oof


def _dense_logit(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def nested_probe(X: np.ndarray, y: np.ndarray, groups: Sequence[str], folds,
                 Cs: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                 inner_splits: int = 5, seed: int = 0
                 ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float]]:
    """A dense linear probe as a stacked feature, leak-free (plan deviation 1).

    For each outer fold: choose C by inner grouped CV on the training rows
    (mean log loss), then return the logit of
    * inner out-of-fold scores at that C for the training rows, and
    * the scores of a probe fitted on all training rows for the test rows.
    Nothing from the test fold is seen. Returns ``(extras, chosen Cs)``.
    """
    groups = np.asarray(groups)
    extras, chosen = [], []
    for tr, te in folds:
        inner = grouped_folds(y[tr], groups[tr], inner_splits, seed)
        best, best_ll, best_oof = None, np.inf, None
        for C in Cs:
            oof = np.full(len(tr), np.nan)
            for itr, ite in inner:
                m = _dense_logit(C).fit(X[tr][itr], y[tr][itr])
                oof[ite] = m.predict_proba(X[tr][ite])[:, 1]
            ll = weighted_log_loss(y[tr], oof)
            if ll < best_ll:
                best, best_ll, best_oof = C, ll, oof
        m = _dense_logit(best).fit(X[tr], y[tr])
        p_te = m.predict_proba(X[te])[:, 1]
        clip = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        extras.append((clip(best_oof)[:, None], clip(p_te)[:, None]))
        chosen.append(float(best))
    return extras, chosen


# ---------------------------------------------------------------- bootstrap


def cluster_weights(groups: Sequence[str], reps: int, rng: np.random.Generator
                    ) -> np.ndarray:
    """``(reps, n)`` unit weights: how often each unit's problem was drawn."""
    g = np.asarray(groups)
    keys, inv = np.unique(g, return_inverse=True)
    counts = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=reps)
    return counts[:, inv].astype(float)


def problem_draws(problem_ids: Sequence[str], reps: int, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``(keys, counts)``: one bootstrap of problems, shared by every analysis
    so that all contrasts are paired. ``counts`` is ``(reps, len(keys))``."""
    keys = np.unique(np.asarray(problem_ids))
    counts = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=reps)
    return keys, counts.astype(float)


def weights_for(groups: Sequence[str], keys: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Unit weights for any subset of units, from the shared problem draws."""
    idx = np.searchsorted(keys, np.asarray(groups))
    return counts[:, idx]


def bootstrap_p(rep: np.ndarray, floor: float) -> float:
    r = rep[np.isfinite(rep)]
    if not len(r):
        return float("nan")
    return float(max(floor, min(1.0, 2 * min(np.mean(r <= 0), np.mean(r >= 0)))))


def compare(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, W: np.ndarray) -> dict:
    """AUC(a) − AUC(b) and log loss(b) − log loss(a) (both positive when ``a``
    is better), with cluster-bootstrap CIs over the weight matrix ``W``."""
    y = np.asarray(y, bool)
    point_auc = weighted_auc(y, pa) - weighted_auc(y, pb)
    point_ll = weighted_log_loss(y, pb) - weighted_log_loss(y, pa)
    ra = np.array([weighted_auc(y, pa, w) - weighted_auc(y, pb, w) for w in W])
    rl = np.array([weighted_log_loss(y, pb, w) - weighted_log_loss(y, pa, w) for w in W])
    fl = 1 / len(W)
    ok = ra[np.isfinite(ra)]
    return {
        "auc_a": weighted_auc(y, pa), "auc_b": weighted_auc(y, pb),
        "delta_auc": point_auc,
        "ci95": [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))] if len(ok) else None,
        "p_bootstrap": bootstrap_p(ra, fl),
        "delta_log_loss": point_ll,
        "log_loss_ci95": [float(np.quantile(rl, 0.025)), float(np.quantile(rl, 0.975))],
    }


def mean_difference(values: np.ndarray, W: np.ndarray) -> dict:
    """Mean of per-unit ``values`` with a cluster-bootstrap CI and p (vs 0)."""
    v = np.asarray(values, float)
    rep = (W @ v) / W.sum(1)
    return {"mean": float(v.mean()),
            "ci95": [float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))],
            "p_bootstrap": bootstrap_p(rep, 1 / len(W)), "n_units": int(len(v))}


def reading(ci: Sequence[float] | None, reject: bool | None, delta: float,
            margin: float) -> str:
    """Plan §5's language."""
    if ci is None or not np.isfinite(delta):
        return "not evaluable"
    if reject and delta > 0:
        return "increment detected"
    if ci[0] > -margin and ci[1] < margin:
        return "no practically relevant increment"
    return "unresolved"


# ---------------------------------------------------------------- Q3 numbers

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalise_number(tok: str) -> str | None:
    """``"1,200"`` → ``"1200"``, ``"3.50"`` → ``"3.5"``, ``"4.0"`` → ``"4"``."""
    t = tok.replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return None
    neg = t.startswith("-")
    t = t.lstrip("-")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    t = t.lstrip("0") or "0"
    if t.startswith("."):
        t = "0" + t
    return ("-" + t) if neg and t != "0" else t


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUM.finditer(text or ""):
        n = normalise_number(m.group(0).rstrip(","))
        if n is not None:
            out.add(n)
    return out


def plain_number(answer) -> str | None:
    """The final answer as a normalised plain number, or ``None`` if it is not
    one (fractions, expressions, text)."""
    if answer is None:
        return None
    t = str(answer).strip().replace("$", "").replace(",", "")
    t = re.sub(r"\\[,!; ]", "", t)
    return normalise_number(t) if re.fullmatch(r"-?\d+(?:\.\d+)?", t) else None


@dataclass
class MentionUnit:
    problem_id: str
    dataset: str
    answer: str                 # this problem's normalised answer
    context_numbers: set[str]   # numbers in the question and the prefix
    explanation_numbers: set[str]


def mention_contrast(units: Sequence[MentionUnit],
                     answers: Mapping[str, tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit ``own`` and ``other`` (plan Q3).

    ``answers`` maps every problem id to ``(dataset, normalised answer)``.
    ``other`` averages, over every other problem of the same dataset whose
    answer is absent from the unit's context, whether the unit's explanation
    contains that answer.
    """
    own = np.zeros(len(units))
    other = np.zeros(len(units))
    by_ds: dict[str, list[tuple[str, str]]] = {}
    for pid, (ds, a) in answers.items():
        by_ds.setdefault(ds, []).append((pid, a))
    for k, u in enumerate(units):
        own[k] = float(u.answer in u.explanation_numbers)
        pool = [a for pid, a in by_ds.get(u.dataset, [])
                if pid != u.problem_id and a not in u.context_numbers]
        other[k] = float(np.mean([a in u.explanation_numbers for a in pool])) if pool else np.nan
    return own, other


# ---------------------------------------------------------------- Q5


def icc1(values: Sequence[float], groups: Sequence[str]) -> float:
    """One-way random-effects ICC(1) of ``values`` grouped by ``groups``."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    keys = np.unique(g)
    k = len(keys)
    n = len(v)
    if k < 2 or n <= k:
        return float("nan")
    grand = v.mean()
    sizes = np.array([np.sum(g == key) for key in keys], float)
    means = np.array([v[g == key].mean() for key in keys])
    msb = np.sum(sizes * (means - grand) ** 2) / (k - 1)
    msw = sum(np.sum((v[g == key] - m) ** 2) for key, m in zip(keys, means)) / (n - k)
    n0 = (n - np.sum(sizes ** 2) / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def oof_for(designs: Mapping[str, Design], y: np.ndarray, folds,
            progress: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    out = {}
    for name, d in designs.items():
        if progress:
            progress(name)
        out[name] = out_of_fold(d, y, folds)
    return out
