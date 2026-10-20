"""Control directions and outcome reading for the RQ3 follow-up pilot (D74).

The registered ``random_dir`` control is a random *unit* direction at the
candidate's coefficient. Projecting it out removes ``|h · u|``, about 1.1% of
a layer-20 activation's norm against 21.8% for the candidate (D68). Raising
the coefficient to compensate is no remedy: at about 19 it no longer removes
a component; it reflects and amplifies one, which is a different edit.

Two controls here are the **same edit** as the candidate, a projection-out at
the same coefficient, and are matched on what the edit removes:

* :func:`variance_matched_directions` draws random directions from the
  activations' own second-moment geometry, accepting a draw only if the mean
  squared projection on calibration activations is within ``tol`` of the
  candidate's. This rules out "any edit that removes this much".
* :func:`permuted_label_directions` reruns the candidate's estimator, a
  difference in means, after permuting the tail and pre-stabilisation labels
  within each problem. This rules out "any direction this estimator finds in
  these windows".

The outcome reading (:func:`read_outcome`) makes truncation visible. In the
registered run, all 13 answer changes at dose 2.0 were continuations cut off
by a 200-token cap.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np


def second_moment(X: np.ndarray, u: np.ndarray) -> float:
    """Mean squared projection of the rows of ``X`` on unit ``u``."""
    u = u / np.linalg.norm(u)
    return float(np.mean((X @ u) ** 2))


def variance_matched_directions(
    X: np.ndarray,
    candidate: np.ndarray,
    n: int,
    rng: np.random.Generator,
    tol: float = 0.10,
    max_draws: int = 20000,
) -> tuple[list[np.ndarray], dict]:
    """Random unit directions whose projection second moment matches the candidate's.

    Draws ``u ∝ V diag(s^(2a)) Vᵀ z`` from the SVD of the calibration matrix
    ``X``, with ``z`` isotropic Gaussian. ``a = 0`` gives directions spread
    over the activations' row space; larger ``a`` concentrates them on
    high-variance directions. ``a`` is set by bisection so the *median* second
    moment equals the candidate's, then draws are accepted only within
    ``tol`` after being made orthogonal to the candidate. Raises
    ``ValueError`` if the candidate's second moment cannot be reached, which
    happens when it exceeds what the top singular direction gives.
    """
    c = candidate / np.linalg.norm(candidate)
    target = second_moment(X, c)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    top = float(s[0] ** 2 / X.shape[0])
    if target > top * (1 + tol):
        raise ValueError(f"candidate second moment {target:.3g} exceeds the top "
                         f"direction's {top:.3g}; no random direction can match it")

    def draw(a: float, k: int) -> np.ndarray:
        z = rng.standard_normal((k, Vt.shape[0]))
        U = (z * s ** (2 * a)) @ Vt
        U -= np.outer(U @ c, c)                       # orthogonal to the candidate
        return U / np.linalg.norm(U, axis=1, keepdims=True)

    def median_m(a: float) -> float:
        U = draw(a, 64)
        return float(np.median(np.mean((X @ U.T) ** 2, axis=0)))

    lo, hi = 0.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if median_m(mid) < target:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    out, tried = [], 0
    while len(out) < n and tried < max_draws:
        U = draw(a, 64)
        tried += 64
        for u in U:
            if abs(second_moment(X, u) / target - 1) <= tol:
                out.append(u)
                if len(out) == n:
                    break
    if len(out) < n:
        raise ValueError(f"only {len(out)} of {n} directions within ±{tol:.0%} "
                         f"after {tried} draws")
    return out, {"target_second_moment": target, "exponent": a, "draws": tried,
                 "accepted_ratios": [second_moment(X, u) / target for u in out]}


def permuted_label_directions(
    tail_by_problem: Mapping[str, np.ndarray],
    pre_by_problem: Mapping[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    X: np.ndarray | None = None,
    candidate: np.ndarray | None = None,
    tol: float = 0.10,
    max_draws: int = 2000,
) -> tuple[list[np.ndarray], dict]:
    """Difference-in-means directions after permuting labels within each problem.

    Every problem keeps its own vectors and its own count of tail and pre
    labels. Only which vector carries which label is shuffled, so the
    estimator, the windows and the problems are those of the candidate.

    Given ``X`` and ``candidate``, each draw is made orthogonal to the
    candidate and kept only if its second moment on ``X`` is within ``tol``
    of the candidate's. On ``main`` the raw permuted directions ranged from
    0.33 to 1.20 times the candidate's size, with cosines up to 0.62 with it
    (D74). A control that partly *is* the candidate, or is a third of its
    size, does not control for it.
    """
    pids = sorted(set(tail_by_problem) & set(pre_by_problem))
    c = None if candidate is None else candidate / np.linalg.norm(candidate)
    target = None if X is None or c is None else second_moment(X, c)
    out, raw_cos, tried = [], [], 0
    while len(out) < n and tried < max_draws:
        tried += 1
        t_all, p_all = [], []
        for pid in pids:
            T, P = np.asarray(tail_by_problem[pid]), np.asarray(pre_by_problem[pid])
            pool = np.vstack([T, P])
            perm = rng.permutation(len(pool))
            t_all.append(pool[perm[:len(T)]])
            p_all.append(pool[perm[len(T):]])
        d = np.vstack(t_all).mean(0) - np.vstack(p_all).mean(0)
        d = d / np.linalg.norm(d)
        if c is not None:
            raw_cos.append(float(d @ c))
            d = d - (d @ c) * c
            d = d / np.linalg.norm(d)
        if target is not None and abs(second_moment(X, d) / target - 1) > tol:
            continue
        out.append(d)
    if len(out) < n:
        raise ValueError(f"only {len(out)} of {n} permuted directions within ±{tol:.0%} "
                         f"after {tried} permutations")
    return out, {"permutations_tried": tried, "raw_cosines_with_candidate": raw_cos,
                 "accepted_ratios": ([second_moment(X, d) / target for d in out]
                                     if target is not None else None)}


def position_direction(
    vectors_by_problem: Mapping[str, np.ndarray],
    relative_position_by_problem: Mapping[str, np.ndarray],
    split_at: float = 0.5,
) -> np.ndarray:
    """The candidate's estimator with position labels: late minus early.

    Pass only boundaries *before* each tail, so the labels carry position
    and nothing about the tail. A candidate close to this direction is about
    progress through the solution, not about the tail.
    """
    late, early = [], []
    for pid, V in vectors_by_problem.items():
        rel = np.asarray(relative_position_by_problem[pid])
        V = np.asarray(V)
        late.append(V[rel >= split_at])
        early.append(V[rel < split_at])
    d = np.vstack(late).mean(0) - np.vstack(early).mean(0)
    return d / np.linalg.norm(d)


_REPEAT = re.compile(r"(\b\w+(?:\W+\w+){7}\b)(?:.*?\1){2}", re.S)


def degenerate(text: str, max_non_ascii: float = 0.3) -> bool:
    """A continuation that repeats an 8-word run three times, or is mostly non-ASCII."""
    if not text:
        return False
    if sum(ord(ch) > 127 for ch in text) / len(text) > max_non_ascii:
        return True
    return bool(_REPEAT.search(text))


def read_outcome(finished: bool, answer_from_text: str | None,
                 forced_answer: str | None) -> tuple[str | None, str]:
    """The answer an intervened continuation ends on, and how it was obtained.

    A continuation that finished is read from its own text. One that hit the
    cap is not read as if it had finished. The model is made to commit, by the
    registered forcing call, and that answer is used and labelled ``forced``.
    """
    if finished:
        return answer_from_text, "finished"
    return forced_answer, "forced_after_cap"


def mean_removed_norm(removed_norm_sum: float, n_positions: int) -> float | None:
    return removed_norm_sum / n_positions if n_positions else None


def within_tolerance(control: Sequence[float | None], candidate: float | None,
                     tol: float = 0.10) -> list[bool]:
    """Per control replicate: is its realised removed norm within ``tol`` of the candidate's?"""
    if not candidate:
        return [False] * len(control)
    return [c is not None and abs(c / candidate - 1) <= tol for c in control]
