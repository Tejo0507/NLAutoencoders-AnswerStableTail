"""The stopping-policy pilot's statistics. Pure numpy/scipy; no I/O.

Implements the analysis of `docs/STOPPING_POLICY_PILOT_PROTOCOL.md`. The unit
is the problem: a policy walks a problem's chunk boundaries in order and
stops at the first whose score reaches the threshold, so each problem
contributes one decision (stop at one boundary, or never stop). Boundaries
are nested in problems and never counted as independent units.

Safety at a boundary uses the registered definition under the 256-token
forced answer: the forced answer's correctness equals the final answer's.
How a forced answer still cut off at 256 tokens is labelled is a fixed rule
(`TRUNCATION_RULES`); the protocol's primary rule counts it as unsafe.

Threshold selection follows Selection with Guaranteed Risk (SGR; Geifman and
El-Yaniv, 2017): a binary search over a fixed grid of thresholds, with a
one-sided Clopper-Pearson upper bound on selective risk computed at
confidence 1 - delta / ceil(log2 m) at each of its ceil(log2 m) steps. By the
union bound, every bound the search computes holds simultaneously with
probability at least 1 - delta; the output is the most permissive threshold
whose bound is at most the calibration target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

#: How a 256-token forced answer that did not finish is labelled.
#: "cut_off_unsafe" (primary): unsafe, whatever its partial answer parses to;
#: "as_parsed": its parsed partial answer is used as stored (O256);
#: "as_no_answer": it counts as no answer (O256-u).
TRUNCATION_RULES = ("cut_off_unsafe", "as_parsed", "as_no_answer")


def is_safe(forced_correct: bool | None, finished: bool, final_correct: bool, rule: str) -> bool:
    """Safety of a stop: the forced answer's correctness equals the final answer's.

    `forced_correct` is None when no answer could be parsed; no answer counts as
    an incorrect answer, as in Stage 1's labels.
    """
    if rule not in TRUNCATION_RULES:
        raise ValueError(f"unknown truncation rule {rule!r}")
    if not finished:
        if rule == "cut_off_unsafe":
            return False
        if rule == "as_no_answer":
            forced_correct = None
    return bool(forced_correct) == bool(final_correct)


@dataclass
class Problem:
    """One problem's boundaries in chunk order."""

    problem_id: str
    scores: np.ndarray      # policy score per boundary (higher: more ready to stop)
    safe: np.ndarray        # bool per boundary
    saving: np.ndarray      # net saving per boundary, in [0, 1]
    finished: np.ndarray    # bool per boundary: the forced answer finished


def net_saving(prefix_tokens: int, forced_tokens: int, total_tokens: int,
               signal_tokens: int = 0) -> float:
    """max(0, 1 - (prefix + forced + signal) / total), with exact generated counts."""
    return max(0.0, 1.0 - (prefix_tokens + forced_tokens + signal_tokens) / total_tokens)


@dataclass
class Walk:
    stopped: np.ndarray     # bool per problem
    safe: np.ndarray        # bool per problem (True when not stopped)
    saving: np.ndarray      # float per problem (0 when not stopped)
    cut_off: np.ndarray     # bool per problem: stopped at an unfinished forced answer


def walk(problems: Sequence[Problem], tau: float) -> Walk:
    stopped, safe, saving, cut = [], [], [], []
    for p in problems:
        hit = np.flatnonzero(p.scores >= tau)
        if hit.size:
            k = int(hit[0])
            stopped.append(True)
            safe.append(bool(p.safe[k]))
            saving.append(float(p.saving[k]))
            cut.append(not bool(p.finished[k]))
        else:
            stopped.append(False)
            safe.append(True)
            saving.append(0.0)
            cut.append(False)
    return Walk(np.array(stopped, bool), np.array(safe, bool), np.array(saving, float),
                np.array(cut, bool))


def cp_upper(x: int, n: int, conf: float) -> float:
    """One-sided Clopper-Pearson upper bound for x events in n trials (1 when n = 0)."""
    from scipy.stats import beta
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(beta.ppf(conf, x + 1, n - x))


def cp_interval(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval."""
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x))
    return lo, hi


def selective_counts(w: Walk) -> tuple[int, int]:
    """(unsafe stops, stops)."""
    return int((~w.safe & w.stopped).sum()), int(w.stopped.sum())


def sgr_threshold(problems: Sequence[Problem], grid: Sequence[float], target: float,
                  delta: float) -> dict:
    """Selection with Guaranteed Risk over a fixed threshold grid.

    `grid` is sorted ascending (permissive to strict). Returns the chosen
    threshold (None if no tested threshold's bound is at most `target`), the
    per-step record, and the confidence used per step.
    """
    g = np.sort(np.asarray(grid, float))
    m = len(g)
    if m == 0:
        raise ValueError("empty threshold grid")
    k = max(1, math.ceil(math.log2(m)))
    conf = 1.0 - delta / k
    lo, hi = 0, m - 1                  # indices: lo most permissive, hi strictest
    best = None
    steps = []
    for _ in range(k):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        w = walk(problems, float(g[mid]))
        x, n = selective_counts(w)
        b = cp_upper(x, n, conf)
        ok = n > 0 and b <= target
        steps.append({"tau": float(g[mid]), "n_stopped": n, "n_unsafe": x, "bound": b,
                      "accepted": ok})
        if ok:
            best = float(g[mid])
            hi = mid - 1               # try more permissive
        else:
            lo = mid + 1               # need stricter
    return {"tau": best, "steps": steps, "k": k, "conf_per_step": conf, "grid_size": m}


def summarise(w: Walk, conf: float = 0.95) -> dict:
    x, n = selective_counts(w)
    out = {
        "n_problems": int(w.stopped.size), "n_stopped": n, "n_unsafe": x,
        "coverage": float(w.stopped.mean()) if w.stopped.size else None,
        "selective_risk": (x / n) if n else None,
        "selective_risk_ci95": list(cp_interval(x, n, 0.95)) if n else None,
        f"selective_risk_upper_{int(conf * 100)}": cp_upper(x, n, conf) if n else None,
        "mean_net_saving": float(w.saving.mean()) if w.saving.size else None,
        "cut_off_share_of_stops": float(w.cut_off[w.stopped].mean()) if n else None,
    }
    return out


def safety_gate(w: Walk, alpha: float, conf: float = 0.95) -> dict:
    """The protocol's safety reading: one-sided upper bound <= alpha, at coverage > 0.

    In a pilot this is reported, not used as a verdict (protocol §1).
    """
    x, n = selective_counts(w)
    if n == 0:
        return {"reading": "never stops (coverage 0): safe but useless", "bound": None}
    b = cp_upper(x, n, conf)
    return {"reading": "bound at or below alpha" if b <= alpha else "bound above alpha",
            "bound": b, "alpha": alpha}


def bootstrap_mean_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> list[float]:
    n = len(values)
    if n == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, n, (reps, n))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_saving(w1: Walk, w0: Walk, reps: int, rng: np.random.Generator) -> dict:
    """P1 minus P0 net saving per problem: mean, SD (sigma_d) and a problem bootstrap CI."""
    d = w1.saving - w0.saving
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if d.size > 1 else None,
            "ci95": bootstrap_mean_ci(d, reps, rng), "n": int(d.size)}


def threshold_grid(train_scores: np.ndarray, n: int) -> np.ndarray:
    """A fixed grid: `n` quantiles of the training scores (frozen before calibration)."""
    return np.unique(np.quantile(np.asarray(train_scores, float), np.linspace(0, 1, n)))


def build_problems(rows: Sequence[Mapping], score: Callable[[Mapping], float], rule: str
                   ) -> list[Problem]:
    """Group boundary rows into problems in chunk order.

    Each row needs: problem_id, chunk, forced_correct (bool or None), finished,
    final_correct, prefix_tokens, forced_tokens, total_tokens.
    """
    by: dict[str, list[Mapping]] = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    out = []
    for pid in sorted(by):
        rs = sorted(by[pid], key=lambda r: r["chunk"])
        out.append(Problem(
            pid,
            np.array([score(r) for r in rs], float),
            np.array([is_safe(r["forced_correct"], r["finished"], r["final_correct"], rule)
                      for r in rs], bool),
            np.array([net_saving(r["prefix_tokens"], r["forced_tokens"], r["total_tokens"])
                      for r in rs], float),
            np.array([bool(r["finished"]) for r in rs], bool)))
    return out
