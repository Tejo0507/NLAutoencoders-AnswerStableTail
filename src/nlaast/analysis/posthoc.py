"""Post-hoc audit quantities (D68): what the registered numbers can and cannot mean.

Every function here is a *reading aid* for a result that already exists. None
of them replaces a registered test, and none is called by the pipeline. They
are kept out of ``stats.py`` so that nothing registered can come to depend on
them by accident.

Four questions motivated them, each raised by the 2026-10-19 audit
(``docs/SCIENTIFIC_AUDIT_2026-10-19.md``):

* **Where does a stopping rule's safe rate come from?** A rule that never
  fires on a problem is counted as safe there (``evaluate_stop``), so at an
  aggressive budget a safe rate can be mostly abstention.
  :func:`decompose_decisions` separates the two.
* **What would a rule with no information score?** :func:`random_first_boundary_null`
  stops at the first boundary on a random subset of problems, sized to reach a
  target mean saving. It is the reference a signal has to beat before its
  safe rate says anything about the signal.
* **How much can any rule save while staying safe?** :func:`oracle_safe_saving`
  is the mean, over problems, of the largest saving at which stopping is safe.
  A matched budget above it forces unsafe stops on every rule by construction.
* **Do the claim- and sample-level tests survive their own clustering?**
  :func:`cluster_bootstrap_mean` and :func:`paired_problem_means` re-express a
  pooled comparison at the problem level, the unit D29 and D41 already use
  for O3 and RQ3.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..baselines.stopping import StopDecision


def decompose_decisions(
    decisions: Sequence[StopDecision],
    tail_start: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Split a safe rate into abstentions and safe stops.

    ``safe_rate = (n_abstained + n_safe_stops) / n``. The two terms mean
    different things: an abstention is safe by definition, whatever the
    signal knows, while a safe stop is the signal being right about a problem
    it acted on. Comparing two rules' safe rates at a matched mean saving
    therefore compares a mixture of the two unless the mixture is reported.

    ``tail_start``, if given, adds where the stops fell relative to the
    Answer-Stable Tail (negative = before the tail began).
    """
    n = len(decisions)
    fired = [d for d in decisions if d.boundary is not None]
    n_abst = n - len(fired)
    n_safe_stops = sum(1 for d in fired if d.safe)
    out: dict[str, Any] = {
        "n": n,
        "n_fired": len(fired),
        "n_abstained": n_abst,
        "n_safe": n_abst + n_safe_stops,
        "n_safe_stops": n_safe_stops,
        "safe_rate": (n_abst + n_safe_stops) / n if n else float("nan"),
        "fire_rate": len(fired) / n if n else float("nan"),
        "share_of_safe_that_is_abstention": (
            n_abst / (n_abst + n_safe_stops) if (n_abst + n_safe_stops) else float("nan")),
        "safe_given_fired": n_safe_stops / len(fired) if fired else float("nan"),
        "mean_tokens_saved": (float(np.mean([d.tokens_saved_fraction for d in decisions]))
                              if n else float("nan")),
        "share_fired_at_first_boundary": (
            float(np.mean([d.boundary == 0 for d in fired])) if fired else float("nan")),
    }
    if tail_start is not None:
        rel = [d.boundary - tail_start[d.problem_id] for d in fired
               if tail_start.get(d.problem_id) is not None]
        out["n_fired_with_tail"] = len(rel)
        out["share_fired_before_tail"] = (
            float(np.mean([r < 0 for r in rel])) if rel else float("nan"))
        out["median_boundary_minus_tail_start"] = (
            float(np.median(rel)) if rel else float("nan"))
    return out


def random_first_boundary_null(
    first_saving: Sequence[float],
    first_safe: Sequence[bool],
    target_saving: float,
    reps: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Safe rate of a rule that knows nothing, at a target mean saving.

    The rule stops at the first boundary on a uniformly random subset of
    problems and abstains on the rest. The subset size is chosen per draw so
    the mean saving over *all* problems is as close to ``target_saving`` as a
    whole number of problems allows, which is how the registered sweep matches
    budgets too.

    Stopping at the first boundary maximises the saving per stop, so this rule
    reaches a given budget with the fewest stops and the most abstentions. It
    is therefore the information-free rule that the abstention term of
    :func:`decompose_decisions` favours most, and the natural floor for a
    signal's safe rate in a regime where most stops are unsafe.
    """
    s0 = np.asarray(first_saving, dtype=float)
    ok0 = np.asarray(first_safe, dtype=bool)
    n = len(s0)
    if n == 0:
        return {"available": False, "reason": "no problems"}
    rates, savings = np.empty(reps), np.empty(reps)
    for r in range(reps):
        order = rng.permutation(n)
        cum = np.cumsum(s0[order]) / n
        k = int(np.argmin(np.abs(cum - target_saving))) + 1
        fired = order[:k]
        rates[r] = (n - k + int(ok0[fired].sum())) / n
        savings[r] = cum[k - 1]
    alpha = (1 - confidence) / 2
    return {
        "available": True,
        "n": n,
        "reps": reps,
        "target_saving": float(target_saving),
        "mean_achieved_saving": float(savings.mean()),
        "mean_safe_rate": float(rates.mean()),
        "interval": [float(np.quantile(rates, alpha)), float(np.quantile(rates, 1 - alpha))],
        "interval_kind": "central range of the null distribution over random subsets",
    }


def oracle_safe_saving(
    rows: Sequence[Mapping[str, Any]],
    safe_at: Callable[[Mapping[str, Any], int], bool | None],
) -> dict[str, Any]:
    """The most a rule could save on average without a single unsafe stop.

    For each problem, the largest token saving among boundaries where stopping
    is safe (``safe_at`` returns True), or 0 if there is none - abstaining is
    always safe. The mean over problems bounds every rule: at any matched
    budget above it, some stops must be unsafe whatever the signal.

    ``safe_at`` may return None for a boundary whose safety is unknown (for
    example, no forced answer was generated there). Unknown boundaries are
    skipped, so the result is then a **lower bound** on the oracle, and the
    count of skipped boundaries is reported.
    """
    best, unknown = [], 0
    for row in rows:
        total = row["total_tokens"]
        b = 0.0
        for i, used in enumerate(row["boundary_tokens"]):
            ok = safe_at(row, i)
            if ok is None:
                unknown += 1
                continue
            if ok and total:
                b = max(b, max(0.0, 1.0 - used / total))
        best.append(b)
    return {
        "n_problems": len(best),
        "mean_max_safe_saving": float(np.mean(best)) if best else float("nan"),
        "median_max_safe_saving": float(np.median(best)) if best else float("nan"),
        "n_boundaries_unknown": unknown,
        "is_lower_bound": unknown > 0,
    }


def cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[Any],
    reps: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Mean with a percentile interval that resamples whole clusters.

    Claims within one explanation, and explanations within one problem, are
    not independent draws; an interval that treats them as such is too narrow
    by an amount that depends on how alike they are. Resampling the cluster
    (here, the problem) is the same unit of analysis the primary comparison
    uses.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ids = np.unique(c)
    index = {k: np.flatnonzero(c == k) for k in ids}
    boot = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        boot[r] = v[np.concatenate([index[k] for k in pick])].mean()
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(v.mean()),
        "low": float(np.quantile(boot, alpha)),
        "high": float(np.quantile(boot, 1 - alpha)),
        "n_units": int(len(v)),
        "n_clusters": int(len(ids)),
        "reps": reps,
        "method": "percentile bootstrap over clusters",
    }


def paired_problem_means(
    records: Sequence[Mapping[str, Any]],
    kind_key: str,
    value_key: str,
    a: str,
    b: str,
    problem_key: str = "problem_id",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-problem means of ``value_key`` for kinds ``a`` and ``b``, paired.

    Only problems that have at least one record of both kinds are returned, in
    sorted order, so the two arrays line up element by element.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for r in records:
        k = r[kind_key]
        if k not in (a, b):
            continue
        acc.setdefault(r[problem_key], {}).setdefault(k, []).append(float(r[value_key]))
    pids = sorted(p for p, v in acc.items() if a in v and b in v)
    xa = np.array([np.mean(acc[p][a]) for p in pids])
    xb = np.array([np.mean(acc[p][b]) for p in pids])
    return pids, xa, xb
