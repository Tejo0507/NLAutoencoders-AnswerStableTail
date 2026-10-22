"""Contrasts that respect the sampling structure (D72).

The registered F10 test compares about four reconstruction samples per
problem with Mann–Whitney. The registered RQ2 test compares 6,007 claims from
120 problems with Wilcoxon. Both treat dependent observations as independent,
which is the error D29 and D41 corrected in O3 and RQ3. Duplicating every
observation within a problem leaves the evidence unchanged and shrinks those
p-values. The functions here are invariant to that duplication by
construction, and ``tests/test_clustered.py`` checks it.

Two estimands, chosen for the question each test asks:

* **F10** asks whether reconstruction differs between tail and matched windows
  *within a trace*. The unit is the problem. The estimand is the mean, over
  problems, of the within-problem difference between window-kind means
  (:func:`paired_problem_contrast`). A mixed model with a random intercept
  per problem estimates the same contrast weighted by precision, and is
  reported beside it (:func:`mixed_contrast`).
* **RQ2** asks about the population of claims. The estimand stays the
  claim-weighted mean. Only its uncertainty must be clustered: a percentile
  bootstrap over problems (``posthoc.cluster_bootstrap_mean``), a Wilcoxon
  test on problem means, and a mixed model with problem and
  explanation-within-problem effects.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .posthoc import paired_problem_means


def paired_problem_contrast(
    records: Sequence[Mapping[str, Any]],
    kind_key: str,
    value_key: str,
    a: str,
    b: str,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Within-problem contrast ``a − b``, with problems as the unit.

    Effect sizes: the mean difference, in the value's own units; ``d_z``, the
    mean over the SD of the per-problem differences; and the matched-pairs
    rank-biserial correlation, the paired analogue of the registered
    Cliff's δ.
    """
    pids, xa, xb = paired_problem_means(records, kind_key, value_key, a, b)
    d = xa - xb
    n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
    return {
        "n_problems": n,
        "mean_difference": float(d.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "d_z": float(d.mean() / d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else float("nan"),
        "rank_biserial_paired": rbc,
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(nz) else 1.0,
        "share_of_problems_a_lower": float(np.mean(d < 0)),
    }


def mixed_contrast(
    records: Sequence[Mapping[str, Any]],
    kind_key: str,
    value_key: str,
    a: str,
    b: str,
    nested_key: str | None = None,
) -> dict[str, Any]:
    """``value ~ 1[kind == a]`` with a random intercept per problem.

    If ``nested_key`` is given, it adds a variance component for that grouping
    within each problem (a window, an explanation). Fitted by REML. Returns the
    fixed effect for ``a`` relative to ``b``.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = [r for r in records if r[kind_key] in (a, b)]
    df = pd.DataFrame({"y": [float(r[value_key]) for r in rows],
                       "is_a": [float(r[kind_key] == a) for r in rows],
                       "problem": [r["problem_id"] for r in rows]})
    vc = None
    if nested_key is not None:
        df["nest"] = [str(r[nested_key]) for r in rows]
        vc = {"nest": "0 + C(nest)"}
    model = smf.mixedlm("y ~ is_a", df, groups=df["problem"], vc_formula=vc)
    fit = model.fit(reml=True, method=["lbfgs", "powell"])
    lo, hi = fit.conf_int().loc["is_a"]
    return {"estimate": float(fit.params["is_a"]), "se": float(fit.bse["is_a"]),
            "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["is_a"]),
            "n_obs": len(df), "n_problems": int(df["problem"].nunique()),
            "converged": bool(fit.converged),
            "random_effects": "problem" + (f" + {nested_key} within problem" if nested_key else "")}


def mixed_mean(values: Sequence[float], problems: Sequence[str],
               nested: Sequence[str] | None = None) -> dict[str, Any]:
    """Intercept-only mixed model: the mean with problem (and nested) random effects."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"y": np.asarray(values, float), "problem": list(problems)})
    vc = None
    if nested is not None:
        df["nest"] = [str(x) for x in nested]
        vc = {"nest": "0 + C(nest)"}
    fit = smf.mixedlm("y ~ 1", df, groups=df["problem"], vc_formula=vc).fit(
        reml=True, method=["lbfgs", "powell"])
    lo, hi = fit.conf_int().loc["Intercept"]
    return {"estimate": float(fit.params["Intercept"]), "se": float(fit.bse["Intercept"]),
            "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["Intercept"]),
            "n_obs": len(df), "converged": bool(fit.converged)}
