"""How many problems the O3 head-to-head needs, given how it is now tested.

The matched-budget comparison is a McNemar test over problems, so its power is
set by the number of *discordant* problems - those where one rule preserved
the verified outcome and the other did not. Concordant problems, however many,
contribute nothing. Sizing the corpus by total problems is therefore sizing it
by the wrong quantity, and the pilot's own discordance rate is what converts
one into the other.

Run with no arguments to size from the pilot, or pass ``--run <dir>``.

This exists because the pilot's O3 finding was reported at p = 0.0039 by
pairing over nine requested budgets that all resolved to a single operating
point. Tested over problems the same data give p = 0.125 on four discordant
problems. The question "how much more data would settle it" only has an answer
under the corrected test, and that answer is what this script computes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

#: BH with m tests gives the smallest p a q-value of m * p, so a test has to
#: clear roughly ``q / m`` to survive correction.
#:
#: This default is **the pilot's evaluable count**: nine tests were
#: pre-registered and five produced a p-value, and ``benjamini_hochberg``
#: corrects over the evaluable ones. It is therefore a property of the run it
#: was read off, not a constant - a run on which more of the family becomes
#: evaluable has a larger m and a stricter target. Pass ``--family-size`` to
#: match the run being sized; the script warns when the run's own evaluable
#: count disagrees with the value in use.
#:
#: ``m`` is now read from the run being sized rather than assumed, so the
#: sizing and the analysis cannot drift apart; this constant is only the
#: fallback for a run whose analysis has not been written yet. The convention -
#: correct over the evaluable tests, and report the pre-registered count as a
#: sensitivity - is settled in D38 and implemented in
#: ``nlaast.analysis.stats.benjamini_hochberg``. It is not an independent
#: decision here and must not be changed without changing both.
#:
#: Note that the pilot's own evaluable count becomes **6** once D38's fix is
#: applied and its analysis is re-run, because the zero-discordance comparison
#: is a test with p = 1.0 rather than a missing one. That tightens the target
#: from 0.0100 to 0.0083 and leaves the sizing conclusion unchanged: eight
#: discordant problems either way.
DEFAULT_FAMILY_SIZE = 5
DEFAULT_Q = 0.05


def exact_p(b: int, d: int) -> float:
    """Two-sided McNemar exact p for b of d discordant pairs favouring A."""
    if d == 0:
        return float("nan")
    return float(stats.binomtest(b, d, 0.5).pvalue)


def required_discordant(p_target: float) -> int:
    """Fewest discordant problems that can reach ``p_target``, best case.

    The best case is every discordant pair pointing the same way, which gives
    ``p = 2 * 0.5**d``. Anything less lopsided needs more.
    """
    d = 1
    while d < 1000:
        if exact_p(0, d) <= p_target:
            return d
        d += 1
    return -1


def power_at(n_problems: int, rate: float, phi: float, p_target: float,
             n_sims: int, rng) -> float:
    """Probability of clearing ``p_target``, simulating the whole pipeline.

    ``rate`` is the chance a problem is discordant and ``phi`` the chance a
    discordant problem favours the readout. ``phi = 0`` reproduces the pilot,
    where all four discordant problems favoured semantic entropy; a larger phi
    is the honest hedge, since four observations pin it down poorly.
    """
    d = rng.binomial(n_problems, rate, size=n_sims)
    hits = 0
    for di in d:
        if di == 0:
            continue
        b = rng.binomial(di, phi)
        if exact_p(int(b), int(di)) <= p_target:
            hits += 1
    return hits / n_sims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/pilot", type=Path)
    ap.add_argument(
        "--family-size", type=int, default=None,
        help="BH family size m. Default: read from the run's own analysis "
             f"(bh_family_size), falling back to {DEFAULT_FAMILY_SIZE}.")
    ap.add_argument("--q", type=float, default=DEFAULT_Q)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    results = json.loads((args.run / "analysis" / "results.json").read_text())
    observed = {}
    for t in results["tests"]["tests"]:
        if not t["name"].startswith("O3"):
            continue
        n = t.get("notes") or {}
        if n.get("test") != "mcnemar_exact":
            continue
        other = t["name"].split("minus ")[1].split(" at ")[0]
        observed[other] = {
            "n_problems": n["n_problems"],
            "n_discordant": n["n_discordant"],
            "favouring_nla": n["a_safe_b_not"],
            "favouring_other": n["b_safe_a_not"],
            "p": t["p_value"],
        }

    if not observed:
        print("no McNemar-tested O3 comparison in this run")
        return 1

    # The target is only right if it uses the m this run's analysis actually
    # corrected over, so m is read from the run rather than assumed. Sizing
    # against a stale constant silently targets the wrong p (D38). An explicit
    # --family-size still wins, and where m came from is always printed.
    run_m = results["tests"].get("bh_family_size")
    if run_m is None:
        run_m = results["tests"].get("n_evaluated")

    if args.family_size is not None:
        family_size, source = args.family_size, "--family-size"
        if run_m is not None and int(run_m) != family_size:
            print(f"WARNING: this run's analysis corrected over m = {run_m}, "
                  f"but --family-size {family_size} was given. The target "
                  f"below is not the one the analysis applied.\n")
    elif run_m:
        family_size, source = int(run_m), "this run's analysis"
    else:
        family_size, source = DEFAULT_FAMILY_SIZE, "fallback default"
        print(f"NOTE: this run records no BH family size; falling back to "
              f"m = {DEFAULT_FAMILY_SIZE}.\n")

    p_target = args.q / family_size
    d_min = required_discordant(p_target)
    print(f"target p for q < {args.q} in a family of {family_size} "
          f"evaluable tests ({source}): {p_target:.4f}")
    print(f"fewest discordant problems that can reach it (all one way): {d_min}\n")

    rng = np.random.default_rng(args.seed)
    for other, obs in sorted(observed.items()):
        d, n = obs["n_discordant"], obs["n_problems"]
        rate = d / n if n else 0.0
        print(f"--- {other} ---")
        print(f"  observed: {d} discordant of {n} problems "
              f"({obs['favouring_nla']} favour the readout, "
              f"{obs['favouring_other']} favour {other}), p = {obs['p']:.4f}")
        if d == 0:
            hi = stats.binomtest(0, n, 0.5).proportion_ci(
                confidence_level=0.95, method="exact").high
            print(f"  no discordant problems: the two rules preserved the "
                  f"outcome on the same {n} problems. The discordance rate is "
                  f"under {hi:.3f} with 95% confidence, so a corpus of about "
                  f"{int(np.ceil(d_min / hi)) if hi else 0} problems would be "
                  f"needed before {d_min} discordant ones are even expected.")
            print()
            continue
        print(f"  discordance rate {rate:.3f}; {d_min} discordant problems "
              f"need about {int(np.ceil(d_min / rate))} problems in the arm\n")
        print(f"  {'phi':>5}{'N=24':>8}{'N=50':>8}{'N=80':>8}"
              f"{'N=120':>8}{'N=250':>8}")
        for phi in (0.0, 0.1, 0.2, 0.3):
            row = [power_at(n_, rate, phi, p_target, args.sims, rng)
                   for n_ in (24, 50, 80, 120, 250)]
            print(f"  {phi:>5.1f}" + "".join(f"{v:>8.2f}" for v in row))
        print("\n  phi is the chance a discordant problem favours the readout;"
              "\n  0.0 is the pilot's own pattern and the most optimistic case.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
