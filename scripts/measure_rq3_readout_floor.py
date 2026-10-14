"""Can RQ3's registered readout ever move on this model?

RQ3 asks whether tail-window states carry a direction whose ablation
*selectively changes verification-like behaviour* without changing final
answers. That construct is registered (`Project_Review_II.md` line 122); the
measure is not, and the one the causal stage uses is a count of lexical
rechecking markers (`VERIFICATION_MARKERS`). A comparison needs the outcome to
vary, and `paired_arm_contrast` needs at least `MIN_PAIRS` problems where it
does.

So the question of whether RQ3 is evaluable at all on this model is a question
about a **base rate**, and the traces answer it without running an
intervention: how often does an unintervened trace contain any marker? Nothing
an edit does to the residual stream can create a comparison from an outcome
that is identically zero across the arms.

This script measures that rate over whatever traces are on disk and projects it
onto the configured causal arm. It is the evidence behind D41's conclusion that
the obstacle is the model rather than the corpus size, and it is worth re-
running whenever the corpus grows - **the projection has moved against the
readout as it has, not for it**: the interval's upper bound falls as N rises,
so each additional trace without a marker makes the arm less viable rather than
more.

Read-only and CPU-only. No model is loaded and no intervention is run, so it is
safe while a run is still generating.

    python scripts/measure_rq3_readout_floor.py
    python scripts/measure_rq3_readout_floor.py --run runs/main --arm 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy import stats  # noqa: E402

from nlaast.causal.interventions import (  # noqa: E402
    MIN_PAIRS, VERIFICATION_MARKERS, count_verification_markers)


def measure(path: Path) -> tuple[int, int, int, dict[str, int]]:
    """``(marker_bearing_traces, traces, occurrences, per_marker)``."""
    n = bearing = occurrences = 0
    per_marker: dict[str, int] = {}
    if not path.exists():
        return 0, 0, 0, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        if not trace.get("ok"):
            continue
        n += 1
        text = trace.get("trace") or ""
        count = count_verification_markers(text)
        if not count:
            continue
        bearing += 1
        occurrences += count
        low = text.lower()
        for marker in VERIFICATION_MARKERS:
            hits = low.count(marker)
            if hits:
                per_marker[marker] = per_marker.get(marker, 0) + hits
    return bearing, n, occurrences, per_marker


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", type=Path, default=None,
                    help="run directory; repeatable. Default: runs/pilot and "
                         "runs/main, pooled.")
    ap.add_argument("--arm", type=int, default=60,
                    help="problems in the causal arm (causal.max_problems)")
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    runs = args.run or [ROOT / "runs" / "pilot", ROOT / "runs" / "main"]
    runs = [r if r.is_absolute() else ROOT / r for r in runs]

    total = bearing = occurrences = 0
    per_marker: dict[str, int] = {}
    print(f"{len(VERIFICATION_MARKERS)} markers in the registered list\n")
    for run in runs:
        b, n, occ, pm = measure(run / "traces" / "traces.jsonl")
        if not n:
            print(f"  {run.name:<8} no usable traces")
            continue
        print(f"  {run.name:<8} {b} marker-bearing of {n} traces, "
              f"{occ} occurrence(s)")
        total += n
        bearing += b
        occurrences += occ
        for k, v in pm.items():
            per_marker[k] = per_marker.get(k, 0) + v

    if not total:
        print("\nno traces to measure")
        return 1

    rate = bearing / total
    ci = stats.binomtest(bearing, total, 0.5).proportion_ci(
        confidence_level=args.confidence, method="exact")
    print(f"\npooled: {bearing}/{total} traces carry any marker = {rate:.4f}")
    print(f"        {int(100 * args.confidence)}% exact CI "
          f"[{ci.low:.4f}, {ci.high:.4f}]")
    print(f"        markers that occur at all: "
          f"{per_marker if per_marker else 'none'}")
    if occurrences:
        print(f"        {occurrences} occurrence(s) in "
              f"{total} traces, so even a trace that has one has about "
              f"{occurrences / max(bearing, 1):.1f}")

    print(f"\nprojected onto a {args.arm}-problem causal arm, against "
          f"MIN_PAIRS = {MIN_PAIRS}:")
    print(f"  at the point rate          {args.arm * rate:.2f} "
          f"marker-bearing problems")
    print(f"  at the interval's top end  {args.arm * ci.high:.2f}")
    verdict = ("UNEVALUABLE" if args.arm * ci.high < MIN_PAIRS
               else "possible at the top of the interval")
    print(f"  verdict: {verdict}")
    if ci.high:
        print(f"\n{MIN_PAIRS} marker-bearing problems are not expected until "
              f"the arm holds about {MIN_PAIRS / ci.high:.0f} problems, even "
              f"at the interval's top end.")
    print("\nThe bound falls as the corpus grows while the count does not, so "
          "more traces make this readout less viable rather than more. The "
          "constraint is the model's prose, not N: a construct cannot be "
          "measured by an instrument with no range on the population it is "
          "applied to. Routes out are in DECISIONS.md D41 - register a "
          "validated in-construct measure before a confirmatory run, or amend "
          "the question and say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
