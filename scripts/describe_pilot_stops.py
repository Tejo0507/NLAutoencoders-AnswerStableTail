"""Where the stopping-policy pilot's P1 stops on the evaluation set (post hoc, descriptive). CPU, read only.

The pilot's `evaluation.json` reports P1's mean net saving but not where in the
trace its stops fall. This script applies the frozen threshold with the same
first-crossing rule as `nlaast.analysis.stopping_pilot.walk` and reports, over
the problems on which P1 stops, the stop's prefix length and its forced-answer
length as shares of the trace's tokens. It is added after the results were seen
(D102) and reads only the pilot's frozen outputs. It writes one new file and
refuses to overwrite anything.

    python scripts/describe_pilot_stops.py <pilot repo dir>
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping

POLICY = "P1_same_label_probe"


def describe(rows: Iterable[Mapping], tau: float, policy: str = POLICY) -> dict:
    by: dict[str, list[Mapping]] = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    prefix, forced, index, n_boundaries = [], [], [], []
    for pid in sorted(by):
        rs = sorted(by[pid], key=lambda r: r["chunk"])
        k = next((i for i, r in enumerate(rs) if r[policy] >= tau), None)
        if k is None:
            continue
        r = rs[k]
        prefix.append(r["prefix_tokens"] / r["total_tokens"])
        forced.append(r["forced_tokens"] / r["total_tokens"])
        index.append(k + 1)
        n_boundaries.append(len(rs))
    if not prefix:
        return {"n_problems": len(by), "n_stopped": 0}
    return {
        "n_problems": len(by), "n_stopped": len(prefix),
        "prefix_share_mean": statistics.mean(prefix),
        "prefix_share_median": statistics.median(prefix),
        "forced_share_mean": statistics.mean(forced),
        "stop_boundary_mean": statistics.mean(index),
        "boundaries_per_stopped_problem_mean": statistics.mean(n_boundaries),
        "stops_at_first_boundary": sum(1 for i in index if i == 1),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runs = Path(argv[0]) / "runs"
    cal_path = runs / "pilot_calibration" / "calibration.json"
    recorded = (runs / "pilot_calibration" / "calibration.sha256").read_text().strip()
    if hashlib.sha256(cal_path.read_bytes()).hexdigest() != recorded:
        raise SystemExit("the calibration changed after it was frozen")
    tau = json.loads(cal_path.read_text(encoding="utf-8"))["policies"][POLICY]["tau"]
    with open(runs / "pilot_eval" / "pilot_scores.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    out = {"status": "post hoc, descriptive (D102); frozen threshold, first-crossing rule",
           "tau": tau, POLICY: describe(rows, tau)}
    dest = runs / "pilot_posthoc"
    dest.mkdir(exist_ok=False)
    (dest / "stop_positions.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
