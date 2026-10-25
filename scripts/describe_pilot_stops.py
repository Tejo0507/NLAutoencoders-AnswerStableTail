"""Where the stopping-policy pilot's P1 stops on the evaluation set (post hoc, descriptive). CPU, read only.

The pilot's `evaluation.json` reports P1's mean net saving but not where in the
trace its stops fall. This script applies the frozen threshold with the same
first-crossing rule as `nlaast.analysis.stopping_pilot.walk` and reports, over
the problems on which P1 stops, the stop's prefix length and its forced-answer
length as shares of the trace's tokens, and how many stops save nothing once
the forced answer is charged (`net_saving`'s rule). It is added after the
results were seen (D102, D103) and reads only the pilot's frozen outputs. It
writes one new file under `runs/pilot_posthoc/` and refuses to overwrite.

    python scripts/describe_pilot_stops.py <pilot repo dir> [--out NAME.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nlaast.analysis.stopping_pilot import net_saving  # noqa: E402

POLICY = "P1_same_label_probe"


def describe(rows: Iterable[Mapping], tau: float, policy: str = POLICY) -> dict:
    by: dict[str, list[Mapping]] = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    prefix, forced, index, n_boundaries, saving, at_end = [], [], [], [], [], []
    for pid in sorted(by):
        rs = sorted(by[pid], key=lambda r: r["chunk"])
        k = next((i for i, r in enumerate(rs) if r[policy] >= tau), None)
        if k is None:
            continue
        r = rs[k]
        prefix.append(r["prefix_tokens"] / r["total_tokens"])
        forced.append(r["forced_tokens"] / r["total_tokens"])
        saving.append(net_saving(r["prefix_tokens"], r["forced_tokens"], r["total_tokens"]))
        at_end.append(r["prefix_tokens"] >= r["total_tokens"])
        index.append(k + 1)
        n_boundaries.append(len(rs))
    if not prefix:
        return {"n_problems": len(by), "n_stopped": 0}
    zero = sum(1 for s in saving if s == 0.0)
    return {
        "n_problems": len(by), "n_stopped": len(prefix),
        "prefix_share_mean": statistics.mean(prefix),
        "prefix_share_median": statistics.median(prefix),
        "forced_share_mean": statistics.mean(forced),
        "stop_boundary_mean": statistics.mean(index),
        "boundaries_per_stopped_problem_mean": statistics.mean(n_boundaries),
        "stops_at_first_boundary": sum(1 for i in index if i == 1),
        "stops_with_zero_saving": zero,
        "stops_at_trace_end": sum(at_end),
        "problems_with_a_saving_stop": len(prefix) - zero,
        "share_with_a_saving_stop": (len(prefix) - zero) / len(by),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path, help="the pilot's repository clone (read only)")
    ap.add_argument("--out", default="stop_positions.json",
                    help="file name under runs/pilot_posthoc/ (never overwritten)")
    a = ap.parse_args(argv)
    runs = a.repo / "runs"
    cal_path = runs / "pilot_calibration" / "calibration.json"
    recorded = (runs / "pilot_calibration" / "calibration.sha256").read_text().strip()
    if hashlib.sha256(cal_path.read_bytes()).hexdigest() != recorded:
        raise SystemExit("the calibration changed after it was frozen")
    tau = json.loads(cal_path.read_text(encoding="utf-8"))["policies"][POLICY]["tau"]
    with open(runs / "pilot_eval" / "pilot_scores.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    out = {"status": "post hoc, descriptive (D102, D103); frozen threshold, first-crossing rule",
           "tau": tau, POLICY: describe(rows, tau)}
    dest = runs / "pilot_posthoc"
    dest.mkdir(exist_ok=True)
    path = dest / a.out
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
