"""Independent check of ``reanalyse_o3_registered.py`` through the original code path (D69).

The reanalysis computes curves with the matrix engine in
``analysis/o3_registered.py``. This script recomputes its point estimates with
the code the registered pipeline uses:
- ``score_baselines.build_rows(outcome="forced")``;
- ``stopping.sweep_threshold`` over ``complete_threshold_grid``;
- ``stopping.at_matched_budget``;
- ``posthoc.oracle_safe_saving``.

It recomputes the oracle bound, every rule's safe rate and saving at every
budget, and the design-A gaps. It fails if any differs by more than 1e-9.

    python scripts/verify_o3_registered.py --config main

Writes ``runs/<run_id>_o3_registered/verification.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.posthoc import oracle_safe_saving
from nlaast.baselines.stopping import (
    OUTCOME_FORCED,
    at_matched_budget,
    complete_threshold_grid,
    sweep_threshold,
)
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json

RULES = ("nla_readout", "semantic_entropy", "probe", "convergence")


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reanalyse_o3_registered as ro
    from score_baselines import build_rows

    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("verify")
    out_dir = cfg.dir.parent / f"{cfg.run_id}_o3_registered"
    got = read_json(out_dir / "o3_registered.json")
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    _, completed, _, _, _, _ = ro.load_forced(cfg.dir.parent / "main_forced", ro.sha256(traces_path))
    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    rows = build_rows([traces[p] for p in pids], OUTCOME_FORCED, completed)

    def safe_at(row, i):
        if not row["answer_known"][i]:
            return None
        return equivalent(row["answers"][i], row["gold"]) == equivalent(row["final_answer"], row["gold"])

    checks, worst = [], 0.0

    def check(name, expected, actual):
        nonlocal worst
        d = abs(expected - actual)
        worst = max(worst, d)
        checks.append({"check": name, "expected": expected, "independent": actual, "abs_diff": d})

    for design, key in (("A", "design_A_shared_menu"), ("B", "design_B_all_boundaries")):
        menu = None if design == "B" else menu_sets
        restricted = [rows[pid] for pid in pids]

        def safe_on_menu(row, i, menu=menu):
            # Off the menu a rule cannot stop, so the boundary is not a safe stop.
            if menu is not None and i not in menu.get(row["problem_id"], set()):
                return False
            return safe_at(row, i)

        oracle = oracle_safe_saving(restricted, safe_on_menu)
        check(f"{design}: oracle bound", got[key]["oracle_bound"], oracle["mean_max_safe_saving"])
        for rule in RULES:
            per = []
            for r in restricted:
                sc = scores[rule][r["problem_id"]]
                if menu is not None:
                    sc = [s if i in menu.get(r["problem_id"], set()) else None for i, s in enumerate(sc)]
                per.append({**r, "scores": sc})
            vals = [s for r in per for s in r["scores"] if s is not None and np.isfinite(s)]
            curve = sweep_threshold(per, complete_threshold_grid(vals) + [np.inf], True)
            for row in got[key]["per_budget"]:
                if row[rule] is None:
                    continue
                pt = at_matched_budget(curve, row["budget"], tolerance=ro.TOL)
                if pt is None:
                    checks.append({"check": f"{design}: {rule} at {row['budget']:.4f}",
                                   "error": "engine matched, independent path did not"})
                    worst = np.inf
                    continue
                check(f"{design}: {rule} safe rate at {row['fraction_of_oracle']}·B*",
                      row[rule]["safe_rate"], pt.safe_rate)
                check(f"{design}: {rule} saving at {row['fraction_of_oracle']}·B*",
                      row[rule]["saving"], pt.mean_tokens_saved)
    ok = worst <= 1e-9
    out = {"ok": bool(ok), "max_abs_diff": worst, "n_checks": len(checks), "checks": checks}
    write_json(out_dir / "verification.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "max_abs_diff", "n_checks")}))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
