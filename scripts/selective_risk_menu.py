"""Selective risk at matched coverage on design A's menu (D83 deviation 1). CPU.

The complement to the registered estimand: among problems a rule stops in
before the end, how often is the stop unsafe, at the same share of problems
stopped? The rules and scores are D79's; the stored registered analysis is
checked first, and nothing here enters any verdict. Post hoc on ``main`` and
``main_nla240``; specified before any outcome for ``rep1``.

    python scripts/selective_risk_menu.py --config main
    python scripts/selective_risk_menu.py --config main --run-id main_nla240
    python scripts/selective_risk_menu.py --config replication --forced-dir runs/rep1_forced

Writes ``runs/<run_id>_selective/selective_menu.json`` and refuses to replace it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.o3_registered import no_signal_expectations, rule_matrix
from nlaast.analysis.position import position_scores, residualise_on_position
from nlaast.analysis.selective import fire_probability, selective_comparisons
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

COVERAGES = (0.25, 0.5, 0.75)
COVERAGE_TOL = 0.03
RULES = ("nla_readout", "probe", "convergence", "position_tokens", "position_tokens_falling",
         "readout_residualised")


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", default=None, help="default: runs/main_forced")
    p.add_argument("--reps", type=int, default=2000)
    args = p.parse_args()
    cfg = load_config(args)
    out_path = cfg.dir.parent / f"{cfg.run_id}_selective" / "selective_menu.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import reanalyse_o3_registered as ro

    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("selective")
    forced_dir = Path(args.forced_dir) if args.forced_dir else cfg.dir.parent / "main_forced"
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    stored = read_json(cfg.dir.parent / f"{cfg.run_id}_o3_registered" / "o3_registered.json")
    for k, path in {"traces": traces_path, "forced": forced_dir / "forced.jsonl",
                    "reconstructions": cfg.dir / "nla" / "reconstructions.jsonl",
                    "boundary_scores": cfg.dir / "baselines" / "boundary_scores.json"}.items():
        if stored["inputs_sha256"][k] != ro.sha256(path):
            raise SystemExit(f"{k} changed since the registered analysis")
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    _, completed, _, _, agree, _ = ro.load_forced(forced_dir, ro.sha256(traces_path))
    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    menu = {pid: np.array([i in menu_sets.get(pid, set()) for i in range(len(traces[pid]["chunks"]))])
            for pid in pids}
    problems = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)
    if abs(ro.oracle_bound(problems, menu) - stored["design_A_shared_menu"]["oracle_bound"]) > 1e-9:
        raise SystemExit("the stored design-A oracle bound did not reproduce")
    prefix = {q: [int(c["prefix_tokens"]) for c in traces[q]["chunks"]] for q in pids}
    total = {q: int(traces[q]["n_tokens"]) for q in pids}
    sc = {r: scores[r] for r in ("nla_readout", "probe", "convergence")}
    sc["position_tokens"] = {q: position_scores(prefix[q], total[q], "tokens") for q in pids}
    sc["position_tokens_falling"] = {q: position_scores(prefix[q], total[q], "tokens_falling") for q in pids}
    sc["readout_residualised"] = residualise_on_position(
        {q: scores["nla_readout"][q] for q in pids}, prefix, menu)
    mats = {r: rule_matrix(problems, sc[r], menu) for r in RULES}
    ES, EQ = no_signal_expectations(problems, menu, ro.QGRID)
    FP = fire_probability(problems, menu, ro.QGRID)
    res = selective_comparisons(mats, (FP, ES, EQ), COVERAGES, COVERAGE_TOL, args.reps,
                                np.random.default_rng(ro.SEED))
    out = {"status": ("specified before any rep1 outcome; secondary (D83 deviation 1)"
                      if cfg.run_id == "rep1" else "post hoc (D83 deviation 1)"),
           "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(), "seed": ro.SEED,
           "reps": args.reps, "n_problems": len(pids), "instrument_agreement": agree,
           "coverage_tolerance": COVERAGE_TOL, "selective_risk_at_coverage": res}
    write_json(out_path, out)
    for c, row in res.items():
        print(f"coverage {c}: no signal {row['no_signal']}")
        for r in RULES:
            e = row[r]
            if e["at"] is None:
                print(f"   {r}: not reached")
                continue
            print(f"   {r}: risk {e['at']['selective_risk']:.3f} saving {e['at']['saving_among_stops']:.3f}"
                  f"  Δ vs no signal {e['risk_minus_no_signal']:+.3f} {e['ci95']}")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
