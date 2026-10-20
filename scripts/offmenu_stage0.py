"""Stage 0 of the leakage-resistant design (D83): cheap signals at every boundary. CPU.

Implements ``docs/LEAKAGE_RESISTANT_DESIGN.md`` §2 on ``main``. Every chunk
boundary of every trace is a unit, so no unit is chosen with the tail start.
It asks how much stopping-safety information position, the registered probe,
a same-label activation probe and the raw trace text carry there, and how
those signals do as stopping rules against no signal and the oracle.
There is no NLA here: it exists only on the menu.

    python scripts/offmenu_stage0.py --config main --forced-dir runs/main_forced

Writes ``runs/<run_id>_offmenu_stage0/stage0.json`` and refuses to replace it
(``--force``). Nothing under ``runs/<run_id>/`` is written.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.information import (
    Design,
    compare,
    grouped_folds,
    nested_probe,
    out_of_fold,
    problem_draws,
    weighted_auc,
    weighted_log_loss,
    weights_for,
    within_problem_concordance,
)
from nlaast.analysis.o3_registered import Problem
from nlaast.analysis.position import position_scores
from nlaast.analysis.selective import fire_probability, selective_comparisons
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.trace.chunking import Chunk, prefix_text

PLAN = Path(__file__).resolve().parents[1] / "docs" / "LEAKAGE_RESISTANT_DESIGN.md"
SEED = 20261020
REPS = 2000
N_FOLDS = 10
R_CHARS = 600
MODELS = ("pos", "pos+probe", "pos+A", "pos+R", "pos+A+R")
COVERAGES = (0.25, 0.5, 0.75)
COVERAGE_TOL = 0.03
INFO_CONTRASTS = {
    "Stage0-P A over pos": ("pos+A", "pos"),
    "probe over pos": ("pos+probe", "pos"),
    "R over pos": ("pos+R", "pos"),
    "A over pos+probe (head to head)": ("pos+A", "pos+probe"),
    "R over pos+A": ("pos+A+R", "pos+A"),
}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", required=True)
    p.add_argument("--reps", type=int, default=REPS)
    args = p.parse_args()
    cfg = load_config(args)
    out_path = cfg.dir.parent / f"{cfg.run_id}_offmenu_stage0" / "stage0.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import position_truncation_audit as pta
    import reanalyse_o3_registered as ro
    from extended_forcing_analysis import tokenizer
    from force_extended import generated_part, registered_completions
    from information_audit import activations_for

    logging.basicConfig(level=logging.ERROR)
    t0 = time.monotonic()
    def progress(msg):
        print(f"  [{time.monotonic() - t0:6.0f}s] {msg}", flush=True)

    forced_dir = Path(args.forced_dir)
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    prov, completed, _, _, agree, _ = ro.load_forced(forced_dir, ro.sha256(traces_path))
    if agree is None or agree < ro.MIN_AGREEMENT:
        raise SystemExit(f"forcing instrument agreement {agree}")
    pids = sorted(traces)
    problems = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)
    unknown = sum(int((~pr.known).sum()) for pr in problems)
    if unknown:
        raise SystemExit(f"{unknown} boundaries have no measured outcome; Stage 0 needs all")
    base = read_json(cfg.dir / "baselines" / "boundary_scores.json")
    ts = {r["problem_id"]: r.get("tail_start") for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}

    units = []
    for pr in problems:
        t = traces[pr.problem_id]
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")}) for c in t["chunks"]]
        for i in range(pr.n):
            units.append({"problem_id": pr.problem_id, "chunk": i, "dataset": t.get("dataset"),
                          "prefix_tokens": int(t["chunks"][i]["prefix_tokens"]),
                          "probe": base[pr.problem_id]["probe"][i],
                          "R": prefix_text(chunks, i, t["trace"])[-R_CHARS:],
                          "Y_reg": bool(pr.safe[i]),
                          "Y_tail": None if ts.get(pr.problem_id) is None else bool(i >= ts[pr.problem_id])})
    progress(f"{len(units)} boundary units in {len(pids)} problems")
    X = activations_for(units, cfg.dir / "acts", cfg.target.layer)
    keys, counts = problem_draws(pids, args.reps, np.random.default_rng(SEED))

    def numeric(sub, spec):
        cols = []
        for u in sub:
            row = []
            if "pos" in spec:
                row += [np.log(max(1, u["prefix_tokens"])), u["chunk"], float(u["dataset"] == "math")]
            if "probe" in spec:
                q = u["probe"]
                row += [0.0 if q is None else float(_logit(q)), float(q is None)]
            cols.append(row)
        return np.asarray(cols, float)

    out = {"status": "exploratory (D83 Stage 0): every boundary is a unit; no NLA",
           "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
           "plan_sha256": ro.sha256(PLAN), "seed": SEED, "reps": args.reps,
           "n_problems": len(pids), "n_units": len(units),
           "label_rates": {lab: float(np.mean([u[lab] for u in units if u[lab] is not None]))
                           for lab in ("Y_reg", "Y_tail")}}
    oof_reg = None
    for lab in ("Y_reg", "Y_tail"):
        keep = [k for k, u in enumerate(units) if u[lab] is not None]
        sub = [units[k] for k in keep]
        y = np.array([u[lab] for u in sub], bool)
        groups = np.array([u["problem_id"] for u in sub])
        folds = grouped_folds(y, groups, N_FOLDS, SEED)
        progress(f"{lab}: nested activation probe on {len(y)} units")
        extras, Cs = nested_probe(X[keep], y, groups, folds, seed=SEED)
        oof = {}
        for spec in MODELS:
            parts = spec.split("+")
            d = Design(numeric(sub, parts), (([u["R"] for u in sub],) if "R" in parts else ()))
            oof[spec] = out_of_fold(d, y, folds, extras if "A" in parts else None)
            progress(f"{lab} {spec}: AUC {weighted_auc(y, oof[spec]):.3f}")
        W = weights_for(groups, keys, counts)
        out[lab] = {"n_units": int(len(y)), "positive_rate": float(y.mean()), "chosen_C": Cs,
                    "auc": {m: weighted_auc(y, v) for m, v in oof.items()},
                    "log_loss": {m: weighted_log_loss(y, v) for m, v in oof.items()},
                    "within_problem_concordance": {m: within_problem_concordance(y, v, groups)
                                                   for m, v in oof.items()},
                    "contrasts": {k: compare(y, oof[a], oof[b], W)
                                  for k, (a, b) in INFO_CONTRASTS.items()}}
        if lab == "Y_reg":
            oof_reg = (sub, oof)

    # ---- stopping policies over every boundary (descriptive) ----
    sub, oof = oof_reg
    learned = {m: {} for m in MODELS}
    for k, u in enumerate(sub):
        for m in MODELS:
            learned[m].setdefault(u["problem_id"], []).append(float(oof[m][k]))
    prefix = {q: [int(c["prefix_tokens"]) for c in traces[q]["chunks"]] for q in pids}
    total = {q: int(traces[q]["n_tokens"]) for q in pids}
    sc = {f"learned {m}": learned[m] for m in MODELS}
    sc["registered probe"] = {q: base[q]["probe"] for q in pids}
    sc["position_tokens"] = {q: position_scores(prefix[q], total[q], "tokens") for q in pids}
    sc["position_tokens_falling"] = {q: position_scores(prefix[q], total[q], "tokens_falling")
                                     for q in pids}
    pairs = {f"{r} - no_signal": (r, "no_signal") for r in sc}
    pairs["learned pos+A - learned pos"] = ("learned pos+A", "learned pos")
    pairs["learned pos+A - position_tokens_falling"] = ("learned pos+A", "position_tokens_falling")
    pairs["learned pos+A - registered probe"] = ("learned pos+A", "registered probe")

    reg48 = registered_completions(traces, forced_dir)
    tok = tokenizer(cfg)
    suffix = cfg.generation.force_answer_suffix
    net = []
    for pr in problems:
        q = pr.problem_id
        sav = np.array([max(0.0, 1 - (prefix[q][i] + len(tok(generated_part(reg48[(q, i)]["text"], suffix),
                                                               add_special_tokens=False)["input_ids"]))
                             / total[q]) for i in range(pr.n)])
        net.append(Problem(q, sav, pr.safe, pr.known))
    out["policies"] = {}
    for name, probs in (("registered savings", problems), ("net savings", net)):
        progress(f"policies, {name}")
        bstar, grid, mats, (ES, EQ), gaps = pta.build_design(ro, probs, sc, None, args.reps, ro.SEED,
                                                             pairs, ro.MIN_BUDGETS)
        cols = {r: [ro.match(m.curve()[0], b, ro.TOL) for b in grid] for r, m in mats.items()}
        # D83 deviation 1 (post hoc): selective risk at matched coverage
        FP = fire_probability(probs, None, ro.QGRID)
        selective = selective_comparisons(mats, (FP, ES, EQ), COVERAGES, COVERAGE_TOL, args.reps,
                                          np.random.default_rng(ro.SEED))
        out["policies"][name] = {
            "selective_risk_at_coverage (deviation 1, post hoc)": selective,
            "oracle_bound": bstar, "budgets": grid,
            "gaps": {k: {"gap": v["gap"], "ci95": v["ci95"], "p": v["p_bootstrap"],
                         "n_budgets_matched": v["n_budgets_matched"]} for k, v in gaps.items()},
            "at_half_bstar": {r: (None if cols[r][4] is None else {
                "saving": float(mats[r].curve()[0][cols[r][4]]),
                "safe_rate": float(mats[r].curve()[1][cols[r][4]]),
                "n_stops": int(mats[r].fired[:, cols[r][4]].sum()),
                "n_safe_stops": int((mats[r].fired[:, cols[r][4]] & mats[r].safe[:, cols[r][4]]).sum())})
                for r in mats},
        }
    out["elapsed_s"] = time.monotonic() - t0
    if out_path.exists():
        # a regeneration must reproduce every number the old file holds
        old = read_json(out_path)
        for lab in ("Y_reg", "Y_tail"):
            for k, v in old[lab]["contrasts"].items():
                if abs(out[lab]["contrasts"][k]["delta_auc"] - v["delta_auc"]) > 1e-9:
                    raise SystemExit(f"{lab} {k} did not reproduce; not overwriting")
        for n, pol in old["policies"].items():
            for k, v in pol["gaps"].items():
                a, b = out["policies"][n]["gaps"][k]["gap"], v["gap"]
                if not ((a != a and b != b) or abs(a - b) <= 1e-9):
                    raise SystemExit(f"policy {n} {k} did not reproduce; not overwriting")
        out["reproduced_previous"] = True
    write_json(out_path, out)
    summ = {lab: {k: [round(v["delta_auc"], 4), [round(x, 4) for x in v["ci95"]], v["p_bootstrap"]]
                  for k, v in out[lab]["contrasts"].items()} for lab in ("Y_reg", "Y_tail")}
    summ["policies"] = {n: {k: [None if v["gap"] != v["gap"] else round(v["gap"], 4), v["ci95"]]
                            for k, v in pol["gaps"].items()} for n, pol in out["policies"].items()}
    print(json.dumps(summ, indent=1, default=str))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
