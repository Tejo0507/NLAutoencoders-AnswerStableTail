"""O3 re-scored on the registered (forced-answer) outcome (D69). CPU only.

Implements ``docs/O3_REGISTERED_REANALYSIS_PLAN.md`` (SHA-256 recorded in D69)
on the stored run and the completed forced answers from
``scripts/force_all_boundaries.py``. **Corrected and post hoc**: the
registered O3 test stays in the record as computed, and nothing under
``runs/<run_id>/`` is written.

Before it reports anything it:

* checks that the completion file was generated from these traces, with this
  instrument;
* stops if the instrument reproduced on fewer than 0.95 of re-forced stored
  boundaries (plan §2);
* reproduces the historical parsed-outcome head-to-heads exactly, from the
  run's own functions;
* checks that restricting the readout to its own menu changes nothing.

    python scripts/reanalyse_o3_registered.py --config main
    python scripts/reanalyse_o3_registered.py --config main --run-id main_nla240

Writes ``runs/<run_id>_o3_registered/o3_registered.json`` and refuses to
replace it (``--force``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.o3_registered import (
    Problem,
    bootstrap_p,
    gap_over_budgets,
    holm,
    match,
    no_signal_expectations,
    no_signal_monte_carlo,
    oracle_bound,
    rule_matrix,
)
from nlaast.analysis.stats import paired_binary_test
from nlaast.baselines.stopping import OUTCOME_FORCED, OUTCOME_PARSED
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS = 2000
MC_DRAWS = 2000
TOL = 0.02
FRACTIONS = [round(0.1 * k, 1) for k in range(1, 10)]
MARGIN = 0.05
MIN_AGREEMENT = 0.95
MIN_BUDGETS = 3
QGRID = np.linspace(0.0, 1.0, 2001)
RULES = ("nla_readout", "semantic_entropy", "probe", "convergence")
COMPARATORS = ("semantic_entropy", "probe", "convergence")
PLAN = Path(__file__).resolve().parents[1] / "docs" / "O3_REGISTERED_REANALYSIS_PLAN.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scripts():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def load_forced(forced_dir: Path, traces_sha: str):
    prov = read_json(forced_dir / "provenance.json")
    if prov.get("traces_sha256") != traces_sha:
        raise SystemExit(f"{forced_dir} was generated from other traces "
                         f"({prov.get('traces_sha256')} vs {traces_sha})")
    rows = read_jsonl(forced_dir / "forced.jsonl")
    completed = {(r["problem_id"], int(r["index"])): r["forced_answer"]
                 for r in rows if r["purpose"] == "missing"}
    regenerated = {(r["problem_id"], int(r["index"])): r["forced_answer"]
                   for r in rows if r["purpose"] == "repro"}
    finished = {(r["problem_id"], int(r["index"])): bool(r["forced_finished"])
                for r in rows if r["purpose"] == "missing"}
    rep = [r for r in rows if r["purpose"] == "repro"]
    agree = sum(bool(r["agrees_with_stored"]) for r in rep) / len(rep) if rep else None
    return prov, completed, regenerated, finished, agree, len(rep)


def problems_for(traces, pids, outcome, completed=None, prefer="stored"):
    _scripts()
    from score_baselines import build_rows

    rows = build_rows([traces[p] for p in pids], outcome, completed, prefer)
    out = []
    for pid in pids:
        r = rows[pid]
        total = r["total_tokens"]
        final_ok = equivalent(r["final_answer"], r["gold"])
        out.append(Problem(
            pid,
            np.array([max(0.0, 1 - t / total) for t in r["boundary_tokens"]]),
            np.array([bool(k) and (equivalent(a, r["gold"]) == final_ok)
                      for a, k in zip(r["answers"], r["answer_known"])]),
            np.array(r["answer_known"], bool)))
    return out


def rule_scores(cfg, log, pids):
    _scripts()
    import aggregate_results as ag

    _, info, nla_rows = ag.nla_stopping_curve(cfg, log)
    nla = {r["problem_id"]: r["scores"] for r in nla_rows}
    base = read_json(cfg.dir / "baselines" / "boundary_scores.json")
    menu = {}
    for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl"):
        if r["window_kind"] == "gaussian_control" or not r.get("integrity_ok"):
            continue
        menu.setdefault(r["problem_id"], set()).add(int(r["chunk_index"]))
    scores = {"nla_readout": nla,
              "semantic_entropy": {p: base[p]["semantic_entropy"] for p in pids},
              "probe": {p: base[p]["probe"] for p in pids},
              "convergence": {p: base[p]["convergence"] for p in pids}}
    return scores, menu, sorted(info["problem_ids"])


def analyse(problems, scores, menu, rng, finished=None, traces=None, reps=REPS,
            mc_draws=MC_DRAWS):
    """Everything plan §5-§8 asks for, on one design and one problem set."""
    bstar = oracle_bound(problems, menu)
    grid = [f * bstar for f in FRACTIONS]
    rm = {r: rule_matrix(problems, scores[r], menu) for r in RULES}
    ES, EQ = no_signal_expectations(problems, menu, QGRID)
    curves = {r: rm[r].curve() for r in RULES}
    curves["no_signal"] = (ES.mean(0), EQ.mean(0))

    per_budget = []
    for f, b in zip(FRACTIONS, grid):
        row = {"fraction_of_oracle": f, "budget": b}
        for r in (*RULES, "no_signal"):
            k = match(curves[r][0], b, TOL)
            if k is None:
                row[r] = None
                continue
            entry = {"saving": float(curves[r][0][k]), "safe_rate": float(curves[r][1][k])}
            if r != "no_signal":
                entry["fire_rate"] = float(rm[r].fired[:, k].mean())
                entry["n_abstained"] = int((~rm[r].fired[:, k]).sum())
                entry["n_safe_stops"] = int((rm[r].fired[:, k] & rm[r].safe[:, k]).sum())
            row[r] = entry
        per_budget.append(row)

    pairs = {f"nla_readout - {c}": ("nla_readout", c) for c in COMPARATORS}
    pairs.update({f"{r} - no_signal": (r, "no_signal") for r in RULES})
    point = {}
    for name, (a, b) in pairs.items():
        g, n, diffs = gap_over_budgets(curves[a], curves[b], grid, TOL)
        point[name] = {"gap": g, "n_budgets_matched": n, "per_budget": diffs}

    # paired problem bootstrap, every curve rebuilt and re-matched
    n = len(problems)
    boot = {name: np.full(reps, np.nan) for name in pairs}
    for i in range(reps):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        c = {r: rm[r].curve(w) for r in RULES}
        ww = w / w.sum()
        c["no_signal"] = (ww @ ES, ww @ EQ)
        for name, (a, b) in pairs.items():
            g, k, _ = gap_over_budgets(c[a], c[b], grid, TOL)
            if k >= MIN_BUDGETS:
                boot[name][i] = g
    for name in pairs:
        r = boot[name]
        ok = r[np.isfinite(r)]
        evaluable = point[name]["n_budgets_matched"] >= MIN_BUDGETS
        point[name].update({
            "evaluable": evaluable,
            "ci95": ([float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
                     if evaluable and len(ok) else None),
            "p_bootstrap": bootstrap_p(r, 1 / reps) if evaluable else float("nan"),
            "n_replicates_evaluable": int(len(ok)),
        })
    f1 = holm({k: point[k]["p_bootstrap"] for k in pairs if k.startswith("nla_readout - ")
               and k != "nla_readout - no_signal"})
    f2 = holm({k: point[k]["p_bootstrap"] for k in pairs if k.endswith(" - no_signal")})

    # no-signal: the planned Monte Carlo beside the closed form (deviation 1)
    mc = no_signal_monte_carlo(problems, menu, grid, TOL, mc_draws, rng)
    closed = [row["no_signal"]["safe_rate"] if row["no_signal"] else None for row in per_budget]
    mc_check = {"monte_carlo_draws": mc_draws, "monte_carlo": mc, "closed_form": closed,
                "max_abs_difference": max(abs(a - b) for a, b in zip(mc, closed)
                                          if a is not None and b is not None)}

    # secondary: McNemar and decomposition at 0.5 B*
    mid = grid[FRACTIONS.index(0.5)]
    secondary = {"budget": mid, "mcnemar": {}, "cut_off_share_of_stops": {}}
    cols = {r: match(curves[r][0], mid, TOL) for r in RULES}
    for c in COMPARATORS:
        ka, kb = cols["nla_readout"], cols[c]
        if ka is None or kb is None:
            secondary["mcnemar"][c] = None
            continue
        t = paired_binary_test(f"nla_readout vs {c}", rm["nla_readout"].safe[:, ka].tolist(),
                               rm[c].safe[:, kb].tolist(), 0.95)
        secondary["mcnemar"][c] = {
            "p": t.p_value, "n_discordant": t.notes.get("n_discordant"),
            "nla_safe_other_not": t.notes.get("a_safe_b_not"),
            "other_safe_nla_not": t.notes.get("b_safe_a_not"),
            "rate_difference_ci95": ([t.interval.low, t.interval.high] if t.interval else None)}
    if finished is not None:
        for r in RULES:
            k = cols[r]
            if k is None:
                continue
            fb = [(p.problem_id, int(b)) for p, b in zip(problems, rm[r].boundary[:, k]) if b >= 0]
            cut = 0
            for pid, b in fb:
                stored = next(x for x in traces[pid]["boundaries"] if x["index"] == b)
                fin = stored.get("forced_finished") if stored.get("evaluated") else finished.get((pid, b))
                cut += fin is False
            secondary["cut_off_share_of_stops"][r] = cut / len(fb) if fb else None

    return {"n_problems": n, "oracle_bound": bstar, "budgets": grid,
            "per_budget": per_budget, "gaps": point, "holm_F1": f1, "holm_F2": f2,
            "no_signal_check": mc_check, "secondary": secondary}


def verdict(design: dict) -> dict:
    """Plan §7, applied mechanically."""
    g, f1, f2 = design["gaps"], design["holm_F1"], design["holm_F2"]
    beats_chance = {r: bool(f2[f"{r} - no_signal"]["reject"]) and
                    g[f"{r} - no_signal"]["gap"] > 0 for r in RULES}
    informative = any(beats_chance.values())
    per = {}
    for c in COMPARATORS:
        key = f"nla_readout - {c}"
        e = g[key]
        if not e["evaluable"]:
            per[c] = "not evaluable"
            continue
        lo, hi = e["ci95"]
        established = (e["gap"] > 0 and bool(f1[key]["reject"])
                       and beats_chance["nla_readout"])
        per[c] = ("established" if established else
                  "equivalent within 0.05" if lo > -MARGIN and hi < MARGIN else
                  "advantage of 0.05 excluded" if hi < MARGIN else "unresolved")
    evaluable = [c for c in COMPARATORS if per[c] != "not evaluable"]
    if not informative:
        overall = "uninformative: no rule beats the no-signal rule"
    elif evaluable and all(per[c] == "established" for c in evaluable):
        overall = "established"
    elif any(per[c] in ("advantage of 0.05 excluded", "equivalent within 0.05")
             and beats_chance[c] for c in evaluable):
        overall = "refuted as a practically relevant advantage"
    else:
        overall = "unresolved"
    return {"rule_beats_no_signal": beats_chance, "informative": informative,
            "per_comparator": per, "overall": overall}


def confirmatory_verdict(design: dict, margin: float = MARGIN) -> dict:
    """D75's pre-registered hypotheses, decided mechanically.

    H1: readout − probe (layer 20, the registered probe) > 0.
    H2: readout − no-signal > 0.
    Both are two-sided bootstrap tests, Holm-adjusted over {H1, H2} at 0.05.
    For H1, an upper 95% bound below ``margin`` excludes a practically
    relevant advantage, and an interval inside ±``margin`` is equivalence.
    Anything else is inconclusive. Nothing short of the interval is called
    equivalence.
    """
    g = design["gaps"]
    h = {"H1_readout_minus_probe": g["nla_readout - probe"],
         "H2_readout_minus_no_signal": g["nla_readout - no_signal"]}
    adj = holm({k: v["p_bootstrap"] for k, v in h.items()})
    out = {}
    for k, v in h.items():
        ci = v["ci95"]
        if not v["evaluable"] or ci is None:
            reading = "not evaluable"
        elif adj[k]["reject"] and v["gap"] > 0:
            reading = "supported"
        elif adj[k]["reject"] and v["gap"] < 0:
            reading = "reversed: the comparator is better"
        elif ci[0] > -margin and ci[1] < margin:
            reading = f"equivalent within ±{margin}"
        elif ci[1] < margin:
            reading = f"advantage of {margin} or more excluded"
        else:
            reading = "inconclusive"
        out[k] = {"gap": v["gap"], "ci95": ci, "p": v["p_bootstrap"],
                  "p_holm": adj[k]["p_holm"], "n_budgets": v["n_budgets_matched"],
                  "reading": reading}
    return out


def reproduce_historical(cfg, log, regist):
    _scripts()
    import aggregate_results as ag

    budgets = regist["o3_comparison_basis"]["budgets"]
    nla_curve, info, nla_rows = ag.nla_stopping_curve(cfg, log)
    base_curves, base_rows = ag.baseline_curves_on(cfg, info["problem_ids"], log)
    stored = {t["name"]: t for t in regist["tests"]["tests"]}
    out = {}
    for other in ("semantic_entropy", "probe"):
        name = f"O3: safe-stopping rate, nla_readout minus {other} at matched budget"
        t, unm = ag.o3_head_to_head(other, nla_curve, base_curves[other], nla_rows,
                                    base_rows.get(other), budgets, "nla_problem_set",
                                    {}, 0.95, log)
        if unm or abs(t.p_value - stored[name]["p_value"]) > 1e-12:
            raise SystemExit(f"{name}: the historical result did not reproduce")
        out[other] = {"p_value": t.p_value, "safe_rate_nla": t.notes["safe_rate_nla"],
                      f"safe_rate_{other}": t.notes[f"safe_rate_{other}"]}
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", default=None,
                   help="completion output (default: runs/main_forced)")
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--confirmatory", default=None, metavar="PLAN",
                   help="D75 replication: design A only, no historical analysis to "
                        "reproduce; refuses unless PLAN hashes to --plan-sha256")
    p.add_argument("--plan-sha256", default=None)
    args = p.parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_o3_registered" / "o3_registered.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("o3_registered")
    rng = np.random.default_rng(SEED)
    forced_dir = Path(args.forced_dir) if args.forced_dir else cfg.dir.parent / "main_forced"

    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    prov, completed, regenerated, finished, agree, n_rep = load_forced(
        forced_dir, sha256(traces_path))
    conf = args.confirmatory is not None
    if conf:
        plan_sha = sha256(Path(args.confirmatory))
        if plan_sha != args.plan_sha256:
            raise SystemExit(f"{args.confirmatory} hashes to {plan_sha}, not the frozen "
                             f"{args.plan_sha256}; refusing to analyse")
    results_path = cfg.dir / "analysis" / "results.json"
    out = {
        "status": ("confirmatory (D75); plan frozen before the outcomes" if conf else
                   "corrected, post hoc (D69); the registered O3 test is unchanged"),
        "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(), "seed": SEED,
        "reps": args.reps, "tolerance": TOL, "margin": MARGIN,
        "plan_sha256": plan_sha if conf else sha256(PLAN),
        "inputs_sha256": {
            "traces": sha256(traces_path),
            "boundary_scores": sha256(cfg.dir / "baselines" / "boundary_scores.json"),
            "reconstructions": sha256(cfg.dir / "nla" / "reconstructions.jsonl"),
            "results": sha256(results_path) if results_path.exists() else None,
            "forced": sha256(forced_dir / "forced.jsonl"),
        },
        "instrument": {"n_completed_boundaries": len(completed),
                       "n_reproducibility_checks": n_rep,
                       "reproducibility_agreement": agree,
                       "completion_provenance": prov},
    }
    if agree is None or agree < MIN_AGREEMENT:
        out["blocked"] = (f"the forcing instrument reproduced on {agree} of re-forced "
                          f"stored boundaries, below {MIN_AGREEMENT} (plan §2)")
        write_json(path, out)
        print(out["blocked"])
        return 2

    if not conf:
        regist = read_json(results_path)
        out["historical_parsed_outcome_reproduced"] = reproduce_historical(cfg, log, regist)

    scores, menu_sets, pids = rule_scores(cfg, log, list(traces))
    menu = {pid: np.array([i in menu_sets.get(pid, set()) for i in range(len(traces[pid]["chunks"]))])
            for pid in pids}
    forced = problems_for(traces, pids, OUTCOME_FORCED, completed)
    parsed = problems_for(traces, pids, OUTCOME_PARSED)

    # the readout's curve is unchanged by its own menu (plan §4)
    a = rule_matrix(forced, scores["nla_readout"], menu).curve()
    b = rule_matrix(forced, scores["nla_readout"], None).curve()
    if not (np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])):
        raise SystemExit("restricting the readout to its own menu changed its curve")

    out["oracle_bound"] = {"registered_outcome_menu": oracle_bound(forced, menu),
                           "parsed_outcome_all_boundaries": oracle_bound(parsed)}
    out["design_A_shared_menu"] = analyse(forced, scores, menu, rng, finished, traces, args.reps)
    out["design_A_shared_menu"]["verdict"] = verdict(out["design_A_shared_menu"])
    if conf:
        # Design A only: outcomes were measured on the menu alone (D75).
        out["confirmatory"] = confirmatory_verdict(out["design_A_shared_menu"])
        from coverage_robustness import probe_at_layer
        out["secondary_probe_layers"] = {}
        for layer in (14, 24):
            ps, _ = probe_at_layer(cfg, layer, traces, log)
            sc = {**scores, "probe": {p: ps[p] for p in pids}}
            s = analyse(forced, sc, menu, np.random.default_rng(SEED), reps=args.reps, mc_draws=50)
            out["secondary_probe_layers"][layer] = {
                k: {"gap": v["gap"], "ci95": v["ci95"], "p": v["p_bootstrap"]}
                for k, v in s["gaps"].items() if "semantic" not in k}
    else:
        out["oracle_bound"]["registered_outcome_all_boundaries"] = oracle_bound(forced)
        out["design_B_all_boundaries"] = analyse(forced, scores, None, rng, finished, traces,
                                                 args.reps)
        out["design_B_all_boundaries"]["note"] = (
            "secondary; the readout's menu straddles the tail start, so its comparisons "
            "here carry the AST's information (plan §1.3) and are not used for the verdict")

    # sensitivity: one generation pass wherever a regenerated answer exists
    sens = problems_for(traces, pids, OUTCOME_FORCED, {**completed, **regenerated},
                        prefer="completed")
    s = analyse(sens, scores, menu, np.random.default_rng(SEED), None, None, reps=200,
                mc_draws=50)
    out["sensitivity_regenerated_preferred"] = {
        k: {"gap": v["gap"], "ci95": v["ci95"]} for k, v in s["gaps"].items()}

    # strata (exploratory, descriptive): by dataset and MATH subject
    strata: dict[str, list[str]] = {}
    for pid in pids:
        key = ("gsm8k" if pid.startswith("gsm8k") else
               "math_algebra" if "algebra" in pid else "math_counting_and_probability")
        strata.setdefault(key, []).append(pid)
    out["strata"] = {}
    for key, ps in strata.items():
        sub = [pr for pr in forced if pr.problem_id in set(ps)]
        st = analyse(sub, scores, {p: menu[p] for p in ps}, np.random.default_rng(SEED),
                     None, None, reps=500, mc_draws=50)
        out["strata"][key] = {"n_problems": len(sub), "oracle_bound": st["oracle_bound"],
                              "gaps": {k: {"gap": v["gap"], "ci95": v["ci95"],
                                           "n_budgets_matched": v["n_budgets_matched"]}
                                       for k, v in st["gaps"].items()}}

    write_json(path, out)
    print(json.dumps(out, indent=1, default=str)[:20000])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
