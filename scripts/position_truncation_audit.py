"""Position-only control and forced-answer truncation audit, design A (D79). CPU only.

Implements ``docs/POSITION_TRUNCATION_PLAN.md`` on a run that already has a
registered-outcome O3 analysis (``reanalyse_o3_registered.py``). Exploratory on
``main`` and ``main_nla240``; on ``rep1`` a secondary analysis that never enters
the D75 verdict. Nothing under ``runs/<run_id>/`` or the forced-completion
directory is written.

Before it reports anything new it reproduces, from the same seed, every stored
design-A comparison of ``runs/<run_id>_o3_registered/o3_registered.json`` (gap,
interval and p-value) and D69's cut-off share of stops at 0.5 x B*.

    python scripts/position_truncation_audit.py --config main
    python scripts/position_truncation_audit.py --config main --run-id main_nla240
    python scripts/position_truncation_audit.py --config replication \\
        --forced-dir runs/<rep1 forced dir> --confirmatory

Writes ``runs/<run_id>_position_audit/position_truncation.json`` and refuses to
replace it (``--force``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.o3_registered import (
    no_signal_expectations,
    oracle_bound,
    rule_matrix,
)
from nlaast.analysis.position import (
    falling_reading,
    majority_answer,
    paired_comparisons,
    position_reading,
    position_scores,
    rate,
    relabelled,
    residualise_on_position,
    safe_vector,
    scenario_answers,
    stop_cut_off_share,
    within_problem_spearman,
)
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

PLAN = Path(__file__).resolve().parents[1] / "docs" / "POSITION_TRUNCATION_PLAN.md"
REGISTERED_RULES = ("nla_readout", "semantic_entropy", "probe", "convergence")
NEW_RULES = ("position_tokens", "position_index", "relative_position", "readout_residualised")
FAMILY_X = {
    "readout - position_tokens": ("nla_readout", "position_tokens"),
    "position_tokens - no_signal": ("position_tokens", "no_signal"),
    "probe - position_tokens": ("probe", "position_tokens"),
    "readout_residualised - no_signal": ("readout_residualised", "no_signal"),
}
#: Plan deviation 3: post hoc on main and main_nla240, specified in advance for rep1.
FAMILY_Y = {
    "readout - position_tokens_falling": ("nla_readout", "position_tokens_falling"),
    "position_tokens_falling - no_signal": ("position_tokens_falling", "no_signal"),
}
DESCRIPTIVE = {
    "readout - relative_position": ("nla_readout", "relative_position"),
    "relative_position - no_signal": ("relative_position", "no_signal"),
    "position_index - no_signal": ("position_index", "no_signal"),
    "readout - position_index": ("nla_readout", "position_index"),
    "readout_residualised - position_tokens": ("readout_residualised", "position_tokens"),
    "readout_residualised - probe": ("readout_residualised", "probe"),
}
REPRO_TOL = 1e-9


def registered_pairs(ro) -> dict[str, tuple[str, str]]:
    pairs = {f"nla_readout - {c}": ("nla_readout", c) for c in ro.COMPARATORS}
    pairs.update({f"{r} - no_signal": (r, "no_signal") for r in ro.RULES})
    return pairs


def cut_off_map(traces, pids, finished) -> dict[tuple[str, int], bool | None]:
    """Was the forced completion *used as the outcome* at each boundary cut off?

    Mirrors ``forced_answer_sequence(prefer="stored")`` and D69's secondary:
    the stored flag where the scan evaluated the boundary, else the completion's.
    """
    out: dict[tuple[str, int], bool | None] = {}
    for pid in pids:
        for b in traces[pid]["boundaries"]:
            key = (pid, int(b["index"]))
            if b.get("evaluated"):
                fin = b.get("forced_finished")
            else:
                fin = finished.get(key)
            out[key] = None if fin is None else (not bool(fin))
    return out


def build_design(ro, problems, scores, menu, reps, seed, pairs, min_budgets):
    """B*, budgets, every rule's matrix, and all pairs on one shared bootstrap."""
    bstar = oracle_bound(problems, menu)
    grid = [f * bstar for f in ro.FRACTIONS]
    mats = {r: rule_matrix(problems, scores[r], menu) for r in scores}
    ES, EQ = no_signal_expectations(problems, menu, ro.QGRID)
    gaps = paired_comparisons(len(problems), mats, (ES, EQ), pairs, grid, ro.TOL, reps,
                              np.random.default_rng(seed), min_budgets)
    return bstar, grid, mats, (ES, EQ), gaps


def readings(ro, gaps, confirmatory):
    f1 = ro.holm({k: gaps[k]["p_bootstrap"] for k in gaps if k.startswith("nla_readout - ")
                  and k != "nla_readout - no_signal"})
    f2 = ro.holm({k: gaps[k]["p_bootstrap"] for k in gaps
                  if k.endswith(" - no_signal") and k.split(" - ")[0] in ro.RULES})
    design = {"gaps": gaps, "holm_F1": f1, "holm_F2": f2}
    hx = ro.holm({k: gaps[k]["p_bootstrap"] for k in FAMILY_X})
    hy = ro.holm({k: gaps[k]["p_bootstrap"] for k in FAMILY_Y})
    out = {"holm_F1": f1, "holm_F2": f2, "d69_verdict": ro.verdict(design),
           "holm_X": hx, "position_reading": position_reading(gaps, hx),
           "holm_Y": hy, "falling_reading": falling_reading(gaps, hy)}
    if confirmatory:
        out["d75_confirmatory"] = ro.confirmatory_verdict(design)
    return out


def check_reproduction(stored_gaps, gaps):
    bad = []
    for k, s in stored_gaps.items():
        g = gaps[k]
        for f in ("gap", "p_bootstrap"):
            a, b = s[f], g[f]
            if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= REPRO_TOL):
                bad.append((k, f, a, b))
        if (s["ci95"] is None) != (g["ci95"] is None) or (
                s["ci95"] is not None and max(abs(x - y) for x, y in zip(s["ci95"], g["ci95"]))
                > REPRO_TOL):
            bad.append((k, "ci95", s["ci95"], g["ci95"]))
    return bad


def descriptive_truncation(cfg, traces, pids, cut, menu_kinds, tail_start, regenerated_rows):
    cap = cfg.generation.max_new_tokens
    tr = [{"dataset": t.get("dataset"),
           "hit_cap": bool(t.get("truncated")) or int(t.get("n_tokens", 0)) >= cap,
           "final_unparsed": t.get("final_answer") is None} for t in traces.values()]
    arm = set(pids)
    bnd, cont, agree = [], [], []
    for pid in pids:
        t = traces[pid]
        ts = tail_start.get(pid)
        for b in t["boundaries"]:
            i = int(b["index"])
            c = cut.get((pid, i))
            kinds = menu_kinds.get((pid, i))
            rec = {"dataset": t.get("dataset"),
                   "source": "stored" if b.get("evaluated") else "completed",
                   "relative_to_tail_start": ("no tail" if ts is None else
                                              "before" if i < ts else "at or after"),
                   "window_kind": "+".join(sorted(kinds)) if kinds else "off menu",
                   "cut_off": c}
            bnd.append(rec)
            if not b.get("evaluated"):
                continue
            ca = list(b.get("continuation_answers") or [])
            for a in ca:
                cont.append({"dataset": t.get("dataset"), "unparsed": a is None})
            if ca:
                maj = majority_answer(ca, equivalent)
                fa, pa = b.get("forced_answer"), b.get("parsed_answer")
                status = "cut off" if c else "finished" if c is False else "unknown"
                agree.append({
                    "cut_off": status,
                    "cut_off_x_position": f"{status}, {rec['relative_to_tail_start']}",
                    "majority_equals_final": (None if maj is None
                                              else equivalent(maj, t.get("final_answer"))),
                    "dataset": t.get("dataset"),
                    "forced_unparsed": fa is None,
                    "forced_vs_majority_disagree": (None if maj is None
                                                    else not equivalent(fa, maj)),
                    "forced_vs_parsed_disagree": (None if pa is None
                                                  else not equivalent(fa, pa)),
                })
    menu_bnd = [r for r in bnd if r["window_kind"] != "off menu"]
    repro = [{"stored_cut_off": cut.get((r["problem_id"], int(r["index"]))),
              "disagrees": not bool(r["agrees_with_stored"])}
             for r in regenerated_rows if r["problem_id"] in arm]
    for r in repro:
        r["split"] = {True: "cut off", False: "finished", None: "unknown"}[r["stored_cut_off"]]
    return {
        "traces": {"n": len(tr), "token_cap": cap,
                   "hit_token_cap": rate(tr, "hit_cap", "dataset") | {"all": rate(tr, "hit_cap")["all"]},
                   "final_answer_unparsed": rate(tr, "final_unparsed", "dataset")
                   | {"all": rate(tr, "final_unparsed")["all"]}},
        "forced_cut_off_arm_boundaries": {
            "all": rate(bnd, "cut_off")["all"],
            "by_dataset": rate(bnd, "cut_off", "dataset"),
            "by_source": rate(bnd, "cut_off", "source"),
            "by_position": rate(bnd, "cut_off", "relative_to_tail_start")},
        "forced_cut_off_menu_boundaries": {
            "all": rate(menu_bnd, "cut_off")["all"],
            "by_window_kind": rate(menu_bnd, "cut_off", "window_kind"),
            "by_position": rate(menu_bnd, "cut_off", "relative_to_tail_start")},
        "continuations_unparsed": rate(cont, "unparsed", "dataset") | {"all": rate(cont, "unparsed")["all"]},
        "forced_unparsed_by_cut_off": rate(agree, "forced_unparsed", "cut_off"),
        "forced_vs_continuation_majority_disagree_by_cut_off":
            rate(agree, "forced_vs_majority_disagree", "cut_off"),
        "forced_vs_continuation_majority_disagree_by_cut_off_and_position":
            rate(agree, "forced_vs_majority_disagree", "cut_off_x_position"),
        "continuation_majority_equals_final_by_cut_off": rate(agree, "majority_equals_final", "cut_off"),
        "continuation_majority_equals_final_by_cut_off_and_position":
            rate(agree, "majority_equals_final", "cut_off_x_position"),
        "forced_vs_parsed_disagree_by_cut_off": rate(agree, "forced_vs_parsed_disagree", "cut_off"),
        "reforced_disagreement_by_stored_cut_off": rate(repro, "disagrees", "split"),
    }


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", default=None,
                   help="completion output (default: runs/main_forced)")
    p.add_argument("--reps", type=int, default=None,
                   help="bootstrap replicates (default: the stored analysis's)")
    p.add_argument("--confirmatory", action="store_true",
                   help="rep1: also apply the D75 readings in each truncation scenario")
    args = p.parse_args()
    cfg = load_config(args)
    out_path = cfg.dir.parent / f"{cfg.run_id}_position_audit" / "position_truncation.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import reanalyse_o3_registered as ro
    from score_baselines import build_rows

    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("position_audit")
    stored_path = cfg.dir.parent / f"{cfg.run_id}_o3_registered" / "o3_registered.json"
    if not stored_path.exists():
        print(f"{stored_path} is missing: run reanalyse_o3_registered.py first; the audit "
              f"must reproduce it before reporting anything (plan §A.2)", file=sys.stderr)
        return 2
    stored = read_json(stored_path)
    reps = args.reps or int(stored["reps"])
    if args.confirmatory and not str(stored.get("status", "")).startswith("confirmatory"):
        print("--confirmatory needs a confirmatory registered analysis", file=sys.stderr)
        return 2

    forced_dir = Path(args.forced_dir) if args.forced_dir else cfg.dir.parent / "main_forced"
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    prov, completed, _, finished, agree, n_rep = ro.load_forced(forced_dir, ro.sha256(traces_path))
    if agree is None or agree < ro.MIN_AGREEMENT:
        print(f"forcing instrument reproduced on {agree}; below {ro.MIN_AGREEMENT}", file=sys.stderr)
        return 2
    for k in ("traces", "forced", "reconstructions", "boundary_scores"):
        now = ro.sha256({"traces": traces_path, "forced": forced_dir / "forced.jsonl",
                         "reconstructions": cfg.dir / "nla" / "reconstructions.jsonl",
                         "boundary_scores": cfg.dir / "baselines" / "boundary_scores.json"}[k])
        if stored["inputs_sha256"][k] != now:
            print(f"{k} changed since the registered analysis ({now})", file=sys.stderr)
            return 2

    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    menu = {pid: np.array([i in menu_sets.get(pid, set()) for i in range(len(traces[pid]["chunks"]))])
            for pid in pids}
    registered = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)
    rows = build_rows([traces[q] for q in pids], OUTCOME_FORCED, completed, "stored")

    # The safety labels this script builds must be the registered ones.
    for pr in registered:
        r = rows[pr.problem_id]
        mine = safe_vector(r["answers"], r["answer_known"], r["gold"], r["final_answer"], equivalent)
        if not np.array_equal(mine, pr.safe):
            raise SystemExit(f"{pr.problem_id}: rebuilt safety labels differ from the registered ones")

    prefix = {q: [int(c["prefix_tokens"]) for c in traces[q]["chunks"]] for q in pids}
    total = {q: int(traces[q]["n_tokens"]) for q in pids}
    sc = {r: scores[r] for r in REGISTERED_RULES}
    sc["position_tokens"] = {q: position_scores(prefix[q], total[q], "tokens") for q in pids}
    sc["position_tokens_falling"] = {q: position_scores(prefix[q], total[q], "tokens_falling")
                                     for q in pids}
    sc["position_index"] = {q: position_scores(prefix[q], total[q], "index") for q in pids}
    sc["relative_position"] = {q: position_scores(prefix[q], total[q], "relative") for q in pids}
    sc["readout_residualised"] = residualise_on_position(
        {q: scores["nla_readout"][q] for q in pids}, prefix, menu)

    pairs = {**registered_pairs(ro), **FAMILY_X, **FAMILY_Y, **DESCRIPTIVE}
    cut = cut_off_map(traces, pids, finished)

    out = {
        "status": ("secondary to the D75 confirmatory analysis, specified before any rep1 "
                   "outcome (D79); not part of the verdict" if args.confirmatory else
                   "exploratory (D79): the readout/probe/no-signal results here were seen "
                   "before the plan; the position results were not"),
        "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
        "plan_sha256": ro.sha256(PLAN), "seed": ro.SEED, "reps": reps,
        "n_problems": len(pids), "registered_analysis": str(stored_path),
        "forced_dir": str(forced_dir), "instrument_agreement": agree,
    }

    # ---- registered outcome: reproduce, then Part A ----
    bstar, grid, mats, _, gaps = build_design(ro, registered, sc, menu, reps, ro.SEED, pairs,
                                              ro.MIN_BUDGETS)
    stored_A = stored["design_A_shared_menu"]
    bad = check_reproduction(stored_A["gaps"], gaps)
    if bad or abs(bstar - stored_A["oracle_bound"]) > REPRO_TOL:
        raise SystemExit(f"the stored design-A analysis did not reproduce: {bad[:5]}")
    mid = grid[ro.FRACTIONS.index(0.5)]
    cols = {r: [ro.match(m.curve()[0], b, ro.TOL) for b in grid] for r, m in mats.items()}
    shares = {r: stop_cut_off_share(mats[r], cols[r], registered, cut) for r in mats}
    d69_share = stored_A["secondary"].get("cut_off_share_of_stops", {})
    for r, v in d69_share.items():
        k = cols[r][ro.FRACTIONS.index(0.5)]
        fb = [(pr.problem_id, int(b)) for pr, b in zip(registered, mats[r].boundary[:, k]) if b >= 0]
        legacy = sum(cut.get(x) is True for x in fb) / len(fb) if fb else None
        if v is not None and (legacy is None or abs(legacy - v) > REPRO_TOL):
            raise SystemExit(f"D69's cut-off share for {r} did not reproduce ({legacy} vs {v})")
    out["reproduced"] = {"design_A_gaps": sorted(stored_A["gaps"]), "oracle_bound": bstar,
                         "d69_cut_off_share_at_half_bstar": d69_share}

    pos = {q: [float(t) for t in prefix[q]] for q in pids}
    out["registered_outcome"] = {
        "oracle_bound": bstar, "budgets": grid,
        "gaps": {k: gaps[k] for k in (*FAMILY_X, *FAMILY_Y, *DESCRIPTIVE, *registered_pairs(ro))},
        **readings(ro, gaps, args.confirmatory),
        # A rule that never stops is safe (as registered), so a safe rate is
        # abstentions plus safe stops; both are reported.
        "safe_rate_at_budget": {
            r: [None if k is None else {"saving": float(mats[r].curve()[0][k]),
                                        "safe_rate": float(mats[r].curve()[1][k]),
                                        "n_stops": int(mats[r].fired[:, k].sum()),
                                        "n_safe_stops": int((mats[r].fired[:, k]
                                                             & mats[r].safe[:, k]).sum())}
                for k in cols[r]] for r in mats},
        "spearman_with_prefix_tokens_over_menu": {
            "nla_readout": within_problem_spearman({q: scores["nla_readout"][q] for q in pids}, pos, menu),
            "probe": within_problem_spearman({q: scores["probe"][q] for q in pids}, pos, menu)},
        "cut_off_share_of_stops_by_budget": shares,
        "half_bstar_budget": mid,
    }

    # ---- Part B ----
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    kinds: dict[tuple[str, int], set[str]] = {}
    for r in recon:
        if r["window_kind"] == "gaussian_control" or not r.get("integrity_ok"):
            continue
        kinds.setdefault((r["problem_id"], int(r["chunk_index"])), set()).add(r["window_kind"])
    tail_start = {r["problem_id"]: r.get("tail_start") for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    regen_rows = [r for r in read_jsonl(forced_dir / "forced.jsonl") if r["purpose"] == "repro"]
    out["truncation_descriptive"] = descriptive_truncation(cfg, traces, pids, cut, kinds,
                                                           tail_start, regen_rows)

    out["truncation_scenarios"] = {}
    base_readings = readings(ro, gaps, args.confirmatory)
    for scen in ("uncommitted", "committed_to_final"):
        probs = []
        n_changed = 0
        for pr in registered:
            r = rows[pr.problem_id]
            co = [cut.get((pr.problem_id, i)) for i in range(pr.n)]
            ans = scenario_answers(r["answers"], co, r["final_answer"], scen)
            s = safe_vector(ans, r["answer_known"], r["gold"], r["final_answer"], equivalent)
            n_changed += int((s != pr.safe).sum())
            probs.append(relabelled(pr, s))
        # Scores do not depend on the outcome, so only the labels change.
        b2, g2, _, _, gp = build_design(ro, probs, sc, menu, reps, ro.SEED, pairs, ro.MIN_BUDGETS)
        rd = readings(ro, gp, args.confirmatory)
        flips = {
            "d69_overall": [base_readings["d69_verdict"]["overall"], rd["d69_verdict"]["overall"]],
            "d69_per_comparator": {c: [base_readings["d69_verdict"]["per_comparator"][c],
                                       rd["d69_verdict"]["per_comparator"][c]]
                                   for c in ro.COMPARATORS},
            "position_overall": [base_readings["position_reading"]["overall"],
                                 rd["position_reading"]["overall"]],
            "position_beats_no_signal": [base_readings["position_reading"]["position_beats_no_signal"],
                                         rd["position_reading"]["position_beats_no_signal"]],
            "falling_overall": [base_readings["falling_reading"]["overall"],
                                rd["falling_reading"]["overall"]],
        }
        if args.confirmatory:
            flips["d75"] = {k: [base_readings["d75_confirmatory"][k]["reading"],
                                rd["d75_confirmatory"][k]["reading"]]
                            for k in base_readings["d75_confirmatory"]}
        changed = []
        def walk(path, v):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    walk(f"{path}.{kk}", vv)
            elif v[0] != v[1]:
                changed.append(path.lstrip("."))
        walk("", flips)
        out["truncation_scenarios"][scen] = {
            "n_boundary_labels_changed": n_changed, "oracle_bound": b2, "budgets": g2,
            "gaps": {k: {"gap": v["gap"], "ci95": v["ci95"], "p": v["p_bootstrap"],
                         "n_budgets_matched": v["n_budgets_matched"]} for k, v in gp.items()},
            "readings": rd, "registered_vs_scenario": flips,
            "readings_that_change": changed,
        }

    write_json(out_path, out)
    summary = {k: {"gap": round(gaps[k]["gap"], 4), "ci95": gaps[k]["ci95"],
                   "p": gaps[k]["p_bootstrap"]} for k in (*FAMILY_X, *FAMILY_Y, *DESCRIPTIVE)}
    print(json.dumps({"position_reading": out["registered_outcome"]["position_reading"],
                      "holm_X": out["registered_outcome"]["holm_X"],
                      "falling_reading": out["registered_outcome"]["falling_reading"],
                      "holm_Y": out["registered_outcome"]["holm_Y"], "gaps": summary,
                      "scenario_changes": {s: v["readings_that_change"]
                                           for s, v in out["truncation_scenarios"].items()}},
                     indent=1, default=str))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
