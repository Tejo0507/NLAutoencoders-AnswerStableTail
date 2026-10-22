"""What changes when a cut-off forced answer may finish (D82). CPU only.

Implements ``docs/EXTENDED_FORCING_PLAN.md`` §3–§5 from the output of
``force_extended.py``. Every outcome is scored on design A's shared menu, with
the D69/D79 machinery and seed. The registered result is reproduced first, and
nothing here replaces it.

    python scripts/extended_forcing_analysis.py --config main --run-id main_nla240 \\
        --forced-dir runs/main_forced
    python scripts/extended_forcing_analysis.py --config replication \\
        --forced-dir runs/rep1_forced --confirmatory

Writes ``runs/<run_id>_extforced/analysis.json`` and refuses to replace it
(``--force``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.o3_registered import Problem
from nlaast.analysis.position import (
    majority_answer,
    position_scores,
    residualise_on_position,
    safe_vector,
)
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

PLAN = Path(__file__).resolve().parents[1] / "docs" / "EXTENDED_FORCING_PLAN.md"
STABILITY_GATE = 0.95
TOKEN_TOL = 2
KEY_COMPARISONS = ("nla_readout - probe", "nla_readout - no_signal", "probe - no_signal",
                   "readout - position_tokens_falling", "position_tokens_falling - no_signal",
                   "readout_residualised - no_signal", "readout - position_tokens")


def net_saving(prefix_tokens: int, forced_tokens: int, total_tokens: int) -> float:
    """Plan §3: a stop must emit its answer, so the forced tokens are charged."""
    return max(0.0, 1 - (prefix_tokens + forced_tokens) / total_tokens)


def cut_off_reading(f: float | None) -> str | None:
    """Plan §5, fixed before any extended answer existed."""
    if f is None:
        return None
    if f >= 0.80:
        return "the 48-token outcome mostly measured commitment within 48 tokens"
    if f <= 0.20:
        return "the registered reading is supported"
    return "mixed"


def agreement_by_position(ext, tail_start, cont, finals, equivalent) -> dict:
    """Plan §4.5: 256-token answers against the final answer and the
    continuations' majority, split by position relative to ``i*``. Agreement
    is recomputed from the answers, never read from a stored flag."""
    out = {}
    for name, keep in (("before", lambda i, t: i < t), ("at or after", lambda i, t: i >= t)):
        rows = [(k, e) for k, e in ext.items()
                if tail_start.get(k[0]) is not None and keep(k[1], tail_start[k[0]])]
        wc = [(k, e) for k, e in rows if k in cont]
        out[name] = {
            "n": len(rows),
            "equals_final": (float(np.mean([equivalent(e["answer"], finals[k[0]]) for k, e in rows]))
                             if rows else None),
            "still_cut_off": float(np.mean([not e["finished"] for _, e in rows])) if rows else None,
            "n_with_continuations": len(wc),
            "equals_continuation_majority": (float(np.mean([equivalent(e["answer"], cont[k])
                                                            for k, e in wc])) if wc else None),
        }
    return out


def tokenizer(cfg):
    from transformers import AutoTokenizer

    from nlaast import paths
    return AutoTokenizer.from_pretrained(
        str(paths.quantised_models() / cfg.target.repo_id.replace("/", "__")))


def outcome_answers(row, menu_mask, ext, reg48, cont, outcome, pid):
    """Per-boundary answers and known flags under one outcome (plan §3)."""
    answers = list(row["answers"])
    known = list(row["answer_known"])
    for i in range(len(answers)):
        if not menu_mask[i]:
            continue
        if outcome in ("O256", "O256-u"):
            e = ext.get((pid, i))
            if e is not None:
                answers[i] = None if (outcome == "O256-u" and not e["finished"]) else e["answer"]
        elif outcome == "Ocont":
            c = cont.get((pid, i))
            if c is None:
                known[i] = False
            else:
                answers[i] = c
    return answers, known


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", required=True)
    p.add_argument("--confirmatory", action="store_true",
                   help="rep1: also apply D75's readings in each cell (sensitivity only)")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--ext-dir", default=None, help="default: runs/<run_id>_extforced")
    args = p.parse_args()
    cfg = load_config(args)
    ext_dir = Path(args.ext_dir) if args.ext_dir else cfg.dir.parent / f"{cfg.run_id}_extforced"
    out_path = ext_dir / "analysis.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import position_truncation_audit as pta
    import reanalyse_o3_registered as ro
    from force_extended import generated_part, registered_completions
    from score_baselines import build_rows

    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("extended")
    forced_dir = Path(args.forced_dir)
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    stored = read_json(cfg.dir.parent / f"{cfg.run_id}_o3_registered" / "o3_registered.json")
    reps = args.reps or int(stored["reps"])
    for k, path in {"traces": traces_path, "forced": forced_dir / "forced.jsonl",
                    "reconstructions": cfg.dir / "nla" / "reconstructions.jsonl",
                    "boundary_scores": cfg.dir / "baselines" / "boundary_scores.json"}.items():
        if stored["inputs_sha256"][k] != ro.sha256(path):
            raise SystemExit(f"{k} changed since the registered analysis")
    eprov = read_json(ext_dir / "provenance.json")
    if eprov["traces_sha256"] != ro.sha256(traces_path) or \
            eprov["registered_forced_sha256"] != ro.sha256(forced_dir / "forced.jsonl"):
        raise SystemExit(f"{ext_dir} was generated from other inputs")

    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    _, completed, _, finished, agree, _ = ro.load_forced(forced_dir, ro.sha256(traces_path))
    ext_rows = read_jsonl(ext_dir / "forced_ext.jsonl")
    ext = {(r["problem_id"], int(r["index"])): r for r in ext_rows if r["purpose"] == "cut_off"}
    chk = [r for r in ext_rows if r["purpose"] == "check"]

    out = {"status": ("sensitivity to D75, pre-specified (D82)" if args.confirmatory else
                      "exploratory (D82): the registered results were seen"),
           "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
           "plan_sha256": ro.sha256(PLAN), "seed": ro.SEED, "reps": reps,
           "ext_provenance": eprov}

    # ---- instrument checks (plan §4) ----
    stab = sum(r["agrees_with_registered"] for r in chk) / len(chk) if chk else None
    tok_diff = max(abs(r["n_tokens"] - r["registered_retokenised_tokens"]) for r in chk) if chk else None
    out["instrument"] = {
        "n_check": len(chk), "stability": stab, "stability_gate": STABILITY_GATE,
        "check_token_count_max_abs_diff": tok_diff,
        "n_cut_off_forced": len(ext),
        "cut_off_begins_with_registered": (np.mean([r["begins_with_registered"] for r in ext.values()])
                                           if ext else None),
        "still_cut_off_at_budget": np.mean([not r["finished"] for r in ext.values()]) if ext else None,
        "mean_tokens_cut_off": np.mean([r["n_tokens"] for r in ext.values()]) if ext else None,
    }
    if stab is None or stab < STABILITY_GATE:
        out["blocked"] = f"instrument unstable: {stab} < {STABILITY_GATE} (plan §4.1)"
        write_json(out_path, out)
        print(out["blocked"])
        return 2
    use_net = tok_diff is not None and tok_diff <= TOKEN_TOL

    # ---- shared design-A inputs, exactly as D79 ----
    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    menu = {pid: np.array([i in menu_sets.get(pid, set()) for i in range(len(traces[pid]["chunks"]))])
            for pid in pids}
    registered = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)
    rows = build_rows([traces[q] for q in pids], OUTCOME_FORCED, completed, "stored")
    prefix = {q: [int(c["prefix_tokens"]) for c in traces[q]["chunks"]] for q in pids}
    total = {q: int(traces[q]["n_tokens"]) for q in pids}
    sc = {r: scores[r] for r in pta.REGISTERED_RULES}
    for kind, name in (("tokens", "position_tokens"), ("tokens_falling", "position_tokens_falling"),
                       ("index", "position_index"), ("relative", "relative_position")):
        sc[name] = {q: position_scores(prefix[q], total[q], kind) for q in pids}
    sc["readout_residualised"] = residualise_on_position(
        {q: scores["nla_readout"][q] for q in pids}, prefix, menu)
    pairs = {**pta.registered_pairs(ro), **pta.FAMILY_X, **pta.FAMILY_Y, **pta.DESCRIPTIVE}

    # registered answers on the menu, and their token counts
    reg48 = registered_completions(traces, forced_dir)
    tok = tokenizer(cfg)
    suffix = cfg.generation.force_answer_suffix
    tok48 = {k: len(tok(generated_part(v["text"], suffix), add_special_tokens=False)["input_ids"])
             for k, v in reg48.items() if menu.get(k[0]) is not None and menu[k[0]][k[1]]}
    cont = {}
    for pid in pids:
        for b in traces[pid]["boundaries"]:
            ca = b.get("continuation_answers") or []
            if b.get("evaluated") and ca:
                m = majority_answer(ca, equivalent)
                if m is not None:
                    cont[(pid, int(b["index"]))] = m

    def problems(outcome, saving):
        probs = []
        for pr in registered:
            pid = pr.problem_id
            r = rows[pid]
            ans, known = outcome_answers(r, menu[pid], ext, reg48, cont, outcome, pid)
            safe = safe_vector(ans, known, r["gold"], r["final_answer"], equivalent)
            sav = pr.saving.copy()
            if saving == "net":
                for i in np.flatnonzero(menu[pid]):
                    e = ext.get((pid, i)) if outcome != "O48" else None
                    ft = e["n_tokens"] if e is not None else tok48[(pid, int(i))]
                    sav[i] = net_saving(prefix[pid][i], ft, total[pid])
            probs.append(Problem(pid, sav, safe, np.array(known, bool)))
        if outcome == "Ocont":
            probs = [q for q in probs if q.known[menu[q.problem_id]].all()]
        return probs

    # O48 with registered savings must be the registered analysis
    base = problems("O48", "registered")
    for a, b in zip(base, registered):
        if not (np.array_equal(a.safe, b.safe) and np.allclose(a.saving, b.saving)):
            raise SystemExit(f"{a.problem_id}: O48 does not reproduce the registered labels")
    bstar, _, _, _, gaps = pta.build_design(ro, base, sc, menu, reps, ro.SEED, pairs, ro.MIN_BUDGETS)
    bad = pta.check_reproduction(stored["design_A_shared_menu"]["gaps"], gaps)
    if bad:
        raise SystemExit(f"the registered design-A analysis did not reproduce: {bad[:3]}")
    out["reproduced_registered"] = True

    cells = {}
    savings = ("registered", "net") if use_net else ("registered",)
    for outcome in ("O48", "O256", "O256-u", "Ocont"):
        for saving in savings:
            probs = problems(outcome, saving)
            menu_c = {q.problem_id: menu[q.problem_id] for q in probs}
            sc_c = {k: {q.problem_id: v[q.problem_id] for q in probs} for k, v in sc.items()}
            b2, grid, _, _, gp = pta.build_design(ro, probs, sc_c, menu_c, reps, ro.SEED, pairs,
                                                  ro.MIN_BUDGETS)
            rd = pta.readings(ro, gp, args.confirmatory)
            cells[f"{outcome}/{saving}"] = {
                "n_problems": len(probs), "oracle_bound": b2,
                "n_menu_labels_changed_vs_O48": int(sum(
                    (q.safe != r0.safe)[menu[q.problem_id]].sum()
                    for q, r0 in zip(probs, [x for x in base if x.problem_id in menu_c]))),
                "gaps": {k: {"gap": gp[k]["gap"], "ci95": gp[k]["ci95"], "p": gp[k]["p_bootstrap"],
                             "n_budgets_matched": gp[k]["n_budgets_matched"]}
                         for k in KEY_COMPARISONS if k in gp},
                "d69_verdict": rd["d69_verdict"], "position_reading": rd["position_reading"],
                "falling_reading": rd["falling_reading"],
                **({"d75_sensitivity": rd["d75_confirmatory"]} if args.confirmatory else {}),
            }
    out["cells"] = cells
    out["net_savings_used"] = use_net

    # ---- what cut-off answers mean (plan §5) ----
    ts = {r["problem_id"]: r.get("tail_start") for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    finals = {q: traces[q].get("final_answer") for q in traces}
    by_pos = agreement_by_position(ext, ts, cont, finals, equivalent)
    f = by_pos["before"]["equals_final"]
    out["cut_off_meaning"] = {"f_256_equals_final": f, "reading": cut_off_reading(f),
                              "by_position": by_pos}
    # per-boundary labels, for the D81 information audit under O256 (plan §4)
    o256 = {q.problem_id: q for q in problems("O256", "registered")}
    with open(ext_dir / "labels_O256.jsonl", "w", encoding="utf-8") as fh:
        for pid in pids:
            for i in np.flatnonzero(menu[pid]):
                fh.write(json.dumps({"problem_id": pid, "chunk": int(i),
                                     "Y_O256": bool(o256[pid].safe[i])}) + "\n")
    out["labels_O256_sha256"] = ro.sha256(ext_dir / "labels_O256.jsonl")
    # ---- which readings depend on the budget ----
    ref = cells["O48/net" if use_net else "O48/registered"]
    tgt = cells["O256/net" if use_net else "O256/registered"]
    out["budget_dependence"] = {
        "compared": ["O48/net" if use_net else "O48/registered", "O256/net" if use_net else "O256/registered"],
        "d69_overall": [ref["d69_verdict"]["overall"], tgt["d69_verdict"]["overall"]],
        "d69_per_comparator": {c: [ref["d69_verdict"]["per_comparator"][c],
                                   tgt["d69_verdict"]["per_comparator"][c]] for c in ro.COMPARATORS},
        "position_overall": [ref["position_reading"]["overall"], tgt["position_reading"]["overall"]],
        "falling_overall": [ref["falling_reading"]["overall"], tgt["falling_reading"]["overall"]],
        **({"d75": {k: [ref["d75_sensitivity"][k]["reading"], tgt["d75_sensitivity"][k]["reading"]]
                    for k in ref["d75_sensitivity"]}} if args.confirmatory else {}),
    }
    write_json(out_path, out)
    print(json.dumps({"instrument": out["instrument"], "cut_off_meaning": out["cut_off_meaning"],
                      "budget_dependence": out["budget_dependence"],
                      "key_gaps": {c: {k: [round(v["gap"], 4), v["ci95"]] for k, v in cell["gaps"].items()}
                                   for c, cell in cells.items()}}, indent=1, default=str))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
