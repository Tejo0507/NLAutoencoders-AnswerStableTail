"""Post-hoc claim-evidence audit of a completed run (D68). Read-only, CPU-only.

Re-derives the quantities the 2026-10-19 scientific audit relies on, from the
run's stored artifacts, so that every number in
``docs/SCIENTIFIC_AUDIT_2026-10-19.md`` can be regenerated and checked. Every
analysis here is **post hoc and exploratory**: it was chosen after the
registered results were seen, it changes no registered result, and it is not
a substitute for any registered test.

Sections, each in its own key of the output:

* ``o3`` - the registered head-to-heads reproduced from the run's own
  functions (the script stops if they do not reproduce), then where each
  rule's safe rate comes from (abstention vs safe stops), an information-free
  reference rule at the same budgets, a shuffled-score null for the readout,
  how much any rule could save while staying safe, the implemented versus the
  registered safety outcome at the tail start, and what each signal costs.
* ``rq1`` - the tail-fraction/correctness association with trace length and
  dataset as covariates, and what the tail start marks (statement versus
  forced computation).
* ``rq2`` - the registered claim-level test re-expressed at the problem level.
* ``f10`` - the registered sample-level contrasts re-expressed as paired
  problem-level contrasts.
* ``family`` - the Benjamini-Hochberg family recomputed with the problem-level
  versions substituted, under both conventions D38 reports.
* ``rq3`` - the readout's base rate in the model's prose, the size of the
  candidate edit against the random-direction control, and output length by
  dose.

    python scripts/audit_claims_posthoc.py --config main

Writes ``runs/<run_id>_claims_audit/claims_audit.json`` and refuses to
replace it (``--force`` to override), because the audit document is read from
that file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats

from _stage import base_parser, load_config

from nlaast.analysis.posthoc import (
    cluster_bootstrap_mean,
    decompose_decisions,
    oracle_safe_saving,
    paired_problem_means,
    random_first_boundary_null,
)
from nlaast.analysis.stats import TestResult, benjamini_hochberg, bh_at_family_size
from nlaast.baselines.stopping import (
    at_matched_budget,
    complete_threshold_grid,
    decisions_at,
    sweep_threshold,
)
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS_NULL = 20000
REPS_SHUFFLE = 1000
REPS_BOOT = 10000

#: Inputs whose hashes are recorded, relative to the run directory.
INPUTS = (
    "analysis/results.json",
    "ast/ast.jsonl",
    "traces/traces.jsonl",
    "baselines/boundary_scores.json",
    "nla/reconstructions.jsonl",
    "nla/verbalisations.jsonl",
    "faithfulness/audits.jsonl",
    "causal/interventions.jsonl",
    "causal/summary.json",
)

#: A broader verification lexicon than RQ3's twenty markers, used only to ask
#: whether the construct is rare or merely mis-measured. It is not a
#: replacement readout and nothing is tested with it (D35, D41).
BROAD_VERIFICATION = re.compile(
    r"\b(verif\w*|check\w*|confirm\w*|double[- ]check|substitut\w* (?:\w+ )?back"
    r"|plug\w* (?:\w+ )?(?:back|in))\b")
#: Arithmetic in a forced completion: the model computing rather than stating.
ARITHMETIC = re.compile(r"(=|\\times|\\cdot|\\frac|[0-9]\s*[\+\-\*/×]\s*[0-9])")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scripts():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


# ------------------------------------------------------------------- O3


def compared_points(nla_curve, other_curve, budgets):
    """The operating-point pair ``o3_head_to_head`` tests, by the same rule."""
    pts = [(b, at_matched_budget(nla_curve, b), at_matched_budget(other_curve, b))
           for b in budgets]
    pts = [p for p in pts if p[1] and p[2]]
    if not pts:
        return None
    _, a, c = min(pts, key=lambda r: abs(r[1].mean_tokens_saved - r[2].mean_tokens_saved))
    return a, c


def shuffled_readout_null(nla_rows, target, reps, rng):
    """The readout's safe rate with its scores drawn at random, positions kept.

    The readout can fire only at the windows the verbaliser was run on, and
    those positions are chosen from the tail start. Replacing each window's
    score with a random draw from the pooled scores keeps where the rule *can*
    fire and destroys what the score says, so the gap between this null and
    the readout is what the reconstruction score contributes beyond its
    window positions.
    """
    pool = np.array([s for r in nla_rows for s in r["scores"] if s is not None])
    rates = []
    for _ in range(reps):
        new = []
        for r in nla_rows:
            out, last, cur = [], object(), None
            for s in r["scores"]:
                if s is None:
                    out.append(None)
                    continue
                if s != last:
                    cur, last = float(rng.choice(pool)), s
                out.append(cur)
            new.append({**r, "scores": out})
        vals = [s for r in new for s in r["scores"] if s is not None]
        pt = at_matched_budget(sweep_threshold(new, complete_threshold_grid(vals), True),
                               target)
        if pt is not None:
            rates.append(pt.safe_rate)
    rates = np.array(rates)
    return {"reps": reps, "n_matched": int(len(rates)),
            "mean_safe_rate": float(rates.mean()) if len(rates) else float("nan"),
            "interval": ([float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]
                         if len(rates) else None)}


def o3_section(cfg, log, rng):
    _scripts()
    import aggregate_results as ag

    results = read_json(cfg.dir / "analysis" / "results.json")
    registered = {t["name"]: t for t in results["tests"]["tests"]}
    budgets = results["o3_comparison_basis"]["budgets"]
    nla_curve, info, nla_rows = ag.nla_stopping_curve(cfg, log)
    base_curves, base_rows = ag.baseline_curves_on(cfg, info["problem_ids"], log)
    ast = {r["problem_id"]: r for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")}
    tail = {p: a.get("tail_start") for p, a in ast.items()}
    out: dict = {"n_problems": len(nla_rows), "window": [budgets[0], budgets[-1]],
                 "comparisons": {}}

    rows_by_pid = {r["problem_id"]: r for r in base_rows["probe"]}

    def safe_parsed(row, i):
        return (equivalent(row["answers"][i], row["gold"])
                == equivalent(row["final_answer"], row["gold"]))

    def safe_forced(row, i):
        t = traces[row["problem_id"]]
        b = next((x for x in t.get("boundaries", []) if x["index"] == i), None)
        if b is None or b.get("forced_answer") is None and not b.get("evaluated"):
            return None
        return (equivalent(b.get("forced_answer"), t["gold"])
                == equivalent(t["final_answer"], t["gold"]))

    pids = sorted(rows_by_pid)
    first_saving = [max(0.0, 1 - rows_by_pid[p]["boundary_tokens"][0]
                        / rows_by_pid[p]["total_tokens"]) for p in pids]
    first_safe = [safe_parsed(rows_by_pid[p], 0) for p in pids]

    for other in ("semantic_entropy", "probe"):
        name = f"O3: safe-stopping rate, nla_readout minus {other} at matched budget"
        t, unmatched = ag.o3_head_to_head(other, nla_curve, base_curves[other], nla_rows,
                                          base_rows.get(other), budgets, "nla_problem_set",
                                          {}, 0.95, log)
        reg = registered[name]
        if unmatched or abs(t.p_value - reg["p_value"]) > 1e-12:
            raise SystemExit(f"{name}: did not reproduce the registered result "
                             f"(p {t.p_value} vs {reg['p_value']}); refusing to report")
        a, c = compared_points(nla_curve, base_curves[other], budgets)
        nla_dec = decisions_at(nla_rows, a.threshold, True)
        oth_dec = decisions_at(base_rows[other], c.threshold, True)
        target = (a.mean_tokens_saved + c.mean_tokens_saved) / 2
        out["comparisons"][other] = {
            "reproduced": {"p_value": t.p_value, "safe_rate_nla": t.notes["safe_rate_nla"],
                           f"safe_rate_{other}": t.notes[f"safe_rate_{other}"],
                           "n_discordant": t.notes["n_discordant"]},
            "budget": target,
            "nla_readout": decompose_decisions(nla_dec, tail),
            other: decompose_decisions(oth_dec, tail),
            "information_free_rule": random_first_boundary_null(
                first_saving, first_safe, target, REPS_NULL, rng),
            "readout_scores_shuffled": shuffled_readout_null(nla_rows, target, REPS_SHUFFLE, rng),
        }

    out["oracle"] = {
        "implemented_outcome": oracle_safe_saving(base_rows["probe"], safe_parsed),
        "registered_outcome_lower_bound": oracle_safe_saving(base_rows["probe"], safe_forced),
        "note": ("mean over problems of the largest saving at which stopping is safe; "
                 "any matched budget above it forces unsafe stops on every rule. The "
                 "registered outcome is known only at boundaries the AST scan evaluated, "
                 "so its value is a lower bound"),
    }

    at_tail = {"n": 0, "safe_implemented": 0, "safe_registered": 0, "saving": []}
    for p in pids:
        ts = tail.get(p)
        if ts is None:
            continue
        row = rows_by_pid[p]
        at_tail["n"] += 1
        at_tail["safe_implemented"] += bool(safe_parsed(row, ts))
        at_tail["safe_registered"] += bool(safe_forced(row, ts))
        at_tail["saving"].append(max(0.0, 1 - row["boundary_tokens"][ts] / row["total_tokens"]))
    n = at_tail["n"]
    out["stop_at_tail_start"] = {
        "n": n,
        "safe_rate_implemented_outcome": at_tail["safe_implemented"] / n,
        "safe_rate_registered_outcome": at_tail["safe_registered"] / n,
        "mean_saving": float(np.mean(at_tail["saving"])),
        "note": ("implemented: the answer parsed from the prefix text "
                 "(score_baselines.build_rows); registered: the answer forced from the "
                 "prefix (PROJECT_PLAN section 5). Criterion 1 makes the registered "
                 "rate 1.0 by construction"),
    }

    tr = [t for t in traces.values() if t.get("ok")]
    tokens = np.array([t["n_tokens"] for t in tr], dtype=float)
    se = np.array([sum(s.get("n_tokens", 0) for s in t.get("entropy_samples", []))
                   for t in tr], dtype=float)
    verb = read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")
    vtok: dict[str, int] = {}
    for v in verb:
        n_tok = sum(int(s.get("n_tokens", 0)) for s in v.get("samples", []))
        n_tok += sum(int(s.get("n_tokens", 0)) for s in v.get("noise_samples", []))
        vtok[v["problem_id"]] = vtok.get(v["problem_id"], 0) + n_tok
    tr_by = {t["problem_id"]: t["n_tokens"] for t in tr}
    ratio_v = [vtok[p] / tr_by[p] for p in vtok if tr_by.get(p)]
    out["signal_cost"] = {
        "mean_trace_tokens": float(tokens.mean()),
        "semantic_entropy_sampled_tokens_per_problem": float(se.mean()),
        "semantic_entropy_tokens_over_trace_tokens": float(np.mean(se / tokens)),
        "verbaliser_generated_tokens_per_problem": float(np.mean(list(vtok.values()))),
        "verbaliser_tokens_over_trace_tokens": float(np.mean(ratio_v)),
        "verbaliser_note": ("7B verbaliser tokens, including the Gaussian control; "
                            "reconstructor forward passes are additional"),
        "note": ("the matched budget counts tokens of the trace not generated; the "
                 "tokens a signal costs to compute are outside it"),
    }
    return out


# ------------------------------------------------------------------- RQ1


def rq1_section(cfg):
    import statsmodels.api as sm

    _scripts()
    from detect_answer_stable_tail import answer_stated_at

    ast = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")}
    X, y = [], []
    for a in ast:
        if a.get("tail_start") is None:
            continue
        t = traces[a["problem_id"]]
        X.append([a["tail_fraction"], np.log(a["n_chunks"]), np.log(t["n_tokens"]),
                  float(t["dataset"] == "math")])
        y.append(int(bool(a["final_correct"])))
    X, y = np.array(X), np.array(y)
    models = {}
    for label, cols in (("tail_fraction", [0]),
                        ("tail_fraction + log_chunks + dataset", [0, 1, 3]),
                        ("tail_fraction + log_tokens + dataset", [0, 2, 3])):
        m = sm.Logit(y, sm.add_constant(X[:, cols])).fit(disp=0)
        lo, hi = m.conf_int()[1]
        models[label] = {"coef_tail_fraction": float(m.params[1]),
                         "p_value": float(m.pvalues[1]), "ci95": [float(lo), float(hi)]}
    rho = stats.spearmanr(X[:, 0], X[:, 2])

    n = before = before_computing = computing = 0
    post_share = []
    for a in ast:
        ts = a.get("tail_start")
        t = traces[a["problem_id"]]
        stated = answer_stated_at(t)["answer_stated_at_permissive"]
        if stated is not None and t.get("chunks"):
            pre = t["chunks"][stated]["prefix_tokens"]
            post_share.append(max(0, t["n_tokens"] - pre) / t["n_tokens"])
        if ts is None or a["status"] != "ok":
            continue
        b = next(x for x in t["boundaries"] if x["index"] == ts)
        c = bool(ARITHMETIC.search(b.get("forced_text") or ""))
        n += 1
        computing += c
        if stated is not None and ts < stated:
            before += 1
            before_computing += c
    return {
        "n": int(len(y)), "n_incorrect": int((1 - y).sum()),
        "logistic_correct_on": models,
        "spearman_tail_fraction_vs_log_tokens": [float(rho.statistic), float(rho.pvalue)],
        "median_tokens_correct": float(np.median(np.exp(X[y == 1, 2]))),
        "median_tokens_incorrect": float(np.median(np.exp(X[y == 0, 2]))),
        "what_the_tail_start_marks": {
            "n_ok_tails": n,
            "forced_completion_computes": computing / n,
            "n_tail_before_answer_stated_permissive": before,
            "forced_completion_computes_when_tail_precedes_statement": (
                before_computing / before if before else float("nan")),
            "note": ("'computes' = the 48-token forced completion at the tail start "
                     "contains arithmetic; the stated-answer index uses the permissive "
                     "extractor, so it differs from the registered summary's strict one"),
        },
        "tokens_after_answer_first_stated": {
            "mean_share": float(np.mean(post_share)),
            "median_share": float(np.median(post_share)),
        },
    }


# ------------------------------------------------------------- RQ2, F10


def rq2_section(cfg, rng):
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    pid, diff, dep, kind = [], [], [], []
    zero = total = 0
    for a in audits:
        for c in a.get("claims", []):
            pe = c["paraphrase_effects"]
            pid.append(a["problem_id"])
            kind.append(a["window_kind"])
            diff.append(c["delete_effect"] - float(np.mean(pe)))
            dep.append(float(bool(c["reconstruction_dependent"])))
            zero += sum(1 for e in pe if e == 0.0)
            total += len(pe)
    diff_a, dep_a, pid_a = np.array(diff), np.array(dep), np.array(pid)
    per_problem = np.array([diff_a[pid_a == p].mean() for p in np.unique(pid_a)])
    w = stats.wilcoxon(per_problem)
    return {
        "n_claims": len(diff), "n_problems": int(len(per_problem)),
        "mean_paired_difference_problem_cluster": cluster_bootstrap_mean(
            diff_a, pid_a, REPS_BOOT, rng),
        "dependent_rate_problem_cluster": cluster_bootstrap_mean(dep_a, pid_a, REPS_BOOT, rng),
        "problem_level_wilcoxon_p": float(w.pvalue),
        "share_of_problems_with_positive_mean": float(np.mean(per_problem > 0)),
        "paraphrase_effects_exactly_zero": zero / total,
        "dependent_rate_by_window_kind": {
            k: float(dep_a[np.array(kind) == k].mean())
            for k in ("tail", "matched_position", "matched_length")},
    }


def f10_section(cfg, rng):
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    out = {}
    for other in ("matched_position", "matched_length", "gaussian_control"):
        pids, a, b = paired_problem_means(recon, "window_kind", "cosine", "tail", other)
        d = a - b
        boot = [rng.choice(d, len(d)).mean() for _ in range(REPS_BOOT)]
        out[other] = {
            "n_problems": len(pids),
            "mean_paired_difference": float(d.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
            "share_of_problems_tail_lower": float(np.mean(d < 0)),
            "note": "same rows as the registered test, averaged within problem and window kind",
        }
    return out


def family_section(cfg, rq2, f10):
    """The registered family with problem-level RQ2 and F10 tests substituted."""
    results = read_json(cfg.dir / "analysis" / "results.json")
    tests = []
    for t in results["tests"]["tests"]:
        p = t.get("p_value")
        name = t["name"]
        if name.startswith("RQ2:"):
            p = rq2["problem_level_wilcoxon_p"]
        for other in f10:
            if name == f"F10: reconstruction cosine, tail vs {other}":
                p = f10[other]["wilcoxon_p"]
        tests.append(TestResult(name=name, statistic=float("nan"),
                                p_value=float("nan") if p is None else float(p),
                                effect=float("nan"), effect_name=""))
    corrected = benjamini_hochberg(tests, 0.05)
    planned = bh_at_family_size(corrected, results["tests"]["n_tests"], 0.05)
    reg = {t["name"]: t for t in results["tests"]["tests"]}
    rows = {}
    for t in corrected:
        rows[t.name] = {
            "p_registered": reg[t.name].get("p_value"),
            "p_problem_level": None if not np.isfinite(t.p_value) else t.p_value,
            "q_evaluable": t.q_value, "significant_evaluable": t.significant,
            "significant_registered": reg[t.name].get("significant"),
            "significant_planned_family": planned["q_values"].get(t.name, {}).get("significant"),
        }
    return {"tests": rows,
            "n_significant_evaluable": sum(1 for r in rows.values() if r["significant_evaluable"]),
            "n_significant_planned_family": planned["n_significant"],
            "planned_family_size": planned["family_size"]}


# ------------------------------------------------------------------- RQ3


def rq3_section(cfg):
    _scripts()
    from nlaast.activations.store import ActivationStore
    from nlaast.causal.interventions import random_direction
    from run_interventions import fit_tail_direction

    rows = read_jsonl(cfg.dir / "causal" / "interventions.jsonl")
    by: dict = {}
    for r in rows:
        by.setdefault((r["intervention"], float(r.get("coefficient") or 0.0)), []).append(r)
    arms = {f"{k[0]}@{k[1]}": {
        "n": len(v),
        "answer_changed": sum(bool(r["answer_changed"]) for r in v),
        "marker_bearing": sum(int(r.get("verification_markers") or 0) > 0 for r in v),
        "mean_tokens_generated": float(np.mean([r.get("tokens_generated", 0) for r in v])),
    } for k, v in sorted(by.items())}

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")}
    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    d, info = fit_tail_direction(SimpleNamespace(target=SimpleNamespace(layer=layer)),
                                 ast_rows, traces, store, logging.getLogger(__name__))
    u = d / np.linalg.norm(d)
    rng = np.random.default_rng(SEED)
    rands = [random_direction(len(u), rng, reference=u) for _ in range(5)]
    H = [store.get(r["problem_id"], layer)[0] for r in ast_rows
         if traces[r["problem_id"]].get("split") == "eval" and store.has(r["problem_id"], layer)]
    H = np.vstack(H).astype(np.float64)
    norm = np.linalg.norm(H, axis=1)
    cand = np.abs(H @ u)
    rand = np.mean([np.abs(H @ r) for r in rands], axis=0)

    broad_traces = broad_tails = 0
    ast = {a["problem_id"]: a for a in ast_rows}
    for t in traces.values():
        if not t.get("ok"):
            continue
        broad_traces += bool(BROAD_VERIFICATION.search(t["trace"].lower()))
        ts = ast[t["problem_id"]].get("tail_start")
        if ts is not None and ts < len(t["chunks"]):
            broad_tails += bool(BROAD_VERIFICATION.search(
                t["trace"][t["chunks"][ts]["char_start"]:].lower()))
    return {
        "arms": arms,
        "direction": info,
        "edit_size_at_dose_1": {
            "n_eval_vectors": int(len(H)),
            "candidate_removed_norm_share": float(np.mean(cand / norm)),
            "random_removed_norm_share": float(np.mean(rand / norm)),
            "ratio_candidate_to_random": float(cand.mean() / rand.mean()),
            "note": ("projection removal of a unit direction removes |h.u|; the random "
                     "control is a unit direction too, so its edit is not matched in "
                     "size. Measured on eval-split chunk-boundary activations, a proxy "
                     "for the generated positions the arms edit"),
        },
        "broad_verification_lexicon": {
            "traces_with_any": broad_traces, "tails_with_any": broad_tails,
            "n_traces": sum(1 for t in traces.values() if t.get("ok")),
            "note": "exploratory screen of the construct's base rate; not a readout",
        },
    }


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_claims_audit" / "claims_audit.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("claims_audit")
    rng = np.random.default_rng(SEED)

    out = {
        "status": "post hoc, exploratory (D68); changes no registered result",
        "run_id": cfg.run_id,
        "created": utcnow(),
        "git": git_commit(),
        "seed": SEED,
        "reps": {"null": REPS_NULL, "shuffle": REPS_SHUFFLE, "bootstrap": REPS_BOOT},
        "inputs_sha256": {p: sha256(cfg.dir / p) for p in INPUTS},
    }
    out["o3"] = o3_section(cfg, log, rng)
    out["rq1"] = rq1_section(cfg)
    out["rq2"] = rq2_section(cfg, rng)
    out["f10"] = f10_section(cfg, rng)
    out["family"] = family_section(cfg, out["rq2"], out["f10"])
    out["rq3"] = rq3_section(cfg)
    write_json(path, out)
    print(json.dumps(out, indent=2, default=str))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
