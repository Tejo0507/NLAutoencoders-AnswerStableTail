"""Power audit for D83 Stage 1, from stored data only. CPU.

Stage 1's primary estimand is Δ₁ = AUC(pos+A+E) − AUC(pos+A), judged against
an equivalence margin of ±0.01 AUC. This script simulates Stage 1's analysis
on stored data, refitting everything each time, to ask:

* how wide its problem-cluster bootstrap CI will be at N problems × K points;
* how often a true Δ₁ near 0 would be declared *equivalent* rather than
  inconclusive;
* whether the bootstrap CI's width matches the spread of Δ̂ across samples.

Two proxies, because no NLA text exists off the menu:

* **P1 (real text):** ``main_nla240``'s menu units, with the explanation
  text E, subsampled to N problems and at most K units each.
* **P2 (Stage 1's exact sampling design):** ``main``'s every-boundary units
  (D83 Stage 0), N problems × K uniformly random boundaries, with the raw
  trace text R standing in for E.

Labels: ``Y_reg`` (O48), ``Y_tail`` and ``Y_commit`` (D79's committed-to-final
reading, a pessimistic stand-in for O256's class balance). No ``rep1``
outcome is read.

    python scripts/stage1_power_audit.py --reps 10

Writes ``runs/stage1_power_audit/power.json``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from _stage import load_config

from nlaast.analysis.information import (
    Design,
    grouped_folds,
    nested_probe,
    out_of_fold,
    problem_draws,
    weighted_auc,
    weights_for,
)
from nlaast.analysis.position import safe_vector, scenario_answers
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.trace.chunking import Chunk, prefix_text

MARGIN = 0.01
SEED = 20261021
BOOT = 1000


def _args(config, run_id=None):
    import argparse
    return argparse.Namespace(config=config, overrides=[], run_id=run_id)


def menu_units():
    """P1: main_nla240's menu units with E text and three labels."""
    import information_audit as ia
    import reanalyse_o3_registered as ro

    cfg = load_config(_args("main", "main_nla240"))
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")}
    fdir = cfg.dir.parent / "main_forced"
    _, completed, _, finished, _, _ = ro.load_forced(fdir, ro.sha256(cfg.dir / "traces" / "traces.jsonl"))
    units, pids, _ = ia.build_units(cfg, ro, traces, completed, finished, fdir)
    X = ia.activations_for(units, cfg.dir.parent / "main" / "acts", cfg.target.layer)
    return units, X, "E"


def boundary_units():
    """P2: every boundary of main (Stage 0), with R text and three labels."""
    import information_audit as ia
    import position_truncation_audit as pta
    import reanalyse_o3_registered as ro
    from score_baselines import build_rows

    cfg = load_config(_args("main"))
    tpath = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(tpath) if t.get("ok")}
    fdir = cfg.dir.parent / "main_forced"
    _, completed, _, finished, _, _ = ro.load_forced(fdir, ro.sha256(tpath))
    pids = sorted(traces)
    probs = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)
    rows = build_rows([traces[q] for q in pids], OUTCOME_FORCED, completed, "stored")
    cut = pta.cut_off_map(traces, pids, finished)
    ts = {r["problem_id"]: r.get("tail_start") for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    units = []
    for pr in probs:
        q = pr.problem_id
        t, r = traces[q], rows[q]
        commit = safe_vector(scenario_answers(r["answers"], [cut.get((q, i)) for i in range(pr.n)],
                                              r["final_answer"], "committed_to_final"),
                             r["answer_known"], r["gold"], r["final_answer"], equivalent)
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")}) for c in t["chunks"]]
        for i in range(pr.n):
            units.append({"problem_id": q, "chunk": i, "dataset": t.get("dataset"),
                          "prefix_tokens": int(t["chunks"][i]["prefix_tokens"]),
                          "R": prefix_text(chunks, i, t["trace"])[-600:],
                          "Y_reg": bool(pr.safe[i]), "Y_commit": bool(commit[i]),
                          "Y_tail": None if ts.get(q) is None else bool(i >= ts[q])})
    X = ia.activations_for(units, cfg.dir / "acts", cfg.target.layer)
    return units, X, "R"


def pos_matrix(units):
    return np.array([[np.log(max(1, u["prefix_tokens"])), u["chunk"], float(u["dataset"] == "math")]
                     for u in units])


def simulate(units, X, text_key, label, N, K, r):
    rng = np.random.default_rng(SEED + 1000 * r + N + K)
    elig = [k for k, u in enumerate(units) if u[label] is not None]
    by_p: dict[str, list[int]] = {}
    for k in elig:
        by_p.setdefault(units[k]["problem_id"], []).append(k)
    pids = sorted(by_p)
    take = rng.choice(len(pids), size=min(N, len(pids)), replace=False)
    idx = []
    for j in sorted(take):
        ks = by_p[pids[j]]
        idx += sorted(rng.choice(ks, size=min(K, len(ks)), replace=False).tolist())
    sub = [units[k] for k in idx]
    y = np.array([u[label] for u in sub], bool)
    groups = np.array([u["problem_id"] for u in sub])
    if len(np.unique(y)) < 2:
        return None
    folds = grouped_folds(y, groups, 10, SEED + r)
    extras, _ = nested_probe(X[idx], y, groups, folds, seed=SEED + r)
    pos = pos_matrix(sub)
    pa = out_of_fold(Design(pos), y, folds, extras)
    pae = out_of_fold(Design(pos, ([u[text_key] for u in sub],)), y, folds, extras)
    keys, counts = problem_draws(groups, BOOT, np.random.default_rng(SEED + r))
    W = weights_for(groups, keys, counts)
    rep = np.array([weighted_auc(y, pae, w) - weighted_auc(y, pa, w) for w in W])
    rep = rep[np.isfinite(rep)]
    d = weighted_auc(y, pae) - weighted_auc(y, pa)
    lo, hi = np.quantile(rep, [0.025, 0.975])
    return {"delta": float(d), "ci95": [float(lo), float(hi)], "se_boot": float(rep.std()),
            "n_units": len(idx), "n_problems": int(len(set(groups))), "positive_rate": float(y.mean()),
            "auc_A": float(weighted_auc(y, pa)), "equivalent": bool(lo > -MARGIN and hi < MARGIN)}


def summarise(runs):
    runs = [x for x in runs if x is not None]
    se = np.array([x["se_boot"] for x in runs])
    d = np.array([x["delta"] for x in runs])
    from scipy.stats import norm
    se_m = float(np.median(se))
    return {"n_sims": len(runs), "median_se_boot": se_m,
            "median_ci_half_width": float(np.median([(x["ci95"][1] - x["ci95"][0]) / 2 for x in runs])),
            "mean_delta": float(d.mean()), "sd_delta_across_sims": float(d.std(ddof=1)) if len(d) > 1 else None,
            "share_declared_equivalent": float(np.mean([x["equivalent"] for x in runs])),
            "positive_rate": float(np.mean([x["positive_rate"] for x in runs])),
            "power_to_detect_0.01_at_median_se": float(norm.cdf(0.01 / se_m - 1.96)),
            "p_equivalent_if_true_delta_0_at_median_se":
                float(2 * norm.cdf((MARGIN - 1.96 * se_m) / se_m) - 1) if MARGIN > 1.96 * se_m else 0.0,
            "sims": runs}


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--designs", default="200x4,200x8,250x4,250x8")
    p.add_argument("--labels", default="Y_reg,Y_commit,Y_tail")
    args = p.parse_args()
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    logging.basicConfig(level=logging.ERROR)
    out_path = Path(__file__).resolve().parents[1] / "runs" / "stage1_power_audit" / "power.json"
    t0 = time.monotonic()
    designs = [tuple(int(v) for v in s.split("x")) for s in args.designs.split(",")]
    labels = args.labels.split(",")
    out = {"status": "design-stage power audit for D83 Stage 1 (D85); stored data only; no rep1 outcome",
           "created": utcnow(), "git": git_commit(), "seed": SEED, "boot": BOOT, "margin": MARGIN,
           "reps": args.reps, "proxies": {}}
    for name, loader in (("P1_menu_real_text", menu_units), ("P2_boundaries_raw_text", boundary_units)):
        units, X, tk = loader()
        out["proxies"][name] = {"n_units_available": len(units), "text": tk, "cells": {}}
        for N, K in designs:
            if name.startswith("P1") and K > 5:
                continue      # the menu has about 5 units per problem
            for lab in labels:
                runs = []
                for r in range(args.reps):
                    runs.append(simulate(units, X, tk, lab, N, K, r))
                    print(f"[{time.monotonic() - t0:6.0f}s] {name} N={N} K={K} {lab} rep {r}", flush=True)
                out["proxies"][name]["cells"][f"N{N}_K{K}_{lab}"] = summarise(runs)
                write_json(out_path, out)
    out["elapsed_s"] = time.monotonic() - t0
    write_json(out_path, out)
    for name, pr in out["proxies"].items():
        for cell, s in pr["cells"].items():
            print(f"{name} {cell}: SE {s['median_se_boot']:.4f} half-width {s['median_ci_half_width']:.4f} "
                  f"equivalent {s['share_declared_equivalent']:.2f} power(0.01) "
                  f"{s['power_to_detect_0.01_at_median_se']:.2f} pos-rate {s['positive_rate']:.2f}")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
