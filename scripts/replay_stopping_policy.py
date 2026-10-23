"""Replay stopping policies on stored rep1 Stage 1 positions. CPU, read only.

A *planning* analysis for `docs/STOPPING_POLICY_PROPOSAL.md` §6. It is not
an experiment, it is not confirmatory, and it chooses nothing: it estimates
how much net saving varies across problems, and how coverage and selective
risk move with the threshold, so that the proposal's sample-size scenarios
rest on measured spreads instead of assumed ones.

Inputs (read only): `runs/rep1_stage1/{positions,forced,reconstructions}.jsonl`,
`runs/rep1/traces/traces.jsonl`, `runs/rep1/ast/ast.jsonl` and the layer-20
activations in `runs/rep1/acts/`. They are the stored outputs of Stage 1
(D96): 1,600 outcome-blind positions in 400 problems, with 256-token forced
answers and their exact generated-token counts.

Policies (out-of-fold, problem-grouped, as Stage 1 fits them):
- P0, position only: logistic on [log prefix tokens, chunk index, MATH];
- P1, same-label probe: P0's features plus a nested-CV linear probe on the
  layer-20 activation, trained for the same label.

Within each problem the policy walks its candidate positions in order and
stops at the first whose predicted P(safe) reaches τ. **Limit:** Stage 1
drew 4 positions per problem, so this is a thinned walk, not every boundary.

Net saving of a stop = max(0, 1 - (prefix tokens + forced tokens) / trace
tokens), with the exact generated count. P0 and P1 add no generated tokens.

Labels: `as_stored` is Stage 1's Y_ext. `conservative` also counts a
completion still cut off at 256 tokens as unsafe.

Thresholds here are chosen on the same data they are scored on. That is
fine for planning spreads and wrong for any claim, so every number is
labelled in-sample.

    python scripts/replay_stopping_policy.py --runs <runs dir> --out <new dir>

`--out` must not exist yet and must not lie inside `--runs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 20261023
RISK_TARGETS = (0.025, 0.05, 0.10)
COVERAGES = (0.25, 0.50, 0.75)
STATUS = ("planning replay on existing data (rep1 Stage 1); exploratory; in-sample "
          "thresholds; not confirmatory and not evidence for any claim")


# ---- pure policy arithmetic (tested on synthetic problems) ----


def problems_from_units(units, scores, label):
    """Per problem: candidates in chunk order with score, safe flag and saving."""
    by = {}
    for u, s in zip(units, scores):
        by.setdefault(u["problem_id"], []).append(
            (u["chunk"], float(s), bool(u[label]),
             max(0.0, 1.0 - (u["prefix_tokens"] + u["forced_tokens"]) / u["total_tokens"])))
    out = {}
    for pid, rows in by.items():
        rows.sort()
        out[pid] = (np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                    np.array([r[3] for r in rows]))
    return out


def walk(problems, tau, order):
    """For threshold tau: per problem (stopped, safe, saving), in `order`."""
    stopped, safe, saving = [], [], []
    for pid in order:
        s, ok, sv = problems[pid]
        hit = np.flatnonzero(s >= tau)
        if hit.size:
            k = hit[0]
            stopped.append(True)
            safe.append(bool(ok[k]))
            saving.append(float(sv[k]))
        else:
            stopped.append(False)
            safe.append(True)
            saving.append(0.0)
    return np.array(stopped), np.array(safe), np.array(saving)


def summary(stopped, safe, saving):
    n_stop = int(stopped.sum())
    return {"coverage": float(stopped.mean()),
            "selective_risk": float((~safe[stopped]).mean()) if n_stop else None,
            "n_stopped": n_stop, "mean_net_saving": float(saving.mean())}


def tau_grid(problems, n=200):
    allv = np.concatenate([p[0] for p in problems.values()])
    return np.unique(np.quantile(allv, np.linspace(0, 1, n)))


def tau_at_risk(problems, order, target, grid):
    """Most permissive tau (lowest) whose in-sample selective risk <= target."""
    for tau in grid:                         # ascending: most permissive first
        st, sf, _ = walk(problems, tau, order)
        if st.any() and (~sf[st]).mean() <= target:
            return float(tau)
    return None


def tau_at_coverage(problems, order, coverage, grid):
    best, gap = None, math.inf
    for tau in grid:
        c = walk(problems, tau, order)[0].mean()
        if abs(c - coverage) < gap:
            best, gap = float(tau), abs(c - coverage)
    return best


def paired(sav1, sav0):
    d = sav1 - sav0
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "n": int(d.size)}


def bootstrap_ci(values_fn, n_problems, reps, rng):
    est = []
    for _ in range(reps):
        idx = rng.integers(0, n_problems, n_problems)
        v = values_fn(idx)
        if v is not None:
            est.append(v)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))] if est else None


def c1_n_stopped(r, alpha, target=0.8, conf=0.95):
    """Smallest n with P(one-sided Clopper-Pearson upper bound <= alpha) >= target."""
    from scipy.stats import beta, binom

    def power(n):
        xs = [x for x in range(n + 1) if x < n and beta.ppf(conf, x + 1, n - x) <= alpha]
        return binom.cdf(max(xs), n, r) if xs else 0.0
    if r >= alpha:
        return None
    n = 10
    while power(n) < target:
        n = int(n * 1.05) + 1
        if n > 100000:
            return None
    for m in range(max(10, int(n / 1.06)), n + 1):
        if power(m) >= target:
            return m
    return n


def c2_n(sd, delta):
    from scipy.stats import norm
    return {"n_power80": math.ceil(((norm.ppf(.975) + norm.ppf(.8)) * sd / delta) ** 2),
            "n_halfwidth": math.ceil((norm.ppf(.975) * sd / (delta / 2)) ** 2)}


# ---- the replay ----


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_units(runs: Path):
    import stage1_offmenu as s1
    s1.OUT = runs / "rep1_stage1"            # read only: build_units only reads
    cfg = SimpleNamespace(dir=runs / "rep1", target=SimpleNamespace(layer=20))
    units = s1.build_units(cfg)
    for u in units:
        u["conservative"] = bool(u["Y_ext"]) and bool(u["forced_finished"])
        u["as_stored"] = bool(u["Y_ext"])
    return units, cfg


def oof_scores(units, X, label, seed):
    from nlaast.analysis.information import Design, grouped_folds, nested_probe, out_of_fold
    y = np.array([u[label] for u in units])
    g = np.array([u["problem_id"] for u in units])
    pos = np.array([[np.log(max(1, u["prefix_tokens"])), u["chunk"], float(u["dataset"] == "math")]
                    for u in units])
    folds = grouped_folds(y, g, 10, seed)
    extras, _ = nested_probe(X, y, g, folds, seed=seed)
    return {"P0_position": out_of_fold(Design(pos), y, folds),
            "P1_same_label_probe": out_of_fold(Design(pos), y, folds, extras)}


def replay(units, X, reps, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for label in ("as_stored", "conservative"):
        sc = oof_scores(units, X, label, seed)
        probs = {k: problems_from_units(units, v, label) for k, v in sc.items()}
        order = sorted(next(iter(probs.values())))
        grids = {k: tau_grid(p) for k, p in probs.items()}
        res = {"label_safe_rate": float(np.mean([u[label] for u in units])),
               "curves": {}, "matched_risk": {}, "matched_coverage": {}}
        for k, p in probs.items():
            res["curves"][k] = [{"tau": float(t), **summary(*walk(p, t, order))}
                                for t in grids[k][::5]]
        for kind, targets, pick in (("matched_risk", RISK_TARGETS, tau_at_risk),
                                    ("matched_coverage", COVERAGES, tau_at_coverage)):
            for tgt in targets:
                taus = {k: pick(p, order, tgt, grids[k]) for k, p in probs.items()}
                cell = {"target": tgt, "tau": taus}
                walks = {k: walk(probs[k], t, order) for k, t in taus.items() if t is not None}
                for k, w in walks.items():
                    cell[k] = summary(*w)
                    st, sf, sv = w
                    cell[k]["selective_risk_ci95"] = bootstrap_ci(
                        lambda i, st=st, sf=sf: (float((~sf[i][st[i]]).mean())
                                                 if st[i].any() else None),
                        len(order), reps, rng)
                    cell[k]["mean_net_saving_ci95"] = bootstrap_ci(
                        lambda i, sv=sv: float(sv[i].mean()), len(order), reps, rng)
                if len(walks) == 2:
                    s1 = walks["P1_same_label_probe"][2]
                    s0 = walks["P0_position"][2]
                    cell["P1_minus_P0"] = paired(s1, s0)
                    cell["P1_minus_P0"]["mean_diff_ci95"] = bootstrap_ci(
                        lambda i: float((s1[i] - s0[i]).mean()), len(order), reps, rng)
                res[kind][str(tgt)] = cell
        out[label] = res
    return out


def sample_size_refinement(res):
    """The proposal's §6 tables, re-read with the replay's measured spreads."""
    rows = []
    for label, r in res.items():
        for tgt, cell in r["matched_risk"].items():
            p1 = cell.get("P1_same_label_probe")
            if not p1 or p1["selective_risk"] is None or "P1_minus_P0" not in cell:
                continue
            sd = cell["P1_minus_P0"]["sd_diff"]
            row = {"label": label, "alpha_cal_in_sample": float(tgt),
                   "P1_coverage": p1["coverage"], "P1_selective_risk": p1["selective_risk"],
                   "sd_diff": sd,
                   "C2_n": {str(d): c2_n(sd, d) for d in (0.02, 0.05, 0.10)}}
            r_obs = p1["selective_risk"]
            if r_obs < 0.05 and p1["coverage"] > 0:
                n = c1_n_stopped(r_obs, 0.05)
                row["C1_alpha_0.05"] = {"n_stopped": n, "N_eval": (math.ceil(n / p1["coverage"])
                                                                   if n else None)}
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    runs, out = a.runs.resolve(), a.out.resolve()
    if out.exists():
        print(f"{out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    if runs == out or runs in out.parents:
        print("--out must not be inside --runs: the replay never writes there", file=sys.stderr)
        return 1
    import information_audit as ia
    from nlaast.provenance import git_commit, utcnow

    inputs = {rel: _sha(runs / rel) for rel in (
        "rep1_stage1/positions.jsonl", "rep1_stage1/forced.jsonl",
        "rep1_stage1/reconstructions.jsonl", "rep1/traces/traces.jsonl", "rep1/ast/ast.jsonl")}
    units, cfg = load_units(runs)
    X = ia.activations_for(units, cfg.dir / "acts", cfg.target.layer)
    res = replay(units, X, a.reps, a.seed)
    doc = {"status": STATUS, "created": utcnow(), "git": git_commit(), "seed": a.seed,
           "reps": a.reps, "inputs_sha256": inputs, "n_units": len(units),
           "n_problems": len({u["problem_id"] for u in units}),
           "limits": ["4 outcome-blind positions per problem: a thinned walk, not every boundary",
                      "thresholds chosen in-sample", "one model, one layer, linear probe"],
           "replay": res, "sample_size_refinement": sample_size_refinement(res)}
    out.mkdir(parents=True)
    (out / "replay.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    after = {rel: _sha(runs / rel) for rel in inputs}
    if after != inputs:
        print("an input changed during the replay", file=sys.stderr)
        return 1
    print(f"wrote {out / 'replay.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
