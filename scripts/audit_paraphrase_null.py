"""RQ2's paraphrase null: what each draw was, and the result without the invalid ones (D70).

CPU only. The paraphrase texts were never stored, but they are a
deterministic function of the stored explanations
(``faithfulness/claims.paraphrase_claim``), so they are regenerated here and
classified by ``faithfulness/paraphrase_qc``. Two checks run before any
number is used, and either failing stops the script:

1. every regenerated claim equals the stored ``claim_text``;
2. every ``identical`` draw has a stored effect of exactly 0, which is what
   ties each regenerated draw to its stored effect.

Then the registered RQ2 quantities are recomputed under D70's nulls N1 and
N2. The registered result is reproduced beside them first. **Post hoc**: the
registered RQ2 result stays the result of record.

    python scripts/audit_paraphrase_null.py --config main

Writes ``runs/<run_id>_paraphrase_qc/paraphrase_qc.json``; refuses to replace
it (``--force``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from scipy import stats

from _stage import base_parser, load_config

from nlaast.analysis.posthoc import cluster_bootstrap_mean
from nlaast.faithfulness.claims import paraphrase_claim, split_claims
from nlaast.faithfulness.paraphrase_qc import VALID_N1, VALID_N2, classify, token_change
from nlaast.logging_utils import read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS = 10000


def regenerate(audits, cfg):
    """One record per claim: stored numbers plus each draw's class."""
    fc = cfg.faithfulness
    claims, mismatched, zero_violations = [], 0, 0
    for e_i, a in enumerate(audits):
        stored = a.get("claims", [])
        if not stored:
            continue
        regen = split_claims(a["explanation"], fc.max_claims_per_explanation)
        if [c.text for c in regen] != [c["claim_text"] for c in stored]:
            mismatched += 1
            continue
        for c, s in zip(regen, stored):
            draws = []
            for v, eff in enumerate(s["paraphrase_effects"]):
                text = paraphrase_claim(c.text, v)
                cls = classify(c.text.strip(), text)
                if cls == "identical" and eff != 0.0:
                    zero_violations += 1
                draws.append({"variant": v, "class": cls, "effect": float(eff),
                              "token_change": token_change(c.text, text)})
            claims.append({"explanation": e_i, "problem_id": a["problem_id"],
                           "window_kind": a["window_kind"],
                           "delete_effect": float(s["delete_effect"]),
                           "registered_threshold": float(s["null_threshold"]),
                           "registered_dependent": bool(s["reconstruction_dependent"]),
                           "noise": bool(s["reproduced_under_noise"]),
                           "draws": draws})
    return claims, mismatched, zero_violations


def recompute(claims, valid, percentile, rng):
    """Dependence under a null restricted to ``valid`` classes (all if None)."""
    by_exp: dict[int, list[float]] = {}
    for c in claims:
        for d in c["draws"]:
            if valid is None or d["class"] in valid:
                by_exp.setdefault(c["explanation"], []).append(abs(d["effect"]))
    thr = {e: float(np.percentile(v, percentile)) for e, v in by_exp.items() if v}
    dep, dep_nn, pid, diff, diff_pid = [], [], [], [], []
    not_evaluable = 0
    for c in claims:
        if c["explanation"] not in thr:
            not_evaluable += 1
            continue
        d = c["delete_effect"] > thr[c["explanation"]]
        dep.append(float(d))
        dep_nn.append(float(d and not c["noise"]))
        pid.append(c["problem_id"])
        effs = [x["effect"] for x in c["draws"] if valid is None or x["class"] in valid]
        if effs:
            diff.append(c["delete_effect"] - float(np.mean(effs)))
            diff_pid.append(c["problem_id"])
    pid_a, diff_a, dpid = np.array(pid), np.array(diff), np.array(diff_pid)
    per_problem = np.array([diff_a[dpid == p].mean() for p in np.unique(dpid)])
    return {
        "n_claims_evaluable": len(dep), "n_claims_not_evaluable": not_evaluable,
        "median_threshold": float(np.median(list(thr.values()))),
        "dependent_rate": cluster_bootstrap_mean(np.array(dep), pid_a, REPS, rng),
        "dependent_and_not_noise_rate": cluster_bootstrap_mean(np.array(dep_nn), pid_a, REPS, rng),
        "delete_minus_paraphrase": cluster_bootstrap_mean(diff_a, dpid, REPS, rng),
        "problem_level_wilcoxon_p": float(stats.wilcoxon(per_problem).pvalue),
        "share_of_problems_positive": float(np.mean(per_problem > 0)),
    }


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_paraphrase_qc" / "paraphrase_qc.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    claims, mismatched, zero_viol = regenerate(audits, cfg)
    if mismatched or zero_viol:
        print(f"checks failed: {mismatched} explanations with mismatched claims, "
              f"{zero_viol} identical draws with a non-zero effect", file=sys.stderr)
        return 2

    draws = [d for c in claims for d in c["draws"]]
    by_class = Counter(d["class"] for d in draws)
    by_variant = {v: Counter(d["class"] for d in draws if d["variant"] == v)
                  for v in sorted({d["variant"] for d in draws})}
    zero = sum(1 for d in draws if d["effect"] == 0.0)
    rng = np.random.default_rng(SEED)
    pct = cfg.faithfulness.null_percentile

    reg = recompute(claims, None, pct, rng)
    stored_rate = float(np.mean([c["registered_dependent"] for c in claims]))
    if abs(reg["dependent_rate"]["estimate"] - stored_rate) > 1e-12:
        print(f"the registered rate did not reproduce ({reg['dependent_rate']['estimate']} "
              f"vs {stored_rate})", file=sys.stderr)
        return 2

    out = {
        "status": "post hoc (D70); the registered RQ2 result is unchanged",
        "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(), "seed": SEED,
        "checks": {"explanations_with_mismatched_claims": mismatched,
                   "identical_draws_with_nonzero_effect": zero_viol,
                   "n_claims": len(claims), "n_draws": len(draws)},
        "draws_with_zero_effect": zero,
        "draws_by_class": dict(by_class),
        "share_by_class": {k: v / len(draws) for k, v in by_class.items()},
        "by_variant": {str(k): dict(v) for k, v in by_variant.items()},
        "token_change_by_class": {
            k: float(np.mean([d["token_change"] for d in draws if d["class"] == k]))
            for k in by_class},
        "claims_with_no_valid_draw": {
            "N1": sum(1 for c in claims if not any(d["class"] in VALID_N1 for d in c["draws"])),
            "N2": sum(1 for c in claims if not any(d["class"] in VALID_N2 for d in c["draws"]))},
        "registered_null": reg,
        "N1_valid_draws": recompute(claims, VALID_N1, pct, rng),
        "N2_surface_verified_draws": recompute(claims, VALID_N2, pct, rng),
    }
    write_json(path, out)
    print(json.dumps(out, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
