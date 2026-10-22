"""F10 and RQ2 with the right unit of analysis, beside the registered tests (D72). CPU only.

The registered tests stay as computed (`runs/<run_id>/analysis/results.json`).
This script first reproduces each one from the stored rows. It then adds the
analyses ``nlaast.analysis.clustered`` describes:
- F10 as a paired, problem-level contrast, with a mixed model beside it;
- RQ2's claim-weighted mean with problem-clustered uncertainty, a problem-level
  test and a nested mixed model.
Finally it recomputes the Benjamini–Hochberg family under both conventions
D38 reports, with the corrected tests in place of the registered ones.
**Corrected, post hoc.**

    python scripts/corrected_f10_rq2.py --config main

Writes ``runs/<run_id>_corrected_tests/corrected_tests.json``; refuses to
replace it (``--force``).
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
from scipy import stats

from _stage import base_parser, load_config

from nlaast.analysis.clustered import mixed_contrast, mixed_mean, paired_problem_contrast
from nlaast.analysis.posthoc import cluster_bootstrap_mean
from nlaast.analysis.stats import TestResult, benjamini_hochberg, bh_at_family_size, cliffs_delta
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS = 10000


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_corrected_tests" / "corrected_tests.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    res = read_json(cfg.dir / "analysis" / "results.json")
    reg = {t["name"]: t for t in res["tests"]["tests"]}

    # ---------------------------------------------------------------- F10
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    for r in recon:
        r["window"] = f"{r['window_kind']}:{r['chunk_index']}"
    tail = [r["cosine"] for r in recon if r["window_kind"] == "tail"]
    f10 = {}
    for other in ("matched_position", "matched_length", "gaussian_control"):
        name = f"F10: reconstruction cosine, tail vs {other}"
        vals = [r["cosine"] for r in recon if r["window_kind"] == other]
        p_reg = float(stats.mannwhitneyu(tail, vals, alternative="two-sided").pvalue)
        if abs(p_reg - reg[name]["p_value"]) > 1e-9 * max(1.0, reg[name]["p_value"]) \
                and not (p_reg < 1e-100 and reg[name]["p_value"] < 1e-100):
            print(f"{name}: registered p did not reproduce ({p_reg} vs {reg[name]['p_value']})",
                  file=sys.stderr)
            return 2
        f10[other] = {
            "registered": {"unit": "sample", "test": "Mann-Whitney U, unpaired",
                           "n": reg[name]["n"], "p": reg[name]["p_value"],
                           "cliffs_delta": cliffs_delta(tail, vals)},
            "problem_level": paired_problem_contrast(recon, "window_kind", "cosine",
                                                     "tail", other, REPS, rng),
            "mixed_model": mixed_contrast(recon, "window_kind", "cosine", "tail", other,
                                          nested_key="window"),
        }

    # ---------------------------------------------------------------- RQ2
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    diff, dep, pid, expl = [], [], [], []
    for a in audits:
        e_id = f"{a['problem_id']}|{a['window_kind']}|{a.get('sample_index')}|{a.get('chunk_index', '')}"
        for c in a.get("claims", []):
            diff.append(c["delete_effect"] - float(np.mean(c["paraphrase_effects"])))
            dep.append(float(bool(c["reconstruction_dependent"])))
            pid.append(a["problem_id"])
            expl.append(e_id)
    diff_a, pid_a = np.array(diff), np.array(pid)
    per_problem = np.array([diff_a[pid_a == p].mean() for p in np.unique(pid_a)])
    name = "RQ2: deletion effect minus paraphrase effect, per claim"
    rq2 = {
        "registered": {"unit": "claim", "test": "Wilcoxon signed-rank on claims",
                       "n": reg[name]["n"], "p": reg[name]["p_value"],
                       "cohens_d_paired": reg[name]["effect"],
                       "interval": reg[name].get("interval")},
        "claim_weighted_mean_cluster_bootstrap": cluster_bootstrap_mean(diff_a, pid_a, REPS, rng),
        "problem_level": {"n_problems": len(per_problem),
                          "wilcoxon_p": float(stats.wilcoxon(per_problem).pvalue),
                          "d_z": float(per_problem.mean() / per_problem.std(ddof=1)),
                          "share_positive": float(np.mean(per_problem > 0))},
        "mixed_model_problem_and_explanation": mixed_mean(diff_a, pid, expl),
        "dependent_rate_cluster_bootstrap": cluster_bootstrap_mean(np.array(dep), pid_a, REPS, rng),
    }

    # ------------------------------------------------------------- family
    def family(p_of):
        tests = []
        for t in res["tests"]["tests"]:
            p = p_of(t)
            tests.append(TestResult(name=t["name"], statistic=float("nan"),
                                    p_value=float("nan") if p is None else float(p),
                                    effect=float("nan"), effect_name=""))
        corrected = benjamini_hochberg(tests, 0.05)
        planned = bh_at_family_size(corrected, res["tests"]["n_tests"], 0.05)
        return {t.name: {"p": None if not np.isfinite(t.p_value) else t.p_value,
                         "q_evaluable": t.q_value, "significant_evaluable": t.significant,
                         "significant_planned_family":
                             planned["q_values"].get(t.name, {}).get("significant")}
                for t in corrected}

    def corrected_p(kind):
        def f(t):
            n = t["name"]
            if n.startswith("RQ2:"):
                return (rq2["problem_level"]["wilcoxon_p"] if kind == "problem"
                        else rq2["mixed_model_problem_and_explanation"]["p"])
            for other, v in f10.items():
                if n == f"F10: reconstruction cosine, tail vs {other}":
                    return (v["problem_level"]["wilcoxon_p"] if kind == "problem"
                            else v["mixed_model"]["p"])
            return t.get("p_value")
        return f

    fams = {"registered": family(lambda t: t.get("p_value")),
            "corrected_problem_level": family(corrected_p("problem")),
            "corrected_mixed_model": family(corrected_p("mixed"))}
    counts = {k: {"evaluable": sum(1 for v in fam.values() if v["significant_evaluable"]),
                  "planned_family": sum(1 for v in fam.values() if v["significant_planned_family"])}
              for k, fam in fams.items()}

    out = {"status": "corrected, post hoc (D72); the registered tests are unchanged",
           "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(), "seed": SEED,
           "reps": REPS, "f10": f10, "rq2": rq2, "family": fams,
           "n_significant": counts}
    write_json(path, out)
    print(json.dumps(out, indent=1, default=str))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
