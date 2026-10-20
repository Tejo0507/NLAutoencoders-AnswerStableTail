"""Does the registered-outcome O3 result depend on a seed or a layer? (D73) CPU only.

Two kinds of variation exist in the stored artifacts without new inference:

* **Verbaliser sampling.** Every window was verbalised twice with different
  seeds (``sample_index`` 0 and 1). Rebuilding the readout's score from one
  sample at a time gives two independent seed replicates of the readout's
  side of the comparison.
* **Probe layer.** Activations were stored at layers 14, 20 and 24 (F7). The
  probe comparator is refitted at 14 and 24 by the registered procedure
  (``score_baselines.fit_probe``). A refit at layer 20 must reproduce the
  stored scores, or the script stops.

Each variant is scored by ``reanalyse_o3_registered.analyse`` on design A,
the shared menu, with the plan's estimands. **Exploratory and post hoc.** The
generation seed, the model and the autoencoder's layer cannot be varied
without new inference; ``docs/COVERAGE_PLAN.md`` covers those.

    python scripts/coverage_robustness.py --config main

Writes ``runs/<run_id>_coverage/coverage_robustness.json``; refuses to replace
it (``--force``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast import config as config_mod
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS = 1000


def readout_from_sample(cfg, pids, n_chunks, sample):
    """The readout's carried-forward score from one verbaliser sample only."""
    per: dict[str, dict[int, list[float]]] = {}
    for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl"):
        if (r["window_kind"] == "gaussian_control" or not r.get("integrity_ok")
                or int(r["sample_index"]) != sample):
            continue
        per.setdefault(r["problem_id"], {}).setdefault(int(r["chunk_index"]), []).append(r["cosine"])
    out = {}
    for pid in pids:
        last, scores = None, []
        for i in range(n_chunks[pid]):
            vals = per.get(pid, {}).get(i)
            if vals:
                last = float(np.mean(vals))
            scores.append(last)
        out[pid] = scores
    return out


def probe_at_layer(cfg, layer, traces, log):
    from score_baselines import build_rows, fit_probe

    lcfg = config_mod.from_mapping({**cfg.to_dict(),
                                    "target": {**cfg.to_dict()["target"], "layer": layer}})
    ok = [t for t in traces.values()]
    base = build_rows(ok)
    _, scores, info = fit_probe(lcfg, ok, base, log)
    return scores, info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_coverage" / "coverage_robustness.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    sys.path.insert(0, here)
    import reanalyse_o3_registered as ro

    logging.basicConfig(level=logging.ERROR)
    log = logging.getLogger("coverage")
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    _, completed, _, _, agree, _ = ro.load_forced(cfg.dir.parent / "main_forced",
                                                  ro.sha256(traces_path))
    if agree is None or agree < ro.MIN_AGREEMENT:
        print("forcing instrument below the reproducibility bar; nothing computed", file=sys.stderr)
        return 2
    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    n_chunks = {p: len(traces[p]["chunks"]) for p in pids}
    menu = {p: np.array([i in menu_sets.get(p, set()) for i in range(n_chunks[p])]) for p in pids}
    problems = ro.problems_for(traces, pids, OUTCOME_FORCED, completed)

    # the layer-20 refit must reproduce the stored probe scores
    stored = read_json(cfg.dir / "baselines" / "boundary_scores.json")
    refit20, _ = probe_at_layer(cfg, cfg.target.layer, traces, log)
    diffs = [abs(a - b) for p in pids for a, b in zip(refit20[p], stored[p]["probe"])
             if a is not None and b is not None]
    none_mismatch = sum((a is None) != (b is None) for p in pids
                        for a, b in zip(refit20[p], stored[p]["probe"]))
    if none_mismatch or (diffs and max(diffs) > 1e-9):
        print(f"layer-20 refit does not reproduce the stored probe (max diff "
              f"{max(diffs) if diffs else None}, {none_mismatch} coverage mismatches)",
              file=sys.stderr)
        return 2

    def summary(s, name):
        g = s["gaps"]
        return {"variant": name, "oracle_bound": s["oracle_bound"],
                "gaps": {k: {"gap": v["gap"], "ci95": v["ci95"], "p": v["p_bootstrap"],
                             "n_budgets_matched": v["n_budgets_matched"]} for k, v in g.items()},
                "verdict": ro.verdict(s)}

    variants = {}
    rng = np.random.default_rng(SEED)
    variants["as_analysed"] = summary(ro.analyse(problems, scores, menu, rng, reps=REPS, mc_draws=50),
                                      "readout from both samples, probe at layer 20")
    for s in (0, 1):
        sc = {**scores, "nla_readout": readout_from_sample(cfg, pids, n_chunks, s)}
        variants[f"readout_sample_{s}"] = summary(
            ro.analyse(problems, sc, menu, np.random.default_rng(SEED), reps=REPS, mc_draws=50),
            f"readout from verbaliser sample {s} only")
    probe_info = {}
    for layer in (14, 24):
        ps, info = probe_at_layer(cfg, layer, traces, log)
        probe_info[layer] = {k: info.get(k) for k in ("n_examples",) if k in info}
        sc = {**scores, "probe": {p: ps[p] for p in pids}}
        variants[f"probe_layer_{layer}"] = summary(
            ro.analyse(problems, sc, menu, np.random.default_rng(SEED), reps=REPS, mc_draws=50),
            f"probe refitted at layer {layer}")

    out = {"status": "exploratory, post hoc (D73)", "run_id": cfg.run_id, "created": utcnow(),
           "git": git_commit(), "seed": SEED, "reps": REPS,
           "probe_refit_check": {"layer": cfg.target.layer, "max_abs_diff": max(diffs) if diffs else 0.0,
                                 "n_scores_compared": len(diffs)},
           "variants": variants}
    write_json(path, out)
    print(json.dumps(out, indent=1, default=str)[:15000])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
