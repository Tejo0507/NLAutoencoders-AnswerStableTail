"""F8, second half: does the headline comparison survive bf16 target activations?

``PROJECT_PLAN.md`` §6 registers F8 as "Q1/Q2 above, plus bf16-CPU replication
of the headline comparison on a subsample". Q1 (activation drift) was measured
on the pilot. This script runs the part of the replication this machine can
run, and it is **partial by decision (D55)**: the target's layer-20 activations
are recomputed in bf16, and everything downstream of them is held fixed.

What changes between the two arms is one thing - the precision of the target
that produced the activations the autoencoder reads:

* ``runs/<run>_f8_bf16/`` - activations from the bf16 target;
* ``runs/<run>_f8_nf4/``  - the run's own NF4 activations for the same problems.

Both arms then run the unchanged `nla` stage (NF4 verbaliser and reconstructor)
on an identical task list, so batching and every sampling seed are identical
too, and the unchanged `analysis` stage computes O3 on the subsample. Comparing
the NF4 arm with the run's original NLA output on the same problems gives the
sampling-noise floor that a bf16-vs-NF4 difference has to clear.

Not replicated, and stated as such: the verbaliser and reconstructor stay NF4;
the behavioural arms (forced answers, semantic-entropy samples, probe) stay on
the NF4 target. Replicating those needs ~41 GB of bf16 weights and bf16
generation on a 6 GB GPU (D55).

Phases::

    python scripts/run_f8_bf16_headline.py --config main --phase prepare
    python scripts/run_f8_bf16_headline.py --config main --phase bf16      # needs the bf16 target
    python scripts/run_autoencoder.py --config main --run-id main_f8_bf16 --set nla.max_problems=0
    python scripts/run_autoencoder.py --config main --run-id main_f8_nf4  --set nla.max_problems=0
    python scripts/aggregate_results.py --config main --run-id main_f8_bf16 --set nla.max_problems=0
    python scripts/aggregate_results.py --config main --run-id main_f8_nf4  --set nla.max_problems=0
    python scripts/run_f8_bf16_headline.py --config main --phase compare
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from _stage import base_parser

from nlaast import config as config_mod
from nlaast import logging_utils, provenance
from nlaast.activations.store import ActivationStore
from nlaast.data import datasets
from nlaast.logging_utils import read_json, read_jsonl, write_json

#: Upstream stages each arm needs, copied unchanged from the run.
_COPY = ("data", "traces", "ast", "baselines")
_TARGET_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def arm_dir(cfg, arm: str) -> Path:
    return cfg.dir.parent / f"{cfg.run_id}_f8_{arm}"


def select_subsample(cfg) -> list[str]:
    """The registered subsample size, drawn from the NLA arm as the arm draws.

    ``robustness.quantisation_subsample`` problems out of those the run's NLA
    arm covers, by the arm's own ``stratified_cap`` - so the subsample is a
    proportional slice of both datasets rather than the GSM8K-only prefix a
    sorted-id cut would give.
    """
    nla_ids = {r["problem_id"] for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
               if r["window_kind"] != "gaussian_control"}
    rows = [r for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl") if r["problem_id"] in nla_ids]
    picked = datasets.stratified_cap(rows, int(cfg.robustness.quantisation_subsample))
    return sorted(r["problem_id"] for r in picked)


def _block_extra(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k not in ("positions", "kinds")}


def prepare(cfg, log) -> int:
    ids = select_subsample(cfg)
    src = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    for arm in ("bf16", "nf4"):
        d = arm_dir(cfg, arm)
        if d.exists():
            log.error("%s exists; refusing to overwrite", d)
            return 1
        d.mkdir(parents=True)
        for stage in _COPY:
            shutil.copytree(cfg.dir / stage, d / stage)
        (d / "acts").mkdir()
    nf4 = ActivationStore(arm_dir(cfg, "nf4") / "acts")
    for pid in ids:
        arr, meta = src.get(pid, layer)
        nf4.put(pid, layer, arr, meta["positions"], meta["kinds"], _block_extra(meta))
    write_json(arm_dir(cfg, "bf16") / "subsample.json", {
        "problem_ids": ids, "n": len(ids), "layer": layer,
        "rule": "datasets.stratified_cap over the NLA arm's problems, "
                "cap = robustness.quantisation_subsample",
        "source_run": cfg.run_id,
    })
    shutil.copy2(arm_dir(cfg, "bf16") / "subsample.json", arm_dir(cfg, "nf4") / "subsample.json")
    log.info("F8 headline: %d problems prepared in %s and %s", len(ids),
             arm_dir(cfg, "bf16"), arm_dir(cfg, "nf4"))
    return 0


def bf16_activations(cfg, log) -> int:
    """Layer-20 activations from the bf16 target, at the run's own positions."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from nlaast.models.loading import free_cuda, snapshot
    from run_falsification_tests import f8_placement, f8_replay_sequence

    d = arm_dir(cfg, "bf16")
    ids = read_json(d / "subsample.json")["problem_ids"]
    layer = cfg.target.layer
    src_store = ActivationStore(cfg.dir / "acts")
    out_store = ActivationStore(d / "acts")
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")}
    todo = [p for p in ids if not out_store.has(p, layer)]
    log.info("F8 bf16: %d problems, %d to do", len(ids), len(todo))

    manifest = provenance.Manifest.open(d, cfg.to_dict(), cfg.hash())
    manifest.start_stage("f8_bf16_acts", n_problems=len(ids))
    src = snapshot(cfg.target.repo_id, _TARGET_REVISION, local_files_only=True)
    manifest.record_resource("target_bf16", repo_id=cfg.target.repo_id,
                             revision=_TARGET_REVISION, local_path=str(src),
                             precision="bf16")
    conf = AutoConfig.from_pretrained(src)
    conf.num_hidden_layers = layer + 1
    tok = AutoTokenizer.from_pretrained(src)
    offload = d / "f8_offload"
    placement = f8_placement(log, offload_dir=offload)
    log.info("F8 bf16: placement %s", placement["record"])
    model = AutoModelForCausalLM.from_pretrained(
        src, config=conf, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        **placement["kwargs"]).eval()

    captured: dict = {}

    def hook(_m, _i, o):
        captured["h"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook)
    per_problem, skipped = [], []
    started = time.monotonic()
    try:
        for n, pid in enumerate(todo, 1):
            t = traces[pid]
            arr, meta = src_store.get(pid, layer)
            messages = [{"role": "system", "content": cfg.generation.system_prompt},
                        {"role": "user", "content": t["question"]}]
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            full, reason = f8_replay_sequence(prompt_ids, t, meta,
                                              cfg.target.max_position_embeddings)
            if full is None:
                skipped.append({"problem_id": pid, "reason": reason})
                log.warning("F8 bf16: skipping %s - %s", pid, reason)
                continue
            with torch.inference_mode():
                model(input_ids=torch.tensor([full], dtype=torch.long), use_cache=False)
            hidden = captured["h"][0].numpy()
            pos = meta.get("resolved_positions") or meta["positions"]
            if max(pos) >= hidden.shape[0]:
                skipped.append({"problem_id": pid, "reason": "position out of range"})
                continue
            b = hidden[np.asarray(pos, dtype=int)]
            out_store.put(pid, layer, b, meta["positions"], meta["kinds"],
                          {**_block_extra(meta), "precision": "bf16",
                           "source_revision": _TARGET_REVISION})
            a = arr.astype(np.float64)
            bb = b.astype(np.float64)
            cs = (a * bb).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(bb, axis=1) + 1e-12)
            per_problem.append({"problem_id": pid, "n": int(len(cs)),
                                "mean_cosine": float(cs.mean()),
                                "min_cosine": float(cs.min()),
                                "norm_ratio_nf4_over_bf16": float(
                                    np.mean(np.linalg.norm(a, axis=1)
                                            / (np.linalg.norm(bb, axis=1) + 1e-12)))})
            log.info("F8 bf16 [%d/%d] %s n=%d mean cos(nf4,bf16)=%.4f (%.0fs)",
                     n, len(todo), pid, len(cs), cs.mean(), time.monotonic() - started)
    finally:
        handle.remove()
        del model
        free_cuda()
        if offload.exists():
            shutil.rmtree(offload, ignore_errors=True)
    prev = read_json(d / "q1.json") if (d / "q1.json").exists() else {"per_problem": []}
    done = {r["problem_id"]: r for r in prev["per_problem"]}
    done.update({r["problem_id"]: r for r in per_problem})
    rows = list(done.values())
    write_json(d / "q1.json", {
        "per_problem": rows, "skipped": skipped, "placement": placement["record"],
        "n_problems": len(rows), "n_vectors": sum(r["n"] for r in rows),
        "mean_of_problem_means": float(np.mean([r["mean_cosine"] for r in rows])) if rows else None,
    })
    complete = all(out_store.has(p, layer) for p in ids)
    manifest.finish_stage("f8_bf16_acts", status="complete" if complete else "partial",
                          metrics={"n_problems": len(rows), "n_skipped": len(skipped)})
    return 0 if complete else 1


# --- compare --------------------------------------------------------------

def window_scores(recon_rows: list[dict], problems: set[str]) -> dict[tuple, float]:
    """Mean reconstruction cosine per (problem, window kind, chunk), as the
    O3 score uses it: Gaussian controls and integrity failures excluded."""
    acc: dict[tuple, list[float]] = {}
    for r in recon_rows:
        if r["problem_id"] not in problems or r["window_kind"] == "gaussian_control":
            continue
        if not r.get("integrity_ok"):
            continue
        acc.setdefault((r["problem_id"], r["window_kind"], int(r["chunk_index"])),
                       []).append(float(r["cosine"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_scores(a: dict, b: dict, n_boot: int, conf: float, seed: int) -> dict:
    from scipy import stats

    from nlaast.analysis.stats import paired_difference

    keys = sorted(set(a) & set(b))
    x = [a[k] for k in keys]
    y = [b[k] for k in keys]
    if len(keys) < 3:
        return {"n_windows": len(keys)}
    iv = paired_difference(x, y, n_boot, conf, seed)
    return {"n_windows": len(keys), "mean_first": float(np.mean(x)),
            "mean_second": float(np.mean(y)),
            "pearson_r": float(stats.pearsonr(x, y).statistic),
            "spearman_rho": float(stats.spearmanr(x, y).statistic),
            "mean_abs_diff": float(np.mean(np.abs(np.subtract(x, y)))),
            "diff": iv.to_dict()}


def o3_rows(run: Path) -> list[dict]:
    p = run / "analysis" / "tables" / "hypothesis_tests.json"
    return [t for t in read_json(p) if str(t.get("test", "")).startswith("O3")] if p.exists() else []


def compare(cfg, log) -> int:
    bf, nf = arm_dir(cfg, "bf16"), arm_dir(cfg, "nf4")
    ids = set(read_json(bf / "subsample.json")["problem_ids"])
    conf, nb, seed = cfg.analysis.confidence, cfg.analysis.bootstrap_iterations, cfg.analysis.seed
    r_bf = read_jsonl(bf / "nla" / "reconstructions.jsonl")
    r_nf = read_jsonl(nf / "nla" / "reconstructions.jsonl")
    r_orig = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    s_bf, s_nf, s_orig = (window_scores(r, ids) for r in (r_bf, r_nf, r_orig))

    def integrity(rows):
        act = [r for r in rows if r["problem_id"] in ids and r["window_kind"] != "gaussian_control"]
        return {"n": len(act), "ok_rate": float(np.mean([bool(r.get("integrity_ok")) for r in act]))
                if act else None}

    out: dict[str, Any] = {
        "_meta": {
            "registered_spec": "PROJECT_PLAN.md §6 F8: bf16-CPU replication of the "
                               "headline comparison on a subsample",
            "scope": "PARTIAL (D55): target activations bf16; verbaliser, reconstructor "
                     "and behavioural arms unchanged (NF4)",
            "subsample": sorted(ids), "n_problems": len(ids),
            "target_revision": _TARGET_REVISION,
        },
        "q1_activation_drift": read_json(bf / "q1.json"),
        "integrity": {"bf16": integrity(r_bf), "nf4_rerun": integrity(r_nf),
                      "nf4_original": integrity(r_orig)},
        "window_score_bf16_vs_nf4_rerun": paired_scores(s_bf, s_nf, nb, conf, seed),
        "window_score_nf4_rerun_vs_original": paired_scores(s_nf, s_orig, nb, conf, seed),
        "o3": {"bf16": o3_rows(bf), "nf4_rerun": o3_rows(nf),
               "main_n120": o3_rows(cfg.dir)},
    }
    nla_info = {}
    for arm, d in (("bf16", bf), ("nf4_rerun", nf)):
        res = d / "analysis" / "results.json"
        if res.exists():
            nla_info[arm] = read_json(res).get("nla_stopping")
    out["nla_stopping"] = nla_info
    write_json(bf.parent / f"{cfg.run_id}_f8_comparison.json", out)
    log.info("wrote %s", bf.parent / f"{cfg.run_id}_f8_comparison.json")
    return 0


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--phase", choices=["prepare", "bf16", "compare"], required=True)
    args = p.parse_args()
    cfg = config_mod.load(args.config, args.overrides)
    log = logging_utils.setup(arm_dir(cfg, "bf16") if args.phase != "prepare" else None)
    log.info("F8 headline (%s): run=%s config_hash=%s", args.phase, cfg.run_id, cfg.hash())
    return {"prepare": prepare, "bf16": bf16_activations, "compare": compare}[args.phase](cfg, log)


if __name__ == "__main__":
    sys.exit(main())
