"""F5's sweeps over the continuation temperature and the chunker (§6, D52).

``PROJECT_PLAN.md`` §6 registers F5 as "sweep ``K``, ``T``, ``w``, chunker;
report rank stability of conclusions". The ``ast --sweep`` pass covers ``K``
(downwards only) and unanimity from the recorded evidence. ``T`` and the
chunker cannot be simulated from it: continuations were sampled at one
temperature, and a different chunker puts boundaries where nothing was
generated. This script generates that evidence.

Conditions are taken from the configuration, which fixed them on 2026-10-06
before any data existed (ad44f99), not chosen here:

* ``T=<t>`` for every ``robustness.ast_sweep_temperature`` other than the base
  ``ast.continuation_temperature`` - the base condition is the run itself;
* ``chunker=<mode>`` for ``chunk.alt_mode``, at the base temperature.

Everything else is held at the run's values: the canonical traces are read,
not regenerated; the boundary scan is ``generate_traces.collect_boundaries``,
the procedure the run used; forced answers are greedy as before; K, the
continuation budget, the scan cap and the seeds derive exactly as in the
`traces` stage. Under a temperature condition the seeds are the run's own
(common random numbers), so a difference between conditions is not seed noise
layered on top of the temperature change.

Outputs go to ``runs/<run>_f5/`` and never into the run directory. Each
condition is checkpointed per problem and resumes. ``--analyse`` is CPU-only and
compares every finished condition with the base run.

    python scripts/run_f5_sweep.py --config main --condition T=0.6
    python scripts/run_f5_sweep.py --config main --analyse
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from _stage import base_parser, load_config

from nlaast import config as config_mod
from nlaast import logging_utils, provenance
from nlaast.analysis.stats import (
    benjamini_hochberg,
    paired_difference,
    paired_rate_difference_ci,
    unpaired_test,
    TestResult,
)
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.trace.chunking import chunk_trace

#: Fields of a trace row a condition keeps unchanged from the run.
_KEEP = ("id", "problem_id", "dataset", "split", "level", "question", "gold",
         "trace", "token_ids", "offsets", "n_tokens", "prompt_tokens",
         "truncated", "final_answer", "final_correct", "ok")

RQ1_NAME = "RQ1: tail fraction, correct vs incorrect final answers"


# --- conditions ---------------------------------------------------------

def sweep_conditions(cfg) -> dict[str, dict[str, Any]]:
    """The registered conditions, keyed by name. The base run is not one."""
    base_t = float(cfg.ast.continuation_temperature)
    out: dict[str, dict[str, Any]] = {}
    for t in cfg.robustness.ast_sweep_temperature:
        t = float(t)
        if abs(t - base_t) < 1e-9:
            continue
        out[f"T={t:.1f}"] = {"factor": "temperature", "temperature": t,
                           "chunk_mode": "sentence"}
    mode = cfg.chunk.alt_mode
    if mode and mode != "sentence":
        out[f"chunker={mode}"] = {"factor": "chunker", "temperature": base_t,
                                  "chunk_mode": mode}
    return out


def condition_config(cfg, cond: dict[str, Any]):
    """The run's config with only this condition's temperature changed.

    The chunker mode is not a config field the `traces` stage reads (it always
    used the sentence splitter), so it is passed to ``chunk_trace`` directly
    and recorded beside the hash rather than inside it.
    """
    d = cfg.to_dict()
    d["ast"] = {**d["ast"], "continuation_temperature": float(cond["temperature"])}
    return config_mod.from_mapping(d)


def condition_hash(ccfg, cond: dict[str, Any]) -> str:
    """Config hash plus the chunker mode, which the config hash does not cover."""
    blob = json.dumps({"config_hash": ccfg.hash(), "chunk_mode": cond["chunk_mode"]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def slug(name: str) -> str:
    return name.replace("=", "_").replace(".", "p")


def out_root(cfg) -> Path:
    return cfg.dir.parent / f"{cfg.run_id}_f5"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


# --- one problem ----------------------------------------------------------

def condition_row(trace: dict, chunks, boundaries: list[dict], capped: bool,
                  max_boundaries: int, name: str) -> dict:
    """A trace row under a condition: the run's trace, new chunks and evidence."""
    row = {k: trace.get(k) for k in _KEEP}
    row.update({
        "condition": name,
        "n_chunks": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
        "boundaries": boundaries,
        "boundary_scan_capped": capped,
        "max_boundaries_evaluated": max_boundaries,
    })
    return row


def run_condition(cfg, name: str, cond: dict, traces: list[dict], log,
                  limit: int | None) -> int:
    from generate_traces import collect_boundaries
    from nlaast.models.target import TargetModel

    ccfg = condition_config(cfg, cond)
    root = out_root(cfg)
    cdir = root / slug(name)
    cdir.mkdir(parents=True, exist_ok=True)
    ev_path = cdir / "evidence.jsonl"
    done = {r["problem_id"] for r in read_jsonl(ev_path)} if ev_path.exists() else set()
    todo = [t for t in traces if t["problem_id"] not in done]
    if limit:
        todo = todo[:limit]
    log.info("F5 %s: %d problems, %d already done, %d to do",
             name, len(traces), len(done), len(todo))

    stage = f"f5:{name}"
    manifest = provenance.Manifest.open(root, None, ccfg.hash(),
                                        overrides=[f"condition={name}"])
    manifest.start_stage(stage, condition=cond, condition_name=name,
                         condition_hash=condition_hash(ccfg, cond),
                         base_config_hash=cfg.hash(),
                         source_traces=str(cfg.dir / "traces" / "traces.jsonl"),
                         source_traces_sha256=sha256_file(cfg.dir / "traces" / "traces.jsonl"),
                         n_problems=len(traces))
    write_json(cdir / "condition.json", {
        "condition": name, **cond,
        "config_hash": ccfg.hash(), "condition_hash": condition_hash(ccfg, cond),
        "base_config_hash": cfg.hash(),
        "base_run": cfg.run_id,
        "k_continuations": ccfg.ast.k_continuations,
        "continuation_max_new_tokens": ccfg.ast.continuation_max_new_tokens,
        "max_boundaries_evaluated": ccfg.ast.max_boundaries_evaluated,
        "force_answer_max_new_tokens": ccfg.generation.force_answer_max_new_tokens,
        "top_p": ccfg.generation.top_p,
        "seed": ccfg.seed,
        "seed_rule": ("forced: derive(seed,'force',pid,first_index); continuations: "
                      "derive(seed,'cont',pid,first_index) % 2**31 - as in the traces stage"),
        "resolved_config": ccfg.to_dict(),
    })
    if not todo:
        manifest.finish_stage(stage, status="complete", output=str(ev_path),
                              metrics={"n_problems": len(done)})
        return 0

    model = TargetModel(ccfg)
    manifest.record_resource("target", repo_id=ccfg.target.repo_id,
                             revision=getattr(model, "revision", None),
                             local_path=str(model.model_path),
                             precision=ccfg.target.precision)
    started = time.monotonic()
    try:
        with logging_utils.JsonlWriter(ev_path) as w:
            for n, t in enumerate(todo, 1):
                t0 = time.monotonic()
                chunks = chunk_trace(t["trace"], ccfg.chunk,
                                     token_offsets=[tuple(o) for o in t["offsets"]],
                                     mode=cond["chunk_mode"])
                bounds, capped = collect_boundaries(
                    model, ccfg, t["problem_id"], t["question"], t["trace"],
                    chunks, t.get("final_answer"))
                w.write(condition_row(t, chunks, bounds, capped,
                                      ccfg.ast.max_boundaries_evaluated, name))
                el = time.monotonic() - started
                log.info("F5 %s [%d/%d] %s chunks=%d evaluated=%d %.0fs (eta %.0f min)",
                         name, n, len(todo), t["problem_id"], len(chunks),
                         sum(1 for b in bounds if b.get("evaluated")),
                         time.monotonic() - t0, el / n * (len(todo) - n) / 60)
    finally:
        model.close()
    n_done = len(read_jsonl(ev_path))
    complete = n_done == len(traces)
    manifest.finish_stage(stage, status="complete" if complete else "partial",
                          output=str(ev_path),
                          metrics={"n_problems": n_done, "n_expected": len(traces)})
    return 0


# --- analysis -------------------------------------------------------------

def detect_rows(cfg, evidence_rows: list[dict]) -> list[dict]:
    from detect_answer_stable_tail import detect_one
    return [detect_one(cfg, r) for r in evidence_rows]


def token_fraction(r: dict) -> float | None:
    if r.get("tail_start") is None or not r.get("total_tokens"):
        return None
    return float(r["tail_tokens"]) / float(r["total_tokens"])


def rq1_test(rows: list[dict], confidence: float) -> TestResult:
    tail = [r for r in rows if r.get("tail_start") is not None]
    return unpaired_test(RQ1_NAME,
                         [r["tail_fraction"] for r in tail if r.get("final_correct")],
                         [r["tail_fraction"] for r in tail if not r.get("final_correct")],
                         confidence)


def q_under_substitution(family: list[dict], name: str, p: float,
                         q: float = 0.05) -> dict:
    """BH q of ``name`` if its p-value in the run's family were ``p``.

    Every other test keeps its recorded p-value; the family is whatever was
    evaluable in the run, which is the convention the analysis reports (D38).
    """
    tests = []
    for t in family:
        tname = t.get("test", t.get("name"))
        pv = p if tname == name else t.get("p_value")
        if pv is None or not np.isfinite(pv):
            continue
        tests.append(TestResult(tname, float("nan"), float(pv), float("nan"), ""))
    benjamini_hochberg(tests, q)
    hit = next((t for t in tests if t.name == name), None)
    return {"q_value": None if hit is None else hit.q_value,
            "significant": None if hit is None else hit.significant,
            "family_size": len(tests)}


def compare(base: list[dict], cond: list[dict], *, same_chunker: bool,
            n_boot: int, confidence: float, seed: int) -> dict:
    """Paired comparison of a condition with the base run, over problems."""
    b = {r["problem_id"]: r for r in base}
    c = {r["problem_id"]: r for r in cond}
    shared = sorted(set(b) & set(c))
    both = [p for p in shared
            if b[p].get("tail_start") is not None and c[p].get("tail_start") is not None]

    def diff(get):
        x = [get(c[p]) for p in both]
        y = [get(b[p]) for p in both]
        iv = paired_difference(x, y, n_boot, confidence, seed)
        return {"n": iv.n, "mean_condition": float(np.mean(x)) if x else None,
                "mean_base": float(np.mean(y)) if y else None,
                **iv.to_dict()}

    from scipy import stats
    tf_c = [token_fraction(c[p]) for p in both]
    tf_b = [token_fraction(b[p]) for p in both]
    rho = stats.spearmanr(tf_c, tf_b) if len(both) >= 3 else None

    def nontrivial(r):
        return r.get("tail_start") is not None and r["tail_start"] != r["n_chunks"] - 1
    only_c = sum(1 for p in shared if nontrivial(c[p]) and not nontrivial(b[p]))
    only_b = sum(1 for p in shared if nontrivial(b[p]) and not nontrivial(c[p]))

    out = {
        "n_shared": len(shared),
        "n_tail_in_both": len(both),
        "tail_presence_agreement": (sum(1 for p in shared
                                        if (b[p].get("tail_start") is None)
                                        == (c[p].get("tail_start") is None))
                                    / len(shared)) if shared else None,
        "tail_token_fraction_diff": diff(token_fraction),
        "spearman_tail_token_fraction": (
            None if rho is None else {"rho": float(rho.statistic),
                                      "p_value": float(rho.pvalue), "n": len(both)}),
        "nontrivial_tail_rate_diff": {
            "condition_only": only_c, "base_only": only_b,
            **paired_rate_difference_ci(only_c, only_b, len(shared), confidence).to_dict()},
    }
    if same_chunker:
        # Chunk indices mean the same thing only under the same chunker.
        out["tail_fraction_diff"] = diff(lambda r: r["tail_fraction"])
        out["tail_start_identical"] = (sum(1 for p in both
                                           if b[p]["tail_start"] == c[p]["tail_start"])
                                       / len(both)) if both else None
        out["tail_start_diff_chunks"] = diff(lambda r: float(r["tail_start"]))
    return out


def forced_reproduction(base_traces: dict, cond_rows: list[dict]) -> dict:
    """Greedy forced answers at boundaries both scans evaluated must agree.

    A determinism check on the regeneration, meaningful only under the same
    chunker (the same boundary index is the same prefix).
    """
    n = same = 0
    for r in cond_rows:
        bt = base_traces.get(r["problem_id"])
        if not bt:
            continue
        bb = {x["index"]: x for x in bt.get("boundaries", []) if x.get("evaluated")}
        for x in r["boundaries"]:
            if x.get("evaluated") and x["index"] in bb:
                n += 1
                same += int(x.get("forced_answer") == bb[x["index"]].get("forced_answer"))
    return {"n_boundaries_compared": n, "forced_answer_identical": same,
            "rate": same / n if n else None}


def analyse(cfg, log) -> int:
    from detect_answer_stable_tail import summarise

    root = out_root(cfg)
    base_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    base_traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
                   if t.get("ok")}
    family = read_json(cfg.dir / "analysis" / "tables" / "hypothesis_tests.json")
    conf, nb, seed = cfg.analysis.confidence, cfg.analysis.bootstrap_iterations, cfg.analysis.seed

    base_rq1 = rq1_test(base_rows, conf)
    result: dict[str, Any] = {
        "_meta": {
            "base_run": cfg.run_id, "base_config_hash": cfg.hash(),
            "base_condition": {"temperature": cfg.ast.continuation_temperature,
                               "chunk_mode": "sentence"},
            "registered_spec": "PROJECT_PLAN.md §6 F5: sweep K, T, w, chunker; "
                               "report rank stability of conclusions",
            "conditions_from": {"temperature": list(cfg.robustness.ast_sweep_temperature),
                                "chunker": cfg.chunk.alt_mode},
            "bootstrap_iterations": nb, "confidence": conf, "seed": seed,
            "note": ("each condition's RQ1 test is a robustness re-test and is not "
                     "added to the run's BH family; q_under_substitution says what "
                     "the run's RQ1 q would be had this condition's p been the one "
                     "recorded"),
        },
        "base": {"summary": summarise(base_rows),
                 "rq1": base_rq1.to_dict()},
        "conditions": {},
    }
    for name, cond in sweep_conditions(cfg).items():
        ev_path = root / slug(name) / "evidence.jsonl"
        ev = read_jsonl(ev_path) if ev_path.exists() else []
        entry: dict[str, Any] = {"condition": cond, "n_problems": len(ev),
                                 "n_expected": len(base_traces)}
        if len(ev) < len(base_traces):
            entry["status"] = "incomplete" if ev else "not_run"
            result["conditions"][name] = entry
            continue
        ccfg = condition_config(cfg, cond)
        rows = detect_rows(ccfg, ev)
        logging_utils.write_jsonl(root / slug(name) / "ast.jsonl", rows)
        summ = summarise(rows)
        write_json(root / slug(name) / "summary.json", summ)
        t = rq1_test(rows, conf)
        same_chunker = cond["chunk_mode"] == "sentence"
        entry.update({
            "status": "ran",
            "config_hash": ccfg.hash(),
            "condition_hash": condition_hash(ccfg, cond),
            "evidence_sha256": sha256_file(ev_path),
            "n_scan_capped": sum(1 for r in ev if r.get("boundary_scan_capped")),
            "mean_chunks": float(np.mean([r["n_chunks"] for r in ev])),
            "n_boundaries_evaluated": sum(1 for r in ev for b in r["boundaries"]
                                          if b.get("evaluated")),
            "summary": summ,
            "rq1": t.to_dict(),
            "rq1_q_under_substitution": q_under_substitution(family, RQ1_NAME, t.p_value),
            "rq1_same_direction_as_base": bool(np.sign(t.effect) == np.sign(base_rq1.effect)),
            "vs_base": compare(base_rows, rows, same_chunker=same_chunker,
                               n_boot=nb, confidence=conf, seed=seed),
        })
        if same_chunker:
            entry["forced_answer_reproduction"] = forced_reproduction(base_traces, ev)
        result["conditions"][name] = entry
        log.info("F5 %s: mean tail fraction %.3f (base %.3f), RQ1 delta %.3f p=%.4f",
                 name, summ.get("mean_tail_fraction", float("nan")),
                 result["base"]["summary"]["mean_tail_fraction"], t.effect, t.p_value)
    write_json(root / "comparison.json", result)
    log.info("wrote %s", root / "comparison.json")
    return 0


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--condition", default=None,
                   help="condition to generate, e.g. T=0.6 or chunker=newline")
    p.add_argument("--analyse", action="store_true",
                   help="CPU only: compare every finished condition with the base run")
    args = p.parse_args()
    cfg = load_config(args)
    root = out_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    log = logging_utils.setup(root)
    log.info("F5 sweep: base run=%s config_hash=%s out=%s", cfg.run_id, cfg.hash(), root)

    if args.analyse:
        return analyse(cfg, log)
    conds = sweep_conditions(cfg)
    if args.condition not in conds:
        log.error("unknown condition %r; registered: %s", args.condition, sorted(conds))
        return 2
    traces = [t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")]
    return run_condition(cfg, args.condition, conds[args.condition], traces, log,
                         args.limit)


if __name__ == "__main__":
    sys.exit(main())
