"""The stopping-policy pilot (`docs/STOPPING_POLICY_PILOT_PROTOCOL.md`). GPU for `force`; CPU otherwise.

Nothing runs without a frozen protocol: `configs/stopping_pilot_protocol.json`
with `"status": "frozen"` and every decision filled. The template with the
open decisions is `configs/stopping_pilot_protocol.template.json`.

Phases, in the order the protocol fixes (each refuses to run out of order):

    check                     validate the frozen protocol and print its SHA-256
    force     --run pilot_cal   256-token forced answer at every boundary, exact counts,
                                plus the re-forcing and stability checks   [GPU]
    fit                       fit P0 (position) and P1 (same-label probe) on rep1
                              Stage 1 (read only) and freeze them
    score     --run pilot_cal   both policies' scores at every boundary
    calibrate                 SGR threshold per policy on pilot_cal; frozen and hashed
    force     --run pilot_eval  as above; refuses until the calibration exists  [GPU]
    score     --run pilot_eval
    evaluate                  one pass on pilot_eval with the frozen thresholds

Traces and activations come from the existing stages, run before `force`:

    python scripts/prepare_fresh_problems.py --config pilot_cal     (then pilot_eval)
    python scripts/generate_traces.py --config pilot_cal --no-entropy
    python scripts/extract_activations.py --config pilot_cal --layers 20

Every output goes to a new directory under `--runs`; nothing is overwritten,
and rep1's stored outputs (`--train-runs`) are only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PROTOCOL = ROOT / "configs" / "stopping_pilot_protocol.json"
POLICIES = ("P0_position", "P1_same_label_probe")
#: Every decision the protocol must fill before anything runs (protocol §5).
REQUIRED = ("alpha", "alpha_justification", "alpha_cal", "delta", "eval_confidence",
            "truncation_rule", "forcing_budget", "tau_grid_n", "n_cal", "n_eval",
            "calibration_source", "delta_save", "gpu_hours_cap", "gates", "seeds", "p3_scope")


# ---- protocol --------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL) -> dict:
    """The frozen protocol, or SystemExit naming what is missing."""
    if not path.exists():
        raise SystemExit(f"no frozen protocol at {path}; the pilot does not run without one")
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("status") != "frozen":
        raise SystemExit(f"protocol status is {p.get('status')!r}, not 'frozen'")
    missing = [k for k in REQUIRED if p.get(k) in (None, "", {})]
    if missing:
        raise SystemExit(f"protocol decisions not filled: {missing}")
    if p["calibration_source"] != "fresh":
        raise SystemExit("only a fresh calibration set is implemented (protocol §3.1)")
    if p["p3_scope"] != "none":
        raise SystemExit("P3 (NLA) is not implemented in the pilot (protocol §3.7)")
    return p


def _new_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} exists and is not empty; refusing to overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- force (GPU) -----------------------------------------------------------


def config_counts(run: str) -> dict:
    """The problem counts a run's config asks for."""
    import yaml
    data = yaml.safe_load((ROOT / "configs" / f"{run}.yaml").read_text(encoding="utf-8"))["data"]
    return {"gsm8k": int(data["n_gsm8k"]), "math": int(data["n_math"])}


def check_configs(proto: dict) -> None:
    for run, key in (("pilot_cal", "n_cal"), ("pilot_eval", "n_eval")):
        want = {k: int(v) for k, v in proto[key].items()}
        if config_counts(run) != want:
            raise SystemExit(f"configs/{run}.yaml asks for {config_counts(run)}, but the "
                             f"frozen protocol's {key} is {want}")


def cmd_force(a, proto) -> int:
    if a.run == "pilot_eval" and not (a.runs / "pilot_calibration" / "calibration.json").exists():
        raise SystemExit("evaluation forcing refuses to start before the calibration is frozen")
    check_configs(proto)
    from force_all_boundaries import block_size, blocks
    from _stage import base_parser, load_config

    from nlaast.data.answers import PERMISSIVE, extract_answer
    from nlaast.logging_utils import JsonlWriter, read_jsonl, setup as setup_logging
    from nlaast.models.target import TargetModel
    from nlaast.seeding import derive
    from nlaast.trace.chunking import Chunk, prefix_text

    cfg = load_config(base_parser("x").parse_args(["--config", a.run]))
    run_dir = a.runs / a.run
    log = setup_logging(run_dir)
    budget = int(proto["forcing_budget"])
    traces = {t["problem_id"]: t for t in read_jsonl(run_dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    chunks = {p: [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")})
                  for c in t["chunks"]] for p, t in traces.items()}
    rng = np.random.default_rng(int(proto["seeds"]["force_checks"]))
    every = [(p, i) for p in sorted(chunks) for i in range(len(chunks[p]))]
    n_chk = min(int(proto["gates"]["n_reforce_check"]), len(every))
    check = {every[j] for j in rng.choice(len(every), n_chk, replace=False)}
    model = TargetModel(cfg)
    writer = JsonlWriter(run_dir / "pilot_forced.jsonl", key="id")
    gcfg = cfg.generation
    blk = block_size(gcfg.batch_size)

    def prompt(p, i):
        t = traces[p]
        return model.build_prompt(t["question"], prefix_text(chunks[p], i, t["trace"])
                                  + gcfg.force_answer_suffix)

    work = [(purpose, p, b) for purpose, sel in
            (("ext", {p: list(range(len(c))) for p, c in chunks.items()}),
             ("repro", {p: [i for (q, i) in check if q == p] for p in chunks}))
            for p in sorted(sel) for b in blocks(sel[p], blk) if b]
    t0 = time.monotonic()
    try:
        with writer:
            for n, (purpose, p, b) in enumerate(work):
                todo = [i for i in b if not writer.has(f"{p}|{i}|{purpose}")]
                if not todo:
                    continue
                gens = model.generate([prompt(p, i) for i in todo], max_new_tokens=budget,
                                      temperature=0.0, top_p=1.0, do_sample=False,
                                      seed=derive(cfg.seed, "pilot_force", p, todo[0]),
                                      batch_size=len(todo))
                for i, g in zip(todo, gens):
                    text = gcfg.force_answer_suffix + g.text
                    writer.write({"id": f"{p}|{i}|{purpose}", "problem_id": p, "chunk": i,
                                  "purpose": purpose, "budget": budget,
                                  "answer": extract_answer(text, PERMISSIVE),
                                  "finished": bool(g.finished),
                                  # the exact generated count, recorded at generation
                                  "n_tokens": int(g.n_tokens),
                                  "prefix_tokens": chunks[p][i].prefix_tokens})
                if (n + 1) % 20 == 0:
                    log.info("%d/%d blocks, %.1fs per block since start", n + 1, len(work),
                             (time.monotonic() - t0) / (n + 1))
    finally:
        model.close()
    return 0


# ---- labelled boundary rows --------------------------------------------------


def boundary_rows(run_dir: Path) -> list[dict]:
    """Every forced boundary of a run, with what the policies and labels need."""
    from nlaast.data.answers import equivalent
    from nlaast.logging_utils import read_jsonl

    traces = {t["problem_id"]: t for t in read_jsonl(run_dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    rows = []
    for f in read_jsonl(run_dir / "pilot_forced.jsonl"):
        if f["purpose"] != "ext":
            continue
        t = traces[f["problem_id"]]
        rows.append({"problem_id": f["problem_id"], "chunk": f["chunk"],
                     "dataset": t.get("dataset"),
                     "forced_correct": equivalent(f["answer"], t.get("gold")),
                     "finished": f["finished"],
                     "final_correct": equivalent(t.get("final_answer"), t.get("gold")),
                     "prefix_tokens": int(f["prefix_tokens"]), "forced_tokens": int(f["n_tokens"]),
                     "total_tokens": int(t["n_tokens"])})
    return rows


def instrument(run_dir: Path, rows: list[dict], gates: dict) -> dict:
    """The protocol's instrument checks: re-forcing agreement and attrition."""
    from nlaast.logging_utils import read_jsonl

    forced = list(read_jsonl(run_dir / "pilot_forced.jsonl"))
    ext = {(r["problem_id"], r["chunk"]): r for r in forced if r["purpose"] == "ext"}
    rep = [r for r in forced if r["purpose"] == "repro"]
    agree = [ext[(r["problem_id"], r["chunk"])]["answer"] == r["answer"] for r in rep
             if (r["problem_id"], r["chunk"]) in ext]
    traces = [t for t in read_jsonl(run_dir / "traces" / "traces.jsonl") if t.get("ok")]
    planned = sum(len(t["chunks"]) for t in traces)
    out = {"reforce_agreement": float(np.mean(agree)) if agree else None,
           "n_reforce": len(agree), "attrition_share": len(rows) / planned if planned else 0.0,
           "planned_boundaries": planned, "forced_boundaries": len(rows)}
    out["passed"] = (out["reforce_agreement"] is not None
                     and out["reforce_agreement"] >= gates["reforce_agreement"]
                     and out["attrition_share"] >= gates["attrition"])
    return out


# ---- policies: fit on rep1 Stage 1, frozen ----------------------------------


def training_label(unit: dict, rule: str) -> bool:
    """A Stage 1 unit's safety label under the protocol's truncation rule.

    Stage 1 stores Y_ext (the parsed 256-token answer) and Y_ext_u (a cut-off
    answer counted as no answer); the primary rule also counts a cut-off as unsafe.
    """
    if rule == "cut_off_unsafe":
        return bool(unit["Y_ext"]) and bool(unit["forced_finished"])
    if rule == "as_parsed":
        return bool(unit["Y_ext"])
    if rule == "as_no_answer":
        return bool(unit["Y_ext_u"])
    raise SystemExit(f"unknown truncation rule {rule!r}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _pos(rows) -> np.ndarray:
    return np.array([[np.log(max(1, r["prefix_tokens"])), r["chunk"],
                      float(r.get("dataset") == "math")] for r in rows], float)


def cmd_fit(a, proto) -> int:
    """P0: logistic on position features. P1: P0's features plus a linear probe on the
    standardised layer-20 activation, stacked from out-of-fold probe outputs, as the
    planning replay fits them. Training labels follow the protocol's truncation rule."""
    import pickle

    import information_audit as ia
    import replay_stopping_policy as rp
    from sklearn.linear_model import LogisticRegression

    from nlaast.analysis.information import _dense_logit, grouped_folds, nested_probe

    out = _new_dir(a.runs / "pilot_models")
    units, cfg = rp.load_units(a.train_runs)
    rule = proto["truncation_rule"]
    y = np.array([training_label(u, rule) for u in units])
    g = np.array([u["problem_id"] for u in units])
    X = ia.activations_for(units, cfg.dir / "acts", 20)
    pos = _pos(units)
    seed = int(proto["seeds"]["fit"])
    folds = grouped_folds(y, g, 10, seed)
    extras, cs = nested_probe(X, y, g, folds, seed=seed)
    c_final = float(np.median(cs))
    probe = _dense_logit(c_final).fit(X, y)          # the same model family as Stage 1's probe
    oof_probe = np.full(len(y), np.nan)
    for k, (_, te) in enumerate(folds):
        oof_probe[te] = extras[k][1][:, 0]            # logit of the out-of-fold probe score
    p0 = LogisticRegression(max_iter=5000).fit(pos, y)
    p1 = LogisticRegression(max_iter=5000).fit(np.column_stack([pos, oof_probe]), y)
    path = out / "models.pkl"
    tmp = out / "models.pkl.partial"          # renamed only once complete
    with open(tmp, "wb") as fh:
        pickle.dump({"p0": p0, "p1_stack": p1, "probe": probe, "probe_C": c_final,
                     "features": ["log_prefix_tokens", "chunk", "is_math"],
                     "training": f"rep1 Stage 1, {len(units):,} units", "label_rule": rule}, fh)
    tmp.replace(path)
    (out / "models.json").write_text(json.dumps({
        "sha256": sha256(path), "probe_C": c_final, "n_units": len(units),
        "label_rule": rule, "positive_rate": float(y.mean()),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
    print(json.dumps({"models": str(path), "sha256": sha256(path)}))
    return 0


def _models(runs: Path):
    import pickle
    meta = json.loads((runs / "pilot_models" / "models.json").read_text(encoding="utf-8"))
    path = runs / "pilot_models" / "models.pkl"
    if sha256(path) != meta["sha256"]:
        raise SystemExit("the frozen models changed after fitting")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def cmd_score(a, proto) -> int:
    import information_audit as ia

    run_dir = a.runs / a.run
    rows = boundary_rows(run_dir)
    m = _models(a.runs)
    X = ia.activations_for(rows, run_dir / "acts", 20)
    pos = _pos(rows)
    probe_logit = logit(m["probe"].predict_proba(X)[:, 1])   # as stacked when fitting
    s0 = m["p0"].predict_proba(pos)[:, 1]
    s1 = m["p1_stack"].predict_proba(np.column_stack([pos, probe_logit]))[:, 1]
    path = run_dir / "pilot_scores.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r, a0, a1 in zip(rows, s0, s1):
            fh.write(json.dumps({**r, "P0_position": float(a0), "P1_same_label_probe": float(a1)})
                     + "\n")
    print(json.dumps({"scores": str(path), "n": len(rows)}))
    return 0


def _scored(run_dir: Path) -> list[dict]:
    from nlaast.logging_utils import read_jsonl
    return list(read_jsonl(run_dir / "pilot_scores.jsonl"))


# ---- calibrate and evaluate (CPU) -------------------------------------------


def cmd_calibrate(a, proto) -> int:
    from nlaast.analysis.stopping_pilot import build_problems, sgr_threshold, threshold_grid

    if (a.runs / "pilot_eval" / "pilot_forced.jsonl").exists():
        raise SystemExit("evaluation forcing already exists; calibration must come first")
    run_dir = a.runs / "pilot_cal"
    rows = _scored(run_dir)
    inst = instrument(run_dir, rows, proto["gates"])
    if not inst["passed"]:
        raise SystemExit(f"calibration instrument failed: {inst}")
    out = _new_dir(a.runs / "pilot_calibration")
    res = {"protocol_sha256": sha256(a.protocol), "instrument": inst, "policies": {}}
    for pol in POLICIES:
        probs = build_problems(rows, lambda r, pol=pol: r[pol], proto["truncation_rule"])
        grid = threshold_grid(np.array([r[pol] for r in rows]), int(proto["tau_grid_n"]))
        res["policies"][pol] = {**sgr_threshold(probs, grid, float(proto["alpha_cal"]),
                                                float(proto["delta"])), "n_problems": len(probs)}
    path = out / "calibration.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "calibration.sha256").write_text(sha256(path) + "\n", encoding="utf-8")
    print(json.dumps({k: v["tau"] for k, v in res["policies"].items()}))
    return 0


def cmd_evaluate(a, proto) -> int:
    from nlaast.analysis.stopping_pilot import (TRUNCATION_RULES, build_problems,
                                                paired_saving, safety_gate, summarise, walk)

    cal_path = a.runs / "pilot_calibration" / "calibration.json"
    recorded = (a.runs / "pilot_calibration" / "calibration.sha256").read_text().strip()
    if sha256(cal_path) != recorded:
        raise SystemExit("the calibration changed after it was frozen")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    run_dir = a.runs / "pilot_eval"
    rows = _scored(run_dir)
    inst = instrument(run_dir, rows, proto["gates"])
    out = _new_dir(a.runs / "pilot_evaluation")
    alpha, conf = float(proto["alpha"]), float(proto["eval_confidence"])
    rng = np.random.default_rng(int(proto["seeds"]["bootstrap"]))
    res = {"status": "exploratory pilot; not a safety certification (protocol §1)",
           "protocol_sha256": sha256(a.protocol), "calibration_sha256": recorded,
           "instrument": inst, "policies": {}, "sensitivity": {}}
    walks = {}
    for pol in POLICIES:
        tau = cal["policies"][pol]["tau"]
        probs = build_problems(rows, lambda r, pol=pol: r[pol], proto["truncation_rule"])
        if tau is None:
            res["policies"][pol] = {"tau": None, "reading": "no calibrated threshold: never stops"}
            continue
        w = walk(probs, tau)
        walks[pol] = w
        res["policies"][pol] = {"tau": tau, **summarise(w, conf),
                                "safety_gate": safety_gate(w, alpha, conf)}
        for rule in TRUNCATION_RULES:
            alt = walk(build_problems(rows, lambda r, pol=pol: r[pol], rule), tau)
            res["sensitivity"].setdefault(pol, {})[rule] = summarise(alt, conf)
    if len(walks) == 2:
        res["P1_minus_P0_net_saving"] = paired_saving(walks["P1_same_label_probe"],
                                                      walks["P0_position"], 2000, rng)
    path = out / "evaluation.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v.get("selective_risk"), v.get("coverage"))
                      for k, v in res["policies"].items()}))
    return 0


def cmd_check(a, proto) -> int:
    check_configs(proto)
    print(json.dumps({"protocol": str(a.protocol), "sha256": sha256(a.protocol),
                      "status": proto["status"]}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "force", "fit", "score", "calibrate", "evaluate"])
    ap.add_argument("--run", choices=["pilot_cal", "pilot_eval"])
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--train-runs", type=Path, default=ROOT / "runs",
                    help="where rep1 and rep1_stage1 are stored (read only)")
    a = ap.parse_args()
    proto = load_protocol(a.protocol)
    if a.command in ("force", "score") and not a.run:
        ap.error(f"{a.command} needs --run")
    return {"check": cmd_check, "force": cmd_force, "fit": cmd_fit, "score": cmd_score,
            "calibrate": cmd_calibrate, "evaluate": cmd_evaluate}[a.command](a, proto)


if __name__ == "__main__":
    sys.exit(main())
