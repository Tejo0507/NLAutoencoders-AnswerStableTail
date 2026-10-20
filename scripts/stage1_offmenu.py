"""D83 Stage 1: the NLA at positions chosen without the tail. GPU and CPU.

Implements ``docs/LEAKAGE_RESISTANT_DESIGN.md`` §3 on ``rep1``'s problems.
Subcommands, in order:

* ``positions`` (CPU): the first N problems of the frozen order with ``ok``
  traces, and K boundaries each, drawn by ``select_positions``. That
  function receives only the problem id and the number of chunks, and is
  the leak barrier: no AST, forced answer or continuation reaches it.
* ``verbalise``, ``reconstruct`` (GPU): the registered NLA phases
  (``run_autoencoder.run_verbalise`` / ``run_reconstruct``) on those
  positions, with window kind ``offmenu``.
* ``force`` (GPU): the D82 instrument at 256 tokens at every position. It
  records the 256-token answer and the answer of the first 48 generated
  tokens, plus a separate 48-token forcing at 50 seeded positions as the
  stability gate.
* ``analyse`` (CPU): Δ₁ = AUC(pos+A+E) − AUC(pos+A) on O256 safety, with the
  secondaries and the compute-charged policies.

All outputs go to ``runs/rep1_stage1/``. Nothing under ``runs/rep1/`` is
written.

    python scripts/stage1_offmenu.py positions
    python scripts/stage1_offmenu.py verbalise
    python scripts/stage1_offmenu.py reconstruct
    python scripts/stage1_offmenu.py force
    python scripts/stage1_offmenu.py analyse
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from _stage import load_config

from nlaast.logging_utils import JsonlWriter, read_jsonl, setup as setup_logging, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.seeding import derive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "rep1_stage1"
DESIGN = ROOT / "docs" / "LEAKAGE_RESISTANT_DESIGN.md"
#: D83 deviation 2 (power audit, D85): every ok rep1 problem, not the first 200.
N_PROBLEMS = 400
MIN_PROBLEMS = 300
#: Attrition gate: share of drawn positions that need a passing verbalisation.
MIN_TEXT_SHARE = 0.75
K = 4
POS_SEED = 20261020
BUDGET = 256
REG_BUDGET = 48
N_CHECK = 50
MARGIN = 0.01
KIND = "offmenu"


def select_positions(problem_id: str, n_chunks: int, k: int = K, seed: int = POS_SEED) -> list[int]:
    """K chunk boundaries uniformly without replacement, all if fewer.

    The only inputs are the problem id and the number of chunks (the leak
    barrier of design §3.2)."""
    if n_chunks <= k:
        return list(range(n_chunks))
    rng = np.random.default_rng(derive(seed, "offmenu", problem_id))
    return sorted(int(i) for i in rng.choice(n_chunks, size=k, replace=False))


def tost_reading(ci: list[float] | None, margin: float = MARGIN) -> str:
    """D83 deviation 2: superiority and equivalence read together from the 95% CI.

    The first version checked "probe better" before equivalence, so a
    significant but negligible negative Δ would have been called "probe
    better" with its whole CI inside the margin."""
    if ci is None:
        return "not evaluable"
    lo, hi = ci
    if lo > margin:
        return "practically relevant increment"
    if hi < -margin:
        return "probe practically better"
    if -margin < lo and hi < margin:
        if lo > 0:
            return "detectable but negligible increment (< 0.01 AUC)"
        if hi < 0:
            return "text slightly worse; no practically relevant difference"
        return "equivalent within ±0.01"
    return "inconclusive: the CI crosses the ±0.01 margin"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: Overridable only for code tests on a fixture (``--out-dir``, ``--config``).
CONFIG, RUN_ID = "replication", None


def _cfg():
    import argparse
    return load_config(argparse.Namespace(config=CONFIG, overrides=[], run_id=RUN_ID))


def cmd_positions(args) -> int:
    cfg = _cfg()
    out = OUT / "positions.jsonl"
    if out.exists():
        print(f"{out} exists; positions are drawn once", file=sys.stderr)
        return 1
    order = [json.loads(x)["id"] for x in (cfg.dir / "data" / "problems.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    # only these two fields are read from each trace
    n_chunks = {}
    for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl"):
        if t.get("ok"):
            n_chunks[t["problem_id"]] = int(t["n_chunks"])
    st = json.loads((cfg.dir / "manifest.json").read_text(encoding="utf-8"))["stages"].get("traces", {})
    if st.get("status") != "complete":
        print("the traces stage is not complete; positions are drawn once it is", file=sys.stderr)
        return 1
    # D83 deviation 2: every ok problem of the frozen order, up to N_PROBLEMS
    pids = [p for p in order if p in n_chunks][:N_PROBLEMS]
    if len(pids) < MIN_PROBLEMS:
        print(f"only {len(pids)} ok traces; below {MIN_PROBLEMS}, refusing", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"problem_id": p, "chunk": i} for p in pids for i in select_positions(p, n_chunks[p])]
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    rng = np.random.default_rng(POS_SEED)
    check = sorted(rng.choice(len(rows), size=N_CHECK, replace=False).tolist())
    write_json(OUT / "provenance.json", {
        "status": "D83 Stage 1", "created": utcnow(), "git": git_commit(),
        "design_sha256": _sha(DESIGN), "n_problems": len(pids), "k": K, "seed": POS_SEED,
        "n_positions": len(rows), "positions_sha256": _sha(out), "check_rows": check,
        "traces_sha256": _sha(cfg.dir / "traces" / "traces.jsonl")})
    print(f"{len(rows)} positions in {len(pids)} problems; sha {_sha(out)}")
    return 0


def _tasks(cfg):
    from nlaast.activations.store import ActivationStore
    store = ActivationStore(cfg.dir / "acts")
    tasks = []
    for r in read_jsonl(OUT / "positions.jsonl"):
        _, meta = store.get(r["problem_id"], cfg.target.layer)
        row = {int(c): j for j, c in enumerate(meta.get("chunk_indices", []))}.get(r["chunk"])
        if row is None:
            raise SystemExit(f"{r['problem_id']}|{r['chunk']}: no stored activation")
        tasks.append({"problem_id": r["problem_id"], "window_kind": KIND,
                      "chunk_index": r["chunk"], "row": row})
    return store, tasks


def cmd_verbalise(args) -> int:
    import run_autoencoder as ra
    cfg = _cfg()
    log = setup_logging(OUT)
    store, tasks = _tasks(cfg)
    return 0 if ra.run_verbalise(cfg, tasks, store, cfg.target.layer,
                                 OUT / "verbalisations.jsonl", log) else 1


def cmd_reconstruct(args) -> int:
    import run_autoencoder as ra
    cfg = _cfg()
    log = setup_logging(OUT)
    store, _ = _tasks(cfg)
    m = ra.run_reconstruct(cfg, store, cfg.target.layer, OUT / "verbalisations.jsonl",
                           OUT / "reconstructions.jsonl", log)
    write_json(OUT / "reconstruct_summary.json", m)
    return 0


def cmd_force(args) -> int:
    from force_all_boundaries import block_size, blocks

    from nlaast.data.answers import PERMISSIVE, extract_answer
    from nlaast.models.target import TargetModel
    from nlaast.trace.chunking import Chunk, prefix_text

    cfg = _cfg()
    log = setup_logging(OUT)
    gcfg = cfg.generation
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")}
    pos = read_jsonl(OUT / "positions.jsonl")
    check = set(json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))["check_rows"])
    by_p: dict[str, list[int]] = {}
    for r in pos:
        by_p.setdefault(r["problem_id"], []).append(r["chunk"])
    chk_p: dict[str, list[int]] = {}
    for j in check:
        chk_p.setdefault(pos[j]["problem_id"], []).append(pos[j]["chunk"])
    chunks_of = {p: [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                                "token_end", "prefix_tokens")})
                     for c in traces[p]["chunks"]] for p in by_p}
    blk = block_size(gcfg.batch_size)
    model = TargetModel(cfg)
    tok = model.tokenizer
    writer = JsonlWriter(OUT / "forced.jsonl", key="id")

    def prompt(p, i):
        t = traces[p]
        return model.build_prompt(t["question"], prefix_text(chunks_of[p], i, t["trace"])
                                  + gcfg.force_answer_suffix)

    work = [(purpose, budget, p, b) for purpose, budget, sel in
            (("ext", BUDGET, by_p), ("check48", REG_BUDGET, chk_p))
            for p in sorted(sel) for b in blocks(sel[p], blk)]
    t0 = time.monotonic()
    try:
        with writer:
            for n, (purpose, budget, p, b) in enumerate(work):
                todo = [i for i in b if not writer.has(f"{p}|{i}|{purpose}")]
                if not todo:
                    continue
                gens = model.generate([prompt(p, i) for i in todo], max_new_tokens=budget,
                                      temperature=0.0, top_p=1.0, do_sample=False,
                                      seed=derive(cfg.seed, "force", p, todo[0]),
                                      batch_size=len(todo))
                for i, g in zip(todo, gens):
                    text = gcfg.force_answer_suffix + g.text
                    first48 = gcfg.force_answer_suffix + tok.decode(g.token_ids[:REG_BUDGET],
                                                                     skip_special_tokens=True)
                    writer.write({"id": f"{p}|{i}|{purpose}", "problem_id": p, "chunk": i,
                                  "purpose": purpose, "budget": budget,
                                  "answer": extract_answer(text, PERMISSIVE),
                                  "answer_first48": extract_answer(first48, PERMISSIVE),
                                  "finished": bool(g.finished), "n_tokens": int(g.n_tokens),
                                  "finished_within_48": bool(g.finished) and g.n_tokens <= REG_BUDGET,
                                  "text": text,
                                  "prefix_tokens": chunks_of[p][i].prefix_tokens})
                if (n + 1) % 20 == 0:
                    log.info("%d/%d blocks, %.1fs per block", n + 1, len(work),
                             (time.monotonic() - t0) / (n + 1))
    finally:
        model.close()
    return 0


def build_units(cfg):
    """Every position, with features, texts and labels (design §3.3-3.4)."""
    from nlaast.data.answers import equivalent

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")}
    ts = {r["problem_id"]: r.get("tail_start") for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    forced = {(r["problem_id"], r["chunk"]): r for r in read_jsonl(OUT / "forced.jsonl")
              if r["purpose"] == "ext"}
    expl: dict[tuple[str, int], list[str]] = {}
    cos: dict[tuple[str, int], list[float]] = {}
    for r in read_jsonl(OUT / "reconstructions.jsonl"):
        if r["window_kind"] != KIND or not r.get("integrity_ok"):
            continue
        k = (r["problem_id"], int(r["chunk_index"]))
        expl.setdefault(k, []).append(r.get("explanation") or "")
        cos.setdefault(k, []).append(float(r["cosine"]))
    units = []
    for r in read_jsonl(OUT / "positions.jsonl"):
        p, i = r["problem_id"], r["chunk"]
        t = traces[p]
        f = forced.get((p, i))
        if f is None:
            raise SystemExit(f"{p}|{i}: no forced answer; the run is incomplete")
        final_ok = equivalent(t.get("final_answer"), t.get("gold"))
        safe = lambda a: equivalent(a, t.get("gold")) == final_ok  # noqa: E731
        units.append({
            "problem_id": p, "chunk": i, "dataset": t.get("dataset"),
            "prefix_tokens": int(t["chunks"][i]["prefix_tokens"]), "total_tokens": int(t["n_tokens"]),
            "E": "\n".join(expl.get((p, i), [])), "n_ok_samples": len(expl.get((p, i), [])),
            "readout": float(np.mean(cos[(p, i)])) if (p, i) in cos else None,
            "Y_ext": safe(f["answer"]),
            "Y_ext_u": safe(f["answer"]) if f["finished"] else safe(None),
            "Y_48": safe(f["answer_first48"]),
            "Y_tail": None if ts.get(p) is None else bool(i >= ts[p]),
            "forced_tokens": f["n_tokens"], "forced_finished": f["finished"],
        })
    return units


def cmd_analyse(args) -> int:
    import information_audit as ia

    from nlaast.analysis.information import (
        Design, compare, grouped_folds, nested_probe, out_of_fold, problem_draws,
        reading, weighted_auc, weights_for,
    )
    from nlaast.analysis.o3_registered import holm  # noqa: F401  (single primary; kept for parity)

    cfg = _cfg()
    out_path = OUT / "analysis.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite", file=sys.stderr)
        return 1
    prov = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    if _sha(OUT / "positions.jsonl") != prov["positions_sha256"]:
        raise SystemExit("positions.jsonl changed since it was drawn")
    units = build_units(cfg)
    chk = [r for r in read_jsonl(OUT / "forced.jsonl") if r["purpose"] == "check48"]
    ext = {(r["problem_id"], r["chunk"]): r for r in read_jsonl(OUT / "forced.jsonl")
           if r["purpose"] == "ext"}
    from nlaast.data.answers import equivalent
    stab = np.mean([equivalent(c["answer"], ext[(c["problem_id"], c["chunk"])]["answer_first48"])
                    or (c["answer"] is None and ext[(c["problem_id"], c["chunk"])]["answer_first48"] is None)
                    for c in chk]) if chk else None
    with_text = sum(u["n_ok_samples"] > 0 for u in units)
    out = {"status": "D83 Stage 1 (pre-specified)", "created": utcnow(), "git": git_commit(),
           "design_sha256": _sha(DESIGN), "positions_sha256": prov["positions_sha256"],
           "n_units": len(units), "n_problems": len({u["problem_id"] for u in units}),
           "n_units_with_passing_verbalisation": with_text,
           "instrument": {"first48_vs_separate_48_agreement": stab, "gate": 0.95,
                          "still_cut_off_at_256": float(np.mean([not u["forced_finished"] for u in units]))},
           "label_rates": {k: float(np.mean([u[k] for u in units if u[k] is not None]))
                           for k in ("Y_ext", "Y_ext_u", "Y_48", "Y_tail")}}
    reasons = []
    if stab is None or stab < 0.95:
        reasons.append("stability gate failed")
    if with_text < MIN_TEXT_SHARE * len(units):
        reasons.append(f"attrition: {with_text} of {len(units)} units have a passing "
                       f"verbalisation, below {MIN_TEXT_SHARE:.0%}")
    X = ia.activations_for(units, cfg.dir / "acts", cfg.target.layer)
    keys, counts = problem_draws([u["problem_id"] for u in units], 2000, np.random.default_rng(POS_SEED))
    pos = np.array([[np.log(max(1, u["prefix_tokens"])), u["chunk"], float(u["dataset"] == "math")]
                    for u in units])
    rd = np.array([[0.0 if u["readout"] is None else u["readout"], float(u["readout"] is None)]
                   for u in units])
    E = [u["E"] for u in units]
    for lab in ("Y_ext", "Y_48", "Y_tail", "Y_ext_u"):
        keep = np.array([u[lab] is not None for u in units])
        y = np.array([bool(u[lab]) for u in units])[keep]
        g = np.array([u["problem_id"] for u in units])[keep]
        idx = np.flatnonzero(keep)
        folds = grouped_folds(y, g, 10, POS_SEED)
        extras, Cs = nested_probe(X[idx], y, g, folds, seed=POS_SEED)
        Es = [E[k] for k in idx]
        oof = {"pos": out_of_fold(Design(pos[idx]), y, folds),
               "pos+A": out_of_fold(Design(pos[idx]), y, folds, extras),
               "pos+E": out_of_fold(Design(pos[idx], (Es,)), y, folds),
               "pos+A+E": out_of_fold(Design(pos[idx], (Es,)), y, folds, extras),
               "pos+A+readout": out_of_fold(Design(np.c_[pos[idx], rd[idx]]), y, folds, extras)}
        W = weights_for(g, keys, counts)
        res = {"n_units": int(len(y)), "positive_rate": float(y.mean()), "chosen_C": Cs,
               "auc": {m: weighted_auc(y, v) for m, v in oof.items()},
               "contrasts": {
                   "Δ1 E over pos+A": compare(y, oof["pos+A+E"], oof["pos+A"], W),
                   "E over pos": compare(y, oof["pos+E"], oof["pos"], W),
                   "A over pos": compare(y, oof["pos+A"], oof["pos"], W),
                   "readout over pos+A": compare(y, oof["pos+A+readout"], oof["pos+A"], W),
                   "pos+E minus pos+A": compare(y, oof["pos+E"], oof["pos+A"], W)}}
        if lab == "Y_ext":
            c = res["contrasts"]["Δ1 E over pos+A"]
            p = c["p_bootstrap"]
            verdict = ("inconclusive: " + "; ".join(reasons)) if reasons else tost_reading(c["ci95"])
            se = (c["ci95"][1] - c["ci95"][0]) / 3.92
            out["primary"] = {"delta": c["delta_auc"], "ci95": c["ci95"], "p": p, "reading": verdict,
                              "underpowered_flag": bool(se > 0.005),
                              "d81_style_reading": reading(c["ci95"], p < 0.05, c["delta_auc"], MARGIN)}
            rob = []
            for s in (POS_SEED + 1, POS_SEED + 2, POS_SEED + 3, POS_SEED + 4, POS_SEED + 5):
                f2 = grouped_folds(y, g, 10, s)
                e2, _ = nested_probe(X[idx], y, g, f2, seed=s)
                rob.append(weighted_auc(y, out_of_fold(Design(pos[idx], (Es,)), y, f2, e2))
                           - weighted_auc(y, out_of_fold(Design(pos[idx]), y, f2, e2)))
            out["primary"]["cv_seed_robustness"] = rob
        out[lab] = res
    write_json(out_path, out)
    print(json.dumps({"primary": out["primary"], "instrument": out["instrument"],
                      "label_rates": out["label_rates"]}, indent=1, default=str))
    return 0


def main() -> int:
    import argparse
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["positions", "verbalise", "reconstruct", "force", "analyse"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", default=None, help="code tests only: a fixture directory")
    p.add_argument("--config", default="replication", help="code tests only")
    p.add_argument("--run-id", default=None, help="code tests only")
    args = p.parse_args()
    global OUT, CONFIG, RUN_ID
    if args.out_dir:
        OUT = Path(args.out_dir)
    CONFIG, RUN_ID = args.config, args.run_id
    return {"positions": cmd_positions, "verbalise": cmd_verbalise, "reconstruct": cmd_reconstruct,
            "force": cmd_force, "analyse": cmd_analyse}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
