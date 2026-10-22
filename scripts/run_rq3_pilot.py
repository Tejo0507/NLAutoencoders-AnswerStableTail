"""RQ3 follow-up, stage 1 pilot: is the tail direction's effect specific? (D74) GPU; NOT RUN.

**Running this needs its own dated decision** (D74 records the configuration
and the go/no-go rules). ``--dry-run`` does everything that needs no model on
CPU: it fits the candidate, builds and checks both size-matched controls, and
reports the exact work and a cost range.

The pilot asks a *new* question, distinct from RQ3: is the tail computation
causally redundant along the candidate direction? It asks it on the 60
problems the registered causal arm already used, so it is **exploratory**. Its
only job is to measure the inputs a confirmatory run would need, chiefly the
answer-change rate of a control matched in edit size
(``docs/RQ3_FOLLOWUP_DESIGN.md``).

Changes from the registered causal stage, each a defect found on ``main`` (D74):

* The controls remove as much as the candidate does
  (``causal/controls.py``): variance-matched random directions and
  permuted-label difference-in-means directions, both projected out at the
  candidate's coefficient.
* Continuations may finish. They are capped at ``MAX_NEW_TOKENS``, not 200,
  and one that hits the cap is made to commit by the registered forcing call.
  The answer is never parsed out of a truncated text.
* Every row records its text, whether it finished, the norm actually removed
  per edited position, and a degeneration flag.

Reads ``runs/<run_id>/`` (traces, AST, activations, the registered causal
arm's problem list). Writes only ``--out-dir``, by default
``runs/<run_id>_rq3_pilot/``.

    python scripts/run_rq3_pilot.py --config main --dry-run
    python scripts/run_rq3_pilot.py --config main          # needs a decision
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from _stage import base_parser, load_config

from nlaast.activations.store import ActivationStore
from nlaast.causal.controls import (
    degenerate,
    mean_removed_norm,
    permuted_label_directions,
    position_direction,
    read_outcome,
    second_moment,
    variance_matched_directions,
    within_tolerance,
)
from nlaast.causal.interventions import intervention_spans, projection_editor
from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import JsonlWriter, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.seeding import derive
from nlaast.trace.chunking import prefix_text

#: The pilot's configuration, recorded in D74. Changing any of these is a new
#: decision, not a tweak.
CANDIDATE_DOSES = (0.0, 1.5, 2.0)
CONTROL_DOSES = (1.5, 2.0)
N_VARIANCE_MATCHED = 5
N_PERMUTED = 5
MAX_NEW_TOKENS = 768
SIZE_TOL = 0.10
SEED = 20261019
#: Single-stream throughput measured on this GPU (D5); the hook needs batch 1.
TOKENS_PER_SECOND = 20.0


def window_vectors(cfg, ast_rows, traces, store):
    """Train-split tail and matched-position vectors per problem, chosen as
    ``run_interventions.fit_tail_direction`` chooses them."""
    layer = cfg.target.layer
    tail, pre = {}, {}
    for row in ast_rows:
        pid = row["problem_id"]
        t = traces.get(pid)
        if t is None or t.get("split") != "train" or not store.has(pid, layer) or not row.get("windows"):
            continue
        w = row["windows"]
        if "tail" not in w or "matched_position" not in w:
            continue
        arr, meta = store.get(pid, layer)
        c2r = {c: i for i, c in enumerate(meta.get("chunk_indices", []))}
        for kind, bucket in (("tail", tail), ("matched_position", pre)):
            rows = [arr[c2r[c]] for c in range(int(w[kind][0]), int(w[kind][1]) + 1) if c in c2r]
            if rows:
                bucket[pid] = np.vstack(rows)
    keep = sorted(set(tail) & set(pre))
    return {p: tail[p] for p in keep}, {p: pre[p] for p in keep}


def calibration_matrix(cfg, traces, store, split):
    layer = cfg.target.layer
    rows = [store.get(p, layer)[0] for p, t in sorted(traces.items())
            if t.get("split") == split and store.has(p, layer)]
    return np.vstack(rows).astype(np.float64)


def build_directions(cfg, log):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_interventions import fit_tail_direction

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")}
    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    store = ActivationStore(cfg.dir / "acts")
    cand, info = fit_tail_direction(cfg, ast_rows, traces, store, log)
    u = cand / np.linalg.norm(cand)
    tail, pre = window_vectors(cfg, ast_rows, traces, store)
    # The registered estimator, recomputed from the same vectors.
    check = np.vstack(list(tail.values())).mean(0) - np.vstack(list(pre.values())).mean(0)
    if not np.allclose(check / np.linalg.norm(check), u, atol=1e-5):
        raise SystemExit("window vectors do not reproduce the registered candidate direction")
    rng = np.random.default_rng(SEED)
    X_train = calibration_matrix(cfg, traces, store, "train")
    X_eval = calibration_matrix(cfg, traces, store, "eval")
    vm, vm_info = variance_matched_directions(X_train, u, N_VARIANCE_MATCHED, rng, tol=SIZE_TOL)
    pl, pl_info = permuted_label_directions(tail, pre, N_PERMUTED, rng, X=X_train,
                                            candidate=u, tol=SIZE_TOL)
    pos = position_direction(*pre_tail_positions(cfg, ast_rows, traces, store))
    target_eval = second_moment(X_eval, u)

    def removed_share(v):
        return float(np.mean(np.abs(X_eval @ v) / np.linalg.norm(X_eval, axis=1)))

    checks = {
        "candidate": {"direction": info, "eval_second_moment": target_eval,
                      "eval_removed_norm_share": removed_share(u)},
        "variance_matched": {**vm_info,
                             "eval_ratio_to_candidate": [second_moment(X_eval, v) / target_eval for v in vm],
                             "eval_removed_norm_share": [removed_share(v) for v in vm]},
        "permuted_label": {**pl_info,
                           "eval_ratio_to_candidate": [second_moment(X_eval, v) / target_eval for v in pl],
                           "eval_removed_norm_share": [removed_share(v) for v in pl],
                           "cosine_with_candidate_after_orthogonalising": [float(v @ u) for v in pl]},
        "position_direction": {"cosine_with_candidate": float(pos @ u),
                               "eval_ratio_to_candidate": second_moment(X_eval, pos) / target_eval,
                               "eval_removed_norm_share": removed_share(pos),
                               "note": "secondary control; reported, not part of go/no-go"},
    }
    arms = [("ablate_dir", d, u, "tail_dim") for d in CANDIDATE_DOSES]
    for d in CONTROL_DOSES:
        arms += [("variance_matched", d, v, f"vm_{k}") for k, v in enumerate(vm)]
        arms += [("permuted_label", d, v, f"pl_{k}") for k, v in enumerate(pl)]
        arms.append(("position_direction", d, pos, "pos"))
    return traces, ast_rows, arms, checks


def pre_tail_positions(cfg, ast_rows, traces, store):
    """Train-split boundary activations strictly before each tail, with their
    relative positions: the inputs of ``position_direction``."""
    layer = cfg.target.layer
    ast = {r["problem_id"]: r for r in ast_rows}
    vecs, rels = {}, {}
    for pid, t in traces.items():
        if t.get("split") != "train" or not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        n = len(t["chunks"])
        ts = ast.get(pid, {}).get("tail_start")
        keep = [(i, c) for i, c in enumerate(meta.get("chunk_indices", []))
                if ts is None or c < ts]
        if len(keep) < 2:
            continue
        vecs[pid] = np.vstack([arr[i] for i, _ in keep])
        rels[pid] = np.array([c / max(1, n - 1) for _, c in keep])
    return vecs, rels


def targets_from_registered_arm(cfg, ast_rows):
    pids = sorted({r["problem_id"] for r in read_jsonl(cfg.dir / "causal" / "interventions.jsonl")})
    by = {r["problem_id"]: r for r in ast_rows}
    return [by[p] for p in pids if by.get(p, {}).get("tail_start") is not None]


#: Mean generated tokens per candidate dose on ``main`` (capped at 200, so a
#: floor), and the registered random control's mean.
_MAIN_MEAN_TOKENS = {0.0: 64.6, 0.5: 66.6, 1.0: 71.8, 1.5: 140.2, 2.0: 183.6}
_CONTROL_MEAN_TOKENS = 65.0


def cost_estimate(n_problems: int, arms: list) -> dict:
    """Token and time range for these arms. The lower end assumes controls
    behave like the registered random control and the candidate like main's
    capped run. The upper end assumes every generation runs to the cap. Forced
    answers after a cap add at most ``force_answer_max_new_tokens`` each."""
    low = n_problems * sum(_MAIN_MEAN_TOKENS.get(d, MAX_NEW_TOKENS) if a == "ablate_dir"
                           else _CONTROL_MEAN_TOKENS for a, d, _, _ in arms)
    high = n_problems * len(arms) * MAX_NEW_TOKENS
    return {"generations": n_problems * len(arms),
            "tokens_low": low, "tokens_high": high,
            "hours_low": low / TOKENS_PER_SECOND / 3600,
            "hours_high": high / TOKENS_PER_SECOND / 3600,
            "assumed_tokens_per_second": TOKENS_PER_SECOND}


def run(cfg, out_dir, traces, targets, arms, log):
    from nlaast.models.target import TargetModel
    from nlaast.trace.chunking import Chunk

    model = TargetModel(cfg)
    writer = JsonlWriter(out_dir / "pilot.jsonl", key="id")
    started = time.monotonic()
    try:
        with writer:
            for n, row in enumerate(targets):
                pid = row["problem_id"]
                t = traces[pid]
                chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                                     "token_end", "prefix_tokens")})
                          for c in t["chunks"]]
                ts = int(row["tail_start"])
                prefix = prefix_text(chunks, ts, t["trace"])
                prompt = model.build_prompt(t["question"], prefix)
                prompt_tokens = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                tail_tokens = max(1, int(row.get("tail_tokens") or 1))
                pre_len = chunks[ts - 1].prefix_tokens - chunks[max(0, ts - 1 - (len(chunks) - ts))].prefix_tokens
                span = intervention_spans(prompt_tokens, tail_tokens, pre_len)["tail"]
                for name, dose, vec, did in arms:
                    key = f"{pid}|{name}|{dose}|{did}"
                    if writer.has(key):
                        continue
                    edit = projection_editor(vec, dose, span)
                    edit.reset()
                    with model.patch_layer(edit):
                        gen = model.generate([prompt], max_new_tokens=MAX_NEW_TOKENS,
                                             temperature=0.0, top_p=1.0, do_sample=False,
                                             seed=derive(cfg.seed, name, pid, did), batch_size=1)[0]
                    text_answer = extract_answer(prefix + gen.text, PERMISSIVE)
                    forced = None
                    if not gen.finished:
                        forced = extract_answer(model.force_answer(t["question"], prefix + gen.text),
                                                PERMISSIVE)
                    answer, source = read_outcome(gen.finished, text_answer, forced)
                    writer.write({
                        "id": key, "problem_id": pid, "arm": name, "dose": dose, "direction_id": did,
                        "text": gen.text, "finished": bool(gen.finished), "n_tokens": gen.n_tokens,
                        "n_positions_edited": int(edit.n_positions_edited),
                        "mean_removed_norm": mean_removed_norm(edit.removed_norm_sum,
                                                               edit.n_positions_edited),
                        "answer": answer, "answer_source": source,
                        "baseline_answer": t.get("final_answer"),
                        "answer_changed": not equivalent(answer, t.get("final_answer")),
                        "degenerate": degenerate(gen.text)})
                log.info("[%d/%d] %s done (%.1f min elapsed)", n + 1, len(targets), pid,
                         (time.monotonic() - started) / 60)
    finally:
        model.close()


def summarise(rows: list[dict], reps: int = 10000, seed: int = SEED) -> dict:
    """Pilot metrics and the D74 go/no-go reading. Pure: rows in, dict out."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    out: dict = {"n_problems": len(by)}
    d0 = [r for r in rows if r["arm"] == "ablate_dir" and r["dose"] == 0.0]
    out["dose0_identity"] = (sum(not r["answer_changed"] for r in d0) / len(d0)) if d0 else None
    out["finished_share"] = {f"{a}@{d}": float(np.mean([r["finished"] for r in rows
                                                        if r["arm"] == a and r["dose"] == d]))
                             for a, d in sorted({(r["arm"], r["dose"]) for r in rows})}
    per_dose = {}
    for d in CONTROL_DOSES:
        diffs = {"variance_matched": [], "permuted_label": []}
        size_ok = {"variance_matched": [], "permuted_label": []}
        cand_rates, ctrl_rates = [], {"variance_matched": [], "permuted_label": []}
        degen_changes = []
        for pid, rs in by.items():
            c = [r for r in rs if r["arm"] == "ablate_dir" and r["dose"] == d]
            if not c:
                continue
            c = c[0]
            cand_rates.append(float(c["answer_changed"]))
            if c["answer_changed"]:
                degen_changes.append(bool(c["degenerate"]))
            for arm in diffs:
                ctl = [r for r in rs if r["arm"] == arm and r["dose"] == d]
                if not ctl:
                    continue
                m = float(np.mean([r["answer_changed"] for r in ctl]))
                ctrl_rates[arm].append(m)
                diffs[arm].append(float(c["answer_changed"]) - m)
                size_ok[arm].extend(within_tolerance([r["mean_removed_norm"] for r in ctl],
                                                     c["mean_removed_norm"], SIZE_TOL))
        entry = {"candidate_change_rate": float(np.mean(cand_rates)) if cand_rates else None,
                 "degenerate_share_of_candidate_changes":
                     float(np.mean(degen_changes)) if degen_changes else None}
        for arm, ds in diffs.items():
            if not ds:
                entry[arm] = None
                continue
            ds = np.array(ds)
            boot = [ds[rng.integers(0, len(ds), len(ds))].mean() for _ in range(reps)]
            entry[arm] = {"control_change_rate": float(np.mean(ctrl_rates[arm])),
                          "paired_difference": float(ds.mean()),
                          "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                          "size_matched_share": float(np.mean(size_ok[arm])) if size_ok[arm] else None}
        per_dose[str(d)] = entry
    out["per_dose"] = per_dose
    out["decision"] = go_no_go(out)
    return out


def go_no_go(s: dict) -> dict:
    """D74's rules, applied mechanically at dose 2.0."""
    reasons = []
    e = s["per_dose"].get("2.0") or {}
    if s.get("dose0_identity") is None or s["dose0_identity"] < 1.0:
        reasons.append("dose 0 did not reproduce the baseline answer on every problem")
    for arm in ("variance_matched", "permuted_label"):
        a = e.get(arm)
        if a is None:
            reasons.append(f"{arm}: no rows")
        elif a["size_matched_share"] is None or a["size_matched_share"] < 0.9:
            reasons.append(f"{arm}: realised edit size within ±10% on "
                           f"{a['size_matched_share']} of replicates (< 0.9)")
    if reasons:
        return {"verdict": "invalid manipulation", "reasons": reasons}
    ds = (e.get("degenerate_share_of_candidate_changes") or 0.0)
    worst = max(e[a]["control_change_rate"] for a in ("variance_matched", "permuted_label"))
    gap = min(e[a]["paired_difference"] for a in ("variance_matched", "permuted_label"))
    lo = min(e[a]["ci95"][0] for a in ("variance_matched", "permuted_label"))
    if ds >= 0.5:
        return {"verdict": "no-go", "reasons": [
            f"{ds:.0%} of candidate answer changes are degenerate continuations: "
            "the outcome measures disruption, not redundancy"]}
    if gap <= 0.05:
        return {"verdict": "no-go", "reasons": [
            f"candidate exceeds the size-matched controls by at most {gap:.3f} (≤ 0.05)"]}
    if worst <= 0.10 and lo > 0:
        return {"verdict": "go", "reasons": [
            f"gap ≥ {gap:.3f}, control rate ≤ {worst:.3f}; register the confirmatory run "
            "on the 83 held-out problems with N from this control rate"]}
    return {"verdict": "go only if N for power 0.8 is ≤ 240", "reasons": [
        f"gap {gap:.3f}, control rate {worst:.3f}, CI lower bound {lo:.3f}"]}


#: Stage 0's threshold: with completion allowed, at least this share of the
#: 60 problems must change answer at dose 2.0 for controls to be worth running.
STAGE0_MIN_CHANGE = 0.10


def stage0_decision(rows: list[dict]) -> dict:
    """Is there an answer effect at all, once continuations may finish? (D74)

    On ``main`` every dose-2.0 answer change was a continuation cut off at 200
    tokens. If, allowed to finish, the candidate changes fewer than
    ``STAGE0_MIN_CHANGE`` of answers, nothing remains whose specificity the
    controls could test.
    """
    d0 = [r for r in rows if r["arm"] == "ablate_dir" and r["dose"] == 0.0]
    d2 = [r for r in rows if r["arm"] == "ablate_dir" and r["dose"] == 2.0]
    if not d2 or not d0:
        return {"verdict": "incomplete", "n_dose0": len(d0), "n_dose2": len(d2)}
    identity = sum(not r["answer_changed"] for r in d0) / len(d0)
    rate = sum(r["answer_changed"] for r in d2) / len(d2)
    out = {"n_problems": len(d2), "dose0_identity": identity,
           "dose2_change_rate": rate,
           "dose2_finished_share": sum(r["finished"] for r in d2) / len(d2),
           "dose2_degenerate_share": sum(r["degenerate"] for r in d2) / len(d2),
           "dose2_mean_tokens": float(np.mean([r["n_tokens"] for r in d2]))}
    if identity < 1.0:
        out["verdict"] = "invalid: dose 0 did not reproduce the baseline"
    elif rate < STAGE0_MIN_CHANGE:
        out["verdict"] = ("no-go: allowed to finish, the candidate changes too few answers "
                          "for a specificity test")
    else:
        out["verdict"] = "go to stage 1: run the size-matched controls"
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--dry-run", action="store_true", help="CPU only: directions, checks, cost")
    p.add_argument("--stage0", action="store_true",
                   help="candidate only, doses 0 and 2.0: is there an effect at all? (D74)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--summarise", action="store_true", help="CPU: summarise existing rows")
    args = p.parse_args()
    cfg = load_config(args)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.dir.parent / f"{cfg.run_id}_rq3_pilot"
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("rq3_pilot")
    if args.summarise:
        rows = read_jsonl(out_dir / "pilot.jsonl")
        s = summarise(rows) if any(r["arm"] != "ablate_dir" for r in rows) else {}
        s["stage0"] = stage0_decision(rows)
        write_json(out_dir / "summary.json", s)
        print(json.dumps(s, indent=1))
        return 0
    traces, ast_rows, arms, checks = build_directions(cfg, log)
    if args.stage0:
        arms = [a for a in arms if a[0] == "ablate_dir" and a[1] in (0.0, 2.0)]
    targets = targets_from_registered_arm(cfg, ast_rows)
    if args.limit:
        # Smoke runs only: a limited run must write to its own --out-dir.
        if not args.out_dir:
            print("--limit needs an explicit --out-dir", file=sys.stderr)
            return 1
        targets = targets[: args.limit]
    plan = {"status": "dry run: nothing generated" if args.dry_run else "running",
            "stage": "0 (candidate only)" if args.stage0 else "1 (candidate and controls)",
            "created": utcnow(), "git": git_commit(), "seed": SEED,
            "configuration": {"candidate_doses": CANDIDATE_DOSES, "control_doses": CONTROL_DOSES,
                              "n_variance_matched": N_VARIANCE_MATCHED, "n_permuted_label": N_PERMUTED,
                              "max_new_tokens": MAX_NEW_TOKENS, "size_tolerance": SIZE_TOL,
                              "problems": [r["problem_id"] for r in targets]},
            "checks": checks,
            "cost": cost_estimate(len(targets), arms)}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / ("dry_run.json" if args.dry_run else "plan.json"), plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "configuration"}, indent=1, default=str))
    if args.dry_run:
        return 0
    run(cfg, out_dir, traces, targets, arms, log)
    rows = read_jsonl(out_dir / "pilot.jsonl")
    s = {} if args.stage0 else summarise(rows)
    s["stage0"] = stage0_decision(rows)
    write_json(out_dir / "summary.json", s)
    print(json.dumps({k: s[k] for k in ("stage0", "decision") if k in s}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
