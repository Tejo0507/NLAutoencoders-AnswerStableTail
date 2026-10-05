"""Stage `nla`: run the released autoencoder on tail and control windows (O2).

Two passes, because the verbaliser and the reconstructor are each a 7B-class
model and only one fits in 6.44 GB at a time:

1. **Verbalise.** Load the AV, generate descriptions for every selected
   activation window, and for the verbaliser-only control (a Gaussian vector at
   matched L2 norm, per Li et al.). Unload.
2. **Reconstruct.** Load the AR, map every description back to a vector and
   score it against the original activation. Unload.

Windows. For each problem with a detected tail: ``windows_per_problem``
positions spread across the tail, plus - when enabled - the matched-position
and matched-length control windows. The controls are what make falsification
test F10 possible: without them, a reconstruction result on the tail could
simply be a result about late token positions.

Integrity. The verbaliser's documented off-distribution failure is CJK output
instead of an English ``<explanation>``. Because the activations here come from
a 4-bit target model, every sample is gated and the failure rate is reported.
The round-trip fidelity is compared against the 0.752 in-distribution value on
the checkpoint card; a large shortfall means the NLA arm is compromised and
says so rather than being quietly reported as a finding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.logging_utils import JsonlWriter, read_jsonl, write_json, write_jsonl
from nlaast.models import loading
from nlaast.nla.reconstructor import ActivationReconstructor, fraction_variance_explained
from nlaast.nla.verbalizer import ActivationVerbalizer
from nlaast.seeding import derive, rng

STAGE = "nla"


def select_windows(cfg, ast_row: dict, n_rows: int) -> list[tuple[str, int]]:
    """``(window_kind, chunk_index)`` pairs to verbalise for one problem.

    Tail positions are spread over the tail rather than clustered at its start,
    so a reconstruction difference cannot be an artefact of always reading the
    same relative offset.
    """
    out: list[tuple[str, int]] = []
    windows = ast_row.get("windows") or {}
    k = max(1, cfg.nla.windows_per_problem)

    tail = windows.get("tail")
    if tail:
        lo, hi = int(tail[0]), int(tail[1])
        if hi >= lo:
            picks = np.unique(np.linspace(lo, hi, min(k, hi - lo + 1)).round().astype(int))
            out += [("tail", int(i)) for i in picks]

    if cfg.nla.verbalise_controls:
        for kind in ("matched_position", "matched_length"):
            w = windows.get(kind)
            if not w:
                continue
            lo, hi = int(w[0]), int(w[1])
            if hi >= lo:
                picks = np.unique(np.linspace(lo, hi, min(k, hi - lo + 1)).round().astype(int))
                out += [(kind, int(i)) for i in picks]

    return [(kind, i) for kind, i in out if 0 <= i < n_rows]


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--phase", choices=["verbalise", "reconstruct", "both"],
                   default="both", help="run one phase only (each loads a 7B model)")
    args = p.parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    if not ast_rows:
        log.error("no AST results - run scripts/detect_answer_stable_tail.py first")
        return 1
    with_tail = [r for r in ast_rows if r.get("tail_start") is not None]
    log.info("%d/%d problems have a detected tail", len(with_tail), len(ast_rows))
    cap = args.limit or (cfg.nla.max_problems or None)
    if cap and len(with_tail) > cap:
        # Deterministic prefix by problem id, so a resumed or extended run
        # keeps the same NLA subset rather than drifting to new problems.
        with_tail = sorted(with_tail, key=lambda r: r["problem_id"])[:cap]
        log.info("NLA arm capped at %d problems (nla.max_problems); the "
                 "behavioural arms keep the full set", len(with_tail))
    if not with_tail:
        manifest.finish_stage(STAGE, status="skipped",
                              metrics={"reason": "no problem has a detected tail"})
        return 0

    manifest.start_stage(STAGE, n_problems=len(with_tail))
    out_dir = cfg.stage_dir(STAGE)
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer

    # Build the work list once so both phases agree on it exactly.
    tasks = []
    for row in with_tail:
        pid = row["problem_id"]
        if not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        chunk_to_row = {c: i for i, c in enumerate(meta.get("chunk_indices", []))}
        for kind, chunk_i in select_windows(cfg, row, arr.shape[0]):
            r = chunk_to_row.get(chunk_i)
            if r is None:
                continue
            tasks.append({"problem_id": pid, "window_kind": kind,
                          "chunk_index": chunk_i, "row": r})
    log.info("%d activation windows to verbalise", len(tasks))
    if not tasks:
        manifest.finish_stage(STAGE, status="skipped",
                              metrics={"reason": "no activation windows available"})
        return 0

    verb_path = out_dir / "verbalisations.jsonl"
    if args.phase in ("verbalise", "both"):
        if not run_verbalise(cfg, tasks, store, layer, verb_path, log):
            manifest.finish_stage(STAGE, status="failed",
                                  error="verbalisation phase failed")
            return 1

    recon_path = out_dir / "reconstructions.jsonl"
    metrics: dict = {}
    if args.phase in ("reconstruct", "both"):
        metrics = run_reconstruct(cfg, store, layer, verb_path, recon_path, log)
        write_json(out_dir / "summary.json", metrics)

    manifest.finish_stage(STAGE, status="complete", output=str(out_dir), metrics=metrics)
    return 0


def run_verbalise(cfg, tasks, store, layer, out_path: Path, log) -> bool:
    """Phase 1: AV resident, generate every description."""
    writer = JsonlWriter(out_path, key="id")
    todo = [t for t in tasks
            if not writer.has(f"{t['problem_id']}|{t['window_kind']}|{t['chunk_index']}")]
    log.info("verbalise: %d windows, %d remaining", len(tasks), len(todo))
    if not todo:
        return True

    av_dir = loading.ensure_nf4(cfg.nla.av_repo).local_path
    try:
        av = ActivationVerbalizer(cfg, av_dir)
    except Exception:
        log.exception("could not load the activation verbaliser")
        return False

    started = time.monotonic()
    n_ok = n_bad = 0
    # Verbalising several activations in one forward pass is the difference
    # between hours and days at this scale: single-stream decoding on this GPU
    # runs at ~20 tok/s, batched at ~45.
    chunk = max(1, cfg.generation.batch_size)
    try:
        with writer:
            for start in range(0, len(todo), chunk):
                group = todo[start : start + chunk]
                vectors = np.stack([store.get(t["problem_id"], layer)[0][t["row"]]
                                    for t in group])
                seeds = [derive(cfg.nla.seed, t["problem_id"], t["window_kind"],
                                t["chunk_index"]) for t in group]

                samples = av.verbalise(vectors, seed=seeds[0] % (2**31),
                                       n_samples=cfg.nla.n_samples,
                                       batch_size=len(group))

                # Verbaliser-only control (Li et al.): same prompt, same norm,
                # Gaussian vector. Claims surviving this are the verbaliser's
                # priors, not information about the target activation.
                noise = [[] for _ in group]
                if cfg.faithfulness.verbaliser_only_control:
                    g = av.gaussian_control(
                        vectors, rng(cfg.nla.seed, "noise", group[0]["problem_id"], start)
                    )
                    noise = av.verbalise(
                        g, seed=(seeds[0] + 7) % (2**31),
                        n_samples=cfg.faithfulness.n_verbaliser_only,
                        source="gaussian_control", batch_size=len(group),
                    )

                for j, t in enumerate(group):
                    ok = sum(1 for s in samples[j] if s.ok)
                    n_ok += ok
                    n_bad += len(samples[j]) - ok
                    writer.write({
                        "id": f"{t['problem_id']}|{t['window_kind']}|{t['chunk_index']}",
                        "problem_id": t["problem_id"],
                        "window_kind": t["window_kind"],
                        "chunk_index": t["chunk_index"],
                        "activation_row": t["row"],
                        "activation_norm": float(np.linalg.norm(vectors[j])),
                        "samples": [s.to_dict() for s in samples[j]],
                        "noise_samples": [s.to_dict() for s in noise[j]],
                        "n_ok": ok,
                    })
                done = start + len(group)
                rate = (time.monotonic() - started) / done
                log.info("[%d/%d] verbalised  ok=%d/%d so far  (eta %.1f min)",
                         done, len(todo), n_ok, n_ok + n_bad,
                         rate * (len(todo) - done) / 60)
    finally:
        av.close()

    total = n_ok + n_bad
    log.info("verbalisation integrity: %d/%d samples well-formed English (%.1f%%)",
             n_ok, total, 100 * n_ok / total if total else 0)
    if total and n_ok / total < 0.5:
        # Upstream's documented off-distribution signature. Surfaced loudly -
        # if the verbaliser is failing this often, nothing built on its output
        # is interpretable.
        log.error("more than half of verbalisations failed the integrity gate. "
                  "This is the off-distribution signature upstream documents; "
                  "the 4-bit target activations are the prime suspect (see F8).")
    return True


def run_reconstruct(cfg, store, layer, verb_path: Path, out_path: Path, log) -> dict:
    """Phase 2: AR resident, score every description against its activation."""
    rows = read_jsonl(verb_path)
    if not rows:
        log.error("no verbalisations to reconstruct")
        return {"error": "no verbalisations"}

    ar_dir = loading.ensure_nf4(cfg.nla.ar_repo).local_path
    ar = ActivationReconstructor(cfg, ar_dir)

    out, started = [], time.monotonic()
    try:
        for i, row in enumerate(rows):
            arr, _ = store.get(row["problem_id"], layer)
            gold = arr[row["activation_row"]]
            for s in row["samples"]:
                text = s.get("explanation") or s.get("text") or ""
                if not text.strip():
                    continue
                r = ar.score(text, gold)
                out.append({
                    "id": f"{row['id']}|s{s['sample_index']}",
                    "problem_id": row["problem_id"],
                    "window_kind": row["window_kind"],
                    "chunk_index": row["chunk_index"],
                    "sample_index": s["sample_index"],
                    "integrity_ok": s["ok"],
                    "explanation": text,
                    **r.to_dict(),
                })
            # The control is scored against the *same* gold activation: the
            # question is whether a noise-driven description reconstructs the
            # real vector, which is the floor the tail numbers sit above.
            for s in row.get("noise_samples", []):
                text = s.get("explanation") or s.get("text") or ""
                if not text.strip():
                    continue
                r = ar.score(text, gold)
                out.append({
                    "id": f"{row['id']}|noise{s['sample_index']}",
                    "problem_id": row["problem_id"],
                    "window_kind": "gaussian_control",
                    "chunk_index": row["chunk_index"],
                    "sample_index": s["sample_index"],
                    "integrity_ok": s["ok"],
                    "explanation": text,
                    **r.to_dict(),
                })
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                rate = (time.monotonic() - started) / (i + 1)
                log.info("[%d/%d] reconstructed (eta %.1f min)",
                         i + 1, len(rows), rate * (len(rows) - i - 1) / 60)
    finally:
        ar.close()

    write_jsonl(out_path, out)

    by_kind: dict[str, dict] = {}
    for kind in sorted({r["window_kind"] for r in out}):
        sel = [r for r in out if r["window_kind"] == kind]
        by_kind[kind] = {
            **fraction_variance_explained([r["mse"] for r in sel]),
            "mean_cosine_direct": float(np.mean([r["cosine"] for r in sel])),
        }

    real = [r for r in out if r["window_kind"] != "gaussian_control"]
    metrics = {
        "n_reconstructions": len(out),
        "by_window_kind": by_kind,
        "overall": fraction_variance_explained([r["mse"] for r in real]) if real else {},
        "reference_fve": cfg.nla.reference_fve,
        "integrity_ok_rate": float(np.mean([r["integrity_ok"] for r in out])) if out else 0.0,
    }
    fve = metrics["overall"].get("fve", float("nan"))
    log.info("reconstruction FVE %.3f against the %.3f in-distribution reference",
             fve, cfg.nla.reference_fve)
    for k, v in by_kind.items():
        log.info("  %-18s n=%-4d mean cosine %.3f  FVE %.3f",
                 k, v["n"], v["mean_cosine_direct"], v["fve"])
    if np.isfinite(fve) and fve < 0.5 * cfg.nla.reference_fve:
        log.error("round-trip fidelity is far below the checkpoint's reported "
                  "in-distribution value. Treat the NLA arm as compromised "
                  "until F8 (quantisation drift) is resolved.")
        metrics["fidelity_warning"] = True
    return metrics


if __name__ == "__main__":
    sys.exit(main())
