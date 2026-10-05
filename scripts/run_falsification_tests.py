"""Stage `robustness`: the falsification battery (F1-F10).

Every test here asks "what would show this interpretation is wrong?" and then
runs it. A test that fails is reported as a failure; the point is to find out
whether the result is true, not to make it look positive.

| id | alternative explanation under test |
|---|---|
| F1 | the readout adds nothing over trace length |
| F2 | it adds nothing over problem difficulty |
| F3 | the claims come from verbaliser priors, not the activation |
| F4 | the tail is an artefact of the answer parser |
| F5 | the tail is an artefact of its own definition |
| F6 | the probe leaks |
| F7 | layer 20 is special only by construction |
| F8 | 4-bit quantisation drives the result |
| F10 | the effect is about token position, not the tail |

**F8 must run while the bf16 target weights are still on disk.** It is the one
test with a hard ordering constraint: the three checkpoints cannot coexist on
this machine, and once the bf16 target is evicted to make room for the
verbaliser, re-measuring drift costs a 15 GB download. ``--only F8_quantisation``
runs it alone, which is how the orchestrator sequences it.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.analysis.stats import unpaired_test
from nlaast.baselines.probe import cross_validate
from nlaast.data.answers import PERMISSIVE, STRICT, equivalent, extract_answer
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.trace.chunking import Chunk, chunk_trace, prefix_text

STAGE = "robustness"


def chunks_of(trace):
    return [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                       "token_end", "prefix_tokens")})
            for c in trace.get("chunks", [])]


# --- F1 ---
def f1_length(cfg, traces, ast_rows, log) -> dict:
    """Does the tail just track trace length?"""
    rows = [(r, traces[r["problem_id"]]) for r in ast_rows
            if r["problem_id"] in traces and r.get("tail_start") is not None]
    if len(rows) < 5:
        return {"status": "skipped", "reason": f"only {len(rows)} tails"}
    lengths = np.array([t["n_tokens"] for _, t in rows], dtype=float)
    fracs = np.array([r["tail_fraction"] for r, _ in rows], dtype=float)
    tail_tokens = np.array([r["tail_tokens"] for r, _ in rows], dtype=float)
    r_frac = float(np.corrcoef(lengths, fracs)[0, 1]) if len(rows) > 2 else float("nan")
    r_tok = float(np.corrcoef(lengths, tail_tokens)[0, 1]) if len(rows) > 2 else float("nan")
    return {
        "status": "ran",
        "n": len(rows),
        "corr_length_vs_tail_fraction": r_frac,
        "corr_length_vs_tail_tokens": r_tok,
        "interpretation": (
            "A tail fraction strongly correlated with length would mean the "
            "construct is largely a length statistic. Absolute tail tokens "
            "correlating with length is expected and not itself a problem."
        ),
    }


# --- F2 ---
def f2_difficulty(cfg, traces, ast_rows, log) -> dict:
    """Does the tail track problem difficulty rather than redundancy?"""
    by_level: dict[str, list[float]] = {}
    by_dataset: dict[str, list[float]] = {}
    for r in ast_rows:
        if r.get("tail_start") is None:
            continue
        if r.get("level"):
            by_level.setdefault(str(r["level"]), []).append(r["tail_fraction"])
        by_dataset.setdefault(str(r.get("dataset")), []).append(r["tail_fraction"])
    out = {
        "status": "ran",
        "by_level": {k: {"n": len(v), "mean_tail_fraction": float(np.mean(v))}
                     for k, v in sorted(by_level.items())},
        "by_dataset": {k: {"n": len(v), "mean_tail_fraction": float(np.mean(v))}
                       for k, v in sorted(by_dataset.items())},
    }
    if len(by_dataset) == 2:
        a, b = list(by_dataset.values())
        if len(a) >= 3 and len(b) >= 3:
            out["dataset_test"] = unpaired_test(
                "F2: tail fraction by dataset", a, b, cfg.analysis.confidence
            ).to_dict()
    return out


# --- F3 ---
def f3_verbaliser_only(cfg, log) -> dict:
    """The Li et al. control, promoted to a headline number.

    If claims generated from Gaussian noise at matched norm reproduce the
    claims generated from real activations, the verbalisation reflects the
    verbaliser's priors and not the target model's state.
    """
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    if not audits and not recon:
        return {"status": "skipped", "reason": "no NLA output"}

    out: dict = {"status": "ran"}
    claims = [c for a in audits for c in a.get("claims", [])]
    if claims:
        noise = np.array([bool(c["reproduced_under_noise"]) for c in claims])
        dep = np.array([bool(c["reconstruction_dependent"]) for c in claims])
        out.update(
            n_claims=len(claims),
            reproduced_from_noise=float(noise.mean()),
            dependent=float(dep.mean()),
            dependent_and_not_noise=float((dep & ~noise).mean()),
            mean_noise_similarity=float(np.mean([c["noise_similarity"] for c in claims])),
        )
    real = [r["cosine"] for r in recon if r["window_kind"] != "gaussian_control"]
    ctrl = [r["cosine"] for r in recon if r["window_kind"] == "gaussian_control"]
    if len(real) >= 3 and len(ctrl) >= 3:
        out["reconstruction_test"] = unpaired_test(
            "F3: reconstruction cosine, real activation vs Gaussian control",
            real, ctrl, cfg.analysis.confidence,
        ).to_dict()
        out["mean_cosine_real"] = float(np.mean(real))
        out["mean_cosine_control"] = float(np.mean(ctrl))
    out["interpretation"] = (
        "The control must reconstruct the real activation substantially worse "
        "than the real verbalisation does. If it does not, the NLA arm is "
        "measuring the verbaliser, not the target model."
    )
    return out


# --- F4 ---
def f4_parser(cfg, traces, ast_rows, log) -> dict:
    """Re-parse every boundary with the stricter extractor and compare.

    Also re-segments with the alternative chunker, because the tail index is
    defined in chunks and a different segmentation is a different index.
    """
    agree = disagree = both_null = one_null = 0
    for t in traces.values():
        chunks = chunks_of(t)
        for i in range(len(chunks)):
            prefix = prefix_text(chunks, i, t["trace"])
            a = extract_answer(prefix, PERMISSIVE)
            b = extract_answer(prefix, STRICT)
            if a is None and b is None:
                both_null += 1
            elif a is None or b is None:
                one_null += 1
            elif equivalent(a, b):
                agree += 1
            else:
                disagree += 1

    final_agree = sum(
        1 for t in traces.values()
        if equivalent(extract_answer(t["trace"], PERMISSIVE),
                      extract_answer(t["trace"], STRICT))
    )
    alt_chunks = {}
    for t in traces.values():
        n_alt = len(chunk_trace(t["trace"], cfg.chunk,
                                token_offsets=[tuple(o) for o in t["offsets"]],
                                mode=cfg.chunk.alt_mode))
        alt_chunks[t["problem_id"]] = (t["n_chunks"], n_alt)

    total = agree + disagree + both_null + one_null
    return {
        "status": "ran",
        "n_boundaries": total,
        "both_parsers_agree": agree / total if total else float("nan"),
        "disagree": disagree / total if total else float("nan"),
        "strict_only_null": one_null / total if total else float("nan"),
        "both_null": both_null / total if total else float("nan"),
        "final_answer_agreement": final_agree / len(traces) if traces else float("nan"),
        "mean_chunks_primary": float(np.mean([v[0] for v in alt_chunks.values()])),
        "mean_chunks_alternative": float(np.mean([v[1] for v in alt_chunks.values()])),
        "interpretation": (
            "The strict parser deliberately refuses bare trailing numbers, so a "
            "high strict-only-null rate at intermediate boundaries is expected "
            "and is not disagreement. Disagreement on the *final* answer would "
            "be the real problem."
        ),
    }


# --- F5 ---
def f5_ast_sensitivity(cfg, log) -> dict:
    path = cfg.dir / "ast" / "sweep_F5.json"
    if not path.exists():
        return {"status": "skipped",
                "reason": "run scripts/detect_answer_stable_tail.py --sweep first"}
    sweep = read_json(path)
    rates = {k: v.get("tail_rate") for k, v in sweep.items()}
    fracs = {k: v.get("mean_tail_fraction") for k, v in sweep.items()}
    vals = [v for v in rates.values() if v is not None]
    return {
        "status": "ran",
        "tail_rate_by_setting": rates,
        "mean_tail_fraction_by_setting": fracs,
        "tail_rate_range": (max(vals) - min(vals)) if vals else float("nan"),
        "interpretation": (
            "A tail rate that swings widely across settings means the construct "
            "is an artefact of its own parameters rather than of the model."
        ),
    }


# --- F6 ---
def f6_probe_leakage(cfg, traces, log) -> dict:
    """Shuffle labels within problem; the probe must collapse to chance."""
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    X, y, groups = [], [], []
    for pid, t in traces.items():
        if not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        chunks = chunks_of(t)
        for row_i, ci in enumerate(meta.get("chunk_indices", [])):
            if ci >= len(chunks):
                continue
            ans = extract_answer(prefix_text(chunks, ci, t["trace"]), PERMISSIVE)
            if ans is None:
                continue
            X.append(arr[row_i])
            y.append(1 if equivalent(ans, t.get("gold")) else 0)
            groups.append(pid)
    if len(X) < 30 or len(set(y)) < 2:
        return {"status": "skipped", "reason": f"{len(X)} examples, {len(set(y))} classes"}

    X, y = np.vstack(X), np.asarray(y)
    real = cross_validate(X, y, groups, cfg.probe).to_dict()
    shuf = cross_validate(X, y, groups, cfg.probe, shuffle_labels=True).to_dict()
    return {
        "status": "ran",
        "n": len(y),
        "real_auc": real["auc"], "real_accuracy": real["accuracy"],
        "shuffled_auc": shuf["auc"], "shuffled_accuracy": shuf["accuracy"],
        "positive_rate": real["positive_rate"],
        "leak_suspected": bool(
            np.isfinite(shuf["auc"]) and abs(shuf["auc"] - 0.5) > 0.15
        ),
        "interpretation": (
            "Labels are permuted within problem, preserving per-problem class "
            "balance and destroying only the activation-label link. A shuffled "
            "AUC away from 0.5 means the probe is reading something it should "
            "not."
        ),
    }


# --- F7 ---
def f7_layer_sweep(cfg, traces, log) -> dict:
    """Probe accuracy at other layers.

    The autoencoder is layer-20-only and cannot be swept, so this tests the
    weaker claim: is layer 20 unusual for decodability, or just the layer the
    checkpoint happens to use?
    """
    store = ActivationStore(cfg.dir / "acts")
    out: dict = {"status": "ran", "layers": {}}
    for layer in cfg.robustness.layer_sweep:
        X, y, groups = [], [], []
        for pid, t in traces.items():
            if not store.has(pid, layer):
                continue
            arr, meta = store.get(pid, layer)
            chunks = chunks_of(t)
            for row_i, ci in enumerate(meta.get("chunk_indices", [])):
                if ci >= len(chunks):
                    continue
                ans = extract_answer(prefix_text(chunks, ci, t["trace"]), PERMISSIVE)
                if ans is None:
                    continue
                X.append(arr[row_i])
                y.append(1 if equivalent(ans, t.get("gold")) else 0)
                groups.append(pid)
        if len(X) < 30 or len(set(y)) < 2:
            out["layers"][str(layer)] = {"status": "unavailable", "n": len(X)}
            continue
        rep = cross_validate(np.vstack(X), np.asarray(y), groups, cfg.probe).to_dict()
        out["layers"][str(layer)] = {"n": len(y), "auc": rep["auc"],
                                     "accuracy": rep["accuracy"]}
    available = [v for v in out["layers"].values() if "auc" in v]
    if not available:
        out["status"] = "skipped"
        out["reason"] = ("only the target layer was extracted; re-run "
                         "scripts/extract_activations.py --layers 14 20 24")
    return out


# --- F8 ---
def f8_quantisation(cfg, traces, log, subsample: int) -> dict:
    """Does 4-bit quantisation move the activations the autoencoder sees?

    The released autoencoder was trained on bf16 activations. This study feeds
    it activations from a 4-bit target because three 7B checkpoints do not fit
    in 6.44 GB otherwise. That is a deviation, and this measures its size
    rather than assuming it away.

    The bf16 model is loaded truncated to the layers actually needed
    (``layer + 1``), which is the only way ~11 GB of weights fit beside the
    rest of the system in 15.6 GB of RAM.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from nlaast.models.loading import free_cuda, snapshot

    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    usable = [t for t in traces.values() if store.has(t["problem_id"], layer)]
    if not usable:
        return {"status": "skipped", "reason": "no stored activations"}
    usable = sorted(usable, key=lambda t: t["problem_id"])[:subsample]

    try:
        src = snapshot(cfg.target.repo_id, cfg.target.revision)
    except Exception as exc:
        return {"status": "blocked",
                "reason": f"bf16 weights unavailable: {exc!r}",
                "note": ("the bf16 source was evicted to make room for the "
                         "autoencoder checkpoints; re-running needs a fresh "
                         "15 GB download")}

    log.info("F8: loading bf16 target truncated to %d layers from %s", layer + 1, src)
    conf = AutoConfig.from_pretrained(src)
    conf.num_hidden_layers = layer + 1
    try:
        tok = AutoTokenizer.from_pretrained(src)
        bf16 = AutoModelForCausalLM.from_pretrained(
            src, config=conf, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="cpu",
        ).eval()
    except Exception as exc:
        log.exception("F8: bf16 load failed")
        return {"status": "failed", "reason": repr(exc),
                "note": "15.6 GB of RAM against ~11 GB of truncated bf16 weights"}

    captured: dict = {}

    def hook(_m, _i, o):
        captured["h"] = (o[0] if isinstance(o, tuple) else o).detach().float()

    handle = bf16.model.layers[layer].register_forward_hook(hook)
    cosines, norm_ratios, per_problem = [], [], []
    started = time.monotonic()
    try:
        for t in usable:
            pid = t["problem_id"]
            arr, meta = store.get(pid, layer)
            messages = [
                {"role": "system", "content": cfg.generation.system_prompt},
                {"role": "user", "content": t["question"]},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(prompt + t["trace"], return_tensors="pt",
                      add_special_tokens=False)["input_ids"]
            if ids.shape[1] > cfg.target.max_position_embeddings:
                continue
            with torch.inference_mode():
                bf16(input_ids=ids, use_cache=False)
            hidden = captured["h"][0]

            resolved = meta.get("resolved_positions") or []
            rows = []
            for i, p in enumerate(resolved):
                if p < hidden.shape[0]:
                    rows.append((i, p))
            if not rows:
                continue
            a = np.stack([arr[i] for i, _ in rows]).astype(np.float64)
            b = np.stack([hidden[p].numpy() for _, p in rows]).astype(np.float64)
            cs = (np.sum(a * b, axis=1)
                  / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))
            nr = np.linalg.norm(a, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)
            cosines.extend(cs.tolist())
            norm_ratios.extend(nr.tolist())
            per_problem.append({"problem_id": pid, "n": len(rows),
                                "mean_cosine": float(cs.mean())})
            log.info("F8: %s  n=%d  mean cos(nf4, bf16)=%.4f  (%.0fs elapsed)",
                     pid, len(rows), cs.mean(), time.monotonic() - started)
    finally:
        handle.remove()
        del bf16
        free_cuda()

    if not cosines:
        return {"status": "failed", "reason": "no comparable positions"}
    c = np.asarray(cosines)
    return {
        "status": "ran",
        "n_problems": len(per_problem),
        "n_vectors": int(c.size),
        "mean_cosine": float(c.mean()),
        "median_cosine": float(np.median(c)),
        "p05_cosine": float(np.percentile(c, 5)),
        "min_cosine": float(c.min()),
        "mean_norm_ratio_nf4_over_bf16": float(np.mean(norm_ratios)),
        "per_problem": per_problem,
        "interpretation": (
            "The verbaliser normalises its input to a fixed L2 norm, so only "
            "direction matters. A mean cosine near 1.0 means the 4-bit model "
            "presents the autoencoder with essentially the same directions it "
            "was trained on; a substantially lower value makes the NLA arm's "
            "inputs off-distribution and the arm's results unsafe to read."
        ),
    }


# --- F10 ---
def f10_position(cfg, log) -> dict:
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    if not recon:
        return {"status": "skipped", "reason": "no reconstructions"}
    out: dict = {"status": "ran", "by_window_kind": {}}
    for kind in sorted({r["window_kind"] for r in recon}):
        vals = [r["cosine"] for r in recon if r["window_kind"] == kind]
        out["by_window_kind"][kind] = {
            "n": len(vals), "mean_cosine": float(np.mean(vals)),
            "std_cosine": float(np.std(vals)),
        }
    tail = [r["cosine"] for r in recon if r["window_kind"] == "tail"]
    for other in ("matched_position", "matched_length"):
        vals = [r["cosine"] for r in recon if r["window_kind"] == other]
        if len(tail) >= 3 and len(vals) >= 3:
            out[f"tail_vs_{other}"] = unpaired_test(
                f"F10: tail vs {other}", tail, vals, cfg.analysis.confidence
            ).to_dict()
    out["interpretation"] = (
        "If the tail does not differ from a matched-position window, any "
        "reconstruction result on the tail is a result about looking late in a "
        "trace, not about the tail."
    )
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these tests (e.g. F8_quantisation)")
    args = p.parse_args()
    cfg, manifest, log = setup(args)

    wanted = list(args.only) if args.only else list(cfg.robustness.enabled_tests)
    if should_skip(manifest, STAGE, args.force, log) and not args.only:
        return 0

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    if not traces:
        log.error("no traces available")
        return 1

    out_dir = cfg.stage_dir(STAGE)
    path = out_dir / "robustness.json"
    results = read_json(path) if path.exists() else {}

    manifest.start_stage(STAGE, tests=wanted)
    runners = {
        "F1_length": lambda: f1_length(cfg, traces, ast_rows, log),
        "F2_difficulty": lambda: f2_difficulty(cfg, traces, ast_rows, log),
        "F3_verbaliser_only": lambda: f3_verbaliser_only(cfg, log),
        "F4_parser": lambda: f4_parser(cfg, traces, ast_rows, log),
        "F5_ast_sensitivity": lambda: f5_ast_sensitivity(cfg, log),
        "F6_probe_leakage": lambda: f6_probe_leakage(cfg, traces, log),
        "F7_layer_sweep": lambda: f7_layer_sweep(cfg, traces, log),
        "F8_quantisation": lambda: f8_quantisation(
            cfg, traces, log, cfg.robustness.quantisation_subsample),
        "F10_position": lambda: f10_position(cfg, log),
    }

    for name in wanted:
        fn = runners.get(name)
        if fn is None:
            log.warning("unknown robustness test %r", name)
            continue
        log.info("--- %s ---", name)
        try:
            results[name] = fn()
        except Exception as exc:
            log.exception("%s raised", name)
            results[name] = {"status": "failed", "reason": repr(exc)}
        log.info("%s: %s", name,
                 {k: v for k, v in results[name].items()
                  if k not in ("per_problem", "interpretation")})
        write_json(path, results)

    summary = {k: v.get("status") for k, v in results.items()}
    log.info("robustness summary: %s", summary)
    manifest.finish_stage(STAGE, status="complete", output=str(path),
                          metrics={"statuses": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
