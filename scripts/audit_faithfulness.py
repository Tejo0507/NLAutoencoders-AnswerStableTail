"""Stage `faithfulness`: the claim-level audit (O4 / RQ2).

Takes the verbalisations from the `nla` stage and, with the reconstructor
resident, runs every arm of the audit: delete, paraphrase (the null), resample,
the independent-probe control and the verbaliser-only control.

Only the reconstructor is loaded. The perturbations are textual, so no further
verbaliser sampling is needed - the resample arm reuses the alternative samples
the `nla` stage already generated for the same activation, which is both
cheaper and more honest, since those really are alternative descriptions of
that vector rather than of a perturbed one.

A claim counts as reconstruction-dependent only if deleting it moves
reconstruction further than the 95th percentile of the paraphrase null for the
same explanation. The null is what makes the threshold non-arbitrary.

This is a **RECAP-inspired** evaluation, not RECAP. RECAP co-trains the target
model with linear decodability heads and so cannot be applied retrospectively
to a released checkpoint.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.baselines.probe import CorrectnessProbe
from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.faithfulness.audit import audit_explanation, summarise
from nlaast.logging_utils import JsonlWriter, read_jsonl, write_json
from nlaast.models import loading
from nlaast.nla.reconstructor import ActivationReconstructor
from nlaast.trace.chunking import Chunk, prefix_text

STAGE = "faithfulness"


def build_independent_probe(cfg, log):
    """A probe trained only on activations, never on verbalised text.

    Its independence is the point: it is the second opinion in the RECAP-
    inspired design. It is fitted on the train split and used to score the
    activations behind the audited explanations, which come from the eval
    split, so it has never seen them.
    """
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer

    X, y = [], []
    for pid, t in traces.items():
        if t.get("split") != "train" or not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        chunks = [Chunk(**{k: c[k] for k in
                           ("index", "text", "char_start", "char_end",
                            "token_end", "prefix_tokens")})
                  for c in t.get("chunks", [])]
        for row_i, chunk_i in enumerate(meta.get("chunk_indices", [])):
            if chunk_i >= len(chunks):
                continue
            ans = extract_answer(prefix_text(chunks, chunk_i, t["trace"]), PERMISSIVE)
            if ans is None:
                continue
            X.append(arr[row_i])
            y.append(1 if equivalent(ans, t.get("gold")) else 0)

    if len(X) < 20 or len(set(y)) < 2:
        log.warning("independent probe unavailable: %d train examples, %d classes",
                    len(X), len(set(y)))
        return None, {"available": False, "n": len(X), "n_classes": len(set(y))}

    probe = CorrectnessProbe(cfg.probe).fit(np.vstack(X), np.asarray(y))
    return probe, {"available": True, "n": len(X), "positive_rate": float(np.mean(y))}


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    verbs = read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")
    if not verbs:
        log.error("no verbalisations - run scripts/run_autoencoder.py first")
        return 1
    if args.limit:
        verbs = verbs[: args.limit]

    manifest.start_stage(STAGE, n_verbalisations=len(verbs))
    out_dir = cfg.stage_dir(STAGE)
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer

    probe, probe_info = build_independent_probe(cfg, log)
    log.info("independent probe: %s", probe_info)

    ar_dir = loading.ensure_nf4(cfg.nla.ar_repo).local_path
    ar = ActivationReconstructor(cfg, ar_dir)

    writer = JsonlWriter(out_dir / "audits.jsonl", key="id")
    started = time.monotonic()
    audits = []

    try:
        with writer:
            for i, row in enumerate(verbs):
                pid = row["problem_id"]
                if not store.has(pid, layer):
                    continue
                arr, _ = store.get(pid, layer)
                gold = arr[row["activation_row"]]

                usable = [s for s in row["samples"]
                          if s.get("ok") and (s.get("explanation") or "").strip()]
                if not usable:
                    continue
                noise_texts = [
                    s.get("explanation") or s.get("text") or ""
                    for s in row.get("noise_samples", [])
                    if (s.get("explanation") or s.get("text") or "").strip()
                ]

                # Independent-probe score for this activation: one scalar per
                # window, not per claim. A correctness probe says whether the
                # *state* is decodably correct, not whether a particular
                # sentence about it is true, so attaching it to individual
                # claims would be a category error.
                probe_score = None
                if probe is not None:
                    probe_score = float(probe.predict_proba(gold[None, :])[0])

                for s in usable:
                    key = f"{row['id']}|s{s['sample_index']}"
                    if writer.has(key):
                        continue
                    alternatives = [
                        o.get("explanation") or ""
                        for o in usable if o["sample_index"] != s["sample_index"]
                    ]
                    audit = audit_explanation(
                        problem_id=pid,
                        window_kind=row["window_kind"],
                        sample_index=s["sample_index"],
                        explanation=s["explanation"],
                        activation=gold,
                        scorer=lambda text, act: ar.score(text, act).cosine,
                        cfg=cfg.faithfulness,
                        resample_alternatives=alternatives,
                        noise_explanations=noise_texts,
                    )
                    d = audit.to_dict()
                    d["id"] = key
                    d["chunk_index"] = row["chunk_index"]
                    d["independent_probe_score"] = probe_score
                    writer.write(d)
                    audits.append(audit)

                if (i + 1) % 10 == 0 or i + 1 == len(verbs):
                    rate = (time.monotonic() - started) / (i + 1)
                    log.info("[%d/%d] audited (eta %.1f min)",
                             i + 1, len(verbs), rate * (len(verbs) - i - 1) / 60)
    finally:
        ar.close()

    all_rows = read_jsonl(out_dir / "audits.jsonl")
    metrics = summarise_from_rows(all_rows)
    metrics["independent_probe"] = probe_info
    metrics["note"] = (
        "RECAP-inspired: adopts the principle of independent verification. "
        "It does not reproduce the RECAP training procedure, which requires "
        "co-training the target model."
    )
    write_json(out_dir / "summary.json", metrics)

    log.info("claims audited: %d across %d explanations",
             metrics.get("n_claims", 0), metrics.get("n_explanations", 0))
    log.info("reconstruction-dependent: %.3f | reproduced from noise: %.3f | "
             "dependent and not noise: %.3f",
             metrics.get("dependent_fraction", float("nan")),
             metrics.get("noise_reproduced_fraction", float("nan")),
             metrics.get("dependent_and_not_noise_fraction", float("nan")))

    manifest.finish_stage(STAGE, status="complete", output=str(out_dir), metrics=metrics)
    return 0


def summarise_from_rows(rows: list[dict]) -> dict:
    """Aggregate directly from the persisted records, so the summary can be
    recomputed after a resume without re-running the audit."""
    claims = [c for r in rows for c in r.get("claims", [])]
    if not claims:
        return {"n_explanations": len(rows), "n_claims": 0}

    dep = np.array([bool(c["reconstruction_dependent"]) for c in claims])
    noise = np.array([bool(c["reproduced_under_noise"]) for c in claims])
    eff = np.array([float(c["delete_effect"]) for c in claims])
    para = np.array([e for c in claims for e in c.get("paraphrase_effects", [])], dtype=float)
    resample = np.array([c["resample_effect"] for c in claims
                         if c.get("resample_effect") is not None], dtype=float)

    out = {
        "n_explanations": len(rows),
        "n_claims": len(claims),
        "mean_claims_per_explanation": len(claims) / len(rows),
        "mean_base_cosine": float(np.mean([r["base_cosine"] for r in rows])),
        "dependent_fraction": float(dep.mean()),
        "noise_reproduced_fraction": float(noise.mean()),
        "dependent_and_not_noise_fraction": float((dep & ~noise).mean()),
        "mean_delete_effect": float(np.mean(eff)),
        "mean_paraphrase_effect": float(np.mean(np.abs(para))) if para.size else float("nan"),
        "mean_resample_effect": float(np.mean(resample)) if resample.size else float("nan"),
    }
    by_kind = {}
    for kind in sorted({r["window_kind"] for r in rows}):
        sel = [r for r in rows if r["window_kind"] == kind]
        cl = [c for r in sel for c in r.get("claims", [])]
        if not cl:
            continue
        by_kind[kind] = {
            "n_explanations": len(sel),
            "n_claims": len(cl),
            "dependent_fraction": float(np.mean(
                [bool(c["reconstruction_dependent"]) for c in cl])),
            "noise_reproduced_fraction": float(np.mean(
                [bool(c["reproduced_under_noise"]) for c in cl])),
            "mean_base_cosine": float(np.mean([r["base_cosine"] for r in sel])),
        }
    out["by_window_kind"] = by_kind
    return out


if __name__ == "__main__":
    sys.exit(main())
