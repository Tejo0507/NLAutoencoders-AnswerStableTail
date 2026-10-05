"""Stage `causal`: causal validation of the tail (O5 / RQ3).

Five arms, all with the target model resident:

* ``truncate`` - stop at the tail start and force an answer;
* ``filler`` - replace the tail text with neutral filler of matched token length;
* ``ablate_dir`` - project the candidate tail direction out of the layer-K
  residual stream over the tail span, swept over a dose coefficient;
* ``random_dir`` - the same edit with a random direction of matched norm,
  drawn in the orthogonal complement of the candidate;
* ``matched_position`` - the candidate direction applied to the
  pre-stabilisation window instead of the tail.

The candidate direction is a difference in means between tail and
matched-position activations, fitted on the **train split only**. Fitting it on
the problems it is evaluated on would make the result circular.

Zhang & Nanda's standard is implemented as a gate, not a caveat:
``direction_claim_supported`` requires the candidate to beat both controls and
to show a monotone dose-response before the result is reported as supporting a
tail direction. A partial result is reported as partial.

The review already scopes O5 as an extension contingent on compute. The stage
is budgeted by ``causal.max_problems`` and resumes, so a partial arm set is an
expected outcome rather than a failure.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.baselines.probe import difference_in_means_direction
from nlaast.causal.interventions import (
    InterventionResult,
    count_verification_markers,
    direction_claim_supported,
    dose_response,
    filler_text,
    projection_editor,
    random_direction,
    summarise_arm,
)
from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import JsonlWriter, read_jsonl, write_json
from nlaast.models.target import TargetModel
from nlaast.seeding import derive, rng
from nlaast.trace.chunking import Chunk, prefix_text

STAGE = "causal"


def fit_tail_direction(cfg, ast_rows, traces, store, log):
    """Difference in means between tail and matched-position activations.

    Difference in means rather than a probe weight: it is the less
    model-dependent estimator, and a causal claim should not rest on a
    classifier's regularisation path.
    """
    layer = cfg.target.layer
    tail_vecs, pre_vecs, n_problems = [], [], 0

    for row in ast_rows:
        pid = row["problem_id"]
        t = traces.get(pid)
        if t is None or t.get("split") != "train":
            continue
        if not store.has(pid, layer) or not row.get("windows"):
            continue
        arr, meta = store.get(pid, layer)
        chunk_to_row = {c: i for i, c in enumerate(meta.get("chunk_indices", []))}
        w = row["windows"]
        if "tail" not in w or "matched_position" not in w:
            continue
        used = False
        for kind, bucket in (("tail", tail_vecs), ("matched_position", pre_vecs)):
            lo, hi = int(w[kind][0]), int(w[kind][1])
            for c in range(lo, hi + 1):
                r = chunk_to_row.get(c)
                if r is not None:
                    bucket.append(arr[r])
                    used = True
        n_problems += int(used)

    if len(tail_vecs) < 5 or len(pre_vecs) < 5:
        log.warning("tail direction unavailable: %d tail / %d pre-stabilisation "
                    "vectors on the train split", len(tail_vecs), len(pre_vecs))
        return None, {"available": False, "n_tail": len(tail_vecs),
                      "n_pre": len(pre_vecs), "n_problems": n_problems}

    d = difference_in_means_direction(np.vstack(tail_vecs), np.vstack(pre_vecs))
    info = {
        "available": True,
        "n_tail": len(tail_vecs),
        "n_pre": len(pre_vecs),
        "n_problems": n_problems,
        "split": "train",
        "estimator": "difference_in_means",
        "layer": layer,
    }
    log.info("tail direction fitted on %d train problems (%d tail / %d pre vectors)",
             n_problems, len(tail_vecs), len(pre_vecs))
    return d, info


def chunks_of(trace: dict) -> list[Chunk]:
    return [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                       "token_end", "prefix_tokens")})
            for c in trace.get("chunks", [])]


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    if not traces or not ast_rows:
        log.error("need traces and AST results first")
        return 1

    store = ActivationStore(cfg.dir / "acts")
    layer, ccfg = cfg.target.layer, cfg.causal
    direction, dir_info = fit_tail_direction(cfg, ast_rows, traces, store, log)

    # Interventions are evaluated on the eval split only - the direction came
    # from the train split.
    targets = [r for r in ast_rows
               if r.get("tail_start") is not None
               and traces.get(r["problem_id"], {}).get("split") == "eval"
               and r.get("windows")]
    targets = targets[: (args.limit or ccfg.max_problems)]
    log.info("%d eval-split problems with a tail available for intervention", len(targets))
    if not targets:
        manifest.finish_stage(STAGE, status="skipped",
                              metrics={"reason": "no eval-split problem has a tail",
                                       "direction": dir_info})
        return 0

    manifest.start_stage(STAGE, n_problems=len(targets))
    out_dir = cfg.stage_dir(STAGE)
    writer = JsonlWriter(out_dir / "interventions.jsonl", key="id")

    model = TargetModel(cfg)
    started = time.monotonic()
    try:
        with writer:
            for i, row in enumerate(targets):
                try:
                    run_problem(model, cfg, row, traces[row["problem_id"]],
                                direction, writer, log)
                except Exception:
                    log.exception("intervention failed for %s", row["problem_id"])
                if (i + 1) % 5 == 0 or i + 1 == len(targets):
                    rate = (time.monotonic() - started) / (i + 1)
                    log.info("[%d/%d] interventions (eta %.1f min)",
                             i + 1, len(targets), rate * (len(targets) - i - 1) / 60)
    finally:
        model.close()

    rows = read_jsonl(out_dir / "interventions.jsonl")
    results = [InterventionResult(**{k: v for k, v in r.items() if k != "id"})
               for r in rows]
    metrics = {
        "n_results": len(results),
        "direction": dir_info,
        "arms": {a: summarise_arm(results, a).to_dict()
                 for a in sorted({r.intervention for r in results})},
        "dose_response_ablate": dose_response(results, "ablate_dir"),
        "verdict": direction_claim_supported(results),
    }
    write_json(out_dir / "summary.json", metrics)

    log.info("intervention arms:")
    for a, s in metrics["arms"].items():
        log.info("  %-18s n=%-4d answer-change %.3f  marker delta %+.2f  accuracy %.3f",
                 a, s["n"], s["answer_change_rate"], s["marker_delta"], s["accuracy"])
    v = metrics["verdict"]
    log.info("RQ3 direction claim: supported=%s (beats random=%s, beats matched "
             "position=%s, monotone dose=%s, answers preserved=%s)",
             v["supported"], v["beats_random_direction"], v["beats_matched_position"],
             v["monotone_dose_response"], v["answers_preserved"])

    manifest.finish_stage(STAGE, status="complete", output=str(out_dir), metrics=metrics)
    return 0


def run_problem(model, cfg, ast_row, trace, direction, writer, log) -> None:
    ccfg = cfg.causal
    pid = ast_row["problem_id"]
    question, gold = trace["question"], trace.get("gold")
    chunks = chunks_of(trace)
    tail_start = int(ast_row["tail_start"])
    if tail_start <= 0 or tail_start >= len(chunks):
        return

    baseline_answer = trace.get("final_answer")
    baseline_correct = bool(trace.get("final_correct"))
    baseline_markers = count_verification_markers(
        trace["trace"][chunks[tail_start - 1].char_end :]
    )
    prefix = prefix_text(chunks, tail_start - 1, trace["trace"])
    prompt_tokens = len(model.tokenizer(model.build_prompt(question, prefix),
                                        add_special_tokens=False)["input_ids"])
    tail_tokens = max(1, int(ast_row.get("tail_tokens") or 1))

    def record(name, coefficient, text, n_tokens, direction_id=None, notes=None):
        answer = extract_answer(prefix + text, PERMISSIVE)
        key = f"{pid}|{name}|{coefficient}|{direction_id}"
        if writer.has(key):
            return
        writer.write(InterventionResult(
            problem_id=pid, intervention=name, coefficient=float(coefficient),
            answer=answer, baseline_answer=baseline_answer, gold=gold,
            answer_changed=not equivalent(answer, baseline_answer),
            correct=equivalent(answer, gold), baseline_correct=baseline_correct,
            tokens_generated=n_tokens,
            verification_markers=count_verification_markers(text),
            baseline_verification_markers=baseline_markers,
            direction_id=direction_id, notes=notes or {},
        ).to_dict())

    # -- truncate ---------------------------------------------------------
    if "truncate" in ccfg.interventions:
        forced = model.force_answer(question, prefix, seed=derive(cfg.seed, "trunc", pid))
        record("truncate", 0.0, forced, 0)

    # -- filler -----------------------------------------------------------
    if "filler" in ccfg.interventions:
        filler = filler_text(tail_tokens, ccfg.filler_text, model.tokenizer)
        forced = model.force_answer(question, prefix + " " + filler,
                                    seed=derive(cfg.seed, "filler", pid))
        record("filler", 0.0, filler + forced, tail_tokens)

    if direction is None:
        return

    # -- direction ablation over the tail span ----------------------------
    # The span starts where the tail starts; everything before it is left
    # untouched, which is what makes this an intervention on the tail rather
    # than on the whole sequence.
    span = (prompt_tokens, prompt_tokens + tail_tokens + 64)
    pre_len = chunks[tail_start - 1].prefix_tokens - (
        chunks[max(0, tail_start - 1 - (len(chunks) - tail_start))].prefix_tokens
    )
    pre_span = (max(0, prompt_tokens - max(1, pre_len)), prompt_tokens)

    arms = []
    if "ablate_dir" in ccfg.interventions:
        for c in ccfg.dose_coefficients:
            arms.append(("ablate_dir", c, direction, span, "tail_dim"))
    if "matched_position" in ccfg.interventions:
        arms.append(("matched_position", 1.0, direction, pre_span, "tail_dim"))
    if "random_dir" in ccfg.interventions:
        r = rng(cfg.causal.seed, pid)
        for k in range(ccfg.n_random_directions):
            arms.append(("random_dir", 1.0,
                         random_direction(cfg.target.d_model, r, reference=direction),
                         span, f"random_{k}"))

    for name, coeff, vec, sp, did in arms:
        key = f"{pid}|{name}|{float(coeff)}|{did}"
        if writer.has(key):
            continue
        edit = projection_editor(vec, coeff, sp)
        with model.patch_layer(edit):
            gen = model.generate(
                [model.build_prompt(question, prefix)],
                max_new_tokens=min(cfg.ast.continuation_max_new_tokens, 256),
                temperature=0.0, top_p=1.0, do_sample=False,
                seed=derive(cfg.seed, name, pid, did), batch_size=1,
            )[0]
        record(name, coeff, gen.text, gen.n_tokens, direction_id=did,
               notes={"span": list(sp)})


if __name__ == "__main__":
    sys.exit(main())
