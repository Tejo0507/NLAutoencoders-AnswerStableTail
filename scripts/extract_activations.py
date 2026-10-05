"""Stage `acts`: layer-20 residual-stream activations at chunk boundaries.

Extraction follows the released autoencoder's own convention exactly: layer K
is the *output* of decoder block K (``model.model.layers[K]``), which is HF's
``hidden_states[K+1]``. Upstream's ``nla/datagen/extractors.py`` defines it that
way and the checkpoint was trained on vectors produced that way; an off-by-one
would feed the verbaliser a different layer with no error raised.

One vector per chunk boundary, at the boundary's last generated token. The
caller later selects tail, matched-position and matched-length windows from
these rows, so the activation stage does not need to know what the tail is -
and does not, which keeps the `ast` stage independent of this one.

``--layers`` extracts additional layers for falsification test F7. The NLA is
layer-20-only, so the extra layers feed the probe and position analyses alone.
"""

from __future__ import annotations

import sys
import time

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.logging_utils import read_jsonl, write_json
from nlaast.models.target import TargetModel

STAGE = "acts"


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="layers to extract (default: the configured target layer)")
    args = p.parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    traces = [t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")]
    if not traces:
        log.error("no usable traces - run scripts/generate_traces.py first")
        return 1
    if args.limit:
        traces = traces[: args.limit]

    layers = args.layers or [cfg.target.layer]
    store = ActivationStore(cfg.stage_dir(STAGE))
    todo = [(t, L) for t in traces for L in layers
            if not store.has(t["problem_id"], L)]
    log.info("%d traces x %d layers; %d blocks to extract", len(traces), len(layers), len(todo))

    manifest.start_stage(STAGE, n_traces=len(traces), layers=layers)
    if not todo:
        manifest.finish_stage(STAGE, status="complete", metrics=store.summary())
        return 0

    model = TargetModel(cfg)
    started = time.monotonic()
    failures = []
    try:
        for i, (trace, layer) in enumerate(todo):
            pid = trace["problem_id"]
            chunks = trace.get("chunks", [])
            if not chunks:
                failures.append({"problem_id": pid, "reason": "no chunks"})
                continue
            positions = [int(c["token_end"]) for c in chunks]
            try:
                acts, meta = model.activations_at(
                    trace["question"], trace["trace"], positions, layer=layer,
                    # The ids the model actually sampled. Re-tokenising the
                    # decoded text is not guaranteed to reproduce them, and a
                    # one-token shift here would read every activation at the
                    # wrong position with no visible symptom.
                    trace_token_ids=trace.get("token_ids"),
                )
            except Exception as exc:
                log.exception("extraction failed for %s L%d", pid, layer)
                failures.append({"problem_id": pid, "layer": layer, "reason": repr(exc)})
                continue

            arr = acts.numpy()
            if arr.shape != (len(positions), cfg.target.d_model):
                failures.append({
                    "problem_id": pid, "layer": layer,
                    "reason": f"unexpected shape {arr.shape}",
                })
                continue

            store.put(
                pid, layer, arr, positions,
                kinds=["chunk_boundary"] * len(positions),
                extra={
                    "chunk_indices": [int(c["index"]) for c in chunks],
                    "prompt_tokens": meta["prompt_tokens"],
                    "trace_tokens": meta["trace_tokens"],
                    "resolved_positions": meta["resolved"],
                    "n_generated_tokens": trace.get("n_tokens"),
                    # Recorded so alignment is auditable rather than assumed.
                    # ``alignment_exact`` means the sampled token ids were
                    # used; ``retokenisation_drift`` is how far re-tokenising
                    # the decoded text would have moved every position.
                    "alignment_exact": meta["alignment_exact"],
                    "retokenisation_drift": meta["retokenisation_drift"],
                    "clamped_positions": meta["clamped_positions"],
                    "layer_convention": "output of decoder block K == hidden_states[K+1]",
                },
            )
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rate = (time.monotonic() - started) / (i + 1)
                log.info("[%d/%d] %s L%d  %s  (eta %.1f min)",
                         i + 1, len(todo), pid, layer, arr.shape,
                         rate * (len(todo) - i - 1) / 60)
    finally:
        model.close()

    summary = store.summary()
    summary["failures"] = failures
    summary["n_failed"] = len(failures)
    summary["layers"] = layers
    # Alignment audit across the whole store. A non-zero clamp count or a
    # non-exact block means some activation was not read where the chunk
    # boundary said it was.
    drifts, clamps, inexact = [], 0, 0
    for rec in store:
        try:
            _, m = store.get(rec["problem_id"], rec["layer"], verify=False)
        except Exception:
            continue
        drifts.append(int(m.get("retokenisation_drift", 0)))
        clamps += int(m.get("clamped_positions", 0))
        inexact += int(not m.get("alignment_exact", False))
    if drifts:
        summary["alignment"] = {
            "n_blocks_checked": len(drifts),
            "n_inexact_alignment": inexact,
            "total_clamped_positions": clamps,
            "retokenisation_drift_nonzero": sum(1 for d in drifts if d != 0),
            "max_abs_retokenisation_drift": max(abs(d) for d in drifts),
            "note": ("drift is how far re-tokenising the decoded trace would "
                     "have moved every position; it is reported because the "
                     "sampled ids are used instead, making alignment exact"),
        }
        log.info("alignment: %d/%d blocks would have drifted under "
                 "re-tokenisation (max %d tokens); %d positions clamped",
                 summary["alignment"]["retokenisation_drift_nonzero"],
                 len(drifts), summary["alignment"]["max_abs_retokenisation_drift"],
                 clamps)
    summary["elapsed_min"] = round((time.monotonic() - started) / 60, 1)
    write_json(cfg.stage_dir(STAGE) / "summary.json", summary)
    log.info("activations: %s", {k: v for k, v in summary.items() if k != "failures"})
    if failures:
        log.warning("%d extraction failures (recorded in summary.json)", len(failures))

    manifest.finish_stage(STAGE, status="complete" if summary["n_blocks"] else "failed",
                          output=str(cfg.stage_dir(STAGE)), metrics=summary)
    return 0 if summary["n_blocks"] else 1


if __name__ == "__main__":
    sys.exit(main())
