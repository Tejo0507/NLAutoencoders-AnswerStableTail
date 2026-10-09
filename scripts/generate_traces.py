"""Stage `traces`: generate reasoning traces and the evidence the tail test needs.

For every problem this writes, in one pass while the target model is resident:

* the **canonical trace** (greedy, so the analysed object is deterministic given
  the checkpoint) with its token ids and exact character offsets;
* the **chunking** of that trace and the intermediate answer parsed at each
  chunk boundary;
* for each boundary, a **forced answer** from the truncated prefix
  (AST criterion 1);
* for each boundary, **K resampled continuations** and their answers
  (AST criterion 2 - the control Mo et al. show the plain agreement rule lacks);
* **semantic-entropy samples** for the whole problem (the uncertainty baseline).

Generation is by far the most expensive stage, and loading the model is
expensive too, so everything that needs the target model is collected here
rather than spread over several stages that would each pay the load cost. The
stage checkpoints per problem and resumes.

Boundary pruning. Evaluating K continuations at every boundary of a 30-chunk
trace is roughly 30xK generations per problem. Criterion 3 (persistence) means
only a *suffix* of boundaries can ever qualify, so boundaries are evaluated from
the end backwards and the scan stops once a boundary fails - the earlier ones
cannot be the tail start regardless of what they would have shown. This is an
exact optimisation of the stated definition, not an approximation of it.
"""

from __future__ import annotations

import sys
import time

from _stage import base_parser, setup, should_skip

from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import JsonlWriter, read_jsonl
from nlaast.models.target import TargetModel
from nlaast.seeding import derive
from nlaast.trace.chunking import chunk_trace, prefix_text

STAGE = "traces"

#: How much of each continuation and forced answer to keep for auditing. The
#: answer always appears at the end, and keeping whole continuations would
#: multiply the trace file by K.
_CONTINUATION_KEEP_CHARS = 400


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--no-entropy", action="store_true",
                   help="skip the semantic-entropy samples")
    args = p.parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    problems = read_jsonl(cfg.dir / "data" / "problems.jsonl")
    if not problems:
        log.error("no problems - run scripts/prepare_problems.py first")
        return 1
    if args.limit:
        problems = problems[: args.limit]

    manifest.start_stage(STAGE, n_problems=len(problems))
    out_path = cfg.stage_dir(STAGE) / "traces.jsonl"
    writer = JsonlWriter(out_path, key="id")
    todo = [pr for pr in problems if not writer.has(pr["id"])]
    log.info("%d problems, %d already done, %d to do",
             len(problems), len(problems) - len(todo), len(todo))

    if not todo:
        # Nothing left to generate. Report the same metrics the generating path
        # reports rather than a bare count: a resumed run and a run that
        # finished in one go describe the same corpus, and the manifest is what
        # the execution report reads.
        metrics = corpus_metrics(read_jsonl(out_path))
        metrics["resumed_without_generating"] = True
        log.info("traces already complete: %s", metrics)
        manifest.finish_stage(STAGE, status="complete", output=str(out_path),
                              metrics=metrics)
        return 0

    model = TargetModel(cfg)
    started = time.monotonic()
    n_done = 0

    try:
        with writer:
            for i, prob in enumerate(todo):
                t0 = time.monotonic()
                try:
                    row = process_problem(model, cfg, prob, log, skip_entropy=args.no_entropy)
                except Exception as exc:
                    # A failed problem is written as a failure row, not skipped:
                    # otherwise a resume would retry it forever, and the failure
                    # rate would not appear in the metrics.
                    log.exception("problem %s failed", prob["id"])
                    writer.write({"id": prob["id"], "problem_id": prob["id"],
                                  "error": repr(exc), "ok": False})
                    continue
                row["elapsed_s"] = round(time.monotonic() - t0, 1)
                writer.write(row)
                n_done += 1
                rate = (time.monotonic() - started) / max(1, n_done)
                log.info(
                    "[%d/%d] %s  chunks=%d  tokens=%d  answer=%s  %.0fs  (eta %.0f min)",
                    i + 1, len(todo), prob["id"], row.get("n_chunks", 0),
                    row.get("n_tokens", 0), str(row.get("final_answer"))[:20],
                    row["elapsed_s"], rate * (len(todo) - i - 1) / 60,
                )
    finally:
        model.close()

    metrics = corpus_metrics(read_jsonl(out_path))
    metrics["elapsed_min"] = round((time.monotonic() - started) / 60, 1)
    log.info("traces complete: %s", metrics)
    manifest.finish_stage(STAGE, status="complete" if metrics["n_ok"] else "failed",
                          output=str(out_path), metrics=metrics)
    return 0 if metrics["n_ok"] else 1


def corpus_metrics(rows: list[dict]) -> dict:
    """What the trace corpus looks like, from the persisted rows only.

    Computed from the file rather than from the loop's own counters so a
    resumed run and a run that finished in one go report the same thing. The
    per-dataset breakdown is here because the achieved N per benchmark is what
    the stratified comparisons downstream can actually support - a corpus that
    stopped early is usually unbalanced, and that has to be visible.
    """
    ok = [r for r in rows if r.get("ok")]
    by_dataset: dict[str, int] = {}
    for r in ok:
        key = str(r.get("dataset"))
        by_dataset[key] = by_dataset.get(key, 0) + 1
    return {
        "n_traces": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "by_dataset": by_dataset,
        "mean_tokens": (sum(r.get("n_tokens", 0) for r in ok) / len(ok)) if ok else 0,
        "mean_chunks": (sum(r.get("n_chunks", 0) for r in ok) / len(ok)) if ok else 0,
        "accuracy": (sum(1 for r in ok if r.get("final_correct")) / len(ok)) if ok else 0,
        "truncated_rate": (sum(1 for r in ok if r.get("truncated")) / len(ok)) if ok else 0,
        "n_scan_capped": sum(1 for r in ok if r.get("boundary_scan_capped")),
    }


def process_problem(model: TargetModel, cfg, prob: dict, log, skip_entropy: bool = False) -> dict:
    """Everything the target model is needed for, for one problem."""
    gcfg, acfg, secfg = cfg.generation, cfg.ast, cfg.semantic_entropy
    pid, question, gold = prob["id"], prob["question"], prob["gold"]

    # -- canonical trace --------------------------------------------------
    canonical = model.generate(
        [model.build_prompt(question)],
        max_new_tokens=gcfg.max_new_tokens,
        temperature=gcfg.temperature,
        top_p=gcfg.top_p,
        do_sample=not gcfg.canonical_greedy,
        seed=derive(cfg.seed, "canonical", pid),
        batch_size=1,
    )[0]
    text = canonical.text
    final_answer = extract_answer(text, PERMISSIVE)
    chunks = chunk_trace(text, cfg.chunk, token_offsets=canonical.offsets)

    row: dict = {
        "id": pid,
        "problem_id": pid,
        "dataset": prob.get("dataset"),
        "split": prob.get("split"),
        "level": prob.get("level"),
        "question": question,
        "gold": gold,
        "trace": text,
        "token_ids": canonical.token_ids,
        "offsets": canonical.offsets,
        "n_tokens": canonical.n_tokens,
        "prompt_tokens": canonical.prompt_tokens,
        # ``finished`` false means the token cap was hit, so the "final" answer
        # may not be the model's final answer. The AST stage marks these.
        "truncated": not canonical.finished,
        "final_answer": final_answer,
        "final_correct": equivalent(final_answer, gold),
        "n_chunks": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
        "ok": True,
    }

    # -- per-boundary evidence -------------------------------------------
    boundaries: list[dict] = []
    capped = False
    if final_answer is not None and len(chunks) >= 2:
        # Backwards scan: criterion 3 means only a suffix can qualify, so the
        # scan runs from the end and stops at the first boundary that fails.
        # Exact, not an approximation.
        #
        # Boundaries are taken in blocks so the forced-answer generations can
        # be batched. A block may evaluate a few boundaries past the first
        # failure; that costs a little generation and changes no result,
        # because the `ast` stage reads the evidence, not the order
        # it was gathered in.
        min_i = int(acfg.min_boundary_fraction * len(chunks))
        floor = max(min_i, len(chunks) - acfg.max_boundaries_evaluated)
        capped = floor > min_i
        collected: dict[int, dict] = {}
        block = max(1, min(4, cfg.generation.batch_size))
        idx = len(chunks) - 1
        stop = False
        while idx >= floor and not stop:
            group = list(range(idx, max(floor - 1, idx - block), -1))
            for ev in boundary_evidence_batch(model, cfg, pid, question, text,
                                              chunks, group, final_answer):
                collected[ev["index"]] = ev
            for g in group:
                ev = collected[g]
                if not (ev["forced_matches_final"]
                        and ev["continuations_matching"] == ev["n_continuations"]):
                    stop = True
                    break
            idx -= len(group)
        # Any boundary not evaluated still needs its parsed answer, which the
        # convergence baseline reads.
        for idx, c in enumerate(chunks):
            if idx in collected:
                boundaries.append(collected[idx])
            else:
                boundaries.append({
                    "index": idx,
                    "parsed_answer": extract_answer(prefix_text(chunks, idx, text), PERMISSIVE),
                    "forced_answer": None,
                    "forced_matches_final": False,
                    "continuation_answers": [],
                    "continuations_matching": 0,
                    "n_continuations": 0,
                    "prefix_tokens": c.prefix_tokens,
                    "evaluated": False,
                })
        boundaries.sort(key=lambda b: b["index"])
    else:
        for idx, c in enumerate(chunks):
            boundaries.append({
                "index": idx,
                "parsed_answer": extract_answer(prefix_text(chunks, idx, text), PERMISSIVE),
                "forced_answer": None, "forced_matches_final": False,
                "continuation_answers": [], "continuations_matching": 0,
                "n_continuations": 0, "prefix_tokens": c.prefix_tokens,
                "evaluated": False,
            })
    row["boundaries"] = boundaries
    row["boundary_scan_capped"] = capped
    row["max_boundaries_evaluated"] = acfg.max_boundaries_evaluated

    # -- semantic-entropy samples ----------------------------------------
    if not skip_entropy and secfg.n_samples > 0:
        prompts = [model.build_prompt(question)] * secfg.n_samples
        samples = model.generate(
            prompts,
            max_new_tokens=secfg.max_new_tokens,
            temperature=secfg.temperature,
            top_p=secfg.top_p,
            do_sample=True,
            seed=derive(cfg.seed, "entropy", pid) % (2**31),
        )
        row["entropy_samples"] = [
            {"answer": extract_answer(s.text, PERMISSIVE), "n_tokens": s.n_tokens}
            for s in samples
        ]
    else:
        row["entropy_samples"] = []

    return row


def boundary_evidence_batch(model, cfg, pid, question, text, chunks,
                            indices: list[int], final_answer) -> list[dict]:
    """Forced answer and K resampled continuations at several boundaries.

    Batched because single-stream decoding on this GPU runs at roughly 20
    tok/s against 45 tok/s batched, and the forced-answer call is short enough
    that per-call overhead dominates it.
    """
    acfg = cfg.ast
    gcfg = cfg.generation
    prefixes = {i: prefix_text(chunks, i, text) for i in indices}

    forced_prompts = [
        model.build_prompt(question, prefixes[i] + gcfg.force_answer_suffix)
        for i in indices
    ]
    forced_gens = model.generate(
        forced_prompts,
        max_new_tokens=gcfg.force_answer_max_new_tokens,
        temperature=0.0, top_p=1.0, do_sample=False,
        seed=derive(cfg.seed, "force", pid, indices[0]),
        batch_size=len(indices),
    )

    # All continuations for all boundaries in this block, in one batch.
    cont_by_index: dict[int, list] = {i: [] for i in indices}
    if acfg.k_continuations > 0:
        prompts, owners = [], []
        for i in indices:
            for _ in range(acfg.k_continuations):
                prompts.append(model.build_prompt(question, prefixes[i]))
                owners.append(i)
        gens = model.generate(
            prompts,
            max_new_tokens=acfg.continuation_max_new_tokens,
            temperature=acfg.continuation_temperature,
            top_p=gcfg.top_p,
            do_sample=True,
            seed=derive(cfg.seed, "cont", pid, indices[0]) % (2**31),
        )
        for owner, g in zip(owners, gens):
            cont_by_index[owner].append(g)

    out = []
    for i, fg in zip(indices, forced_gens):
        prefix = prefixes[i]
        forced_text = gcfg.force_answer_suffix + fg.text
        forced = extract_answer(forced_text, PERMISSIVE)

        cont_answers, cont_tails = [], []
        for c in cont_by_index[i]:
            # The continuation is read together with its prefix: an answer may
            # have been stated before the continuation started, and the
            # question is what answer this trajectory ends on.
            cont_answers.append(extract_answer(prefix + c.text, PERMISSIVE))
            # Keep the end of each continuation: enough to audit the tail
            # decision later, and enough to re-derive the answer parse. Without
            # the text a parser fix could not be applied retrospectively and
            # every continuation would have to be regenerated.
            cont_tails.append(c.text[-_CONTINUATION_KEEP_CHARS:])

        out.append({
            "index": i,
            "parsed_answer": extract_answer(prefix, PERMISSIVE),
            "forced_answer": forced,
            "forced_text": forced_text[-_CONTINUATION_KEEP_CHARS:],
            # Whether the forced completion ended on its own or hit
            # force_answer_max_new_tokens. A completion cut off mid-derivation
            # can fail criterion 1 for a budget reason rather than an
            # evidential one, which moves the tail start later; the `ast`
            # stage reports how often that happened.
            "forced_finished": bool(fg.finished),
            "forced_matches_final": equivalent(forced, final_answer),
            "continuation_answers": cont_answers,
            "continuation_tails": cont_tails,
            "continuations_matching": sum(
                1 for a in cont_answers if equivalent(a, final_answer)),
            "n_continuations": len(cont_answers),
            "prefix_tokens": chunks[i].prefix_tokens,
            "evaluated": True,
        })
    return out


if __name__ == "__main__":
    sys.exit(main())
