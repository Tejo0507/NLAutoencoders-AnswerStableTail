"""Stage `ast`: Answer-Stable Tail detection (O1 / RQ1).

Reads the boundary evidence the `traces` stage collected and applies the
pre-registered criteria. **This stage does not read activations.** That is the
point: the review's first distinction is that answer stability and causal
redundancy are different properties, and the tail has to be defined without
reference to any activation description for the comparison in O3 to mean
anything. The stage DAG enforces it: this stage depends on `traces`, not on
`acts`.

Edge cases are written out with a status code and counted, never dropped.

``--sweep`` re-runs detection under alternative parameter settings for
falsification test F5 (does the tail survive its own definition being
perturbed?) without regenerating anything.
"""

from __future__ import annotations

import sys
from collections import Counter

from _stage import base_parser, setup, should_skip

from nlaast.ast_detect import (
    BoundaryEvidence,
    TailStatus,
    detect_ast,
    matched_windows,
)
from nlaast.data.answers import PERMISSIVE, STRICT, equivalent, extract_answer
from nlaast.logging_utils import read_jsonl, write_json, write_jsonl
from nlaast.seeding import rng
from nlaast.trace.chunking import Chunk, prefix_text

STAGE = "ast"


def answer_stated_at(trace: dict) -> dict:
    """The first boundary at which the trace has actually *said* the answer.

    This is not the same thing as the Answer-Stable Tail start, and the
    difference is the central interpretive caveat of the whole construct.

    The AST criteria ask whether the answer is *determined* from a prefix: an
    answer forced out of the prefix, and answers from independent
    continuations of it, all match the final answer. A prefix can satisfy that
    while the trace has not yet performed the arithmetic - the model finishes
    it inside the forced answer instead. On the first real traces that happened
    plainly: one trace's tail began at chunk 3 of 8, two chunks before the
    arithmetic producing the answer appeared.

    For the stopping comparison (O3) determinacy is exactly the right notion -
    a rule that stops there loses nothing. But a window that precedes the
    answer being stated contains genuine computation, not post-answer
    redundancy, which matters for how a verbalisation of that window may be
    read. So the gap is measured and reported rather than left implicit.

    Returns the first boundary index under the strict and the permissive
    extractor at which the parsed answer is equivalent to the final answer.
    """
    final = trace.get("final_answer")
    chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                         "token_end", "prefix_tokens")})
              for c in trace.get("chunks", [])]
    out: dict[str, int | None] = {"answer_stated_at_strict": None,
                                  "answer_stated_at_permissive": None}
    if final is None or not chunks:
        return out
    text = trace.get("trace", "")
    for i in range(len(chunks)):
        prefix = prefix_text(chunks, i, text)
        if (out["answer_stated_at_strict"] is None
                and equivalent(extract_answer(prefix, STRICT), final)):
            out["answer_stated_at_strict"] = i
        if (out["answer_stated_at_permissive"] is None
                and equivalent(extract_answer(prefix, PERMISSIVE), final)):
            out["answer_stated_at_permissive"] = i
        if all(v is not None for v in out.values()):
            break
    return out


def evidence_from_row(row: dict) -> list[BoundaryEvidence]:
    return [
        BoundaryEvidence(
            index=b["index"],
            parsed_answer=b.get("parsed_answer"),
            forced_answer=b.get("forced_answer"),
            forced_matches_final=bool(b.get("forced_matches_final")),
            continuation_answers=list(b.get("continuation_answers", [])),
            continuations_matching=int(b.get("continuations_matching", 0)),
            prefix_tokens=int(b.get("prefix_tokens", 0)),
        )
        for b in row.get("boundaries", [])
    ]


def detect_one(cfg, trace: dict, *, require_unanimous=None, convergence_window=None,
               min_boundary_fraction=None, k_cap: int | None = None) -> dict:
    """Run detection on one trace. The keyword arguments drive the F5 sweep."""
    acfg = cfg.ast
    evidence = evidence_from_row(trace)
    if k_cap is not None:
        # Simulate a smaller K by truncating the recorded continuations. Exact,
        # because the continuations are independent samples.
        for e in evidence:
            e.continuation_answers = e.continuation_answers[:k_cap]
            e.continuations_matching = sum(
                1 for a in e.continuation_answers
                if equivalent(a, trace.get("final_answer"))
            )

    res = detect_ast(
        problem_id=trace["problem_id"],
        evidence=evidence,
        final_answer=trace.get("final_answer"),
        gold=trace.get("gold"),
        n_chunks=int(trace.get("n_chunks", 0)),
        total_tokens=int(trace.get("n_tokens", 0)),
        truncated=bool(trace.get("truncated")),
        require_unanimous=(acfg.require_unanimous if require_unanimous is None
                           else require_unanimous),
        convergence_window=(acfg.convergence_window if convergence_window is None
                            else convergence_window),
        min_boundary_fraction=(acfg.min_boundary_fraction if min_boundary_fraction is None
                               else min_boundary_fraction),
    )
    out = res.to_dict()
    out["dataset"] = trace.get("dataset")
    out["split"] = trace.get("split")
    out["level"] = trace.get("level")
    out.update(answer_stated_at(trace))
    stated = out["answer_stated_at_strict"]
    out["tail_starts_before_answer_stated"] = (
        None if (stated is None or res.tail_start is None)
        else bool(res.tail_start < stated)
    )
    out["chunks_from_tail_start_to_statement"] = (
        None if (stated is None or res.tail_start is None) else stated - res.tail_start
    )

    if res.tail_start is not None:
        out["windows"] = {
            k: list(v) for k, v in matched_windows(
                res.tail_start, res.n_chunks,
                rng(cfg.seed, "windows", trace["problem_id"])
            ).items()
        }
    else:
        out["windows"] = {}
    return out


def summarise(rows: list[dict]) -> dict:
    status = Counter(r["status"] for r in rows)
    with_tail = [r for r in rows if r["tail_start"] is not None]
    n = len(rows)
    out = {
        "n": n,
        "status_counts": dict(status),
        "status_shares": {k: v / n for k, v in status.items()} if n else {},
        "n_with_tail": len(with_tail),
        "tail_rate": len(with_tail) / n if n else 0.0,
    }

    # How often the *final* boundary qualifies, and how often the tail is
    # nothing but that boundary.
    #
    # This matters for reading `tail_rate`. At the last boundary the prefix is
    # the whole trace, so forcing an answer from it reproduces the final answer
    # and continuations from it restate the same answer - both criteria are
    # close to tautological there. A tail therefore almost always exists, and
    # `tail_rate` sits near 1 whenever the final answer parses and generation
    # was not truncated. It is an upper bound on nothing interesting. The
    # informative quantity is `tail_fraction`: how much of the trace the tail
    # covers. A tail consisting only of the final chunk is recorded separately
    # as trivial, because it means no redundancy was detected.
    last_qualifies = sum(
        1 for r in rows
        if r.get("evidence")
        and r["evidence"][-1].get("forced_matches_final")
        and r["evidence"][-1].get("continuations_unanimous")
    )
    trivial = [r for r in with_tail if r["tail_start"] == r["n_chunks"] - 1]
    out["final_boundary_qualifies_rate"] = last_qualifies / n if n else 0.0
    out["n_trivial_tail"] = len(trivial)
    out["trivial_tail_share"] = len(trivial) / len(with_tail) if with_tail else 0.0
    out["n_nontrivial_tail"] = len(with_tail) - len(trivial)
    out["nontrivial_tail_rate"] = (len(with_tail) - len(trivial)) / n if n else 0.0
    out["tail_rate_caveat"] = (
        "The final boundary satisfies both criteria almost by construction - "
        "its prefix is the whole trace - so tail_rate is close to 1 whenever "
        "the final answer parses and generation was not truncated. Read "
        "tail_fraction and nontrivial_tail_rate instead."
    )

    if with_tail:
        fr = [r["tail_fraction"] for r in with_tail]
        tt = [r["tail_tokens"] for r in with_tail]
        out.update(
            mean_tail_fraction=sum(fr) / len(fr),
            median_tail_fraction=sorted(fr)[len(fr) // 2],
            min_tail_fraction=min(fr),
            max_tail_fraction=max(fr),
            mean_tail_tokens=sum(tt) / len(tt),
            total_tail_tokens=sum(tt),
            total_tokens=sum(r["total_tokens"] for r in with_tail),
        )
        out["tail_token_share"] = (
            out["total_tail_tokens"] / out["total_tokens"] if out["total_tokens"] else 0.0
        )
    # Determinacy versus statement. See answer_stated_at() - a tail that
    # begins before the trace has said the answer contains computation, not
    # post-answer redundancy, and that changes how a verbalisation of the
    # window may be read.
    gaps = [r["chunks_from_tail_start_to_statement"] for r in rows
            if r.get("chunks_from_tail_start_to_statement") is not None]
    if gaps:
        before = [g for g in gaps if g > 0]
        out["determinacy_vs_statement"] = {
            "n": len(gaps),
            "tail_starts_before_answer_stated": len(before) / len(gaps),
            "mean_chunks_before_statement": sum(gaps) / len(gaps),
            "max_chunks_before_statement": max(gaps),
            "n_unstated": sum(
                1 for r in rows
                if r.get("tail_start") is not None
                and r.get("answer_stated_at_strict") is None),
            "note": ("A positive gap means the AST criteria were satisfied "
                     "before the trace stated the answer: forcing elicited it "
                     "from the prefix. Correct for a stopping rule, but such a "
                     "window is not post-answer redundancy."),
        }

    # How often the cheap agreement rule fires earlier than the AST. This is
    # the quantitative form of Mo et al.'s point and it is reported whether or
    # not it flatters the AST.
    both = [r for r in rows
            if r["tail_start"] is not None and r.get("convergence_start") is not None]
    if both:
        out["convergence_vs_ast"] = {
            "n": len(both),
            "convergence_earlier": sum(1 for r in both
                                       if r["convergence_start"] < r["tail_start"]) / len(both),
            "same": sum(1 for r in both
                        if r["convergence_start"] == r["tail_start"]) / len(both),
            "convergence_later": sum(1 for r in both
                                     if r["convergence_start"] > r["tail_start"]) / len(both),
            "mean_gap_chunks": sum(r["tail_start"] - r["convergence_start"]
                                   for r in both) / len(both),
        }
    return out


def run_sweep(cfg, traces: list[dict], log) -> dict:
    """F5: does the tail survive its own definition being perturbed?

    Two settings are swept, and one that used to be swept is not.

    ``k`` (resampled continuations) is simulated by truncating the recorded
    continuations, which is exact because they are independent samples - but
    only *downwards*. A requested ``k`` above the number actually generated
    cannot be simulated at all; it silently reproduced the base result and was
    then reported as "stable across settings", which is the one thing a
    falsification test must never do. Such a setting is now recorded as
    ``not_applicable`` with the reason.

    ``require_unanimous`` genuinely changes criterion 2 and is swept.

    ``convergence_window`` is **not** an AST parameter. It parameterises the
    Liu & Wang agreement baseline, and sweeping it leaves every tail untouched
    by construction; reporting an unchanged tail fraction across it looked like
    evidence of robustness and was an identity. It is swept here against the
    quantity it actually moves - where the agreement rule fires relative to the
    tail - and reported under its own key.
    """
    available_k = max(
        (max((len(b.get("continuation_answers") or []) for b in t.get("boundaries", [])),
             default=0) for t in traces),
        default=0,
    )
    sweep: dict = {
        "_meta": {
            "n_traces": len(traces),
            "max_continuations_recorded": available_k,
            "configured_k": cfg.ast.k_continuations,
            "note": ("k can only be simulated downwards from the recorded "
                     "continuations; convergence_window is a baseline "
                     "parameter and does not enter the tail criteria"),
        }
    }

    for k in cfg.robustness.ast_sweep_k:
        key = f"k={k}"
        if k > available_k:
            sweep[key] = {
                "status": "not_applicable",
                "reason": (f"only {available_k} continuations were generated per "
                           f"boundary, so k={k} cannot be simulated; raise "
                           f"ast.k_continuations and regenerate to test it"),
            }
            continue
        sweep[key] = {"status": "ran",
                      **summarise([detect_one(cfg, t, k_cap=k) for t in traces])}

    sweep["require_unanimous=False"] = {
        "status": "ran",
        **summarise([detect_one(cfg, t, require_unanimous=False) for t in traces]),
    }

    # The baseline parameter, swept against what it actually changes.
    conv: dict = {}
    for w in (1, 2, 3):
        rows = [detect_one(cfg, t, convergence_window=w) for t in traces]
        s = summarise(rows)
        conv[f"convergence_window={w}"] = {
            "n_fired": sum(1 for r in rows if r.get("convergence_start") is not None),
            "convergence_vs_ast": s.get("convergence_vs_ast"),
            "mean_tail_fraction": s.get("mean_tail_fraction"),
        }
    sweep["baseline_convergence_window"] = {
        "status": "ran",
        "settings": conv,
        "note": ("the tail is identical under every setting here by "
                 "construction - convergence_window is not an AST parameter. "
                 "What moves is where the agreement baseline fires relative to "
                 "the tail."),
    }

    ran = {k: v for k, v in sweep.items()
           if isinstance(v, dict) and v.get("status") == "ran"
           and "mean_tail_fraction" in v}
    log.info("F5 sweep: mean tail fraction %s",
             {k: round(v["mean_tail_fraction"], 3) for k, v in ran.items()})
    log.info("F5 sweep: non-trivial tail rate %s",
             {k: round(v["nontrivial_tail_rate"], 3) for k, v in ran.items()})
    skipped = [k for k, v in sweep.items()
               if isinstance(v, dict) and v.get("status") == "not_applicable"]
    if skipped:
        log.warning("F5 sweep: %s could not be simulated from the recorded "
                    "continuations and is reported as not applicable", skipped)
    return sweep


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--sweep", action="store_true",
                   help="also run the F5 definition-sensitivity sweep")
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

    manifest.start_stage(STAGE, n_traces=len(traces))
    out_dir = cfg.stage_dir(STAGE)

    rows = [detect_one(cfg, t) for t in traces]
    write_jsonl(out_dir / "ast.jsonl", rows)
    summary = summarise(rows)
    write_json(out_dir / "summary.json", summary)

    log.info("status: %s", summary["status_counts"])
    log.info("tail detected on %d/%d problems (%.1f%%); %d are trivial "
             "(final chunk only), so %.1f%% have a non-trivial tail",
             summary["n_with_tail"], summary["n"], 100 * summary["tail_rate"],
             summary["n_trivial_tail"], 100 * summary["nontrivial_tail_rate"])
    log.info("the final boundary qualifies on %.1f%% of problems - near "
             "tautological, so tail_rate is near its ceiling by construction; "
             "read tail_fraction",
             100 * summary["final_boundary_qualifies_rate"])
    if summary["n_with_tail"]:
        log.info("tail fraction: mean %.3f, median %.3f, range %.2f-%.2f | "
                 "tail tokens are %.1f%% of all generated tokens",
                 summary["mean_tail_fraction"], summary["median_tail_fraction"],
                 summary["min_tail_fraction"], summary["max_tail_fraction"],
                 100 * summary["tail_token_share"])
    if "convergence_vs_ast" in summary:
        c = summary["convergence_vs_ast"]
        log.info("agreement rule fires earlier than the AST on %.1f%% of problems "
                 "(mean gap %.2f chunks)", 100 * c["convergence_earlier"], c["mean_gap_chunks"])

    if args.sweep:
        write_json(out_dir / "sweep_F5.json", run_sweep(cfg, traces, log))

    manifest.finish_stage(STAGE, status="complete",
                          output=str(out_dir / "ast.jsonl"), metrics=summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
