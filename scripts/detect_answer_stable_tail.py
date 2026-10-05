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
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_jsonl, write_json, write_jsonl
from nlaast.seeding import rng

STAGE = "ast"


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
    if with_tail:
        fr = [r["tail_fraction"] for r in with_tail]
        tt = [r["tail_tokens"] for r in with_tail]
        out.update(
            mean_tail_fraction=sum(fr) / len(fr),
            median_tail_fraction=sorted(fr)[len(fr) // 2],
            mean_tail_tokens=sum(tt) / len(tt),
            total_tail_tokens=sum(tt),
            total_tokens=sum(r["total_tokens"] for r in with_tail),
        )
        out["tail_token_share"] = (
            out["total_tail_tokens"] / out["total_tokens"] if out["total_tokens"] else 0.0
        )
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
    log.info("tail detected on %d/%d problems (%.1f%%)",
             summary["n_with_tail"], summary["n"], 100 * summary["tail_rate"])
    if summary["n_with_tail"]:
        log.info("mean tail fraction %.3f | tail tokens are %.1f%% of all generated tokens",
                 summary["mean_tail_fraction"], 100 * summary["tail_token_share"])
    if "convergence_vs_ast" in summary:
        c = summary["convergence_vs_ast"]
        log.info("agreement rule fires earlier than the AST on %.1f%% of problems "
                 "(mean gap %.2f chunks)", 100 * c["convergence_earlier"], c["mean_gap_chunks"])

    if args.sweep:
        sweep = {}
        for k in cfg.robustness.ast_sweep_k:
            srows = [detect_one(cfg, t, k_cap=k) for t in traces]
            sweep[f"k={k}"] = summarise(srows)
        for w in (1, 2, 3):
            srows = [detect_one(cfg, t, convergence_window=w) for t in traces]
            sweep[f"convergence_window={w}"] = summarise(srows)
        srows = [detect_one(cfg, t, require_unanimous=False) for t in traces]
        sweep["require_unanimous=False"] = summarise(srows)
        write_json(out_dir / "sweep_F5.json", sweep)
        log.info("F5 sweep: tail rates %s",
                 {k: round(v["tail_rate"], 3) for k, v in sweep.items()})

    manifest.finish_stage(STAGE, status="complete",
                          output=str(out_dir / "ast.jsonl"), metrics=summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
