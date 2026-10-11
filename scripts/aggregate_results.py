"""Stage `analysis`: aggregation, statistics and the pre-registered test family.

Answers the four questions the review asks, in order, and writes one
machine-readable table per question.

* **RQ1 / O1** - tail corpus, status accounting, agreement between the AST and
  the cheap convergence rule.
* **Primary RQ / O3** - the four stopping rules on the safety/saving curve at
  matched token budget, with problem-level paired bootstrap intervals.
* **RQ2 / O4** - claim-level faithfulness, with the verbaliser-only control
  reported next to it rather than in a footnote.
* **RQ3 / O5** - the causal verdict, which is reported as unsupported unless
  both controls and the dose-response are satisfied.

The family of tests is fixed here and corrected with Benjamini-Hochberg (F9).
Tests that could not be evaluated stay in the family rather than being dropped,
because shrinking the family would make the correction look kinder than it is.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage import base_parser, setup, should_skip

from nlaast.analysis import tables
from nlaast.analysis.stats import (
    TestResult,
    benjamini_hochberg,
    paired_binary_test,
    paired_test,
    summarise_tests,
    unpaired_test,
)
from nlaast.baselines.stopping import (
    MATCH_TOLERANCE,
    OperatingPoint,
    at_matched_budget,
    complete_threshold_grid,
    decisions_at,
    granularity_bound,
)
from nlaast.logging_utils import read_json, read_jsonl, write_json

STAGE = "analysis"
NLA_RULE = "nla_readout"


def load_curves(cfg) -> tuple[dict, list[float], dict]:
    path = cfg.dir / "baselines" / "curves.json"
    if not path.exists():
        return {}, [], {}
    raw = read_json(path)
    curves = {
        rule: [OperatingPoint(**p) for p in pts]
        for rule, pts in raw.get("curves", {}).items()
    }
    return (curves, raw.get("matched_budgets", []),
            raw.get("curve_resolution", {}))


def nla_stopping_curve(cfg, log):
    """Turn the verbalised readout into a stopping rule, so it can be compared.

    The NLA produces a description, not a number, so a score has to be derived
    from it. The score used is reconstruction cosine at the boundary: the
    cheapest defensible reading of "the readout says this state is
    well-characterised". That choice is a limitation and is reported as one -
    a different reduction of text to a scalar could give a different curve, and
    this study does not claim to have found the best one.
    """
    from nlaast.baselines.stopping import sweep_threshold
    from nlaast.data.answers import PERMISSIVE, extract_answer
    from nlaast.trace.chunking import Chunk, prefix_text

    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    if not recon:
        return None, {"available": False, "reason": "no reconstructions"}, []

    by_problem: dict[str, dict[int, list[float]]] = {}
    for r in recon:
        if r["window_kind"] == "gaussian_control" or not r.get("integrity_ok"):
            continue
        by_problem.setdefault(r["problem_id"], {}).setdefault(
            int(r["chunk_index"]), []).append(float(r["cosine"]))
    if not by_problem:
        return None, {"available": False, "reason": "no usable reconstructions"}, []

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    per_problem = []
    for pid, per_chunk in by_problem.items():
        t = traces.get(pid)
        if t is None:
            continue
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start",
                                             "char_end", "token_end", "prefix_tokens")})
                  for c in t.get("chunks", [])]
        if not chunks:
            continue
        n = len(chunks)
        scores: list[float | None] = [None] * n
        for ci, vals in per_chunk.items():
            if 0 <= ci < n:
                scores[ci] = float(np.mean(vals))
        # The NLA is only run at selected windows, so most boundaries have no
        # score. Carrying the last observed score forward is what a deployed
        # rule would have to do; boundaries before the first scored one stay
        # unscored and the rule simply cannot fire there.
        last = None
        for i in range(n):
            if scores[i] is None:
                scores[i] = last
            else:
                last = scores[i]
        per_problem.append({
            "problem_id": pid,
            "scores": scores,
            "answers": [extract_answer(prefix_text(chunks, i, t["trace"]), PERMISSIVE)
                        for i in range(n)],
            "boundary_tokens": [c.prefix_tokens for c in chunks],
            "final_answer": t.get("final_answer"),
            "gold": t.get("gold"),
            "total_tokens": int(t.get("n_tokens", 0)),
        })

    vals = [s for row in per_problem for s in row["scores"]
            if s is not None and np.isfinite(s)]
    if not vals:
        return None, {"available": False, "reason": "no finite NLA scores"}, []
    curve = sweep_threshold(per_problem, complete_threshold_grid(vals),
                            higher_fires=True)
    info = {"available": True, "n_problems": len(per_problem),
            "problem_ids": sorted(row["problem_id"] for row in per_problem),
            "score": "mean reconstruction cosine at the boundary, carried forward",
            "curve_resolution": {
                "n_distinct_scores": len({float(v) for v in vals}),
                "max_operating_points_at_any_n": granularity_bound(vals),
                "operating_points": len({round(p.mean_tokens_saved, 9)
                                         for p in curve}),
            }}
    log.info("NLA stopping curve built from %d problems", len(per_problem))
    return curve, info, per_problem


def baseline_curves_on(cfg, problem_ids, log) -> tuple[dict, dict]:
    """Re-sweep the cheap rules over one problem set.

    The NLA arm runs on fewer problems than the behavioural arms - it is an
    order of magnitude more expensive per problem, and some of its samples fail
    the integrity gate. Reading each rule off a curve built from a *different*
    set of problems and calling that a matched-budget comparison confounds the
    rule with the problems: a baseline evaluated on easier problems looks safer
    for a reason that has nothing to do with the rule.

    So when the NLA arm is present, the head-to-head is run on the problems it
    actually covers, with every baseline re-swept over exactly those. The
    full-corpus curves are still reported, because the behavioural arms'
    achieved N is a result in its own right.
    """
    from nlaast.baselines.stopping import sweep_threshold

    scores_path = cfg.dir / "baselines" / "boundary_scores.json"
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    if not scores_path.exists() or not traces_path.exists():
        return {}, {}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_baselines import build_rows, threshold_grid

    wanted = set(problem_ids)
    traces = [t for t in read_jsonl(traces_path)
              if t.get("ok") and t["problem_id"] in wanted]
    if not traces:
        return {}, {}
    base = build_rows(traces)
    scores = {pid: s for pid, s in read_json(scores_path).items() if pid in wanted}
    if not scores:
        return {}, {}

    out, rows_by_rule = {}, {}
    for rule in ("convergence", "semantic_entropy", "probe"):
        per_problem = [
            {**base[pid], "scores": scores[pid][rule]}
            for pid in base
            if pid in scores and any(s is not None for s in scores[pid].get(rule, []))
        ]
        if not per_problem:
            continue
        grid = threshold_grid(scores, rule)
        if not grid:
            continue
        out[rule] = sweep_threshold(per_problem, grid, higher_fires=True)
        rows_by_rule[rule] = per_problem
    log.info("re-swept %d baseline rules over the %d problems the NLA arm covers",
             len(out), len(traces))
    return out, rows_by_rule


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0
    manifest.start_stage(STAGE)

    out_dir = cfg.stage_dir(STAGE)
    tbl_dir = out_dir / "tables"
    tests: list[TestResult] = []
    results: dict = {}

    # ---------------------------------------------------------------- RQ1
    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    if ast_rows:
        corpus = tables.ast_corpus_table(ast_rows)
        tables.write_table(corpus, tbl_dir, "ast_corpus")
        tables.write_table(tables.ast_status_table(ast_rows), tbl_dir, "ast_status")
        results["rq1"] = read_json(cfg.dir / "ast" / "summary.json")

        with_tail = corpus[corpus["tail_start"].notna()]
        if len(with_tail) >= 3:
            # Is the tail longer on correct answers than on incorrect ones?
            # A yes would mean the construct is partly tracking difficulty,
            # which matters for F2.
            tests.append(unpaired_test(
                "RQ1: tail fraction, correct vs incorrect final answers",
                with_tail[with_tail["final_correct"]]["tail_fraction"].tolist(),
                with_tail[~with_tail["final_correct"]]["tail_fraction"].tolist(),
                cfg.analysis.confidence,
            ))

    # -------------------------------------------------------- primary RQ
    curves, budgets, curve_resolution = load_curves(cfg)
    nla_curve, nla_info, nla_rows = nla_stopping_curve(cfg, log)
    if nla_curve:
        curves[NLA_RULE] = nla_curve
    results["nla_stopping"] = nla_info

    if curves:
        from nlaast.baselines.stopping import budget_grid

        budgets = budget_grid(curves) or budgets
        comparison = tables.stopping_comparison_table(curves, budgets)
        tables.write_table(comparison, tbl_dir, "stopping_at_matched_budget")
        results["matched_budgets"] = budgets

        # Head-to-head at each matched budget, against every cheaper signal.
        if NLA_RULE in curves:
            # On the problems the NLA arm actually covers. It runs on fewer
            # problems than the behavioural arms, and comparing curves built
            # from different problem sets confounds the rule with the problems.
            restricted, restricted_rows = baseline_curves_on(
                cfg, nla_info.get("problem_ids") or [], log)
            comparison_curves = dict(curves)
            if restricted:
                comparison_curves.update(restricted)
                h2h_budgets = budget_grid({NLA_RULE: curves[NLA_RULE], **restricted})
                basis = "nla_problem_set"
            else:
                h2h_budgets = budgets
                basis = "full_corpus"
            results["o3_curve_resolution"] = {
                **curve_resolution,
                NLA_RULE: nla_info.get("curve_resolution", {}),
            }
            results["o3_comparison_basis"] = {
                "basis": basis,
                "n_problems_nla": nla_info.get("n_problems"),
                "rules_resswept": sorted(restricted),
                "budgets": h2h_budgets,
                "note": ("the head-to-head tests below are computed on the "
                         "problems the NLA arm covers, with every baseline "
                         "re-swept over exactly those problems; the table above "
                         "is the full corpus, where each arm's achieved N "
                         "differs and is reported per row"),
            }
            if restricted:
                tables.write_table(
                    tables.stopping_comparison_table(comparison_curves, h2h_budgets),
                    tbl_dir, "stopping_on_nla_problem_set")

            for other in ("convergence", "probe", "semantic_entropy"):
                if other not in comparison_curves:
                    continue
                # Only budgets where *both* rules have an operating point
                # genuinely close to the target. A rule whose nearest
                # achievable saving is far away cannot be compared there, and
                # using its nearest point anyway compares safety at different
                # budgets - the one thing the matched-budget design exists to
                # prevent.
                rows, discarded = [], 0
                for b in h2h_budgets:
                    a = at_matched_budget(comparison_curves[NLA_RULE], b)
                    c = at_matched_budget(comparison_curves[other], b)
                    if a and c:
                        rows.append((b, a, c))
                    else:
                        discarded += 1
                if not rows:
                    # Why it could not be matched decides what to do about it.
                    # The thresholds are now exhaustive, so the curve is not
                    # being failed by the sweep. Either the rule's score is
                    # too coarse to put an operating point in the window - in
                    # which case no corpus size will help and the rule needs a
                    # graded score - or the curves genuinely do not overlap,
                    # which more problems can change.
                    res = (curve_resolution or {}).get(other, {})
                    bound = res.get("max_operating_points_at_any_n")
                    n_points = len({round(p.mean_tokens_saved, 9)
                                    for p in comparison_curves[other]})
                    limited = bool(res.get("granularity_limited")) or n_points <= 3
                    reason = (
                        "the rule's score is too coarse to have an operating "
                        "point in the shared budget window; its curve has the "
                        "same few points at any corpus size, so a larger "
                        "corpus would not make this comparison evaluable"
                        if limited else
                        "no budget where both rules have an operating point "
                        "within tolerance, with the thresholds exhausted")
                    log.warning(
                        "O3: %s cannot be compared with %s at any budget - %s",
                        NLA_RULE, other, reason)
                    results.setdefault("o3_unmatched", {})[other] = {
                        "reason": reason,
                        "granularity_limited": limited,
                        "operating_points": n_points,
                        "max_operating_points_at_any_n": bound,
                        "thresholds_exhaustive": True,
                        "tolerance": MATCH_TOLERANCE,
                        "budgets_tried": len(h2h_budgets),
                    }
                    # The test stays in the pre-registered family as
                    # unevaluable rather than disappearing from it (D16):
                    # dropping it would shrink m and make the Benjamini-
                    # Hochberg correction look kinder than it is, and would
                    # hide that the comparison was planned and could not be
                    # made.
                    tests.append(TestResult(
                        name=(f"O3: safe-stopping rate, {NLA_RULE} minus "
                              f"{other} at matched budget"),
                        statistic=float("nan"), p_value=float("nan"),
                        effect=float("nan"), effect_name="cohens_d_paired",
                        n=0,
                        notes={"skipped": (
                            f"no budget at which both rules have an operating "
                            f"point within {MATCH_TOLERANCE} of the target - "
                            f"{reason}"),
                            "granularity_limited": limited,
                            "operating_points": n_points,
                            "max_operating_points_at_any_n": bound,
                            "budgets_tried": len(h2h_budgets),
                            "match_tolerance": MATCH_TOLERANCE,
                            "comparison_basis": basis},
                    ))
                    continue
                if discarded:
                    log.info("O3: %s vs %s compared at %d of %d budgets; %d "
                             "discarded as not genuinely matched",
                             NLA_RULE, other, len(rows), len(h2h_budgets), discarded)

                # Distinct operating-point pairs, not requested budgets. A
                # curve is a step function, so a span of budgets resolves to a
                # few points and the rest are repeats. Testing over the
                # repeats sets the p-value by the size of the budget grid: on
                # this pilot all nine budgets reach one pair, and a signed-rank
                # test over nine identical differences gives p = 2^-8, against
                # 0.0625 for five budgets and 1e-6 for twenty-five.
                by_point: dict[tuple[float, float], tuple[float, Any, Any]] = {}
                for b, a, c in rows:
                    by_point.setdefault((a.threshold, c.threshold), (b, a, c))
                distinct = list(by_point.values())
                if len(distinct) < len(rows):
                    log.info("O3: %s vs %s - %d budgets resolve to %d distinct "
                             "operating points; the comparison is made once per "
                             "point, over problems",
                             NLA_RULE, other, len(rows), len(distinct))

                # The comparison is made at the closest-matched point, and the
                # unit of analysis is the problem: safety is a per-problem
                # binary outcome, and both rules are scored on the same
                # problems.
                b_at, a_pt, c_pt = min(
                    distinct,
                    key=lambda r: abs(r[1].mean_tokens_saved - r[2].mean_tokens_saved))
                nla_dec = decisions_at(nla_rows, a_pt.threshold, True)
                other_dec = decisions_at(
                    restricted_rows.get(other, []) or nla_rows, c_pt.threshold, True)
                shared = sorted(
                    {d.problem_id for d in nla_dec} & {d.problem_id for d in other_dec})
                nla_safe = {d.problem_id: d.safe for d in nla_dec}
                other_safe = {d.problem_id: d.safe for d in other_dec}

                t = paired_binary_test(
                    f"O3: safe-stopping rate, {NLA_RULE} minus {other} at matched budget",
                    [nla_safe[p] for p in shared],
                    [other_safe[p] for p in shared],
                    cfg.analysis.confidence,
                )
                t.notes["n_budgets_requested"] = len(rows)
                t.notes["n_distinct_operating_points"] = len(distinct)
                t.notes["n_budgets_discarded_unmatched"] = discarded
                t.notes["match_tolerance"] = MATCH_TOLERANCE
                t.notes["budget_nla"] = a_pt.mean_tokens_saved
                t.notes[f"budget_{other}"] = c_pt.mean_tokens_saved
                t.notes["budget_gap"] = abs(
                    a_pt.mean_tokens_saved - c_pt.mean_tokens_saved)
                # Which way the difference runs, in the outcome's own units,
                # so a reader cannot take a significant q as a win.
                t.notes["safe_rate_nla"] = t.notes.pop("safe_rate_a", None)
                t.notes[f"safe_rate_{other}"] = t.notes.pop("safe_rate_b", None)
                t.notes["direction"] = (
                    "nla_readout better" if (t.notes["safe_rate_nla"] or 0)
                    > (t.notes[f"safe_rate_{other}"] or 0) else
                    "nla_readout worse" if (t.notes["safe_rate_nla"] or 0)
                    < (t.notes[f"safe_rate_{other}"] or 0) else "tied")
                t.notes["comparison_basis"] = basis
                tests.append(t)

                results.setdefault("o3_rate_differences", {})[other] = (
                    t.interval.to_dict() if t.interval else None)

    # ---------------------------------------------------------------- RQ2
    faith_path = cfg.dir / "faithfulness" / "summary.json"
    if faith_path.exists():
        results["rq2"] = read_json(faith_path)
        audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
        claims = [c for a in audits for c in a.get("claims", [])]
        if len(claims) >= 6:
            # The decisive comparison: do deletions move reconstruction more
            # than meaning-preserving paraphrases of the same claim?
            tests.append(paired_test(
                "RQ2: deletion effect minus paraphrase effect, per claim",
                [float(c["delete_effect"]) for c in claims],
                [float(np.mean(np.abs(c["paraphrase_effects"])))
                 if c.get("paraphrase_effects") else np.nan for c in claims],
                cfg.analysis.bootstrap_iterations, cfg.analysis.confidence,
                cfg.analysis.seed,
            ))
            dep = sum(1 for c in claims if c["reconstruction_dependent"])
            noise = sum(1 for c in claims if c["reproduced_under_noise"])
            results["rq2_headline"] = {
                "n_claims": len(claims),
                "reconstruction_dependent": dep / len(claims),
                "reproduced_from_noise": noise / len(claims),
                "dependent_and_not_noise": sum(
                    1 for c in claims
                    if c["reconstruction_dependent"] and not c["reproduced_under_noise"]
                ) / len(claims),
            }

    # Reconstruction by window kind (F10: is this about the tail, or about
    # looking late in a trace?)
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    if recon:
        rt = tables.reconstruction_table(recon)
        tables.write_table(rt, tbl_dir, "reconstruction_by_window")
        results["reconstruction"] = rt.to_dict(orient="records")
        tail = [r["cosine"] for r in recon if r["window_kind"] == "tail"]
        for other in ("matched_position", "matched_length", "gaussian_control"):
            vals = [r["cosine"] for r in recon if r["window_kind"] == other]
            if len(tail) >= 3 and len(vals) >= 3:
                tests.append(unpaired_test(
                    f"F10: reconstruction cosine, tail vs {other}",
                    tail, vals, cfg.analysis.confidence,
                ))

    # ---------------------------------------------------------------- RQ3
    causal_path = cfg.dir / "causal" / "summary.json"
    if causal_path.exists():
        causal = read_json(causal_path)
        results["rq3"] = causal
        rows = read_jsonl(cfg.dir / "causal" / "interventions.jsonl")
        abl = [r["verification_markers"] - r["baseline_verification_markers"]
               for r in rows if r["intervention"] == "ablate_dir"]
        rnd = [r["verification_markers"] - r["baseline_verification_markers"]
               for r in rows if r["intervention"] == "random_dir"]
        if len(abl) >= 3 and len(rnd) >= 3:
            tests.append(unpaired_test(
                "RQ3: marker change, candidate direction vs matched-random direction",
                abl, rnd, cfg.analysis.confidence,
            ))
        if rows:
            tables.write_table(pd.DataFrame(rows), tbl_dir, "interventions")

    # ------------------------------------------------- multiple testing (F9)
    tests = benjamini_hochberg(tests, cfg.analysis.fdr_q)
    tbl = tables.tests_table(tests)
    tables.write_table(tbl, tbl_dir, "hypothesis_tests")
    results["tests"] = summarise_tests(tests)
    results["fdr_q"] = cfg.analysis.fdr_q

    write_json(out_dir / "results.json", results)

    log.info("hypothesis tests: %d in the family, %d evaluated, %d significant at q=%.2f",
             results["tests"]["n_tests"], results["tests"]["n_evaluated"],
             results["tests"]["n_significant"], cfg.analysis.fdr_q)
    for t in tests:
        log.info("  %-72s effect=%+.3f q=%s", t.name[:72], t.effect,
                 "n/a" if t.q_value is None else f"{t.q_value:.4f}")

    manifest.finish_stage(STAGE, status="complete", output=str(out_dir), metrics={
        "n_tests": results["tests"]["n_tests"],
        "n_significant": results["tests"]["n_significant"],
        "rules_compared": sorted(curves),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
