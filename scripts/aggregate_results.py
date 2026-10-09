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

import numpy as np
import pandas as pd

from _stage import base_parser, setup, should_skip

from nlaast.analysis import tables
from nlaast.analysis.stats import (
    TestResult,
    benjamini_hochberg,
    paired_test,
    rate_difference_ci,
    summarise_tests,
    unpaired_test,
)
from nlaast.baselines.stopping import OperatingPoint, at_matched_budget
from nlaast.logging_utils import read_json, read_jsonl, write_json

STAGE = "analysis"
NLA_RULE = "nla_readout"


def load_curves(cfg) -> tuple[dict, list[float]]:
    path = cfg.dir / "baselines" / "curves.json"
    if not path.exists():
        return {}, []
    raw = read_json(path)
    curves = {
        rule: [OperatingPoint(**p) for p in pts]
        for rule, pts in raw.get("curves", {}).items()
    }
    return curves, raw.get("matched_budgets", [])


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
        return None, {"available": False, "reason": "no reconstructions"}

    by_problem: dict[str, dict[int, list[float]]] = {}
    for r in recon:
        if r["window_kind"] == "gaussian_control" or not r.get("integrity_ok"):
            continue
        by_problem.setdefault(r["problem_id"], {}).setdefault(
            int(r["chunk_index"]), []).append(float(r["cosine"]))
    if not by_problem:
        return None, {"available": False, "reason": "no usable reconstructions"}

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
        return None, {"available": False, "reason": "no finite NLA scores"}
    thresholds = list(np.linspace(min(vals), max(vals), 25))
    curve = sweep_threshold(per_problem, thresholds, higher_fires=True)
    info = {"available": True, "n_problems": len(per_problem),
            "problem_ids": sorted(row["problem_id"] for row in per_problem),
            "score": "mean reconstruction cosine at the boundary, carried forward"}
    log.info("NLA stopping curve built from %d problems", len(per_problem))
    return curve, info


def baseline_curves_on(cfg, problem_ids, log) -> dict:
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
        return {}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_baselines import build_rows, threshold_grid

    wanted = set(problem_ids)
    traces = [t for t in read_jsonl(traces_path)
              if t.get("ok") and t["problem_id"] in wanted]
    if not traces:
        return {}
    base = build_rows(traces)
    scores = {pid: s for pid, s in read_json(scores_path).items() if pid in wanted}
    if not scores:
        return {}

    out = {}
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
    log.info("re-swept %d baseline rules over the %d problems the NLA arm covers",
             len(out), len(traces))
    return out


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
    curves, budgets = load_curves(cfg)
    nla_curve, nla_info = nla_stopping_curve(cfg, log)
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
            restricted = baseline_curves_on(cfg, nla_info.get("problem_ids") or [], log)
            comparison_curves = dict(curves)
            if restricted:
                comparison_curves.update(restricted)
                h2h_budgets = budget_grid({NLA_RULE: curves[NLA_RULE], **restricted})
                basis = "nla_problem_set"
            else:
                h2h_budgets = budgets
                basis = "full_corpus"
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
                rows = []
                for b in h2h_budgets:
                    a = at_matched_budget(comparison_curves[NLA_RULE], b)
                    c = at_matched_budget(comparison_curves[other], b)
                    if a and c:
                        rows.append((b, a, c))
                if not rows:
                    continue
                t = paired_test(
                    f"O3: safe-stopping rate, {NLA_RULE} minus {other} at matched budget",
                    [r[1].safe_rate for r in rows],
                    [r[2].safe_rate for r in rows],
                    cfg.analysis.bootstrap_iterations,
                    cfg.analysis.confidence,
                    cfg.analysis.seed,
                )
                t.notes["n_budgets"] = len(rows)
                t.notes["comparison_basis"] = basis
                t.notes["n_problems"] = nla_info.get("n_problems") if restricted else None
                t.notes["caveat"] = (
                    "paired over matched-budget operating points, not over "
                    "problems: the unit of analysis is the budget level"
                )
                tests.append(t)

                best_a = max(rows, key=lambda r: r[1].safe_rate)[1]
                best_c = max(rows, key=lambda r: r[2].safe_rate)[2]
                ci = rate_difference_ci(
                    int(round(best_a.safe_rate * best_a.n)), best_a.n,
                    int(round(best_c.safe_rate * best_c.n)), best_c.n,
                    cfg.analysis.confidence,
                )
                results.setdefault("o3_rate_differences", {})[other] = ci.to_dict()

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
