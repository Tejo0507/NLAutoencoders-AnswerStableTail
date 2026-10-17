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


def rq3_paired_contrast(rows: list[dict], confidence: float) -> dict[str, Any]:
    """The RQ3 family test: candidate vs matched-random, paired over problems.

    Built from ``interventions.jsonl`` rather than from the summary, because
    the family test needs the per-problem values. It reuses the causal module's
    own definitions - the eligible outcome, the dose the controls run at, and
    the magnitude convention - so this test and the verdict in
    ``causal/summary.json`` are the same contrast rather than two different
    ones with the same name.

    Returns the basis (counts, dose, outcome) always, and a ``TestResult`` only
    when the contrast could actually be made. A contrast that could not be made
    contributes no p-value, which is what keeps it out of the Benjamini-
    Hochberg denominator while staying visible (D38).
    """
    from nlaast.causal.interventions import (
        VERDICT_OUTCOME,
        InterventionResult,
        paired_arm_contrast,
    )
    from nlaast.analysis.stats import paired_test

    if not rows:
        return {"available": False, "reason": "no intervention rows"}

    # `paired_arm_contrast` works on InterventionResult, so rehydrate the
    # fields it reads rather than duplicating its pairing logic here.
    recs, skipped = [], 0
    for r in rows:
        try:
            recs.append(InterventionResult(
                problem_id=r["problem_id"],
                intervention=r["intervention"],
                coefficient=float(r.get("coefficient", 0.0)),
                answer=r.get("answer"), baseline_answer=r.get("baseline_answer"),
                gold=r.get("gold"),
                answer_changed=bool(r.get("answer_changed", False)),
                correct=bool(r.get("correct", False)),
                baseline_correct=bool(r.get("baseline_correct", False)),
                tokens_generated=int(r.get("tokens_generated", 0) or 0),
                verification_markers=int(r.get("verification_markers", 0) or 0),
                baseline_verification_markers=int(
                    r.get("baseline_verification_markers", 0) or 0),
                direction_id=r.get("direction_id"),
                notes=r.get("notes") or {},
            ))
        except (KeyError, TypeError, ValueError):
            skipped += 1

    c = paired_arm_contrast(recs, "ablate_dir", "random_dir",
                            outcome=VERDICT_OUTCOME, compare_at=1.0,
                            confidence=confidence)
    out: dict[str, Any] = {"available": True, "malformed_rows_skipped": skipped,
                           **c}
    name = ("RQ3: marker change, candidate direction vs matched-random "
            "direction")

    # Did the independent variable actually get applied?
    #
    # This decides whether an identical-arms result is a *tested null* or a
    # broken experiment, and the two must not be reported the same way. D38
    # established that exact agreement between two arms is a null that was
    # tested and should count towards the Benjamini-Hochberg denominator. That
    # reasoning assumes the arms differed in what was done to them. Under D34
    # the pilot's direction arms edited no positions at all, so their
    # agreement says nothing about the direction - it says the edit never
    # happened.
    #
    # So the test is only evaluable when the candidate demonstrably edited
    # something. An unrecorded count (pilot artifacts predate the counter)
    # counts as not demonstrated, which is the conservative reading.
    #
    # A problem whose continuation generated no tokens has no positions the
    # edit could reach - the span covers the generated continuation - so its
    # zero count says nothing about whether the editor works. It is counted
    # and excluded from the requirement rather than allowed to void the test
    # (on `main`, one trivial-tail problem of 60 voided it this way).
    cand = [r for r in recs if r.intervention == "ablate_dir"
            and abs(r.coefficient - 1.0) < 1e-9]
    no_continuation = [r for r in cand if r.tokens_generated <= 0]
    edits = [int((r.notes or {}).get("n_positions_edited", -1)) for r in cand
             if r.tokens_generated > 0]
    known = [e for e in edits if e >= 0]
    out["candidate_edits_recorded"] = len(known)
    out["candidate_edits_min"] = min(known) if known else None
    out["candidate_rows_without_continuation"] = len(no_continuation)
    out["intervention_demonstrated"] = bool(known) and all(e > 0 for e in known)
    if not out["intervention_demonstrated"]:
        out["test_result"] = TestResult(
            name=name, statistic=float("nan"), p_value=float("nan"),
            effect=float("nan"), effect_name="cohens_d_paired",
            n=int(c.get("n_pairs", 0)),
            notes={"skipped": (
                "the candidate arm is not demonstrated to have edited any "
                "position at dose 1.0, so the arms' agreement is evidence "
                "about the intervention rather than about the direction "
                "(D34); no p-value is reported and this test does not enter "
                "the BH denominator"),
                "test": "wilcoxon_signed_rank_paired_on_problems",
                "outcome": VERDICT_OUTCOME, "dose": 1.0,
                "n_pairs": c.get("n_pairs", 0),
                "candidate_edits_recorded": len(known),
                "broken_independent_variable": True},
        )
        return out

    if c.get("p_value") is None:
        out["test_result"] = TestResult(
            name=name, statistic=float("nan"), p_value=float("nan"),
            effect=float("nan"), effect_name="cohens_d_paired",
            n=int(c.get("n_pairs", 0)),
            notes={"skipped": c.get("reason", "contrast could not be made"),
                   "test": "wilcoxon_signed_rank_paired_on_problems",
                   "outcome": VERDICT_OUTCOME, "dose": 1.0,
                   "n_pairs": c.get("n_pairs", 0)},
        )
        return out

    # Re-derive the identical test from the contrast's own paired arrays, so
    # the family carries the interval and effect size rather than a
    # reconstructed shell.
    t = paired_test(name, c["paired_candidate_values"],
                    c["paired_control_values"], confidence=confidence)
    if not np.isfinite(t.p_value) and c.get("p_value") == 1.0:
        # Exactly equal arms, with the intervention demonstrated. Wilcoxon is
        # undefined once the zero differences are dropped, but the two-sided
        # exact answer is p = 1.0 by the same argument as D38: this is a null
        # that was tested and not rejected, so it counts towards m rather
        # than disappearing from the family.
        t.p_value = 1.0
        t.statistic = 0.0
    t.notes = {
        "test": "wilcoxon_signed_rank_paired_on_problems",
        "unit_of_analysis": "problem",
        "outcome": VERDICT_OUTCOME,
        "dose": c.get("dose"),
        "n_pairs": c["n_pairs"],
        "n_pairs_favouring_candidate": c.get("n_pairs_favouring_candidate"),
        "n_pairs_favouring_control": c.get("n_pairs_favouring_control"),
        "n_pairs_tied": c.get("n_pairs_tied"),
        "median_paired_difference": c.get("median_paired_difference"),
        "beats": c.get("beats"),
    }
    out["test_result"] = t
    return out


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

    # Two exclusions, both counted, and they are not the same kind. The
    # Gaussian control is dropped by design - it is the noise arm. The
    # integrity gate drops samples whose verbalisation is not well-formed
    # English, and the arm's score is a mean over what survives it, so a rate
    # that moves between runs changes what the number is a mean of.
    #
    # Recording only the surviving problem count hid both. The pilot excludes
    # 132 of 378 samples here while `n_problems` stays at 21 - a problem keeps
    # its remaining samples - and nothing said whether those 132 were 126
    # controls and 6 gate failures (they are, a 2.4% gate rate matching the
    # nla stage's own 0.976) or something closer to a compromised arm.
    by_problem: dict[str, dict[int, list[float]]] = {}
    excluded = {"gaussian_control": 0, "integrity_failed": 0}
    for r in recon:
        if r["window_kind"] == "gaussian_control":
            excluded["gaussian_control"] += 1
            continue
        if not r.get("integrity_ok"):
            excluded["integrity_failed"] += 1
            continue
        by_problem.setdefault(r["problem_id"], {}).setdefault(
            int(r["chunk_index"]), []).append(float(r["cosine"]))
    used = sum(len(v) for per in by_problem.values() for v in per.values())
    exclusions = {
        "n_samples_total": len(recon),
        "n_samples_used": used,
        "n_samples_excluded_as_control": excluded["gaussian_control"],
        "n_samples_excluded_by_integrity_gate": excluded["integrity_failed"],
        "integrity_exclusion_rate": (
            excluded["integrity_failed"]
            / (len(recon) - excluded["gaussian_control"])
            if len(recon) - excluded["gaussian_control"] else 0.0),
        "note": ("the Gaussian control is excluded by design - it is the noise "
                 "arm, not a readout; the integrity gate excludes samples "
                 "whose verbalisation is not well-formed English, so the score "
                 "is a mean over what survived both"),
    }
    if not by_problem:
        return None, {"available": False, "reason": "no usable reconstructions",
                      "exclusions": exclusions}, []
    if exclusions["integrity_exclusion_rate"] > 0.25:
        log.warning("NLA stopping arm: the integrity gate excluded %.1f%% of "
                    "activation-driven samples (%d of %d); the arm's score is "
                    "a mean over the remainder and should be read with the "
                    "nla stage's own integrity rate",
                    100 * exclusions["integrity_exclusion_rate"],
                    excluded["integrity_failed"],
                    len(recon) - excluded["gaussian_control"])

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    per_problem = []
    dropped_no_trace, dropped_no_chunks = [], []
    for pid, per_chunk in by_problem.items():
        t = traces.get(pid)
        if t is None:
            dropped_no_trace.append(pid)
            continue
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start",
                                             "char_end", "token_end", "prefix_tokens")})
                  for c in t.get("chunks", [])]
        if not chunks:
            dropped_no_chunks.append(pid)
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

    # A problem with reconstructions but no usable trace row cannot be scored.
    # It is an exclusion like the others and is named, not merely subtracted:
    # these are the problems the arm covers on paper and not in the comparison.
    exclusions["n_problems_with_usable_samples"] = len(by_problem)
    exclusions["n_problems_scored"] = len(per_problem)
    exclusions["problems_dropped_no_usable_trace"] = sorted(dropped_no_trace)
    exclusions["problems_dropped_no_chunks"] = sorted(dropped_no_chunks)
    if dropped_no_trace or dropped_no_chunks:
        log.warning("NLA stopping arm: %d of %d problems with usable samples "
                    "could not be scored (%d without a usable trace row, %d "
                    "without chunks)", len(by_problem) - len(per_problem),
                    len(by_problem), len(dropped_no_trace),
                    len(dropped_no_chunks))

    vals = [s for row in per_problem for s in row["scores"]
            if s is not None and np.isfinite(s)]
    if not vals:
        return None, {"available": False, "reason": "no finite NLA scores",
                      "exclusions": exclusions}, []
    curve = sweep_threshold(per_problem, complete_threshold_grid(vals),
                            higher_fires=True)
    info = {"available": True, "n_problems": len(per_problem),
            "problem_ids": sorted(row["problem_id"] for row in per_problem),
            "score": "mean reconstruction cosine at the boundary, carried forward",
            "exclusions": exclusions,
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


def o3_head_to_head(
    other: str,
    nla_curve: list,
    other_curve: list,
    nla_rows: list[dict],
    other_rows: list[dict] | None,
    h2h_budgets: list[float],
    basis: str,
    full_resolution: dict,
    confidence: float,
    log,
) -> tuple[TestResult, dict | None]:
    """One head-to-head: the readout against one cheaper rule, at matched budget.

    Returns the family's ``TestResult`` always, and an ``o3_unmatched`` record
    when the comparison could not be made. A comparison that could not be made
    is a result - it was planned and is reported as absent rather than dropped
    (D16) - and it carries no p-value, so it does not enter the
    Benjamini-Hochberg denominator (D37, D38).

    Lifted out of ``main`` so the study's primary comparison can be exercised
    without a run directory. It had no test of its own, and the refusal below
    is the reason it needed one.
    """
    name = (f"O3: safe-stopping rate, {NLA_RULE} minus {other} "
            f"at matched budget")

    # The paired test needs the other rule's own per-problem scores, not merely
    # its curve: `decisions_at` reads `row['scores']`, so handing it a
    # different rule's rows computes *that* rule's decisions at this rule's
    # threshold.
    #
    # This used to fall back to `nla_rows` when the rule had no re-swept rows,
    # which is exactly that substitution - the baseline's threshold applied to
    # the NLA readout's cosines, reported as a McNemar test between the two
    # rules. It is reachable whenever a baseline is scored on the full corpus
    # but not on the problems the NLA arm covers, which is a per-rule
    # condition: `baseline_curves_on` omits a rule with no usable score on
    # that subset while keeping the others.
    #
    # A comparison without the rule's own rows is not a null result, it is an
    # absent one, and it is recorded as such.
    if not other_rows:
        reason = (
            f"{other} has a curve but no per-problem scores on the {basis} "
            f"the comparison is made on, so the paired test has no rows of "
            f"its own to run over; scoring {other}'s threshold against the "
            f"readout's own values would compare the readout with itself")
        log.warning("O3: %s cannot be compared with %s - %s",
                    NLA_RULE, other, reason)
        return (
            TestResult(
                name=name, statistic=float("nan"), p_value=float("nan"),
                effect=float("nan"), effect_name="cohens_d_paired", n=0,
                notes={"skipped": reason,
                       "per_problem_scores_available": False,
                       "comparison_basis": basis},
            ),
            {"reason": reason,
             "granularity_limited": False,
             "per_problem_scores_available": False,
             "comparison_basis": basis},
        )

    # Only budgets where *both* rules have an operating point genuinely close
    # to the target. A rule whose nearest achievable saving is far away cannot
    # be compared there, and using its nearest point anyway compares safety at
    # different budgets - the one thing the matched-budget design exists to
    # prevent.
    rows, discarded = [], 0
    for b in h2h_budgets:
        a = at_matched_budget(nla_curve, b)
        c = at_matched_budget(other_curve, b)
        if a and c:
            rows.append((b, a, c))
        else:
            discarded += 1

    if not rows:
        # Why it could not be matched decides what to do about it. The
        # thresholds are now exhaustive, so the curve is not being failed by
        # the sweep. Either the rule's score is too coarse to put an operating
        # point in the window - in which case the rule needs a graded score -
        # or the curves simply do not overlap on this corpus.
        #
        # Neither verdict licenses a claim about *all* corpus sizes. The
        # resolution bound is computed from the scores observed here, so a
        # score whose range grows with the data has a bound that grows too
        # (granularity_bound).
        #
        # It is computed from the scores on the problems actually compared,
        # not from `curve_resolution`, which `score_baselines` writes over the
        # whole corpus. The two scopes differ - on the pilot the probe takes
        # 272 distinct values corpus-wide and 203 on the 21 problems the NLA
        # arm covers - and a reason that quotes a count from one scope to
        # explain a failure in the other describes a different curve from the
        # one that could not be matched. The corpus-wide figures are kept
        # beside it rather than dropped.
        vals = [s for r in other_rows for s in r["scores"]
                if s is not None and np.isfinite(s)]
        n_distinct = len({float(v) for v in vals})
        bound = granularity_bound(vals) if vals else None
        n_points = len({round(p.mean_tokens_saved, 9) for p in other_curve})
        limited = bool(bound is not None and bound <= 3) or n_points <= 3
        reason = (
            f"the rule's score is too coarse to have an operating point in "
            f"the shared budget window: it takes {n_distinct} distinct values "
            f"on the {len(other_rows)} problems compared, bounding its curve "
            f"at {bound} points. A larger corpus helps only if it widens the "
            f"score's range; if the score is bounded by construction it "
            f"cannot, and the rule needs a graded score"
            if limited else
            f"the thresholds are exhausted, so the two curves do not come "
            f"within tolerance of a shared budget on this corpus rather than "
            f"the sweep having missed a point; {other} reaches {n_points} "
            f"distinct savings here")
        log.warning("O3: %s cannot be compared with %s at any budget - %s",
                    NLA_RULE, other, reason)
        return (
            TestResult(
                name=name, statistic=float("nan"), p_value=float("nan"),
                effect=float("nan"), effect_name="cohens_d_paired", n=0,
                notes={"skipped": (
                    f"no budget at which both rules have an operating point "
                    f"within {MATCH_TOLERANCE} of the target - {reason}"),
                    "granularity_limited": limited,
                    "operating_points": n_points,
                    "max_operating_points_at_any_n": bound,
                    "n_distinct_scores": n_distinct,
                    "n_problems_compared": len(other_rows),
                    "budgets_tried": len(h2h_budgets),
                    "match_tolerance": MATCH_TOLERANCE,
                    "comparison_basis": basis},
            ),
            {"reason": reason,
             "granularity_limited": limited,
             "operating_points": n_points,
             "max_operating_points_at_any_n": bound,
             "n_distinct_scores": n_distinct,
             "thresholds_exhaustive": True,
             "tolerance": MATCH_TOLERANCE,
             "budgets_tried": len(h2h_budgets),
             # Which problems the numbers above describe, and the corpus-wide
             # figures beside them, so the two scopes cannot be read as one.
             "diagnostics_basis": basis,
             "n_problems_compared": len(other_rows),
             "full_corpus_resolution": full_resolution or None},
        )

    if discarded:
        log.info("O3: %s vs %s compared at %d of %d budgets; %d discarded as "
                 "not genuinely matched",
                 NLA_RULE, other, len(rows), len(h2h_budgets), discarded)

    # Distinct operating-point pairs, not requested budgets. A curve is a step
    # function, so a span of budgets resolves to a few points and the rest are
    # repeats. Testing over the repeats sets the p-value by the size of the
    # budget grid: on the pilot all nine budgets reach one pair, and a
    # signed-rank test over nine identical differences gives p = 2^-8, against
    # 0.0625 for five budgets and 1e-6 for twenty-five.
    by_point: dict[tuple[float, float], tuple[float, Any, Any]] = {}
    for b, a, c in rows:
        by_point.setdefault((a.threshold, c.threshold), (b, a, c))
    distinct = list(by_point.values())
    if len(distinct) < len(rows):
        log.info("O3: %s vs %s - %d budgets resolve to %d distinct operating "
                 "points; the comparison is made once per point, over problems",
                 NLA_RULE, other, len(rows), len(distinct))

    # The comparison is made at the closest-matched point, and the unit of
    # analysis is the problem: safety is a per-problem binary outcome, and both
    # rules are scored on the same problems.
    _, a_pt, c_pt = min(
        distinct,
        key=lambda r: abs(r[1].mean_tokens_saved - r[2].mean_tokens_saved))
    nla_dec = decisions_at(nla_rows, a_pt.threshold, True)
    other_dec = decisions_at(other_rows, c_pt.threshold, True)
    shared = sorted(
        {d.problem_id for d in nla_dec} & {d.problem_id for d in other_dec})
    nla_safe = {d.problem_id: d.safe for d in nla_dec}
    other_safe = {d.problem_id: d.safe for d in other_dec}

    t = paired_binary_test(name, [nla_safe[p] for p in shared],
                           [other_safe[p] for p in shared], confidence)
    t.notes["n_budgets_requested"] = len(rows)
    t.notes["n_distinct_operating_points"] = len(distinct)
    t.notes["n_budgets_discarded_unmatched"] = discarded
    t.notes["match_tolerance"] = MATCH_TOLERANCE
    t.notes["budget_nla"] = a_pt.mean_tokens_saved
    t.notes[f"budget_{other}"] = c_pt.mean_tokens_saved
    t.notes["budget_gap"] = abs(a_pt.mean_tokens_saved - c_pt.mean_tokens_saved)
    # How many problems each arm was scored on, and how many they share. The
    # test runs on the intersection, so a silent loss of problems between the
    # two arms would otherwise show up only as a smaller `n_problems`.
    t.notes["n_problems_nla_arm"] = len(nla_dec)
    t.notes[f"n_problems_{other}_arm"] = len(other_dec)
    t.notes["n_problems_shared"] = len(shared)
    # Which way the difference runs, in the outcome's own units, so a reader
    # cannot take a significant q as a win.
    t.notes["safe_rate_nla"] = t.notes.pop("safe_rate_a", None)
    t.notes[f"safe_rate_{other}"] = t.notes.pop("safe_rate_b", None)
    t.notes["direction"] = (
        "nla_readout better" if (t.notes["safe_rate_nla"] or 0)
        > (t.notes[f"safe_rate_{other}"] or 0) else
        "nla_readout worse" if (t.notes["safe_rate_nla"] or 0)
        < (t.notes[f"safe_rate_{other}"] or 0) else "tied")
    t.notes["comparison_basis"] = basis
    return t, None


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
                t, unmatched = o3_head_to_head(
                    other,
                    comparison_curves[NLA_RULE], comparison_curves[other],
                    nla_rows, restricted_rows.get(other),
                    h2h_budgets, basis,
                    (curve_resolution or {}).get(other, {}),
                    cfg.analysis.confidence, log,
                )
                tests.append(t)
                if unmatched is not None:
                    results.setdefault("o3_unmatched", {})[other] = unmatched
                else:
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
        # Paired over problems and read at the controls' dose.
        #
        # This test was `unpaired_test` (Mann-Whitney U) over every
        # `ablate_dir` row against every `random_dir` row, which was wrong
        # twice. Both arms run on the *same* problems, so discarding the
        # pairing is the error D29 corrected in the primary O3 comparison -
        # the unit of analysis is the problem. And pooling the candidate's
        # whole dose sweep put dose 0.0 into the candidate arm, which is a
        # no-op edit by definition, diluting the candidate against undiluted
        # controls; `summarise_arm`'s docstring already warned against exactly
        # that, so the registered test contradicted the gate beside it.
        #
        # `rq3_paired_contrast` rebuilds the same contrast the gate uses, so
        # the family test and the verdict cannot disagree.
        pair = rq3_paired_contrast(rows, cfg.analysis.confidence)
        results["rq3_family_test_basis"] = {
            k: v for k, v in pair.items() if k != "test_result"}
        t = pair.get("test_result")
        if t is not None:
            tests.append(t)
        if rows:
            tables.write_table(pd.DataFrame(rows), tbl_dir, "interventions")

    # ------------------------------------------------- multiple testing (F9)
    tests = benjamini_hochberg(tests, cfg.analysis.fdr_q)
    tbl = tables.tests_table(tests)
    tables.write_table(tbl, tbl_dir, "hypothesis_tests")
    results["tests"] = summarise_tests(tests, cfg.analysis.fdr_q)
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
