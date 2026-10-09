"""Stage `baselines`: the three comparison signals (O3).

* **answer convergence** (Liu & Wang) - agreement between parsed intermediate
  answers across a window;
* **hidden-state correctness probe** (Zhang et al.) - a linear probe on the
  layer-K residual stream at each boundary;
* **semantic entropy** (Farquhar et al., symbolic-equivalence variant) - one
  confidence value per problem.

Each becomes a per-boundary score, so all of them - and later the verbalised
NLA readout - can be swept over a threshold and compared on the same
safety/saving curve at matched token budget. Comparing rules at their own
natural operating points would compare different budgets, under which a rule
that simply stops later always looks safer.

The probe is fitted on the **train** split and scored on the **eval** split,
with folds grouped by problem. Boundaries inside one trace are strongly
correlated, so an ungrouped split would report an inflated probe accuracy and
make the NLA arm look worse by comparison for the wrong reason.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from _stage import base_parser, setup, should_skip

from nlaast.activations.store import ActivationStore
from nlaast.baselines.probe import CorrectnessProbe, cross_validate
from nlaast.baselines.semantic_entropy import confidence_score, semantic_entropy
from nlaast.baselines.stopping import budget_grid, sweep_threshold
from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import read_jsonl, write_json, write_jsonl
from nlaast.trace.chunking import Chunk, prefix_text

STAGE = "baselines"


def convergence_scores(trace: dict, window: int) -> list[float]:
    """Per-boundary score for the agreement rule.

    1.0 once the last ``window`` parsed answers agree, else 0.0. Expressed as a
    score rather than a boundary so it sweeps like the others; thresholding at
    any value in (0, 1] recovers the original rule.
    """
    parsed = [b.get("parsed_answer") for b in trace.get("boundaries", [])]
    out = []
    for i in range(len(parsed)):
        lo = i - window + 1
        if lo < 0 or any(parsed[j] is None for j in range(lo, i + 1)):
            out.append(0.0)
            continue
        agree = all(equivalent(parsed[lo], parsed[j]) for j in range(lo + 1, i + 1))
        out.append(1.0 if agree else 0.0)
    return out


def build_rows(traces: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-problem scaffolding the sweep needs: answers and token counts at
    each boundary."""
    rows = {}
    for t in traces:
        chunks = [Chunk(**{k: c[k] for k in
                           ("index", "text", "char_start", "char_end",
                            "token_end", "prefix_tokens")})
                  for c in t.get("chunks", [])]
        # The answer a rule "walks away with" is the one parsed from the prefix
        # it stopped at, which is what the safety definition compares.
        answers = [extract_answer(prefix_text(chunks, i, t["trace"]), PERMISSIVE)
                   for i in range(len(chunks))]
        rows[t["problem_id"]] = {
            "problem_id": t["problem_id"],
            "dataset": t.get("dataset"),
            "split": t.get("split"),
            "answers": answers,
            "boundary_tokens": [c.prefix_tokens for c in chunks],
            "final_answer": t.get("final_answer"),
            "gold": t.get("gold"),
            "total_tokens": int(t.get("n_tokens", 0)),
            "n_chunks": len(chunks),
        }
    return rows


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    traces = [t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")]
    ast_rows = {r["problem_id"]: r for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}
    if not traces:
        log.error("no traces - run scripts/generate_traces.py first")
        return 1
    if args.limit:
        traces = traces[: args.limit]

    manifest.start_stage(STAGE, n_traces=len(traces))
    out_dir = cfg.stage_dir(STAGE)
    base = build_rows(traces)

    # -- 1. answer convergence -------------------------------------------
    conv = {t["problem_id"]: convergence_scores(t, cfg.ast.convergence_window)
            for t in traces}

    # -- 2. semantic entropy ---------------------------------------------
    entropy: dict[str, dict] = {}
    for t in traces:
        answers = [s.get("answer") for s in t.get("entropy_samples", [])]
        res = semantic_entropy(answers, t.get("final_answer"))
        entropy[t["problem_id"]] = {
            **res.to_dict(), "confidence": confidence_score(res)
        }
    write_jsonl(out_dir / "semantic_entropy.jsonl",
                [{"id": k, "problem_id": k, **v} for k, v in entropy.items()])

    # -- 3. hidden-state correctness probe --------------------------------
    probe_report, probe_scores, probe_info = fit_probe(cfg, traces, base, log)
    write_json(out_dir / "probe.json", {"report": probe_report, **probe_info})

    # -- per-boundary score table ----------------------------------------
    scores: dict[str, dict[str, list[float | None]]] = {}
    for pid, row in base.items():
        n = row["n_chunks"]
        scores[pid] = {
            "convergence": conv.get(pid, [0.0] * n),
            # Semantic entropy is a per-problem quantity, so it is constant
            # across boundaries. That is a real property of the signal, not a
            # modelling shortcut: it cannot say *when* to stop, only how
            # confident the answer is. The sweep treats it accordingly.
            "semantic_entropy": [entropy.get(pid, {}).get("confidence")] * n,
            "probe": probe_scores.get(pid, [None] * n),
        }
    write_json(out_dir / "boundary_scores.json", scores)

    # -- sweeps at matched budget ----------------------------------------
    curves = {}
    for rule in ("convergence", "semantic_entropy", "probe"):
        per_problem = [
            {**base[pid], "scores": scores[pid][rule]}
            for pid in base
            if any(s is not None for s in scores[pid][rule])
        ]
        if not per_problem:
            log.warning("rule %s has no usable scores", rule)
            continue
        thresholds = threshold_grid(scores, rule)
        curves[rule] = sweep_threshold(per_problem, thresholds, higher_fires=True)

    budgets = budget_grid(curves)
    write_json(out_dir / "curves.json", {
        "curves": {k: [p.to_dict() for p in v] for k, v in curves.items()},
        "matched_budgets": budgets,
    })

    metrics = {
        "n_problems": len(base),
        "rules": sorted(curves),
        "probe_auc": probe_report.get("auc"),
        "probe_accuracy": probe_report.get("accuracy"),
        "probe_n": probe_report.get("n_eval"),
        "mean_semantic_entropy": float(np.mean(
            [v["semantic_entropy"] for v in entropy.values()]
        )) if entropy else float("nan"),
        "mean_entropy_clusters": float(np.mean(
            [v["n_clusters"] for v in entropy.values()]
        )) if entropy else float("nan"),
        "matched_budgets": budgets,
    }
    log.info("baselines: %s", metrics)
    manifest.finish_stage(STAGE, status="complete", output=str(out_dir), metrics=metrics)
    return 0


def threshold_grid(scores: dict, rule: str, n: int = 25) -> list[float]:
    vals = [s for p in scores.values() for s in p[rule]
            if s is not None and np.isfinite(s)]
    if not vals:
        return []
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        # A constant score still needs two points so the curve has a
        # "never fires" end and an "always fires" end.
        return [lo - 1e-6, lo + 1e-6]
    return list(np.linspace(lo, hi, n))


def fit_probe(cfg, traces, base, log):
    """Train the correctness probe on the train split, score every boundary."""
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer

    X, y, groups, index = [], [], [], []
    for t in traces:
        pid = t["problem_id"]
        if not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        answers = base[pid]["answers"]
        gold = base[pid]["gold"]
        for row_i, chunk_i in enumerate(meta.get("chunk_indices", range(arr.shape[0]))):
            if chunk_i >= len(answers):
                continue
            ans = answers[chunk_i]
            if ans is None:
                # No intermediate answer means there is nothing whose
                # correctness the probe could be predicting.
                continue
            X.append(arr[row_i])
            y.append(1 if equivalent(ans, gold) else 0)
            groups.append(pid)
            index.append((pid, chunk_i))

    if len(X) < 20:
        log.warning("only %d probe examples - skipping the probe", len(X))
        return ({"error": f"insufficient data ({len(X)} examples)"}, {},
                {"n_examples": len(X)})

    X = np.vstack(X)
    y = np.asarray(y)
    log.info("probe data: %d boundaries, %d problems, positive rate %.3f",
             len(y), len(set(groups)), y.mean())

    cv = cross_validate(X, y, groups, cfg.probe)
    report = cv.to_dict()
    log.info("probe (grouped CV): accuracy=%.3f auc=%.3f brier=%.3f",
             report["accuracy"], report["auc"], report["brier"])

    # Per-boundary scores for the stopping sweep come from the grouped
    # cross-validation's out-of-fold predictions, so every score was produced
    # by a probe that never saw that problem. Fitting on the train split and
    # scoring everything would put in-sample probabilities on the train-split
    # problems, and those problems are in the sweep too - which would inflate
    # the baseline the primary research question is measured against, in the
    # direction that makes the verbalised readout look worse for the wrong
    # reason. The grouped folds are hashed by problem id, so this is leak-free
    # and uses every problem rather than only the eval split.
    oof = np.asarray(cv.oof, dtype=float) if cv.oof else np.full(len(y), np.nan)
    by_problem: dict[str, list[float | None]] = {
        pid: [None] * base[pid]["n_chunks"] for pid in base
    }
    n_scored = 0
    for (pid, chunk_i), prob in zip(index, oof):
        if chunk_i < len(by_problem[pid]) and np.isfinite(prob):
            by_problem[pid][chunk_i] = float(prob)
            n_scored += 1

    # A probe is also fitted on the train split alone - not for the sweep, but
    # because the causal stage and the report quote its direction, and because
    # the eval-split metrics are the ones comparable with earlier runs.
    train_mask = np.array([base[g]["split"] == "train" for g in groups])
    eval_mask = ~train_mask
    train_split_fit: dict[str, object] = {"available": False}
    if train_mask.sum() >= 10 and len(np.unique(y[train_mask])) >= 2:
        fitted = CorrectnessProbe(cfg.probe).fit(X[train_mask], y[train_mask])
        train_split_fit = {
            "available": True,
            "n_train": int(train_mask.sum()),
            "n_eval": int(eval_mask.sum()),
            "direction_norm": float(np.linalg.norm(fitted.direction)),
        }
        if eval_mask.sum() >= 3 and len(np.unique(y[eval_mask])) >= 2:
            from sklearn.metrics import roc_auc_score

            train_split_fit["eval_auc"] = float(
                roc_auc_score(y[eval_mask], fitted.predict_proba(X[eval_mask]))
            )
    else:
        log.warning("train split too small or single-class (%d rows, %d classes); "
                    "no train-split probe fit. The sweep is unaffected - it uses "
                    "out-of-fold scores.",
                    int(train_mask.sum()), len(np.unique(y[train_mask])))

    info = {
        "n_examples": int(len(y)),
        "n_problems": len(set(groups)),
        "positive_rate": float(y.mean()),
        "probe_source": "grouped_cv_out_of_fold",
        "n_boundaries_scored": n_scored,
        "n_boundaries_unscored": int(len(y) - n_scored),
        "layer": layer,
        "train_split_fit": train_split_fit,
    }
    return report, by_problem, info


if __name__ == "__main__":
    sys.exit(main())
