"""Machine-readable result tables.

Every table is written twice: CSV for inspection and JSON for programmatic
reuse. Nothing is hand-assembled, so the reporting stage can be re-run from
saved stage outputs without touching a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get

log = get(__name__)


def write_table(df: pd.DataFrame, out_dir: Path, name: str,
                index: bool = False) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    js = out_dir / f"{name}.json"
    df.to_csv(csv, index=index)
    df.to_json(js, orient="records", indent=2)
    log.info("wrote table %s (%d rows)", name, len(df))
    return {"csv": str(csv), "json": str(js)}


def markdown(df: pd.DataFrame, floatfmt: str = "%.3f", max_rows: int = 60) -> str:
    """GitHub-flavoured markdown for the execution report."""
    if df.empty:
        return "_(no rows)_"
    shown = df.head(max_rows).copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(
                lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v))
                else floatfmt % v
            )
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    out = "\n".join([header, rule, *body])
    if len(df) > max_rows:
        out += f"\n\n_({len(df) - max_rows} further rows omitted)_"
    return out


def ast_corpus_table(ast_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Per-problem tail corpus - the O6 release artefact."""
    return pd.DataFrame(
        [
            {
                "problem_id": r["problem_id"],
                "dataset": r.get("dataset"),
                "split": r.get("split"),
                "level": r.get("level"),
                "status": r["status"],
                "n_chunks": r["n_chunks"],
                "tail_start": r["tail_start"],
                "tail_fraction": r["tail_fraction"],
                "tail_tokens": r["tail_tokens"],
                "total_tokens": r["total_tokens"],
                "convergence_start": r.get("convergence_start"),
                "final_correct": r["final_correct"],
            }
            for r in ast_rows
        ]
    )


def ast_status_table(ast_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Edge-case accounting. Required because the plan says edge cases are
    recorded rather than dropped - this is where that is visible."""
    df = ast_corpus_table(ast_rows)
    if df.empty:
        return df
    out = (
        df.groupby("status")
        .agg(n=("problem_id", "count"),
             mean_tail_fraction=("tail_fraction", "mean"),
             mean_chunks=("n_chunks", "mean"),
             accuracy=("final_correct", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    out["share"] = out["n"] / out["n"].sum()
    return out


def stopping_comparison_table(curves: dict[str, Sequence[Any]],
                              budgets: Sequence[float]) -> pd.DataFrame:
    """The O3 headline: every rule read off its own curve at matched budget."""
    from ..baselines.stopping import at_matched_budget

    rows = []
    for budget in budgets:
        for rule, curve in curves.items():
            pt = at_matched_budget(curve, budget)
            if pt is None:
                continue
            rows.append(
                {
                    "target_budget": budget,
                    "rule": rule,
                    "threshold": pt.threshold,
                    "n": pt.n,
                    "achieved_tokens_saved": pt.mean_tokens_saved,
                    "safe_rate": pt.safe_rate,
                    "accuracy": pt.accuracy,
                    "fire_rate": pt.fire_rate,
                }
            )
    return pd.DataFrame(rows)


def reconstruction_table(nla_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Reconstruction fidelity by window kind - the F10 position control."""
    if not nla_rows:
        return pd.DataFrame()
    df = pd.DataFrame(nla_rows)
    if "window_kind" not in df or "cosine" not in df:
        return pd.DataFrame()
    agg = (
        df.groupby("window_kind")
        .agg(
            n=("cosine", "count"),
            mean_cosine=("cosine", "mean"),
            median_cosine=("cosine", "median"),
            std_cosine=("cosine", "std"),
            mean_mse=("mse", "mean"),
        )
        .reset_index()
    )
    # Same convention as the released checkpoints: MSE = 2(1 - cos), and FVE
    # against an orthogonal-prediction baseline of 2.0.
    agg["fve"] = 1.0 - agg["mean_mse"] / 2.0
    return agg


def tests_table(tests: Sequence[Any]) -> pd.DataFrame:
    rows = []
    for t in tests:
        d = t.to_dict() if hasattr(t, "to_dict") else dict(t)
        iv = d.get("interval") or {}
        rows.append(
            {
                "test": d["name"],
                "n": d.get("n"),
                "effect": d.get("effect"),
                "effect_name": d.get("effect_name"),
                "ci_low": iv.get("low"),
                "ci_high": iv.get("high"),
                "p_value": d.get("p_value"),
                "q_value": d.get("q_value"),
                "significant": d.get("significant"),
            }
        )
    return pd.DataFrame(rows)
