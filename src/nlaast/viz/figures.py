"""Result figures.

Every figure is a pure function of saved stage outputs, so they regenerate from
``runs/<id>/`` without touching a model. Nothing is hand-edited.

Each function returns ``None`` when its inputs are missing, so the reporting
stage produces whatever the run actually supports rather than failing because
one stage did not finish.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..logging_utils import get  # noqa: E402

log = get(__name__)

DPI = 200
#: Colour-blind-safe, and distinguishable in greyscale print.
PALETTE = {
    "convergence": "#4C72B0",
    "probe": "#DD8452",
    "semantic_entropy": "#55A868",
    "nla_readout": "#C44E52",
    "tail": "#C44E52",
    "matched_position": "#4C72B0",
    "matched_length": "#8172B3",
    "gaussian_control": "#937860",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figure %s", path.name)
    return str(path)


def fig_tail_distribution(ast_rows: Sequence[dict], out_dir: Path) -> str | None:
    """Tail fraction distribution and the status accounting beside it."""
    rows = [r for r in ast_rows if r.get("tail_start") is not None]
    if not rows:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    fracs = [r["tail_fraction"] for r in rows]
    ax1.hist(fracs, bins=min(20, max(5, len(fracs) // 2)),
             color=PALETTE["tail"], alpha=0.85, edgecolor="white")
    ax1.axvline(float(np.mean(fracs)), color="black", linestyle="--", linewidth=1.2,
                label=f"mean {np.mean(fracs):.2f}")
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1, "Answer-Stable Tail as a fraction of the trace",
           "tail fraction (chunks)", "problems")

    from collections import Counter

    counts = Counter(r["status"] for r in ast_rows)
    labels = list(counts)
    ax2.barh(labels, [counts[k] for k in labels], color="#4C72B0", alpha=0.85)
    for i, k in enumerate(labels):
        ax2.text(counts[k], i, f" {counts[k]}", va="center", fontsize=8)
    _style(ax2, "Detection status (edge cases retained, not dropped)",
           "problems", "")
    return _save(fig, out_dir, "fig_tail_distribution")


def fig_stopping_curves(curves: dict[str, Sequence[Any]], out_dir: Path) -> str | None:
    """Safety against tokens saved - the O3 comparison."""
    usable = {k: v for k, v in curves.items() if v}
    if not usable:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for rule, pts in usable.items():
        xs = [p["mean_tokens_saved"] if isinstance(p, dict) else p.mean_tokens_saved
              for p in pts]
        safe = [p["safe_rate"] if isinstance(p, dict) else p.safe_rate for p in pts]
        acc = [p["accuracy"] if isinstance(p, dict) else p.accuracy for p in pts]
        order = np.argsort(xs)
        c = PALETTE.get(rule, "#777777")
        ax1.plot(np.asarray(xs)[order], np.asarray(safe)[order], "o-",
                 color=c, label=rule, markersize=3, linewidth=1.4)
        ax2.plot(np.asarray(xs)[order], np.asarray(acc)[order], "o-",
                 color=c, label=rule, markersize=3, linewidth=1.4)
    _style(ax1, "Safe stopping at matched token budget",
           "mean fraction of tokens saved", "safe-stop rate")
    _style(ax2, "Accuracy at matched token budget",
           "mean fraction of tokens saved", "accuracy at stop")
    ax1.legend(fontsize=8, frameon=False)
    return _save(fig, out_dir, "fig_stopping_curves")


def fig_reconstruction_by_window(recon_rows: Sequence[dict], out_dir: Path) -> str | None:
    """Reconstruction fidelity by window kind - the F10 position control."""
    if not recon_rows:
        return None
    kinds = [k for k in ("tail", "matched_position", "matched_length",
                         "gaussian_control")
             if any(r["window_kind"] == k for r in recon_rows)]
    if not kinds:
        return None
    data = [[r["cosine"] for r in recon_rows if r["window_kind"] == k] for k in kinds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "black",
                               "markeredgecolor": "black", "markersize": 4})
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(PALETTE.get(k, "#999999"))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(kinds) + 1))
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=8)
    for i, vals in enumerate(data, start=1):
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center",
                va="bottom", fontsize=7, color="#555555")
    _style(ax, "Reconstruction fidelity by window kind\n"
               "(fidelity is not claim-level faithfulness)",
           "", "cosine(reconstructed, original)")
    return _save(fig, out_dir, "fig_reconstruction_by_window")


def fig_faithfulness(audit_rows: Sequence[dict], out_dir: Path) -> str | None:
    """Deletion effect against the paraphrase null, and the noise control."""
    claims = [c for a in audit_rows for c in a.get("claims", [])]
    if not claims:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    delete = np.array([c["delete_effect"] for c in claims], dtype=float)
    para = np.array([np.mean(np.abs(c["paraphrase_effects"]))
                     if c.get("paraphrase_effects") else np.nan
                     for c in claims], dtype=float)
    mask = np.isfinite(delete) & np.isfinite(para)
    ax1.scatter(para[mask], delete[mask], s=18, alpha=0.6, color=PALETTE["tail"],
                edgecolors="white", linewidths=0.4)
    if mask.any():
        lim = float(np.nanmax(np.abs(np.concatenate([delete[mask], para[mask]])))) * 1.1
        ax1.plot([-lim, lim], [-lim, lim], "--", color="black", linewidth=1,
                 label="deletion = paraphrase")
        ax1.set_xlim(-0.02, lim)
        ax1.set_ylim(-lim, lim)
        ax1.legend(fontsize=8, frameon=False)
    _style(ax1, "Deletion effect against the paraphrase null\n"
                "(points above the line carry reconstruction weight)",
           "mean |paraphrase effect|", "deletion effect")

    dep = np.array([bool(c["reconstruction_dependent"]) for c in claims])
    noise = np.array([bool(c["reproduced_under_noise"]) for c in claims])
    cats = ["dependent", "reproduced\nfrom noise", "dependent and\nnot from noise"]
    vals = [dep.mean(), noise.mean(), (dep & ~noise).mean()]
    bars = ax2.bar(cats, vals, color=["#C44E52", "#937860", "#55A868"], alpha=0.85)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.05)
    _style(ax2, f"Claim-level audit (n={len(claims)} claims)", "", "fraction of claims")
    return _save(fig, out_dir, "fig_faithfulness")


def fig_dose_response(causal: dict, out_dir: Path) -> str | None:
    """Dose-response for the candidate direction against its controls."""
    curve = causal.get("dose_response_ablate") or []
    arms = causal.get("arms") or {}
    if not curve:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    xs = [p["coefficient"] for p in curve]
    ax1.plot(xs, [p["marker_delta"] for p in curve], "o-",
             color=PALETTE["tail"], label="candidate tail direction", linewidth=1.5)
    for arm, colour in (("random_dir", "#937860"), ("matched_position", "#4C72B0")):
        if arm in arms and np.isfinite(arms[arm].get("marker_delta", np.nan)):
            ax1.axhline(arms[arm]["marker_delta"], linestyle="--", color=colour,
                        linewidth=1.2, label=arm.replace("_", " "))
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1, "Dose-response: verification markers", "projection coefficient",
           "change in marker count vs baseline")

    ax2.plot(xs, [p["answer_change_rate"] for p in curve], "o-",
             color="#4C72B0", linewidth=1.5)
    ax2.set_ylim(-0.02, 1.02)
    _style(ax2, "Answer change under ablation\n(selectivity requires this to stay low)",
           "projection coefficient", "fraction of answers changed")
    return _save(fig, out_dir, "fig_dose_response")


def fig_quantisation_drift(f8: dict, out_dir: Path) -> str | None:
    """F8: how far 4-bit activations sit from the bf16 ones the NLA expects."""
    per = f8.get("per_problem") or []
    if f8.get("status") != "ran" or not per:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [p["mean_cosine"] for p in per]
    ax.hist(vals, bins=min(15, max(4, len(vals))), color="#55A868", alpha=0.85,
            edgecolor="white")
    ax.axvline(f8["mean_cosine"], color="black", linestyle="--", linewidth=1.2,
               label=f"mean {f8['mean_cosine']:.4f}")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "F8: cosine between 4-bit and bf16 layer-20 activations\n"
               "(the verbaliser normalises input, so only direction matters)",
           "mean cosine per problem", "problems")
    return _save(fig, out_dir, "fig_quantisation_drift")
