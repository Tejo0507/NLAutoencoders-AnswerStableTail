"""Figures for the revised manuscript, from stored analysis outputs only. CPU.

Every number plotted is read from a JSON written by an analysis script:
``runs/main_position_audit/position_truncation.json`` (D79),
``runs/main_nla240_information/information*.json`` (D81) and
``runs/main_offmenu_stage0/stage0.json`` (D83). Nothing is recomputed here.

    python scripts/make_paper_figures_v2.py            # writes paper/figures_v2/

A figure whose input does not exist yet is skipped and reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures_v2"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = {"readout": "#cf222e", "falling": "#8250df", "probe": "#0969da", "rising": "#9a6700",
     "no_signal": "#6e7781", "A": "#1a7f37", "text": "#bc4c00", "oracle": INK}


def _load(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_increments():
    """D81: what each signal adds, menu chunks of main_nla240."""
    a = _load("runs/main_nla240_information/information.json")
    b = _load("runs/main_nla240_information/information_posthoc.json")
    if a is None or b is None:
        return None
    rows = [
        ("registered probe | position", a["Y_reg"]["Q1"]["probe over pos"], a["Y_tail"]["Q1"]["probe over pos"], C["probe"]),
        ("readout cosine | position", a["Y_reg"]["Q1"]["readout over pos"], a["Y_tail"]["Q1"]["readout over pos"], C["readout"]),
        ("readout cosine | position + probe", a["Y_reg"]["Q1"]["Q1-P readout over pos+probe"], a["Y_tail"]["Q1"]["Q1-P readout over pos+probe"], C["readout"]),
        ("trace text (600 chars) | position", a["Y_reg"]["Q2"]["R over pos"], a["Y_tail"]["Q2"]["R over pos"], MUTED),
        ("explanation text | position + trace text", a["Y_reg"]["Q2"]["Q2-P E over pos+R"], a["Y_tail"]["Q2"]["Q2-P E over pos+R"], C["text"]),
        ("noise-vector text | position (null)", a["Y_reg"]["Q2"]["N over pos (null check)"], a["Y_tail"]["Q2"]["N over pos (null check)"], MUTED),
        ("activation probe, same label | position", b["Y_reg"]["contrasts"]["A over pos"], b["Y_tail"]["contrasts"]["A over pos"], C["A"]),
        ("explanation text | position + activation probe", b["Y_reg"]["contrasts"]["E over pos+A"], b["Y_tail"]["contrasts"]["E over pos+A"], C["text"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, k, title in ((axes[0], 1, "Registered safety (48-token forced answer)"),
                         (axes[1], 2, "Tail membership (i ≥ i*)")):
        for j, row in enumerate(rows[::-1]):
            v = row[k]
            ax.errorbar(v["delta_auc"], j, xerr=[[v["delta_auc"] - v["ci95"][0]], [v["ci95"][1] - v["delta_auc"]]],
                        fmt="o", color=row[3], ms=5, capsize=2, lw=1.2)
        ax.axvline(0, color=INK, lw=0.8)
        _axes(ax, title, "ΔAUC (95% problem-cluster bootstrap CI)", "")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
    fig.suptitle("What each signal adds at design-A menu chunks (240 problems, 1,206 chunks; exploratory)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig_v2_1_information_increments")


def _curve(rule_entries):
    xs, ys = [], []
    for e in rule_entries:
        if e is not None:
            xs.append(e["saving"])
            ys.append(1 - e["safe_rate"])
    return xs, ys


def fig_stopping():
    """D79: break rate against mean saving on main's design-A menu."""
    d = _load("runs/main_position_audit/position_truncation.json")
    if d is None:
        return None
    r = d["registered_outcome"]
    s = r["safe_rate_at_budget"]
    budgets = r["budgets"]
    per = r["gaps"]["nla_readout - no_signal"]["per_budget"]
    ns = [(b, 1 - (e["safe_rate"] - g)) for b, e, g in zip(budgets, s["nla_readout"], per)
          if e is not None and g is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for key, label, col in (("nla_readout", "NLA readout (cosine)", C["readout"]),
                            ("position_tokens_falling", "falling position rule (no NLA)", C["falling"]),
                            ("probe", "registered probe", C["probe"]),
                            ("position_tokens", "rising position rule", C["rising"])):
        x, y = _curve(s[key])
        ax.plot(x, y, "o-", color=col, label=label, ms=3.5, lw=1.4)
    ax.plot([b for b, _ in ns], [v for _, v in ns], "s--", color=C["no_signal"], label="no signal", ms=3, lw=1.2)
    ax.axvline(r["oracle_bound"], color=C["oracle"], lw=0.8, ls=":")
    ax.text(r["oracle_bound"], 0.005, " oracle B*", fontsize=7, color=INK)
    _axes(ax, "Registered estimand: answers broken per token saved", "mean saving (fraction of trace)",
          "break rate (1 − safe rate)")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    mid = budgets.index(r["half_bstar_budget"]) if r["half_bstar_budget"] in budgets else 4
    names = [("nla_readout", "readout", C["readout"]), ("position_tokens_falling", "falling rule", C["falling"]),
             ("probe", "probe", C["probe"]), ("position_tokens", "rising rule", C["rising"])]
    for j, (key, label, col) in enumerate(names):
        e = s[key][mid]
        if e is None:
            continue
        ax.bar(j - 0.2, e["n_stops"] - e["n_safe_stops"], 0.4, color=col, alpha=0.45)
        ax.bar(j + 0.2, e["n_safe_stops"], 0.4, color=col)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names], fontsize=8)
    _axes(ax, "Stops at 0.5·B*: unsafe (light) and safe (dark)", "", "problems (of 120)")
    fig.tight_layout()
    return _save(fig, "fig_v2_2_registered_estimand")


def fig_truncation():
    d = _load("runs/main_position_audit/position_truncation.json")
    if d is None:
        return None
    t = d["truncation_descriptive"]
    cut = t["forced_cut_off_arm_boundaries"]["by_position"]
    dis = t["forced_vs_continuation_majority_disagree_by_cut_off_and_position"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    keys = ["before", "at or after"]
    ax.bar(range(2), [cut[k]["rate"] for k in keys], color=[C["readout"], C["probe"]])
    for j, k in enumerate(keys):
        ax.text(j, cut[k]["rate"] + 0.01, f"n={cut[k]['n']}", ha="center", fontsize=7)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["before i*", "at or after i*"], fontsize=8)
    _axes(ax, "48-token forced completions cut off", "", "share of boundaries")
    ax = axes[1]
    keys = ["cut off, before", "finished, before", "finished, at or after"]
    ax.bar(range(3), [dis[k]["rate"] for k in keys], color=[C["readout"], C["rising"], C["probe"]])
    for j, k in enumerate(keys):
        ax.text(j, dis[k]["rate"] + 0.01, f"n={dis[k]['n']}", ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["cut off,\nbefore i*", "finished,\nbefore i*", "finished,\nat/after i*"], fontsize=8)
    _axes(ax, "Forced answer ≠ continuations' majority", "", "share of boundaries")
    fig.suptitle("Forced-answer truncation (main, 120-problem arm)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig_v2_3_truncation")


def fig_stage0():
    d = _load("runs/main_offmenu_stage0/stage0.json")
    if d is None:
        return None
    pol = d["policies"]["registered savings"]
    g = pol["gaps"]
    rows = [(k.replace(" - no_signal", ""), v) for k, v in g.items() if k.endswith(" - no_signal")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, (name, v) in enumerate(rows):
        if v["ci95"] is None:
            continue
        ax.errorbar(v["gap"], j, xerr=[[v["gap"] - v["ci95"][0]], [v["ci95"][1] - v["gap"]]],
                    fmt="o", color=C["A"] if "pos+A" in name else MUTED, capsize=2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    _axes(ax, f"Every boundary, no menu (250 problems): safe-rate gap to no signal; B* = {pol['oracle_bound']:.3f}",
          "mean safe-rate gap over budgets (95% CI)", "")
    fig.tight_layout()
    return _save(fig, "fig_v2_4_stage0_policies")


def fig_selective():
    """D83 deviation 1: selective risk against coverage, menu and every boundary."""
    menu = _load("runs/main_nla240_selective/selective_menu.json")
    s0 = _load("runs/main_offmenu_stage0/stage0.json")
    if menu is None or s0 is None:
        return None
    panels = [
        (menu["selective_risk_at_coverage"], "Design-A menu (240 problems)",
         [("nla_readout", "NLA readout", C["readout"]), ("probe", "registered probe", C["probe"]),
          ("position_tokens_falling", "falling rule", C["falling"]),
          ("position_tokens", "rising rule", C["rising"])]),
        (s0["policies"]["registered savings"]["selective_risk_at_coverage (deviation 1, post hoc)"],
         "Every boundary, no menu (250 problems)",
         [("learned pos+A", "same-label activation probe", C["A"]),
          ("registered probe", "registered probe", C["probe"]),
          ("position_tokens_falling", "falling rule", C["falling"]),
          ("position_tokens", "rising rule", C["rising"])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (res, title, rules) in zip(axes, panels):
        covs = sorted(res, key=float)
        ns = [res[c]["no_signal"] for c in covs]
        ax.plot([float(c) for c in covs], [n["selective_risk"] for n in ns], "s--", color=C["no_signal"],
                label="no signal", ms=4)
        for key, label, col in rules:
            pts = [(float(c), res[c][key]["at"]) for c in covs if res[c][key]["at"] is not None]
            ax.plot([x for x, _ in pts], [a["selective_risk"] for _, a in pts], "o-", color=col,
                    label=label, ms=4)
            for x, a in pts:
                ax.annotate(f"{a['saving_among_stops']:.2f}", (x, a["selective_risk"]), fontsize=6,
                            color=col, xytext=(3, 3), textcoords="offset points")
        _axes(ax, title, "coverage (share of problems stopped before the end)",
              "selective risk (unsafe share of stops)")
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Selective risk at matched coverage; labels: mean saving among stops (post hoc)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig_v2_5_selective_risk")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p.relative_to(ROOT))


def main() -> int:
    for f in (fig_increments, fig_stopping, fig_truncation, fig_stage0, fig_selective):
        out = f()
        print(f"{f.__name__}: {out or 'skipped (input missing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
