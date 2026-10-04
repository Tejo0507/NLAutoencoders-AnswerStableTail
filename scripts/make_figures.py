"""Generate the figures used in the Project Review document.

Writes PNG files into ``figures/`` at the repository root:

    fig1_answer_stable_tail.png    Structure of a reasoning trace and the tail
    fig2_nla_mechanism.png         Activation verbaliser / reconstructor loop
    fig3_literature_argument.png   How the reviewed literature converges
    fig4_proposed_architecture.png Proposed evaluation pipeline

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

INK = "#1a1a1a"
MID = "#5a5a5a"
LIGHT = "#c8c8c8"
PANEL = "#f3f3f3"
ACCENT = "#8a6a3a"
ACCENT_FILL = "#f0e4d0"
GREEN = "#5f7f5f"
GREEN_FILL = "#e6ece6"

# Mean glyph width of DejaVu Sans as a fraction of the point size. Used to fit
# label text to box width without measuring a rendered layout.
CHAR_W = 0.62

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    }
)


def _wrap(text: str, width_in: float, size: float) -> str:
    """Wrap *text* so each line fits within *width_in* inches at *size* points."""
    max_chars = max(8, int(width_in * 72.0 / (CHAR_W * size)))
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
        else:
            # Never hyphenate or split a word: a slightly wide line reads better
            # than a broken term such as "Reconstruct or".
            out.extend(textwrap.wrap(para, max_chars, break_long_words=False) or [""])
    return "\n".join(out)


def box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    *,
    fig_w,
    fill=PANEL,
    edge=MID,
    size=8.0,
    weight="normal",
    lw=1.0,
    title=None,
    title_size=None,
):
    """Draw a rounded box and place wrapped text inside it.

    ``title`` renders as a bold first line above the body text.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    inner_in = (w - 0.03) * fig_w
    cx = x + w / 2

    if title:
        tsize = title_size or size + 0.4
        body = _wrap(text, inner_in, size) if text else ""
        if not body:
            title_y, body_y = y + h * 0.5, None
        else:
            # Split the box in proportion to the rendered height of each part so
            # that a multi-line body cannot ride up into the title.
            title_h = len(_wrap(title, inner_in, tsize).split("\n")) * tsize * 1.3
            body_h = len(body.split("\n")) * size * 1.35
            frac = title_h / (title_h + body_h)
            title_y = y + h * (1.0 - frac / 2)
            body_y = y + h * ((1.0 - frac) / 2)
        ax.text(
            cx,
            title_y,
            _wrap(title, inner_in, tsize),
            ha="center",
            va="center",
            fontsize=tsize,
            fontweight="bold",
            color=INK,
            linespacing=1.3,
        )
        if body:
            ax.text(
                cx,
                body_y,
                body,
                ha="center",
                va="center",
                fontsize=size,
                color=MID,
                linespacing=1.35,
            )
    elif text:
        ax.text(
            cx,
            y + h / 2,
            _wrap(text, inner_in, size),
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=INK,
            linespacing=1.35,
        )


def arrow(ax, start, end, *, color=MID, lw=1.1, rad=0.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- figure 1


def fig_answer_stable_tail() -> None:
    fw = 7.2
    fig, ax = blank_axes((fw, 2.85))

    ax.text(
        0.5,
        0.955,
        "Structure of a reasoning trace and the Answer-Stable Tail",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )

    y, h = 0.50, 0.17
    labels = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    answers = ["--", "17", "23", "41", "41", "41", "41", "41"]
    x0, total = 0.215, 0.765
    n = len(labels)
    w = total / n
    stable_from = 3

    for i, (lab, ans) in enumerate(zip(labels, answers)):
        x = x0 + i * w
        in_tail = i > stable_from
        box(
            ax,
            x + 0.004,
            y,
            w - 0.008,
            h,
            lab,
            fig_w=fw,
            fill=ACCENT_FILL if in_tail else PANEL,
            edge=ACCENT if in_tail else MID,
            size=8,
        )
        ax.text(
            x + w / 2,
            y - 0.085,
            ans,
            ha="center",
            va="center",
            fontsize=8,
            color=ACCENT if in_tail else MID,
            fontweight="bold" if in_tail else "normal",
        )

    ax.text(x0 - 0.018, y + h / 2, "chunk", fontsize=8, color=MID, ha="right", va="center")
    ax.text(
        x0 - 0.018,
        y - 0.085,
        "parsed answer",
        fontsize=8,
        color=MID,
        ha="right",
        va="center",
    )

    xs = x0 + (stable_from + 1) * w
    ax.plot([xs, xs], [y - 0.15, y + h + 0.10], color=INK, lw=1.2, ls=(0, (4, 2)))
    ax.text(
        xs,
        y + h + 0.135,
        "answer stabilises",
        fontsize=8.5,
        color=INK,
        ha="center",
        fontweight="bold",
    )

    x_end = x0 + total
    ax.annotate(
        "",
        xy=(xs, 0.245),
        xytext=(x_end, 0.245),
        arrowprops=dict(arrowstyle="|-|", color=ACCENT, lw=1.3, shrinkA=0, shrinkB=0),
    )
    ax.text(
        (xs + x_end) / 2,
        0.165,
        "Answer-Stable Tail",
        ha="center",
        va="center",
        fontsize=9,
        color=ACCENT,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.055,
        "The parsed answer no longer changes, yet generation continues.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MID,
        style="italic",
    )

    fig.savefig(FIG_DIR / "fig1_answer_stable_tail.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2


def fig_nla_mechanism() -> None:
    fw = 7.2
    fig, ax = blank_axes((fw, 3.0))

    ax.text(
        0.5,
        0.955,
        "Natural Language Autoencoder: verbaliser and reconstructor",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )

    row_y, row_h = 0.52, 0.22

    box(
        ax,
        0.015,
        row_y - 0.06,
        0.20,
        row_h + 0.12,
        "",
        fig_w=fw,
        fill="#fbfbfb",
        edge=LIGHT,
    )
    ax.text(
        0.115,
        row_y + row_h + 0.028,
        "Frozen target model",
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    box(ax, 0.033, row_y, 0.164, row_h, "residual stream at layer L", fig_w=fw, size=7.8)

    box(ax, 0.242, row_y, 0.122, row_h, "activation vector a", fig_w=fw,
        fill=ACCENT_FILL, edge=ACCENT, size=7.8)
    box(ax, 0.394, row_y, 0.132, row_h, "Activation Verbaliser (AV)", fig_w=fw,
        size=8.0, weight="bold")
    box(ax, 0.556, row_y, 0.140, row_h, "natural-language description", fig_w=fw,
        fill="#fbfbfb", edge=MID, size=7.6)
    box(ax, 0.726, row_y, 0.160, row_h, "Activation Reconstructor (AR)", fig_w=fw,
        size=8.0, weight="bold")
    box(ax, 0.916, row_y, 0.072, row_h, "a-hat", fig_w=fw,
        fill=ACCENT_FILL, edge=ACCENT, size=7.8)

    for a, b in [(0.199, 0.240), (0.366, 0.392), (0.528, 0.554), (0.698, 0.724), (0.888, 0.914)]:
        arrow(ax, (a, row_y + row_h / 2), (b, row_y + row_h / 2))

    # Reconstruction objective, routed in the clear space beneath the row. A
    # negative curvature is what bends the arc downward on a right-to-left span.
    ax.add_patch(
        FancyArrowPatch(
            (0.952, row_y - 0.015),
            (0.303, row_y - 0.015),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=ACCENT,
            connectionstyle="arc3,rad=-0.22",
            shrinkA=2,
            shrinkB=2,
        )
    )
    ax.text(
        0.628,
        0.235,
        "minimise squared error between a and a-hat",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ACCENT,
    )
    ax.text(
        0.628,
        0.185,
        "reported fidelity: fraction of variance explained",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ACCENT,
    )

    ax.text(
        0.5,
        0.070,
        "High reconstruction fidelity shows the description retains activation information.\n"
        "It does not show that each claim inside the description is grounded in the activation.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MID,
        style="italic",
        linespacing=1.45,
    )

    fig.savefig(FIG_DIR / "fig2_nla_mechanism.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3


def fig_literature_argument() -> None:
    fw = 6.9
    fig, ax = blank_axes((fw, 5.3))

    steps = [
        ("Extended reasoning is standard practice",
         "Chain-of-thought and reinforcement-trained reasoning models make long traces normal."),
        ("Part of that reasoning is redundant",
         "Traces continue after the parsed answer stops changing, and rechecks mostly confirm."),
        ("Existing signals detect some redundancy",
         "Answer convergence, hidden-state probes and semantic entropy each give a stopping signal."),
        ("Those signals say whether to stop, not what is happening",
         "They are scalar or behavioural, and describe no internal state."),
        ("The visible reasoning text cannot settle it",
         "Chain-of-thought is not a literal account, so tail wording is not a reliable label."),
        ("Activation-level evidence is available",
         "Probing and intervention work shows behaviourally relevant information is present."),
        ("Verbalisers can render activations as text",
         "Natural Language Autoencoders turn a residual-stream vector into readable description."),
        ("Readable does not imply faithful",
         "Fidelity can stay high while claims are unsupported or come from verbaliser knowledge."),
        ("Independent verification is therefore required",
         "Decodability supervision and privileged-information controls test whether text is grounded."),
        ("This motivates the present study",
         "Test whether verbalised activations add faithful, incremental value in the stable region."),
    ]

    top, bottom = 0.985, 0.015
    n = len(steps)
    gap = 0.011
    h = (top - bottom - (n - 1) * gap) / n

    for i, (title, body) in enumerate(steps):
        y = top - (i + 1) * h - i * gap
        final = i == n - 1
        box(
            ax,
            0.062,
            y,
            0.935,
            h,
            "",
            fig_w=fw,
            fill=ACCENT_FILL if final else PANEL,
            edge=ACCENT if final else LIGHT,
            lw=1.2 if final else 1.0,
        )
        inner_in = 0.86 * fw
        ax.text(
            0.080,
            y + h * 0.69,
            _wrap(title, inner_in, 8.4),
            fontsize=8.4,
            fontweight="bold",
            color=INK,
            va="center",
        )
        ax.text(
            0.080,
            y + h * 0.29,
            _wrap(body, inner_in, 7.6),
            fontsize=7.6,
            color=MID,
            va="center",
            linespacing=1.3,
        )
        ax.text(
            0.030,
            y + h / 2,
            str(i + 1),
            fontsize=8.6,
            fontweight="bold",
            color=ACCENT if final else MID,
            ha="center",
            va="center",
        )
        if not final:
            arrow(ax, (0.030, y - 0.0015), (0.030, y - gap + 0.0015), color=LIGHT, lw=0.9)

    fig.savefig(FIG_DIR / "fig3_literature_argument.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4


def fig_proposed_architecture() -> None:
    fw = 7.0
    fig, ax = blank_axes((fw, 6.6))

    ax.text(
        0.5,
        0.985,
        "Proposed evaluation pipeline",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )

    # Legend across the top.
    legend = [
        mpatches.Patch(facecolor=GREEN_FILL, edgecolor=GREEN, label="implemented"),
        mpatches.Patch(facecolor=PANEL, edgecolor=MID, label="proposed"),
        mpatches.Patch(facecolor=ACCENT_FILL, edgecolor=ACCENT, label="core contribution"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handlelength=1.2,
        handleheight=0.85,
        columnspacing=1.6,
    )

    box(ax, 0.045, 0.835, 0.91, 0.062,
        "GSM8K and MATH staged as verified question-answer pairs",
        fig_w=fw, fill=GREEN_FILL, edge=GREEN, size=8,
        title="Stage A   Benchmark preparation", title_size=8.4)

    arrow(ax, (0.5, 0.831), (0.5, 0.800))

    box(ax, 0.045, 0.718, 0.91, 0.078,
        "Reasoning traces, sentence-level chunking, intermediate-answer parsing, verifier labels",
        fig_w=fw, size=8,
        title="Stage B   Trace generation and segmentation", title_size=8.4)

    arrow(ax, (0.5, 0.714), (0.5, 0.683))

    box(ax, 0.045, 0.586, 0.91, 0.095,
        "Independent of any activation description: answer equivalence under truncation, "
        "K resampled continuations, external verifier, matched-position and matched-length controls",
        fig_w=fw, fill=ACCENT_FILL, edge=ACCENT, size=8,
        title="Stage C   Answer-Stable Tail identification",
        title_size=8.4)

    arrow(ax, (0.5, 0.582), (0.5, 0.556))

    ax.text(0.5, 0.542, "activations extracted at tail-candidate windows",
            ha="center", fontsize=7.8, color=MID, style="italic")

    arrow(ax, (0.44, 0.532), (0.26, 0.502), rad=0.12)
    arrow(ax, (0.56, 0.532), (0.74, 0.502), rad=-0.12)

    box(ax, 0.045, 0.372, 0.43, 0.126,
        "Activation verbaliser and reconstructor applied at tail windows to produce "
        "descriptions and reconstruction scores",
        fig_w=fw, size=7.9,
        title="Branch 1   Verbalised activation readout", title_size=8.2)

    box(ax, 0.525, 0.372, 0.43, 0.126,
        "Answer-convergence stopping, hidden-state correctness probe, semantic entropy",
        fig_w=fw, size=7.9,
        title="Branch 2   Reference signals", title_size=8.2)

    arrow(ax, (0.26, 0.368), (0.42, 0.330), rad=-0.12)
    arrow(ax, (0.74, 0.368), (0.58, 0.330), rad=0.12)

    box(ax, 0.125, 0.228, 0.75, 0.090,
        "Safe-stopping accuracy against tokens saved, with paired bootstrap confidence "
        "intervals computed at problem level",
        fig_w=fw, fill=ACCENT_FILL, edge=ACCENT, size=8,
        title="Stage D   Matched-budget comparison", title_size=8.4)

    arrow(ax, (0.30, 0.224), (0.26, 0.198))
    arrow(ax, (0.70, 0.224), (0.74, 0.198))

    box(ax, 0.045, 0.068, 0.43, 0.128,
        "Claim deletion, resampling and paraphrase, scored by reconstruction change "
        "against independent-probe and verbaliser-only controls",
        fig_w=fw, size=7.8,
        title="Stage E   Faithfulness controls", title_size=8.2)

    box(ax, 0.525, 0.068, 0.43, 0.128,
        "Truncation and tail replacement, with matched-position and "
        "matched-random-direction comparisons",
        fig_w=fw, size=7.8,
        title="Stage F   Causal controls (extension)", title_size=8.2)

    ax.text(
        0.5,
        0.030,
        "Stages B to F are proposed work. Stage F is scoped as an extension contingent on compute.",
        ha="center",
        fontsize=7.8,
        color=MID,
        style="italic",
    )

    fig.savefig(FIG_DIR / "fig4_proposed_architecture.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig_answer_stable_tail()
    fig_nla_mechanism()
    fig_literature_argument()
    fig_proposed_architecture()
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"wrote {path.name}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
