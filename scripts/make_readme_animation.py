"""Draw the README's animated schematic of the answer-stable tail.

Illustrative only: the chunks and the forced answers are made up to show the
idea, and the figure says so on every frame. It reads no run and reports no
result. Output: docs/assets/answer_stable_tail.gif

    python scripts/make_readme_animation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "answer_stable_tail.gif"

# The answer the model would give if forced to answer at each chunk boundary.
FORCED = ["7", "18", "18", "24", "36", "36", "24", "36", "42", "42", "42", "42"]
FINAL = "42"
TAIL_START = 8  # the first chunk from which the forced answer never changes

GREY, GREEN, INK = "#d0d7e0", "#dff3e3", "#1f2430"
GREEN_DARK, RED = "#1a7a33", "#c9302c"

N = len(FORCED)
REVEAL = list(range(N))
HOLD_TAIL, HOLD_QUESTION = 5, 7
# frame -> (chunks shown, caption stage)
FRAMES = (
    [(k + 1, 0) for k in REVEAL]
    + [(N, 1)] * HOLD_TAIL
    + [(N, 2)] * HOLD_QUESTION
)

CAPTIONS = {
    0: "A model reasons step by step. Forced to answer at each point, it gives...",
    1: "From chunk 8 on, the answer never changes: the answer-stable tail.",
    2: "Can a verbalised activation (an NLA) tell where the tail starts, "
       "better than cheaper signals?",
}


def draw(i: int, ax) -> None:
    shown, stage = FRAMES[i]
    ax.clear()
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(0.1, 3.95, "Illustrative schematic - not data", fontsize=8,
            color="#6b7280", style="italic")

    for k in range(shown):
        in_tail = stage >= 1 and k >= TAIL_START
        x = 0.1 + k * 1.02
        ax.add_patch(FancyBboxPatch((x, 2.3), 0.9, 0.9, boxstyle="round,pad=0.02",
                                    fc=GREEN if in_tail else GREY, ec="#6b7280", lw=0.8))
        ax.text(x + 0.45, 2.75, str(k), ha="center", va="center", fontsize=10, color=INK)
        ans = FORCED[k]
        ax.text(x + 0.45, 1.65, ans, ha="center", va="center", fontsize=13,
                fontweight="bold", color=GREEN_DARK if ans == FINAL and k >= TAIL_START else RED)
    ax.text(0.1, 1.05, "forced answer at that point", fontsize=8, color="#6b7280")

    if stage >= 1:
        x0 = 0.1 + TAIL_START * 1.02 - 0.06
        ax.plot([x0, x0], [1.3, 3.5], color=GREEN_DARK, lw=2)
        ax.text(x0 + 0.1, 3.6, "answer-stable tail", color=GREEN_DARK, fontsize=10)

    ax.text(6.2, 0.35, CAPTIONS[stage], ha="center", va="center", fontsize=10,
            color=INK, wrap=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.9), dpi=80)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    anim = FuncAnimation(fig, lambda i: draw(i, ax), frames=len(FRAMES))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=2))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(FRAMES)} frames)")


if __name__ == "__main__":
    main()
