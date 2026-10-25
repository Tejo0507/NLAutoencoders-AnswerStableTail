"""`scripts/assemble_manuscript_v2.py`: assembly and reference checks.

Each test writes a miniature manuscript into a temporary directory; nothing
here reads or writes `paper/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import assemble_manuscript_v2 as am  # noqa: E402


def _write(v2: Path, **over):
    files = {
        "00_abstract.md": "# A title\n\n*note*\n\n## Abstract\n\nText.\n",
        "01_introduction.md": "# 1. Introduction\n\n*note*\n\nSignals [1, 2]. See §2 and Fig 1.\n",
        "02_related_work.md": "# 2. Related work\n\nMore [2]. Table 1.\n",
        "03_methods.md": "# 3. Methods\n\nFig 2.\n",
        "04_results.md": "# 4. Results\n\n## 4.1 A\n\nR.\n",
        "05_discussion.md": "# 5. Discussion\n\nD.\n",
        "06_limitations.md": "# 6. Limitations\n\nL.\n",
        "07_reproducibility.md": "# 7. Reproducibility\n\nP.\n",
        "09_declarations.md": "# Declarations\n\n*note*\n\n**Funding.** To be supplied.\n",
        "08_figures_tables.md": "# Figures and tables\n\n*note*\n\n**Figure 1. One.**\n\n"
                                "**Figure 2. Two.**\n\n**Table 1. T.**\n",
        "tables_generated.md": "# Tables\n\n*gen*\n\n## Table 1. T\n\n| a |\n",
        "references.md": "# References\n\n[1] A, \"x\".\n\n[2] B, \"y\".\n",
    }
    files.update(over)
    for name, text in files.items():
        (v2 / name).write_text(text, encoding="utf-8")


def test_a_consistent_manuscript_has_no_problems(tmp_path):
    _write(tmp_path)
    assert am.check(tmp_path) == []


def test_unresolved_and_uncited_references_are_reported(tmp_path):
    _write(tmp_path, **{"references.md": "# References\n\n[1] A, \"x\".\n\n[2] B, \"y\".\n\n[3] C, \"z\".\n"})
    assert "reference [3] is never cited" in am.check(tmp_path)
    _write(tmp_path, **{"02_related_work.md": "# 2. Related work\n\nMore [4]. Table 1.\n"})
    assert any("[4] has no reference entry" in p for p in am.check(tmp_path))


def test_out_of_order_figures_and_missing_headings_are_reported(tmp_path):
    _write(tmp_path, **{"01_introduction.md": "# 1. Introduction\n\nSignals [1, 2]. See §9 and Fig 2.\n",
                        "03_methods.md": "# 3. Methods\n\nFig 1.\n"})
    problems = am.check(tmp_path)
    assert any("not numbered in order of first citation" in p for p in problems)
    assert "§9 is referred to but no such heading exists" in problems


def test_a_note_with_bold_text_is_dropped_whole(tmp_path):
    text = "# 4. Results\n\n*A note with **bold**\nacross lines.*\n\nBody starts here.\n"
    assert am.strip_note(text) == "# 4. Results\n\nBody starts here.\n"


def test_assembly_drops_notes_and_keeps_the_status(tmp_path):
    _write(tmp_path)
    out = am.assemble("Revised candidate.", tmp_path)
    assert out.startswith("# A title")
    assert "> **Status:** Revised candidate." in out
    assert "*note*" not in out and "[1] A" in out and "## Table 1. T" in out
    assert out.index("# 7. Reproducibility") < out.index("# Declarations") < out.index("# Tables")
