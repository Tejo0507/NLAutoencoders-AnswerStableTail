"""Assemble the revised manuscript (`paper/v2/`) into one document, and check its references.

The sections are the source of truth; `paper/v2/MANUSCRIPT.md` is generated.
Do not edit it by hand. Each section's leading italic note (an editorial
note about provenance) is left out of the assembled text. The assembled
document keeps a status banner: the manuscript is a revised candidate until
the authors record its approval.

`--check` assembles nothing and checks, in the sections:
- every numeric citation resolves to `references.md`, and every entry is cited;
- every "Fig n" and "Table n" referred to has a caption, and every caption is
  referred to in the text;
- every "§x.y" cross-reference names a heading that exists.

    python scripts/assemble_manuscript_v2.py            # write MANUSCRIPT.md
    python scripts/assemble_manuscript_v2.py --check    # exit 1 on any problem

The default status is the one stated in the sections; `--status` changes it
only when the authors have approved a new one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "paper" / "v2"
SECTIONS = ("00_abstract.md", "01_introduction.md", "02_related_work.md", "03_methods.md",
            "04_results.md", "05_discussion.md", "06_limitations.md", "07_reproducibility.md")
CAPTIONS = "08_figures_tables.md"
TABLES = "tables_generated.md"
REFERENCES = "references.md"
DEFAULT_STATUS = "Revised candidate, awaiting author review. Not approved, not frozen."
CITE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
ENTRY = re.compile(r"^\[(\d+)\] ", re.M)


def strip_note(text: str) -> str:
    """Drop the first italic paragraph after the heading (the editorial note).

    The note ends at an asterisk that closes the paragraph, so bold text
    inside the note does not end it early.
    """
    return re.sub(r"\A(#[^\n]*\n+)\*[^*].*?\*\n\s*\n", r"\1", text, count=1, flags=re.S)


def read(name: str, v2: Path = V2) -> str:
    return (v2 / name).read_text(encoding="utf-8")


def check(v2: Path = V2) -> list[str]:
    problems: list[str] = []
    body = {s: read(s, v2) for s in SECTIONS}
    captions = read(CAPTIONS, v2)
    text = "\n".join(body.values())
    refs = read(REFERENCES, v2)

    defined = {int(n) for n in ENTRY.findall(refs)}
    cited = {int(x) for m in CITE.finditer(text) for x in m.group(1).split(",")}
    for n in sorted(cited - defined):
        problems.append(f"citation [{n}] has no reference entry")
    for n in sorted(defined - cited):
        problems.append(f"reference [{n}] is never cited")
    if defined != set(range(1, len(defined) + 1)):
        problems.append("reference numbers are not 1..n")
    order = []
    for m in CITE.finditer(text):
        for x in (int(y) for y in m.group(1).split(",")):
            if x not in order:
                order.append(x)
    if order != sorted(order):
        problems.append("references are not numbered in order of first citation")

    for kind in ("Fig", "Table"):
        word = "Figure" if kind == "Fig" else "Table"
        referred = {int(n) for n in re.findall(rf"\b{kind}s? (\d+)", text)}
        referred |= {int(n) for m in re.finditer(rf"\b{kind}s (\d+(?:, \d+)*,? and \d+)", text)
                     for n in re.findall(r"\d+", m.group(1))}
        captioned = {int(n) for n in re.findall(rf"\*\*{word} (\d+)\.", captions)}
        for n in sorted(referred - captioned):
            problems.append(f"{kind} {n} is referred to but has no caption")
        for n in sorted(captioned - referred):
            problems.append(f"{word} {n} has a caption but is never referred to")
        first = []
        for m in re.finditer(rf"\b{kind}s? (\d+)", text):
            if int(m.group(1)) not in first:
                first.append(int(m.group(1)))
        if first != sorted(first):
            problems.append(f"{word.lower()}s are not numbered in order of first citation: {first}")

    headings = set(re.findall(r"^#+ (\d+(?:\.\d+)?)\.? ", text, re.M))
    for ref in sorted(set(re.findall(r"§(\d+(?:\.\d+)?)", text))):
        if ref not in headings:
            problems.append(f"§{ref} is referred to but no such heading exists")
    return problems


def assemble(status: str, v2: Path = V2) -> str:
    parts = []
    abstract = strip_note(read(SECTIONS[0], v2))
    title, _, rest = abstract.partition("\n")
    parts += [title, "", f"> **Status:** {status}", "", rest.strip()]
    for s in SECTIONS[1:]:
        parts += ["", strip_note(read(s, v2)).strip()]
    parts += ["", "# Tables", ""]
    tables = read(TABLES, v2).split("\n## ", 1)[1]
    parts += ["## " + tables.strip()]
    parts += ["", strip_note(read(CAPTIONS, v2)).strip().replace("# Figures and tables",
                                                                 "# Figure and table captions", 1)]
    parts += ["", read(REFERENCES, v2).strip()]
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--out", type=Path, default=V2 / "MANUSCRIPT.md")
    a = ap.parse_args()
    problems = check()
    for p in problems:
        print("PROBLEM", p)
    if a.check:
        print(f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if problems:
        print("not assembled: fix the problems above first", file=sys.stderr)
        return 1
    a.out.write_text(assemble(a.status), encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
