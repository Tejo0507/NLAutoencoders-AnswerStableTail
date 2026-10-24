"""Consolidate the v2 manuscript's bibliography into one file, numbered by first citation.

The v2 sections cite numbers from two lists: `paper/references.md` ([1]-[27],
the frozen report's numbering) and `paper/v2/references_additions.md`
([28]-...). This script

1. reads both lists and every v2 section in reading order;
2. numbers the cited works by first appearance;
3. rewrites each section's numeric citations to the new numbers; and
4. writes `paper/v2/references.md`, one entry per cited work, each noting
   the number it had before and where its verification is recorded.

It refuses to run if a section cites a number that neither list defines, or
if `paper/v2/references.md` already exists (it is a one-time migration;
afterwards the sections and `references.md` are the source of truth).

    python scripts/consolidate_references_v2.py            # migrate
    python scripts/consolidate_references_v2.py --dry-run  # show the mapping only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
V2 = PAPER / "v2"
SECTIONS = ("00_abstract.md", "01_introduction.md", "02_related_work.md", "03_methods.md",
            "04_results.md", "05_discussion.md", "06_limitations.md",
            "07_reproducibility.md", "08_figures_tables.md")
#: A numeric citation: one or more integers in brackets, nothing else.
CITE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
ENTRY = re.compile(r"^\[(\d+)\]\s+(.+)$")


def read_entries(path: Path) -> dict[int, str]:
    """Bibliography entries: lines that start with ``[n]`` and quote a title.

    Prose that merely starts with a citation ("[6] self-verification probes;
    ...") is not an entry; the quoted-title test keeps it out.
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ENTRY.match(line.strip())
        if m and '"' in m.group(2):
            out[int(m.group(1))] = m.group(2).strip()
    return out


def first_citation_order(texts: list[str]) -> list[int]:
    order: list[int] = []
    for t in texts:
        for m in CITE.finditer(t):
            for n in (int(x) for x in m.group(1).split(",")):
                if n not in order:
                    order.append(n)
    return order


def renumber(text: str, mapping: dict[int, int]) -> str:
    def sub(m: re.Match) -> str:
        new = sorted(mapping[int(x)] for x in m.group(1).split(","))
        return "[" + ", ".join(str(n) for n in new) + "]"
    return CITE.sub(sub, text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out_path = V2 / "references.md"
    if out_path.exists() and not a.dry_run:
        print(f"{out_path} exists; the migration has already run", file=sys.stderr)
        return 2
    # the frozen report's list wins; the additions only add numbers it lacks
    old = read_entries(PAPER / "references.md")
    for n, text in read_entries(V2 / "references_additions.md").items():
        old.setdefault(n, text)
    texts = {s: (V2 / s).read_text(encoding="utf-8") for s in SECTIONS}
    order = first_citation_order([texts[s] for s in SECTIONS])
    missing = [n for n in order if n not in old]
    if missing:
        print(f"cited but undefined: {missing}", file=sys.stderr)
        return 1
    mapping = {o: i + 1 for i, o in enumerate(order)}
    for o, n in mapping.items():
        print(f"[{o}] -> [{n}]")
    if a.dry_run:
        return 0
    for s in SECTIONS:
        (V2 / s).write_text(renumber(texts[s], mapping), encoding="utf-8", newline="\n")
    lines = [
        "# References",
        "",
        "*The v2 manuscript's single bibliography, numbered by first citation "
        "(generated once by `scripts/consolidate_references_v2.py` on 2026-10-24 from "
        "`paper/references.md` and `paper/v2/references_additions.md`, which keep their "
        "own numbering as historical records). Each entry notes its former number. "
        "Verification: entries formerly [1]-[27] were checked on 2026-10-19 "
        "(`paper/references.md`); [28]-[32] on 2026-10-21 and [33]-[42] on 2026-10-23 "
        "(`paper/v2/references_additions.md`), each against its arXiv record, ACL "
        "Anthology or proceedings page, or DOI/journal record. These are machine and "
        "agent checks; no human has yet verified them for this manuscript.*",
        "",
    ]
    for o in order:
        lines.append(f"[{mapping[o]}] {old[o]} (Formerly [{o}].)")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{len(order)} cited works -> {out_path}")
    unused = sorted(set(old) - set(order))
    print(f"defined but not cited in v2 (left out): {unused}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
