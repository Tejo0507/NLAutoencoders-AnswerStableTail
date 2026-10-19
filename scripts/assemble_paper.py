"""Assemble `paper/` into one submission document, with the gaps named.

The literature review was submitted as a single Word document rendered from one
Markdown file by `build_review_docx.py`. The final report has to be submitted
the same way, and `paper/` is six separate section files with no front matter,
no table of contents, no bibliography and no assembly step. This is that step.

**It will not pretend the report is finished.** Sections are declared below with
what each one waits on, and a section that does not exist is written into the
document as an explicit placeholder naming its blocker — not silently omitted,
and not stubbed with prose that reads like a finding. The exit code says which
state the document is in:

  0  every declared section is present
  2  the document assembled, with placeholders for the sections still pending

So this can be run today, on a report whose Results do not exist, and the
output is an honest document that says so. Run it again when they do and the
placeholders disappear.

A section that exists but still carries an unresolved `**[VERIFY]**` or
`**[N]**` marker counts as incomplete too, so a draft cannot pass for finished
by being present.

The references are read from `paper/references.md`, the version-controlled
copy of `Project_Review_II.md` §References. The drafts' `[n]` citations are
numbered against that list; the copy was checked against each source on
2026-10-19 and two author lists corrected, as its header records.

    python scripts/assemble_paper.py
    python scripts/assemble_paper.py --out paper/REPORT.md
    python scripts/assemble_paper.py --check      # report state, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
REFERENCES = PAPER / "references.md"

#: Inline markers that say a statement is not finished. A section containing
#: one is reported as incomplete, whatever its declaration says.
UNRESOLVED_MARKERS = ("**[VERIFY]**", "**[N]**")


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    #: Filename under `paper/`, or None for a section with no file yet.
    filename: str | None
    #: What the section is waiting on. Empty when nothing.
    waiting_on: str = ""


#: The report's structure, in submission order. Taken from `paper/README.md`,
#: which is the record of what was planned and what each piece waits on.
SECTIONS: tuple[Section, ...] = (
    Section("", "Abstract", "00_abstract.md"),
    Section("1", "Introduction", "01_introduction.md"),
    Section("2", "Related work", "02_related_work.md"),
    Section("3", "Method", "03_method.md"),
    Section("4", "Experimental setup", "04_experimental_setup.md"),
    Section("5", "Statistical analysis plan", "05_analysis_plan.md"),
    Section("6", "Results", "06_results.md"),
    Section("7", "Discussion", "07_discussion.md"),
    Section("8", "Limitations and threats to validity", "08_limitations.md"),
)

TITLE = "Natural Language Autoencoders for Redundancy Diagnostics"
SUBTITLE = "Project Review II: Method, Results and Discussion"

#: The review's title page fields, left blank there too. Kept identical so the
#: two documents present the same front matter.
FRONT_MATTER_FIELDS = (
    "Student name", "Registration number", "Programme / Department",
    "Supervisor", "Date of submission",
)


def references() -> list[str]:
    """The numbered bibliography, from `paper/references.md`."""
    if not REFERENCES.exists():
        return []
    return [line.strip() for line in REFERENCES.read_text(encoding="utf-8").splitlines()
            if re.match(r"^\[\d{1,2}\]", line.strip())]


def unresolved_markers(text: str) -> list[str]:
    """The unfinished-statement markers a section still carries."""
    return [m for m in UNRESOLVED_MARKERS if m in text]


def cited_numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", text)}


def strip_draft_notes(body: str) -> tuple[str, list[str]]:
    """Separate the drafting note under a section heading from its content.

    Each draft opens with an italic note saying what it is and what it waits
    on. That is a note to the authors, not part of the report, so it is pulled
    out and reported rather than carried into a submission document.
    """
    lines = body.splitlines()
    out, notes, i = [], [], 0
    # Keep the heading, then take any immediately following italic block.
    while i < len(lines) and not lines[i].startswith("#"):
        i += 1
    if i < len(lines):
        out.append(lines[i])
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("*"):
        note = []
        while i < len(lines) and lines[i].strip():
            note.append(lines[i].strip())
            i += 1
        notes.append(" ".join(note).strip("*").strip())
    out.extend(lines[i:])
    return "\n".join(out).strip(), notes


def placeholder(section: Section) -> str:
    """A section that does not exist, written as what it is."""
    heading = (f"# {section.number}. {section.title}" if section.number
               else f"# {section.title}")
    return "\n".join([
        heading, "",
        f"**Not written.** Waiting on {section.waiting_on}.", "",
        "This placeholder is deliberate. The section is part of the planned "
        "report and its absence is a state of the work, so it is recorded "
        "here rather than left out of the document: a reader of this draft "
        "should be able to see what is missing without consulting the "
        "repository. Nothing in it is a finding, and no number in it is "
        "provisional, because there is no number in it.",
    ])


def assemble() -> tuple[str, list[Section], list[str]]:
    """``(markdown, pending_sections, notes)``."""
    parts: list[str] = []
    pending: list[Section] = []
    notes: list[str] = []

    parts.append(f"# {TITLE}\n")
    parts.append(f"**{SUBTITLE}**\n")
    parts.append("---\n")
    for field in FRONT_MATTER_FIELDS:
        parts.append(f"**{field}:** _______________________\n")
    parts.append("---\n")

    # Table of contents, with the pending sections marked in it.
    parts.append("## Table of Contents\n")
    for s in SECTIONS:
        label = f"{s.number}. {s.title}" if s.number else s.title
        present = bool(s.filename and (PAPER / s.filename).exists())
        parts.append(f"- {label}" + ("" if present else "  *(not written)*"))
    parts.append("")
    parts.append("---\n")

    body_text = []
    for s in SECTIONS:
        path = (PAPER / s.filename) if s.filename else None
        if path is None or not path.exists():
            pending.append(s)
            parts.append(placeholder(s))
            parts.append("\n---\n")
            continue
        raw = path.read_text(encoding="utf-8")
        content, section_notes = strip_draft_notes(raw)
        notes += [f"{s.title}: {n}" for n in section_notes]
        markers = unresolved_markers(content)
        if markers and not s.waiting_on:
            s = Section(s.number, s.title, s.filename,
                        "unresolved " + " and ".join(markers) + " markers")
        if s.waiting_on:
            pending.append(s)
        parts.append(content)
        parts.append("\n---\n")
        body_text.append(content)

    refs = references()
    if refs:
        used = cited_numbers("\n".join(body_text))
        parts.append("## References\n")
        parts.append("*Numbering follows `Project_Review_II.md`. Each entry was "
                     "checked against its source on 2026-10-19; "
                     "`paper/references.md` records how, and the two "
                     "corrections made.*\n")
        for entry in refs:
            n = int(re.match(r"^\[(\d{1,2})\]", entry).group(1))
            parts.append(entry + ("" if n in used else "  *(not yet cited)*"))
            parts.append("")
    return "\n".join(parts).rstrip() + "\n", pending, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PAPER / "REPORT.md")
    ap.add_argument("--check", action="store_true",
                    help="report the document's state and write nothing")
    args = ap.parse_args()

    markdown, pending, notes = assemble()
    missing = [s for s in pending if not (s.filename
                                          and (PAPER / s.filename).exists())]
    incomplete = [s for s in pending if s not in missing]

    print(f"{len(SECTIONS)} declared sections, "
          f"{len(SECTIONS) - len(missing)} present, {len(missing)} not written")
    for s in missing:
        label = f"{s.number}. {s.title}" if s.number else s.title
        print(f"  not written  {label}\n               waiting on {s.waiting_on}")
    for s in incomplete:
        label = f"{s.number}. {s.title}" if s.number else s.title
        print(f"  present      {label}\n               still waiting on "
              f"{s.waiting_on}")
    if notes:
        print("\ndrafting notes lifted out of the submission text:")
        for n in notes:
            print(f"  - {n}")

    if not args.check:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)} "
              f"({len(markdown.splitlines())} lines)")

    if missing or incomplete:
        print("\nNot submission-ready: the sections above are pending. The "
              "document assembled anyway, with each gap named in place, so it "
              "can be read and circulated as a draft that states what it is "
              "missing.")
        return 2
    print("\nEvery declared section is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
