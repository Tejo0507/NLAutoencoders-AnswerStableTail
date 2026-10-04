"""Render Project_Review_II.md as a submission-ready Word document.

Handles the Markdown subset used by the review: ATX headings, paragraphs with
bold and italic runs, images with captions, pipe tables, block quotes, and
ordered and unordered lists.

Document conventions applied here:
  * a title page carrying the project and candidate details, with no page number
  * one page geometry throughout, so no page differs in size or orientation
  * all body and heading text in black, so nothing renders as a blue hyperlink
  * in-text citations are internal links to their reference entry, styled black
    and without underline so they read as plain text but remain clickable
  * table rows never split across a page break, and header rows repeat
  * page numbers centred in the footer; chapter headings start a new page

Usage:
    python scripts/build_docx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Project_Review_II.md"
DST = ROOT / "Project_Review_II.docx"

BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x44, 0x44, 0x44)

BODY_FONT = "Cambria"
HEAD_FONT = "Cambria"
TABLE_FONT = "Calibri"

INLINE = re.compile(
    r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\[\d{1,2}\])"
)
CITATION = re.compile(r"^\[(\d{1,2})\]$")


def bookmark_name(n: int | str) -> str:
    return f"ref{n}"


# ------------------------------------------------------------------ helpers


def unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    )


def style_run(run, *, size=None, color=DARK, font=BODY_FONT):
    run.font.name = font
    run.font.color.rgb = color
    if size is not None:
        run.font.size = Pt(size)
    # East-Asian font binding, otherwise Word may substitute a different face.
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font)


def add_citation_link(paragraph, label, anchor, *, size=None, font=BODY_FONT):
    """Append '[n]' as an internal link to its reference entry.

    Styled black and without underline: it must read as ordinary text in print
    while staying clickable on screen.
    """
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("w:anchor"), anchor)

    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")

    rfonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), font)
    rpr.append(rfonts)

    col = OxmlElement("w:color")
    col.set(qn("w:val"), "1A1A1A")
    rpr.append(col)

    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "none")
    rpr.append(underline)

    if size is not None:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))  # half-points
        rpr.append(sz)

    run.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = label
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_runs(
    paragraph,
    text,
    *,
    size=None,
    color=DARK,
    font=BODY_FONT,
    base_bold=False,
    link_citations=True,
):
    """Append inline-formatted runs. Link syntax renders as plain label text."""
    text = unescape(text)
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos : m.start()])
            r.bold = base_bold
            style_run(r, size=size, color=color, font=font)
        tok = m.group(0)
        cite = CITATION.match(tok)
        if cite and link_citations:
            add_citation_link(
                paragraph, tok, bookmark_name(cite.group(1)), size=size, font=font
            )
            pos = m.end()
            continue
        if cite:
            r = paragraph.add_run(tok)
            r.bold = base_bold
            style_run(r, size=size, color=color, font=font)
            pos = m.end()
            continue
        if tok.startswith("**"):
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
            style_run(r, size=size, color=color, font=font)
        elif tok.startswith("*"):
            r = paragraph.add_run(tok[1:-1])
            r.italic = True
            r.bold = base_bold
            style_run(r, size=size, color=color, font=font)
        elif tok.startswith("`"):
            r = paragraph.add_run(tok[1:-1])
            style_run(r, size=size, color=color, font="Consolas")
        else:  # [label](target) renders as the label only, never as a blue link
            label = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            r = paragraph.add_run(label.group(1) if label else tok)
            r.bold = base_bold
            style_run(r, size=size, color=color, font=font)
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        r.bold = base_bold
        style_run(r, size=size, color=color, font=font)


def add_page_number_footer(section) -> None:
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    style_run(run, size=9, color=GREY)
    for kind, payload in (("begin", None), ("instr", "PAGE"), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = payload
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def set_cell_background(cell, hex_color: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def repeat_header_row(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def keep_row_whole(row) -> None:
    """Stop a row breaking mid-cell across a page boundary."""
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def add_bookmark(paragraph, name: str, bid: int) -> None:
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bid))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bid))
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def is_table_sep(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ------------------------------------------------------------------ builder


class Builder:
    def __init__(self) -> None:
        self.doc = Document()
        self._configure_styles()
        self._setup_first_section()
        # Chapter 1 must start on a fresh page, not under the trailing lines of
        # the front matter, so the first H1 already owes a break.
        self.pending_chapter_break = True
        self.front_matter = True
        self.in_references = False
        self._bookmark_id = 1000

    def _configure_styles(self) -> None:
        d = self.doc
        normal = d.styles["Normal"]
        normal.font.name = BODY_FONT
        normal.font.size = Pt(11)
        normal.font.color.rgb = DARK
        pf = normal.paragraph_format
        pf.space_after = Pt(8)
        pf.line_spacing = 1.25

        # The stock heading styles are blue; force them to black.
        for name, size in (
            ("Title", 22),
            ("Heading 1", 16),
            ("Heading 2", 13),
            ("Heading 3", 11.5),
            ("Heading 4", 11),
        ):
            try:
                st = d.styles[name]
            except KeyError:
                continue
            st.font.name = HEAD_FONT
            st.font.size = Pt(size)
            st.font.color.rgb = BLACK
            st.font.bold = True
            st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
            st.paragraph_format.space_after = Pt(6)
            st.paragraph_format.keep_with_next = True

    def _setup_first_section(self) -> None:
        s = self.doc.sections[0]
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
        s.different_first_page_header_footer = True  # title page carries no number
        add_page_number_footer(s)

    # -- structural -----------------------------------------------------

    def page_break(self) -> None:
        p = self.doc.add_paragraph()
        p.add_run().add_break(WD_BREAK.PAGE)

    def usable_width_cm(self) -> float:
        s = self.doc.sections[-1]
        # Arithmetic on Length objects yields a plain EMU int, so rewrap it.
        return Emu(s.page_width - s.left_margin - s.right_margin).cm

    # -- content --------------------------------------------------------

    def title_page(self, title: str, subtitle: str, fields: list[str]) -> None:
        d = self.doc
        for _ in range(4):
            d.add_paragraph()

        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
        style_run(r, size=22, color=BLACK, font=HEAD_FONT)

        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        r = p.add_run(subtitle)
        style_run(r, size=13, color=GREY, font=HEAD_FONT)

        for _ in range(5):
            d.add_paragraph()

        for line in fields:
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(14)
            label, _, _rest = line.partition(":")
            r = p.add_run(f"{label.strip()}:  ")
            r.bold = True
            style_run(r, size=11.5, color=DARK)
            r = p.add_run("_" * 34)
            style_run(r, size=11.5, color=GREY)

        self.page_break()

    def heading(self, level: int, text: str) -> None:
        text = unescape(text)
        if level == 1 and self.pending_chapter_break:
            self.page_break()
        h = self.doc.add_heading(level=min(level, 4))
        h.paragraph_format.keep_with_next = True
        add_runs(h, text, size=None, color=BLACK, font=HEAD_FONT, base_bold=True)
        for r in h.runs:
            r.bold = True
            r.font.color.rgb = BLACK
        if level == 1:
            self.pending_chapter_break = True
            self.front_matter = False

    def paragraph(self, text: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # A reference entry is the target of every in-text citation to it, so it
        # carries the bookmark and renders its own leading number as plain text.
        ref = re.match(r"^\[(\d{1,2})\]\s", text) if self.in_references else None
        if ref:
            p.paragraph_format.space_after = Pt(7)
            p.paragraph_format.left_indent = Cm(0.95)
            p.paragraph_format.first_line_indent = Cm(-0.95)
            add_runs(p, text, link_citations=False)
            add_bookmark(p, bookmark_name(ref.group(1)), self._bookmark_id)
            self._bookmark_id += 1
            return

        if self.front_matter:
            # Separates each contents group from the tight list above it.
            p.paragraph_format.space_before = Pt(10)
        add_runs(p, text)

    def caption(self, text: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(14)
        add_runs(p, text, size=9.5, color=GREY)
        for r in p.runs:
            r.italic = True

    def image(self, path: Path) -> None:
        if not path.exists():
            self.paragraph(f"[missing figure: {path.name}]")
            return
        avail = self.usable_width_cm()
        self.doc.add_picture(str(path), width=Cm(min(15.5, avail)))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.paragraphs[-1].paragraph_format.space_before = Pt(10)
        self.doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
        self.doc.paragraphs[-1].paragraph_format.keep_with_next = True

    def quote(self, text: str) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.right_indent = Cm(1.0)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, text, size=10.5, color=DARK)
        for r in p.runs:
            r.italic = True

    def bullet(self, text: str, numbered: bool = False) -> None:
        p = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        if self.front_matter:
            # The contents listing is set tight so it holds to a single page.
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            add_runs(p, text, size=10)
        else:
            p.paragraph_format.space_after = Pt(4)
            add_runs(p, text)

    def table(self, header: list[str], rows: list[list[str]]) -> None:
        ncols = len(header)

        # Bind the preceding line to the table so a heading or lead-in sentence
        # cannot be left stranded at the foot of a page above it.
        if self.doc.paragraphs:
            self.doc.paragraphs[-1].paragraph_format.keep_with_next = True

        t = self.doc.add_table(rows=1 + len(rows), cols=ncols)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False

        size = 7.5 if ncols >= 6 else (8.5 if ncols >= 4 else 9.0)

        # Allocate column width from mean content length, compressed so a short
        # column is not starved, then floored so nothing becomes a sliver.
        lengths = []
        for c in range(ncols):
            vals = [len(header[c])] + [len(r[c]) if c < len(r) else 0 for r in rows]
            lengths.append(max(6.0, sum(vals) / len(vals)))
        lo, hi = min(lengths), max(lengths)
        if hi > lo:
            lengths = [1.0 + 1.8 * (v - lo) / (hi - lo) for v in lengths]

        avail = self.usable_width_cm() - 0.1
        total = sum(lengths)
        raw = [avail * v / total for v in lengths]

        floor_cm = min(2.1, avail / ncols)
        if min(raw) < floor_cm:
            short = [i for i, w in enumerate(raw) if w < floor_cm]
            spare = avail - floor_cm * len(short)
            rest = sum(raw[i] for i in range(ncols) if i not in short) or 1.0
            raw = [
                floor_cm if i in short else spare * raw[i] / rest
                for i in range(ncols)
            ]
        widths = [Cm(w) for w in raw]

        hdr = t.rows[0]
        repeat_header_row(hdr)
        for row in t.rows:
            keep_row_whole(row)
        for c, text in enumerate(header):
            cell = hdr.cells[c]
            cell.text = ""
            set_cell_background(cell, "EDEDED")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, text, size=size, color=BLACK, font=TABLE_FONT, base_bold=True)
            for r in p.runs:
                r.bold = True

        for ri, row in enumerate(rows, start=1):
            for c in range(ncols):
                cell = t.rows[ri].cells[c]
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                add_runs(
                    p,
                    row[c] if c < len(row) else "",
                    size=size,
                    color=DARK,
                    font=TABLE_FONT,
                )

        # Width must be stamped on every cell, not just the column object.
        for row in t.rows:
            for c, w in enumerate(widths):
                row.cells[c].width = w

        # Chain every row to the one below it so Word moves the whole table to
        # the next page rather than splitting it. The last row is left unchained,
        # otherwise the table would also drag the following paragraph along.
        for row in t.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True

        self.doc.add_paragraph().paragraph_format.space_after = Pt(6)


# ------------------------------------------------------------------ parse


def build() -> None:
    b = Builder()
    lines = SRC.read_text(encoding="utf-8").splitlines()

    # The Markdown title block becomes a formatted title page instead.
    b.title_page(
        "Natural Language Autoencoders for Redundancy Diagnostics",
        "Project Review I: Introduction and Literature Review",
        [
            "Student name",
            "Registration number",
            "Programme / Department",
            "Supervisor",
            "Date of submission",
        ],
    )

    i = 0
    # Skip the Markdown front matter already rendered as the title page.
    while i < len(lines) and not lines[i].startswith("## Abstract of Scope"):
        i += 1

    while i < len(lines):
        raw = lines[i].rstrip()
        s = raw.strip()

        if not s:
            i += 1
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
            i += 1
            continue

        if s.startswith("## References"):
            b.in_references = True

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            # Treat the two chapter headings as level 1, everything else below.
            b.heading(level, text)
            i += 1
            continue

        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", s)
        if m:
            b.image(ROOT / m.group(1))
            i += 1
            continue

        if s.startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            header = split_row(s)
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            b.table(header, rows)
            i = j
            continue

        if s.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                buf.append(lines[i].strip()[2:])
                i += 1
            b.quote(" ".join(buf))
            continue

        if re.match(r"^[-*]\s+", s):
            b.bullet(re.sub(r"^[-*]\s+", "", s))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", s):
            b.bullet(re.sub(r"^\d+\.\s+", "", s), numbered=True)
            i += 1
            continue

        # Figure captions are centred and italicised rather than justified.
        if re.match(r"^\*\*Figure\s", s):
            b.caption(s)
            i += 1
            continue

        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if re.match(r"^(#{1,6}\s|>\s|[-*]\s|\d+\.\s|\||!\[)", nxt.strip()):
                break
            if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", nxt.strip()):
                break
            buf.append(nxt.strip())
            i += 1
        b.paragraph(" ".join(buf))

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DST
    b.doc.save(out)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
