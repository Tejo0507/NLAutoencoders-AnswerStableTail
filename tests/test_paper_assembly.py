"""Assembling the report, and refusing to pretend it is finished.

The literature review was submitted as one Word document rendered from one
Markdown file. The final report has to be submitted the same way and `paper/`
is a set of section files, so something has to join them. The risk in writing
that something is that it quietly produces a document which *looks* complete:
a missing Results section omitted rather than named, or a stub whose prose
reads like a finding.

These tests pin the opposite behaviour - a declared section that does not exist
is written into the document as an explicit placeholder naming its blocker, the
table of contents marks it, and the exit code says the document is not
submission-ready.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import assemble_paper as ap  # noqa: E402


@pytest.fixture(scope="module")
def built():
    markdown, pending, notes = ap.assemble()
    return markdown, pending, notes


class TestTheDocumentNamesWhatIsMissing:
    def test_every_declared_section_appears_in_the_document(self, built):
        markdown, _, _ = built
        for s in ap.SECTIONS:
            assert s.title in markdown, f"{s.title} is not in the document"

    def test_a_section_with_no_file_becomes_a_named_placeholder(self, monkeypatch):
        absent = ap.Section("6", "Results", None, "the main run")
        monkeypatch.setattr(ap, "SECTIONS", (*ap.SECTIONS[:6], absent))
        markdown, pending, _ = ap.assemble()
        assert absent in pending
        assert "**Not written.**" in markdown
        assert "Waiting on the main run." in markdown

    def test_the_placeholder_states_no_finding_and_no_number(self):
        """A stub that reads like a result is the failure mode here."""
        s = ap.Section("6", "Results", None, "the main run")
        text = ap.placeholder(s)
        assert "Not written" in text
        assert "the main run" in text
        assert "no number in it" in text

    def test_the_table_of_contents_marks_the_unwritten_sections(self, built):
        markdown, _, _ = built
        toc = markdown.split("## Table of Contents", 1)[1].split("---", 1)[0]
        for s in ap.SECTIONS:
            label = f"{s.number}. {s.title}" if s.number else s.title
            line = next(ln for ln in toc.splitlines() if label in ln)
            present = bool(s.filename and (ap.PAPER / s.filename).exists())
            assert ("(not written)" in line) is not present

    def test_a_section_present_but_still_waiting_is_reported_as_pending(self, monkeypatch):
        """A file that exists but is declared as waiting is not 'done'."""
        waiting = ap.Section("4", "Experimental setup", "04_experimental_setup.md",
                             "the achieved N")
        monkeypatch.setattr(ap, "SECTIONS", (ap.SECTIONS[1], waiting))
        _, pending, _ = ap.assemble()
        assert [s.title for s in pending] == ["Experimental setup"]

    def test_an_unresolved_marker_makes_a_present_section_pending(self, tmp_path, monkeypatch):
        """Presence is not completion: a [VERIFY] or [N] left in the text counts."""
        monkeypatch.setattr(ap, "PAPER", tmp_path)
        (tmp_path / "a.md").write_text("# 1. A\n\n*Final.*\n\nDone.\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("# 2. B\n\n*Final.*\n\nThe N is **[N]**.\n",
                                       encoding="utf-8")
        monkeypatch.setattr(ap, "SECTIONS", (ap.Section("1", "A", "a.md"),
                                             ap.Section("2", "B", "b.md")))
        _, pending, _ = ap.assemble()
        assert [(s.title, s.waiting_on) for s in pending] == [("B", "unresolved **[N]** markers")]

    def test_the_report_as_it_stands_is_complete(self, built):
        """Every declared section exists and none carries an unfinished marker."""
        markdown, pending, _ = built
        assert pending == []
        assert "**Not written.**" not in markdown
        assert not ap.unresolved_markers(markdown)


class TestDraftingNotesDoNotReachTheDocument:
    def test_the_italic_note_under_a_heading_is_lifted_out(self):
        body, notes = ap.strip_draft_notes(
            "# 3. Method\n\n*Draft. Waiting on nothing.*\n\nReal content.\n")
        assert "Draft." not in body
        assert "Real content." in body
        assert body.startswith("# 3. Method")
        assert notes == ["Draft. Waiting on nothing."]

    def test_a_section_without_a_note_is_untouched(self):
        body, notes = ap.strip_draft_notes("# 1. Introduction\n\nContent.\n")
        assert notes == []
        assert "Content." in body

    def test_italic_body_text_further_down_is_not_mistaken_for_a_note(self):
        body, notes = ap.strip_draft_notes(
            "# 1. Introduction\n\nContent.\n\n*An emphasised aside.*\n")
        assert "*An emphasised aside.*" in body
        assert notes == []

    def test_every_real_section_carries_a_note_and_loses_it(self, built):
        markdown, _, notes = built
        assert notes, "the drafts each open with a note; none was found"
        assert "*Draft." not in markdown


class TestReferences:
    def test_the_bibliography_comes_from_the_tracked_copy(self):
        """Not from the gitignored review, so the report builds from the repository."""
        assert ap.REFERENCES == ap.PAPER / "references.md"
        refs = ap.references()
        assert len(refs) == 27
        assert refs[0].startswith("[1]")

    def test_every_citation_in_the_text_has_an_entry(self, built):
        markdown, _, _ = built
        numbers = {int(r[1:r.index("]")]) for r in ap.references()}
        assert ap.cited_numbers(markdown.split("## References", 1)[0]) <= numbers

    def test_each_entry_is_numbered_once_and_in_order(self):
        import re
        numbers = [int(re.match(r"^\[(\d+)\]", r).group(1))
                   for r in ap.references()]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))

    def test_an_entry_no_section_cites_is_marked_rather_than_dropped(self, built):
        """Dropping it would hide that the bibliography and the text disagree."""
        markdown, _, _ = built
        cited = ap.cited_numbers(markdown.split("## References", 1)[0])
        assert cited, "the drafts carry [n] citations"
        uncited = [r for r in ap.references()
                   if int(r[1:r.index("]")]) not in cited]
        refs_block = markdown.split("## References", 1)[1]
        assert refs_block.count("*(not yet cited)*") == len(uncited)

    def test_the_citation_scanner_reads_bracketed_numbers_only(self):
        assert ap.cited_numbers("as [7] and [21] show") == {7, 21}
        assert ap.cited_numbers("see section [a] or figure 7") == set()


class TestTheDeclarationMatchesTheRecord:
    """`paper/README.md`'s status table is the plan; `SECTIONS` executes it.

    The repository already has a test asserting `configs/base.yaml` cannot
    drift from `config.py` for the same reason: two statements of one thing
    diverge silently, and here the divergence would be a section that the plan
    lists and the document omits.
    """

    def _table(self):
        import re
        text = (ap.PAPER / "README.md").read_text(encoding="utf-8")
        rows = []
        for line in text.splitlines():
            m = re.match(r"^\|\s*([0-9]+|—)\s*\|\s*([^|]+?)\s*\|", line)
            if m:
                number = "" if m.group(1) == "—" else m.group(1)
                title = re.sub(r"\[`?([^`\]]+)`?\]\([^)]*\)", r"\1",
                               m.group(2)).strip()
                rows.append((number, title))
        return rows

    def test_the_readme_lists_a_row_for_every_declared_section(self):
        rows = self._table()
        assert rows, "the status table was not found or could not be parsed"
        assert len(rows) == len(ap.SECTIONS)

    def test_the_same_sections_are_declared_on_both_sides(self):
        """As a set: the README lists the abstract last, the document puts it
        first, and both are right about their own ordering."""
        assert ({n for n, _ in self._table()}
                == {s.number for s in ap.SECTIONS})

    def test_the_document_leads_with_the_abstract(self):
        assert ap.SECTIONS[0].title == "Abstract"
        assert [s.number for s in ap.SECTIONS[1:]] == [
            str(i) for i in range(1, 6)] + ["6", "7", "8"]

    def test_every_declared_file_exists_or_is_declared_absent(self):
        for s in ap.SECTIONS:
            if s.filename is None:
                continue
            assert (ap.PAPER / s.filename).exists(), (
                f"{s.filename} is declared but missing; declare it as "
                f"filename=None with what it waits on, or restore the file")

    def test_a_section_with_no_file_states_what_it_waits_on(self):
        for s in ap.SECTIONS:
            if s.filename is None:
                assert s.waiting_on, f"{s.title} is absent with no blocker given"


class TestFrontMatter:
    def test_the_title_page_fields_match_the_review(self, built):
        markdown, _, _ = built
        head = markdown.split("## Table of Contents", 1)[0]
        for field in ap.FRONT_MATTER_FIELDS:
            assert f"**{field}:**" in head

    def test_the_document_opens_with_the_project_title(self, built):
        markdown, _, _ = built
        assert markdown.startswith(f"# {ap.TITLE}")
