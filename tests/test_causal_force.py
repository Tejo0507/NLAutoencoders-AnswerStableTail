"""`run_interventions.py --force` must regenerate, not replay.

The interventions writer resumes by row id. `--force` only bypassed the
stage-complete check, so the writer found every id already present and
skipped it: the pilot's D34 validation finished in 11 s and re-validated the
pre-fix no-op rows. `--force` now moves the previous rows aside first.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from nlaast.logging_utils import JsonlWriter, get  # noqa: E402

OLD = [{"id": "p0|ablate_dir|1.0|None", "notes": {}},
       {"id": "p1|ablate_dir|1.0|None", "notes": {}}]


@pytest.fixture(scope="module")
def ri():
    return importlib.import_module("run_interventions")


@pytest.fixture
def rows_path(tmp_path):
    p = tmp_path / "interventions.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in OLD), encoding="utf-8")
    return p


def test_without_archiving_the_writer_would_replay_every_row(rows_path):
    """The defect, stated: a resumed writer treats every old row as done."""
    w = JsonlWriter(rows_path, key="id")
    assert all(w.has(r["id"]) for r in OLD)


def test_archiving_leaves_nothing_to_resume(ri, rows_path):
    archived = ri.archive_previous_rows(rows_path, get())
    w = JsonlWriter(rows_path, key="id")
    assert not any(w.has(r["id"]) for r in OLD), "every arm is regenerated"
    assert archived is not None and archived.name.startswith("interventions.superseded-")


def test_the_old_rows_are_kept_byte_for_byte(ri, rows_path):
    before = rows_path.read_bytes()
    archived = ri.archive_previous_rows(rows_path, get())
    assert archived.read_bytes() == before
    assert not rows_path.exists()


def test_a_second_archive_with_the_same_stamp_does_not_overwrite(ri, rows_path):
    """Two --force runs on rows with one mtime must keep both sets."""
    first = ri.archive_previous_rows(rows_path, get())
    rows_path.write_text(json.dumps({"id": "new"}) + "\n", encoding="utf-8")
    st = first.stat()
    os.utime(rows_path, (st.st_atime, st.st_mtime))
    second = ri.archive_previous_rows(rows_path, get())
    assert first != second and first.exists() and second.exists()
    assert "new" in second.read_text(encoding="utf-8")


def test_nothing_to_archive_is_a_no_op(ri, tmp_path):
    assert ri.archive_previous_rows(tmp_path / "interventions.jsonl", get()) is None
    empty = tmp_path / "interventions.jsonl"
    empty.write_text("", encoding="utf-8")
    assert ri.archive_previous_rows(empty, get()) is None
    assert empty.exists()
