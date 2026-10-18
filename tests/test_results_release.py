"""The run-artifact release (D63): what is packaged, and what the scan reports.

The package must hold exactly the classified runs, byte for byte, with a
checksum list that verifies and an archive that is reproducible; a run nobody
classified must stop the build rather than ship or vanish. The scan must find
local paths and tokens without printing a token, and must not mistake the
"\\n" escapes of JSON-encoded model text for drive paths.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_results_release as B  # noqa: E402
import scan_release as S  # noqa: E402


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    for name in B.RELEASED:
        if name.endswith(".json"):
            root.mkdir(exist_ok=True)
            (root / name).write_text('{"f8": 1}')
        else:
            (root / name / "sub").mkdir(parents=True)
            (root / name / "manifest.json").write_text(f'{{"run": "{name}"}}')
            (root / name / "sub" / "rows.jsonl").write_text(f'{{"id": "{name}"}}\n')
    (root / "main" / "manifest.json.lock").write_bytes(b"")
    (root / "tiny").mkdir()
    (root / "tiny" / "fixture.json").write_text("{}")
    return root


def _build(runs, out, commit="c0ffee"):
    return B.build(runs, out, "draft", commit, "2026-10-18", "a.tar.gz")


def test_the_package_holds_exactly_the_released_runs_byte_for_byte(runs, tmp_path):
    res = _build(runs, tmp_path / "out")
    stage = tmp_path / "out" / B.TOP
    staged = {p.relative_to(stage / "runs").as_posix()
              for p in (stage / "runs").rglob("*") if p.is_file()}
    expected = {p.relative_to(runs).as_posix() for p in runs.rglob("*")
                if p.is_file() and p.name not in B.SKIP_NAMES
                and p.relative_to(runs).parts[0] in B.RELEASED}
    assert staged == expected
    assert res["run_files"] == len(expected)
    for rel in expected:
        assert (stage / "runs" / rel).read_bytes() == (runs / rel).read_bytes()
    assert not any("tiny" in s or s.endswith(".lock") for s in staged)
    meta = json.loads((stage / "RUNS.json").read_text())
    assert meta["source_commit"] == "c0ffee" and "DRAFT" in meta["status"]
    assert {r["path"] for r in meta["runs"]} == {f"runs/{n}" for n in B.RELEASED}
    assert "DRAFT" in (stage / "README.md").read_text(encoding="utf-8")
    assert "{SOURCE_COMMIT}" not in (stage / "README.md").read_text(encoding="utf-8")


def test_checksums_cover_every_file_and_the_archive_matches_the_staging(runs, tmp_path):
    _build(runs, tmp_path / "out")
    out = tmp_path / "out"
    stage = out / B.TOP
    lines = (stage / "SHA256SUMS.txt").read_text().splitlines()
    listed = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
        assert hashlib.sha256((stage / rel).read_bytes()).hexdigest() == digest
    on_disk = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert on_disk - set(listed) == {"SHA256SUMS.txt"}
    with tarfile.open(out / "a.tar.gz", "r:gz") as tf:
        members = tf.getmembers()
        assert all(m.isfile() and m.name.startswith(f"{B.TOP}/") and ".." not in m.name
                   for m in members)
        assert {m.name[len(B.TOP) + 1:] for m in members} == on_disk
        for m in members:
            assert tf.extractfile(m).read() == (stage / m.name[len(B.TOP) + 1:]).read_bytes()


def test_the_archive_is_reproducible(runs, tmp_path):
    a = _build(runs, tmp_path / "one")["archive_sha256"]
    b = _build(runs, tmp_path / "two")["archive_sha256"]
    assert a == b


def test_an_unclassified_run_stops_the_build(runs, tmp_path):
    (runs / "new_run").mkdir()
    with pytest.raises(SystemExit, match="unclassified"):
        _build(runs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_missing_released_run_stops_the_build(runs, tmp_path):
    (runs / "main_f8_comparison.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        _build(runs, tmp_path / "out")


def test_an_existing_output_is_not_overwritten(runs, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="refusing"):
        _build(runs, tmp_path / "out")


def test_the_scan_finds_paths_and_tokens_without_printing_the_token(tmp_path, capsys, monkeypatch):
    token = "hf_" + "A" * 34
    (tmp_path / "log.txt").write_text(f"loaded from D:\\models\\nf4\\target with token={token}\n")
    (tmp_path / "trace.jsonl").write_text('{"text": "So the answer is:\\n\\\\[ 5 \\\\]"}\n')
    monkeypatch.setattr(sys, "argv", ["scan_release.py", str(tmp_path), "--identity", "someone-local"])
    assert S.main() == 1
    out = capsys.readouterr().out
    assert token not in out
    assert "log.txt\tabs_windows_path" in out and "log.txt\thf_token" in out
    assert "trace.jsonl" not in out


def test_the_scan_is_clean_on_clean_text(tmp_path, monkeypatch):
    (tmp_path / "rows.jsonl").write_text('{"text": "Thus:\\nx = 2"}\n')
    monkeypatch.setattr(sys, "argv", ["scan_release.py", str(tmp_path)])
    assert S.main() == 0
