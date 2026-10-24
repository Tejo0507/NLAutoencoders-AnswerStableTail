"""The run-artifact release (D63-D65): what is packaged, sanitised and scanned.

The package must hold exactly the classified runs, with a checksum list that
verifies and an archive that is reproducible; a run nobody classified must
stop the build rather than ship or vanish. Local path roots are replaced in
the staged copy only, keeping the rest of each path, and every changed file is
recorded with its original hash; everything else stays byte for byte. Nothing
the scan still finds may reach an archive. The scan must find local paths and
tokens without printing a token, and must not mistake the "\\n" escapes of
JSON-encoded model text for drive paths.
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


def _build(runs, out, commit="c0ffee", identities=frozenset()):
    return B.build(runs, out, "draft", commit, "2026-10-18", "a.tar.gz", set(identities))


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


def test_an_extra_run_outside_runs_is_staged_under_its_name(runs, tmp_path):
    """A released entry kept in an isolated tree is packaged like any other run."""
    outside = tmp_path / "experiment" / "main_paraphrase_llm"
    (outside / "sub").mkdir(parents=True)
    (outside / "analyse.json").write_text('{"n_claims": 246}')
    (outside / "sub" / "rows.jsonl").write_text('{"id": "x"}\n')
    import shutil
    shutil.rmtree(runs / "main_paraphrase_llm")
    res = _build_extra(runs, tmp_path / "out", {"main_paraphrase_llm": outside})
    stage = tmp_path / "out" / B.TOP
    assert (stage / "runs" / "main_paraphrase_llm" / "analyse.json").read_bytes() == \
        (outside / "analyse.json").read_bytes()
    assert (stage / "runs" / "main_paraphrase_llm" / "sub" / "rows.jsonl").exists()
    assert res["run_files"] > 0
    assert (outside / "analyse.json").read_text() == '{"n_claims": 246}'   # source untouched


def test_an_unclassified_extra_run_stops_the_build(runs, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(SystemExit, match="unclassified extra run"):
        _build_extra(runs, tmp_path / "out", {"not_classified": outside})


def test_an_extra_run_also_under_runs_stops_the_build(runs, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(SystemExit, match="both under"):
        _build_extra(runs, tmp_path / "out", {"main_paraphrase_llm": outside})


def _build_extra(runs, out, extra):
    return B.build(runs, out, "draft", "c0ffee", "2026-10-18", "a.tar.gz", set(), extra=extra)


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


# --- sanitisation (D65) ------------------------------------------------------

REPO = "NLAutoencoders-AnswerStableTail"


@pytest.mark.parametrize("original, released", [
    (f"D:\\\\{REPO}\\\\runs\\\\main\\\\traces", "<repo>\\\\runs\\\\main\\\\traces"),
    ("D:\\\\nla_models\\\\nf4\\\\Qwen__Qwen2.5-7B-Instruct", "<local>\\\\nf4\\\\Qwen__Qwen2.5-7B-Instruct"),
    ("E:\\\\SomeOtherProject\\\\.cache\\\\huggingface", "<local>\\\\.cache\\\\huggingface"),
    ("C:\\\\Users\\\\someone\\\\cache\\\\hub", "<local>\\\\cache\\\\hub"),
    (f"D:/{REPO}/data/raw/gsm8k/test.jsonl", "<repo>/data/raw/gsm8k/test.jsonl"),
])
def test_json_paths_lose_their_root_and_keep_the_rest(original, released):
    text = json.dumps({"p": "x"}).replace('"x"', f'"{original}"')
    new, n = B.sanitise_text(text, "json")
    assert n == 1
    assert json.loads(new)["p"] == json.loads(f'"{released}"')
    assert "SomeOtherProject" not in new and "someone" not in new


@pytest.mark.parametrize("value, released", [
    (f"D:\\\\{REPO}", "<repo>"), ("E:\\\\hf_cache", "<local>"), (f"D:/{REPO}", "<repo>")])
def test_a_bare_root_that_is_a_whole_value_is_sanitised_and_scanned(value, released, tmp_path):
    text = '{"path": "' + value + '", "text": "so s:\\\\frac{1}{2} and s:\\n  x"}'
    new, n = B.sanitise_text(text, "json")
    assert n == 1 and json.loads(new)["path"] == released
    assert json.loads(new)["text"] == json.loads(text)["text"]
    (tmp_path / "env.json").write_text(text)
    assert set(S.scan(tmp_path, set())[1]["env.json"]) == {"abs_windows_root"}


def test_json_escapes_in_model_text_are_not_paths():
    text = json.dumps({"text": "The answer is:\n\\[ 5 \\]\nSo:\n- a\\b"})
    assert B.sanitise_text(text, "json") == (text, 0)


def test_log_paths_with_one_or_two_backslashes_are_sanitised():
    log = (f"INFO wrote D:\\{REPO}\\runs\\main\\x.json\n"
           f'  File "D:\\{REPO}\\.venv\\Lib\\site-packages\\t.py", line 3\n'
           "INFO loading from E:\\hf_cache\\hub\\models--Qwen\n"
           f"INFO repr 'D:\\\\{REPO}\\\\runs'\n")
    new, n = B.sanitise_text(log, "text")
    assert n == 4
    assert "<repo>\\runs\\main\\x.json" in new and "<repo>\\.venv\\Lib" in new
    assert "<local>\\hub\\models--Qwen" in new and "<repo>\\\\runs" in new
    assert ":\\" not in new


@pytest.fixture
def runs_with_paths(runs):
    (runs / "main" / "env").mkdir()
    env = {"disks": {"model_cache": {"path": "E:\\SomeOtherProject\\.cache\\huggingface",
                                     "free_gb": 12.5},
                     "repo": {"path": f"D:\\{REPO}"}},
           "environment": {"python_executable": f"D:\\{REPO}\\.venv\\Scripts\\python.exe"},
           "revision": "a09a35458c702b33eeacc393d103063234e8bc28"}
    (runs / "main" / "env" / "environment.json").write_text(json.dumps(env, indent=2))
    (runs / "pilot" / "run.log").write_text(
        f"10:00 INFO run=pilot dir=D:\\{REPO}\\runs\\pilot\n10:01 INFO no path here\n")
    return runs


def test_sanitised_files_are_recorded_with_their_original_hash(runs_with_paths, tmp_path):
    res = _build(runs_with_paths, tmp_path / "out")
    stage = tmp_path / "out" / B.TOP
    record = json.loads((stage / "TRANSFORMS.json").read_text())
    assert res["sanitised_files"] == record["n_files"] == 2
    env = record["files"]["runs/main/env/environment.json"]
    src = runs_with_paths / "main" / "env" / "environment.json"
    assert env["original_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    released = stage / "runs" / "main" / "env" / "environment.json"
    assert env["released_sha256"] == hashlib.sha256(released.read_bytes()).hexdigest()
    assert set(env["fields"]) == {"disks.model_cache.path", "environment.python_executable",
                                  "disks.repo.path"}
    assert record["files"]["runs/pilot/run.log"]["lines"] == [1]
    # Provenance that is not a path root survives exactly.
    out = json.loads(released.read_text())
    assert out["disks"]["model_cache"] == {"path": "<local>\\.cache\\huggingface", "free_gb": 12.5}
    assert out["environment"]["python_executable"] == "<repo>\\.venv\\Scripts\\python.exe"
    assert out["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    # A bare root - drive and one directory - is a local path too.
    assert out["disks"]["repo"]["path"] == "<repo>"


def test_sanitisation_never_touches_the_source_runs_and_spares_other_files(runs_with_paths, tmp_path):
    before = {p: p.read_bytes() for p in runs_with_paths.rglob("*") if p.is_file()}
    _build(runs_with_paths, tmp_path / "out")
    assert {p: p.read_bytes() for p in runs_with_paths.rglob("*") if p.is_file()} == before
    stage = tmp_path / "out" / B.TOP
    changed = set(json.loads((stage / "TRANSFORMS.json").read_text())["files"])
    for p, data in before.items():
        rel = f"runs/{p.relative_to(runs_with_paths).as_posix()}"
        if p.name in B.SKIP_NAMES or rel.startswith("runs/tiny/"):
            continue
        if rel not in changed:
            assert (stage / rel).read_bytes() == data, rel
    assert S.scan(stage, set())[1] == {}


def test_a_token_in_a_run_stops_the_build_before_any_archive(runs, tmp_path):
    (runs / "main" / "leak.txt").write_text("hf_" + "B" * 34)
    with pytest.raises(SystemExit, match="scan failed"):
        _build(runs, tmp_path / "out")
    assert not (tmp_path / "out" / "a.tar.gz").exists()


def test_a_path_the_transformation_cannot_reach_stops_the_build(runs, tmp_path):
    import numpy as np
    with open(runs / "main" / "sub" / "block.npz", "wb") as fh:
        np.savez(fh, meta=np.asarray([json.dumps({"src": "D:\\models\\nf4\\x"})], dtype=object))
    with pytest.raises(SystemExit, match="block.npz"):
        _build(runs, tmp_path / "out")
    assert not (tmp_path / "out" / "a.tar.gz").exists()


def _retain(runs, monkeypatch, content: bytes, pinned: bytes):
    target = runs / "main_f6_diag_validation" / "validate.py"
    target.write_bytes(content)
    monkeypatch.setattr(B, "RETAINED", {"runs/main_f6_diag_validation/validate.py": {
        "sha256": hashlib.sha256(pinned).hexdigest(), "reason": "test"}})
    return target


def test_a_retained_file_ships_unmodified_only_with_its_pinned_content(runs, tmp_path, monkeypatch):
    code = f'R = Path(r"D:\\{REPO}")\n'.encode()
    _retain(runs, monkeypatch, code, code)
    _build(runs, tmp_path / "out")
    stage = tmp_path / "out" / B.TOP
    assert (stage / "runs" / "main_f6_diag_validation" / "validate.py").read_bytes() == code
    record = json.loads((stage / "TRANSFORMS.json").read_text())
    assert "runs/main_f6_diag_validation/validate.py" not in record["files"]
    assert "runs/main_f6_diag_validation/validate.py" in record["retained_unmodified"]


def test_a_retained_file_with_other_content_stops_the_build(runs, tmp_path, monkeypatch):
    pinned = f'R = Path(r"D:\\{REPO}")\n'.encode()
    _retain(runs, monkeypatch, f'R = Path(r"E:\\Elsewhere\\x")\n'.encode(), pinned)
    with pytest.raises(SystemExit, match="validate.py"):
        _build(runs, tmp_path / "out")


def test_a_retained_file_still_fails_on_anything_but_a_path(runs, tmp_path, monkeypatch):
    code = ("token = 'hf_" + "C" * 34 + "'\n").encode()
    _retain(runs, monkeypatch, code, code)
    with pytest.raises(SystemExit, match="validate.py"):
        _build(runs, tmp_path / "out")


def test_a_local_identity_stops_the_build(runs, tmp_path):
    (runs / "main" / "note.txt").write_text("run by someone-local on this machine")
    with pytest.raises(SystemExit, match="identity"):
        _build(runs, tmp_path / "out", identities={"someone-local"})
