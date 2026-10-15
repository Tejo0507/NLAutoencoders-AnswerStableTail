"""F8 must never start the 15 GB bf16 download on its own.

F8 is in every config's `enabled_tests`, so `run_falsification_tests.py
--config main` runs it. It asked `snapshot` for the bf16 target, which calls
`snapshot_download`; on a run with no earlier F8 result to carry forward - any
fresh run, including `main` - that started a 15.3 GB download onto a cache
volume with 12 GB free. The documented post-trace order runs this stage without
`--only`, so avoiding the generic orchestrator did not avoid the download.

F8 now resolves the snapshot with `local_files_only` unless `--allow-download`
is given, and reports `blocked` with the reason when the weights are absent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_falsification_tests as rft  # noqa: E402

LOG = logging.getLogger("test_f8")


class _Target:
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    revision = None
    layer = 20


class _Cfg:
    target = _Target()

    def __init__(self, path):
        self.dir = path


@pytest.fixture
def calls(monkeypatch, tmp_path):
    seen = []

    class Store:
        def __init__(self, *_):
            pass

        def has(self, *_):
            return True

    def fake_snapshot(repo_id, revision=None, allow_patterns=None,
                      local_files_only=False):
        seen.append(local_files_only)
        raise FileNotFoundError("not in the local cache")

    monkeypatch.setattr(rft, "ActivationStore", Store)
    import nlaast.models.loading as loading
    monkeypatch.setattr(loading, "snapshot", fake_snapshot)
    return seen, _Cfg(tmp_path)


TRACES = {"p0": {"problem_id": "p0"}}


def test_by_default_it_only_looks_on_disk(calls):
    seen, cfg = calls
    out = rft.f8_quantisation(cfg, TRACES, LOG, subsample=1)
    assert seen == [True]
    assert out["status"] == "blocked"
    assert out["download_attempted"] is False
    assert "--allow-download" in out["reason"]


def test_a_download_happens_only_when_asked_for(calls):
    seen, cfg = calls
    out = rft.f8_quantisation(cfg, TRACES, LOG, subsample=1, allow_download=True)
    assert seen == [False]
    assert out["download_attempted"] is True


def test_an_earlier_measurement_is_still_carried_forward(calls):
    """The pilot's F8 result must survive a later `--force` pass unchanged."""
    _, cfg = calls
    previous = {"status": "ran", "mean_cosine": 0.987}
    out = rft.f8_quantisation(cfg, TRACES, LOG, subsample=1, previous=previous)
    assert out["status"] == "ran"
    assert out["mean_cosine"] == 0.987
    assert out["reused_from_earlier_run"] is True


def test_the_flag_is_on_the_command_line():
    import inspect
    assert "--allow-download" in inspect.getsource(rft.main)
