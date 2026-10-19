"""The F6 resize audit's plumbing: which run it reads, and what it will not overwrite.

The audit's numbers are quoted in ``docs/F6_RESIZE_AUDIT.md`` from the file it
writes, so a rerun must not replace that file silently, and ``--run-id`` must
select the run it names rather than the config's default. Its donor
reconstruction must also track ``permute_labels`` exactly, or every tiling and
truncation count it reports is about some other shuffle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _stage  # noqa: E402
import audit_f6_resize as A  # noqa: E402
from nlaast import paths  # noqa: E402
from nlaast.baselines.probe import permute_labels  # noqa: E402


@pytest.mark.parametrize("n_groups", [2, 3, 7, 25])
@pytest.mark.parametrize("seed", [0, 1, 43])
def test_donor_map_reproduces_permute_labels(n_groups, seed):
    rng = np.random.default_rng(seed)
    groups = [f"g{g:02d}" for g in range(n_groups) for _ in range(int(rng.integers(1, 6)))]
    y = rng.integers(0, 2, size=len(groups))
    shuffled, _ = permute_labels(y, groups, "across_groups", seed)
    donors = A.donor_map(groups, seed)
    by_g = {g: y[[i for i, h in enumerate(groups) if h == g]] for g in set(groups)}
    rebuilt = np.empty_like(y)
    for g in set(groups):
        idx = [i for i, h in enumerate(groups) if h == g]
        rebuilt[idx] = np.resize(by_g[donors[g]], len(idx))
    assert np.array_equal(rebuilt, shuffled)


def test_load_config_applies_run_id():
    args = _stage.base_parser("x").parse_args(["--config", "main", "--run-id", "elsewhere"])
    assert _stage.load_config(args).run_id == "elsewhere"
    args = _stage.base_parser("x").parse_args(["--config", "main"])
    assert _stage.load_config(args).run_id == "main"


def _existing_output(root: Path, run_id: str) -> Path:
    out = root / f"{run_id}_f6_audit" / "f6_resize_audit.json"
    out.parent.mkdir(parents=True)
    out.write_text('{"recorded": true}')
    return out


@pytest.mark.parametrize("argv, run_id", [(["--config", "main"], "main"),
                                          (["--config", "main", "--run-id", "other"], "other")])
def test_an_existing_audit_is_not_overwritten_without_force(tmp_path, monkeypatch, argv, run_id):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    out = _existing_output(tmp_path, run_id)

    def must_not_run(cfg):
        raise AssertionError("audit ran despite an existing output")

    monkeypatch.setattr(A, "audit", must_not_run)
    monkeypatch.setattr(sys, "argv", ["audit_f6_resize.py", *argv])
    assert A.main() == 1
    assert out.read_text() == '{"recorded": true}'


def test_force_rewrites_the_audit_for_the_named_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    out = _existing_output(tmp_path, "other")
    seen = []
    monkeypatch.setattr(A, "audit", lambda cfg: seen.append(cfg.run_id) or {"fresh": 1})
    monkeypatch.setattr(sys, "argv", ["audit_f6_resize.py", "--config", "main",
                                      "--run-id", "other", "--force"])
    assert A.main() == 0
    assert seen == ["other"]
    assert '"fresh": 1' in out.read_text()
