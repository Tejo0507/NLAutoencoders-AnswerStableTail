"""F8's partial bf16 replication driver: preparing the two arms, and comparing them.

The two arms must differ in one thing only, the precision of the target that
produced the activations, so ``prepare`` copies the run's upstream stages into
both and refuses to touch an arm that already exists. The comparison scores
windows exactly as O3 does (no Gaussian controls, no integrity failures), or a
bf16-vs-NF4 difference could be an exclusion difference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_f8_bf16_headline as F  # noqa: E402
from nlaast import config as config_mod  # noqa: E402
from nlaast import paths  # noqa: E402
from nlaast.activations.store import ActivationStore  # noqa: E402
from nlaast.logging_utils import read_json, write_jsonl  # noqa: E402

LOG = logging.getLogger("test_f8_headline")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A minimal source run: four problems in the NLA arm, two per dataset."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    d = config_mod.from_mapping({"run_id": "src"}).to_dict()
    d["robustness"] = {**d["robustness"], "quantisation_subsample": 2}
    cfg = config_mod.from_mapping(d)
    ids = ["gsm8k-a", "gsm8k-b", "math-a", "math-b"]
    for stage in F._COPY:
        (cfg.dir / stage).mkdir(parents=True)
    write_jsonl(cfg.dir / "ast" / "ast.jsonl",
                [{"problem_id": p, "dataset": p.split("-")[0]} for p in ids])
    (cfg.dir / "nla").mkdir()
    write_jsonl(cfg.dir / "nla" / "reconstructions.jsonl",
                [{"problem_id": p, "window_kind": "tail"} for p in ids]
                + [{"problem_id": "gsm8k-control-only", "window_kind": "gaussian_control"}])
    store = ActivationStore(cfg.dir / "acts")
    for i, p in enumerate(ids):
        store.put(p, cfg.target.layer, np.full((2, 4), float(i)), [3, 7],
                  ["tail", "tail"], {"chunk_indices": [1, 2]})
    return cfg


def test_prepare_builds_both_arms_from_a_stratified_subsample(run):
    assert F.prepare(run, LOG) == 0
    sub = read_json(F.arm_dir(run, "bf16") / "subsample.json")
    # One problem per dataset, not the two GSM8K ids a sorted prefix would give.
    assert sub["problem_ids"] == ["gsm8k-a", "math-a"]
    assert read_json(F.arm_dir(run, "nf4") / "subsample.json") == sub
    for arm in ("bf16", "nf4"):
        for stage in F._COPY:
            assert (F.arm_dir(run, arm) / stage).is_dir()
    # The NF4 arm carries the run's own activations for the subsample; the
    # bf16 arm starts empty, to be filled from the bf16 target.
    nf4 = ActivationStore(F.arm_dir(run, "nf4") / "acts")
    arr, meta = nf4.get("math-a", run.target.layer)
    assert np.array_equal(arr, np.full((2, 4), 2.0, dtype=np.float32))
    assert meta["positions"] == [3, 7]
    assert not nf4.has("gsm8k-b", run.target.layer)
    assert not ActivationStore(F.arm_dir(run, "bf16") / "acts").has("gsm8k-a", run.target.layer)


@pytest.mark.parametrize("present", ["bf16", "nf4"])
def test_prepare_refuses_before_creating_anything_when_either_arm_exists(run, present):
    F.arm_dir(run, present).mkdir()
    (F.arm_dir(run, present) / "keep.txt").write_text("existing")
    assert F.prepare(run, LOG) == 1
    other = "nf4" if present == "bf16" else "bf16"
    assert not F.arm_dir(run, other).exists()
    assert [p.name for p in F.arm_dir(run, present).iterdir()] == ["keep.txt"]


def _row(pid, kind="tail", chunk=1, cos=0.9, ok=True):
    return {"problem_id": pid, "window_kind": kind, "chunk_index": chunk,
            "cosine": cos, "integrity_ok": ok}


def test_window_scores_exclude_controls_integrity_failures_and_other_problems():
    rows = [_row("p", cos=0.8), _row("p", cos=0.6),
            _row("p", kind="gaussian_control", cos=0.1),
            _row("p", cos=0.0, ok=False),
            _row("p", kind="matched_position", cos=0.5),
            _row("q", cos=0.3)]
    s = F.window_scores(rows, {"p"})
    assert s == {("p", "tail", 1): pytest.approx(0.7), ("p", "matched_position", 1): 0.5}


def test_paired_scores_report_only_the_count_below_three_shared_windows():
    a = {("p", "tail", 1): 0.9, ("p", "tail", 2): 0.8}
    assert F.paired_scores(a, dict(a), 200, 0.95, 0) == {"n_windows": 2}


def test_paired_scores_pair_on_shared_windows_and_are_zero_for_identical_arms():
    a = {("p", "tail", i): 0.5 + 0.1 * i for i in range(4)}
    b = {**a, ("q", "tail", 0): 0.0}           # unshared: ignored
    out = F.paired_scores(a, b, 200, 0.95, 0)
    assert out["n_windows"] == 4
    assert out["mean_abs_diff"] == 0.0
    assert out["diff"]["estimate"] == 0.0
    assert out["pearson_r"] == pytest.approx(1.0)
