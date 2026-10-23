"""`scripts/make_paper_figures_v2.py`: reads stored outputs, writes only figures.

The figures are regenerated from stored JSON. These tests point the script at
a temporary runs directory and a temporary output directory, so nothing here
reads or writes `runs/` or `paper/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_paper_figures_v2 as mpf  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    runs, out = tmp_path / "runs", tmp_path / "figs"
    runs.mkdir()
    monkeypatch.setattr(mpf, "RUNS", runs)
    monkeypatch.setattr(mpf, "OUT", out)
    monkeypatch.setattr(mpf, "ROOT", tmp_path)
    return runs, out


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_every_figure_is_skipped_when_its_input_is_missing(dirs):
    _, out = dirs
    for f in (mpf.fig_increments, mpf.fig_stopping, mpf.fig_truncation, mpf.fig_stage0,
              mpf.fig_selective, mpf.fig_replication):
        assert f() is None
    assert not out.exists()


def test_the_replication_figure_is_drawn_from_stored_values(dirs):
    runs, out = dirs
    gap = {"gap": 0.047, "ci95": [0.019, 0.075]}
    cells = {c: {"n_problems": 380, "gaps": {"nla_readout - probe": gap,
                                             "nla_readout - no_signal": gap}}
             for c in ("O48/registered", "O256/registered")}
    _write(runs / "rep1_extforced" / "analysis.json", {"cells": cells})
    inc = {"delta_auc": 0.07, "ci95": [0.05, 0.09]}
    _write(runs / "rep1_stage1" / "analysis.json", {
        "n_units": 1600, "n_problems": 400,
        "Y_ext": {"contrasts": {"E over pos": inc, "A over pos": inc,
                                "Δ1 E over pos+A": inc}}})
    assert mpf.fig_replication() is not None
    assert [p.name for p in out.iterdir()] == ["fig_v2_6_replication.png"]


def test_stage0_rule_names_cover_the_rules_stage0_writes():
    """A Stage 0 rule without a plain name would reach the axis as code."""
    import offmenu_stage0 as s0
    src = Path(s0.__file__).read_text(encoding="utf-8")
    for name in ("learned pos", "learned pos+A", "registered probe",
                 "position_tokens_falling", "position_tokens"):
        assert f'"{name}' in src
        assert name in mpf.STAGE0_NAMES
