"""`scripts/make_paper_tables_v2.py`: copies stored numbers into Tables 2-3.

These tests use a temporary runs directory and output file; nothing here
reads or writes `runs/` or `paper/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_paper_tables_v2 as mpt  # noqa: E402


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_numbers_are_written_as_the_record_writes_them():
    assert mpt.num(-0.0082) == "−0.008"
    assert mpt.num(0.047) == "+0.047"
    assert mpt.ci([-0.036, 0.024]) == "[−0.036, 0.024]"


def test_missing_inputs_skip_the_table(tmp_path):
    assert mpt.table2(tmp_path) is None
    assert mpt.table3(tmp_path) is None


def test_table3_copies_the_verdict_verbatim(tmp_path):
    h = {"gap": 0.047076, "ci95": [0.018998, 0.075438], "p": 0.002, "p_holm": 0.002,
         "reading": "supported"}
    _w(tmp_path / "rep1_o3_registered" / "o3_registered.json", {
        "confirmatory": {"H1_readout_minus_probe": h, "H2_readout_minus_no_signal": h},
        "design_A_shared_menu": {"n_problems": 380},
        "instrument": {"reproducibility_agreement": 0.965}})
    sens = {"H1_readout_minus_probe": {**h, "reading": "equivalent within ±0.05"},
            "H2_readout_minus_no_signal": h}
    _w(tmp_path / "rep1_extforced" / "analysis.json", {
        "cut_off_meaning": {"f_256_equals_final": 0.8667, "reading": "commitment"},
        "cells": {c: {"d75_sensitivity": sens}
                  for c in ("O48/registered", "O256/registered", "O256-u/registered",
                            "Ocont/registered")}})
    g = {"gap": 0.1, "ci95": [0.0, 0.2]}
    keys_x = ("readout - position_tokens", "position_tokens - no_signal",
              "probe - position_tokens", "readout_residualised - no_signal")
    keys_y = ("readout - position_tokens_falling", "position_tokens_falling - no_signal")
    _w(tmp_path / "rep1_position_audit" / "position_truncation.json", {"registered_outcome": {
        "gaps": {k: g for k in keys_x + keys_y},
        "holm_X": {k: {"p_holm": 0.5} for k in keys_x},
        "holm_Y": {k: {"p_holm": 0.5} for k in keys_y},
        "position_reading": {"overall": "unresolved"},
        "falling_reading": {"overall": "unresolved"}}})
    _w(tmp_path / "rep1_stage1" / "analysis.json", {
        "n_units": 1600, "n_problems": 400,
        "primary": {"delta": 2.3e-05, "ci95": [-0.0041, 0.0041],
                    "reading": "equivalent within ±0.01"}})
    t = mpt.table3(tmp_path)
    assert "| H1: readout − registered probe | +0.047 | [0.019, 0.075] | 0.002 | 0.002 | **supported** |" in t
    assert "equivalent within ±0.05" in t and "equivalent within ±0.01" in t
    assert "not about faithfulness" in t
