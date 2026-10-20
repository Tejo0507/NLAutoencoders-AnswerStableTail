"""The annotation tooling never scores blank labels and keeps problems on one side."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_claim_annotation_sheet as sheet  # noqa: E402


def _write(path, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "claim", *sheet.FIELDS])
        w.writeheader()
        for i, lab in enumerate(labels):
            w.writerow({"item": f"item{i:03d}", "claim": 0, **{k: lab for k in sheet.FIELDS}})


def test_scoring_refuses_blank_labels(tmp_path):
    _write(tmp_path / "a.csv", ["yes", ""])
    _write(tmp_path / "b.csv", ["yes", "no"])
    with pytest.raises(SystemExit):
        sheet.score(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"))


def test_scoring_reports_kappa_per_field(tmp_path):
    _write(tmp_path / "a.csv", ["yes", "no", "yes", "no"])
    _write(tmp_path / "b.csv", ["yes", "no", "yes", "no"])
    k = sheet.score(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"))
    assert set(k) == set(sheet.FIELDS) and all(v == 1.0 for v in k.values())


def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "claim", *sheet.FIELDS])
        w.writeheader()
        for item, claim, lab in rows:
            w.writerow({"item": item, "claim": claim, **{k: lab for k in sheet.FIELDS}})


def _key(kinds_parts):
    return {"items": {f"item{i:03d}": {"kind": k, "part": part, "problem_id": f"p{i}"}
                      for i, (k, part) in enumerate(kinds_parts)}}


def test_report_keeps_controls_out_of_kappa_and_shows_their_labels(tmp_path):
    """Raters who agree only on the controls: pooled kappa looks fine, the
    protocol's kappa on real claims does not."""
    kinds = [("real", "heldout")] * 6 + [("gaussian_control", "heldout")] * 6 + [("real", "dev")]
    a = ["yes", "no", "yes", "no", "yes", "no"] + ["no"] * 6 + ["yes"]
    b = ["no", "yes", "no", "yes", "no", "yes"] + ["no"] * 6 + ["yes"]
    _write_rows(tmp_path / "a.csv", [(f"item{i:03d}", 0, x) for i, x in enumerate(a)])
    _write_rows(tmp_path / "b.csv", [(f"item{i:03d}", 0, x) for i, x in enumerate(b)])
    r = sheet.report(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), _key(kinds))
    f = r["fields"]["P2_supported"]
    assert r["n_heldout_claims"] == 12 and r["n_heldout_real_claims"] == 6   # dev excluded
    assert f["kappa_heldout_real"]["kappa"] == pytest.approx(-1.0)
    assert f["kappa_pooled_with_controls"] > f["kappa_heldout_real"]["kappa"]
    assert not f["reliable"]
    assert f["label_shares_heldout"]["gaussian_control"]["rater_a"] == {"no": 1.0}


def test_report_calls_a_single_label_kappa_undefined_not_perfect(tmp_path):
    kinds = [("real", "heldout")] * 5
    _write_rows(tmp_path / "a.csv", [(f"item{i:03d}", 0, "none") for i in range(5)])
    _write_rows(tmp_path / "b.csv", [(f"item{i:03d}", 0, "none") for i in range(5)])
    r = sheet.report(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), _key(kinds))
    k = r["fields"]["P5_mechanism"]
    assert k["kappa_heldout_real"]["defined"] is False and k["kappa_heldout_real"]["kappa"] is None
    assert k["reliable"] is False
    assert k["kappa_pooled_with_controls"] == 1.0          # what the legacy score says


def test_report_refuses_dev_only_sheets_and_unknown_items(tmp_path):
    _write_rows(tmp_path / "a.csv", [("item000", 0, "yes")])
    _write_rows(tmp_path / "b.csv", [("item000", 0, "yes")])
    with pytest.raises(SystemExit):
        sheet.report(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), _key([("real", "dev")]))
    with pytest.raises(SystemExit):
        sheet.report(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), {"items": {}})


def test_split_keeps_every_problem_on_one_side():
    items = [{"problem_id": f"p{i // 3}", "kind": "real"} for i in range(90)]
    dev = sheet.split_by_problem(items, np.random.default_rng(0))
    n_dev = sum(1 for it in items if it["problem_id"] in dev)
    assert sheet.N_DEV <= n_dev < sheet.N_DEV + 3
    assert all(p.startswith("p") for p in dev)


def test_sample_includes_blind_controls():
    verbs = [{"problem_id": f"gsm8k-{i}", "window_kind": k,
              "samples": [{"ok": True, "explanation": "x"}] * 2,
              "noise_samples": [{"ok": True, "explanation": "n"}]}
             for i in range(40) for k in ("tail", "matched_position", "matched_length")]
    items = sheet.sample_items(verbs, np.random.default_rng(1))
    kinds = [k for k, _, _ in items]
    assert kinds.count("real") == sheet.N_REAL
    assert kinds.count("gaussian_control") == sheet.N_NOISE
    assert kinds.count("mismatched_trace") == sheet.N_MISMATCH
