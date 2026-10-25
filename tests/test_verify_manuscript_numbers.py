"""`scripts/verify_manuscript_numbers.py`: the manuscript's numbers against their sources.

Every test builds its own runs directory, manuscript files and registry in a
temporary directory; nothing here reads `runs/` or `paper/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_manuscript_numbers as vmn  # noqa: E402


def _setup(tmp_path, results_text="the gap was +0.047 [0.019, 0.075] and 4 tokens",
           gap=0.047076, abstract="by 0.047 (95% CI 0.019 to 0.075)"):
    runs = tmp_path / "runs"
    (runs / "r").mkdir(parents=True)
    (runs / "r" / "out.json").write_text(json.dumps(
        {"h": {"gap": gap, "ci95": [0.018998, 0.075438]}, "diff": 4, "rate": 0.8667,
         "sel": {"0.5": {"risk": 0.0105}}}), encoding="utf-8")
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "04_results.md").write_text(results_text + "\nrisk 0.011 and 86.7%", encoding="utf-8")
    (v2 / "00_abstract.md").write_text(abstract, encoding="utf-8")
    reg = {"facts": [
        {"id": "N1", "root": "runs", "file": "r/out.json", "value": ["h", "gap"], "ci": ["h", "ci95"],
         "fmt": "signed3", "written": "+0.047 [0.019, 0.075]", "files": ["04_results.md"],
         "variants": [{"text": "0.047 (95% CI 0.019 to 0.075)", "files": ["00_abstract.md"]}]},
        {"id": "N2", "root": "runs", "file": "r/out.json", "value": ["diff"], "fmt": "int",
         "written": "4 tokens", "files": ["04_results.md"]},
        {"id": "N3", "root": "runs", "file": "r/out.json", "value": ["sel", "0.5", "risk"],
         "fmt": "plain3", "written": "risk 0.011", "files": ["04_results.md"]},
        {"id": "N4", "root": "runs", "file": "r/out.json", "value": ["rate"], "fmt": "pct1",
         "written": "86.7%", "files": ["04_results.md"]},
    ]}
    return reg, {"runs": runs}, v2


def test_formats_follow_the_record():
    assert vmn.fmt_value(-0.0082, "signed3") == "−0.008"
    assert vmn.fmt_value(2.3e-05, "signed5") == "+0.00002"
    assert vmn.fmt_value(0.5645, "pct0") == "56%"
    assert vmn.fmt_value(1598, "intc") == "1,598"
    assert vmn.fmt_ci(-0.036, 0.024, "signed3") == "[−0.036, 0.024]"


def test_everything_agrees(tmp_path):
    reg, roots, v2 = _setup(tmp_path)
    failures, checked = vmn.check(reg, roots, v2)
    assert failures == [] and checked == 4


def test_a_changed_source_value_fails(tmp_path):
    reg, roots, v2 = _setup(tmp_path, gap=0.0481)
    failures, _ = vmn.check(reg, roots, v2)
    assert any(f.startswith("N1: source gives '+0.048") for f in failures)


def test_a_number_missing_from_the_text_fails(tmp_path):
    reg, roots, v2 = _setup(tmp_path, results_text="the gap was +0.047 [0.019, 0.076] and 4 tokens")
    failures, _ = vmn.check(reg, roots, v2)
    assert any("not found in 04_results.md" in f for f in failures)


def test_a_number_inside_a_longer_number_does_not_count(tmp_path):
    reg, roots, v2 = _setup(tmp_path, results_text="+0.047 [0.019, 0.075] and 14 tokens")
    failures, _ = vmn.check(reg, roots, v2)
    assert any(f.startswith("N2:") for f in failures)


def test_a_variant_with_other_decimals_fails(tmp_path):
    reg, roots, v2 = _setup(tmp_path, abstract="by 0.047 (95% CI 0.019 to 0.075)")
    reg["facts"][0]["variants"][0]["text"] = "0.047 (95% CI 0.019 to 0.074)"
    failures, _ = vmn.check(reg, roots, v2)
    assert any("disagrees with" in f for f in failures)


def test_negated_gaps_swap_the_interval(tmp_path):
    reg, roots, v2 = _setup(tmp_path)
    fact = {"root": "runs", "file": "r/out.json", "value": ["h", "gap"], "ci": ["h", "ci95"],
            "fmt": "plain3", "neg": True}
    assert vmn.expected(fact, roots["runs"]) == "−0.047 [−0.075, −0.019]"


def test_a_missing_source_is_a_failure_not_a_pass(tmp_path):
    reg, _, v2 = _setup(tmp_path)
    failures, checked = vmn.check(reg, {"runs": tmp_path / "nowhere"}, v2)
    assert checked == 0 and len(failures) == 4


def test_facts_under_the_pilot_root_are_read_from_it(tmp_path):
    reg, roots, v2 = _setup(tmp_path)
    for fact in reg["facts"]:
        fact["root"] = "pilot"
    failures, checked = vmn.check(reg, {"pilot": roots["runs"]}, v2)
    assert failures == [] and checked == 4
    failures, checked = vmn.check(reg, roots, v2)          # no --pilot given
    assert checked == 0 and all("source pilot:" in f for f in failures)


def test_main_passes_every_given_root(tmp_path, monkeypatch):
    reg, _, _ = _setup(tmp_path)
    path = tmp_path / "reg.json"
    path.write_text(json.dumps(reg), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(vmn, "check", lambda registry, roots: (seen.update(roots) or ([], 1)))
    assert vmn.main(["--runs", "a", "--d76", "b", "--pilot", "c", "--registry", str(path)]) == 0
    assert seen == {"runs": Path("a"), "d76": Path("b"), "pilot": Path("c")}
