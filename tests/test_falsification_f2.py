"""F2's two registered stratifications.

The pre-registration asks F2 to stratify "by MATH level and by gold-answer
magnitude". Only the level half was implemented, so the battery reported F2 as
having run while covering one of the two strata it names. These tests pin the
magnitude half, which is descriptive: tertiles of the observed magnitudes, no
threshold chosen, and no contribution to the hypothesis-test family.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

# ``scripts/`` is not a package: each stage script imports ``_stage`` as a
# sibling top-level module, which only resolves with this directory on the path.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def f2():
    return importlib.import_module("run_falsification_tests")


CFG = types.SimpleNamespace(analysis=types.SimpleNamespace(confidence=0.95))


def _rows(n, dataset="gsm8k", level=None):
    return [{"problem_id": f"{dataset}-{i:03}", "dataset": dataset,
             "level": level, "tail_start": 2, "tail_fraction": 0.1 + 0.01 * i}
            for i in range(n)]


def _traces(rows, golds):
    return {r["problem_id"]: {"gold": g} for r, g in zip(rows, golds)}


class TestGoldMagnitude:
    def test_plain_integers_and_decimals(self, f2):
        assert f2.gold_magnitude("160") == 160.0
        assert f2.gold_magnitude("3.5") == 3.5

    def test_sign_is_dropped_because_magnitude_is_asked_for(self, f2):
        assert f2.gold_magnitude("-42") == 42.0

    def test_currency_percent_and_thousands_separators(self, f2):
        assert f2.gold_magnitude("$1,200") == 1200.0
        assert f2.gold_magnitude("35%") == 35.0

    def test_latex_and_list_golds_have_no_magnitude(self, f2):
        """MATH answers are often not numbers, and must not be coerced."""
        assert f2.gold_magnitude(r"\dfrac{9}{7}") is None
        assert f2.gold_magnitude("x+1") is None
        assert f2.gold_magnitude(None) is None
        assert f2.gold_magnitude("") is None

    def test_a_comma_separated_list_is_not_a_number(self, f2):
        """The bug this test was written for.

        Stripping commas unconditionally turned MATH's list answers into huge
        numbers - "4,6,14,15" parsed as 461415 - which would have filled the
        top magnitude tertile with answers that have no magnitude. Three of
        the pilot's 24 golds contain a comma, so this was live.
        """
        assert f2.gold_magnitude("4,6,14,15") is None
        assert f2.gold_magnitude("1,2") is None

    def test_latex_thousands_separators_are_normalised_not_discarded(self, f2):
        """MATH writes 1000000 as "1,\\!000,\\!000"; that is formatting."""
        assert f2.gold_magnitude(r"1,\!000,\!000") == 1000000.0
        assert f2.gold_magnitude(r"9,\!240") == 9240.0


class TestF2Magnitude:
    def test_the_stratification_is_reported_with_its_own_coverage(self, f2):
        """Non-numeric golds reduce coverage; that is stated, not hidden."""
        rows = _rows(6)
        tr = _traces(rows, ["1", "2", "3", r"\frac{1}{2}", "x+1", "9"])
        out = f2.f2_difficulty(CFG, tr, rows, None)
        mag = out["by_gold_magnitude"]
        assert mag["n_with_numeric_gold"] == 4
        assert mag["n_tailed"] == 6
        assert mag["coverage"] == pytest.approx(4 / 6)

    def test_tertiles_are_positional_so_no_cut_is_chosen(self, f2):
        rows = _rows(9)
        tr = _traces(rows, [str(v) for v in (1, 2, 3, 100, 200, 300, 9000, 9001, 9002)])
        bins = f2.f2_difficulty(CFG, tr, rows, None)["by_gold_magnitude"]["tertiles"]
        assert [bins[k]["n"] for k in ("low", "mid", "high")] == [3, 3, 3]
        assert bins["low"]["gold_magnitude_max"] == 3.0
        assert bins["high"]["gold_magnitude_min"] == 9000.0

    def test_too_few_numeric_golds_is_reported_not_guessed(self, f2):
        rows = _rows(4)
        tr = _traces(rows, ["1", r"\frac{1}{2}", "x", "y"])
        mag = f2.f2_difficulty(CFG, tr, rows, None)["by_gold_magnitude"]
        assert "tertiles" not in mag
        assert "numeric gold" in mag["reason"]

    def test_problems_without_a_tail_are_excluded_from_both_strata(self, f2):
        rows = _rows(6)
        rows[0]["tail_start"] = None
        rows[1]["tail_start"] = None
        tr = _traces(rows, ["1", "2", "3", "4", "5", "6"])
        out = f2.f2_difficulty(CFG, tr, rows, None)
        assert out["by_gold_magnitude"]["n_tailed"] == 4
        assert out["by_gold_magnitude"]["n_with_numeric_gold"] == 4

    def test_the_level_stratification_is_unchanged(self, f2):
        """The half that already worked must keep working."""
        rows = _rows(3, dataset="math", level="Level 5") + _rows(3, dataset="math",
                                                                 level="Level 2")
        for i, r in enumerate(rows):
            r["problem_id"] = f"math-{i:03}"
        tr = _traces(rows, ["1"] * 6)
        out = f2.f2_difficulty(CFG, tr, rows, None)
        assert set(out["by_level"]) == {"Level 5", "Level 2"}
        assert out["by_level"]["Level 5"]["n"] == 3

    def test_a_missing_trace_does_not_raise(self, f2):
        """`ast.jsonl` and `traces.jsonl` are separate files and can disagree."""
        rows = _rows(4)
        out = f2.f2_difficulty(CFG, {}, rows, None)
        assert out["by_gold_magnitude"]["n_with_numeric_gold"] == 0
        assert out["status"] == "ran"
