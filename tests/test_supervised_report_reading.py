"""The supervised report must not overstate its label-shuffle check.

On `main` the classification report called a training-split CV AUROC with
label-shuffle p = 0.0796 "separated" from its null - the cut-off was p < 0.10,
the null's maximum (0.911) exceeded the observed 0.747, and the training split
held 6 incorrect answers. The check is a leakage check on the CV score, not a
test of the held-out result, and the report now says which is which.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def wsr():
    return importlib.import_module("write_supervised_report")


def null(p):
    return {"p_value": p, "observed": 0.747, "null_mean": 0.509,
            "null_sd": 0.172, "null_max": 0.911, "n_permutations": 200}


def test_p_of_0_08_is_not_called_separated(wsr):
    text = wsr.shuffle_reading(null(0.0796), 101, minority=6)
    assert "Not separated from the null at α = 0.05" in text
    assert "separated from it" not in text


def test_the_reading_says_what_the_check_is_for(wsr):
    text = wsr.shuffle_reading(null(0.73), 97)
    assert "training-split" in text
    assert "neither corroborates nor contradicts the held-out estimate" in text


def test_the_minority_count_is_stated_when_given(wsr):
    assert "only 6 minority-class" in wsr.shuffle_reading(null(0.08), 101, minority=6)


def test_a_clear_separation_is_still_reported_as_one(wsr):
    text = wsr.shuffle_reading(null(0.005), 200)
    assert "separated from the null at α = 0.05" in text
    assert "Not separated" not in text


def test_the_boundary_is_alpha_itself(wsr):
    assert "Not separated" not in wsr.shuffle_reading(null(0.05), 200)
    assert "Not separated" in wsr.shuffle_reading(null(0.0501), 200)
