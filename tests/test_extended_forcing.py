"""The extended-budget forcing pass's selection and outcome logic (D82).

Synthetic fixtures with known answers; they test the implementation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extended_forcing_analysis import (  # noqa: E402
    agreement_by_position,
    cut_off_reading,
    net_saving,
    outcome_answers,
)
from force_extended import generated_part, select  # noqa: E402


def _reg(finished):
    return {"finished": finished, "text": "t", "answer": "1"}


def test_selection_takes_every_cut_off_menu_boundary_and_samples_finished_ones():
    menu = {"a": {0, 2}, "b": {1, 3}}
    reg = {("a", 0): _reg(False), ("a", 2): _reg(True), ("b", 1): _reg(True),
           ("b", 3): _reg(False), ("a", 1): _reg(False)}       # a|1 is off the menu
    cut, check = select(menu, reg, n_check=1, seed=0)
    assert cut == {"a": [0], "b": [3]}
    drawn = [(p, i) for p, v in check.items() for i in v]
    assert len(drawn) == 1 and drawn[0] in {("a", 2), ("b", 1)}
    assert select(menu, reg, 1, 0) == (cut, check)


def test_selection_refuses_a_menu_boundary_without_a_registered_completion():
    with pytest.raises(SystemExit):
        select({"a": {0}}, {}, 1, 0)
    with pytest.raises(SystemExit):
        select({"a": {0}}, {("a", 0): {"finished": None, "text": None, "answer": None}}, 1, 0)


def test_generated_part_strips_the_suffix_only_when_present():
    suf = "\n\nTherefore, the answer is"
    assert generated_part(suf + " 18.", suf) == " 18."
    assert generated_part(" 18.", suf) == " 18."
    assert generated_part(None, suf) == ""


def _row():
    return {"answers": ["5", "7", "9", "9"], "answer_known": [True, True, True, True]}


def test_o256_replaces_only_cut_off_menu_answers_and_o256u_drops_unfinished():
    menu = np.array([True, True, False, True])
    ext = {("p", 0): {"answer": "9", "finished": True},
           ("p", 1): {"answer": "8", "finished": False},
           ("p", 2): {"answer": "1", "finished": True}}           # off the menu: ignored
    a, k = outcome_answers(_row(), menu, ext, {}, {}, "O256", "p")
    assert a == ["9", "8", "9", "9"] and all(k)
    a, k = outcome_answers(_row(), menu, ext, {}, {}, "O256-u", "p")
    assert a == ["9", None, "9", "9"]
    a, k = outcome_answers(_row(), menu, ext, {}, {}, "O48", "p")
    assert a == _row()["answers"]


def test_net_saving_charges_the_forced_tokens_and_floors_at_zero():
    assert net_saving(100, 0, 400) == pytest.approx(0.75)
    assert net_saving(100, 48, 400) == pytest.approx(0.63)
    assert net_saving(100, 256, 400) == pytest.approx(0.11)
    assert net_saving(300, 256, 400) == 0.0          # finishing costs more than the trace saved


@pytest.mark.parametrize("f,reading", [(0.80, "commitment"), (0.95, "commitment"),
                                       (0.20, "supported"), (0.0, "supported"),
                                       (0.5, "mixed"), (0.2001, "mixed"), (0.7999, "mixed")])
def test_cut_off_reading_boundaries_are_the_frozen_ones(f, reading):
    assert reading in cut_off_reading(f)
    assert cut_off_reading(None) is None


def test_agreement_is_split_at_the_tail_start():
    from nlaast.data.answers import equivalent
    # stored flags deliberately wrong: agreement must come from the answers
    ext = {("p", 1): {"answer": "9", "finished": True, "matches_final": False},
           ("p", 2): {"answer": "4", "finished": False, "matches_final": True},
           ("p", 5): {"answer": "9", "finished": True, "matches_final": False},
           ("q", 0): {"answer": "1", "finished": True, "matches_final": False}}  # no tail
    out = agreement_by_position(ext, {"p": 3, "q": None}, {("p", 1): "9", ("p", 5): "8"},
                                {"p": "9", "q": "1"}, equivalent)
    assert out["before"]["n"] == 2 and out["before"]["equals_final"] == 0.5
    assert out["before"]["still_cut_off"] == 0.5
    assert out["before"]["equals_continuation_majority"] == 1.0
    assert out["at or after"]["n"] == 1 and out["at or after"]["equals_continuation_majority"] == 0.0


def test_ocont_uses_the_majority_and_marks_missing_menu_boundaries_unknown():
    menu = np.array([True, False, False, True])
    cont = {("p", 0): "9"}
    a, k = outcome_answers(_row(), menu, {}, {}, cont, "Ocont", "p")
    assert a[0] == "9" and k == [True, True, True, False]
