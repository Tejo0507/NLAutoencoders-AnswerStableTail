"""D83 Stage 1's position draw: the leak barrier (design §3.2)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402

from stage1_offmenu import K, select_positions, tost_reading  # noqa: E402


@pytest.mark.parametrize("ci,word", [
    ([0.012, 0.03], "practically relevant increment"),
    ([0.002, 0.008], "negligible increment"),
    ([-0.008, -0.002], "slightly worse"),
    ([-0.004, 0.003], "equivalent"),
    ([-0.03, -0.012], "probe practically better"),
    ([-0.012, -0.002], "inconclusive"),
    ([-0.002, 0.015], "inconclusive"),
])
def test_tost_reading_separates_statistical_from_practical_difference(ci, word):
    assert word in tost_reading(ci)
    assert tost_reading(None) == "not evaluable"


def test_the_draw_sees_only_the_problem_id_and_the_number_of_chunks():
    params = list(inspect.signature(select_positions).parameters)
    assert params == ["problem_id", "n_chunks", "k", "seed"]


def test_draw_is_deterministic_distinct_sorted_and_in_range():
    a = select_positions("gsm8k-test-00001", 13)
    assert a == select_positions("gsm8k-test-00001", 13)
    assert len(a) == K == len(set(a)) and a == sorted(a)
    assert all(0 <= i < 13 for i in a)


def test_short_traces_take_every_boundary_and_problems_differ():
    assert select_positions("p", 3) == [0, 1, 2]
    draws = {tuple(select_positions(f"p{j}", 20)) for j in range(50)}
    assert len(draws) > 40


def test_positions_are_roughly_uniform_over_chunks():
    import numpy as np
    counts = np.zeros(10)
    for j in range(4000):
        for i in select_positions(f"q{j}", 10):
            counts[i] += 1
    share = counts / counts.sum()
    assert np.all(np.abs(share - 0.1) < 0.015)
