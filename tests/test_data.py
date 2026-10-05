"""Benchmark loading, validation and splitting.

These run against the real staged corpus, so they also serve as a data
integrity check: a corrupted download or a bad re-encode shows up here rather
than as an unexplained accuracy drop eight hours into a run.
"""

import pytest

from nlaast import paths
from nlaast.config import DataConfig
from nlaast.data import datasets
from nlaast.data.answers import equivalent, extract_answer

pytestmark = pytest.mark.skipif(
    not (paths.DATA_RAW / "gsm8k" / "test.jsonl").exists(),
    reason="benchmarks not staged; run scripts/prepare_problems.py",
)

SMALL = DataConfig(n_gsm8k=20, n_math=10)


@pytest.fixture(scope="module")
def problems():
    return datasets.load_problems(SMALL)


class TestLoading:
    def test_counts(self, problems):
        assert len([p for p in problems if p.dataset == "gsm8k"]) == 20
        assert len([p for p in problems if p.dataset == "math"]) == 10

    def test_every_problem_has_a_gold_answer(self, problems):
        assert all(p.gold and p.gold.strip() for p in problems)

    def test_every_problem_has_a_question(self, problems):
        assert all(len(p.question.strip()) > 10 for p in problems)

    def test_math_carries_level_and_subject(self, problems):
        math = [p for p in problems if p.dataset == "math"]
        assert all(p.level for p in math)
        assert all(p.subject for p in math)

    def test_validation_passes(self, problems):
        assert datasets.validate(problems)["ok"]


class TestSplitting:
    def test_both_splits_are_populated(self, problems):
        splits = {p.split for p in problems}
        assert splits == {"train", "eval"}

    def test_split_is_deterministic(self):
        a = {p.id: p.split for p in datasets.load_problems(SMALL)}
        b = {p.id: p.split for p in datasets.load_problems(SMALL)}
        assert a == b

    def test_split_is_stable_as_the_set_grows(self):
        """A problem must keep its split when more problems are added, or a
        probe fitted during the pilot would be evaluated on its own training
        data in the main run."""
        small = {p.id: p.split for p in datasets.load_problems(SMALL)}
        big = {p.id: p.split
               for p in datasets.load_problems(DataConfig(n_gsm8k=60, n_math=40))}
        shared = set(small) & set(big)
        assert len(shared) >= 10
        assert all(small[i] == big[i] for i in shared)

    def test_subsample_is_not_a_head_slice(self):
        """MATH files are grouped by level, so taking the first N would select
        only easy problems."""
        math = [p for p in datasets.load_problems(DataConfig(n_gsm8k=0, n_math=40))
                if p.dataset == "math"]
        assert len({p.level for p in math}) >= 3


class TestIntegrity:
    def test_no_replacement_characters(self, problems):
        # U+FFFD means a bad decode somewhere in the staging path.
        for p in problems:
            assert "�" not in p.question, p.id
            assert "�" not in (p.gold_solution or ""), p.id

    def test_gold_is_recoverable_from_the_gold_solution(self, problems):
        """An upper bound on answer-extraction accuracy. If this is low, every
        downstream 'the answer changed' measurement inherits the error rate."""
        checked = [p for p in problems if p.gold_solution]
        hits = sum(1 for p in checked
                   if equivalent(extract_answer(p.gold_solution), p.gold))
        assert hits / len(checked) >= 0.95, f"{hits}/{len(checked)}"

    def test_ids_are_unique(self, problems):
        ids = [p.id for p in problems]
        assert len(ids) == len(set(ids))
