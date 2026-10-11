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


class TestProcessingOrder:
    """Every stage resumes per problem, so the achieved N is whatever
    completed. That is only a usable sample if any prefix is representative -
    concatenating the datasets would put all GSM8K before all MATH, and a run
    halted partway would have no MATH at all.
    """

    def test_both_datasets_appear_early(self, problems):
        head = problems[: len(problems) // 4]
        assert {p.dataset for p in head} == {"gsm8k", "math"}

    def test_every_prefix_is_close_to_the_configured_proportion(self, problems):
        target = 20 / 30
        for k in range(6, len(problems) + 1):
            share = sum(1 for p in problems[:k] if p.dataset == "gsm8k") / k
            assert abs(share - target) < 0.2, f"prefix of {k} is {share:.2f} gsm8k"

    def test_the_order_is_deterministic(self):
        a = [p.id for p in datasets.load_problems(SMALL)]
        b = [p.id for p in datasets.load_problems(SMALL)]
        assert a == b

    def test_interleaving_preserves_the_whole_set(self):
        groups = [["a", "b", "c", "d", "e"], ["x", "y"]]
        out = datasets.interleave_by_dataset(groups)
        assert sorted(out) == sorted(groups[0] + groups[1])
        assert len(out) == 7

    def test_a_single_group_is_left_alone(self):
        assert datasets.interleave_by_dataset([["a", "b", "c"]]) == ["a", "b", "c"]
        assert datasets.interleave_by_dataset([]) == []

    def test_a_large_group_does_not_block_a_small_one(self):
        out = datasets.interleave_by_dataset([["g"] * 150, ["m"] * 100])
        # In any prefix of 10, both datasets are present.
        assert "m" in out[:10] and "g" in out[:10]


class TestStratifiedCap:
    """An arm that can only afford a subset must still sample this corpus.

    The NLA arm is capped by `nla.max_problems` and the primary O3 comparison
    is computed on the problems it covers. Capping by sorted problem id would
    take every `gsm8k-...` before any `math-...`, making that comparison a
    GSM8K-only result reported as a corpus-wide one.
    """

    @staticmethod
    def _rows(n_gsm8k, n_math):
        return ([{"problem_id": f"gsm8k-test-{i:05}", "dataset": "gsm8k"}
                 for i in range(n_gsm8k)]
                + [{"problem_id": f"math-algebra-test-{i:05}", "dataset": "math"}
                   for i in range(n_math)])

    def test_the_cap_keeps_both_datasets(self):
        out = datasets.stratified_cap(self._rows(130, 90), 80)
        counts = {d: sum(1 for r in out if r["dataset"] == d)
                  for d in ("gsm8k", "math")}
        assert len(out) == 80
        assert counts["gsm8k"] > 0 and counts["math"] > 0

    def test_the_cap_holds_the_corpus_proportions(self):
        out = datasets.stratified_cap(self._rows(150, 100), 80)
        share = sum(1 for r in out if r["dataset"] == "gsm8k") / len(out)
        assert abs(share - 0.6) < 0.02

    def test_sorting_by_id_would_have_dropped_a_dataset(self):
        """The behaviour being replaced, stated so the fix cannot regress."""
        rows = self._rows(130, 90)
        naive = sorted(rows, key=lambda r: r["problem_id"])[:80]
        assert {r["dataset"] for r in naive} == {"gsm8k"}
        fixed = datasets.stratified_cap(rows, 80)
        assert {r["dataset"] for r in fixed} == {"gsm8k", "math"}

    def test_it_is_a_deterministic_subset_under_extension(self):
        small = datasets.stratified_cap(self._rows(60, 40), 50)
        larger = datasets.stratified_cap(self._rows(150, 100), 50)
        assert ([r["problem_id"] for r in small]
                == [r["problem_id"] for r in larger])

    def test_no_cap_and_an_undersized_corpus_pass_through(self):
        rows = self._rows(5, 5)
        assert datasets.stratified_cap(rows, 0) == rows
        assert datasets.stratified_cap(rows, 50) == rows

    def test_a_stratum_smaller_than_its_share_does_not_overdraw(self):
        out = datasets.stratified_cap(self._rows(95, 5), 50)
        assert len(out) == 50
        assert sum(1 for r in out if r["dataset"] == "math") <= 5
