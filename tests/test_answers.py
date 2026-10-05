"""Answer extraction and equivalence.

Every number in this study is downstream of these functions, so the cases below
are drawn from real GSM8K and MATH formats and from real model output observed
during the pilot, not from invented strings.
"""

import pytest

from nlaast.data.answers import (
    PERMISSIVE,
    STRICT,
    cluster_answers,
    equivalent,
    extract_answer,
    extract_gold,
    find_boxed,
)


class TestFindBoxed:
    def test_simple(self):
        assert find_boxed(r"the answer is $\boxed{42}$") == "42"

    def test_nested_braces(self):
        # A non-greedy regex truncates this to "\frac{1" - the reason the
        # extractor is brace-balanced rather than regex-based.
        assert find_boxed(r"so $\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"

    def test_last_wins(self):
        assert find_boxed(r"first $\boxed{1}$ then $\boxed{7}$") == "7"

    def test_unbalanced_returns_none(self):
        assert find_boxed(r"truncated \boxed{\frac{1") is None

    def test_absent(self):
        assert find_boxed("no box here") is None

    def test_text_wrapper_stripped(self):
        assert find_boxed(r"$\boxed{\text{blue}}$") == "blue"


class TestExtractGold:
    def test_gsm8k(self):
        rec = {"answer": "She makes 9 * 2 = <<9*2=18>>18 every day.\n#### 18"}
        assert extract_gold(rec, "gsm8k") == "18"

    def test_gsm8k_with_commas(self):
        assert extract_gold({"answer": "#### 1,234"}, "gsm8k") == "1234"

    def test_math(self):
        rec = {"solution": r"Therefore the graph has $\boxed{2}$ asymptotes."}
        assert extract_gold(rec, "math") == "2"

    def test_unknown_dataset(self):
        with pytest.raises(ValueError):
            extract_gold({}, "squad")


class TestExtractAnswer:
    def test_answer_is_phrasing(self):
        # Exactly what the system prompt asks the model to produce.
        assert extract_answer("...\n\nThe answer is 18.") == "18"

    def test_answer_is_with_dollar(self):
        # Observed verbatim in pilot output: "The answer is $18."
        assert extract_answer("The answer is $18.") == "18"

    def test_boxed_takes_precedence(self):
        assert extract_answer(r"The answer is 5. Actually $\boxed{7}$") == "7"

    def test_permissive_falls_back_to_last_number(self):
        assert extract_answer("so we get 36 - 30 = 6", PERMISSIVE) == "6"

    def test_strict_refuses_bare_number(self):
        assert extract_answer("so we get 36 - 30 = 6", STRICT) is None

    def test_strict_accepts_explicit(self):
        assert extract_answer("The answer is 6.", STRICT) == "6"

    def test_empty(self):
        assert extract_answer("") is None
        assert extract_answer("no digits at all", PERMISSIVE) is None


class TestEquivalence:
    @pytest.mark.parametrize("a,b", [
        ("18", "18"),
        ("18", "18.0"),
        ("1,234", "1234"),
        ("0.5", r"\frac{1}{2}"),
        ("6", "$6"),
        ("-3", "-3.00"),
    ])
    def test_equivalent(self, a, b):
        assert equivalent(a, b)

    @pytest.mark.parametrize("a,b", [("18", "19"), ("0.5", "0.6"), ("2", "-2")])
    def test_not_equivalent(self, a, b):
        assert not equivalent(a, b)

    def test_none_never_equivalent(self):
        # An unparseable intermediate answer is not evidence that the answer
        # has settled, so None must not match anything - including None.
        assert not equivalent(None, "5")
        assert not equivalent("5", None)
        assert not equivalent(None, None)


class TestClustering:
    def test_equivalent_forms_collapse(self):
        clusters = cluster_answers(["0.5", r"\frac{1}{2}", "1/2", "7"])
        assert len(clusters) == 2
        assert max(len(c.members) for c in clusters) == 3

    def test_unparsed_are_singletons(self):
        # Dropping them would make a problem the model mostly fails on look
        # confident, inverting the entropy signal where it matters most.
        clusters = cluster_answers([None, None, "5"])
        assert len(clusters) == 3

    def test_empty(self):
        assert cluster_answers([]) == []
