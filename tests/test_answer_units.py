"""Unit suffixes on numeric answers.

Observed live during the pilot: the model answered "The answer is 180 minutes."
against a gold of "180". Scoring that as a mismatch would understate accuracy
and manufacture spurious answer instability at chunk boundaries, since the
same value would look like two different answers.

The stripping is deliberately conservative - it must not touch expressions
where the trailing token changes the value.
"""

import pytest

from nlaast.data.answers import _strip_units, equivalent, extract_answer


class TestUnitsAreStripped:
    @pytest.mark.parametrize("text,expected", [
        ("The answer is 180 minutes.", "180"),
        ("The answer is 6 dollars.", "6"),
        ("The answer is 9 eggs per day.", "9"),
        ("The answer is 42 apples.", "42"),
        ("The answer is 1,250 miles.", "1250"),
        ("The answer is 3.5 hours.", "3.5"),
        ("The answer is -7 degrees.", "-7"),
    ])
    def test_extracted_value_matches_a_bare_gold(self, text, expected):
        got = extract_answer(text)
        assert got == expected
        assert equivalent(got, expected)

    def test_boxed_with_units(self):
        assert extract_answer(r"so $\boxed{18 \text{ dollars}}$") == "18"


class TestMathematicalSuffixesAreKept:
    """A unit does not change the value; an operator or symbol does."""

    @pytest.mark.parametrize("value", [
        "3 pi", "2 squared", "5 x", "4 cubed", "2 times", "10 percent",
    ])
    def test_preserved(self, value):
        assert _strip_units(value) == value

    def test_expressions_untouched(self):
        for v in (r"\frac{1}{2}", "2x + 1", "x = 3", "(1,2)", "0.5"):
            assert _strip_units(v) == v

    def test_plain_numbers_untouched(self):
        for v in ("42", "-7", "3.14", "1250"):
            assert _strip_units(v) == v


class TestRegression:
    def test_the_pilot_case(self):
        # gsm8k-test-01218, verbatim.
        assert equivalent(extract_answer("The answer is 180 minutes."), "180")

    def test_long_prose_still_falls_back_to_the_number(self):
        text = ("The answer is obtained by first computing the total of 36 and "
                "then subtracting 30")
        assert extract_answer(text) == "30"
