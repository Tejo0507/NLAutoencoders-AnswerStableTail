"""Answer extraction against strings observed in real pilot output.

Separate from ``test_answers.py`` so it is obvious which cases came from the
model and which were written to pin an intended behaviour. Every string here
was copied from a run.
"""

import pytest

from nlaast.data.answers import PERMISSIVE, STRICT, equivalent, extract_answer


class TestForcedAnswerEchoes:
    """The forced-answer prompt ends with "Therefore, the answer is", and the
    model frequently restates the whole phrase before answering."""

    def test_double_marker(self):
        text = "\n\nTherefore, the answer is The answer is 6."
        assert extract_answer(text) == "6"

    def test_triple_marker(self):
        text = "Therefore, the answer is the final answer is 42."
        assert extract_answer(text) == "42"

    def test_prose_after_marker_falls_back_to_number(self):
        # Observed at a pre-stabilisation boundary, where no answer exists yet.
        text = ("\n\nTherefore, the answer is calculated as follows: we first "
                "need the cost of 18 tickets")
        assert extract_answer(text, PERMISSIVE) == "18"

    def test_strict_rejects_prose(self):
        text = "\n\nTherefore, the answer is calculated as follows and we continue"
        assert extract_answer(text, STRICT) is None


class TestRealTraceEndings:
    @pytest.mark.parametrize("text,expected", [
        ("...\n\nTherefore, David saves $6.\n\nThe answer is 6.", "6"),
        ("So the total cost is 36 dollars.\n\nThe answer is $36.", "36"),
        (r"Hence $\boxed{\frac{1}{2}}$ is the value.", r"\frac{1}{2}"),
        (r"The answer is \(\frac{3}{4}\).", r"\frac{3}{4}"),
        ("The answer is -5.", "-5"),
        ("The answer is 1,250.", "1250"),
        ("The answer is 0.75.", "0.75"),
    ])
    def test_endings(self, text, expected):
        assert extract_answer(text) == expected

    def test_decimal_not_truncated(self):
        # Splitting the candidate on "." would turn 0.5 into 0.
        assert equivalent(extract_answer("The answer is 0.5."), "0.5")

    def test_latex_survives_stripping(self):
        # The first implementation stripped the leading backslash and the
        # trailing brace, producing "frac{1}{2".
        got = extract_answer(r"The answer is $\frac{1}{2}$.")
        assert got == r"\frac{1}{2}"
        assert equivalent(got, "0.5")


class TestIntermediateBoundaries:
    """Parsed answers at chunk boundaries mid-trace, where the permissive
    fallback is doing the work."""

    @pytest.mark.parametrize("text,expected", [
        ("- There are 9 rides.", "9"),
        (r"- Therefore, the total number of tickets needed is \(9 \times 2 = 18\) tickets.", "18"),
        ("- Each ticket costs $2.", "2"),
        ("To determine how much money David saves, we need to calculate", None),
    ])
    def test_boundary_parse(self, text, expected):
        assert extract_answer(text, PERMISSIVE) == expected
