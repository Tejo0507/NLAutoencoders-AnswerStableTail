"""The objective claim checks (D71), on explanations in the verbaliser's own format."""

from __future__ import annotations

import pytest

from nlaast.faithfulness.claim_checks import (
    cohens_kappa,
    expectation_hit,
    final_token_match,
    longest_common_run,
    parse,
    quote_support,
)

EXPLANATION = (
    'Structured math problem with numbered steps.\n\n'
    'The sentence "Adding all the individual durations together" signals a '
    'concluding summation.\n\n'
    'Final token "time. " closes a clause, strongly expecting "Total the combined '
    'times" or "Sum all the final durations."'
)


def test_parse_separates_final_token_context_and_expectations():
    p = parse(EXPLANATION)
    assert p.final_token == "time. "
    assert p.context_quotes == ["Adding all the individual durations together"]
    assert p.expected == ["Total the combined times", "Sum all the final durations."]
    assert p.mechanistic is False


def test_mechanistic_language_is_flagged():
    assert parse('Final token "x" shows the model is verifying its sum.').mechanistic


def test_parse_without_a_final_token_claim():
    p = parse('The text "a b c d" is about apples.')
    assert p.final_token is None and p.context_quotes == ["a b c d"]


@pytest.mark.parametrize("quoted, context, expected", [
    ("time. ", "the total time.", (True, True, True)),
    ("Time.", "the total time.", (True, True, True)),          # case-folded
    ("period. ", "the total time.", (False, False, True)),     # right punctuation only
    ("time", "the total time.", (False, True, False)),
    ("18.", "the answer is 45.", (False, False, True)),
    ("45.", "the answer is 45.", (True, True, True)),
])
def test_final_token_levels(quoted, context, expected):
    m = final_token_match(quoted, context)
    assert (m["exact_suffix"], m["last_word"], m["last_char"]) == expected


def test_quote_support_is_contiguous_and_ignores_short_quotes():
    ctx = "First we add 3 and 4. Adding all the individual durations together gives 7."
    assert quote_support("Adding all the individual durations together", ctx) == 1.0
    # every word occurs, but no two consecutively: a reworded "quote"
    assert quote_support("adding the durations all together now", ctx) == pytest.approx(1 / 6)
    assert quote_support("we add 3 and 4 slowly", ctx) == pytest.approx(5 / 6)
    assert quote_support("add 3", ctx) is None
    assert longest_common_run(["a", "b", "c"], ["x", "b", "c", "a"]) == 2


def test_expectation_hit_reads_the_first_words():
    assert expectation_hit(["Total the times"], "\n\nTotal time = 10") == {
        "first_word": True, "any_word_in_first_10": True}
    assert expectation_hit(["Sum all"], " The total. Sum is 3") == {
        "first_word": False, "any_word_in_first_10": True}
    assert expectation_hit([], "x") is None


def test_kappa():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    from nlaast.faithfulness.claim_checks import kappa_summary
    s = kappa_summary(["a", "b", "a", "b"] * 5, ["a", "b", "a", "b"] * 5, list(range(20)), reps=200)
    assert s["kappa"] == 1.0 and s["ci95"] == [1.0, 1.0]
    s = kappa_summary(["a"] * 8, ["a"] * 8, list(range(8)), reps=50)
    assert s["kappa"] is None and s["ci95"] is None and s["n_replicates_undefined"] == 50
    # resampling is by cluster: all labels of one cluster move together
    s = kappa_summary(["a", "a", "b", "b"], ["a", "b", "a", "b"], [0, 0, 1, 1], reps=100)
    assert s["n_clusters"] == 2 and s["kappa"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        kappa_summary(["a"], ["a"], [])
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohens_kappa(["a"], [])
