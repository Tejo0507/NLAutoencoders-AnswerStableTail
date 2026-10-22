"""D70's classes for paraphrase-null draws, including on the real paraphraser."""

from __future__ import annotations

import pytest

from nlaast.faithfulness.claims import paraphrase_claim
from nlaast.faithfulness.paraphrase_qc import (
    CONTENT_CHANGED,
    IDENTICAL,
    LEXICAL_SWAP,
    NORMALISED_IDENTICAL,
    PREFIX_ONLY,
    REORDER,
    VALID_N1,
    VALID_N2,
    classify,
    token_change,
)

CLAIM = "The text describes a sum of three weekly totals."


@pytest.mark.parametrize("variant, expected", [
    (CLAIM, IDENTICAL),
    ("the text describes a sum of three weekly totals", NORMALISED_IDENTICAL),
    ("The  text describes a sum of three weekly totals!", NORMALISED_IDENTICAL),
    ("In other words, the text describes a sum of three weekly totals.", PREFIX_ONLY),
    ("The passage describes a sum of three weekly totals.", LEXICAL_SWAP),
    ("The text characterises a sum of three weekly totals.", LEXICAL_SWAP),
    ("A sum of three weekly totals, the text describes.", REORDER),
    # meaning-changing rewrites
    ("The text describes a sum of four weekly totals.", CONTENT_CHANGED),
    ("The text describes a sum of 3 weekly totals.", CONTENT_CHANGED),
    ("The text never describes a sum of three weekly totals.", CONTENT_CHANGED),
    ("The text describes a product of three weekly totals.", CONTENT_CHANGED),
    # two swaps at once are not one table pair
    ("The passage characterises a sum of three weekly totals.", CONTENT_CHANGED),
])
def test_classes(variant, expected):
    assert classify(CLAIM, variant) == expected


def test_validity_sets_exclude_non_perturbations_and_content_changes():
    for invalid in (IDENTICAL, NORMALISED_IDENTICAL, CONTENT_CHANGED):
        assert invalid not in VALID_N1 and invalid not in VALID_N2
    assert REORDER in VALID_N1 and REORDER not in VALID_N2


def test_the_real_paraphraser_returns_an_identity_when_no_synonym_matches():
    # The defect D68 measured: variant 0 changes nothing here.
    claim = "Final token closes a clause mid-sentence, expecting a number."
    assert paraphrase_claim(claim, 0) == claim
    assert classify(claim, paraphrase_claim(claim, 0)) == IDENTICAL
    assert classify(claim, paraphrase_claim(claim, 1)) == PREFIX_ONLY
    assert classify(claim, paraphrase_claim(claim, 2)) == REORDER


def test_the_real_paraphraser_swaps_when_a_synonym_matches():
    claim = "This text relates to a shopping problem"
    assert classify(claim, paraphrase_claim(claim, 0)) == LEXICAL_SWAP
    # no comma: variant 2 falls back to a second prefix
    assert classify(claim, paraphrase_claim(claim, 2)) == PREFIX_ONLY


def test_reordering_around_a_thousands_separator_is_flagged_as_unverified():
    # "1,000" is split at its comma: the content words survive, the meaning
    # does not. This is why REORDER is outside the surface-verified set N2.
    claim = "The total is 1,000 apples"
    v = paraphrase_claim(claim, 2)
    assert classify(claim, v) == REORDER and REORDER not in VALID_N2


def _llm_qc():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from paraphrase_null_llm import qc
    return qc


@pytest.mark.parametrize("variant, ok, reason", [
    ("The passage lays out a total formed by adding three weekly figures.", True, "accepted"),
    (CLAIM, False, "identical"),
    ("the text describes a sum of three weekly totals", False, "normalised_identical"),
    ("The passage describes a sum of three weekly totals.", False, "too close to the original"),
    ("Line one.\nLine two about three weekly totals.", False, "empty or multi-line"),
    ("It adds 3 weekly figures.", False, "numbers changed"),
    ("Totals.", False, "length ratio"),
])
def test_rewording_qc(variant, ok, reason):
    qc = _llm_qc()
    assert qc(CLAIM, variant) == (ok, reason)


def test_rewording_qc_keeps_numbers_and_quotes():
    qc = _llm_qc()
    orig = 'Final token "18." closes the sum of 9 and 9.'
    assert qc(orig, 'The last token, "18.", ends the addition of 9 plus 9.')[0]
    assert qc(orig, 'The last token, "18", ends the addition of 9 plus 9.')[1] == "quoted string changed"
    assert qc(orig, 'The last token, "18.", ends the addition of 9 plus 8.')[1] == "numbers changed"


def test_token_change_measures_diversity_not_meaning():
    assert token_change(CLAIM, CLAIM) == 0.0
    assert token_change(CLAIM, "The passage describes a sum of three weekly totals.") == pytest.approx(1 / 9)
    assert token_change("a b", "c d") == 1.0
