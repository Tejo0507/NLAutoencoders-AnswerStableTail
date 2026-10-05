"""Claim splitting, perturbation and the audit.

The audit is the part of the study most able to produce a flattering number by
accident, so the tests pin the two things that stop it: the paraphrase null
sets the deletion threshold, and the verbaliser-only control is applied to
every claim.
"""

import numpy as np
import pytest

from nlaast.config import FaithfulnessConfig
from nlaast.faithfulness.audit import (
    audit_explanation,
    claim_similarity,
    reproduced_in,
    summarise,
)
from nlaast.faithfulness.claims import (
    delete_claim,
    keep_only,
    paraphrase_claim,
    paraphrase_explanation,
    split_claims,
    substitute_claim,
)

#: The shape the verbaliser is prompted to produce: "2-3 text snippets".
EXPLANATION = (
    "The vector relates to arithmetic word problems involving money. "
    "It activates on discussion of comparing two purchase options. "
    "There is a focus on computing a difference between totals."
)

BULLETS = (
    "- The vector relates to arithmetic word problems.\n"
    "- It activates on comparisons between options.\n"
    "- There is a focus on computing differences."
)


class TestSplitting:
    def test_sentences(self):
        claims = split_claims(EXPLANATION)
        assert len(claims) == 3
        assert claims[0].text.startswith("The vector relates")

    def test_bullets_take_precedence(self):
        claims = split_claims(BULLETS)
        assert len(claims) == 3
        assert not claims[0].text.startswith("-")

    def test_offsets_are_exact(self):
        for c in split_claims(EXPLANATION):
            assert EXPLANATION[c.char_start:c.char_end] == c.text

    def test_cap_respected(self):
        text = " ".join(f"This is claim number {i} of many here." for i in range(20))
        assert len(split_claims(text, max_claims=4)) == 4

    def test_empty(self):
        assert split_claims("") == []
        assert split_claims("   ") == []

    def test_fragments_dropped(self):
        assert split_claims("Hi. Ok.") == []


class TestPerturbations:
    def test_delete_removes_exactly_the_claim(self):
        claims = split_claims(EXPLANATION)
        out = delete_claim(EXPLANATION, claims[1])
        assert claims[1].text not in out
        assert claims[0].text in out and claims[2].text in out

    def test_keep_only(self):
        claims = split_claims(EXPLANATION)
        assert keep_only(EXPLANATION, claims[0]) == claims[0].text

    def test_substitute(self):
        claims = split_claims(EXPLANATION)
        out = substitute_claim(EXPLANATION, claims[0], "Something else entirely.")
        assert out.startswith("Something else entirely.")
        assert claims[0].text not in out

    def test_paraphrase_changes_surface_and_is_deterministic(self):
        claims = split_claims(EXPLANATION)
        a = paraphrase_explanation(EXPLANATION, claims[0], 0)
        assert a != EXPLANATION
        assert a == paraphrase_explanation(EXPLANATION, claims[0], 0)

    def test_paraphrase_variants_differ(self):
        text = "The vector relates to arithmetic, which is common."
        variants = {paraphrase_claim(text, v) for v in range(3)}
        assert len(variants) >= 2

    def test_paraphrase_preserves_content_words(self):
        # The null has to be a surface edit; if it changed meaning it would
        # not be a null for the deletion threshold.
        text = "The vector relates to arithmetic word problems."
        out = paraphrase_claim(text, 0)
        for w in ("arithmetic", "word", "problems"):
            assert w in out.lower()


class TestNoiseControl:
    def test_similarity_bounds(self):
        assert claim_similarity("alpha beta gamma", "alpha beta gamma") == 1.0
        assert claim_similarity("alpha beta", "delta epsilon") == 0.0

    def test_stopwords_ignored(self):
        assert claim_similarity("the vector is about arithmetic",
                                "arithmetic") == pytest.approx(1.0)

    def test_reproduced_in_detects_a_match(self):
        hit, sim = reproduced_in(
            "The vector relates to arithmetic word problems.",
            ["It concerns arithmetic word problems in general. And more text here."],
        )
        assert hit and sim >= 0.5

    def test_reproduced_in_rejects_unrelated(self):
        hit, _ = reproduced_in(
            "The vector relates to arithmetic word problems.",
            ["This concerns maritime navigation and shipping lanes entirely."],
        )
        assert not hit


class TestAudit:
    """A scorer stub stands in for the reconstructor, so the audit's logic is
    tested without a 7B model."""

    def _scorer(self, keyword: str | None, base=0.80, drop=0.30):
        """Reconstruction falls only when a content word disappears.

        Keying on a content word rather than an exact sentence matters: a
        meaning-preserving paraphrase keeps the word, so the paraphrase null
        stays near zero and the deletion effect stands out against it. A stub
        that matched the sentence verbatim would penalise paraphrase exactly
        as hard as deletion and the null would swallow every real effect -
        which is what the first version of this test did.
        """
        def score(text, _activation):
            if keyword is not None and keyword not in text.lower():
                return base - drop
            return base
        return score

    def test_detects_a_load_bearing_claim(self):
        # "purchase" appears only in claim 1 and survives paraphrasing.
        audit = audit_explanation(
            "p1", "tail", 0, EXPLANATION, np.zeros(8),
            self._scorer("purchase"), FaithfulnessConfig(),
        )
        by_index = {c.claim_index: c for c in audit.claims}
        assert by_index[1].delete_effect == pytest.approx(0.30)
        assert by_index[1].reconstruction_dependent
        # The other claims carry no weight under this scorer.
        assert not by_index[0].reconstruction_dependent
        assert not by_index[2].reconstruction_dependent

    def test_paraphrase_null_prevents_false_positives(self):
        # A scorer that moves on *any* edit must not mark claims dependent:
        # the paraphrase null absorbs it.
        def jumpy(text, _a):
            return 0.8 if text == EXPLANATION else 0.5

        audit = audit_explanation(
            "p1", "tail", 0, EXPLANATION, np.zeros(8), jumpy, FaithfulnessConfig(),
        )
        assert not any(c.reconstruction_dependent for c in audit.claims)

    def test_noise_control_flags_prior_driven_claims(self):
        audit = audit_explanation(
            "p1", "tail", 0, EXPLANATION, np.zeros(8),
            self._scorer(None), FaithfulnessConfig(),
            noise_explanations=[EXPLANATION],
        )
        assert audit.noise_reproduced_fraction == pytest.approx(1.0)

    def test_resample_arm_uses_the_alternative(self):
        alt = ("The vector concerns geometry puzzles. "
               "It activates on angle computations. "
               "There is a focus on triangles and circles.")
        audit = audit_explanation(
            "p1", "tail", 0, EXPLANATION, np.zeros(8),
            self._scorer(None), FaithfulnessConfig(),
            resample_alternatives=[alt],
        )
        assert all(c.resample_effect is not None for c in audit.claims)

    def test_no_claims_is_recorded_not_crashed(self):
        audit = audit_explanation(
            "p1", "tail", 0, "", np.zeros(8), self._scorer(None),
            FaithfulnessConfig(),
        )
        assert audit.n_claims == 0 and "skipped" in audit.notes

    def test_summarise(self):
        audits = [
            audit_explanation("p1", "tail", 0, EXPLANATION, np.zeros(8),
                              self._scorer("money"),
                              FaithfulnessConfig()),
            audit_explanation("p2", "matched_position", 0, EXPLANATION, np.zeros(8),
                              self._scorer(None), FaithfulnessConfig()),
        ]
        s = summarise(audits)
        assert s["n_explanations"] == 2
        assert s["n_claims"] == 6
        assert 0.0 <= s["dependent_fraction"] <= 1.0
        assert "dependent_and_not_noise_fraction" in s
