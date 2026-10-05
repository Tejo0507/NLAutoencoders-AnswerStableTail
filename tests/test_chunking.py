"""Trace segmentation and token alignment.

The regression at the top of this file is the real one: the first chunker
merged chunks on GSM8K because it treated ``$`` as a LaTeX delimiter, and these
traces write money as ``$2.``. Activation positions are derived from chunk
boundaries, so a merge here silently moves every activation read.
"""

from nlaast.config import ChunkConfig
from nlaast.trace.chunking import (
    _math_spans,
    chunk_trace,
    prefix_text,
    split_sentences,
)

CFG = ChunkConfig()

#: Verbatim from pilot output on gsm8k-test-00389.
REAL_TRACE = (
    "First, let's calculate the cost for Dasha if she buys individual tickets:\n"
    "- Each ride costs 2 tickets.\n"
    "- There are 9 rides.\n"
    "- Each ticket costs $2.\n"
    "- So, the total cost for Dasha is \\(18 \\times 2 = 36\\) dollars.\n\n"
    "Now, we compare the two costs:\n"
    "\\[ 36 - 30 = 6 \\]\n\n"
    "The answer is 6."
)


class TestMathSpans:
    def test_money_is_not_maths(self):
        # The bug: "$2." and "$30" are money, not an unterminated maths span.
        assert _math_spans("Each ticket costs $2. The bracelet is $30.") == []

    def test_latex_paren_is_maths(self):
        spans = _math_spans(r"we get \(9 \times 2 = 18\) tickets.")
        assert len(spans) == 1

    def test_display_maths(self):
        assert len(_math_spans(r"so \[ 36 - 30 = 6 \] follows")) == 1

    def test_dollar_pair_with_command(self):
        assert len(_math_spans(r"the value $\frac{1}{2}$ here")) == 1

    def test_dollar_pair_with_variable(self):
        assert len(_math_spans("solve $x + 1 = 2$ for x")) == 1


class TestSplitting:
    def test_money_does_not_suppress_boundaries(self):
        spans = split_sentences(REAL_TRACE, CFG)
        texts = [REAL_TRACE[s:e].strip() for s, e in spans]
        assert any(t.startswith("- Each ticket costs $2.") for t in texts), texts
        # Before the fix this collapsed to three chunks.
        assert len(spans) >= 6, texts

    def test_does_not_split_inside_latex(self):
        text = r"We compute \(1.5 \times 2 = 3.0\) exactly. Then we stop."
        spans = split_sentences(text, CFG)
        assert len(spans) == 2

    def test_decimal_not_split(self):
        # "3.14" must not end a sentence. Both halves are above the minimum
        # chunk length, so a correct splitter yields exactly two.
        text = "The measured value is 3.14 exactly here. Then we conclude the proof."
        spans = split_sentences(text, CFG)
        assert len(spans) == 2
        assert "3.14" in text[spans[0][0] : spans[0][1]]

    def test_abbreviation_not_split(self):
        text = "Please ask Dr. Smith about the result. Then we can leave the room."
        assert len(split_sentences(text, CFG)) == 2

    def test_blank_line_is_a_boundary(self):
        # Display maths often ends a step with no terminal punctuation, so the
        # paragraph break is the only available boundary.
        text = ("First we set up the equation\n\n"
                "\\[ 36 - 30 = 6 \\quad \\text{dollars} \\]\n\n"
                "and that finishes the derivation.")
        spans = split_sentences(text, CFG)
        assert len(spans) == 3, [text[s:e] for s, e in spans]

    def test_short_fragments_merge_backwards(self):
        text = "This is a full sentence here. Ok."
        assert len(split_sentences(text, CFG)) == 1

    def test_short_leading_fragment_merges_forwards(self):
        # Nothing precedes it, so it has to be absorbed by what follows.
        text = "Ok. This is a full sentence here."
        assert len(split_sentences(text, CFG)) == 1

    def test_max_chunks_respected(self):
        text = " ".join(f"This is sentence number {i} of many." for i in range(100))
        assert len(split_sentences(text, ChunkConfig(max_chunks=10))) <= 10

    def test_newline_mode(self):
        text = "alpha line goes here\nbeta line goes here\n\ngamma line goes here"
        assert len(split_sentences(text, CFG, mode="newline")) == 3

    def test_empty(self):
        assert split_sentences("", CFG) == []
        assert split_sentences("   \n  ", CFG) == []


class TestTokenAlignment:
    def test_offsets_map_to_tokens(self):
        text = "Alpha one two. Beta three four. Gamma five six."
        # One token per 4 characters, contiguous - a stand-in for a real
        # offset_mapping.
        offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
        chunks = chunk_trace(text, CFG, token_offsets=offsets)
        assert len(chunks) == 3
        # Monotone and within range.
        ends = [c.token_end for c in chunks]
        assert ends == sorted(ends)
        assert all(0 <= e < len(offsets) for e in ends)
        assert chunks[-1].token_end == len(offsets) - 1

    def test_boundary_token_is_inside_the_prefix(self):
        # A token straddling a boundary belongs to the prefix: the activation
        # read there has seen the whole sentence.
        text = "Alpha beta gamma. Delta epsilon zeta."
        #                        ^ char 16 is the period; token 2 spans 11-18
        #                          and so straddles the boundary.
        offsets = [(0, 6), (6, 11), (11, 18), (18, 26), (26, 37)]
        chunks = chunk_trace(text, CFG, token_offsets=offsets)
        assert len(chunks) == 2, [c.text for c in chunks]
        assert chunks[0].token_end == 2

    def test_prefix_tokens_accumulate(self):
        text = "Alpha one two. Beta three four. Gamma five six."
        offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
        chunks = chunk_trace(text, CFG, token_offsets=offsets)
        assert [c.prefix_tokens for c in chunks] == [c.token_end + 1 for c in chunks]

    def test_prefix_text_is_sliced_not_joined(self):
        # Re-joining would change whitespace and therefore the tokenisation.
        text = "Alpha one two.  Beta three four.   Gamma five six."
        chunks = chunk_trace(text, CFG)
        assert text.startswith(prefix_text(chunks, 1, text))
        assert prefix_text(chunks, len(chunks) - 1, text) == text

    def test_prefix_before_start(self):
        chunks = chunk_trace("Alpha one two. Beta three four.", CFG)
        assert prefix_text(chunks, -1, "whatever") == ""
