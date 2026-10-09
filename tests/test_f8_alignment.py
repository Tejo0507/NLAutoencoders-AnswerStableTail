"""F8 must compare the two precisions at the *same* tokens.

F8 is the test that decides whether the NLA arm's inputs are in distribution:
it computes layer-20 activations from the 4-bit target and from the bf16 one
and reports the cosine between them. The 4-bit arm read its activations at
absolute positions in `prompt_ids + sampled_trace_ids`, with the sampled ids
replayed verbatim because re-tokenising decoded text is not guaranteed to
reproduce them (DECISIONS.md D6).

The bf16 arm originally tokenised `prompt + trace` as one string. That reopens
exactly the hazard D6 closes - byte-level BPE can merge across the old
prompt/trace boundary - and the failure is invisible: the cosine simply comes
out lower, and a reader would attribute the gap to quantisation, which is the
one thing F8 exists to measure.

These pin the replacement: replay the stored ids, and refuse the comparison
rather than approximate it when the sequence cannot be reproduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_falsification_tests import f8_replay_sequence  # noqa: E402

PROMPT = [1, 2, 3, 4, 5]
TRACE = [10, 11, 12, 13]


def test_the_sampled_ids_are_replayed_verbatim_after_the_prompt():
    ids, reason = f8_replay_sequence(
        PROMPT, {"token_ids": TRACE}, {"prompt_tokens": len(PROMPT)}, 100)
    assert reason is None
    assert ids == PROMPT + TRACE


def test_a_prompt_that_retokenises_differently_is_refused():
    """The silent-shift case: right trace ids, wrong absolute offsets."""
    ids, reason = f8_replay_sequence(
        PROMPT, {"token_ids": TRACE}, {"prompt_tokens": len(PROMPT) + 1}, 100)
    assert ids is None
    assert "recorded at extraction" in reason


def test_a_trace_without_stored_ids_is_refused_not_retokenised():
    ids, reason = f8_replay_sequence(
        PROMPT, {"trace": "some text"}, {"prompt_tokens": len(PROMPT)}, 100)
    assert ids is None
    assert "token_ids" in reason


def test_an_over_long_sequence_is_refused_with_its_length():
    ids, reason = f8_replay_sequence(
        PROMPT, {"token_ids": TRACE}, {"prompt_tokens": len(PROMPT)}, 6)
    assert ids is None
    assert "exceeds 6" in reason


def test_a_missing_prompt_length_record_does_not_block_the_comparison():
    """Older activation blocks predate the field; replaying is still correct."""
    ids, reason = f8_replay_sequence(PROMPT, {"token_ids": TRACE}, {}, 100)
    assert reason is None
    assert ids == PROMPT + TRACE


def test_the_stored_positions_index_the_replayed_sequence():
    """What the alignment is for: position p must be inside the sequence, and
    the first chunk boundary must land after the prompt, not inside it."""
    meta = {"prompt_tokens": len(PROMPT),
            "resolved_positions": [len(PROMPT) + 1, len(PROMPT) + 3]}
    ids, reason = f8_replay_sequence(PROMPT, {"token_ids": TRACE}, meta, 100)
    assert reason is None
    for p in meta["resolved_positions"]:
        assert len(PROMPT) <= p < len(ids)
