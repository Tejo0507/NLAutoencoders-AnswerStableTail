"""The measurement behind "RQ3 is unevaluable on this model".

D41 concludes that the obstacle to RQ3 is the model's prose rather than the
corpus size. That is an empirical claim about a base rate - how often an
unintervened trace contains any of the registered rechecking markers - and it
has to be measured from the traces rather than asserted, because it decides
whether a pre-registered research question can be answered at all.

`scripts/measure_rq3_readout_floor.py` measures it. These tests pin that it
counts what it says it counts, including the cases that would flatter the
readout: a marker inside a longer word, several markers in one trace, and
traces that failed and must not enter the denominator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from measure_rq3_readout_floor import measure  # noqa: E402


def write_traces(tmp_path: Path, traces) -> Path:
    path = tmp_path / "traces" / "traces.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(t) for t in traces) + "\n",
                    encoding="utf-8")
    return path


def trace(text, ok=True):
    return {"problem_id": "p", "ok": ok, "trace": text}


class TestTheRateIsMeasuredNotAsserted:
    def test_a_clean_corpus_has_no_marker_bearing_traces(self, tmp_path):
        path = write_traces(tmp_path, [
            trace("Step one: 2 + 2 = 4. The answer is 4."),
            trace("First, 3 x 3 = 9. Therefore the answer is 9."),
        ])
        bearing, n, occurrences, per_marker = measure(path)
        assert (bearing, n, occurrences) == (0, 2, 0)
        assert per_marker == {}

    def test_one_marker_in_one_trace_is_one_bearing_trace(self, tmp_path):
        path = write_traces(tmp_path, [
            trace("2 + 2 = 4, confirming the total is 4."),
            trace("3 x 3 = 9. The answer is 9."),
        ])
        bearing, n, occurrences, per_marker = measure(path)
        assert (bearing, n, occurrences) == (1, 2, 1)
        assert per_marker == {"confirming": 1}

    def test_several_markers_in_one_trace_is_still_one_bearing_trace(self, tmp_path):
        """The projection is against MIN_PAIRS, which counts problems."""
        path = write_traces(tmp_path, [
            trace("wait, let me check that. Hmm, actually 2 + 2 = 4."),
        ])
        bearing, n, occurrences, _ = measure(path)
        assert bearing == 1
        assert n == 1
        assert occurrences > 1

    def test_failed_traces_do_not_enter_the_denominator(self, tmp_path):
        """A trace that did not complete has no prose to carry a marker, so
        counting it would understate the rate."""
        path = write_traces(tmp_path, [
            trace("confirming the answer is 4."),
            trace("", ok=False),
            trace("", ok=False),
        ])
        bearing, n, _, _ = measure(path)
        assert (bearing, n) == (1, 1)

    def test_a_missing_file_measures_nothing_rather_than_raising(self, tmp_path):
        assert measure(tmp_path / "absent.jsonl") == (0, 0, 0, {})

    def test_an_empty_trace_field_is_not_a_marker(self, tmp_path):
        path = write_traces(tmp_path, [{"problem_id": "p", "ok": True}])
        bearing, n, _, _ = measure(path)
        assert (bearing, n) == (0, 1)


class TestTheCountIsTheStagesOwn:
    def test_it_uses_the_registered_marker_list(self, tmp_path):
        """Not a list of its own: the rate has to be the one the causal stage
        would compute, or it is measuring a different instrument."""
        from nlaast.causal.interventions import VERIFICATION_MARKERS
        path = write_traces(tmp_path, [trace(m) for m in VERIFICATION_MARKERS])
        bearing, n, _, per_marker = measure(path)
        assert n == len(VERIFICATION_MARKERS)
        assert bearing == len(VERIFICATION_MARKERS)
        assert set(per_marker) >= {"wait", "hmm", "recheck"}

    def test_a_substring_match_counts_as_the_stage_counts_it(self, tmp_path):
        """`count_verification_markers` is a substring count, so "waiting"
        contains "wait". That is the instrument's behaviour and this script
        must not quietly improve on it - a rate measured under a stricter rule
        would not be the rate the stage's comparison would see.
        """
        path = write_traces(tmp_path, [trace("waiting for the bus")])
        bearing, _, occurrences, per_marker = measure(path)
        assert bearing == 1
        assert occurrences == 1
        assert per_marker == {"wait": 1}
