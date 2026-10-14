"""What the NLA stopping arm drops, and whether the output says so.

The arm's score is the mean reconstruction cosine at a boundary, taken over the
samples that survive two exclusions: the Gaussian control arm, dropped by
design because it is the noise control, and the integrity gate, which drops a
sample whose verbalisation is not well-formed English. The score is therefore a
mean over a subset, and nothing in `analysis/results.json` said how large a
subset or which kind of exclusion made it so.

That matters because the two exclusions mean opposite things. On the pilot the
arm drops 132 of 378 samples, which looks alarming until it is split: 126 are
controls and 6 are gate failures - a 2.4% rate, matching the `nla` stage's own
0.976 integrity figure. A run where those proportions inverted would have the
same `n_problems` and a score meaning something quite different.

A problem can also have usable samples and still not be scored, if its trace
row is missing or has no chunks. Those were subtracted silently; they are now
named.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_results import nla_stopping_curve  # noqa: E402

LOG = logging.getLogger("test_nla_arm")

TRACE = "Step one. Step two. The answer is 7."
#: Character ends of the three sentences above.
ENDS = [10, 21, len(TRACE)]


class FakeConfig:
    """Only `dir` is read by `nla_stopping_curve`."""

    def __init__(self, path: Path):
        self.dir = path


def chunks():
    return [{"index": i, "text": TRACE[(ENDS[i - 1] if i else 0):ENDS[i]],
             "char_start": ENDS[i - 1] if i else 0, "char_end": ENDS[i],
             "token_end": 4 * (i + 1) - 1, "prefix_tokens": 4 * (i + 1)}
            for i in range(len(ENDS))]


def write_run(path: Path, recon_rows, problems=("p0", "p1")):
    (path / "nla").mkdir(parents=True, exist_ok=True)
    (path / "traces").mkdir(parents=True, exist_ok=True)
    (path / "nla" / "reconstructions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recon_rows) + "\n", encoding="utf-8")
    (path / "traces" / "traces.jsonl").write_text(
        "\n".join(json.dumps({
            "problem_id": pid, "ok": True, "trace": TRACE, "chunks": chunks(),
            "final_answer": "7", "gold": "7", "n_tokens": 12,
        }) for pid in problems) + "\n", encoding="utf-8")
    return FakeConfig(path)


def sample(pid, ci, kind="tail", *, cosine=0.8, integrity_ok=True):
    return {"problem_id": pid, "chunk_index": ci, "window_kind": kind,
            "cosine": cosine, "mse": 0.4, "integrity_ok": integrity_ok}


@pytest.fixture
def run(tmp_path):
    rows = []
    for pid in ("p0", "p1"):
        for ci in (1, 2):
            rows.append(sample(pid, ci, cosine=0.5 + 0.1 * ci))
            rows.append(sample(pid, ci, "gaussian_control", cosine=0.1))
    # One activation-driven sample fails the gate.
    rows.append(sample("p0", 2, cosine=0.95, integrity_ok=False))
    return write_run(tmp_path, rows)


class TestExclusionsAreCounted:
    def test_the_two_kinds_of_exclusion_are_reported_separately(self, run):
        _, info, _ = nla_stopping_curve(run, LOG)
        ex = info["exclusions"]
        assert ex["n_samples_total"] == 9
        assert ex["n_samples_excluded_as_control"] == 4
        assert ex["n_samples_excluded_by_integrity_gate"] == 1
        assert ex["n_samples_used"] == 4

    def test_every_sample_is_accounted_for(self, run):
        """Used plus excluded must equal the file, or something went missing."""
        _, info, _ = nla_stopping_curve(run, LOG)
        ex = info["exclusions"]
        assert (ex["n_samples_used"]
                + ex["n_samples_excluded_as_control"]
                + ex["n_samples_excluded_by_integrity_gate"]
                == ex["n_samples_total"])

    def test_the_gate_rate_excludes_the_control_arm_from_its_denominator(self, run):
        """The control arm is not a candidate for the gate, so counting it
        would dilute the rate the `nla` stage reports."""
        _, info, _ = nla_stopping_curve(run, LOG)
        assert info["exclusions"]["integrity_exclusion_rate"] == pytest.approx(1 / 5)

    def test_a_high_gate_rate_is_warned_about(self, tmp_path, caplog):
        rows = []
        for pid in ("p0", "p1"):
            for ci in (1, 2):
                rows.append(sample(pid, ci, cosine=0.6))
                rows.append(sample(pid, ci, cosine=0.6, integrity_ok=False))
                rows.append(sample(pid, ci, cosine=0.6, integrity_ok=False))
        cfg = write_run(tmp_path, rows)
        with caplog.at_level(logging.WARNING):
            _, info, _ = nla_stopping_curve(cfg, LOG)
        assert info["exclusions"]["integrity_exclusion_rate"] == pytest.approx(2 / 3)
        assert any("integrity gate excluded" in r.getMessage()
                   for r in caplog.records)

    def test_problem_counts_are_reported_beside_the_sample_counts(self, run):
        _, info, _ = nla_stopping_curve(run, LOG)
        ex = info["exclusions"]
        assert ex["n_problems_with_usable_samples"] == 2
        assert ex["n_problems_scored"] == 2
        assert info["n_problems"] == ex["n_problems_scored"]


class TestProblemsThatCannotBeScoredAreNamed:
    def test_a_problem_without_a_usable_trace_row_is_listed(self, tmp_path, caplog):
        rows = [sample("p0", 1), sample("p0", 2),
                sample("ghost", 1), sample("ghost", 2)]
        cfg = write_run(tmp_path, rows, problems=("p0",))
        with caplog.at_level(logging.WARNING):
            _, info, per_problem = nla_stopping_curve(cfg, LOG)
        ex = info["exclusions"]
        assert ex["problems_dropped_no_usable_trace"] == ["ghost"]
        assert ex["n_problems_with_usable_samples"] == 2
        assert ex["n_problems_scored"] == 1
        assert info["n_problems"] == 1
        assert [r["problem_id"] for r in per_problem] == ["p0"]
        assert any("could not be scored" in r.getMessage()
                   for r in caplog.records)

    def test_a_trace_with_no_chunks_is_listed_under_its_own_reason(self, tmp_path):
        cfg = write_run(tmp_path, [sample("p0", 1), sample("p1", 1)])
        path = cfg.dir / "traces" / "traces.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["chunks"] = []
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                        encoding="utf-8")
        _, info, _ = nla_stopping_curve(cfg, LOG)
        assert info["exclusions"]["problems_dropped_no_chunks"] == ["p1"]
        assert info["exclusions"]["problems_dropped_no_usable_trace"] == []
        assert info["n_problems"] == 1

    def test_the_accounting_survives_an_arm_with_nothing_usable(self, tmp_path):
        """An unavailable arm still has to say what it excluded."""
        rows = [sample("p0", 1, integrity_ok=False),
                sample("p0", 2, "gaussian_control")]
        cfg = write_run(tmp_path, rows)
        curve, info, per_problem = nla_stopping_curve(cfg, LOG)
        assert curve is None
        assert info["available"] is False
        assert per_problem == []
        assert info["exclusions"]["n_samples_used"] == 0
        assert info["exclusions"]["n_samples_excluded_by_integrity_gate"] == 1
