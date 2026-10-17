"""F5's temperature and chunker sweeps: which conditions run, and how they are compared.

The sweep's conditions come from the configuration, not from the script, and
everything other than the swept factor must stay at the run's value - a sweep
that silently moved a second setting would report that setting's effect as the
temperature's. The comparison is paired over problems and has to reduce to an
exact zero when a condition reproduces the base run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_f5_sweep as F  # noqa: E402
from nlaast import config as config_mod  # noqa: E402


def cfg(**over):
    base = config_mod.from_mapping({})
    d = base.to_dict()
    for section, values in over.items():
        d[section] = {**d[section], **values}
    return config_mod.from_mapping(d)


def test_conditions_come_from_config_and_exclude_the_base_temperature():
    c = F.sweep_conditions(cfg())
    assert set(c) == {"T=0.6", "T=1.0", "chunker=newline"}
    assert c["T=0.6"] == {"factor": "temperature", "temperature": 0.6,
                          "chunk_mode": "sentence"}
    # The chunker condition runs at the base temperature.
    assert c["chunker=newline"]["temperature"] == 0.8


def test_conditions_follow_a_different_configuration():
    c = F.sweep_conditions(cfg(robustness={"ast_sweep_temperature": [0.5, 0.7]},
                               ast={"continuation_temperature": 0.7},
                               chunk={"alt_mode": "sentence"}))
    assert set(c) == {"T=0.5"}


def test_condition_config_changes_only_the_temperature():
    base = cfg()
    for name, cond in F.sweep_conditions(base).items():
        cc = F.condition_config(base, cond)
        a, b = base.to_dict(), cc.to_dict()
        assert b["ast"]["continuation_temperature"] == cond["temperature"]
        b["ast"]["continuation_temperature"] = a["ast"]["continuation_temperature"]
        assert a == b, name


def test_condition_hash_distinguishes_the_chunker():
    base = cfg()
    conds = F.sweep_conditions(base)
    newline = conds["chunker=newline"]
    sentence = {**newline, "chunk_mode": "sentence"}
    cc = F.condition_config(base, newline)
    # Same config hash (the chunker is not a config field the stage reads) ...
    assert cc.hash() == base.hash()
    # ... but a different condition hash.
    assert F.condition_hash(cc, newline) != F.condition_hash(cc, sentence)


def test_slug_is_filesystem_safe():
    assert F.slug("T=0.6") == "T_0p6"
    assert F.slug("chunker=newline") == "chunker_newline"


def ast_row(pid, tail_start, n_chunks=10, tail_tokens=30, total=100, correct=True):
    return {"problem_id": pid, "tail_start": tail_start, "n_chunks": n_chunks,
            "tail_fraction": None if tail_start is None else (n_chunks - tail_start) / n_chunks,
            "tail_tokens": tail_tokens if tail_start is not None else 0,
            "total_tokens": total, "final_correct": correct}


def test_compare_is_exactly_zero_against_itself():
    rows = [ast_row(f"p{i}", 5 + i % 3, tail_tokens=20 + i) for i in range(12)]
    out = F.compare(rows, rows, same_chunker=True, n_boot=200, confidence=0.95, seed=0)
    assert out["n_tail_in_both"] == 12
    assert out["tail_presence_agreement"] == 1.0
    assert out["tail_start_identical"] == 1.0
    for key in ("tail_token_fraction_diff", "tail_fraction_diff", "tail_start_diff_chunks"):
        assert out[key]["estimate"] == 0.0
    assert out["nontrivial_tail_rate_diff"]["estimate"] == 0.0
    assert out["spearman_tail_token_fraction"]["rho"] == pytest.approx(1.0)


def test_compare_sign_is_condition_minus_base():
    base = [ast_row(f"p{i}", 6, tail_tokens=30) for i in range(10)]
    cond = [ast_row(f"p{i}", 4, tail_tokens=50) for i in range(10)]
    out = F.compare(base, cond, same_chunker=True, n_boot=200, confidence=0.95, seed=0)
    assert out["tail_token_fraction_diff"]["estimate"] == pytest.approx(0.2)
    assert out["tail_start_diff_chunks"]["estimate"] == pytest.approx(-2.0)


def test_compare_under_another_chunker_omits_chunk_index_quantities():
    rows = [ast_row(f"p{i}", 5) for i in range(5)]
    out = F.compare(rows, rows, same_chunker=False, n_boot=100, confidence=0.95, seed=0)
    assert "tail_fraction_diff" not in out
    assert "tail_start_identical" not in out
    assert "tail_token_fraction_diff" in out


def test_compare_counts_tail_presence_and_nontrivial_changes():
    base = [ast_row("a", 5), ast_row("b", None), ast_row("c", 9)]   # c trivial
    cond = [ast_row("a", 5), ast_row("b", 5), ast_row("c", 5)]
    out = F.compare(base, cond, same_chunker=True, n_boot=100, confidence=0.95, seed=0)
    assert out["n_shared"] == 3
    assert out["n_tail_in_both"] == 2
    assert out["tail_presence_agreement"] == pytest.approx(2 / 3)
    nt = out["nontrivial_tail_rate_diff"]
    assert (nt["condition_only"], nt["base_only"]) == (2, 0)
    assert nt["estimate"] == pytest.approx(2 / 3)


def test_q_under_substitution_reproduces_and_moves_with_p():
    family = [
        {"test": "RQ1", "p_value": 0.02},
        {"test": "other", "p_value": 0.001},
        {"test": "untested", "p_value": None},
    ]
    same = F.q_under_substitution(family, "RQ1", 0.02)
    assert same["family_size"] == 2
    assert same["q_value"] == pytest.approx(0.02)
    worse = F.q_under_substitution(family, "RQ1", 0.2)
    assert worse["q_value"] == pytest.approx(0.2)
    assert worse["significant"] is False


def test_forced_reproduction_compares_only_boundaries_both_scans_evaluated():
    base = {"p": {"boundaries": [
        {"index": 0, "evaluated": False, "forced_answer": None},
        {"index": 1, "evaluated": True, "forced_answer": "4"},
        {"index": 2, "evaluated": True, "forced_answer": "5"},
    ]}}
    cond = [{"problem_id": "p", "boundaries": [
        {"index": 0, "evaluated": True, "forced_answer": "3"},
        {"index": 1, "evaluated": True, "forced_answer": "4"},
        {"index": 2, "evaluated": True, "forced_answer": "6"},
    ]}]
    out = F.forced_reproduction(base, cond)
    assert out == {"n_boundaries_compared": 2, "forced_answer_identical": 1, "rate": 0.5}


def test_condition_row_keeps_the_trace_and_replaces_the_evidence():
    from nlaast.trace.chunking import chunk_trace
    trace = {"problem_id": "p", "trace": "First. Second line.\nThird.", "final_answer": "1",
             "n_tokens": 9, "chunks": ["stale"], "boundaries": ["stale"],
             "entropy_samples": [{"answer": "1"}]}
    chunks = chunk_trace(trace["trace"], config_mod.from_mapping({}).chunk, mode="newline")
    row = F.condition_row(trace, chunks, [{"index": 0}], False, 12, "chunker=newline")
    assert row["trace"] == trace["trace"] and row["n_tokens"] == 9
    assert row["n_chunks"] == len(chunks) and row["chunks"][0]["index"] == 0
    assert row["boundaries"] == [{"index": 0}]
    assert "entropy_samples" not in row
    assert row["condition"] == "chunker=newline"
