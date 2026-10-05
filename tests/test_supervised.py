"""Feature construction and evaluation helpers for the two supervised analyses.

The things worth pinning here are the ones that would fail silently: a feature
that is constant (and so carries no information while looking like a
predictor), a nested block that is not actually nested, a preprocessor that
leaks test-fold statistics into the training fold, and a permutation null that
cannot detect a model fitting noise.
"""

import numpy as np
import pandas as pd
import pytest

from nlaast.supervised import (
    BLOCKS,
    FEATURE_BLOCKS,
    INFERENCE_FEATURES,
    build_feature_table,
    make_preprocessor,
    nested_block_scores,
    paired_metric_difference,
    permutation_null,
)
from nlaast.supervised.features import (
    _answer_mentions,
    _gold_magnitude,
    _parse_level,
    features_for_trace,
)


# --- Fixtures ---


def chunk(index, text, prefix_tokens):
    return {"index": index, "text": text, "char_start": 0, "char_end": len(text),
            "token_end": prefix_tokens - 1, "prefix_tokens": prefix_tokens}


def boundary(index, parsed=None, prefix_tokens=10):
    return {"index": index, "parsed_answer": parsed, "forced_answer": None,
            "forced_matches_final": False, "continuation_answers": [],
            "continuations_matching": 0, "n_continuations": 0,
            "prefix_tokens": prefix_tokens, "evaluated": False}


@pytest.fixture
def trace():
    return {
        "id": "gsm8k-test-00001",
        "dataset": "gsm8k",
        "split": "train",
        "question": "Tom has 5 apples and buys 7 more. How many apples?",
        "gold": "12",
        "trace": ("Tom starts with 5 apples. He buys 7 more, so 5 + 7 = 12. "
                  "Let me check that: 5 + 7 = 12, correct. The answer is 12."),
        "n_tokens": 44,
        "prompt_tokens": 30,
        "truncated": False,
        "final_answer": "12",
        "final_correct": True,
        "n_chunks": 4,
        "chunks": [
            chunk(0, "Tom starts with 5 apples.", 8),
            chunk(1, "He buys 7 more, so 5 + 7 = 12.", 20),
            chunk(2, "Let me check that: 5 + 7 = 12, correct.", 34),
            chunk(3, "The answer is 12.", 44),
        ],
        "boundaries": [boundary(0), boundary(1, "12"), boundary(2, "12"),
                       boundary(3, "12")],
        "ok": True,
    }


@pytest.fixture
def ast_row():
    return {"id": "gsm8k-test-00001", "status": "ok", "n_chunks": 4,
            "final_answer": "12", "final_correct": True, "tail_start": 1,
            "tail_fraction": 0.75, "tail_tokens": 36, "total_tokens": 44,
            "convergence_start": 2, "evidence": [], "notes": {}}


@pytest.fixture
def problem():
    return {"id": "gsm8k-test-00001", "dataset": "gsm8k", "split": "train",
            "level": None, "subject": "gsm8k", "gold": "12"}


# --- Feature blocks ---


def test_blocks_are_strictly_nested():
    """``BLOCKS`` must be cumulative, or the reported increments are nonsense."""
    names = list(BLOCKS)
    for earlier, later in zip(names, names[1:]):
        assert set(BLOCKS[earlier]) < set(BLOCKS[later])
    assert BLOCKS[names[-1]] == [c for cols in FEATURE_BLOCKS.values() for c in cols]


def test_no_feature_appears_in_two_blocks():
    seen = [c for cols in FEATURE_BLOCKS.values() for c in cols]
    assert len(seen) == len(set(seen))


def test_inference_subset_is_a_subset_of_the_features():
    assert set(INFERENCE_FEATURES) <= set(BLOCKS["C_convergence"])


def test_no_target_or_identity_column_is_a_feature():
    """The guard against the most consequential possible mistake."""
    forbidden = {"tail_fraction", "final_correct", "tail_start", "tail_tokens",
                 "total_tokens", "id", "split", "ast_status", "gold",
                 "final_answer"}
    assert forbidden.isdisjoint(set(BLOCKS["C_convergence"]))


# --- Individual features ---


def test_features_match_the_hand_computed_trace(trace, ast_row, problem):
    row = features_for_trace(trace, ast_row, problem)

    assert row["tail_fraction"] == pytest.approx(0.75)
    assert row["final_correct"] == 1
    assert row["n_chunks"] == 4
    assert row["n_tokens"] == 44
    # prefix_tokens 8, 20, 34, 44 -> per-chunk 8, 12, 14, 10
    assert row["tokens_per_chunk_mean"] == pytest.approx(11.0)
    # "check" is in chunk 2; one hit over four chunks
    assert row["cue_verify_per_chunk"] == pytest.approx(0.25)
    # convergence_start 2 of n_chunks-1 = 3
    assert row["convergence_position"] == pytest.approx(2 / 3)
    assert row["has_convergence"] == 1
    assert row["n_parsed_boundaries"] == 3


def test_answer_mention_finds_the_value_not_the_announcement(trace):
    """The value appears in chunk 1; the announcement is in chunk 3.

    The system prompt pins the announcement to the last chunk, so a feature
    keyed on it would be constant. This feature must key on the value.
    """
    position, count = _answer_mentions(trace["chunks"], "12")
    assert position == pytest.approx(1 / 3)
    assert count == 3


def test_answer_mention_is_missing_rather_than_zero_when_absent(trace):
    position, count = _answer_mentions(trace["chunks"], "99999")
    assert np.isnan(position)
    assert count == 0


def test_answer_mention_handles_a_single_chunk_trace():
    position, count = _answer_mentions([chunk(0, "The answer is 7.", 5)], "7")
    assert position == 0.0
    assert count == 1


@pytest.mark.parametrize("gold, log10, is_int", [
    ("6277", pytest.approx(np.log10(6277)), 1),
    ("1,234", pytest.approx(np.log10(1234)), 1),
    ("0", 0.0, 1),
    ("-50", pytest.approx(np.log10(50)), 1),
    ("2.5", pytest.approx(np.log10(2.5)), 0),
])
def test_gold_magnitude_is_scale_not_value(gold, log10, is_int):
    value, integer = _gold_magnitude(gold)
    assert value == log10
    assert integer == is_int


def test_gold_magnitude_of_a_symbolic_answer_is_missing():
    """``\\frac{1}{2}`` has no float magnitude; the imputer handles it."""
    value, integer = _gold_magnitude(r"\frac{1}{2}")
    assert np.isnan(value)
    assert integer == 0


@pytest.mark.parametrize("raw, expected", [
    ("Level 3", 3.0), ("Level 5", 5.0), (None, None), ("", None), ("hard", None),
])
def test_parse_level(raw, expected):
    value = _parse_level(raw)
    if expected is None:
        assert np.isnan(value)
    else:
        assert value == expected


# --- Table assembly ---


def test_build_table_drops_failed_traces_and_keeps_the_rest(trace, ast_row, problem):
    failed = {"id": "gsm8k-test-00002", "ok": False, "error": "boom"}
    orphan = dict(trace, id="gsm8k-test-00003")  # no AST record
    table = build_feature_table([trace, failed, orphan], [ast_row], [problem])

    assert list(table["id"]) == ["gsm8k-test-00001"]


def test_build_table_declares_every_block_column(trace, ast_row, problem):
    table = build_feature_table([trace], [ast_row], [problem])
    for column in BLOCKS["C_convergence"]:
        assert column in table.columns


def test_missing_math_level_stays_missing_rather_than_zero(trace, ast_row, problem):
    """A GSM8K level of 0 would be a lie; the pipeline imputes with an indicator."""
    table = build_feature_table([trace], [ast_row], [problem])
    assert table["math_level"].isna().all()


# --- Preprocessing and leakage ---


def frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in BLOCKS["C_convergence"]
            if c not in ("dataset", "subject")}
    data["dataset"] = rng.choice(["gsm8k", "math"], size=n)
    data["subject"] = rng.choice(["algebra", "gsm8k"], size=n)
    return pd.DataFrame(data)


def test_preprocessor_fits_on_train_only():
    """The check that matters: transform must use the *fitted* mean, not its own.

    If scaling were recomputed on the test frame, a shifted test frame would
    come back centred on zero. It must come back shifted.
    """
    columns = BLOCKS["C_convergence"]
    train = frame(60, seed=1)
    pre = make_preprocessor(columns, scale=True).fit(train[columns])

    shifted = train.copy()
    shifted["n_tokens"] = shifted["n_tokens"] + 10.0
    out = pre.transform(shifted[columns])

    assert abs(float(np.asarray(out).mean())) > 0.1


def test_preprocessor_imputes_and_flags_a_fully_missing_column():
    columns = BLOCKS["C_convergence"]
    train = frame(40, seed=2)
    train["math_level"] = np.nan
    out = np.asarray(make_preprocessor(columns, scale=True).fit_transform(train[columns]))

    assert np.isfinite(out).all()


def test_preprocessor_tolerates_an_unseen_category():
    columns = BLOCKS["C_convergence"]
    train = frame(40, seed=3)
    pre = make_preprocessor(columns, scale=True).fit(train[columns])

    unseen = train.copy()
    unseen["dataset"] = "aime"
    assert np.isfinite(np.asarray(pre.transform(unseen[columns]))).all()


# --- Evaluation helpers ---


def test_independent_columns_drops_a_duplicated_indicator():
    """The structural case: ``math_level`` is missing exactly for GSM8K rows.

    The imputer's missingness indicator for it is then identical to the
    ``dataset=gsm8k`` dummy, and a design holding both is rank-deficient.
    """
    from nlaast.supervised.evaluation import independent_columns

    rng = np.random.default_rng(0)
    is_gsm8k = np.repeat([0.0, 1.0], 25)
    design = np.column_stack([
        rng.normal(size=50),   # n_tokens
        is_gsm8k,              # dataset=gsm8k dummy
        is_gsm8k,              # missingindicator_math_level - the duplicate
        np.ones(50),           # a constant column
    ])
    names = ["n_tokens", "dataset_gsm8k", "missingindicator_math_level", "const"]

    pruned, kept, dropped = independent_columns(design, names)

    assert kept == ["n_tokens", "dataset_gsm8k"]
    assert np.linalg.matrix_rank(pruned) == pruned.shape[1]
    assert any("constant" in d for d in dropped)
    assert any("duplicates dataset_gsm8k" in d for d in dropped)


def test_independent_columns_keeps_merely_correlated_columns():
    """Correlated is not redundant; only near-identical columns are dropped."""
    from nlaast.supervised.evaluation import independent_columns

    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    design = np.column_stack([a, a + rng.normal(scale=0.5, size=200)])

    _, kept, dropped = independent_columns(design, ["a", "b"])
    assert kept == ["a", "b"]
    assert dropped == []


def test_nested_block_scores_reports_increments_over_the_previous_block():
    scores = {"A_problem": 0.10, "B_trace": 0.30, "C_convergence": 0.34}
    out = nested_block_scores(
        lambda columns: {"primary": scores[
            next(k for k, v in BLOCKS.items() if v == columns)]}
    )
    assert list(out["block"]) == list(BLOCKS)
    assert np.isnan(out["increment"].iloc[0])
    assert out["increment"].iloc[1] == pytest.approx(0.20)
    assert out["increment"].iloc[2] == pytest.approx(0.04)


def test_paired_difference_detects_a_real_gap_and_reports_its_sign():
    rng = np.random.default_rng(0)
    y = rng.normal(size=120)
    good = y + rng.normal(scale=0.1, size=120)
    bad = y + rng.normal(scale=1.5, size=120)

    from sklearn.metrics import mean_absolute_error
    out = paired_metric_difference(y, good, bad, mean_absolute_error,
                                   n_iterations=500, seed=0)
    assert out["estimate"] < 0
    assert out["excludes_zero"]


def test_paired_difference_on_identical_predictions_is_zero():
    from sklearn.metrics import mean_absolute_error
    y = np.arange(50, dtype=float)
    out = paired_metric_difference(y, y + 1.0, y + 1.0, mean_absolute_error,
                                   n_iterations=200, seed=0)
    assert out["estimate"] == pytest.approx(0.0)
    assert not out["excludes_zero"]


def test_permutation_null_separates_real_signal_from_noise():
    """A score function that reads the labels must beat its own shuffled null."""
    y = np.array([0, 1] * 30, dtype=float)

    def honest(labels):
        # correlates perfectly with whatever labels it is handed, but only the
        # unshuffled call matches the fixed reference
        return float(np.corrcoef(labels, y)[0, 1])

    out = permutation_null(honest, y, n_permutations=100, seed=0)
    assert out["observed"] == pytest.approx(1.0)
    assert out["p_value"] < 0.05


def test_permutation_null_does_not_flag_a_model_that_ignores_its_labels():
    out = permutation_null(lambda labels: 0.42, np.array([0, 1] * 20, dtype=float),
                           n_permutations=50, seed=0)
    assert out["p_value"] == pytest.approx(1.0)
