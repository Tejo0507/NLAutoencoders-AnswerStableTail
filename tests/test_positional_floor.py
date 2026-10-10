"""What the hidden-state probe is actually reading.

The probe is the baseline the primary research question is measured against,
and on the real pilot corpus it scores an AUC of 0.996. That looks
overwhelming until you ask what its label is: *the intermediate answer at this
boundary is already correct* becomes true once the trace has worked the answer
out and stays true afterwards, so it is strongly ordered by position in the
trace. A model that had learned nothing but "this boundary is late" would score
almost as well.

`positional_floor` fits the same model family, on the same grouped folds, on
activation-free positional features. On the pilot it reaches 0.962 - so the
layer-20 probe is 0.034 AUC above a predictor that never sees an activation.
That is the number that belongs next to the probe's, and these tests pin that
the floor is computed the way the probe is.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_baselines import positional_floor  # noqa: E402

LOG = logging.getLogger("test")


def corpus(n_problems=20, n_chunks=10, stable_from=6):
    """Boundaries whose label flips on at a fixed position in every trace.

    The degenerate case the floor exists to detect: the label is a pure
    function of position, so a positional predictor should be near perfect.
    """
    base, index, y, groups = {}, [], [], []
    for p in range(n_problems):
        pid = f"p{p:03d}"
        base[pid] = {"n_chunks": n_chunks,
                     "boundary_tokens": [(i + 1) * 8 for i in range(n_chunks)]}
        for i in range(n_chunks):
            index.append((pid, i))
            y.append(1 if i >= stable_from else 0)
            groups.append(pid)
    return base, index, np.asarray(y), groups


@pytest.fixture
def cfg():
    from nlaast import config as config_mod

    return config_mod.load("pilot", [])


def test_a_label_determined_by_position_is_found_by_the_floor(cfg):
    base, index, y, groups = corpus()
    out = positional_floor(base, index, y, groups, cfg, LOG)
    assert out["auc"] > 0.95
    assert out["n"] == len(y)


def test_a_label_independent_of_position_leaves_the_floor_at_chance(cfg):
    """The contrast that makes the floor meaningful: when correctness does not
    track position, a positional predictor must not score."""
    base, index, y, groups = corpus()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=len(y))
    out = positional_floor(base, index, y, groups, cfg, LOG)
    assert abs(out["auc"] - 0.5) < 0.2, out["auc"]


def test_the_features_are_all_activation_free(cfg):
    base, index, y, groups = corpus()
    out = positional_floor(base, index, y, groups, cfg, LOG)
    assert out["features"] == ["chunk_index", "relative_position", "prefix_tokens"]


def test_a_missing_token_count_does_not_crash_the_floor(cfg):
    """Short boundary_tokens lists appear when a trace was re-chunked; the
    floor must degrade rather than raise, since it is a diagnostic."""
    base, index, y, groups = corpus()
    for pid in base:
        base[pid]["boundary_tokens"] = base[pid]["boundary_tokens"][:3]
    out = positional_floor(base, index, y, groups, cfg, LOG)
    assert np.isfinite(out["auc"]) or np.isnan(out["auc"])


def test_the_note_tells_the_reader_how_to_use_the_number(cfg):
    base, index, y, groups = corpus()
    out = positional_floor(base, index, y, groups, cfg, LOG)
    assert "activation-free" in out["note"]
    assert "position" in out["note"]
