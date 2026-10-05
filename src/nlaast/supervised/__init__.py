"""Supervised analyses over the trace corpus.

Two analyses share one feature table and one splitting discipline:

* **regression** on ``tail_fraction`` - how much of a reasoning trace is
  post-answer redundancy (``scripts/predict_tail_fraction.py``);
* **classification** on ``final_correct`` - whether the trace's verified answer
  is right (``scripts/predict_correctness.py``).

Both targets are quantities the study already defines; neither is invented for
the sake of having something to model. See ``docs/SUPERVISED_ANALYSES.md`` for
the justification against the problem statement.

Nothing in this package reads an activation. That is deliberate: these models
are the *cheap* tier, and the point of measuring them is to establish the floor
that the layer-20 probe and the verbalised readout have to clear.
"""

from __future__ import annotations

from .features import (
    BLOCKS,
    FEATURE_BLOCKS,
    INFERENCE_FEATURES,
    build_feature_table,
    categorical_columns,
    numeric_columns,
)
from .evaluation import (
    bootstrap_metric,
    make_preprocessor,
    nested_block_scores,
    paired_metric_difference,
    permutation_null,
)

__all__ = [
    "BLOCKS",
    "FEATURE_BLOCKS",
    "INFERENCE_FEATURES",
    "build_feature_table",
    "categorical_columns",
    "numeric_columns",
    "bootstrap_metric",
    "make_preprocessor",
    "nested_block_scores",
    "paired_metric_difference",
    "permutation_null",
]
