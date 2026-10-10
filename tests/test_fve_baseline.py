"""Which baseline the reconstruction FVE is measured against.

`FVE = 1 - mean(MSE) / baseline`, and the usual baseline of 2.0 assumes an
uninformative prediction is *orthogonal* to the target. A random direction in
3584 dimensions is. A plausible guess is not — and layer-20 residual streams
are strongly anisotropic, so "guess a typical activation" is a plausible
guess.

Measured on the pilot's own 278 layer-20 vectors, two drawn at random have a
mean cosine of 0.584, so an uninformative prediction achieves MSE 0.832 rather
than 2.0. The same mean reconstruction MSE of 0.268 therefore gives:

    FVE = 0.866  against 2.0
    FVE = 0.678  against 0.832

PROJECT_PLAN §5 specifies the empirical baseline. The code used 2.0, which is
the flattering end of a 0.19 spread on a headline number.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.nla.reconstructor import (
    empirical_baseline_mse,
    fraction_variance_explained,
)


def test_isotropic_vectors_give_a_baseline_near_two():
    """The case the hardcoded 2.0 was right for: random directions in high
    dimensions really are almost orthogonal."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((400, 1024))
    out = empirical_baseline_mse(v, n_pairs=20_000)
    assert out["available"] is True
    assert abs(out["mean_pairwise_cosine"]) < 0.05
    assert abs(out["baseline_mse"] - 2.0) < 0.1


def test_anisotropic_vectors_give_a_much_lower_baseline():
    """The real case. A shared mean direction - which residual streams have -
    makes any two activations similar, so an uninformative prediction is far
    better than orthogonal and 2.0 overstates the headroom."""
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(1024)
    shared /= np.linalg.norm(shared)
    # The shared component has to dominate the per-vector noise, whose norm
    # grows as sqrt(d): 0.5 * sqrt(1024) = 16, so a shared norm of 10 would
    # still leave the vectors nearly orthogonal.
    v = rng.standard_normal((400, 1024)) * 0.5 + shared * 30.0
    out = empirical_baseline_mse(v, n_pairs=20_000)
    assert out["mean_pairwise_cosine"] > 0.5
    assert out["baseline_mse"] < 1.0


def test_the_baseline_changes_the_fve_materially():
    mses = [0.268] * 50
    theoretical = fraction_variance_explained(mses)["fve"]
    empirical = fraction_variance_explained(mses, baseline_mse=0.832)["fve"]
    assert theoretical == pytest.approx(0.866, abs=1e-3)
    assert empirical == pytest.approx(0.678, abs=1e-3)
    assert theoretical - empirical > 0.15


def test_the_baseline_used_is_always_reported_with_the_number():
    """The whole point: an FVE without its baseline is not interpretable."""
    out = fraction_variance_explained([0.268], baseline_mse=0.832)
    assert out["baseline_mse"] == pytest.approx(0.832)
    assert fraction_variance_explained([0.268])["baseline_mse"] == 2.0


def test_direction_is_all_that_matters_so_scale_is_irrelevant():
    """Both vectors are normalised before the MSE, so scaling the sample must
    not move the baseline."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal((200, 256))
    a = empirical_baseline_mse(v, n_pairs=10_000, seed=3)
    b = empirical_baseline_mse(v * 137.0, n_pairs=10_000, seed=3)
    assert a["baseline_mse"] == pytest.approx(b["baseline_mse"], rel=1e-9)


def test_too_few_vectors_is_unavailable_not_a_crash():
    assert empirical_baseline_mse(np.zeros((1, 16)))["available"] is False
    assert empirical_baseline_mse(np.zeros((0, 16)))["available"] is False


def test_zero_vectors_do_not_divide_by_zero():
    out = empirical_baseline_mse(np.zeros((10, 16)), n_pairs=100)
    assert out["available"] is True
    assert np.isfinite(out["baseline_mse"])
