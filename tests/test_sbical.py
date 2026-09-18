"""Tests.

The most important test in this file is
`test_exact_posterior_passes_its_own_calibration_check`. The diagnostics are only
worth anything if they clear a posterior that is correct by construction. If that
test fails, every other result in the package is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sbical import (
    Design,
    Priors,
    coverage_curve,
    degeneracy_index,
    design_matrix,
    exact_amplitude_posterior,
    joint_posterior,
    plugin_amplitude_posterior,
    run_calibration,
    simulate,
)

NOMINAL = Design(half_width=4.0, n_points=32)
PRIORS = Priors()


# --------------------------------------------------------------- simulator


def test_design_matrix_shape_and_columns():
    matrix = design_matrix(NOMINAL)
    assert matrix.shape == (NOMINAL.n_points, 2)
    assert np.allclose(matrix[:, 1], 1.0)
    # unit height, though no sample need land exactly on the centre
    assert 0.9 < matrix[:, 0].max() <= 1.0


def test_simulation_is_reproducible_from_a_seed():
    a = simulate([1.0, 0.5], NOMINAL, np.random.default_rng(7))
    b = simulate([1.0, 0.5], NOMINAL, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_free_signal_matches_the_forward_model():
    theta = np.array([0.8, -0.3])
    noiseless = Design(half_width=4.0, n_points=32, sigma=0.0)
    y = simulate(theta, noiseless, np.random.default_rng(0))
    assert np.allclose(y, design_matrix(noiseless) @ theta)


def test_narrow_sampling_is_more_degenerate_than_wide():
    assert degeneracy_index(Design(half_width=0.5)) > degeneracy_index(
        Design(half_width=6.0)
    )


# --------------------------------------------------------------- posteriors


def test_joint_posterior_matches_a_brute_force_grid():
    """Check the conjugate algebra against direct numerical integration."""
    rng = np.random.default_rng(11)
    design = Design(half_width=4.0, n_points=24, sigma=0.3)
    y = simulate([0.7, -0.2], design, rng)

    mean, cov = joint_posterior(y, design, PRIORS)

    grid = np.linspace(-3, 3, 401)
    aa, bb = np.meshgrid(grid, grid, indexing="ij")
    matrix = design_matrix(design)
    model = aa[..., None] * matrix[:, 0] + bb[..., None] * matrix[:, 1]
    log_like = -0.5 * np.sum((y - model) ** 2, axis=-1) / design.sigma**2
    log_prior = -0.5 * (
        (aa / PRIORS.amplitude_sd) ** 2 + (bb / PRIORS.baseline_sd) ** 2
    )
    post = np.exp(log_like + log_prior - np.max(log_like + log_prior))
    post /= post.sum()

    assert np.sum(post * aa) == pytest.approx(mean[0], abs=2e-3)
    assert np.sum(post * bb) == pytest.approx(mean[1], abs=2e-3)
    var_a = np.sum(post * aa**2) - np.sum(post * aa) ** 2
    assert np.sqrt(var_a) == pytest.approx(np.sqrt(cov[0, 0]), rel=2e-2)


def test_marginal_posterior_is_wider_than_the_plugin():
    """Marginalising over the nuisance must cost uncertainty, never save it."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = simulate([rng.normal(), rng.normal()], NOMINAL, rng)
        exact = exact_amplitude_posterior(y, NOMINAL, PRIORS)
        plug = plugin_amplitude_posterior(y, NOMINAL, PRIORS)
        assert exact.sd > plug.sd


def test_credible_interval_widths_are_correct():
    from sbical.posteriors import Posterior

    post = Posterior(mean=1.0, sd=2.0)
    lo, hi = post.credible_interval(0.95)
    assert (hi - lo) / 2 == pytest.approx(stats.norm.ppf(0.975) * 2.0)
    assert post.cdf(1.0) == pytest.approx(0.5)


def test_credible_interval_rejects_invalid_levels():
    from sbical.posteriors import Posterior

    with pytest.raises(ValueError):
        Posterior(0.0, 1.0).credible_interval(1.0)


def test_plugin_requires_wings_to_exist():
    with pytest.raises(ValueError):
        plugin_amplitude_posterior(
            np.zeros(8), Design(half_width=1.0, n_points=8), PRIORS
        )


# ------------------------------------------------------------- diagnostics


def test_exact_posterior_passes_its_own_calibration_check():
    """The control. A correct posterior must give uniform SBC ranks.

    This is what licenses the package to call the plug-in miscalibrated.
    """
    result = run_calibration(
        "exact", exact_amplitude_posterior, NOMINAL, PRIORS,
        n_simulations=2000, seed=101,
    )
    assert result.sbc_uniformity_pvalue() > 0.01
    assert result.summary()["max_coverage_error"] < 0.04
    assert result.summary()["coverage_at_0.90"] == pytest.approx(0.90, abs=0.03)


def test_plugin_posterior_is_detected_as_miscalibrated():
    result = run_calibration(
        "plug-in", plugin_amplitude_posterior, NOMINAL, PRIORS,
        n_simulations=2000, seed=101,
    )
    assert result.sbc_uniformity_pvalue() < 1e-6
    assert result.summary()["coverage_at_0.90"] < 0.80


def test_point_estimates_alone_would_not_reveal_the_problem():
    """The claim the package exists to make."""
    kwargs = {"design": NOMINAL, "priors": PRIORS,
              "n_simulations": 2000, "seed": 101}
    exact = run_calibration("exact", exact_amplitude_posterior, **kwargs)
    plug = run_calibration("plug-in", plugin_amplitude_posterior, **kwargs)

    assert plug.metrics["rmse"] / exact.metrics["rmse"] < 1.25
    assert plug.metrics["correlation"] > 0.95
    assert abs(plug.metrics["bias"]) < 0.05
    # ... yet the intervals are badly wrong.
    gap = exact.summary()["coverage_at_0.90"] - plug.summary()["coverage_at_0.90"]
    assert gap > 0.10


def test_coverage_of_a_correct_posterior_tracks_the_diagonal():
    rng = np.random.default_rng(5)
    n = 4000
    truths = rng.normal(0.0, 1.0, n)
    sds = np.full(n, 0.4)
    means = truths + rng.normal(0.0, 0.4, n)
    levels, empirical = coverage_curve(truths, means, sds)
    assert np.max(np.abs(empirical - levels)) < 0.03


def test_coverage_detects_overconfidence():
    rng = np.random.default_rng(6)
    n = 4000
    truths = rng.normal(0.0, 1.0, n)
    means = truths + rng.normal(0.0, 0.4, n)
    sds = np.full(n, 0.2)  # half the true scatter
    _, empirical = coverage_curve(truths, means, sds)
    assert empirical[-1] < 0.80


def test_ranks_are_within_bounds():
    result = run_calibration(
        "exact", exact_amplitude_posterior, NOMINAL, PRIORS,
        n_simulations=200, n_posterior_samples=49, seed=9,
    )
    assert result.ranks.min() >= 0
    assert result.ranks.max() <= 49
