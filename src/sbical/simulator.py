"""A deliberately minimal forward model with a signal and a nuisance baseline.

The model is *not* a physical simulator and makes no astrophysical claim. It is a
stand-in chosen because it shares the structure of many spectroscopic amplitude
measurements: a localised feature of interest sits on top of a poorly known
continuum-like baseline, and both are seen through noise.

    y_i = A * g(x_i; centre, width) + b + eps_i ,    eps_i ~ N(0, sigma^2)

`A` is the parameter of interest, `b` is a nuisance parameter, and the two become
degenerate when the feature is sampled over a range too narrow to separate it
from a constant offset.

The model is linear in ``(A, b)``, which is the point: with Gaussian priors the
exact posterior is available in closed form, so the calibration diagnostics in
`sbical.diagnostics` can be verified against a posterior that is known to be
correct before they are trusted on an approximate one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Design", "Priors", "design_matrix", "profile", "sample_prior", "simulate"]


@dataclass(frozen=True)
class Design:
    """Observation design: where the spectrum is sampled, and how noisily.

    Parameters
    ----------
    n_points:
        Number of sampled positions.
    half_width:
        Positions span ``[-half_width, +half_width]`` in units of the feature
        width. Small values sample only the core of the feature and leave the
        baseline poorly constrained, which is the degenerate regime.
    centre, width:
        Location and scale of the feature. Held fixed and known; the point of the
        demonstrator is nuisance handling, not shape fitting.
    sigma:
        Per-point Gaussian noise standard deviation.
    """

    n_points: int = 24
    half_width: float = 1.2
    centre: float = 0.0
    width: float = 1.0
    sigma: float = 0.25

    def positions(self) -> np.ndarray:
        return np.linspace(-self.half_width, self.half_width, self.n_points)


@dataclass(frozen=True)
class Priors:
    """Independent Gaussian priors on the amplitude and the baseline."""

    amplitude_sd: float = 1.0
    baseline_sd: float = 1.0

    def covariance(self) -> np.ndarray:
        return np.diag([self.amplitude_sd**2, self.baseline_sd**2])


def profile(x: np.ndarray, centre: float = 0.0, width: float = 1.0) -> np.ndarray:
    """Unit-height Gaussian feature."""
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def design_matrix(design: Design) -> np.ndarray:
    """Return the ``(n_points, 2)`` matrix whose columns multiply ``(A, b)``."""
    x = design.positions()
    return np.column_stack(
        [profile(x, design.centre, design.width), np.ones_like(x)]
    )


def sample_prior(priors: Priors, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Draw ``size`` parameter vectors ``(A, b)`` from the prior."""
    return np.column_stack(
        [
            rng.normal(0.0, priors.amplitude_sd, size),
            rng.normal(0.0, priors.baseline_sd, size),
        ]
    )


def simulate(
    theta: np.ndarray, design: Design, rng: np.random.Generator
) -> np.ndarray:
    """Generate one noisy observation for parameters ``theta = (A, b)``."""
    theta = np.asarray(theta, dtype=float).reshape(2)
    mean = design_matrix(design) @ theta
    return mean + rng.normal(0.0, design.sigma, mean.shape)


def degeneracy_index(design: Design) -> float:
    """How strongly the amplitude and the baseline trade off against each other.

    Returns the absolute correlation coefficient between the two parameters in
    the least-squares covariance ``(X^T X)^-1``. Values near 1 mean the feature
    cannot be told apart from a constant offset over the sampled range, so the
    baseline must either be marginalised or constrained externally.
    """
    matrix = design_matrix(design)
    gram = matrix.T @ matrix
    covariance = np.linalg.inv(gram)
    denominator = np.sqrt(covariance[0, 0] * covariance[1, 1])
    if denominator == 0:
        return 1.0
    return float(min(1.0, abs(covariance[0, 1] / denominator)))
