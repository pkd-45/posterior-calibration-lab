"""Two estimators for the amplitude: one exact, one a common shortcut.

`exact_amplitude_posterior` marginalises over the nuisance baseline properly. For
this Gaussian linear model that marginalisation is analytic, so the result is a
posterior that is correct by construction. It is the control.

`plugin_amplitude_posterior` does what is very often done in practice instead:
estimate the baseline once, freeze it, and infer the amplitude as though that
estimate were exact. This is the treatment under test. It is not a strawman - it
is fast, it is unbiased in the mean, and its point estimates are good. What it
loses is the uncertainty contributed by the nuisance, and that loss is invisible
to any point-estimate metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .simulator import Design, Priors, design_matrix

__all__ = [
    "Posterior",
    "exact_amplitude_posterior",
    "joint_posterior",
    "plugin_amplitude_posterior",
]


@dataclass(frozen=True)
class Posterior:
    """A one-dimensional Gaussian posterior on the amplitude."""

    mean: float
    sd: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mean, self.sd, size)

    def credible_interval(self, level: float) -> tuple[float, float]:
        """Central credible interval containing probability ``level``."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        half = stats.norm.ppf(0.5 * (1.0 + level)) * self.sd
        return self.mean - half, self.mean + half

    def cdf(self, value: float) -> float:
        return float(stats.norm.cdf(value, loc=self.mean, scale=self.sd))


def joint_posterior(
    y: np.ndarray, design: Design, priors: Priors
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gaussian posterior on ``(A, b)``.

    Standard conjugate result for Bayesian linear regression with Gaussian priors
    and known noise: the posterior precision is the prior precision plus the
    scaled Gram matrix of the design.
    """
    matrix = design_matrix(design)
    noise_precision = 1.0 / design.sigma**2
    prior_precision = np.linalg.inv(priors.covariance())

    precision = prior_precision + noise_precision * (matrix.T @ matrix)
    covariance = np.linalg.inv(precision)
    mean = covariance @ (noise_precision * (matrix.T @ y))
    return mean, covariance


def exact_amplitude_posterior(
    y: np.ndarray, design: Design, priors: Priors
) -> Posterior:
    """Posterior on the amplitude with the baseline marginalised out."""
    mean, covariance = joint_posterior(y, design, priors)
    return Posterior(mean=float(mean[0]), sd=float(np.sqrt(covariance[0, 0])))


def plugin_amplitude_posterior(
    y: np.ndarray, design: Design, priors: Priors, wing_threshold: float = 2.5
) -> Posterior:
    """Posterior on the amplitude with the baseline estimated once and frozen.

    The baseline is estimated from the "line-free" wings, meaning the sampled
    positions further than ``wing_threshold`` feature widths from the centre,
    where the feature contributes negligibly. This is the realistic shortcut: it
    is what continuum estimation from feature-free regions actually does, and it
    gives a nearly unbiased baseline and good amplitude point estimates.

    The flaw is only in the uncertainty. The wing estimate has a variance of its
    own, and freezing it discards that variance, so the reported amplitude
    posterior is narrower than it should be. No point-estimate metric can see
    this.
    """
    matrix = design_matrix(design)
    feature = matrix[:, 0]
    x = design.positions()

    wing = np.abs(x - design.centre) >= wing_threshold * design.width
    if not np.any(wing):
        raise ValueError("no sampled positions lie in the wings; widen the design")

    baseline_hat = float(np.mean(y[wing]))
    residual = y - baseline_hat

    noise_precision = 1.0 / design.sigma**2
    prior_precision = 1.0 / priors.amplitude_sd**2

    precision = prior_precision + noise_precision * float(feature @ feature)
    variance = 1.0 / precision
    mean = variance * noise_precision * float(feature @ residual)
    return Posterior(mean=mean, sd=float(np.sqrt(variance)))
