"""Calibration diagnostics that can be run against any posterior estimator.

Three questions, in increasing order of how often they are actually asked:

1. Are the point estimates close to the truth?  (Almost always asked.)
2. Do the credible intervals contain the truth as often as they claim?
3. Is the whole posterior shape right, not just its width?

Question 1 is answered by `point_estimate_metrics`, question 2 by
`coverage_curve`, and question 3 by `simulation_based_calibration`.

The central claim of this package is that 1 can look excellent while 2 and 3 fail
badly, and that only 2 and 3 tell you whether an inferred parameter can be
interpreted physically.

Simulation-based calibration follows Talts et al. (2018): if the posterior is
correct, then drawing a parameter from the prior, simulating data from it, and
computing the rank of that parameter within its own posterior yields ranks that
are uniformly distributed. Any departure from uniformity is a fault in the
inference, not in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import stats

from .posteriors import Posterior
from .simulator import Design, Priors, sample_prior, simulate

__all__ = [
    "CalibrationResult",
    "Estimator",
    "coverage_curve",
    "point_estimate_metrics",
    "run_calibration",
    "simulation_based_calibration",
]


class Estimator(Protocol):
    """Any callable that turns data into a one-dimensional amplitude posterior."""

    def __call__(self, y: np.ndarray, design: Design, priors: Priors) -> Posterior:
        ...


@dataclass
class CalibrationResult:
    """Everything needed to judge one estimator."""

    name: str
    truths: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    ranks: np.ndarray
    n_rank_bins: int
    nominal_levels: np.ndarray
    empirical_coverage: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        out = dict(self.metrics)
        out["sbc_uniformity_pvalue"] = self.sbc_uniformity_pvalue()
        out["coverage_at_0.90"] = float(
            np.interp(0.90, self.nominal_levels, self.empirical_coverage)
        )
        out["max_coverage_error"] = float(
            np.max(np.abs(self.empirical_coverage - self.nominal_levels))
        )
        return out

    def sbc_uniformity_pvalue(self) -> float:
        """Chi-square p-value against a uniform rank histogram.

        A small p-value means the posterior is demonstrably miscalibrated.
        """
        counts = np.bincount(self.ranks, minlength=self.n_rank_bins + 1)
        expected = np.full(counts.size, self.ranks.size / counts.size)
        statistic = float(np.sum((counts - expected) ** 2 / expected))
        return float(stats.chi2.sf(statistic, df=counts.size - 1))


def simulation_based_calibration(
    estimator: Estimator,
    design: Design,
    priors: Priors,
    n_simulations: int = 2000,
    n_posterior_samples: int = 99,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SBC and return ``(truths, ranks, posteriors)``.

    ``n_posterior_samples`` sets the number of rank bins to
    ``n_posterior_samples + 1``. Using a value one less than a round number keeps
    the histogram bins evenly populated under the null.
    """
    rng = np.random.default_rng(seed)
    truths = np.empty(n_simulations)
    ranks = np.empty(n_simulations, dtype=int)
    means = np.empty(n_simulations)
    sds = np.empty(n_simulations)

    for i in range(n_simulations):
        theta = sample_prior(priors, rng, size=1)[0]
        y = simulate(theta, design, rng)
        posterior = estimator(y, design, priors)

        draws = posterior.sample(rng, n_posterior_samples)
        truths[i] = theta[0]
        ranks[i] = int(np.sum(draws < theta[0]))
        means[i] = posterior.mean
        sds[i] = posterior.sd

    return truths, ranks, np.column_stack([means, sds])


def coverage_curve(
    truths: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical containment fraction of central credible intervals.

    For a correct posterior the returned curve tracks the diagonal. A curve below
    the diagonal means the intervals are too narrow, so a stated 90% interval
    contains the truth less than 90% of the time.
    """
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)

    empirical = np.empty(levels.size)
    for j, level in enumerate(levels):
        half = stats.norm.ppf(0.5 * (1.0 + level)) * sds
        inside = np.abs(truths - means) <= half
        empirical[j] = float(np.mean(inside))
    return levels, empirical


def point_estimate_metrics(truths: np.ndarray, means: np.ndarray) -> dict[str, float]:
    """The metrics that are almost always reported, and are not enough."""
    residual = means - truths
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "correlation": float(np.corrcoef(truths, means)[0, 1]),
    }


def run_calibration(
    name: str,
    estimator: Estimator,
    design: Design,
    priors: Priors,
    n_simulations: int = 2000,
    n_posterior_samples: int = 99,
    seed: int = 0,
) -> CalibrationResult:
    """Run the full battery for one estimator."""
    truths, ranks, posteriors = simulation_based_calibration(
        estimator, design, priors, n_simulations, n_posterior_samples, seed
    )
    means, sds = posteriors[:, 0], posteriors[:, 1]
    levels, empirical = coverage_curve(truths, means, sds)

    return CalibrationResult(
        name=name,
        truths=truths,
        means=means,
        sds=sds,
        ranks=ranks,
        n_rank_bins=n_posterior_samples,
        nominal_levels=levels,
        empirical_coverage=empirical,
        metrics=point_estimate_metrics(truths, means),
    )
