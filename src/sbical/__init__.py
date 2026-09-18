"""Calibration diagnostics for approximate posteriors."""

from .diagnostics import (
    CalibrationResult,
    coverage_curve,
    point_estimate_metrics,
    run_calibration,
    simulation_based_calibration,
)
from .posteriors import (
    Posterior,
    exact_amplitude_posterior,
    joint_posterior,
    plugin_amplitude_posterior,
)
from .simulator import Design, Priors, degeneracy_index, design_matrix, simulate

__version__ = "0.1.0"

__all__ = [
    "CalibrationResult",
    "Design",
    "Posterior",
    "Priors",
    "coverage_curve",
    "degeneracy_index",
    "design_matrix",
    "exact_amplitude_posterior",
    "joint_posterior",
    "plugin_amplitude_posterior",
    "point_estimate_metrics",
    "run_calibration",
    "simulate",
    "simulation_based_calibration",
]
