"""Run the calibration comparison and write the products.

    python3 scripts/run_demo.py --output-dir products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbical import (
    Design,
    Priors,
    degeneracy_index,
    exact_amplitude_posterior,
    plugin_amplitude_posterior,
    run_calibration,
)

NOMINAL = Design(half_width=4.0, n_points=32)
SWEEP = [2.6, 2.8, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0]

EXACT = "#2C5F8A"
PLUGIN = "#C4622D"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("products"))
    parser.add_argument("--n-simulations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    priors = Priors()
    common = {
        "design": NOMINAL, "priors": priors,
        "n_simulations": args.n_simulations, "seed": args.seed,
    }
    exact = run_calibration("exact (marginalised)", exact_amplitude_posterior, **common)
    plugin = run_calibration("plug-in (frozen baseline)", plugin_amplitude_posterior, **common)

    sweep: list[dict[str, float]] = []
    for half_width in SWEEP:
        design = Design(half_width=half_width, n_points=round(8 * half_width))
        kwargs = {"design": design, "priors": priors,
                  "n_simulations": 1500, "seed": args.seed + 1}
        e = run_calibration("e", exact_amplitude_posterior, **kwargs)
        p = run_calibration("p", plugin_amplitude_posterior, **kwargs)
        sweep.append(
            {
                "half_width": half_width,
                "n_wing_points": int(np.sum(np.abs(design.positions()) >= 2.5)),
                "degeneracy_index": degeneracy_index(design),
                "rmse_ratio": p.metrics["rmse"] / e.metrics["rmse"],
                "coverage90_exact": e.summary()["coverage_at_0.90"],
                "coverage90_plugin": p.summary()["coverage_at_0.90"],
            }
        )

    metrics = {
        "note": (
            "Synthetic study of nuisance-parameter handling. The forward model is a "
            "deliberately minimal stand-in and makes no astrophysical claim."
        ),
        "nominal_design": {
            "n_points": NOMINAL.n_points,
            "half_width": NOMINAL.half_width,
            "sigma": NOMINAL.sigma,
            "degeneracy_index": degeneracy_index(NOMINAL),
        },
        "n_simulations": args.n_simulations,
        "seed": args.seed,
        "exact": exact.summary(),
        "plugin": plugin.summary(),
        "rmse_ratio_plugin_over_exact": plugin.metrics["rmse"] / exact.metrics["rmse"],
        "sweep": sweep,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # (a) SBC rank histograms
    ax = axes[0]
    bins = np.linspace(0, exact.n_rank_bins, 20)
    for result, colour in ((exact, EXACT), (plugin, PLUGIN)):
        ax.hist(result.ranks, bins=bins, histtype="step", lw=1.8,
                color=colour, label=result.name)
    ax.axhline(exact.ranks.size / (bins.size - 1), color="0.35", ls="--", lw=1,
               label="uniform (correct)")
    ax.set_xlabel("SBC rank of the true value")
    ax.set_ylabel("count")
    ax.set_title("(a) Simulation-based calibration", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    # (b) coverage
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1, label="perfect")
    for result, colour in ((exact, EXACT), (plugin, PLUGIN)):
        ax.plot(result.nominal_levels, result.empirical_coverage, "o-",
                ms=3.5, lw=1.6, color=colour, label=result.name)
    ax.set_xlabel("stated credible level")
    ax.set_ylabel("fraction actually containing the truth")
    ax.set_title("(b) Interval coverage", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (c) the sweep: RMSE says nothing, coverage says everything
    ax = axes[2]
    hw = [s["half_width"] for s in sweep]
    ax.plot(hw, [s["rmse_ratio"] for s in sweep], "s-", ms=4, lw=1.6,
            color="0.35", label="RMSE ratio (plug-in / exact)")
    ax.plot(hw, [s["coverage90_plugin"] for s in sweep], "o-", ms=4, lw=1.6,
            color=PLUGIN, label="plug-in coverage of its 90% interval")
    ax.plot(hw, [s["coverage90_exact"] for s in sweep], "o-", ms=4, lw=1.6,
            color=EXACT, label="exact coverage of its 90% interval")
    ax.axhline(0.90, color="0.35", ls=":", lw=1)
    ax.set_xlabel("sampled half-width  (feature widths)")
    ax.set_ylabel("ratio  /  coverage")
    ax.set_title("(c) More continuum does not fix it", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(args.output_dir / "calibration.png", dpi=150)

    print(json.dumps({k: metrics[k] for k in
                      ("exact", "plugin", "rmse_ratio_plugin_over_exact")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
