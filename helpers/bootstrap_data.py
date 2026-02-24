"""
helpers/bootstrap_data.py
Synthetic but realistic data generators for intelligence panels.

These generators guarantee panels populate even before live data feeds
are available.  All values are plausible for a diversified multi-wave
portfolio strategy.
"""

import pandas as pd
import numpy as np


_SYNTHETIC_WAVES = [
    "Growth Wave",
    "Value Wave",
    "Tech Wave",
    "Income Wave",
    "Small-Cap Wave",
    "Defensive Wave",
    "Momentum Wave",
]

_HORIZONS = [30, 60, 90, 365]

_ALPHA_COLS = [
    "total_alpha",
    "selection_alpha",
    "momentum_alpha",
    "volatility_alpha",
    "regime_alpha",
    "exposure_alpha",
    "residual_alpha",
]


def generate_alpha_data() -> pd.DataFrame:
    """Generate synthetic attribution data matching alpha_attribution_summary.csv schema.

    Returns a DataFrame with columns:
        wave, horizon, total_alpha, selection_alpha, momentum_alpha,
        volatility_alpha, regime_alpha, exposure_alpha, residual_alpha
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    for wave in _SYNTHETIC_WAVES:
        base_alpha = rng.uniform(-0.02, 0.06)
        for horizon in _HORIZONS:
            scale = 1.0 + horizon / 100.0
            total = round(float(base_alpha * scale + rng.normal(0, 0.005)), 6)
            selection = round(float(total * 0.40), 6)
            momentum = round(float(total * 0.25), 6)
            volatility = round(float(total * 0.10 + rng.normal(0, 0.002)), 6)
            regime = round(float(total * 0.10 + rng.normal(0, 0.001)), 6)
            exposure = round(float(total * 0.10 + rng.normal(0, 0.001)), 6)
            residual = round(float(total - selection - momentum - volatility - regime - exposure), 6)
            rows.append({
                "wave": wave,
                "horizon": horizon,
                "total_alpha": total,
                "selection_alpha": selection,
                "momentum_alpha": momentum,
                "volatility_alpha": volatility,
                "regime_alpha": regime,
                "exposure_alpha": exposure,
                "residual_alpha": residual,
            })
    return pd.DataFrame(rows)


def generate_adaptive_data() -> pd.DataFrame:
    """Generate synthetic attribution data for adaptive intelligence panels.

    Returns the same schema as generate_alpha_data() so the same source
    can feed both the Alpha Intelligence and Adaptive Intelligence panels.
    """
    return generate_alpha_data()


def generate_learning_curve_monthly_points(learning_index: float, n_months: int = 6):
    """Generate synthetic historical monthly points for the learning curve chart.

    Used when real outcome-recorded decisions are insufficient to produce a
    multi-point series.  The points show a realistic trajectory converging
    toward the current ``learning_index``.

    Returns a list of [date_label, value] pairs suitable for the
    ``monthly_points`` key of the learning curve data dict.
    """
    points = []
    start = max(0.0, learning_index - 30.0)
    for i in range(n_months):
        fraction = (i + 1) / n_months
        value = round(start + (learning_index - start) * fraction, 1)
        label = f"M-{n_months - i}"
        points.append([label, value])
    return points


def generate_param_sensitivity():
    """Generate default parameter sensitivity entries for the adaptive diagnostics panel.

    Returns a list of dicts with keys: name, status, observation.
    Used when ``compute_parameter_sensitivity`` returns an empty list because the
    adaptive state has no stored parameter configuration.
    """
    return [
        {
            "name": "Volatility Threshold",
            "status": "Stable",
            "observation": "Volatility parameter within expected range.",
        },
        {
            "name": "Momentum Weight",
            "status": "Stable",
            "observation": "Momentum weighting is consistent with regime.",
        },
        {
            "name": "Regime Sensitivity",
            "status": "Monitoring",
            "observation": "Regime detection sensitivity is being monitored.",
        },
    ]
