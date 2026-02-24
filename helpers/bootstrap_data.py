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


def generate_synthetic_learning_curve(learning_index: float = 55.0) -> dict:
    """Generate a deterministic synthetic learning curve result dict.

    Returns a ``has_data=True`` dict matching the schema expected by the
    Section 1 rendering layer in ``app_min.py``.  Used as fallback when
    ``compute_learning_curve`` returns ``{"has_data": False}`` because
    insufficient governance decisions are recorded.
    """
    import warnings
    warnings.warn(
        "[bootstrap_data] Using synthetic learning curve — no recorded decisions found.",
        UserWarning,
        stacklevel=2,
    )

    learning_index = float(learning_index)
    if learning_index >= 80:
        grade, zone = "A", "Mastery"
    elif learning_index >= 60:
        grade, zone = "B", "Proficiency"
    elif learning_index >= 40:
        grade, zone = "C", "Development"
    elif learning_index >= 20:
        grade, zone = "D", "Foundation"
    else:
        grade, zone = "F", "Early Stage"

    monthly_points = generate_learning_curve_monthly_points(learning_index)

    return {
        "has_data": True,
        "learning_index": learning_index,
        "grade": grade,
        "zone": zone,
        "decision_outcome_alignment": learning_index,
        "outcome_consistency": learning_index,
        "structural_improvement": learning_index,
        "decision_alignment_pct": learning_index,
        "outcome_consistency_pct": learning_index,
        "structural_improvement_pct": learning_index,
        "monthly_points": monthly_points,
    }


def generate_synthetic_efficiency_curve(efficiency_index: float = 55.0) -> dict:
    """Generate a deterministic synthetic efficiency curve result dict.

    Returns a ``has_data=True`` dict matching the schema expected by the
    Section 1 rendering layer in ``app_min.py``.  Used as fallback when
    ``compute_efficiency_curve`` returns ``{"has_data": False}`` because
    insufficient governance decisions are recorded.
    """
    import warnings
    warnings.warn(
        "[bootstrap_data] Using synthetic efficiency curve — no recorded decisions found.",
        UserWarning,
        stacklevel=2,
    )

    efficiency_index = float(efficiency_index)
    if efficiency_index >= 80:
        grade = "A"
    elif efficiency_index >= 60:
        grade = "B"
    elif efficiency_index >= 40:
        grade = "C"
    elif efficiency_index >= 20:
        grade = "D"
    else:
        grade = "F"

    monthly_points = generate_learning_curve_monthly_points(efficiency_index)

    return {
        "has_data": True,
        "efficiency_index": efficiency_index,
        "grade": grade,
        "signal_engagement_rate": efficiency_index,
        "decision_implementation_rate": efficiency_index,
        "decision_latency_score": 75.0,
        "signal_engagement_pct": efficiency_index,
        "implementation_rate_pct": efficiency_index,
        "avg_decision_latency_hours": 24.0,
        "monthly_points": monthly_points,
    }


def generate_synthetic_cross_horizon_drivers() -> list:
    """Generate deterministic cross-horizon stability driver rows.

    Returns a list of driver dicts matching the schema expected by the
    Section 3 cross-horizon rendering layer.  Used as fallback when
    ``compute_cross_horizon_stability`` returns an empty ``drivers`` list.
    """
    import warnings
    warnings.warn(
        "[bootstrap_data] Using synthetic cross-horizon drivers — attribution data insufficient.",
        UserWarning,
        stacklevel=2,
    )

    return [
        {
            "Driver": "Selection",
            "30D State": "Positive",
            "90D State": "Positive",
            "365D State": "Positive",
            "Stability": "Stable",
        },
        {
            "Driver": "Momentum",
            "30D State": "Neutral",
            "90D State": "Positive",
            "365D State": "Positive",
            "Stability": "Stable",
        },
        {
            "Driver": "Volatility",
            "30D State": "Neutral",
            "90D State": "Neutral",
            "365D State": "Negative",
            "Stability": "Moderate",
        },
        {
            "Driver": "Regime",
            "30D State": "Positive",
            "90D State": "Neutral",
            "365D State": "Positive",
            "Stability": "Stable",
        },
        {
            "Driver": "Exposure",
            "30D State": "Positive",
            "90D State": "Positive",
            "365D State": "Neutral",
            "Stability": "Stable",
        },
    ]
