"""
bootstrap_data.py
Deterministic synthetic bootstrap datasets for the seven intelligence panels.

When live data is unavailable, these generators provide stable, non-empty
DataFrames and data structures so panels render immediately on every run.

All functions use a fixed seed (derived from a constant, never from the
current time) so output is fully deterministic across reloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEED = 42

_WAVE_NAMES = [
    "AI & Cloud MegaCap Wave",
    "Clean Transit-Infrastructure Wave",
    "Crypto Broad Growth Wave",
    "Emerging Market Growth Wave",
    "Healthcare Innovation Wave",
    "Income & Dividend Wave",
    "Quantum Computing Wave",
]

_DRIVERS = [
    "Selection",
    "Momentum",
    "Volatility",
    "Regime",
    "Exposure",
]

_HORIZONS = ["30D", "60D", "90D", "365D"]


def bootstrap_alpha_quality_ranking() -> pd.DataFrame:
    """Return a deterministic synthetic Alpha Quality Ranking DataFrame.

    Columns
    -------
    wave, alpha_quality_score, consistency,
    total_alpha_30D, total_alpha_60D, total_alpha_90D, total_alpha_365D
    """
    rng = np.random.default_rng(_SEED)
    rows = []
    for wave in _WAVE_NAMES:
        alphas = {h: float(rng.uniform(-0.05, 0.15)) for h in _HORIZONS}
        consistency = round(sum(1 for v in alphas.values() if v > 0) / len(alphas), 2)
        mean_abs = float(np.mean([abs(v) for v in alphas.values()]))
        score = round(mean_abs * consistency, 6)
        rows.append(
            {
                "wave": wave,
                "alpha_quality_score": score,
                "consistency": consistency,
                "total_alpha_30D": round(alphas["30D"], 6),
                "total_alpha_60D": round(alphas["60D"], 6),
                "total_alpha_90D": round(alphas["90D"], 6),
                "total_alpha_365D": round(alphas["365D"], 6),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("alpha_quality_score", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_capital_pressure_regime() -> pd.DataFrame:
    """Return a deterministic synthetic Capital Pressure Regime DataFrame.

    Columns
    -------
    wave, capital_pressure_score, regime_bin,
    residual_alpha_30D, regime_alpha_30D
    """
    rng = np.random.default_rng(_SEED + 1)
    rows = []
    for wave in _WAVE_NAMES:
        residual = float(rng.uniform(-0.01, 0.02))
        regime_a = float(rng.uniform(-0.01, 0.015))
        score = round(residual + abs(regime_a), 6)
        regime_bin = "High" if score > 0.005 else ("Low" if score < -0.005 else "Neutral")
        rows.append(
            {
                "wave": wave,
                "capital_pressure_score": score,
                "regime_bin": regime_bin,
                "residual_alpha_30D": round(residual, 6),
                "regime_alpha_30D": round(regime_a, 6),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("capital_pressure_score", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_rotation_velocity() -> pd.DataFrame:
    """Return a deterministic synthetic Rotation Velocity DataFrame.

    Columns
    -------
    wave, total_alpha_30D, total_alpha_365D, rotation_velocity, direction
    """
    rng = np.random.default_rng(_SEED + 2)
    rows = []
    for wave in _WAVE_NAMES:
        a30 = float(rng.uniform(-0.05, 0.10))
        a365 = float(rng.uniform(-0.05, 0.10))
        velocity = round(a30 - a365, 6)
        direction = "Accelerating" if velocity >= 0 else "Decelerating"
        rows.append(
            {
                "wave": wave,
                "total_alpha_30D": round(a30, 6),
                "total_alpha_365D": round(a365, 6),
                "rotation_velocity": velocity,
                "direction": direction,
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_velocity=lambda df: df["rotation_velocity"].abs())
        .sort_values("abs_velocity", ascending=False)
        .drop(columns=["abs_velocity"])
        .reset_index(drop=True)
    )


def bootstrap_alpha_ignition_surface() -> pd.DataFrame:
    """Return a deterministic synthetic Alpha Ignition Surface DataFrame.

    Columns
    -------
    wave, horizon, ignition_score, selection_alpha, momentum_alpha,
    volatility_alpha
    """
    rng = np.random.default_rng(_SEED + 3)
    rows = []
    for wave in _WAVE_NAMES:
        for horizon in _HORIZONS:
            sel = float(rng.uniform(-0.02, 0.05))
            mom = float(rng.uniform(-0.02, 0.04))
            vol = float(rng.uniform(-0.015, 0.03))
            score = round(0.5 * sel + 0.3 * mom - 0.2 * abs(vol), 6)
            rows.append(
                {
                    "wave": wave,
                    "horizon": horizon,
                    "ignition_score": score,
                    "selection_alpha": round(sel, 6),
                    "momentum_alpha": round(mom, 6),
                    "volatility_alpha": round(vol, 6),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("ignition_score", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_cross_horizon_stability() -> dict:
    """Return a deterministic synthetic cross_horizon_data dict.

    The returned dict has the structure expected by the
    Cross-Horizon & Attribution Stability panel (Section 3) of the
    Adaptive Intelligence tab.

    Keys
    ----
    drivers : list[dict]
        Each entry has ``Driver``, ``30D State``, ``90D State``,
        ``365D State``, ``Stability``.
    summary : str
        One-line summary sentence.
    """
    rng = np.random.default_rng(_SEED + 4)
    states = ["Positive", "Neutral", "Under Pressure"]
    stability_options = ["Stable", "Mixed", "Unstable"]
    drivers = []
    for drv in _DRIVERS:
        s30 = states[int(rng.integers(0, len(states)))]
        s90 = states[int(rng.integers(0, len(states)))]
        s365 = states[int(rng.integers(0, len(states)))]
        stab = stability_options[int(rng.integers(0, len(stability_options)))]
        drivers.append(
            {
                "Driver": drv,
                "30D State": s30,
                "90D State": s90,
                "365D State": s365,
                "Stability": stab,
            }
        )
    stable_count = sum(1 for d in drivers if d["Stability"] == "Stable")
    summary = (
        f"Bootstrap data: {stable_count}/{len(drivers)} attribution drivers "
        "show stable cross-horizon behavior. Live data will replace this when available."
    )
    return {"drivers": drivers, "summary": summary}


def bootstrap_learning_diagnostics() -> tuple[dict, dict]:
    """Return deterministic synthetic learning_curve_data and efficiency_curve_data.

    The returned dicts have the structure expected by the System Learning &
    Efficiency Panel (Section 1) of the Adaptive Intelligence tab.

    Returns
    -------
    learning_curve_data : dict
    efficiency_curve_data : dict
    """
    rng = np.random.default_rng(_SEED + 5)

    # Build 12 synthetic monthly points
    base_li = float(rng.uniform(40, 70))
    lc_points = []
    for i in range(12):
        month_label = f"2025-{i + 1:02d}"
        value = round(base_li + float(rng.uniform(-5, 8)), 1)
        lc_points.append([month_label, value])

    base_ei = float(rng.uniform(45, 72))
    ec_points = []
    for i in range(12):
        month_label = f"2025-{i + 1:02d}"
        value = round(base_ei + float(rng.uniform(-4, 7)), 1)
        ec_points.append([month_label, value])

    li_final = lc_points[-1][1]
    ei_final = ec_points[-1][1]

    def _grade(score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 65:
            return "B"
        if score >= 50:
            return "C"
        if score >= 35:
            return "D"
        return "F"

    def _zone(score: float) -> str:
        if score >= 70:
            return "Learning"
        if score >= 45:
            return "Developing"
        return "Early Stage"

    learning_curve_data: dict = {
        "has_data": True,
        "learning_index": round(li_final, 1),
        "grade": _grade(li_final),
        "zone": _zone(li_final),
        "monthly_points": lc_points,
        "decision_alignment_pct": round(float(rng.uniform(50, 85)), 1),
        "outcome_consistency_pct": round(float(rng.uniform(45, 80)), 1),
        "structural_improvement_pct": round(float(rng.uniform(40, 75)), 1),
    }

    efficiency_curve_data: dict = {
        "has_data": True,
        "efficiency_index": round(ei_final, 1),
        "grade": _grade(ei_final),
        "monthly_points": ec_points,
        "signal_engagement_pct": round(float(rng.uniform(55, 88)), 1),
        "implementation_rate_pct": round(float(rng.uniform(50, 82)), 1),
        "avg_decision_latency_hours": round(float(rng.uniform(2, 24)), 1),
    }

    return learning_curve_data, efficiency_curve_data


def bootstrap_adaptive_regime_diagnostics() -> list[dict]:
    """Return a deterministic synthetic param_sensitivity list.

    Each entry has ``name`` and ``status`` as expected by the Adaptive
    Threshold & Confidence Calibration panel (Section 5).
    """
    rng = np.random.default_rng(_SEED + 6)
    components = [
        "Volatility Threshold",
        "Momentum Weight",
        "Regime Sensitivity",
        "Confidence Decay",
        "Learning Rate",
    ]
    statuses = ["Stable", "Monitoring", "Review"]
    params = []
    for comp in components:
        status = statuses[int(rng.integers(0, len(statuses)))]
        params.append({"name": comp, "status": status})
    return params
