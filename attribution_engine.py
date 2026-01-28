# ============================================================
# attribution_engine.py
# Core alpha attribution engine
# ============================================================

import numpy as np
import pandas as pd

from .alpha_components import (
    compute_beta_alpha,
    compute_momentum_alpha,
    compute_volatility_alpha,
    compute_allocation_alpha,
    compute_total_alpha,
)


def compute_horizon_attribution(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
    days: int,
) -> dict:
    """
    Compute horizon-specific alpha attribution components.

    Momentum and volatility are ALWAYS computed for all horizons,
    including days=252 (365D), with full lookback = days.
    """

    wave_h = wave_series.tail(days)
    bench_h = benchmark_series.tail(days)

    if len(wave_h) < 2 or len(bench_h) < 2:
        return {
            "beta": 0.0,
            "momentum": 0.0,
            "volatility": 0.0,
            "allocation": 0.0,
            "residual": 0.0,
        }

    beta_alpha = compute_beta_alpha(wave_h, bench_h, engine_weights)

    momentum_alpha = compute_momentum_alpha(
        wave_h,
        bench_h,
        engine_weights,
        lookback=days,
    )

    volatility_alpha = compute_volatility_alpha(
        wave_h,
        bench_h,
        engine_weights,
        lookback=days,
    )

    allocation_alpha = compute_allocation_alpha(
        wave_h,
        bench_h,
        engine_weights,
        lookback=days,
    )

    total_alpha = compute_total_alpha(wave_h, bench_h)

    residual_alpha = (
        total_alpha
        - beta_alpha
        - momentum_alpha
        - volatility_alpha
        - allocation_alpha
    )

    return {
        "beta": beta_alpha,
        "momentum": momentum_alpha,
        "volatility": volatility_alpha,
        "allocation": allocation_alpha,
        "residual": residual_alpha,
    }