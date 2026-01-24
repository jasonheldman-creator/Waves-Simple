"""
alpha_beta.py

Institutional-grade alpha / beta attribution utilities.

This module is intentionally:
- Import-safe
- Streamlit-agnostic
- Side-effect free

It computes rolling alpha/beta statistics using OLS regression:
    wave_return = alpha + beta * benchmark_return + residual

Outputs are structured DataFrames suitable for:
- Wave Health
- WaveScore
- Attribution dashboards
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, List, Optional


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------
# Core regression logic
# ---------------------------------------------------------

def _ols_alpha_beta(
    wave_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """
    Compute alpha, beta, R^2, and tracking error using OLS.

    Parameters
    ----------
    wave_returns : pd.Series
        Return series for the wave
    benchmark_returns : pd.Series
        Return series for the benchmark

    Returns
    -------
    dict
        alpha, beta, r2, tracking_error
    """

    df = pd.concat(
        [wave_returns, benchmark_returns],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) < 10:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r2": np.nan,
            "tracking_error": np.nan,
        }

    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    x_mean = x.mean()
    y_mean = y.mean()

    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.mean((x - x_mean) ** 2)

    if var_x == 0:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r2": np.nan,
            "tracking_error": np.nan,
        }

    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean

    y_hat = alpha + beta * x
    residuals = y - y_hat

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    tracking_error = np.std(residuals, ddof=1)

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "tracking_error": tracking_error,
    }


# ---------------------------------------------------------
# Rolling attribution
# ---------------------------------------------------------

def compute_rolling_alpha_beta(
    wave_returns: pd.Series,
    benchmark_returns: pd.Series,
    windows: Iterable[int] = (30, 60, 90),
    annualize_alpha: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling alpha/beta attribution over multiple windows.

    Parameters
    ----------
    wave_returns : pd.Series
        Daily return series for the wave
    benchmark_returns : pd.Series
        Daily return series for the benchmark
    windows : iterable of int
        Rolling window lengths in trading days
    annualize_alpha : bool
        Whether to annualize alpha

    Returns
    -------
    pd.DataFrame
        Columns: window, alpha, beta, r2, tracking_error
    """

    results: List[Dict[str, float]] = []

    for window in windows:
        wave_slice = wave_returns.tail(window)
        bench_slice = benchmark_returns.tail(window)

        stats = _ols_alpha_beta(wave_slice, bench_slice)

        alpha = stats["alpha"]
        if annualize_alpha and not pd.isna(alpha):
            alpha = alpha * (TRADING_DAYS_PER_YEAR / window)

        results.append({
            "window": window,
            "alpha": alpha,
            "beta": stats["beta"],
            "r2": stats["r2"],
            "tracking_error": stats["tracking_error"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Multi-wave convenience wrapper
# ---------------------------------------------------------

def compute_alpha_beta_for_waves(
    wave_returns_df: pd.DataFrame,
    benchmark_returns_df: pd.DataFrame,
    windows: Iterable[int] = (30, 60, 90),
) -> pd.DataFrame:
    """
    Compute alpha/beta attribution for multiple waves.

    Parameters
    ----------
    wave_returns_df : pd.DataFrame
        Columns = wave_id, index = date, values = returns
    benchmark_returns_df : pd.DataFrame
        Columns = wave_id, index = date, values = benchmark returns
    windows : iterable of int
        Rolling windows in trading days

    Returns
    -------
    pd.DataFrame
        Indexed by wave_id and window
    """

    records: List[Dict[str, object]] = []

    common_waves = set(wave_returns_df.columns).intersection(
        benchmark_returns_df.columns
    )

    for wave_id in sorted(common_waves):
        wave_series = wave_returns_df[wave_id]
        bench_series = benchmark_returns_df[wave_id]

        df = compute_rolling_alpha_beta(
            wave_series,
            bench_series,
            windows=windows,
        )

        for _, row in df.iterrows():
            records.append({
                "wave_id": wave_id,
                "window": int(row["window"]),
                "alpha": row["alpha"],
                "beta": row["beta"],
                "r2": row["r2"],
                "tracking_error": row["tracking_error"],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Module self-check (safe)
# ---------------------------------------------------------

def _module_healthcheck() -> str:
    return "alpha_beta attribution module loaded safely"


if __name__ == "__main__":
    print(_module_healthcheck())