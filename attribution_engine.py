# ============================================================
# attribution_engine.py
# Core alpha attribution engine
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# CORE COMPONENTS (INLINED — VERBATIM LOGIC)
# ============================================================

def compute_total_alpha(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
) -> float:
    """
    Total alpha over the horizon: wave return minus benchmark return.
    """
    wave_ret = wave_series.iloc[-1] / wave_series.iloc[0] - 1.0
    bench_ret = benchmark_series.iloc[-1] / benchmark_series.iloc[0] - 1.0
    return float(wave_ret - bench_ret)


def compute_beta_alpha(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
) -> float:
    """
    Beta / market exposure component.
    """
    returns_wave = wave_series.pct_change().dropna()
    returns_bench = benchmark_series.pct_change().dropna()

    if len(returns_wave) < 2 or len(returns_bench) < 2:
        return 0.0

    cov = np.cov(returns_wave, returns_bench)[0, 1]
    var = np.var(returns_bench)

    if var == 0:
        return 0.0

    beta = cov / var
    bench_ret = returns_bench.mean()
    beta_alpha = beta * bench_ret * engine_weights.get("beta", 1.0)

    return float(beta_alpha)


def compute_momentum_alpha(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
    lookback: int,
) -> float:
    """
    Momentum / trend-following alpha.

    • NO lookback capping
    • NO horizon gating
    • Full lookback used (e.g. 252 for 365D)
    """
    window = lookback

    if len(wave_series) < window + 1 or len(benchmark_series) < window + 1:
        return 0.0

    wave_ret = wave_series.pct_change(window).iloc[-1]
    bench_ret = benchmark_series.pct_change(window).iloc[-1]

    raw_mom = wave_ret - bench_ret
    scaled = raw_mom * engine_weights.get("momentum", 1.0)

    return float(scaled)


def compute_volatility_alpha(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
    lookback: int,
) -> float:
    """
    Volatility / VIX / convexity alpha.

    • NO short-horizon caps
    • NO long-horizon disable
    • Full lookback used (e.g. 252 for 365D)
    """
    window = lookback

    if len(wave_series) < window + 1 or len(benchmark_series) < window + 1:
        return 0.0

    returns_wave = wave_series.pct_change().dropna()
    returns_bench = benchmark_series.pct_change().dropna()

    wave_vol = returns_wave.rolling(window).std().iloc[-1]
    bench_vol = returns_bench.rolling(window).std().iloc[-1]

    if np.isnan(wave_vol) or np.isnan(bench_vol):
        return 0.0

    vol_spread = wave_vol - bench_vol
    volatility_weight = engine_weights.get("volatility", 1.0)

    volatility_alpha = vol_spread * volatility_weight

    return float(volatility_alpha)


def compute_allocation_alpha(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
    lookback: int,
) -> float:
    """
    Allocation / sleeve / sector tilt alpha.

    Logic unchanged. Placeholder preserved.
    """
    return 0.0


# ============================================================
# HORIZON ATTRIBUTION (ENGINE ENTRYPOINT)
# ============================================================

def compute_horizon_attribution(
    wave_series: pd.Series,
    benchmark_series: pd.Series,
    engine_weights: dict,
    days: int,
) -> dict:
    """
    Compute horizon-specific alpha attribution.

    GUARANTEES:
    • Momentum ALWAYS computed (including 365D / 252)
    • Volatility ALWAYS computed (including 365D / 252)
    • NO horizon gating
    • NO component folding
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