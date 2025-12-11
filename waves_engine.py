"""
waves_engine.py — WAVES Intelligence™ Stage 10 Engine

Features
--------
• Loads wave_weights.csv (wave, ticker, weight) and discovers all Waves.
• Stage 10 logic:
    - Wave-specific benchmark blends (ETF baskets).
    - Momentum-tilted weights (6-month trend).
    - Quarterly rebalancing simulation (drift + rebalance).
    - Estimated annual turnover from those rebalances.
    - Beta discipline vs each Wave's benchmark.
    - SmartSafe 2.0 (VIX-based base sweep to BIL).
    - SmartSafe 3.0 (Wave-specific extra sweep based on stress / panic).
• Computes intraday (placeholder), 30D, 60D, 1Y, and since-inception returns
  and alpha vs each Wave's benchmark.
• Computes 1Y volatility, max drawdown, info ratio, hit rate, beta, and
  turnover / execution telemetry.
• Returns a snapshot dict per Wave compatible with app.py:
    {
        "benchmark": "<human readable blend>",
        "metrics": { ... },
        "positions": pd.DataFrame([...])
    }
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None
    _YF_ERROR = str(e)
else:
    _YF_ERROR = None

# ---------------------------------------------------------------------
# Wave configuration: benchmarks + Stage 10 parameters
# ---------------------------------------------------------------------


@dataclass
class WaveConfig:
    benchmark: Dict[str, float]  # ETF -> weight
    beta_target: float = 1.0
    momentum_tilt_strength: float = 0.5  # 0 = none, 1 = strong
    max_weight: float = 0.12
    smartsafe_factor: float = 1.0  # extra sweep amplification for 3.0


# Explicit configs for key Waves. Any Wave not listed falls back to
# heuristic defaults based on its name.
WAVE_CONFIGS: Dict[str, WaveConfig] = {
    # Core equity
    "S&P Wave": WaveConfig(
        benchmark={"SPY": 1.0},
        beta_target=0.95,
        momentum_tilt_strength=0.3,
        max_weight=0.10,
        smartsafe_factor=0.6,
    ),
    "Income Wave": WaveConfig(
        benchmark={"SCHD": 0.6, "BND": 0.4},
        beta_target=0.85,
        momentum_tilt_strength=0.25,
        max_weight=0.08,
        smartsafe_factor=0.7,
    ),
    # Growth / factor Waves
    "Growth Wave": WaveConfig(
        benchmark={"QQQ": 0.5, "SPY": 0.5},
        beta_target=1.1,
        momentum_tilt_strength=0.7,
        max_weight=0.12,
        smartsafe_factor=1.0,
    ),
    "Small Cap Growth Wave": WaveConfig(
        # Small caps vs realistic benchmark
        benchmark={"IWM": 0.6, "QQQ": 0.2, "ARKK": 0.2},
        beta_target=1.2,
        momentum_tilt_strength=0.8,
        max_weight=0.10,
        smartsafe_factor=1.3,
    ),
    "Small to Mid Cap Growth Wave": WaveConfig(
        benchmark={"IWM": 0.5, "QQQ": 0.3, "SPY": 0.2},
        beta_target=1.15,
        momentum_tilt_strength=0.7,
        max_weight=0.10,
        smartsafe_factor=1.1,
    ),
    # AI / Tech Waves
    "AI Wave": WaveConfig(
        # 40% SMH + 30% IGV + 30% AIQ
        benchmark={"SMH": 0.4, "IGV": 0.3, "AIQ": 0.3},
        beta_target=1.15,
        momentum_tilt_strength=0.9,
        max_weight=0.12,
        smartsafe_factor=1.1,
    ),
    "Cloud & Software Wave": WaveConfig(
        # SaaS-aligned benchmark
        benchmark={"WCLD": 0.6, "QQQ": 0.4},
        beta_target=1.1,
        momentum_tilt_strength=0.8,
        max_weight=0.12,
        smartsafe_factor=1.0,
    ),
    "Quantum Computing Wave": WaveConfig(
        benchmark={"QQQ": 0.4, "ARKK": 0.3, "SOXX": 0.3},
        beta_target=1.25,
        momentum_tilt_strength=0.9,
        max_weight=0.10,
        smartsafe_factor=1.4,
    ),
    "Future Power & Energy Wave": WaveConfig(
        benchmark={"ICLN": 0.4, "XLE": 0.3, "QQQ": 0.3},
        beta_target=1.1,
        momentum_tilt_strength=0.8,
        max_weight=0.12,
        smartsafe_factor=1.1,
    ),
    "Clean Transit-Infrastructure Wave": WaveConfig(
        benchmark={"SPY": 0.4, "QQQ": 0.4, "IWM": 0.2},
        beta_target=1.05,
        momentum_tilt_strength=0.7,
        max_weight=0.12,
        smartsafe_factor=1.0,
    ),
    # Crypto Waves
    "Crypto Equity Wave (mid/large cap)": WaveConfig(
        # 50/30/20 WGMI/BLOK/BITQ blend
        benchmark={"WGMI": 0.5, "BLOK": 0.3, "BITQ": 0.2},
        beta_target=1.5,
        momentum_tilt_strength=0.9,
        max_weight=0.15,
        smartsafe_factor=1.6,
    ),
    "Crypto Income Wave": WaveConfig(
        benchmark={"BITO": 0.5, "BLOK": 0.3, "BITQ": 0.2},
        beta_target=1.2,
        momentum_tilt_strength=0.6,
        max_weight=0.12,
        smartsafe_factor=1.5,
    ),
    # SmartSafe / cash Wave
    "SmartSafe Wave": WaveConfig(
        benchmark={"BIL": 1.0},
        beta_target=0.05,
        momentum_tilt_strength=0.0,
        max_weight=1.0,
        smartsafe_factor=0.0,
    ),
}


def _heuristic_benchmark_for_wave(wave: str) -> Dict[str, float]:
    """Fallback benchmark chooser if a Wave is not explicitly configured."""
    name = wave.lower()
    if "s&p" in name or "sp " in name:
        return {"SPY": 1.0}
    if "income" in name or "dividend" in name:
        return {"SCHD": 0.6, "BND": 0.4}
    if "small" in name:
        return {"IWM": 0.7, "QQQ": 0.3}
    if "crypto" in name:
        return {"BITO": 0.6, "BLOK": 0.4}
    if "ai" in name or "cloud" in name or "software" in name:
        return {"QQQ": 0.7, "SMH": 0.3}
    if "power" in name or "energy" in name:
        return {"XLE": 0.6, "ICLN": 0.4}
    if "quantum" in name:
        return {"QQQ": 0.6, "SOXX": 0.4}
    return {"SPY": 1.0}


def get_wave_config(wave: str) -> WaveConfig:
    if wave in WAVE_CONFIGS:
        return WAVE_CONFIGS[wave]
    bench = _heuristic_benchmark_for_wave(wave)
    return WaveConfig(
        benchmark=bench,
        beta_target=1.0,
        momentum_tilt_strength=0.5,
        max_weight=0.12,
        smartsafe_factor=1.0,
    )


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------


def _load_wave_weights(csv_path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load wave_weights.csv and normalize column names.

    Handles:
    • Hidden BOM characters (e.g. '﻿wave')
    • Extra spaces
    • Upper/lower case
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"wave_weights.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean column names: strip BOM, whitespace, lowercase
    clean_map = {}
    for col in df.columns:
        c = col.replace("\ufeff", "").strip().lower()
        clean_map[col] = c
    df = df.rename(columns=clean_map)

    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing columns: {missing}")

    # Clean up values
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = df["weight"].astype(float)

    # Normalize weights per wave
    for wave, sub in df.groupby("wave"):
        total = sub["weight"].sum()
        if total <= 0:
            continue
        df.loc[sub.index, "weight"] = sub["weight"] / total

    return df


def get_available_waves(csv_path: str = "wave_weights.csv") -> List[str]:
    df = _load_wave_weights(csv_path)
    waves = sorted(df["wave"].unique().tolist())
    return waves


# Price cache
_PRICE_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def _load_price_history(
    tickers: List[str],
    period: str = "3y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load adjusted close prices for all tickers over the given period.
    Returns a DataFrame with date index and ticker columns.
    """
    if _YF_ERROR is not None:
        raise RuntimeError(f"yfinance is not available: {_YF_ERROR}")

    tickers = sorted(set(tickers))
    key = (",".join(tickers), period)
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key]

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = {}
        for t in tickers:
            try:
                close[t] = data[(t, "Close")]
            except Exception:
                continue
        prices = pd.DataFrame(close)
    else:
        # Single ticker case
        prices = pd.DataFrame({"Close": data["Close"]})
        prices.columns = tickers

    prices = prices.dropna(how="all")
    _PRICE_CACHE[key] = prices
    return prices


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


def _series_return(s: pd.Series) -> float:
    return float((1.0 + s).prod() - 1.0)


def _max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    return float(dd.min())


def _beta_and_alpha(port: pd.Series, bench: pd.Series) -> Tuple[float, float]:
    common = port.dropna().index.intersection(bench.dropna().index)
    if len(common) < 10:
        return 0.0, 0.0
    rp = port.loc[common]
    rb = bench.loc[common]
    cov = np.cov(rp, rb)[0, 1]
    var_b = np.var(rb)
    if var_b <= 0:
        return 0.0, 0.0
    beta = cov / var_b
    alpha_daily = (rp - beta * rb).mean()
    alpha_annual = float(alpha_daily * 252.0)
    return float(beta), alpha_annual


def _info_ratio(port: pd.Series, bench: pd.Series) -> float:
    common = port.dropna().index.intersection(bench.dropna().index)
    if len(common) < 10:
        return 0.0
    diff = port.loc[common] - bench.loc[common]
    mean_excess = diff.mean()
    vol_excess = diff.std()
    if vol_excess <= 0:
        return 0.0
    return float((mean_excess * 252.0) / (vol_excess * np.sqrt(252.0)))


def _hit_rate(port: pd.Series, bench: pd.Series) -> float:
    common = port.dropna().index.intersection(bench.dropna().index)
    if len(common) < 10:
        return 0.0
    rp = port.loc[common]
    rb = bench.loc[common]
    hits = (rp > rb).sum()
    return float(hits / len(common))


# ---------------------------------------------------------------------
# SmartSafe logic (2.0 + 3.0)
# ---------------------------------------------------------------------


def _load_vix(period: str = "3y") -> pd.Series:
    vix_prices = _load_price_history(["^VIX"], period=period)
    col = vix_prices.columns[0]
    return vix_prices[col]


def _smartsafe_regime(
    vix_level: float,
    wave_dd_60d: float,
    bench_dd_60d: float,
) -> str:
    """
    Determine SmartSafe 3.0 regime from VIX + drawdowns.
    """
    if np.isnan(vix_level):
        vix_level = 18.0

    extra_stress = 0.0
    if wave_dd_60d < bench_dd_60d - 0.05:
        extra_stress = 1.0

    if vix_level < 18 and wave_dd_60d > -0.10:
        return "Normal"
    if (18 <= vix_level < 24) or (wave_dd_60d <= -0.10 - 0.05 * extra_stress):
        return "Caution"
    if (24 <= vix_level < 30) or (wave_dd_60d <= -0.20 - 0.05 * extra_stress):
        return "Stress"
    if vix_level >= 30 or wave_dd_60d <= -0.35:
        return "Panic"
    return "Normal"


_SMARTSAFE_BASE_SWEEP = {
    "Normal": 0.00,
    "Caution": 0.10,
    "Stress": 0.20,
    "Panic": 0.40,
}


# ---------------------------------------------------------------------
# Stage 10: core computation (with drift + quarterly rebalance)
# ---------------------------------------------------------------------


def _compute_wave_series(
    wave: str,
    mode: str,
    weights_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Build the Wave time series and metrics in Stage 10 style.
    Returns dict with keys: benchmark, metrics, positions
    """

    cfg = get_wave_config(wave)

    # Tickers & base weights
    sub = weights_df[weights_df["wave"] == wave].copy()
    if sub.empty:
        raise ValueError(f"No positions for Wave '{wave}' in wave_weights.csv")

    # Mode-aware beta target tweaks (simple)
    mode_lower = (mode or "standard").lower()
    beta_target = cfg.beta_target
    if "amb" in mode_lower:   # Alpha-Minus-Beta
        beta_target = cfg.beta_target * 0.85
    elif "pl" in mode_lower:  # Private Logic
        beta_target = cfg.beta_target * 1.10

    tickers = sub["ticker"].tolist()
    base_weights = sub["weight"].values

    # Benchmark ETFs
    bench_map = cfg.benchmark or _heuristic_benchmark_for_wave(wave)
    bench_tickers = list(bench_map.keys())

    # Price history
    all_tickers = sorted(set(tickers + bench_tickers + ["BIL"]))
    prices = _load_price_history(all_tickers, period="3y")
    returns = prices.pct_change().dropna(how="all")

    # Align base_weights with available tickers
    valid_mask = sub["ticker"].isin(returns.columns)
    sub = sub[valid_mask]
    sub = sub.copy()
    sub["weight"] = sub["weight"] / sub["weight"].sum()
    tickers = sub["ticker"].tolist()
    base_weights = sub["weight"].values

    # 6-month momentum for Stage 10 momentum tilts
    lookback_mom = 126  # ~6 months
    mom = {}
    for t in tickers:
        s = prices[t].dropna()
        if len(s) < lookback_mom + 5:
            mom[t] = 0.0
        else:
            r = s.pct_change().dropna()
            m = _series_return(r[-lookback_mom:])
            mom[t] = m

    if len(mom) == 0:
        mom_scores = np.ones_like(base_weights)
    else:
        mom_series = pd.Series(mom)
        if mom_series.std() > 0:
            z = (mom_series - mom_series.mean()) / (mom_series.std() + 1e-9)
        else:
            z = mom_series * 0.0
        z = z.clip(-2.0, 2.0) / 2.0  # [-1, +1]
        mom_scores = z.reindex(tickers).fillna(0.0).values

    tilt_strength = cfg.momentum_tilt_strength
    adj_weights = base_weights * (1.0 + tilt_strength * mom_scores)
    adj_weights = np.maximum(adj_weights, 0.0)
    if adj_weights.sum() == 0:
        adj_weights = base_weights
    adj_weights = adj_weights / adj_weights.sum()

    # Enforce max_weight cap
    cap = cfg.max_weight
    if cap is not None and cap > 0:
        w = adj_weights.copy()
        for _ in range(3):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if under.any():
                w[under] += excess * (w[under] / w[under].sum())
        adj_weights = w / w.sum()

    # Stage 10: simulate daily drift + quarterly rebalancing
    rebalance_every = 63  # ~63 trading days ≈ quarterly
    dates = returns.index
    port_ret_list: List[float] = []
    turnover_list: List[float] = []

    # initialise weights
    w = adj_weights.copy()
    last_rebalance_w = w.copy()

    ret_matrix = returns[tickers].fillna(0.0).values

    for i, date in enumerate(dates):
        r_vec = ret_matrix[i, :]
        # portfolio return from current weights
        port_ret = float(np.dot(w, r_vec))
        port_ret_list.append(port_ret)

        # update weights after returns (drift)
        w = w * (1.0 + r_vec)
        if w.sum() <= 0:
            w = adj_weights.copy()
        else:
            w = w / w.sum()

        # rebalance periodically back to target adj_weights
        if (i + 1) % rebalance_every == 0:
            # turnover is 0.5 * sum |w_new - w_old|
            turnover = 0.5 * float(np.abs(adj_weights - w).sum())
            turnover_list.append(turnover)
            w = adj_weights.copy()
            last_rebalance_w = w.copy()

    port_returns = pd.Series(port_ret_list, index=dates)

    # Benchmark series (static blend of ETFs)
    bench_returns = pd.Series(0.0, index=dates)
    for t, w_b in bench_map.items():
        if t not in returns.columns:
            extra = _load_price_history([t], period="3y").pct_change().dropna()
            if t in extra.columns:
                returns[t] = extra[t]
        if t in returns.columns:
            bench_returns = bench_returns.add(w_b * returns[t].fillna(0.0), fill_value=0.0)

    # Clip to common period
    common_idx = port_returns.dropna().index.intersection(bench_returns.dropna().index)
    if len(common_idx) < 60:
        common_idx = port_returns.dropna().index

    port_returns = port_returns.loc[common_idx].fillna(0.0)
    bench_returns = bench_returns.loc[common_idx].fillna(0.0)

    # Cumulative for drawdowns / SI
    port_cum = (1.0 + port_returns).cumprod()
    bench_cum = (1.0 + bench_returns).cumprod()

    # Window indices helpers
    def last_n_days(n: int) -> pd.Index:
        return common_idx[-n:] if len(common_idx) >= n else common_idx

    idx_30 = last_n_days(30)
    idx_60 = last_n_days(60)
    idx_252 = last_n_days(252)  # ~1Y

    # Returns and alpha
    ret_30d = _series_return(port_returns.loc[idx_30])
    bench_ret_30d = _series_return(bench_returns.loc[idx_30])
    alpha_30d = ret_30d - bench_ret_30d

    ret_60d = _series_return(port_returns.loc[idx_60])
    bench_ret_60d = _series_return(bench_returns.loc[idx_60])
    alpha_60d = ret_60d - bench_ret_60d

    ret_1y = _series_return(port_returns.loc[idx_252])
    bench_ret_1y = _series_return(bench_returns.loc[idx_252])
    alpha_1y = ret_1y - bench_ret_1y

    ret_si = _series_return(port_returns)
    bench_ret_si = _series_return(bench_returns)
    alpha_si = ret_si - bench_ret_si

    # Vol, beta, IR, hit rate, max drawdown
    vol_1y = float(port_returns.loc[idx_252].std() * np.sqrt(252.0))
    beta_1y, alpha_annual_from_beta = _beta_and_alpha(
        port_returns.loc[idx_252], bench_returns.loc[idx_252]
    )
    ir_1y = _info_ratio(port_returns.loc[idx_252], bench_returns.loc[idx_252])
    hit_rate_1y = _hit_rate(port_returns.loc[idx_252], bench_returns.loc[idx_252])
    maxdd = _max_drawdown(port_cum)

    beta_drift = float(beta_1y - beta_target)

    # SmartSafe state
    vix_series = _load_vix(period="3y")
    vix_series = vix_series.reindex(common_idx).fillna(method="ffill")
    vix_level = float(vix_series.iloc[-1]) if len(vix_series) else 18.0

    # 60D drawdowns for regime selection
    port_cum_60 = port_cum.loc[idx_60]
    bench_cum_60 = bench_cum.loc[idx_60]
    dd_60 = _max_drawdown(port_cum_60)
    bench_dd_60 = _max_drawdown(bench_cum_60)

    regime = _smartsafe_regime(vix_level, dd_60, bench_dd_60)
    base_sweep = _SMARTSAFE_BASE_SWEEP.get(regime, 0.0)
    extra_sweep = base_sweep * cfg.smartsafe_factor

    # Turnover telemetry
    if len(turnover_list) > 0:
        # quarterly turnovers -> approximate 1Y turnover from last 4
        last_turns = turnover_list[-4:] if len(turnover_list) >= 4 else turnover_list
        turnover_1y_est = float(np.mean(last_turns))
    else:
        turnover_1y_est = 0.0

    turnover_1d = float(turnover_1y_est / 252.0)

    if regime in ("Stress", "Panic"):
        execution_regime = "Defensive"
    elif regime == "Caution":
        execution_regime = "Balanced"
    else:
        execution_regime = "Offense"

    intraday_return = 0.0  # placeholder

    # Positions DataFrame (top 10) — show current tilted target weights
    latest_prices = prices.iloc[-1]
    pos_rows = []
    for t, w_t in zip(tickers, adj_weights):
        last_px = float(latest_prices.get(t, np.nan))
        pos_rows.append(
            {
                "ticker": t,
                "weight": float(w_t),
                "last_price": last_px if np.isfinite(last_px) else np.nan,
            }
        )
    positions_df = pd.DataFrame(pos_rows).sort_values("weight", ascending=False)

    # Human-readable benchmark string
    bench_str_parts = []
    for t, w_b in bench_map.items():
        bench_str_parts.append(f"{int(round(w_b * 100))}% {t}")
    benchmark_str = " + ".join(bench_str_parts)

    metrics = {
        "mode": mode_lower,
        "intraday_return": intraday_return,
        "ret_30d": ret_30d,
        "alpha_30d": alpha_30d,
        "ret_60d": ret_60d,
        "alpha_60d": alpha_60d,
        "ret_1y": ret_1y,
        "alpha_1y": alpha_1y,
        "ret_si": ret_si,
        "alpha_si": alpha_si,
        "vol_1y": vol_1y,
        "maxdd": maxdd,
        "ir_1y": ir_1y,
        "hit_rate_1y": hit_rate_1y,
        "beta_1y": beta_1y,
        "beta_target": beta_target,
        "beta_drift": beta_drift,
        "vix_level": vix_level,
        "smartsafe_sweep_fraction": base_sweep,
        "smartsafe3_state": regime,
        "smartsafe3_extra_fraction": extra_sweep,
        "turnover_1d": turnover_1d,
        "turnover_1y_est": turnover_1y_est,
        "rebalance_frequency_days": rebalance_every,
        "execution_regime": execution_regime,
    }

    return {
        "benchmark": benchmark_str,
        "metrics": metrics,
        "positions": positions_df,
    }


def get_wave_snapshot(wave: str, mode: str = "standard") -> Dict[str, object]:
    """
    Public API used by app.py.
    Returns a dict with keys:
        - benchmark: str
        - metrics: dict
        - positions: DataFrame
    """
    weights_df = _load_wave_weights("wave_weights.csv")
    return _compute_wave_series(wave, mode, weights_df)