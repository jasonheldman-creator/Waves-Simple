"""
waves_engine.py

Merged WAVES Intelligence™ engine:

- Reads wave_weights.csv (user-defined Waves)
- Builds a dynamic S&P 500 Wave from sp500_universe.csv
  with an overlay from any "S&P Wave" rows in wave_weights.csv
- Fetches price history via yfinance
- Computes per-Wave performance & alpha vs benchmarks
- Produces top holdings
- Computes simple factor scores (momentum, quality/low-vol proxy)
- Exposes a VIX helper for the console

This file is designed to match the expectations of the
WAVES Super Console app.py.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

WAVE_WEIGHTS_PATH = ROOT / "wave_weights.csv"
SP500_UNIVERSE_PATH = ROOT / "sp500_universe.csv"

# Portion of S&P Wave weight that is driven by the custom overlay
SP500_OVERLAY_SHARE = 0.30  # 30% overlay from wave_weights; 70% broad S&P core

# Lookback for summary calculations
MAX_LOOKBACK_DAYS = 365

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Wave weights & dynamic S&P Wave
# ---------------------------------------------------------------------


def load_wave_weights(path: Path = WAVE_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Load base wave weights from CSV: columns wave,ticker,weight
    Normalizes weights within each wave but does NOT yet apply
    the dynamic S&P logic.
    """
    df = _read_csv(path)
    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv must contain columns {expected_cols}, missing {missing}")

    df = df.copy()
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Normalize within each wave (in case weights don't sum to 1)
    grouped = []
    for wave, sub in df.groupby("wave"):
        total = sub["weight"].sum()
        if total <= 0:
            continue
        sub = sub.copy()
        sub["weight"] = sub["weight"] / total
        grouped.append(sub)
    if not grouped:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    return pd.concat(grouped, ignore_index=True)


def _build_sp500_core(path: Path = SP500_UNIVERSE_PATH) -> pd.DataFrame:
    """
    Build a broad S&P 500 core from sp500_universe.csv.

    Expected columns: ticker, market_cap (optional but preferred).
    If market_cap is missing, all tickers are equal-weighted.
    Returns DataFrame with columns: ticker, core_weight.
    """
    df = _read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("sp500_universe.csv must contain a 'ticker' column.")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0.0)
        # Use market cap weights; if all zero, fall back to equal
        if df["market_cap"].sum() > 0:
            df["core_weight"] = df["market_cap"] / df["market_cap"].sum()
        else:
            df["core_weight"] = 1.0 / len(df)
    else:
        df["core_weight"] = 1.0 / len(df)

    return df[["ticker", "core_weight"]]


def _build_dynamic_sp500_wave(
    base_weights: pd.DataFrame,
    sp500_path: Path = SP500_UNIVERSE_PATH,
    overlay_share: float = SP500_OVERLAY_SHARE,
    wave_name: str = "S&P Wave",
) -> pd.DataFrame:
    """
    Build the dynamic S&P Wave:

    - Start from a broad S&P core based on sp500_universe.csv
      (market-cap weighted if available, else equal).
    - Take any rows in base_weights where wave == wave_name and
      treat them as an overlay (user-specified tilt).
    - Combine core and overlay:
        weight = (1 - overlay_share) * core_weight
                 + overlay_share * overlay_weight

      Overlay weights are normalized within the overlay basket.

    Returns DataFrame columns: wave, ticker, weight
    """
    # Grab overlay rows (if any)
    overlay = base_weights[base_weights["wave"] == wave_name].copy()
    overlay = overlay[["ticker", "weight"]]

    # Normalize overlay weights (relative tilt)
    if not overlay.empty:
        tot = overlay["weight"].sum()
        if tot > 0:
            overlay["weight"] = overlay["weight"] / tot
        else:
            overlay = overlay.iloc[0:0]

    # Load S&P core
    try:
        core = _build_sp500_core(sp500_path)
    except Exception as e:
        # Fallback: if we can't build a core, use overlay only
        if overlay.empty:
            raise RuntimeError(f"Failed to build S&P core and no overlay present: {e}")
        core = overlay.rename(columns={"weight": "core_weight"})
        core["core_weight"] = 1.0 / len(core)

    # Merge overlay into core
    merged = core.merge(overlay, how="left", on="ticker", suffixes=("_core", "_overlay"))
    merged["weight"].fillna(0.0, inplace=True)
    merged["core_weight"].fillna(0.0, inplace=True)

    overlay_share = float(max(0.0, min(overlay_share, 1.0)))
    base_share = 1.0 - overlay_share

    merged["final_weight"] = base_share * merged["core_weight"] + overlay_share * merged["weight"]

    # Normalize again (defensive)
    total = merged["final_weight"].sum()
    if total <= 0:
        # Edge: no usable weights; equal-weight everything
        merged["final_weight"] = 1.0 / len(merged)
    else:
        merged["final_weight"] = merged["final_weight"] / total

    out = merged[["ticker", "final_weight"]].copy()
    out.rename(columns={"final_weight": "weight"}, inplace=True)
    out["wave"] = wave_name

    return out[["wave", "ticker", "weight"]]


def get_dynamic_wave_weights() -> pd.DataFrame:
    """
    Main entry: returns a DataFrame of all Waves with dynamic S&P applied.

    - Loads wave_weights.csv
    - Builds a dynamic "S&P Wave" from sp500_universe.csv + overlay
    - Keeps all other Waves as defined in wave_weights.csv (normalized)
    """
    base = load_wave_weights(WAVE_WEIGHTS_PATH)

    if base.empty:
        return base

    wave_name = "S&P Wave"

    # Non-S&P waves
    non_sp = base[base["wave"] != wave_name].copy()

    # Build dynamic S&P (if there is an S&P Wave in base or sp500_universe exists)
    try:
        dynamic_sp = _build_dynamic_sp500_wave(base, SP500_UNIVERSE_PATH, SP500_OVERLAY_SHARE, wave_name)
    except Exception as e:
        # If something goes wrong, gracefully fall back to whatever S&P rows exist in base
        print(f"[WARN] Dynamic S&P Wave failed, falling back to base S&P weights: {e}")
        dynamic_sp = base[base["wave"] == wave_name].copy()
        if dynamic_sp.empty:
            # still nothing, just return non-S&P waves
            return non_sp

    # Combine
    all_waves = pd.concat([non_sp, dynamic_sp], ignore_index=True)

    # Normalize within each wave one more time for safety
    grouped = []
    for wave, sub in all_waves.groupby("wave"):
        total = sub["weight"].sum()
        if total <= 0:
            continue
        sub = sub.copy()
        sub["weight"] = sub["weight"] / total
        grouped.append(sub)
    if not grouped:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    return pd.concat(grouped, ignore_index=True)


def get_wave_names(weights_df: pd.DataFrame) -> List[str]:
    """
    Return sorted list of unique wave names.
    """
    if weights_df is None or weights_df.empty:
        return []
    return sorted(weights_df["wave"].dropna().unique().tolist())


# ---------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------


def fetch_price_history(tickers: List[str], lookback_days: int = MAX_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for the given tickers.

    - Uses yfinance
    - Returns wide DataFrame with Date index and tickers as columns
    """
    if not tickers:
        return pd.DataFrame()

    tickers_clean = sorted(set(str(t).upper().strip() for t in tickers))
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 5)  # small pad

    data = yf.download(
        tickers_clean,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance returns different shapes depending on # of tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            # pick the first top-level as a fallback
            top0 = data.columns.levels[0][0]
            prices = data[top0].copy()
    else:
        # single series: name is the ticker
        prices = data.copy()

    prices = prices.dropna(how="all")
    prices.index.name = "date"
    return prices


def get_vix_level() -> Optional[float]:
    """
    Fetch current-ish VIX level using yfinance ^VIX.
    """
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        last = df["Close"].dropna().iloc[-1]
        return float(last)
    except Exception as e:
        print(f"[WARN] Failed to fetch VIX: {e}")
        return None


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------


def get_benchmark_for_wave(wave_name: str) -> Optional[str]:
    """
    Map a Wave name to a benchmark ticker. This is heuristic
    and can be customized as needed.
    """
    name = wave_name.lower()

    # S&P / core US equity
    if "s&p" in name or "sp " in name or "core us" in name:
        return "SPY"

    # Growth / tech
    if "growth" in name or "innovation" in name or "quantum" in name:
        return "QQQ"

    # Crypto
    if "crypto" in name or "bitcoin" in name:
        # You can switch to BITO if you prefer ETF
        return "BTC-USD"

    # Income / dividend
    if "income" in name or "yield" in name:
        return "VYM"

    # Energy / power
    if "energy" in name or "power" in name:
        return "XLE"

    # Clean transit / infrastructure
    if "clean transit" in name or "infrastructure" in name:
        return "ICLN"

    # Small cap
    if "small cap" in name or "smid" in name:
        return "IWM"

    # Default fallback
    return "SPY"


# ---------------------------------------------------------------------
# Factors (momentum / quality proxies)
# ---------------------------------------------------------------------


def compute_factor_scores(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple factor proxies from price history only.

    - momentum_score: 6M total return
    - quality_score: negative volatility (lower vol = higher score)
    - factor_score: average of the two
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    # Ensure columns are tickers, rows are dates
    df = price_df.sort_index().copy()

    # Use ~6 months of data where available
    # (but we assume caller already passed a suitable window)
    rets = df.pct_change().dropna(how="all")

    # Momentum: cumulative return over full window
    momentum = (1.0 + rets).prod() - 1.0  # per column

    # Volatility: standard deviation of daily returns
    vol = rets.std()

    # Quality proxy: lower vol = higher quality
    quality = -vol

    # Build DataFrame
    out = pd.DataFrame(
        {
            "momentum_score": momentum,
            "quality_score": quality,
        }
    )

    # Hybrid factor (simple average)
    out["factor_score"] = out[["momentum_score", "quality_score"]].mean(axis=1)

    return out


# ---------------------------------------------------------------------
# Wave summary (returns, alpha, top holdings)
# ---------------------------------------------------------------------


def _compute_portfolio_returns(
    weights: pd.Series,
    lookback_days: int = MAX_LOOKBACK_DAYS,
) -> pd.Series:
    """
    Compute daily portfolio returns from ticker weights.
    """
    tickers = list(weights.index)
    prices = fetch_price_history(tickers, lookback_days=lookback_days)
    if prices.empty:
        return pd.Series(dtype=float)

    # align weights to price columns
    w = weights.reindex(prices.columns).fillna(0.0)
    rets = prices.pct_change().dropna(how="all")
    port_rets = (rets * w).sum(axis=1)
    return port_rets


def _compute_horizon_return(rets: pd.Series, days: int) -> Optional[float]:
    """
    Compute cumulative return over the last N observations (trading days).
    """
    if rets is None or rets.empty:
        return None
    tail = rets.tail(days)
    if tail.empty:
        return None
    return float((1.0 + tail).prod() - 1.0)


def compute_wave_summary(
    wave_name: str,
    weights_df: pd.DataFrame | None = None,
) -> Dict:
    """
    Compute summary metrics for a single Wave:

    - 1D / 30D / 60D returns
    - Alpha vs benchmark over same horizons
    - Top 10 holdings
    - Benchmark symbol

    Returns dict with keys:
        return_1d, alpha_1d,
        return_30d, alpha_30d,
        return_60d, alpha_60d,
        top_holdings (DataFrame),
        benchmark (str or None)
    """
    if weights_df is None:
        weights_df = get_dynamic_wave_weights()

    df = weights_df[weights_df["wave"] == wave_name].copy()
    if df.empty:
        return {}

    # aggregate weights per ticker
    w = (
        df.groupby("ticker")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    # Compute portfolio returns
    port_rets = _compute_portfolio_returns(w, lookback_days=MAX_LOOKBACK_DAYS)

    # Benchmark
    bench_ticker = get_benchmark_for_wave(wave_name)
    bench_rets = None
    if bench_ticker:
        try:
            bench_prices = fetch_price_history([bench_ticker], lookback_days=MAX_LOOKBACK_DAYS)
            if not bench_prices.empty:
                bench_rets = bench_prices.iloc[:, 0].pct_change().dropna()
        except Exception as e:
            print(f"[WARN] Failed to fetch benchmark {bench_ticker} for {wave_name}: {e}")
            bench_rets = None

    # Align for alpha
    if bench_rets is not None and not bench_rets.empty and not port_rets.empty:
        idx = port_rets.index.intersection(bench_rets.index)
        port_rets = port_rets.reindex(idx)
        bench_rets = bench_rets.reindex(idx)
    else:
        bench_rets = None

    # Horizon returns
    ret_1d = _compute_horizon_return(port_rets, 1)
    ret_30d = _compute_horizon_return(port_rets, 30)
    ret_60d = _compute_horizon_return(port_rets, 60)

    if bench_rets is not None and not bench_rets.empty:
        b1 = _compute_horizon_return(bench_rets, 1)
        b30 = _compute_horizon_return(bench_rets, 30)
        b60 = _compute_horizon_return(bench_rets, 60)
        alpha_1d = None if (ret_1d is None or b1 is None) else ret_1d - b1
        alpha_30d = None if (ret_30d is None or b30 is None) else ret_30d - b30
        alpha_60d = None if (ret_60d is None or b60 is None) else ret_60d - b60
    else:
        alpha_1d = alpha_30d = alpha_60d = None

    # Top 10 holdings DF
    top10 = w.head(10).reset_index()
    top10.columns = ["ticker", "weight"]

    summary = {
        "return_1d": ret_1d,
        "alpha_1d": alpha_1d,
        "return_30d": ret_30d,
        "alpha_30d": alpha_30d,
        "return_60d": ret_60d,
        "alpha_60d": alpha_60d,
        "top_holdings": top10,
        "benchmark": bench_ticker,
    }

    return summary