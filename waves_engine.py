"""
waves_engine.py — WAVES Intelligence™ Simple Engine (Stable v1.3)

Light-weight engine for the Streamlit console.

• Reads wave_weights.csv (wave, ticker, weight)
• Cleans and normalizes
• Fetches live prices via yfinance
• Builds daily NAV-based performance history
• Exposes API functions:

    get_available_waves()
    get_wave_snapshot(...)
    get_wave_history(...)

This engine ignores mode differences for now (Standard, AMB, PL)
but keeps arguments so the UI does not break.

"""

import os
import datetime as dt
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------------------------------------------------------
# Paths & Cache
# ---------------------------------------------------------
WAVE_WEIGHTS_CSV = "wave_weights.csv"
_wave_weights_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------
# Load & clean wave_weights.csv
# ---------------------------------------------------------
def _load_wave_weights(csv_path: str = WAVE_WEIGHTS_CSV) -> pd.DataFrame:
    """Load + clean wave_weights.csv and normalize weights per Wave."""
    global _wave_weights_cache

    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Make sure your GitHub repo contains the file."
        )

    df = pd.read_csv(csv_path)

    # Standardize names
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"wave", "ticker", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    # Clean values
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    if df["weight"].isna().any():
        raise ValueError("Non-numeric weight detected.")

    # Remove zero / negative weights
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv contains no usable rows.")

    # Normalize weights by Wave
    grouped = df.groupby("wave")["weight"].transform("sum")
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


# ---------------------------------------------------------
# Public API — Wave List
# ---------------------------------------------------------
def get_available_waves() -> List[str]:
    df = _load_wave_weights()
    return sorted(df["wave"].unique().tolist())


# ---------------------------------------------------------
# Fetch live prices (robust, multi-ticker safe)
# ---------------------------------------------------------
def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    if yf is None:
        raise RuntimeError("yfinance unavailable.")

    if not tickers:
        return {}

    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    latest: Dict[str, float] = {}

    # Single ticker → DataFrame
    if isinstance(data, pd.DataFrame) and "Close" in data:
        series = data["Close"].dropna()
        if len(series) > 0:
            latest[tickers[0]] = float(series.iloc[-1])
        else:
            latest[tickers[0]] = np.nan
        return latest

    # Multi-ticker → dict of Series
    for t in tickers:
        try:
            series = data[t]["Close"].dropna()
            latest[t] = float(series.iloc[-1]) if len(series) else np.nan
        except Exception:
            latest[t] = np.nan

    return latest


# ---------------------------------------------------------
# Build performance history
# ---------------------------------------------------------
def _build_history_from_prices(wave_name: str, lookback_days: int = 365) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance unavailable.")

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No tickers found for Wave '{wave_name}'")

    tickers = sub["ticker"].unique().tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days))

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    # --- FIXED BLOCK — handles single-ticker cleanly ---
    if isinstance(data, pd.DataFrame) and "Close" in data:
        # SINGLE TICKER CASE → Close column is Series
        px = data["Close"].to_frame()   # FIXED LINE
    else:
        # MULTI-TICKER
        px = data["Close"]

    px = px.dropna(how="all")
    if px.empty:
        raise ValueError("No price history returned.")

    # Align weights
    weights = sub.set_index("ticker")["weight"].reindex(px.columns).fillna(0).values
    weights = weights.reshape(1, -1)

    # Compute daily returns
    px = px.sort_index()
    ret = px.pct_change().fillna(0)

    port_ret = (ret.values * weights).sum(axis=1)
    nav = (1 + port_ret).cumprod()

    hist = pd.DataFrame({
        "date": px.index,
        "wave_nav": nav,
        "wave_return": port_ret,
    })

    hist["cum_wave_return"] = hist["wave_nav"] - 1

    # Stub benchmark
    hist["bench_nav"] = 1.0
    hist["bench_return"] = 0.0
    hist["cum_bench_return"] = 0.0

    # Alpha = wave - bench
    hist["daily_alpha"] = hist["wave_return"]
    hist["cum_alpha"] = hist["cum_wave_return"]

    return hist


# ---------------------------------------------------------
# Public API: Snapshots
# ---------------------------------------------------------
def get_wave_snapshot(
    wave_name: str,
    mode: str = "standard",
    as_of: Optional[dt.date] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}'")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    sub["price"] = sub["ticker"].map(prices)
    sub["dollar_weight"] = sub["weight"]

    as_of = as_of or dt.date.today()

    return {
        "wave": wave_name,
        "mode": mode,
        "as_of": as_of,
        "positions": sub[["ticker", "weight", "price", "dollar_weight"]],
    }


# ---------------------------------------------------------
# Public API: History
# ---------------------------------------------------------
def get_wave_history(
    wave_name: str,
    mode: str = "standard",
    lookback_days: int = 365,
    **kwargs: Any,
) -> pd.DataFrame:
    return _build_history_from_prices(wave_name, lookback_days)