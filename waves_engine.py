"""
waves_engine.py — WAVES Intelligence™ Simple Engine (Full Reset)

Purpose
-------
Light-weight engine for the Streamlit "waves-simple" console.

• Reads wave_weights.csv  (columns: wave,ticker,weight)
• Cleans names and normalizes weights per Wave
• Provides:
    - get_available_waves()    -> list of Wave names
    - get_wave_snapshot(...)   -> current positions with live prices
    - get_wave_history_v2(...) -> simple daily history for each Wave

Notes
-----
• This version is intentionally simple and self-contained.
• It accepts **extra keyword arguments** so older app.py calls
  won’t crash even if they pass more parameters than we use.
• History logic is NAV-only. Benchmarks / alpha are stubbed
  but present so the UI can still render tables.
"""

import os
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # Streamlit will show a clear error if prices are needed


# -------------------------------------------------------------------
# Paths & global cache
# -------------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"

# Optional: if we later build a big pre-computed file
FULL_WAVE_HISTORY_CSV = "Full_Wave_History.csv"

_wave_weights_cache: Optional[pd.DataFrame] = None


# -------------------------------------------------------------------
# Core: load & clean wave_weights.csv
# -------------------------------------------------------------------

def _load_wave_weights(csv_path: str = WAVE_WEIGHTS_CSV) -> pd.DataFrame:
    """
    Load and clean wave_weights.csv.

    Expected columns (case-insensitive):
        wave, ticker, weight

    Returns a DataFrame with:
        wave  (str)
        ticker (str, uppercased)
        weight (float), normalized to sum to 1 within each Wave
    """
    global _wave_weights_cache

    # Simple cache – this file rarely changes during a single run
    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found in repo root. "
            f"Make sure you committed the new 12-Wave template."
        )

    df = pd.read_csv(csv_path)

    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    # Clean values
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()]
        raise ValueError(
            "wave_weights.csv has non-numeric weights in 'weight' column. "
            f"Offending rows:\n{bad_rows}"
        )

    # Normalize to 1.0 within each Wave (safe even if already 1.0)
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv is empty after filtering non-positive weights.")

    df["weight"] = df["weight"].astype(float)

    # Normalize by Wave
    grouped = df.groupby("wave")["weight"].transform("sum").replace(0, np.nan)
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


# -------------------------------------------------------------------
# Public API: Wave list
# -------------------------------------------------------------------

def get_available_waves() -> List[str]:
    """
    Return a sorted list of Wave names discovered in wave_weights.csv.
    """
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


# -------------------------------------------------------------------
# Helpers for prices & slugs
# -------------------------------------------------------------------

def _make_wave_slug(wave_name: str) -> str:
    """
    Turn 'S&P 500 Wave' -> 'SP500_Wave', etc.
    Used only for log filenames if/when needed.
    """
    slug = wave_name.replace("&", "and")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch == " ")
    slug = "_".join(slug.split())
    return slug


def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get latest close price for each ticker using yfinance.
    Returns {ticker: price}. Missing quotes become NaN.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    if not tickers:
        return {}

    # yfinance can return Series (single ticker) or DataFrame (multi).
    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    latest_prices: Dict[str, float] = {}

    # Single ticker -> Series
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        # Multi-index not used; treat as one ticker
        last_row = data["Close"].dropna().iloc[-1]
        latest_prices[tickers[0]] = float(last_row)
        return latest_prices

    # Multi-ticker format
    for t in tickers:
        try:
            series = data[t]["Close"].dropna()
            price = float(series.iloc[-1]) if not series.empty else np.nan
        except Exception:
            price = np.nan
        latest_prices[t] = price

    return latest_prices


def _build_history_from_prices(
    wave_name: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Very simple daily NAV history built on the fly from prices.

    • Constant weights (no rebalancing inside the window)
    • NAV starts at 1.0
    • Returns:
        date, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return, daily_alpha, cum_alpha

      Bench columns are currently stubbed to 0 so the UI
      still has the expected fields.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days) + 5)  # buffer
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    # Normalize to DataFrame of shape [date, ticker]
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        # Single ticker -> Series
        px = data["Close"].to_frame(name=tickers[0])
    else:
        # Multi-index [field,ticker] -> take Close
        px = data["Close"]

    px = px.dropna(how="all")
    if px.empty:
        raise ValueError(f"No price history returned for Wave '{wave_name}'.")

    # Align weights
    weights = sub.set_index("ticker")["weight"].reindex(px.columns).fillna(0.0)
    weights = weights.values.reshape(1, -1)

    # Portfolio NAV starting at 1.0
    # Compute daily returns of each ticker, then portfolio daily return.
    px = px.sort_index()
    ret = px.pct_change().fillna(0.0)
    port_ret = (ret * weights).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()

    hist = pd.DataFrame(
        {
            "date": px.index.date,
            "wave_nav": nav.values,
            "wave_return": port_ret.values,
        }
    )

    hist["cum_wave_return"] = hist["wave_nav"] / hist["wave_nav"].iloc[0] - 1.0

    # Stub benchmark series = flat 0 for now so UI can compute deltas
    hist["bench_nav"] = 1.0
    hist["bench_return"] = 0.0
    hist["cum_bench_return"] = 0.0

    # Alpha = wave_return - bench_return (here == wave_return)
    hist["daily_alpha"] = hist["wave_return"] - hist["bench_return"]
    hist["cum_alpha"] = hist["cum_wave_return"] - hist["cum_bench_return"]

    return hist


# -------------------------------------------------------------------
# Public API: Snapshots & History
# -------------------------------------------------------------------

def get_wave_snapshot(
    wave_name: str,
    mode: str = "standard",
    as_of: Optional[dt.date] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Return a simple snapshot dict for a Wave:

        {
          "wave": "<Wave Name>",
          "as_of": <date>,
          "positions": DataFrame[
              ticker, weight, price, dollar_weight
          ]
        }

    • 'mode' and **kwargs are accepted for compatibility but
      are not used in this simple engine.
    """
    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    sub["price"] = sub["ticker"].map(prices)
    # dollar_weight = weight * 100 (arbitrary notional; UI mostly cares about %
    sub["dollar_weight"] = sub["weight"]

    as_of = as_of or dt.date.today()

    return {
        "wave": wave_name,
        "mode": mode,
        "as_of": as_of,
        "positions": sub[["ticker", "weight", "price", "dollar_weight"]],
    }


def get_wave_history(
    wave_name: str,
    mode: str = "standard",
    lookback_days: int = 365,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Primary daily history function (v1).

    Right now this:
      • builds daily NAV history from prices on the fly
      • ignores 'mode' and extra kwargs, but accepts them so
        existing app.py calls don't crash.

    Returns a DataFrame with at least:
        date
        wave_nav
        wave_return
        cum_wave_return
        bench_nav
        bench_return
        cum_bench_return
        daily_alpha
        cum_alpha
    """
    # Later we can switch this to read from FULL_WAVE_HISTORY_CSV
    # if that file is fully populated.
    return _build_history_from_prices(wave_name, lookback_days=lookback_days)


def get_wave_history_v2(
    wave_name: str,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    v2 wrapper used by app.py

    • For now it simply forwards all arguments to get_wave_history().
    • Once Full_Wave_History.csv is fully wired, we can switch
      this to a new implementation while keeping the same name.
    """
    return get_wave_history(wave_name, *args, **kwargs)