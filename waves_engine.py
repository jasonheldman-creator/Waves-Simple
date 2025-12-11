"""
waves_engine.py — WAVES Intelligence™ Simple Engine (hardened v2)

Purpose
-------
Light-weight engine for the Streamlit "waves-simple" console.

• Reads wave_weights.csv  (columns: wave,ticker,weight)
• Optionally can be extended later to read Full_Wave_History.csv
• Cleans names and normalizes weights per Wave
• Provides:
    - get_available_waves()      -> list of Wave names
    - get_wave_snapshot(...)     -> current positions snapshot
    - get_wave_history(...)      -> simple daily NAV history (from yfinance)

Notes
-----
• This version is intentionally simple and defensive.
• It accepts **extra keyword arguments** so old app.py calls still work.
• History logic is NAV-only. Benchmarks / alpha are stubbed but present
  so the UI can still render tables without errors.
"""

import os
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None      # Streamlit will show an error if prices are requested


# ---------------------------------------------------------------------
# Paths & global cache
# ---------------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"

# Optional: future use for pre-computed long history
FULL_WAVE_HISTORY_CSV = "Full_Wave_History.csv"

_wave_weights_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------
# Core: load & clean wave_weights.csv
# ---------------------------------------------------------------------

def _load_wave_weights(csv_path: str = WAVE_WEIGHTS_CSV) -> pd.DataFrame:
    """
    Load and clean wave_weights.csv.

    Expected columns (case-insensitive):
        wave, ticker, weight

    Returns a DataFrame with:
        wave       (str)
        ticker     (str, uppercased)
        weight     (float), normalized to sum to 1 within each Wave
    """
    global _wave_weights_cache

    # Simple cache — this file rarely changes during a session
    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found in repo root.\n"
            f"Make sure you committed the new {csv_path} to GitHub."
        )

    df = pd.read_csv(csv_path)

    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing columns: {missing}")

    # Clean values
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()]
        raise ValueError(
            "wave_weights.csv has non-numeric weights.\n"
            f"Offending rows:\n{bad_rows}"
        )

    # Normalize to 1.0 within each Wave (safe even if already normalized)
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv is empty after filtering weights > 0.")

    grouped = df.groupby("wave")["weight"].transform("sum")
    df["weight"] = df["weight"].astype(float)
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


# ---------------------------------------------------------------------
# Public API: Wave list
# ---------------------------------------------------------------------

def get_available_waves() -> List[str]:
    """
    Return a sorted list of Wave names discovered in wave_weights.csv.
    """
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


# ---------------------------------------------------------------------
# Helpers for price downloads
# ---------------------------------------------------------------------

def _download_ticker_close_series(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> Optional[pd.Series]:
    """
    Download a single ticker's daily Close series from yfinance.

    Returns a Series indexed by date if successful, otherwise None.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    try:
        data = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None

    if not isinstance(data, pd.DataFrame) or "Close" not in data.columns:
        return None

    s = data["Close"].dropna()
    if s.empty:
        return None

    # Normalize index to pure date (no timezone issues)
    s.index = s.index.date
    s.name = ticker
    return s


def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get latest close price for each ticker using yfinance.

    Returns {ticker: price}. Missing quotes become np.nan.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    if not tickers:
        return {}

    end = dt.date.today()
    start = end - dt.timedelta(days=10)

    prices: Dict[str, float] = {}
    for t in tickers:
        s = _download_ticker_close_series(t, start, end)
        if s is None or s.empty:
            prices[t] = float("nan")
        else:
            prices[t] = float(s.iloc[-1])
    return prices


# ---------------------------------------------------------------------
# History helper: build from yfinance prices
# ---------------------------------------------------------------------

def _build_history_from_prices(
    wave_name: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Very simple daily NAV history built from yfinance prices.

    • Constant weights (no rebal inside the window)
    • NAV starts at 1.0
    • Returns:
        date, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days))

    # Download each ticker separately; skip failures instead of crashing.
    price_series: List[pd.Series] = []
    good_tickers: List[str] = []

    for t in tickers:
        s = _download_ticker_close_series(t, start, end)
        if s is None or s.empty:
            # silently skip missing / rate-limited / exotic tickers
            continue
        price_series.append(s)
        good_tickers.append(t)

    if not price_series:
        raise ValueError(
            f"No price history returned from yfinance for any ticker in Wave '{wave_name}'."
        )

    # Align on date index
    px = pd.concat(price_series, axis=1)
    px = px.sort_index()
    px = px.ffill().dropna(how="all")
    if px.empty:
        raise ValueError(
            f"Price matrix is empty after cleaning for Wave '{wave_name}'."
        )

    # Align weights to the tickers that actually have data
    sub_good = sub[sub["ticker"].isin(good_tickers)].copy()
    weights = sub_good.set_index("ticker")["weight"]
    weights = weights / weights.sum()  # renormalize
    weights = weights.reindex(px.columns).fillna(0.0).values.reshape(1, -1)

    # Daily portfolio returns
    ret = px.pct_change().fillna(0.0).values  # shape [days, n_tickers]
    port_ret = (ret * weights).sum(axis=1)    # [days]

    # NAV and cumulative return
    nav = (1.0 + port_ret).cumprod()
    # cumulative return = NAV / NAV[0] - 1
    cum_ret = nav / nav[0] - 1.0

    hist = pd.DataFrame(
        {
            "date": px.index,
            "wave_nav": nav,
            "wave_return": port_ret,
            "cum_wave_return": cum_ret,
        }
    )

    # Stub benchmark series = flat 0 so UI has consistent fields
    hist["bench_nav"] = 1.0
    hist["bench_return"] = 0.0
    hist["cum_bench_return"] = 0.0

    # Alpha vs benchmark (currently just equals wave_return / cum_wave_return)
    hist["daily_alpha"] = hist["wave_return"] - hist["bench_return"]
    hist["cum_alpha"] = hist["cum_wave_return"] - hist["cum_bench_return"]

    # Make sure date is plain date, not Timestamp
    hist["date"] = pd.to_datetime(hist["date"]).dt.date

    return hist


# ---------------------------------------------------------------------
# Public API: Snapshots & History
# ---------------------------------------------------------------------

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
           "mode": mode,
           "as_of": <date>,
           "positions": DataFrame[
               ticker, weight, price, dollar_weight
           ]
        }

    • 'mode' and **kwargs are accepted for compatibility but not yet used
      in this simple engine.
    """
    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    sub = sub.copy()
    sub["price"] = sub["ticker"].map(prices)
    # For now, dollar_weight is just the normalized weight (can later
    # be scaled to a notional portfolio size).
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
      • tries to use a pre-computed Full_Wave_History.csv if it exists
      • otherwise builds daily NAV history from yfinance prices.

    The returned DataFrame MUST have at least:
        date, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha
    """
    # 1) Optional: pre-computed history file (currently probably empty)
    try:
        if os.path.exists(FULL_WAVE_HISTORY_CSV):
            full = pd.read_csv(FULL_WAVE_HISTORY_CSV)
            if not full.empty:
                cols = [c.strip().lower() for c in full.columns]
                full.columns = cols
                if "wave" in cols and "date" in cols and "wave_nav" in cols:
                    sub = full[full["wave"] == wave_name].copy()
                    if not sub.empty:
                        sub["date"] = pd.to_datetime(sub["date"]).dt.date
                        if lookback_days and lookback_days > 0:
                            cutoff = dt.date.today() - dt.timedelta(
                                days=int(lookback_days)
                            )
                            sub = sub[sub["date"] >= cutoff]
                        if not sub.empty:
                            return sub
    except Exception:
        # If anything goes wrong with the pre-computed file, silently
        # fall back to live yfinance logic.
        pass

    # 2) Fallback: build from yfinance prices
    return _build_history_from_prices(wave_name, lookback_days=lookback_days)