"""
waves_engine.py — WAVES Intelligence™ Simple Engine v2

Purpose
-------
Light-weight engine for the Streamlit "waves-simple" console.

• Reads wave_weights.csv  (columns: wave,ticker,weight)
• (Optionally) reads Full_Wave_History.csv if you later pre-compute history.
• Cleans names and normalizes weights per Wave.
• Provides:

    - get_available_waves()      -> list of Wave names
    - get_wave_snapshot(...)     -> current positions snapshot
    - get_wave_history(...)      -> simple daily history per Wave
    - compute_wave_metrics(...)  -> overview metrics per Wave

Notes
-----
• History logic is NAV-only with a stub benchmark = 0 for now.
• If Full_Wave_History.csv exists and has rows for a Wave, we use that.
  Otherwise we build history on the fly from yfinance prices.
• All modes ("standard", "alpha_minus_beta", "private_logic") are treated
  the same for now, but the function signatures accept them so the UI
  doesn’t break later when you add mode-specific behavior.
"""

from __future__ import annotations

import os
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# yfinance is optional; we fail gracefully if it's missing or rate limited
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore


# ---------------------------------------------------------------------------
# Paths & global caches
# ---------------------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"
FULL_WAVE_HISTORY_CSV = "Full_Wave_History.csv"

_wave_weights_cache: Optional[pd.DataFrame] = None
_full_history_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Core: load & clean wave_weights.csv
# ---------------------------------------------------------------------------

def _load_wave_weights(csv_path: str = WAVE_WEIGHTS_CSV) -> pd.DataFrame:
    """
    Load and clean wave_weights.csv.

    Expected columns (case-insensitive):
        wave, ticker, weight

    Returns a DataFrame with:
        wave   (str)
        ticker (str, uppercased)
        weight (float), normalized to sum to 1.0 within each Wave.
    """
    global _wave_weights_cache

    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found in repo root.\n"
            f"Make sure you committed the latest wave_weights.csv."
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

    # Drop rows with non-positive weights
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv is effectively empty after cleaning.")

    # Normalize to 1.0 within each Wave
    grouped = df.groupby("wave")["weight"].transform("sum")
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


# ---------------------------------------------------------------------------
# Optional: pre-computed full history
# ---------------------------------------------------------------------------

def _load_full_history(csv_path: str = FULL_WAVE_HISTORY_CSV) -> Optional[pd.DataFrame]:
    """
    Load Full_Wave_History.csv if present and non-empty.

    Expected columns (case-insensitive, but we're forgiving):
        date, wave, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha

    If the file doesn't exist or has no data rows, returns None.
    """
    global _full_history_cache

    if _full_history_cache is not None:
        return _full_history_cache.copy()

    if not os.path.exists(csv_path):
        _full_history_cache = None
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        _full_history_cache = None
        return None

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    _full_history_cache = df
    return df.copy()


# ---------------------------------------------------------------------------
# Helpers for latest prices (positions snapshot)
# ---------------------------------------------------------------------------

def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get latest close price for each ticker using yfinance.

    Returns {ticker: price}. Missing quotes become NaN.
    """
    prices: Dict[str, float] = {}

    if not tickers:
        return prices

    if yf is None:
        # No yfinance — return NaNs but don't crash
        return {t: np.nan for t in tickers}

    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="5d", interval="1d")
            if data.empty or "Close" not in data.columns:
                price = np.nan
            else:
                price = float(data["Close"].iloc[-1])
        except Exception:
            price = np.nan
        prices[t] = price

    return prices


# ---------------------------------------------------------------------------
# Helpers for building price-based history
# ---------------------------------------------------------------------------

def _build_history_from_prices(
    wave_name: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Build a very simple daily NAV history from yfinance prices.

    • Constant weights (no rebalancing inside the window)
    • NAV starts at 1.0
    • Returns a DataFrame with:
        date, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv.")

    tickers = sub["ticker"].unique().tolist()
    if not tickers:
        raise ValueError(f"Wave '{wave_name}' has no tickers defined.")

    # Grab slightly more history than we need, then trim
    days_str = f"{lookback_days + 7}d"
    price_frames: List[pd.DataFrame] = []

    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period=days_str, interval="1d")
            if data.empty or "Close" not in data.columns:
                continue
            frame = data[["Close"]].rename(columns={"Close": t})
            price_frames.append(frame)
        except Exception:
            # Skip tickers that completely fail
            continue

    if not price_frames:
        raise ValueError(
            f"No price history returned from yfinance for Wave '{wave_name}'."
        )

    # Align all tickers on common dates
    px = pd.concat(price_frames, axis=1, join="inner").sort_index()
    # Keep last lookback_days + 1 rows so pct_change has enough points
    px = px.tail(lookback_days + 1)

    if px.empty:
        raise ValueError(
            f"Price matrix empty after alignment for Wave '{wave_name}'."
        )

    # Align weights to the columns we actually ended up with
    weights = sub.set_index("ticker")["weight"]
    weights = weights.reindex(px.columns).fillna(0.0).values.reshape(1, -1)

    # Compute portfolio returns
    ret = px.pct_change().fillna(0.0)
    port_ret = (ret.values * weights).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()

    hist = pd.DataFrame(
        {
            "date": px.index.date,
            "wave_nav": nav,
            "wave_return": port_ret,
        }
    )

    # Cumulative return relative to 1.0
    hist["cum_wave_return"] = (1.0 + hist["wave_return"]).cumprod() - 1.0

    # Stub benchmark series: flat 0 for now
    hist["bench_nav"] = 1.0
    hist["bench_return"] = 0.0
    hist["cum_bench_return"] = 0.0

    # Alpha = wave_return - bench_return
    hist["daily_alpha"] = hist["wave_return"] - hist["bench_return"]
    hist["cum_alpha"] = hist["cum_wave_return"] - hist["cum_bench_return"]

    return hist


# ---------------------------------------------------------------------------
# Public API: Wave list
# ---------------------------------------------------------------------------

def get_available_waves() -> List[str]:
    """
    Return a sorted list of Wave names discovered in wave_weights.csv.
    """
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


# ---------------------------------------------------------------------------
# Public API: Snapshots & History
# ---------------------------------------------------------------------------

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
            "mode": "<mode>",
            "as_of": <date>,
            "positions": DataFrame[
                ticker, weight, price, dollar_weight
            ]
        }

    'mode' and **kwargs are accepted for compatibility with future,
    more complex logic; they are not used in this simple engine.
    """
    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv.")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    sub["price"] = sub["ticker"].map(prices)
    # For now, dollar_weight is just weight (you can scale later)
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
    Primary daily history function.

    Right now this:
    • Tries to read from Full_Wave_History.csv if it exists and has rows
      for this Wave.
    • Otherwise, builds history from yfinance prices.

    Returns a DataFrame with at least:
        date, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha
    """
    # 1) Try pre-computed file
    full = _load_full_history()
    if full is not None and not full.empty and "wave" in full.columns:
        df = full.copy()
        sub = df[df["wave"] == wave_name].copy() if "wave" in df.columns else pd.DataFrame()
        if not sub.empty:
            # Ensure sorted by date
            if "date" in sub.columns:
                sub["date"] = pd.to_datetime(sub["date"]).dt.date
                sub = sub.sort_values("date")
                if lookback_days and lookback_days > 0:
                    end = sub["date"].max()
                    start = end - dt.timedelta(days=int(lookback_days))
                    sub = sub[sub["date"] >= start]
            return sub

    # 2) Fall back to building from yfinance
    return _build_history_from_prices(wave_name, lookback_days=lookback_days)


def compute_wave_metrics(
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Compute simple overview metrics for all Waves:

    Returns a list of dicts:
        {
            "Wave": "<name>",
            "nav_last": float or NaN,
            "cum_return": float or NaN,   # over lookback_days
            "status": "ok" or error/diagnostic text
        }
    """
    waves = get_available_waves()
    results: List[Dict[str, Any]] = []

    for w in waves:
        nav_last: Optional[float] = np.nan
        cum_ret: Optional[float] = np.nan
        status = "ok"

        try:
            hist = get_wave_history(w, lookback_days=lookback_days)
            if hist is None or hist.empty:
                status = "no history"
            else:
                if "wave_nav" in hist.columns:
                    nav_last = float(hist["wave_nav"].iloc[-1])
                if "cum_wave_return" in hist.columns:
                    cum_ret = float(hist["cum_wave_return"].iloc[-1])
        except Exception as e:
            status = f"error: {type(e).__name__}"

        results.append(
            {
                "Wave": w,
                "nav_last": nav_last,
                "cum_return": cum_ret,
                "status": status,
            }
        )

    return results