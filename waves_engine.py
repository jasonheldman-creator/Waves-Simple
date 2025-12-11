"""
waves_engine.py — WAVES Intelligence™ Hybrid Engine (History + Live)

Purpose
-------
Hybrid engine for the WAVES console:

• Reads wave_weights.csv  (columns: wave,ticker,weight)
• Optionally reads Full_Wave_History.csv for precomputed histories
• Falls back to live yfinance prices when history CSV is missing or incomplete
• Provides:

    get_available_waves()  -> list of Wave names
    get_wave_snapshot(...) -> current positions snapshot (with prices)
    get_wave_history(...)  -> daily NAV / returns / benchmark / alpha
    compute_risk_stats(...) -> summary risk/return stats for a Wave

Notes
-----
• History CSV is OPTIONAL. If it's not present or doesn't contain a given
  Wave, the engine will compute history on the fly using yfinance.
• This file is designed to be robust and forgiving:
    - handles missing tickers
    - handles history gaps
    - does not crash the app on a single Wave's failure
"""

from __future__ import annotations

import os
import math
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # Streamlit will show a clear error if this is None


# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"
FULL_HISTORY_CSV = "Full_Wave_History.csv"  # optional hybrid history file

_wave_weights_cache: Optional[pd.DataFrame] = None
_full_history_cache: Optional[pd.DataFrame] = None

# Simple benchmark mapping by Wave name (case-insensitive keys)
BENCHMARK_BY_WAVE: Dict[str, str] = {
    "ai wave": "QQQ",
    "cloud & software wave": "QQQ",
    "crypto income wave": "BTC-USD",
    "future power & energy wave": "XLE",
    "clean transit-infrastructure wave": "IYT",
    "growth wave": "QQQ",
    "income wave": "AGG",
    "quantum computing wave": "QQQ",
    "small cap growth wave": "IWM",
    "s&p 500 wave": "SPY",
    "smartsafe wave": "BIL",
}


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
        weight (float), normalized to sum to 1.0 within each Wave
    """
    global _wave_weights_cache

    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found in repo root.\n"
            f"Make sure wave_weights.csv is committed."
        )

    df = pd.read_csv(csv_path)

    # Standardize column names
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

    # Drop non-positive
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv is empty after filtering weights > 0.")

    # Normalize to 1.0 within each Wave
    grouped = df.groupby("wave")["weight"].transform("sum")
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


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
# Helpers: history CSV loading and filtering
# ---------------------------------------------------------------------------

def _load_full_history(csv_path: str = FULL_HISTORY_CSV) -> pd.DataFrame:
    """
    Load the optional Full_Wave_History.csv file.

    Expected columns (case-insensitive):
        date, wave, wave_nav, wave_return, cum_wave_return,
        bench_nav, bench_return, cum_bench_return,
        daily_alpha, cum_alpha

    Extra columns are allowed; we just ignore them.
    """
    global _full_history_cache

    if _full_history_cache is not None:
        return _full_history_cache.copy()

    if not os.path.exists(csv_path):
        # No hybrid file yet — that's OK
        _full_history_cache = pd.DataFrame()
        return _full_history_cache.copy()

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "wave" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain at least 'date' and 'wave' columns."
        )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["wave"] = df["wave"].astype(str).str.strip()

    _full_history_cache = df.copy()
    return df.copy()


def _get_history_from_csv(wave_name: str, lookback_days: int) -> pd.DataFrame:
    """
    Attempt to get history for a Wave from the Full_Wave_History.csv file.

    Returns an empty DataFrame if no data is available.
    """
    df = _load_full_history()
    if df.empty:
        return pd.DataFrame()

    df_wave = df[df["wave"].str.lower() == wave_name.lower()].copy()
    if df_wave.empty:
        return pd.DataFrame()

    cutoff = dt.date.today() - dt.timedelta(days=int(lookback_days))
    df_wave = df_wave[df_wave["date"] >= cutoff].copy()
    df_wave = df_wave.sort_values("date")

    return df_wave


# ---------------------------------------------------------------------------
# Helpers: yfinance price download
# ---------------------------------------------------------------------------

def _ensure_yfinance_available() -> None:
    if yf is None:
        raise RuntimeError(
            "yfinance is not available. Add it to requirements.txt or your environment."
        )


def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get latest close price for each ticker using yfinance.

    Uses one download per ticker for robustness.
    Returns:
        {ticker: price} (NaN if quote is missing).
    """
    _ensure_yfinance_available()

    prices: Dict[str, float] = {}
    if not tickers:
        return prices

    end = dt.date.today()
    start = end - dt.timedelta(days=7)

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data, pd.DataFrame) and not data.empty and "Close" in data.columns:
                s = data["Close"].dropna()
                if not s.empty:
                    prices[t] = float(s.iloc[-1])
                    continue
            prices[t] = np.nan
        except Exception:
            prices[t] = np.nan

    return prices


def _build_history_from_prices(
    wave_name: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Build daily NAV + simple benchmark from yfinance.

    • Constant portfolio weights across the window
    • NAV starts at 1.0
    • Returns a DataFrame with columns:

        date
        wave_nav, wave_return, cum_wave_return
        bench_nav, bench_return, cum_bench_return
        daily_alpha, cum_alpha
    """
    _ensure_yfinance_available()

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days))

    # --- Portfolio prices (download each ticker separately) ---
    price_series_list: list[pd.Series] = []

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
            if "Close" not in data.columns:
                continue
            s = data["Close"].dropna()
            if s.empty:
                continue
            s = s.rename(t)
            price_series_list.append(s)
        except Exception:
            continue

    if not price_series_list:
        raise RuntimeError(
            f"Could not build price history from yfinance for Wave '{wave_name}'."
        )

    px = pd.concat(price_series_list, axis=1).sort_index()
    px = px.dropna(how="all")
    if px.empty:
        raise RuntimeError(f"Empty price matrix after cleaning for Wave '{wave_name}'.")

    # Align weights with downloaded tickers
    weights = sub.set_index("ticker")["weight"]
    common = sorted(set(px.columns) & set(weights.index))
    if not common:
        raise RuntimeError(
            f"No overlap between tickers in prices vs weights for Wave '{wave_name}'."
        )

    px = px[common]
    w_vec = weights.loc[common].values.reshape(1, -1)

    # Daily returns and NAV
    px = px.astype(float)
    ret = px.pct_change().fillna(0.0)
    port_ret = (ret.values * w_vec).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()

    hist = pd.DataFrame(
        {
            "date": px.index.date,
            "wave_nav": nav,
            "wave_return": port_ret,
        }
    )
    hist["cum_wave_return"] = hist["wave_nav"] / hist["wave_nav"].iloc[0] - 1.0

    # --- Benchmark series ---
    bench_ticker = BENCHMARK_BY_WAVE.get(wave_name.lower(), "SPY")
    try:
        bdata = yf.download(
            bench_ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(bdata, pd.DataFrame) and not bdata.empty:
            if "Close" in bdata.columns:
                bs = bdata["Close"].dropna()
            else:
                bs = pd.Series(dtype=float)
        else:
            bs = pd.Series(dtype=float)
    except Exception:
        bs = pd.Series(dtype=float)

    if bs.empty:
        # Flat benchmark (0 return)
        hist["bench_nav"] = 1.0
        hist["bench_return"] = 0.0
        hist["cum_bench_return"] = 0.0
    else:
        bs = bs.sort_index()
        # Align with portfolio dates
        bs = bs.reindex(px.index).ffill()
        b_ret = bs.pct_change().fillna(0.0)
        b_nav = (1.0 + b_ret).cumprod()

        hist["bench_nav"] = b_nav.values
        hist["bench_return"] = b_ret.values
        hist["cum_bench_return"] = (
            hist["bench_nav"] / hist["bench_nav"].iloc[0] - 1.0
        )

    # Alpha
    hist["daily_alpha"] = hist["wave_return"] - hist["bench_return"]
    hist["cum_alpha"] = hist["cum_wave_return"] - hist["cum_bench_return"]

    return hist


# ---------------------------------------------------------------------------
# Public API: snapshot & history
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

    * 'mode' and **kwargs are accepted but not used in this version.
    """
    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    sub["price"] = sub["ticker"].map(prices)
    sub["dollar_weight"] = sub["weight"]  # simple placeholder

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
    Hybrid history lookup:

    1) Try to load from Full_Wave_History.csv (if present).
    2) If not available or empty, compute from yfinance prices.

    Mode is currently ignored, but accepted for future compatibility.
    """
    # First try history CSV
    hist = _get_history_from_csv(wave_name, lookback_days)
    if not hist.empty:
        # Ensure required columns exist
        needed = [
            "date",
            "wave_nav",
            "wave_return",
            "cum_wave_return",
            "bench_nav",
            "bench_return",
            "cum_bench_return",
            "daily_alpha",
            "cum_alpha",
        ]
        for col in needed:
            if col not in hist.columns:
                # If missing any, fall back to live builder
                return _build_history_from_prices(wave_name, lookback_days)
        return hist

    # No CSV-based history → build from prices
    return _build_history_from_prices(wave_name, lookback_days)


# ---------------------------------------------------------------------------
# Risk/stats helper
# ---------------------------------------------------------------------------

def compute_risk_stats(hist: pd.DataFrame) -> Dict[str, float]:
    """
    Given a history DataFrame from get_wave_history, compute:

        total_return   (cum_wave_return last)
        bench_return   (cum_bench_return last)
        alpha_total    (cum_alpha last)
        ann_vol        (annualized volatility, 252d)
        sharpe         (simple Sharpe vs benchmark, rf≈0)
        max_drawdown   (max peak-to-trough drawdown)
    """
    if hist is None or hist.empty:
        return {
            "total_return": float("nan"),
            "bench_return": float("nan"),
            "alpha_total": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    wave_ret = pd.Series(hist["wave_return"].values)
    bench_ret = pd.Series(hist["bench_return"].values)

    total_return = float(hist["cum_wave_return"].iloc[-1])
    bench_return = float(hist["cum_bench_return"].iloc[-1])
    alpha_total = float(hist["cum_alpha"].iloc[-1])

    # Volatility & Sharpe
    if wave_ret.std(ddof=1) > 0:
        ann_vol = float(wave_ret.std(ddof=1) * math.sqrt(252))
        excess = wave_ret - bench_ret
        sharpe = float(
            excess.mean() / wave_ret.std(ddof=1) * math.sqrt(252)
        )
    else:
        ann_vol = float("nan")
        sharpe = float("nan")

    # Max drawdown
    nav = pd.Series(hist["wave_nav"].values)
    rolling_max = nav.cummax()
    drawdowns = nav / rolling_max - 1.0
    max_dd = float(drawdowns.min())

    return {
        "total_return": total_return,
        "bench_return": bench_return,
        "alpha_total": alpha_total,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }