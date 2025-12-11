"""
waves_engine.py — WAVES Intelligence™ Engine v2 (Option B)

Purpose
-------
Light-weight but more realistic engine for the Streamlit "waves-simple" console.

• Reads wave_weights.csv (columns: wave,ticker,weight)
• Normalizes weights per Wave
• Fetches prices via yfinance
• Builds daily NAV history per Wave + benchmark
• Computes cumulative returns and alpha
• Provides:

    get_available_waves()  -> list of Wave names
    get_wave_snapshot(...) -> current positions & prices
    get_wave_history(...)  -> daily NAV / returns / alpha

Notes
-----
• Modes are implemented as simple exposure multipliers:
      standard          -> 1.00x
      alpha_minus_beta  -> 0.80x (de-risked)
      private_logic     -> 1.20x (risk-on)
• History is daily close-to-close, auto-adjusted prices.
• Benchmarks are mapped per Wave name; default is SPY.
"""

import os
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # Streamlit will show a nicer error via RuntimeError


# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"

# Optional: if in the future we pre-compute long histories into a CSV,
# we can wire it up here. For now the engine always builds from prices.
FULL_WAVE_HISTORY_CSV = "Full_Wave_History.csv"

_wave_weights_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

MODE_SETTINGS: Dict[str, Dict[str, float]] = {
    "standard": {
        "exposure": 1.00,
    },
    "alpha_minus_beta": {
        "exposure": 0.80,  # dial down beta
    },
    "private_logic": {
        "exposure": 1.20,  # slightly levered flavour
    },
}

# Very simple benchmark mapping. If nothing matches, we fall back to SPY.
WAVE_BENCHMARKS: Dict[str, str] = {
    "s&p 500 wave": "SPY",
    "ai wave": "QQQ",
    "cloud & software wave": "QQQ",
    "crypto income wave": "BTC-USD",
    "future power & energy wave": "XLE",
    "clean transit-infrastructure wave": "IYT",
    "small cap growth wave": "IWM",
    "growth wave": "SPYG",
    "income wave": "AGG",
    "quantum computing wave": "QQQ",
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

    Returns DataFrame with:
        wave   (str)
        ticker (str, uppercased)
        weight (float, normalized to sum to 1 within each Wave)
    """
    global _wave_weights_cache

    if _wave_weights_cache is not None:
        return _wave_weights_cache.copy()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found in repo root. "
            f"Make sure you committed the latest wave_weights.csv."
        )

    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"wave", "ticker", "weight"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing columns: {missing}")

    # Clean values
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        bad = df[df["weight"].isna()]
        raise ValueError(
            "wave_weights.csv has non-numeric weights. "
            f"Offending rows:\n{bad}"
        )

    # Remove non-positive weights
    df = df[df["weight"] > 0].copy()
    if df.empty:
        raise ValueError("wave_weights.csv has no positive weights")

    # Normalize to 1.0 within each Wave
    grouped = df.groupby("wave")["weight"].transform("sum")
    df["weight"] = df["weight"] / grouped

    _wave_weights_cache = df.copy()
    return df


# ---------------------------------------------------------------------------
# Public API: Wave list
# ---------------------------------------------------------------------------

def get_available_waves() -> List[str]:
    """Return a sorted list of Wave names discovered in wave_weights.csv."""
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


# ---------------------------------------------------------------------------
# Helpers: benchmarks, prices, exposure
# ---------------------------------------------------------------------------

def _get_benchmark_for_wave(wave_name: str) -> str:
    """Return benchmark ticker for a Wave (rough mapping, fallback to SPY)."""
    key = wave_name.lower()
    for name_substring, ticker in WAVE_BENCHMARKS.items():
        if name_substring in key:
            return ticker
    return "SPY"


def _get_mode_exposure(mode: str) -> float:
    """Return exposure multiplier for a mode."""
    mode = (mode or "standard").lower()
    if mode in MODE_SETTINGS:
        return MODE_SETTINGS[mode]["exposure"]
    return MODE_SETTINGS["standard"]["exposure"]


def _normalize_price_panel(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Normalize yfinance download output into a [date x ticker] Close-price DataFrame.

    Handles:
        • Single ticker with 'Close' column
        • Multi-ticker with group_by='ticker' MultiIndex style
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()

    # Case 1: single ticker, columns include 'Close'
    if "Close" in data.columns:
        # data["Close"] is a Series; to_frame gives us [date x 1] DataFrame.
        t = tickers[0] if tickers else "TICKER"
        px = data["Close"].to_frame(name=t)
    else:
        # Case 2: group_by='ticker' style: data[ticker]["Close"]
        close_map: Dict[str, pd.Series] = {}
        for t in tickers:
            try:
                series = data[t]["Close"].dropna()
                close_map[t] = series
            except Exception:
                # If we can't get data for a ticker, keep empty series
                close_map[t] = pd.Series(dtype=float)
        px = pd.DataFrame(close_map)

    # Clean index
    px = px.sort_index()
    px = px.loc[~px.index.duplicated(keep="last")]
    return px


def _fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get latest close price for each ticker.

    Returns {ticker: price}. Missing quotes become np.nan but never crash.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment")

    if not tickers:
        return {}

    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    px = _normalize_price_panel(data, tickers)
    latest_prices: Dict[str, float] = {}

    if px.empty:
        # No data at all; return NaNs
        for t in tickers:
            latest_prices[t] = np.nan
        return latest_prices

    last_row = px.dropna(how="all").iloc[-1]
    for t in tickers:
        price = float(last_row.get(t, np.nan))
        latest_prices[t] = price

    return latest_prices


def _build_history_from_prices(
    wave_name: str,
    mode: str = "standard",
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Build daily NAV history for a Wave from yfinance prices.

    • Constant portfolio weights, as-of today.
    • NAV starts at 1.0
    • Returns:
        date
        wave_nav, wave_return
        bench_nav, bench_return
        cum_wave_return, cum_bench_return
        daily_alpha, cum_alpha
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment")

    exposure = _get_mode_exposure(mode)

    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days))

    # ------------------ Wave prices ------------------
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    px = _normalize_price_panel(data, tickers)
    if px.empty:
        raise ValueError(f"No price history returned for tickers: {tickers}")

    # Align weights to columns
    weights = sub.set_index("ticker")["weight"]
    weights = weights.reindex(px.columns).fillna(0.0).values.reshape(1, -1)

    # Compute daily returns & NAV
    ret = px.pct_change().fillna(0.0)
    port_ret = (ret * weights).sum(axis=1)

    # Mode exposure adjustment
    port_ret = port_ret * exposure

    wave_nav = (1.0 + port_ret).cumprod()

    hist = pd.DataFrame(
        {
            "date": px.index.date,
            "wave_nav": wave_nav.values,
            "wave_return": port_ret.values,
        }
    )

    # ------------------ Benchmark series ------------------
    bench_ticker = _get_benchmark_for_wave(wave_name)
    try:
        bench_data = yf.download(
            tickers=bench_ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if "Close" in bench_data.columns:
            bench_px = bench_data["Close"].dropna()
        else:
            # group_by='ticker' format
            bench_px = bench_data[bench_ticker]["Close"].dropna()
    except Exception:
        bench_px = pd.Series(dtype=float)

    if bench_px.empty:
        # Fallback: flat benchmark (0% return)
        hist["bench_nav"] = 1.0
        hist["bench_return"] = 0.0
    else:
        # Align with hist dates
        bench_px = bench_px.sort_index()
        bench_px = bench_px.loc[~bench_px.index.duplicated(keep="last")]
        bench_px = bench_px.reindex(px.index).ffill()

        bench_ret = bench_px.pct_change().fillna(0.0)
        bench_nav = (1.0 + bench_ret).cumprod()

        hist["bench_nav"] = bench_nav.values
        hist["bench_return"] = bench_ret.values

    # ------------------ Cumulative returns & alpha ------------------
    hist["cum_wave_return"] = (1.0 + hist["wave_return"]).cumprod() - 1.0
    hist["cum_bench_return"] = (1.0 + hist["bench_return"]).cumprod() - 1.0
    hist["daily_alpha"] = hist["wave_return"] - hist["bench_return"]
    hist["cum_alpha"] = hist["cum_wave_return"] - hist["cum_bench_return"]

    return hist


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
          "wave": wave_name,
          "mode": mode,
          "as_of": <date>,
          "positions": DataFrame[
              ticker, weight, eff_weight, price, dollar_weight
          ],
        }

    'mode' and **kwargs are accepted for compatibility with older UI.
    """
    df_w = _load_wave_weights()
    sub = df_w[df_w["wave"] == wave_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for Wave '{wave_name}' in wave_weights.csv")

    tickers = sub["ticker"].unique().tolist()
    prices = _fetch_latest_prices(tickers)

    exposure = _get_mode_exposure(mode)
    sub["price"] = sub["ticker"].map(prices)
    sub["eff_weight"] = sub["weight"] * exposure

    # Arbitrary $100 base just for "dollar_weight" display
    base_notional = 100.0
    sub["dollar_weight"] = sub["eff_weight"] * base_notional

    as_of = as_of or dt.date.today()

    positions = sub[["ticker", "weight", "eff_weight", "price", "dollar_weight"]]

    return {
        "wave": wave_name,
        "mode": mode,
        "as_of": as_of,
        "positions": positions,
    }


def get_wave_history(
    wave_name: str,
    mode: str = "standard",
    lookback_days: int = 365,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Primary daily history function (v2).

    Right now this:
        • builds daily NAV history from prices via yfinance
        • implements modes as exposure multipliers
        • computes benchmark NAV + alpha
    """
    return _build_history_from_prices(
        wave_name=wave_name,
        mode=mode,
        lookback_days=lookback_days,
    )