"""
waves_engine.py — WAVES Intelligence™ Vector Engine (Restored w/ compute_history_nav)

Core responsibilities
---------------------
- Load universe & wave weights from CSVs.
- Discover all Waves automatically.
- Build per-Wave NAV history over a lookback window.
- Prefer Full_Wave_History.csv for historical NAV if available.
- Fallback to yfinance-based price history when necessary.
- Expose compute_history_nav() for Streamlit app.
- Provide portfolio-level overview stats (NAV, 365D return, 30D return).
- Provide positions/top holdings for each Wave.

Assumptions about CSVs
----------------------
- wave_weights.csv:
    columns at minimum:
        Wave        : str (Wave name)
        Ticker      : str (ticker symbol)
        Weight      : float (target weight, any scale; will be normalized per-Wave)

- Full_Wave_History.csv (optional but preferred for high accuracy):
    columns at minimum:
        date        : str or datetime (YYYY-MM-DD)
        Wave        : str (Wave name, matching wave_weights.csv)
        NAV         : float  (NAV level; arbitrary starting point)

All file paths are relative to the project root by default.
"""

import os
import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # The app will show a warning if fallback is needed.


# -----------------------------#
# Configuration / Paths
# -----------------------------#

DATA_DIR = "data"
LOG_DIR = "logs"
WAVE_WEIGHTS_FILE = os.path.join(DATA_DIR, "wave_weights.csv")
FULL_HISTORY_FILE = os.path.join(DATA_DIR, "Full_Wave_History.csv")

# Default lookbacks
DEFAULT_LOOKBACK_DAYS = 365
SHORT_LOOKBACK_DAYS = 30

# Simple mode multipliers (can be tuned later)
MODE_MULTIPLIERS = {
    "Standard": 1.0,
    "Alpha-Minus-Beta": 0.8,   # Slightly lower volatility
    "Private Logic": 1.15,     # Slightly higher risk/return
}


# -----------------------------#
# Utility helpers
# -----------------------------#

def _ensure_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure df has a proper datetime index based on date_col."""
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise ValueError("DataFrame must have a 'date' column or DatetimeIndex.")
    return df


def _load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV if it exists; otherwise return None."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[waves_engine] Error loading CSV '{path}': {e}")
        return None


# -----------------------------#
# Core data loading
# -----------------------------#

def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv and normalize weights per Wave.

    Returns
    -------
    DataFrame with columns:
        Wave, Ticker, Weight (normalized so sum per Wave = 1.0)
    """
    df = _load_csv_safe(WAVE_WEIGHTS_FILE)
    if df is None:
        raise FileNotFoundError(f"wave_weights.csv not found at {WAVE_WEIGHTS_FILE}")

    required_cols = {"Wave", "Ticker", "Weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    df = df.copy()
    # Clean whitespace
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Normalize weights per Wave
    df["Weight"] = df["Weight"].astype(float)
    weight_sums = df.groupby("Wave")["Weight"].transform("sum")
    df["Weight"] = df["Weight"] / weight_sums

    return df


def get_all_waves() -> List[str]:
    """Return sorted list of all Waves."""
    df = load_wave_weights()
    waves = sorted(df["Wave"].unique().tolist())
    return waves


def get_wave_positions(wave_name: str) -> pd.DataFrame:
    """
    Return positions (weights) for a specific Wave.

    Columns: Ticker, Weight, Wave
    """
    df = load_wave_weights()
    mask = df["Wave"].str.lower() == wave_name.lower()
    wave_df = df.loc[mask].copy()
    if wave_df.empty:
        raise ValueError(f"No positions found for Wave '{wave_name}'")
    return wave_df[["Wave", "Ticker", "Weight"]]


# -----------------------------#
# NAV / Performance logic
# -----------------------------#

def _load_full_history() -> Optional[pd.DataFrame]:
    """Load Full_Wave_History.csv if it exists."""
    hist = _load_csv_safe(FULL_HISTORY_FILE)
    if hist is None:
        return None

    # Normalize column names
    cols_lower = {c.lower(): c for c in hist.columns}
    # Expect at least "date", "wave", "nav"
    date_col = cols_lower.get("date")
    wave_col = cols_lower.get("wave")
    nav_col = cols_lower.get("nav")

    if not all([date_col, wave_col, nav_col]):
        raise ValueError("Full_Wave_History.csv must have 'date', 'Wave', 'NAV' columns (case insensitive).")

    hist = hist.rename(columns={date_col: "date", wave_col: "Wave", nav_col: "NAV"})
    hist["Wave"] = hist["Wave"].astype(str).str.strip()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values(["Wave", "date"])
    return hist


def _compute_nav_from_prices(price_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Compute NAV series from price history and weights.

    Parameters
    ----------
    price_df : DataFrame
        Columns: tickers, index = date (DatetimeIndex)
    weights : Series
        index: tickers, values: normalized weights (sum to 1)

    Returns
    -------
    DataFrame
        index: date
        columns: ['NAV', 'Return']
    """
    price_df = price_df.copy()
    price_df = price_df[weights.index]  # align
    price_df = price_df.sort_index()

    # Daily returns
    returns = price_df.pct_change().fillna(0.0)

    # Portfolio return as weighted sum
    port_ret = (returns * weights).sum(axis=1)

    # NAV as cumulative product, normalized to 1.0 at start
    nav = (1.0 + port_ret).cumprod()
    result = pd.DataFrame({"NAV": nav, "Return": port_ret})
    return result


def _fetch_price_history(tickers: List[str], lookback_days: int) -> Optional[pd.DataFrame]:
    """
    Fetch adjusted close price history from yfinance.

    Returns DataFrame with index=date, columns=tickers.
    """
    if yf is None:
        print("[waves_engine] yfinance not available; cannot fetch history.")
        return None

    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=lookback_days + 5)  # small buffer

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date + dt.timedelta(days=1),
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
        # yfinance can return different shapes depending on tickers count.
        if isinstance(data.columns, pd.MultiIndex):
            # Pull 'Adj Close' from multiindex
            frames = []
            for ticker in tickers:
                if (ticker, "Adj Close") in data.columns:
                    series = data[(ticker, "Adj Close")].rename(ticker)
                    frames.append(series)
            if not frames:
                return None
            prices = pd.concat(frames, axis=1)
        else:
            # Single ticker case
            if "Adj Close" in data.columns:
                prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
            else:
                return None

        prices.index = pd.to_datetime(prices.index)
        return prices
    except Exception as e:
        print(f"[waves_engine] Error fetching yfinance data: {e}")
        return None


def compute_history_nav(
    wave_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    mode: str = "Standard",
) -> pd.DataFrame:
    """
    Compute NAV history for a Wave over the given lookback window.

    Priority:
        1) Use Full_Wave_History.csv if it contains this Wave;
        2) Otherwise compute from yfinance prices and weights.

    Mode-specific logic:
        - Applies a simple multiplicative scaling to daily returns based on MODE_MULTIPLIERS.

    Returns
    -------
    DataFrame with columns:
        NAV         : normalized NAV series
        Return      : daily returns
        CumReturn   : cumulative return over window (at each point)
    Index
        DatetimeIndex (date)
    """
    mode = mode or "Standard"
    multiplier = MODE_MULTIPLIERS.get(mode, 1.0)

    # 1) Try Full_Wave_History.csv
    full_hist = _load_full_history()
    if full_hist is not None:
        mask_wave = full_hist["Wave"].str.lower() == wave_name.lower()
        hist_wave = full_hist.loc[mask_wave].copy()
        if not hist_wave.empty:
            # Filter by lookback window
            cutoff = dt.datetime.utcnow().date() - dt.timedelta(days=lookback_days)
            hist_wave = hist_wave[hist_wave["date"] >= pd.to_datetime(cutoff)]
            hist_wave = hist_wave.sort_values("date")
            if not hist_wave.empty:
                hist_wave = hist_wave.set_index("date")
                # Normalize NAV to 1.0 at start of this window
                first_nav = hist_wave["NAV"].iloc[0]
                nav = hist_wave["NAV"] / float(first_nav)
                # Daily returns from NAV
                ret = nav.pct_change().fillna(0.0) * multiplier
                nav_scaled = (1.0 + ret).cumprod()
                cum_ret = nav_scaled / nav_scaled.iloc[0] - 1.0
                result = pd.DataFrame(
                    {
                        "NAV": nav_scaled,
                        "Return": ret,
                        "CumReturn": cum_ret,
                    }
                )
                return result

    # 2) Fallback to yfinance-based computation
    positions = get_wave_positions(wave_name)
    tickers = positions["Ticker"].tolist()
    weights = positions.set_index("Ticker")["Weight"]

    price_df = _fetch_price_history(tickers, lookback_days)
    if price_df is None or price_df.empty:
        raise RuntimeError(f"Unable to compute NAV history for '{wave_name}': no price data available.")

    nav_df = _compute_nav_from_prices(price_df, weights)
    # Apply mode multiplier to returns
    nav_df["Return"] = nav_df["Return"] * multiplier
    nav_df["NAV"] = (1.0 + nav_df["Return"]).cumprod()
    nav_df["CumReturn"] = nav_df["NAV"] / nav_df["NAV"].iloc[0] - 1.0
    return nav_df


def get_wave_summary_metrics(
    wave_name: str,
    mode: str = "Standard",
    long_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    short_lookback_days: int = SHORT_LOOKBACK_DAYS,
) -> Dict[str, float]:
    """
    Compute summary metrics (last NAV, 365D return, 30D return).

    Returns
    -------
    dict with keys:
        nav_last
        ret_long
        ret_short
    """
    hist_long = compute_history_nav(wave_name, long_lookback_days, mode)
    nav_last = float(hist_long["NAV"].iloc[-1])
    ret_long = float(hist_long["CumReturn"].iloc[-1])

    # Short window metrics
    if len(hist_long) > short_lookback_days:
        short_slice = hist_long.iloc[-short_lookback_days:]
        ret_short = float(short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0)
    else:
        ret_short = float(ret_long)

    return {
        "nav_last": nav_last,
        "ret_long": ret_long,
        "ret_short": ret_short,
    }


def get_portfolio_overview(
    mode: str = "Standard",
    long_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    short_lookback_days: int = SHORT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Build a portfolio-level overview for all Waves.

    Columns:
        Wave
        NAV_last
        Return_365D
        Return_30D
    """
    waves = get_all_waves()
    rows = []
    for w in waves:
        try:
            metrics = get_wave_summary_metrics(
                wave_name=w,
                mode=mode,
                long_lookback_days=long_lookback_days,
                short_lookback_days=short_lookback_days,
            )
            rows.append(
                {
                    "Wave": w,
                    "NAV_last": metrics["nav_last"],
                    "Return_365D": metrics["ret_long"],
                    "Return_30D": metrics["ret_short"],
                }
            )
        except Exception as e:
            print(f"[waves_engine] Error computing overview for Wave '{w}': {e}")
            rows.append(
                {
                    "Wave": w,
                    "NAV_last": float("nan"),
                    "Return_365D": float("nan"),
                    "Return_30D": float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("Wave").reset_index(drop=True)
    return df