"""
waves_engine.py — WAVES Intelligence™ Engine (History-Aware)

Core responsibilities
---------------------
• Load Wave definitions from wave_weights.csv   (Wave, Ticker, Weight).
• Ensure Full_Wave_History.csv exists and is populated:
    - If missing / empty → call build_full_wave_history.build_full_history().
• For each Wave, provide:
    - 30D / 60D / 1Y / SI returns
    - 30D / 60D / 1Y / SI alpha vs benchmark
    - 1Y volatility, max drawdown, beta, info ratio, hit rate
    - Optional SmartSafe regime based on VIX
    - Latest positions snapshot (weights + last price + market value)
• Simple public API for app.py:
    - get_available_waves()
    - get_wave_snapshot(wave: str, mode: str = "standard") -> dict
    - get_wave_history(wave: str) -> pd.DataFrame
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: used only for SmartSafe / VIX overlays
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

# Filenames
WAVE_WEIGHTS_FILE = "wave_weights.csv"
FULL_HISTORY_FILE = "Full_Wave_History.csv"

# Cache containers
_WEIGHTS_DF: pd.DataFrame | None = None
_HISTORY_DF: pd.DataFrame | None = None


# ---------------------------------------------------------------------
# Basic loaders
# ---------------------------------------------------------------------


def _load_wave_weights() -> pd.DataFrame:
    """Load and normalize wave_weights.csv (Wave / Ticker / Weight)."""
    global _WEIGHTS_DF
    if _WEIGHTS_DF is not None:
        return _WEIGHTS_DF

    if not os.path.exists(WAVE_WEIGHTS_FILE):
        raise FileNotFoundError(
            f"{WAVE_WEIGHTS_FILE} not found in repo root. "
            "This file must exist with columns: Wave,Ticker,Weight."
        )

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    # Normalize column names
    df.columns = [c.replace("\ufeff", "").strip().lower() for c in df.columns]

    expected = {"wave", "ticker", "weight"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{WAVE_WEIGHTS_FILE} is missing required columns: {missing}"
        )

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    if df["weight"].isna().any():
        bad = df[df["weight"].isna()]
        raise ValueError(
            f"Non-numeric weights found in {WAVE_WEIGHTS_FILE}:\n{bad}"
        )

    # Normalize weights per Wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda s: s / s.sum() if s.sum() != 0 else s
    )

    _WEIGHTS_DF = df
    return df


def get_available_waves() -> List[str]:
    """Return a sorted list of Wave names from wave_weights.csv."""
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


# ---------------------------------------------------------------------
# Full_Wave_History.csv loader  (auto-build if missing)
# ---------------------------------------------------------------------


def _ensure_full_history() -> pd.DataFrame:
    """
    Load Full_Wave_History.csv.

    If it does not exist or has no data rows, call
    build_full_wave_history.build_full_history() to generate it.
    """
    global _HISTORY_DF

    if _HISTORY_DF is not None:
        return _HISTORY_DF

    needs_build = False

    if not os.path.exists(FULL_HISTORY_FILE):
        needs_build = True
    else:
        try:
            tmp = pd.read_csv(FULL_HISTORY_FILE)
            if tmp.shape[0] <= 1:
                needs_build = True
        except Exception:
            needs_build = True

    if needs_build:
        try:
            from build_full_wave_history import build_full_history  # type: ignore

            build_full_history()
        except Exception as e:
            raise RuntimeError(
                f"Could not auto-build {FULL_HISTORY_FILE}: {e}"
            )

    df = pd.read_csv(FULL_HISTORY_FILE)

    # Basic normalization
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # Expect core columns
    required = {
        "Date",
        "Wave",
        "Position",
        "Weight",
        "Ticker",
        "Price",
        "MarketValue",
        "NAV",
        "WaveReturn",
        "BenchReturn",
        "Alpha",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{FULL_HISTORY_FILE} is missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Wave"] = df["Wave"].astype(str)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()

    _HISTORY_DF = df
    return df


def get_wave_history(wave: str) -> pd.DataFrame:
    """
    Return history for a given Wave as a daily time series.

    Output columns (per date, one row per date):
        Date
        NAV
        WaveReturn
        BenchReturn
        Alpha
    """
    hist = _ensure_full_history()
    sub = hist[hist["Wave"] == wave].copy()
    if sub.empty:
        raise ValueError(f"No historical data found for Wave '{wave}'")

    # One row per date: take the first row (all tickers share same NAV/returns)
    sub = (
        sub.sort_values(["Date", "Position"])
        .groupby("Date")
        .agg(
            {
                "NAV": "first",
                "WaveReturn": "first",
                "BenchReturn": "first",
                "Alpha": "first",
            }
        )
        .reset_index()
    )

    sub = sub.sort_values("Date")
    return sub


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------


def _window_indices(dates: pd.Series, n: int) -> pd.Series:
    """Return last n dates (or fewer if history shorter)."""
    if len(dates) <= n:
        return dates
    return dates.iloc[-n:]


def _total_return(ret: pd.Series) -> float:
    """Convert daily returns to cumulative return."""
    if ret.empty:
        return 0.0
    return float((1.0 + ret).prod() - 1.0)


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return float(dd.min())


def _beta_and_ir(
    wave_ret: pd.Series, bench_ret: pd.Series
) -> Tuple[float, float, float]:
    """
    Return (beta, info_ratio, hit_rate) for daily returns.
    """
    idx = wave_ret.index.intersection(bench_ret.index)
    if len(idx) < 30:
        return 0.0, 0.0, 0.0

    rp = wave_ret.loc[idx]
    rb = bench_ret.loc[idx]

    # Beta
    cov = np.cov(rp, rb)[0, 1]
    var_b = np.var(rb)
    if var_b <= 0:
        beta = 0.0
    else:
        beta = cov / var_b

    # Information ratio
    diff = rp - rb
    mean_excess = diff.mean()
    vol_excess = diff.std()
    if vol_excess <= 0:
        ir = 0.0
    else:
        ir = (mean_excess * 252.0) / (vol_excess * np.sqrt(252.0))

    # Hit rate
    hits = (rp > rb).sum()
    hit_rate = hits / len(idx)

    return float(beta), float(ir), float(hit_rate)


def _vol_1y(wave_ret: pd.Series) -> float:
    if len(wave_ret) < 21:
        return 0.0
    return float(wave_ret.std() * np.sqrt(252.0))


# ---------------------------------------------------------------------
# SmartSafe / VIX helper (lightweight)
# ---------------------------------------------------------------------


def _get_vix_level() -> float:
    """Fetch recent VIX level via yfinance (if available)."""
    if yf is None:
        return 18.0  # reasonable default
    try:
        data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if data.empty:
            return 18.0
        # prefer "Adj Close" if it exists
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        v = float(data[col].iloc[-1])
        if np.isnan(v):
            return 18.0
        return v
    except Exception:
        return 18.0


def _smartsafe_regime(vix_level: float) -> Tuple[str, float]:
    """
    Map VIX level to a simple SmartSafe regime + sweep fraction.

    This is intentionally simple and display-oriented; the
    underlying engine logic for SmartSafe can be more complex.
    """
    if vix_level < 18:
        return "Normal", 0.00
    if vix_level < 24:
        return "Caution", 0.10
    if vix_level < 30:
        return "Stress", 0.20
    return "Panic", 0.40


# ---------------------------------------------------------------------
# Positions snapshot builder
# ---------------------------------------------------------------------


def _get_latest_positions(wave: str) -> pd.DataFrame:
    """
    Return a positions DataFrame for the latest date in history for this Wave.

    Columns:
        Ticker, Weight, Price, MarketValue
    """
    weights_df = _load_wave_weights()
    hist = _ensure_full_history()

    w_slice = weights_df[weights_df["wave"] == wave].copy()
    if w_slice.empty:
        raise ValueError(f"No positions found for Wave '{wave}' in weights file.")

    h_slice = hist[hist["Wave"] == wave].copy()
    if h_slice.empty:
        # If no history, just return weights with NaN prices
        out = w_slice[["ticker", "weight"]].rename(
            columns={"ticker": "Ticker", "weight": "Weight"}
        )
        out["Price"] = np.nan
        out["MarketValue"] = np.nan
        return out

    latest_date = h_slice["Date"].max()
    latest_rows = h_slice[h_slice["Date"] == latest_date].copy()

    # Merge weights + latest prices
    merged = w_slice.merge(
        latest_rows[["Ticker", "Price", "MarketValue"]],
        left_on="ticker",
        right_on="Ticker",
        how="left",
    )

    merged = merged[["ticker", "weight", "Price", "MarketValue"]]
    merged = merged.rename(columns={"ticker": "Ticker", "weight": "Weight"})

    # Sort descending by Weight
    merged = merged.sort_values("Weight", ascending=False).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------
# Public API: get_wave_snapshot
# ---------------------------------------------------------------------


def get_wave_snapshot(wave: str, mode: str = "standard") -> Dict[str, object]:
    """
    Build the full snapshot for a Wave, used directly by app.py.

    Returns:
        {
            "wave": wave_name,
            "mode": mode,
            "benchmark_label": "<human text>",
            "metrics": { ... numeric stats ... },
            "positions": <DataFrame>,
            "history": <DataFrame of Date,NAV,WaveReturn,BenchReturn,Alpha>
        }
    """

    # Mode normalization
    mode = (mode or "standard").strip().lower()

    # Load data
    hist_series = get_wave_history(wave)
    dates = hist_series["Date"]
    nav = hist_series["NAV"]
    wave_ret = hist_series["WaveReturn"]
    bench_ret = hist_series["BenchReturn"]
    alpha_daily = hist_series["Alpha"]

    # Windows: approximate trading-day lookbacks
    idx_30 = _window_indices(dates, 30)
    idx_60 = _window_indices(dates, 60)
    idx_252 = _window_indices(dates, 252)

    # Reindex return series by these windows
    mask_30 = dates.isin(idx_30)
    mask_60 = dates.isin(idx_60)
    mask_252 = dates.isin(idx_252)

    wave_30 = wave_ret[mask_30]
    bench_30 = bench_ret[mask_30]

    wave_60 = wave_ret[mask_60]
    bench_60 = bench_ret[mask_60]

    wave_1y = wave_ret[mask_252]
    bench_1y = bench_ret[mask_252]

    # Cumulative returns & alpha
    ret_30d = _total_return(wave_30)
    bench_ret_30d = _total_return(bench_30)
    alpha_30d = ret_30d - bench_ret_30d

    ret_60d = _total_return(wave_60)
    bench_ret_60d = _total_return(bench_60)
    alpha_60d = ret_60d - bench_ret_60d

    ret_1y = _total_return(wave_1y)
    bench_ret_1y = _total_return(bench_1y)
    alpha_1y = ret_1y - bench_ret_1y

    ret_si = _total_return(wave_ret)
    bench_ret_si = _total_return(bench_ret)
    alpha_si = ret_si - bench_ret_si

    # Risk statistics
    beta_1y, ir_1y, hit_rate_1y = _beta_and_ir(wave_1y, bench_1y)
    vol_1y = _vol_1y(wave_1y)
    maxdd = _max_drawdown(nav)

    # SmartSafe state
    vix_level = _get_vix_level()
    smartsafe_state, smartsafe_sweep = _smartsafe_regime(vix_level)

    # Positions snapshot
    positions_df = _get_latest_positions(wave)

    # History DataFrame (for charts in app.py)
    history_df = hist_series.copy()

    # Benchmark label: we infer from daily bench_ret vs SPY/QQQ patterns
    # For now, just a simple label, can be upgraded later.
    benchmark_label = "Custom Benchmark"

    metrics = {
        "mode": mode,
        "ret_30d": ret_30d,
        "alpha_30d": alpha_30d,
        "bench_ret_30d": bench_ret_30d,
        "ret_60d": ret_60d,
        "alpha_60d": alpha_60d,
        "bench_ret_60d": bench_ret_60d,
        "ret_1y": ret_1y,
        "alpha_1y": alpha_1y,
        "bench_ret_1y": bench_ret_1y,
        "ret_si": ret_si,
        "alpha_si": alpha_si,
        "bench_ret_si": bench_ret_si,
        "vol_1y": vol_1y,
        "maxdd": maxdd,
        "beta_1y": beta_1y,
        "info_ratio_1y": ir_1y,
        "hit_rate_1y": hit_rate_1y,
        "vix_level": vix_level,
        "smartsafe_state": smartsafe_state,
        "smartsafe_sweep": smartsafe_sweep,
    }

    return {
        "wave": wave,
        "mode": mode,
        "benchmark_label": benchmark_label,
        "metrics": metrics,
        "positions": positions_df,
        "history": history_df,
    }