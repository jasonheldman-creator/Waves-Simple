"""
waves_engine.py — WAVES Intelligence™ Engine
(Restored Baseline + Upgraded Performance)

This version:
- Auto-discovers Waves from wave_weights.csv
- Uses SmartSafe 2.0 *hook only* (no SmartSafe 3.0 logic)
- Computes intraday, 30-day, and 60-day performance and alpha vs benchmark
- Has more robust price download logic
- Falls back to last logged performance metrics if live pulls fail

Assumptions:
- wave_weights.csv: columns for Wave, Ticker, Weight (any reasonable capitalization/spaces)
- list.csv: optional universe with a Ticker column
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR
WEIGHTS_CSV = DATA_DIR / "wave_weights.csv"
UNIVERSE_CSV = DATA_DIR / "list.csv"

LOGS_DIR = BASE_DIR / "logs"
POSITIONS_LOG_DIR = LOGS_DIR / "positions"
PERF_LOG_DIR = LOGS_DIR / "performance"

for d in [LOGS_DIR, POSITIONS_LOG_DIR, PERF_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Benchmarks per Wave (fallback to SPY if not found)
BENCHMARK_MAP: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "S&P Wave": "SPY",
    "AI Wave": "QQQ",
    "AI & Innovation Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",
    "Future Power & Energy Wave": "XLE",
    "Clean Transit-Infrastructure Wave": "IYT",
    "Crypto Income Wave": "BITO",
    "Income Wave": "SCHD",
    "Small Cap Growth Wave": "IWM",
    "Small to Mid Cap Growth Wave": "VO",
    "Cloud & Software Wave": "IGV",
    "SmartSafe Money Market Wave": "BIL",
}

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on headers; keep casing but clean up weird spacing."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _standardize_column(df: pd.DataFrame, candidates: List[str], target: str) -> pd.DataFrame:
    """
    Given a DataFrame and a list of candidate column names, rename the first one found to `target`.

    Matching:
    - case-insensitive
    - whitespace-insensitive
    """
    df = _normalize_headers(df)

    norm_to_original = {str(col).strip().lower(): col for col in df.columns}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_to_original:
            original = norm_to_original[key]
            if original != target:
                df = df.rename(columns={original: target})
            break

    return df


def _load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        return pd.DataFrame(columns=["ticker"])

    df = pd.read_csv(UNIVERSE_CSV)
    df = _standardize_column(df, ["Ticker", "ticker", "Symbol", "symbol"], "ticker")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def _load_wave_weights() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError(f"wave_weights.csv not found at {WEIGHTS_CSV}")

    df = pd.read_csv(WEIGHTS_CSV)
    df = _normalize_headers(df)

    df = _standardize_column(df, ["Wave", "wave", "Portfolio", "portfolio", "Name"], "wave")
    df = _standardize_column(df, ["Ticker", "ticker", "Symbol", "symbol"], "ticker")
    df = _standardize_column(df, ["Weight", "weight", "Wgt", "wgt"], "weight")

    required_cols = ["wave", "ticker", "weight"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in wave_weights.csv: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    grouped = df.groupby(["wave", "ticker"], as_index=False)["weight"].sum()
    grouped["weight"] = grouped.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )
    return grouped


def get_available_waves() -> List[str]:
    """Return sorted list of distinct wave names from wave_weights.csv."""
    weights = _load_wave_weights()
    if "wave" not in weights.columns:
        raise ValueError(
            f"'wave' column not found after normalization. Columns: {list(weights.columns)}"
        )
    waves = sorted(weights["wave"].unique().tolist())
    return waves


def _get_benchmark_for_wave(wave_name: str) -> str:
    """Map a wave to its benchmark ticker. Fallback to SPY."""
    if wave_name in BENCHMARK_MAP:
        return BENCHMARK_MAP[wave_name]

    name_lower = wave_name.lower()
    if "crypto" in name_lower or "digital" in name_lower:
        return "BITO"
    if "ai" in name_lower or "tech" in name_lower or "cloud" in name_lower:
        return "QQQ"
    if "small" in name_lower:
        return "IWM"
    if "income" in name_lower:
        return "SCHD"
    if "smartsafe" in name_lower:
        return "BIL"
    if "s&p" in name_lower:
        return "SPY"
    return "SPY"


def _download_price_series(ticker: str, period: str = "180d") -> pd.Series:
    """
    Download daily adjusted close for a single ticker.

    More robust:
    - Tries yf.download first.
    - If Adj Close missing, falls back to Close.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.Series(dtype=float)

        col = None
        for candidate in ["Adj Close", "Close"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            return pd.Series(dtype=float)

        s = df[col].copy()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float)


def _get_fast_intraday_return(ticker: str) -> Tuple[float, float]:
    """
    Use fast_info to approximate intraday return for a single ticker.
    Returns (last_price, pct_return).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        last_price = float(info.get("last_price", np.nan))
        prev_close = float(info.get("previous_close", np.nan))
        if np.isfinite(last_price) and np.isfinite(prev_close) and prev_close != 0:
            ret = (last_price / prev_close) - 1.0
        else:
            ret = 0.0
        return last_price, ret
    except Exception:
        return np.nan, 0.0


# ---------------------------------------------------------------------
# SmartSafe 2.0 (placeholder hook)
# ---------------------------------------------------------------------


def apply_smartsafe_sweep(positions: pd.DataFrame) -> pd.DataFrame:
    """
    SmartSafe 2.0 sweep *hook*.

    IMPORTANT:
    - This is intentionally minimal and non-destructive.
    - No SmartSafe 3.0 logic in this file.

    Currently: returns positions unchanged.
    """
    return positions


# ---------------------------------------------------------------------
# Performance & logging
# ---------------------------------------------------------------------


def _compute_portfolio_trailing_returns(
    positions: pd.DataFrame,
    benchmark_ticker: str,
    period: str = "180d",
) -> Dict[str, float]:
    """
    Compute 30d and 60d total returns and alpha vs benchmark.

    Returns keys:
        ret_30d, ret_60d, alpha_30d, alpha_60d
    """
    if positions.empty:
        return {
            "ret_30d": 0.0,
            "ret_60d": 0.0,
            "alpha_30d": 0.0,
            "alpha_60d": 0.0,
        }

    tickers = positions["ticker"].unique().tolist()
    weights = positions.set_index("ticker")["weight"]
    weights = weights / weights.sum() if weights.sum() != 0 else weights

    # Download history for each ticker
    price_frames = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if not s.empty:
            price_frames.append(s)

    if not price_frames:
        return {
            "ret_30d": 0.0,
            "ret_60d": 0.0,
            "alpha_30d": 0.0,
            "alpha_60d": 0.0,
        }

    prices_df = pd.concat(price_frames, axis=1).sort_index()
    prices_df = prices_df.ffill().dropna(how="all")

    # Align weights to columns (missing tickers -> 0)
    aligned_weights = weights.reindex(prices_df.columns).fillna(0.0)
    if aligned_weights.sum() != 0:
        aligned_weights = aligned_weights / aligned_weights.sum()

    # Portfolio daily values and returns
    port_values = (prices_df * aligned_weights).sum(axis=1)
    port_returns = port_values.pct_change().dropna()

    # Benchmark
    bench_series = _download_price_series(benchmark_ticker, period=period)
    if bench_series.empty:
        bench_returns = pd.Series(0.0, index=port_returns.index)
    else:
        bench_series = bench_series.reindex(port_returns.index).ffill()
        bench_returns = bench_series.pct_change().dropna()
        port_returns, bench_returns = port_returns.align(bench_returns, join="inner")

    def _window_total_return(r: pd.Series, days: int) -> float:
        if r.empty:
            return 0.0
        sub = r.tail(days)
        if sub.empty:
            return 0.0
        return float((1 + sub).prod() - 1.0)

    port_30 = _window_total_return(port_returns, 30)
    port_60 = _window_total_return(port_returns, 60)
    bench_30 = _window_total_return(bench_returns, 30)
    bench_60 = _window_total_return(bench_returns, 60)

    metrics = {
        "ret_30d": port_30,
        "ret_60d": port_60,
        "alpha_30d": port_30 - bench_30,
        "alpha_60d": port_60 - bench_60,
    }
    return metrics


def _log_positions(wave_name: str, positions: pd.DataFrame) -> None:
    """Write a daily snapshot of positions to logs/positions."""
    if positions.empty:
        return
    today_str = datetime.now().strftime("%Y%m%d")
    file_path = POSITIONS_LOG_DIR / f"{wave_name.replace(' ', '_')}_positions_{today_str}.csv"
    try:
        positions.to_csv(file_path, index=False)
    except Exception:
        pass  # non-fatal


def _log_performance(wave_name: str, metrics: Dict[str, float]) -> None:
    """Append or create performance log for a wave."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    row = {
        "date": today_str,
        "ret_30d": metrics.get("ret_30d", 0.0),
        "ret_60d": metrics.get("ret_60d", 0.0),
        "alpha_30d": metrics.get("alpha_30d", 0.0),
        "alpha_60d": metrics.get("alpha_60d", 0.0),
    }
    file_path = PERF_LOG_DIR / f"{wave_name.replace(' ', '_')}_performance_daily.csv"
    try:
        if file_path.exists():
            existing = pd.read_csv(file_path)
            existing = existing[existing["date"] != today_str]
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.to_csv(file_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(file_path, index=False)
    except Exception:
        pass  # non-fatal


def _load_last_logged_metrics(wave_name: str) -> Optional[Dict[str, float]]:
    """
    If live price downloads fail and return zeros, we can fall back to
    the last recorded performance row for this Wave.
    """
    file_path = PERF_LOG_DIR / f"{wave_name.replace(' ', '_')}_performance_daily.csv"
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        last = df.sort_values("date").iloc[-1]
        return {
            "ret_30d": float(last.get("ret_30d", 0.0)),
            "ret_60d": float(last.get("ret_60d", 0.0)),
            "alpha_30d": float(last.get("alpha_30d", 0.0)),
            "alpha_60d": float(last.get("alpha_60d", 0.0)),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------
# Core Wave Snapshot
# ---------------------------------------------------------------------


def _build_wave_positions(wave_name: str) -> pd.DataFrame:
    """
    Return a positions DataFrame for a given wave, with columns:
    ticker, weight, last_price, intraday_return
    """
    weights = _load_wave_weights()
    wave_df = weights[weights["wave"] == wave_name].copy()
    if wave_df.empty:
        return pd.DataFrame(columns=["ticker", "weight", "last_price", "intraday_return"])

    prices = []
    intraday_returns = []
    for _, row in wave_df.iterrows():
        ticker = row["ticker"]
        last_price, intraday_ret = _get_fast_intraday_return(ticker)
        prices.append(last_price)
        intraday_returns.append(intraday_ret)

    wave_df["last_price"] = prices
    wave_df["intraday_return"] = intraday_returns

    # Apply SmartSafe 2.0 hook (no-op for now)
    wave_df = apply_smartsafe_sweep(wave_df)

    return wave_df


def get_wave_snapshot(wave_name: str) -> Dict:
    """
    High-level function used by app.py.

    Returns:
        {
            "wave_name": str,
            "benchmark": str,
            "positions": DataFrame,
            "metrics": {
                "intraday_return": float,
                "ret_30d": float,
                "ret_60d": float,
                "alpha_30d": float,
                "alpha_60d": float,
            }
        }
    """
    positions = _build_wave_positions(wave_name)
    benchmark = _get_benchmark_for_wave(wave_name)

    # Portfolio intraday return: weighted sum of individual intraday returns
    if not positions.empty:
        w = positions["weight"]
        w = w / w.sum() if w.sum() != 0 else w
        intraday_ret = float((positions["intraday_return"] * w).sum())
    else:
        intraday_ret = 0.0

    trailing = _compute_portfolio_trailing_returns(
        positions,
        benchmark_ticker=benchmark,
        period="180d",
    )

    # If live computation produced all zeros, try to fall back to last logged values
    if (
        trailing["ret_30d"] == 0.0
        and trailing["ret_60d"] == 0.0
        and trailing["alpha_30d"] == 0.0
        and trailing["alpha_60d"] == 0.0
    ):
        logged = _load_last_logged_metrics(wave_name)
        if logged is not None:
            trailing = logged

    metrics = {
        "intraday_return": intraday_ret,
        "ret_30d": trailing["ret_30d"],
        "ret_60d": trailing["ret_60d"],
        "alpha_30d": trailing["alpha_30d"],
        "alpha_60d": trailing["alpha_60d"],
    }

    # Log (non-fatal if fails)
    _log_positions(wave_name, positions)
    _log_performance(wave_name, metrics)

    return {
        "wave_name": wave_name,
        "benchmark": benchmark,
        "positions": positions,
        "metrics": metrics,
    }