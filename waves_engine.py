"""
waves_engine.py — WAVES Intelligence™ Engine
(VIX Ladder + SmartSafe 2.0 + Blended Benchmarks + 1Y & Since Inception)

This version:
- Auto-discovers Waves from wave_weights.csv
- Uses your custom blended benchmarks (multi-ETF) for specific Waves
- Uses single-ETF benchmarks for all other Waves
- Implements a VIX-based risk ladder (no SmartSafe 3.0 anywhere)
- Implements SmartSafe 2.0 *weighted sweep*:
    * When VIX is high, trims highest-volatility holdings first
    * Frees up X% of portfolio and reallocates to BIL (SmartSafe proxy)
- Computes:
    * Intraday return
    * 30-day & 60-day return and alpha
    * 1-year return and alpha (approx. 252 trading days)
    * Since inception return and alpha (full history window)
- Uses robust price download logic
- Falls back to last logged performance metrics if live pulls fail

Assumptions:
- wave_weights.csv: columns for Wave, Ticker, Weight (any reasonable capitalization/spaces)
- list.csv: optional universe with a Ticker column

IMPORTANT:
- No Waves are equal-weighted in this engine. We always use weights from wave_weights.csv,
  normalized within each Wave, not equal splits.
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

# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------

# 1) Default single-ETF benchmarks (used when no custom blend is defined)
BENCHMARK_MAP: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "S&P Wave": "SPY",
    "AI Wave": "QQQ",  # overridden by blended spec below
    "AI & Innovation Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",  # overridden by blended spec below
    "Future Power & Energy Wave": "XLE",  # overridden by blended spec below
    "Clean Transit-Infrastructure Wave": "IYT",  # overridden by blended spec below
    "Crypto Income Wave": "BITO",  # overridden by blended spec below
    "Crypto Equity Wave": "BITO",  # overridden by blended spec below
    "Income Wave": "SCHD",
    "Small Cap Growth Wave": "IWM",  # overridden by blended spec below
    "Small to Mid Cap Growth Wave": "VO",
    "Cloud & Software Wave": "IGV",  # overridden by blended spec below
    "SmartSafe Money Market Wave": "BIL",
}

# 2) Custom blended benchmark specs (your research)
# NOTE: Raw weights are normalized automatically so they don't have to sum to 1 exactly.
# Keys must match the exact Wave names as they appear in wave_weights.csv.

BLENDED_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # AI wave: 50/50 SMH and AIQ
    "AI Wave": {
        "SMH": 0.50,
        "AIQ": 0.50,
    },
    # Clean transit: SPY 40%, QQQ 40%, IWM 20%
    "Clean Transit-Infrastructure Wave": {
        "SPY": 0.40,
        "QQQ": 0.40,
        "IWM": 0.20,
    },
    # Cloud & Software: 50% QQQ, 40% WCLD, 30% HACK (normalized)
    "Cloud & Software Wave": {
        "QQQ": 0.50,
        "WCLD": 0.40,
        "HACK": 0.30,
    },
    # Crypto equity: 50% WGMI, 30% BLOK, 20% BITQ
    # Wire to both possible names so we're safe
    "Crypto Income Wave": {
        "WGMI": 0.50,
        "BLOK": 0.30,
        "BITQ": 0.20,
    },
    "Crypto Equity Wave": {
        "WGMI": 0.50,
        "BLOK": 0.30,
        "BITQ": 0.20,
    },
    # Future power & energy: 40% QQQ, 30% BUG, 30% WCLD
    "Future Power & Energy Wave": {
        "QQQ": 0.40,
        "BUG": 0.30,
        "WCLD": 0.30,
    },
    # Growth: 40% QQQ, 30% BUG, 30% WCLD
    "Growth Wave": {
        "QQQ": 0.40,
        "BUG": 0.30,
        "WCLD": 0.30,
    },
    # Quantum computing: QQQ 50%, SOXX 25%, ARKK 25%
    "Quantum Computing Wave": {
        "QQQ": 0.50,
        "SOXX": 0.25,
        "ARKK": 0.25,
    },
    # Small cap growth: ARKK 40%, IPAY 30%, XLY 30%
    "Small Cap Growth Wave": {
        "ARKK": 0.40,
        "IPAY": 0.30,
        "XLY": 0.30,
    },
}

# VIX ladder (Option B)
# VIX <20  -> 0% shift
# 20–25    -> 15%
# 25–30    -> 30%
# 30–40    -> 50%
# >40      -> 80%
VIX_LEVELS = [
    (40.0, 0.80),
    (30.0, 0.50),
    (25.0, 0.30),
    (20.0, 0.15),
]

# Daily return clipping band (guardrail for extreme bad ticks)
DAILY_RETURN_CLIP = 0.20  # +/- 20% per day max

# Approx trading days per year
TRADING_DAYS_1Y = 252


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on headers; keep casing but clean up spacing."""
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

    # Aggregate duplicates PER WAVE, then normalize inside each wave
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


def _get_single_benchmark_ticker(wave_name: str) -> str:
    """Fallback: single-ETF benchmark ticker for a wave."""
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


def _get_benchmark_for_wave(
    wave_name: str,
) -> Tuple[str, Dict[str, float]]:
    """
    Return:
        display_label: str  (e.g., "50% SMH + 50% AIQ" or "QQQ")
        spec: dict[ticker -> weight] with weights normalized to sum 1
    """
    if wave_name in BLENDED_BENCHMARKS:
        raw = BLENDED_BENCHMARKS[wave_name]
        total = float(sum(raw.values()))
        if total <= 0:
            # fall back to single ticker
            t = _get_single_benchmark_ticker(wave_name)
            return t, {t: 1.0}
        spec = {t: w / total for t, w in raw.items()}
        parts = [f"{int(round(w * 100))}% {t}" for t, w in spec.items()]
        label = " + ".join(parts)
        return label, spec

    # Not in blended set -> use single ETF
    t = _get_single_benchmark_ticker(wave_name)
    return t, {t: 1.0}


def _download_price_series(ticker: str, period: str = "10y") -> pd.Series:
    """
    Download daily adjusted close for a single ticker.

    More robust:
    - Tries yf.download first.
    - If Adj Close missing, falls back to Close.
    - Uses up to ~10 years to support 1Y & since-inception.
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
# VIX & SmartSafe 2.0
# ---------------------------------------------------------------------


def _get_vix_level() -> Optional[float]:
    """
    Fetch latest VIX level using ^VIX.
    If unavailable, return None (no sweep).
    """
    try:
        s = _download_price_series("^VIX", period="1y")
        if s.empty:
            return None
        return float(s.dropna().iloc[-1])
    except Exception:
        return None


def _compute_sweep_fraction_from_vix(vix_level: Optional[float]) -> float:
    """
    Ladder rule (Option B):

    VIX <20   -> 0%
    20–25     -> 15%
    25–30     -> 30%
    30–40     -> 50%
    >40       -> 80%
    """
    if vix_level is None:
        return 0.0
    if vix_level < 20:
        return 0.0

    for threshold, frac in VIX_LEVELS:
        if vix_level >= threshold:
            return frac

    # Between 20 and lowest threshold in VIX_LEVELS:
    return 0.15


def _compute_vol_by_ticker(tickers: List[str], period: str = "90d") -> pd.DataFrame:
    """
    Compute simple historical volatility per ticker using daily returns over `period`.
    Returns DataFrame with columns: ticker, vol
    """
    records = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if s.empty:
            continue
        returns = s.pct_change().dropna()
        if returns.empty:
            continue
        # Clip extreme moves for stability
        returns = returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)
        vol = float(returns.std())
        records.append({"ticker": t, "vol": vol})

    if not records:
        return pd.DataFrame(columns=["ticker", "vol"])

    return pd.DataFrame(records)


def apply_smartsafe_sweep(
    wave_name: str,
    positions: pd.DataFrame,
    vix_level: Optional[float],
) -> pd.DataFrame:
    """
    SmartSafe 2.0 weighted sweep (Option 2).

    - If wave is the SmartSafe Money Market Wave itself, do nothing.
    - Otherwise:
        * Compute sweep_fraction from VIX ladder.
        * Trim highest-volatility tickers first until that fraction is freed.
        * Add a synthetic BIL position with the freed weight.

    Returns a new positions DataFrame with adjusted weights (and BIL if used).
    """
    if positions.empty:
        return positions

    # Don't sweep SmartSafe Wave itself
    if "smartsafe" in wave_name.lower():
        return positions

    sweep_fraction = _compute_sweep_fraction_from_vix(vix_level)
    if sweep_fraction <= 0.0:
        return positions

    df = positions.copy()

    # Normalize weights (never equal-weight; we respect wave_weights.csv ratios)
    total_weight = df["weight"].sum()
    if total_weight <= 0:
        return positions
    df["weight"] = df["weight"] / total_weight

    # Compute volatility per ticker
    tickers = df["ticker"].unique().tolist()
    vol_df = _compute_vol_by_ticker(tickers, period="90d")

    # If we can't compute vol, fall back to simple proportional sweep
    if vol_df.empty:
        trim_amount = sweep_fraction
        df["weight"] = df["weight"] * (1.0 - sweep_fraction)
        bil_last, bil_intraday = _get_fast_intraday_return("BIL")
        bil_row = {
            "ticker": "BIL",
            "weight": trim_amount,
            "last_price": bil_last,
            "intraday_return": bil_intraday,
        }
        df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)
        return df

    # Merge vol into positions, default vol=0 if missing
    df = df.merge(vol_df, on="ticker", how="left")
    df["vol"] = df["vol"].fillna(0.0)

    # Sort by volatility descending (highest vol first = trimmed first)
    df = df.sort_values("vol", ascending=False)

    target_sweep = sweep_fraction
    remaining_sweep = target_sweep
    bil_weight = 0.0

    new_weights = []
    for _, row in df.iterrows():
        w = float(row["weight"])
        if remaining_sweep <= 0.0:
            new_weights.append(w)
            continue

        cut = min(w, remaining_sweep)
        new_w = w - cut
        bil_weight += cut
        remaining_sweep -= cut
        new_weights.append(new_w)

    df["weight"] = new_weights

    # Drop vol column; internal only
    df = df.drop(columns=["vol"])

    # If we didn't actually sweep anything, return original
    if bil_weight <= 0.0:
        return positions

    # Add BIL as SmartSafe allocation
    bil_last, bil_intraday = _get_fast_intraday_return("BIL")
    bil_row = {
        "ticker": "BIL",
        "weight": bil_weight,
        "last_price": bil_last,
        "intraday_return": bil_intraday,
    }
    df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)

    # Re-normalize to ensure sum of weights ~ 1
    total_weight = df["weight"].sum()
    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight

    return df


# ---------------------------------------------------------------------
# Performance & logging
# ---------------------------------------------------------------------


def _compute_composite_benchmark_returns(
    benchmark_spec: Dict[str, float],
    period: str = "10y",
) -> pd.Series:
    """
    Build a composite benchmark return series from a dict[ticker -> weight].
    Weights are assumed to sum to 1 (normalized earlier).
    """
    if not benchmark_spec:
        return pd.Series(dtype=float)

    price_frames = []
    for t in benchmark_spec.keys():
        s = _download_price_series(t, period=period)
        if not s.empty:
            price_frames.append(s)

    if not price_frames:
        return pd.Series(dtype=float)

    prices_df = pd.concat(price_frames, axis=1).sort_index()
    prices_df = prices_df.ffill().dropna(how="all")

    # Align weights to columns
    weights = pd.Series(benchmark_spec)
    weights = weights.reindex(prices_df.columns).fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()

    bench_values = (prices_df * weights).sum(axis=1)
    bench_returns = bench_values.pct_change().dropna()
    bench_returns = bench_returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)
    return bench_returns


def _window_total_return(r: pd.Series, days: Optional[int] = None) -> float:
    """
    Compute total return over the last `days` observations.
    If days is None, use full history ("since inception" for the series).
    """
    if r.empty:
        return 0.0
    if days is not None:
        r = r.tail(days)
        if r.empty:
            return 0.0
    return float((1 + r).prod() - 1.0)


def _compute_portfolio_trailing_returns(
    positions: pd.DataFrame,
    benchmark_spec: Dict[str, float],
    period: str = "10y",
) -> Dict[str, float]:
    """
    Compute total returns and alpha vs composite benchmark over multiple windows:

    - 30d
    - 60d
    - 1y (252 trading days approx)
    - Since inception (full available history)

    Returns keys:
        ret_30d, ret_60d, ret_1y, ret_si,
        alpha_30d, alpha_60d, alpha_1y, alpha_si
    """
    if positions.empty:
        return {
            "ret_30d": 0.0,
            "ret_60d": 0.0,
            "ret_1y": 0.0,
            "ret_si": 0.0,
            "alpha_30d": 0.0,
            "alpha_60d": 0.0,
            "alpha_1y": 0.0,
            "alpha_si": 0.0,
        }

    tickers = positions["ticker"].unique().tolist()
    weights = positions.set_index("ticker")["weight"]
    weights = weights / weights.sum() if weights.sum() != 0 else weights

    # Download history for each portfolio ticker
    price_frames = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if not s.empty:
            price_frames.append(s)

    if not price_frames:
        return {
            "ret_30d": 0.0,
            "ret_60d": 0.0,
            "ret_1y": 0.0,
            "ret_si": 0.0,
            "alpha_30d": 0.0,
            "alpha_60d": 0.0,
            "alpha_1y": 0.0,
            "alpha_si": 0.0,
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
    port_returns = port_returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)

    # Composite benchmark returns
    bench_returns = _compute_composite_benchmark_returns(benchmark_spec, period=period)
    if bench_returns.empty:
        bench_returns = pd.Series(0.0, index=port_returns.index)
    else:
        port_returns, bench_returns = port_returns.align(bench_returns, join="inner")

    # Returns
    port_30 = _window_total_return(port_returns, 30)
    port_60 = _window_total_return(port_returns, 60)
    port_1y = _window_total_return(port_returns, TRADING_DAYS_1Y)
    port_si = _window_total_return(port_returns, None)

    bench_30 = _window_total_return(bench_returns, 30)
    bench_60 = _window_total_return(bench_returns, 60)
    bench_1y = _window_total_return(bench_returns, TRADING_DAYS_1Y)
    bench_si = _window_total_return(bench_returns, None)

    metrics = {
        "ret_30d": port_30,
        "ret_60d": port_60,
        "ret_1y": port_1y,
        "ret_si": port_si,
        "alpha_30d": port_30 - bench_30,
        "alpha_60d": port_60 - bench_60,
        "alpha_1y": port_1y - bench_1y,
        "alpha_si": port_si - bench_si,
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
        "ret_1y": metrics.get("ret_1y", 0.0),
        "ret_si": metrics.get("ret_si", 0.0),
        "alpha_30d": metrics.get("alpha_30d", 0.0),
        "alpha_60d": metrics.get("alpha_60d", 0.0),
        "alpha_1y": metrics.get("alpha_1y", 0.0),
        "alpha_si": metrics.get("alpha_si", 0.0),
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
    If live price downloads fail and return zeros, fall back to
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
            "ret_1y": float(last.get("ret_1y", 0.0)),
            "ret_si": float(last.get("ret_si", 0.0)),
            "alpha_30d": float(last.get("alpha_30d", 0.0)),
            "alpha_60d": float(last.get("alpha_60d", 0.0)),
            "alpha_1y": float(last.get("alpha_1y", 0.0)),
            "alpha_si": float(last.get("alpha_si", 0.0)),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------
# Core Wave Snapshot
# ---------------------------------------------------------------------


def _build_wave_positions(wave_name: str, vix_level: Optional[float]) -> pd.DataFrame:
    """
    Return a positions DataFrame for a given wave, with columns:
    ticker, weight, last_price, intraday_return
    and apply SmartSafe 2.0 weighted sweep based on VIX.
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

    # Apply SmartSafe 2.0 weighted sweep based on VIX
    wave_df = apply_smartsafe_sweep(wave_name, wave_df, vix_level)

    return wave_df


def get_wave_snapshot(wave_name: str) -> Dict:
    """
    High-level function used by app.py.

    Returns:
        {
            "wave_name": str,
            "benchmark": str (display label, may be a blend like "50% SMH + 50% AIQ"),
            "positions": DataFrame,
            "metrics": {
                "intraday_return": float,
                "ret_30d": float,
                "ret_60d": float,
                "ret_1y": float,
                "ret_si": float,
                "alpha_30d": float,
                "alpha_60d": float,
                "alpha_1y": float,
                "alpha_si": float,
                "vix_level": float | None,
                "smartsafe_sweep_fraction": float,
            }
        }
    """
    benchmark_label, benchmark_spec = _get_benchmark_for_wave(wave_name)

    # Get VIX level once per snapshot
    vix_level = _get_vix_level()
    sweep_fraction = _compute_sweep_fraction_from_vix(vix_level)

    positions = _build_wave_positions(wave_name, vix_level=vix_level)

    # Portfolio intraday return: weighted sum of individual intraday returns
    if not positions.empty:
        w = positions["weight"]
        w = w / w.sum() if w.sum() != 0 else w
        intraday_ret = float((positions["intraday_return"] * w).sum())
    else:
        intraday_ret = 0.0

    trailing = _compute_portfolio_trailing_returns(
        positions,
        benchmark_spec=benchmark_spec,
        period="10y",
    )

    # If live computation produced all zeros, try to fall back to last logged values
    if (
        trailing["ret_30d"] == 0.0
        and trailing["ret_60d"] == 0.0
        and trailing["ret_1y"] == 0.0
        and trailing["ret_si"] == 0.0
        and trailing["alpha_30d"] == 0.0
        and trailing["alpha_60d"] == 0.0
        and trailing["alpha_1y"] == 0.0
        and trailing["alpha_si"] == 0.0
    ):
        logged = _load_last_logged_metrics(wave_name)
        if logged is not None:
            trailing = logged

    metrics = {
        "intraday_return": intraday_ret,
        "ret_30d": trailing["ret_30d"],
        "ret_60d": trailing["ret_60d"],
        "ret_1y": trailing["ret_1y"],
        "ret_si": trailing["ret_si"],
        "alpha_30d": trailing["alpha_30d"],
        "alpha_60d": trailing["alpha_60d"],
        "alpha_1y": trailing["alpha_1y"],
        "alpha_si": trailing["alpha_si"],
        "vix_level": vix_level,
        "smartsafe_sweep_fraction": sweep_fraction,
    }

    # Log (non-fatal if fails)
    _log_positions(wave_name, positions)
    _log_performance(wave_name, metrics)

    return {
        "wave_name": wave_name,
        "benchmark": benchmark_label,
        "positions": positions,
        "metrics": metrics,
    }