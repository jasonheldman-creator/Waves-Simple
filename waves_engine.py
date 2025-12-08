# waves_engine.py
"""
WAVES Intelligence™ Engine
- Loads wave definitions
- Builds dynamic S&P 500 Wave weights (Autonomous hybrid model)
- Computes live and historical performance
- Provides helpers for the Streamlit console
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
WAVE_WEIGHTS_PATH = BASE_DIR / "wave_weights.csv"
SP500_UNIVERSE_PATH = BASE_DIR / "sp500_universe.csv"

LOG_DIR = BASE_DIR / "logs"
POS_LOG_DIR = LOG_DIR / "positions"
PERF_LOG_DIR = LOG_DIR / "performance"

POS_LOG_DIR.mkdir(parents=True, exist_ok=True)
PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Benchmarks per Wave (extend this dict as needed)
BENCHMARK_MAP: Dict[str, str] = {
    "S&P Wave": "^GSPC",
    # "Growth Wave": "QQQ",
    # "Future Power Wave": "XLE",
    # etc...
}

# -------------------------------------------------------------------
# CSV helpers
# -------------------------------------------------------------------

def load_wave_weights(path: Path = WAVE_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Loads wave_weights.csv with columns: wave,ticker,weight
    Normalizes weights within each wave.
    """
    if not path.exists():
        raise FileNotFoundError(f"wave_weights.csv not found at {path}")

    df = pd.read_csv(path)
    required_cols = {"wave", "ticker", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing columns: {missing}")

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Normalize weights within each wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )

    return df


def load_sp500_universe(path: Path = SP500_UNIVERSE_PATH) -> pd.DataFrame:
    """
    Expects a CSV with at least:
        ticker, market_cap (market_cap can be empty; falls back to equal-weight)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"sp500_universe.csv not found at {path}. "
            f"Create it with columns: ticker,market_cap"
        )

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("sp500_universe.csv must contain a 'ticker' column.")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "market_cap" not in df.columns:
        df["market_cap"] = np.nan

    return df


# -------------------------------------------------------------------
# Market data helpers
# -------------------------------------------------------------------

def fetch_price_history(
    tickers: List[str],
    lookback_days: int = 260
) -> pd.DataFrame:
    """
    Fetches daily adjusted close prices for the given tickers.
    Returns DataFrame: index = date, columns = tickers
    """
    tickers = list({t.strip().upper() for t in tickers if t})
    if not tickers:
        raise ValueError("No tickers provided to fetch_price_history().")

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # Handle different shapes for single vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: (ticker, field)
        close = data.xs("Close", axis=1, level=1)
    else:
        # Single ticker: columns like ['Open','High','Low','Close',...]
        if "Close" not in data.columns:
            raise ValueError("No 'Close' column returned from yfinance.")
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all")
    return close


def get_vix_level(default: float = 18.0) -> float:
    """
    Fetches recent VIX level. Falls back to default if fetch fails.
    """
    try:
        vix = yf.download(
            tickers=["^VIX"],
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"].dropna()
        if not vix.empty:
            return float(vix.iloc[-1])
    except Exception:
        pass
    return float(default)


# -------------------------------------------------------------------
# Factor model (Hybrid - Option 3)
# -------------------------------------------------------------------

def compute_factor_scores(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid factor model:
        - Momentum (6m / 12m)
        - Volatility (60d)
        - Max drawdown (full history)

    Returns DataFrame indexed by ticker with:
        factor_weight_raw (positive, unnormalized)
    """
    rets = price_df.pct_change().dropna(how="all")

    def _safe_return(series: pd.Series, window: int) -> float:
        if len(series) < window + 1:
            return np.nan
        return float(series.iloc[-1] / series.iloc[-(window + 1)] - 1)

    # Momentum
    mom6 = {c: _safe_return(price_df[c].dropna(), 126) for c in price_df.columns}
    mom12 = {c: _safe_return(price_df[c].dropna(), 252) for c in price_df.columns}

    # Volatility (60d)
    vol60 = {
        c: float(rets[c].dropna().tail(60).std())
        if not rets[c].dropna().empty else np.nan
        for c in price_df.columns
    }

    # Max drawdown
    mdd = {}
    for c in price_df.columns:
        s = price_df[c].dropna()
        if s.empty:
            mdd[c] = np.nan
            continue
        cum_max = s.cummax()
        dd = (s / cum_max) - 1.0
        mdd[c] = float(dd.min())

    factor_df = pd.DataFrame({
        "momentum_6m": pd.Series(mom6),
        "momentum_12m": pd.Series(mom12),
        "vol_60d": pd.Series(vol60),
        "max_drawdown": pd.Series(mdd),
    })

    def zscore(x: np.ndarray) -> np.ndarray:
        m = np.nanmean(x)
        s = np.nanstd(x)
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return (x - m) / s

    factor_df["z_mom6"] = zscore(factor_df["momentum_6m"].values)
    factor_df["z_mom12"] = zscore(factor_df["momentum_12m"].values)
    factor_df["z_vol60"] = zscore(factor_df["vol_60d"].values)
    factor_df["z_mdd"] = zscore(factor_df["max_drawdown"].values)

    # Composite factors
    # Higher momentum is good; lower vol/mdd is good
    factor_df["momentum_score"] = (
        0.5 * factor_df["z_mom6"] + 0.5 * factor_df["z_mom12"]
    )
    factor_df["quality_score"] = (
        -0.6 * factor_df["z_vol60"] + -0.4 * factor_df["z_mdd"]
    )

    factor_df["factor_score"] = (
        0.6 * factor_df["momentum_score"] + 0.4 * factor_df["quality_score"]
    )

    # Convert to positive raw weights
    factor_df["factor_weight_raw"] = np.exp(
        factor_df["factor_score"].fillna(0.0)
    )

    return factor_df


def normalize_weights(raw: pd.Series) -> pd.Series:
    total = raw.sum()
    if total <= 0 or np.isnan(total):
        return pd.Series(
            np.repeat(1.0 / len(raw), len(raw)),
            index=raw.index,
        )
    return raw / total


# -------------------------------------------------------------------
# Dynamic S&P 500 Wave builder
# -------------------------------------------------------------------

def build_dynamic_sp500_weights(
    universe_path: Path = SP500_UNIVERSE_PATH,
    vix_level: Optional[float] = None,
) -> pd.DataFrame:
    """
    Builds autonomous S&P 500 Wave weights:

        40% market-cap base
        30% equal-weight overlay
        20% factor sleeve
        10% alpha sleeve (concentrated factor tilt, VIX-gated)

    Returns DataFrame with columns: wave, ticker, weight
    """
    universe = load_sp500_universe(universe_path)
    tickers = universe["ticker"].unique().tolist()

    if len(tickers) == 0:
        raise ValueError("sp500_universe.csv has no tickers defined.")

    # Market-cap layer
    if universe["market_cap"].notna().sum() > 0:
        mc = universe.set_index("ticker")["market_cap"].fillna(0.0)
        mc = mc.reindex(tickers).fillna(0.0)
        mc_weight = normalize_weights(mc)
    else:
        mc_weight = pd.Series(
            np.repeat(1.0 / len(tickers), len(tickers)),
            index=tickers,
        )

    # Equal-weight
    eq_weight = pd.Series(
        np.repeat(1.0 / len(tickers), len(tickers)),
        index=tickers,
    )

    # Factor sleeve
    prices = fetch_price_history(tickers, lookback_days=260)
    factor_df = compute_factor_scores(prices)
    factor_weight = normalize_weights(factor_df["factor_weight_raw"])

    # Alpha sleeve (more concentrated factor tilt)
    alpha_raw = factor_df["factor_weight_raw"] ** 1.5
    alpha_weight = normalize_weights(alpha_raw)

    # VIX gating for alpha intensity
    if vix_level is None:
        vix_level = get_vix_level()

    if vix_level <= 18:
        alpha_intensity = 1.0
    elif vix_level <= 25:
        alpha_intensity = 0.6
    elif vix_level <= 35:
        alpha_intensity = 0.3
    else:
        alpha_intensity = 0.1

    # Combine layers
    base = (
        0.40 * mc_weight.reindex(tickers) +
        0.30 * eq_weight.reindex(tickers) +
        0.20 * factor_weight.reindex(tickers) +
        0.10 * alpha_intensity * alpha_weight.reindex(tickers)
    )

    combined = normalize_weights(base)

    # Guardrails: floor/ceil vs market-cap layer
    floor = 0.20 * mc_weight
    ceil = 1.50 * mc_weight

    adjusted = combined.clip(lower=floor, upper=ceil)
    final = normalize_weights(adjusted)

    out = pd.DataFrame({
        "wave": "S&P Wave",
        "ticker": final.index.astype(str),
        "weight": final.values,
    })

    return out


# -------------------------------------------------------------------
# Public API: weights & wave list
# -------------------------------------------------------------------

def get_dynamic_wave_weights() -> pd.DataFrame:
    """
    Returns full wave weights with dynamic S&P 500 Wave overriding any static S&P Wave.
    """
    static_df = load_wave_weights()
    # Remove any existing S&P Wave entries
    static_df = static_df[static_df["wave"] != "S&P Wave"].copy()

    try:
        sp500_df = build_dynamic_sp500_weights()
        combined = pd.concat([static_df, sp500_df], ignore_index=True)
    except Exception as e:
        # If something goes wrong, fall back to static only
        print(f"[WARN] Dynamic S&P 500 build failed: {e}")
        combined = static_df

    # Ensure weights normalized per wave
    combined["weight"] = combined.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )
    return combined


def get_wave_names(weights_df: Optional[pd.DataFrame] = None) -> List[str]:
    if weights_df is None:
        weights_df = get_dynamic_wave_weights()
    return sorted(weights_df["wave"].unique())


# -------------------------------------------------------------------
# Performance & summary helpers
# -------------------------------------------------------------------

def get_benchmark_for_wave(wave_name: str) -> Optional[str]:
    return BENCHMARK_MAP.get(wave_name)


def compute_wave_returns_over_window(
    wave_name: str,
    weights_df: pd.DataFrame,
    days: int,
    benchmark_ticker: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes cumulative return and alpha over the last `days` calendar days.
    Returns (wave_return, alpha_vs_benchmark).
    """
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return None, None

    weights = (
        w.groupby("ticker")["weight"]
        .sum()
        .reindex(w["ticker"].unique())
        .fillna(0.0)
    )

    tickers = list(weights.index)
    lookback = days + 5  # pad a few days for weekends/holidays

    try:
        prices = fetch_price_history(tickers, lookback_days=lookback)
    except Exception as e:
        print(f"[WARN] Failed to fetch prices for {wave_name}: {e}")
        return None, None

    # Daily portfolio returns
    rets = prices.pct_change().dropna(how="all")
    # Align weights to columns
    weights_aligned = weights.reindex(prices.columns).fillna(0.0)
    port_rets = (rets * weights_aligned).sum(axis=1)

    if port_rets.empty:
        return None, None

    # Keep last `days` observations
    port_rets = port_rets.tail(days)
    wave_cum = float((1.0 + port_rets).prod() - 1.0)

    alpha = None
    if benchmark_ticker:
        try:
            bench_prices = fetch_price_history(
                [benchmark_ticker],
                lookback_days=lookback,
            )
            bench_rets = bench_prices.pct_change().dropna().iloc[:, 0]
            bench_rets = bench_rets.tail(len(port_rets))
            bench_cum = float((1.0 + bench_rets).prod() - 1.0)
            alpha = wave_cum - bench_cum
        except Exception as e:
            print(f"[WARN] Failed to fetch benchmark {benchmark_ticker}: {e}")
            alpha = None

    return wave_cum, alpha


def get_top_holdings(
    wave_name: str,
    weights_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Returns top N holdings by weight for a given wave.
    """
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame(columns=["ticker", "weight"])
    w = (
        w.groupby("ticker")["weight"]
        .sum()
        .reset_index()
        .sort_values("weight", ascending=False)
        .head(n)
    )
    return w.reset_index(drop=True)


def compute_wave_summary(
    wave_name: str,
    weights_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Returns summary metrics for a wave:
        - 1D, 30D, 60D returns
        - 1D, 30D, 60D alpha
        - top holdings (DataFrame)
    """
    if weights_df is None:
        weights_df = get_dynamic_wave_weights()

    bench = get_benchmark_for_wave(wave_name)
    summary: Dict[str, object] = {}

    for window in [1, 30, 60]:
        ret, alpha = compute_wave_returns_over_window(
            wave_name,
            weights_df,
            days=window,
            benchmark_ticker=bench,
        )
        summary[f"return_{window}d"] = ret
        summary[f"alpha_{window}d"] = alpha

    summary["top_holdings"] = get_top_holdings(wave_name, weights_df, n=10)
    summary["benchmark"] = bench
    return summary