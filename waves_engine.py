"""
waves_engine.py — Vector 1 Fallback (Hardened)

- Reads wave_weights.csv (user-defined Waves)
- Builds a dynamic S&P 500 Wave from sp500_universe.csv
  with an overlay from any "S&P Wave" rows in wave_weights.csv
- Fetches price history via yfinance (with full try/except so Streamlit never crashes)
- Computes per-Wave performance & alpha vs benchmarks
- Produces top holdings
- Computes simple factor scores (momentum, quality/low-vol proxy)
- Exposes a VIX helper for the console
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

WAVE_WEIGHTS_PATH = ROOT / "wave_weights.csv"
SP500_UNIVERSE_PATH = ROOT / "sp500_universe.csv"

# Portion of S&P Wave weight that is driven by the custom overlay
SP500_OVERLAY_SHARE = 0.30  # 30% overlay from wave_weights; 70% broad S&P core

# Lookback for summary calculations
MAX_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Wave weights & dynamic S&P Wave
# ---------------------------------------------------------------------

def load_wave_weights(path: Path = WAVE_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Load base wave weights from CSV: columns wave,ticker,weight
    Normalizes weights within each wave but does NOT yet apply
    the dynamic S&P logic.
    """
    df = _read_csv(path)
    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv must contain columns {expected_cols}, missing {missing}")

    df = df.copy()
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    grouped = []
    for wave, sub in df.groupby("wave"):
        total = sub["weight"].sum()
        if total <= 0:
            continue
        sub = sub.copy()
        sub["weight"] = sub["weight"] / total
        grouped.append(sub)
    if not grouped:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    return pd.concat(grouped, ignore_index=True)


def _build_sp500_core(path: Path = SP500_UNIVERSE_PATH) -> pd.DataFrame:
    """
    Build a broad S&P 500 core from sp500_universe.csv.

    Expected columns: ticker, market_cap (optional but preferred).
    If market_cap is missing, all tickers are equal-weighted.
    Returns DataFrame with columns: ticker, core_weight.
    """
    df = _read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("sp500_universe.csv must contain a 'ticker' column.")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0.0)
        if df["market_cap"].sum() > 0:
            df["core_weight"] = df["market_cap"] / df["market_cap"].sum()
        else:
            df["core_weight"] = 1.0 / len(df)
    else:
        df["core_weight"] = 1.0 / len(df)

    return df[["ticker", "core_weight"]]


def _build_dynamic_sp500_wave(
    base_weights: pd.DataFrame,
    sp500_path: Path = SP500_UNIVERSE_PATH,
    overlay_share: float = SP500_OVERLAY_SHARE,
    wave_name: str = "S&P Wave",
) -> pd.DataFrame:
    """
    Build the dynamic S&P Wave:

    - Start from a broad S&P core based on sp500_universe.csv
      (market-cap weighted if available, else equal).
    - Take any rows in base_weights where wave == wave_name and
      treat them as an overlay (user-specified tilt).
    - Combine core and overlay:
        weight = (1 - overlay_share) * core_weight
                 + overlay_share * overlay_weight

    Returns DataFrame columns: wave, ticker, weight
    """
    overlay = base_weights[base_weights["wave"] == wave_name].copy()
    overlay = overlay[["ticker", "weight"]]

    if not overlay.empty:
        tot = overlay["weight"].sum()
        if tot > 0:
            overlay["weight"] = overlay["weight"] / tot
        else:
            overlay = overlay.iloc[0:0]

    try:
        core = _build_sp500_core(sp500_path)
    except Exception as e:
        # If core fails but we have an overlay, just use overlay.
        if overlay.empty:
            raise RuntimeError(f"Failed to build S&P core and no overlay present: {e}")
        core = overlay.rename(columns={"weight": "core_weight"})
        core["core_weight"] = 1.0 / len(core)

    merged = core.merge(overlay, how="left", on="ticker")
    merged["weight"].fillna(0.0, inplace=True)
    merged["core_weight"].fillna(0.0, inplace=True)

    overlay_share = float(max(0.0, min(overlay_share, 1.0)))
    base_share = 1.0 - overlay_share

    merged["final_weight"] = base_share * merged["core_weight"] + overlay_share * merged["weight"]

    total = merged["final_weight"].sum()
    if total <= 0:
        merged["final_weight"] = 1.0 / len(merged)
    else:
        merged["final_weight"] = merged["final_weight"] / total

    out = merged[["ticker", "final_weight"]].copy()
    out.rename(columns={"final_weight": "weight"}, inplace=True)
    out["wave"] = wave_name

    return out[["wave", "ticker", "weight"]]


def get_dynamic_wave_weights() -> pd.DataFrame:
    """
    Main entry: returns a DataFrame of all Waves with dynamic S&P applied.
    """
    base = load_wave_weights(WAVE_WEIGHTS_PATH)
    if base.empty:
        return base

    wave_name = "S&P Wave"

    non_sp = base[base["wave"] != wave_name].copy()

    try:
        dynamic_sp = _build_dynamic_sp500_wave(base, SP500_UNIVERSE_PATH, SP500_OVERLAY_SHARE, wave_name)
    except Exception as e:
        print(f"[WARN] Dynamic S&P Wave failed, falling back to base S&P weights: {e}")
        dynamic_sp = base[base["wave"] == wave_name].copy()
        if dynamic_sp.empty:
            return non_sp

    all_waves = pd.concat([non_sp, dynamic_sp], ignore_index=True)

    grouped = []
    for wave, sub in all_waves.groupby("wave"):
        total = sub["weight"].sum()
        if total <= 0:
            continue
        sub = sub.copy()
        sub["weight"] = sub["weight"] / total
        grouped.append(sub)
    if not grouped:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    return pd.concat(grouped, ignore_index=True)


def get_wave_names(weights_df: pd.DataFrame) -> List[str]:
    if weights_df is None or weights_df.empty:
        return []
    return sorted(weights_df["wave"].dropna().unique().tolist())


# ---------------------------------------------------------------------
# Market data helpers (hardened)
# ---------------------------------------------------------------------

def fetch_price_history(tickers: List[str], lookback_days: int = MAX_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for the given tickers.
    Returns wide DataFrame with Date index and tickers as columns.

    HARDENED: any yfinance error returns an empty DataFrame instead
    of crashing the Streamlit app.
    """
    if not tickers:
        return pd.DataFrame()

    tickers_clean = sorted(
        set(str(t).upper().strip() for t in tickers if str(t).strip())
    )
    if not tickers_clean:
        return pd.DataFrame()

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 5)

    try:
        data = yf.download(
            tickers_clean,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"[WARN] yfinance download failed for {tickers_clean}: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            top0 = data.columns.levels[0][0]
            prices = data[top0].copy()
    else:
        prices = data.copy()

    prices = prices.dropna(how="all")
    prices.index.name = "date"
    return prices


def get_vix_level() -> Optional[float]:
    """
    Fetch current-ish VIX level using yfinance ^VIX.
    Also hardened with try/except.
    """
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        last = df["Close"].dropna().iloc[-1]
        return float(last)
    except Exception as e:
        print(f"[WARN] Failed to fetch VIX: {e}")
        return None


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------

def get_benchmark_for_wave(wave_name: str) -> Optional[str]:
    """
    Map a Wave name to a benchmark ticker. Heuristic; adjust as needed.
    """
    name = wave_name.lower()

    if "s&p" in name or "sp " in name or "core us" in name:
        return "SPY"
    if "growth" in name or "innovation" in name or "quantum" in name:
        return "QQQ"
    if "crypto" in name or "bitcoin" in name:
        return "BTC-USD"
    if "income" in name or "yield" in name:
        return "VYM"
    if "energy" in name or "power" in name:
        return "XLE"
    if "clean transit" in name or "infrastructure" in name:
        return "ICLN"
    if "small cap" in name or "smid" in name:
        return "IWM"

    return "SPY"


# ---------------------------------------------------------------------
# Factors (momentum / quality proxies)
# ---------------------------------------------------------------------

def compute_factor_scores(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple factor proxies from price history:

    - momentum_score: full-window total return
    - quality_score: negative volatility
    - factor_score: average of the two
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.sort_index().copy()
    rets = df.pct_change().dropna(how="all")

    momentum = (1.0 + rets).prod() - 1.0
    vol = rets.std()
    quality = -vol

    out = pd.DataFrame(
        {
            "momentum_score": momentum,
            "quality_score": quality,
        }
    )
    out["factor_score"] = out[["momentum_score", "quality_score"]].mean(axis=1)
    return out


# ---------------------------------------------------------------------
# Wave summary (returns, alpha, top holdings)
# ---------------------------------------------------------------------

def _compute_portfolio_returns(
    weights: pd.Series,
    lookback_days: int = MAX_LOOKBACK_DAYS,
) -> pd.Series:
    tickers = list(weights.index)
    prices = fetch_price_history(tickers, lookback_days=lookback_days)
    if prices.empty:
        return pd.Series(dtype=float)

    w = weights.reindex(prices.columns).fillna(0.0)
    rets = prices.pct_change().dropna(how="all")
    port_rets = (rets * w).sum(axis=1)
    return port_rets


def _compute_horizon_return(rets: pd.Series, days: int) -> Optional[float]:
    if rets is None or rets.empty:
        return None
    tail = rets.tail(days)
    if tail.empty:
        return None
    return float((1.0 + tail).prod() - 1.0)


def compute_wave_summary(
    wave_name: str,
    weights_df: pd.DataFrame | None = None,
) -> Dict:
    """
    Compute summary metrics for a single Wave:

    - 1D / 30D / 60D returns
    - Alpha vs benchmark over same horizons
    - Top 10 holdings
    - Benchmark symbol
    """
    if weights_df is None:
        weights_df = get_dynamic_wave_weights()

    df = weights_df[weights_df["wave"] == wave_name].copy()
    if df.empty:
        return {}

    w = (
        df.groupby("ticker")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    port_rets = _compute_portfolio_returns(w, lookback_days=MAX_LOOKBACK_DAYS)

    bench_ticker = get_benchmark_for_wave(wave_name)
    bench_rets = None
    if bench_ticker:
        try:
            bench_prices = fetch_price_history([bench_ticker], lookback_days=MAX_LOOKBACK_DAYS)
            if not bench_prices.empty:
                bench_rets = bench_prices.iloc[:, 0].pct_change().dropna()
        except Exception as e:
            print(f"[WARN] Failed to fetch benchmark {bench_ticker} for {wave_name}: {e}")
            bench_rets = None

    if bench_rets is not None and not bench_rets.empty and not port_rets.empty:
        idx = port_rets.index.intersection(bench_rets.index)
        port_rets = port_rets.reindex(idx)
        bench_rets = bench_rets.reindex(idx)
    else:
        bench_rets = None

    ret_1d = _compute_horizon_return(port_rets, 1)
    ret_30d = _compute_horizon_return(port_rets, 30)
    ret_60d = _compute_horizon_return(port_rets, 60)

    if bench_rets is not None and not bench_rets.empty:
        b1 = _compute_horizon_return(bench_rets, 1)
        b30 = _compute_horizon_return(bench_rets, 30)
        b60 = _compute_horizon_return(bench_rets, 60)
        alpha_1d = None if (ret_1d is None or b1 is None) else ret_1d - b1
        alpha_30d = None if (ret_30d is None or b30 is None) else ret_30d - b30
        alpha_60d = None if (ret_60d is None or b60 is None) else ret_60d - b60
    else:
        alpha_1d = alpha_30d = alpha_60d = None

    top10 = w.head(10).reset_index()
    top10.columns = ["ticker", "weight"]

    summary = {
        "return_1d": ret_1d,
        "alpha_1d": alpha_1d,
        "return_30d": ret_30d,
        "alpha_30d": alpha_30d,
        "return_60d": ret_60d,
        "alpha_60d": alpha_60d,
        "top_holdings": top10,
        "benchmark": bench_ticker,
    }

    return summary