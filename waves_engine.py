# waves_engine.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

WAVE_WEIGHTS_CSV = "wave_weights.csv"

# Benchmark blends for each Wave (locked to your latest choices)
# Weights must sum to 1.0 per Wave.
BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Broad market
    "S&P Wave": {
        "SPY": 1.0,
    },

    # Large-cap growth / innovation
    # (you mentioned QQQ + IWF blend; I’m using 0.55 / 0.45 as a realistic split)
    "Growth Wave": {
        "QQQ": 0.55,
        "IWF": 0.45,
    },

    # Small–Mid Cap / cloud & enterprise software growth
    # From your screenshot: IGV 60%, WCLD/CLOU 20%, SPY 20%
    "Small-Mid Cap Growth Wave": {
        "IGV": 0.60,
        "WCLD": 0.20,  # if CLOU is preferred, just swap the ticker
        "SPY": 0.20,
    },

    # Clean Transit–Infrastructure (industrials + consumer disc + SPY context)
    # From your screenshot: FIDU 45%, FDIS 45%, SPY 10%
    "Clean Transit-Infrastructure Wave": {
        "FIDU": 0.45,
        "FDIS": 0.45,
        "SPY": 0.10,
    },

    # Cloud & Enterprise Software Growth Wave (your “small cap growth” rename)
    "Cloud & Enterprise Software Growth Wave": {
        "IGV": 0.60,
        "WCLD": 0.20,
        "SPY": 0.20,
    },

    # Crypto Equity Wave (mid/large cap) – updated to 70% spot BTC proxy + 30% DAPP
    "Crypto Equity Wave (mid/large cap)": {
        "FBTC": 0.70,  # or IBIT – feel free to swap the ticker symbol
        "DAPP": 0.30,
    },

    # Income Wave – SCHD as primary benchmark (high-quality U.S. dividend)
    "Income Wave": {
        "SCHD": 1.0,
    },

    # Quantum Computing Wave – use a focused tech / innovation proxy
    # (No perfect ETF; using NASDAQ and broad tech blend)
    "Quantum Computing Wave": {
        "QQQ": 0.70,
        "VGT": 0.30,
    },

    # AI Wave – heavy software/AI tilt with tech + market context
    # (You can tweak these if you like later.)
    "AI Wave": {
        "IGV": 0.50,
        "VGT": 0.25,
        "QQQ": 0.15,
        "SPY": 0.10,
    },

    # SmartSafe Wave – cash/short-duration U.S. government exposure
    # You have SGOV 70, BIL 20, SHY 10 in wave_weights – mirror that.
    "SmartSafe Wave": {
        "SGOV": 0.70,
        "BIL": 0.20,
        "SHY": 0.10,
    },
}

# Lookback horizon for return series
LOOKBACK_DAYS = 365 * 2  # download 2Y to safely compute 1Y + 60D


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _load_wave_weights(csv_path: str = WAVE_WEIGHTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns: wave, ticker, weight
    for col in ("wave", "ticker", "weight"):
        if col not in df.columns:
            raise RuntimeError(f"wave_weights.csv missing required column: {col}")

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = df["weight"].astype(float)

    return df


def _download_prices(all_tickers: List[str], start_date: datetime) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers in one shot.
    Robust to partial failures: we drop columns with no data.
    """
    if not all_tickers:
        return pd.DataFrame()

    all_tickers = sorted(set(t for t in all_tickers if isinstance(t, str) and t.strip()))
    if not all_tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=all_tickers,
            start=start_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        # If yfinance completely fails, bubble up a readable error.
        raise RuntimeError(f"Price download failed for all tickers: {e}") from e

    if data is None or len(data) == 0:
        raise RuntimeError("No price data returned by yfinance for the requested tickers.")

    # Handle multi-index columns from yf
    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close" in data.columns.get_level_values(0)):
            prices = data["Adj Close"].copy()
        elif ("Close" in data.columns.get_level_values(0)):
            prices = data["Close"].copy()
        else:
            # Fallback: take the last level as tickers
            prices = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        prices = data.copy()

    # Drop columns that are entirely NaN (rate limit / bad ticker)
    prices = prices.dropna(axis=1, how="all")

    # Clean up ticker names
    prices.columns = [c.strip().upper() for c in prices.columns]
    prices = prices.sort_index()

    return prices


def _series_total_return(ret_series: pd.Series, window_days: int) -> float:
    """
    Compute total return over the last `window_days` trading days.
    If fewer days are available, use all available data.
    """
    if ret_series is None or ret_series.empty:
        return np.nan

    n = min(window_days, len(ret_series))
    window = ret_series.iloc[-n:]
    cumulative = (1.0 + window).prod() - 1.0
    return float(cumulative)


# ------------------------------------------------------------
# Core computation
# ------------------------------------------------------------

def compute_all_wave_metrics() -> pd.DataFrame:
    """
    Main engine function.
    Returns a DataFrame with one row per Wave and:
      - ret_60d, alpha_60d
      - ret_1y,  alpha_1y
      - benchmark_label
      - notes (if any issues like dropped tickers)
    """
    weights_df = _load_wave_weights()

    # Collect all tickers needed: holdings + benchmark ETFs
    unique_holdings = set(weights_df["ticker"].unique())
    benchmark_tickers = {sym for wave_map in BENCHMARKS.values() for sym in wave_map.keys()}
    all_tickers = list(unique_holdings.union(benchmark_tickers))

    start_date = datetime.today() - timedelta(days=LOOKBACK_DAYS)
    prices = _download_prices(all_tickers, start_date)

    if prices.empty:
        raise RuntimeError("No price data available for any ticker after cleaning.")

    # Precompute daily returns
    daily_rets = prices.pct_change().dropna(how="all")

    rows = []

    for wave in sorted(weights_df["wave"].unique()):
        wave_slice = weights_df[weights_df["wave"] == wave].copy()
        if wave_slice.empty:
            continue

        # Use only tickers that we actually have prices for
        tickers = [t for t in wave_slice["ticker"] if t in daily_rets.columns]
        dropped = [t for t in wave_slice["ticker"] if t not in daily_rets.columns]

        notes = ""
        if dropped:
            notes = f"Dropped tickers with no price data: {', '.join(sorted(set(dropped)))}"

        if not tickers:
            # No usable tickers at all – record NaNs and move on
            rows.append(
                {
                    "Wave": wave,
                    "Return 60D": np.nan,
                    "Alpha 60D": np.nan,
                    "Return 1Y": np.nan,
                    "Alpha 1Y": np.nan,
                    "Benchmark": ", ".join(f"{s}:{w}" for s, w in BENCHMARKS.get(wave, {}).items()),
                    "Notes": notes or "No usable holdings (all missing price data).",
                }
            )
            continue

        # Normalize weights for available tickers
        w = (
            wave_slice.set_index("ticker")["weight"]
            .reindex(tickers)
            .fillna(0.0)
        )
        if w.sum() <= 0:
            w[:] = 1.0 / len(w)
        else:
            w = w / w.sum()

        # Portfolio daily returns
        wave_rets = (daily_rets[tickers] * w.values).sum(axis=1).dropna()

        # Benchmark daily returns for this wave
        bench_spec = BENCHMARKS.get(wave, None)
        if bench_spec:
            bench_tickers = [s for s in bench_spec.keys() if s in daily_rets.columns]
            if bench_tickers:
                bw = pd.Series(bench_spec).reindex(bench_tickers).fillna(0.0)
                if bw.sum() <= 0:
                    bw[:] = 1.0 / len(bw)
                else:
                    bw = bw / bw.sum()

                bench_rets = (daily_rets[bench_tickers] * bw.values).sum(axis=1).dropna()
                bench_label = ", ".join(f"{s} {w:.0%}" for s, w in bench_spec.items())
            else:
                bench_rets = pd.Series(index=daily_rets.index, dtype=float)
                bench_label = "Benchmark tickers missing price data"
                if notes:
                    notes += " | "
                notes += "No benchmark price data."
        else:
            bench_rets = pd.Series(index=daily_rets.index, dtype=float)
            bench_label = "No benchmark configured"

        # Align both series
        idx = wave_rets.index.intersection(bench_rets.index) if not bench_rets.empty else wave_rets.index
        wave_rets = wave_rets.reindex(idx).dropna()
        bench_rets = bench_rets.reindex(idx).fillna(0.0)

        # Compute window returns + alpha
        ret_60 = _series_total_return(wave_rets, 60)
        ret_1y = _series_total_return(wave_rets, 252)

        bench_60 = _series_total_return(bench_rets, 60) if not bench_rets.empty else 0.0
        bench_1y = _series_total_return(bench_rets, 252) if not bench_rets.empty else 0.0

        alpha_60 = ret_60 - bench_60
        alpha_1y = ret_1y - bench_1y

        rows.append(
            {
                "Wave": wave,
                "Return 60D": ret_60,
                "Alpha 60D": alpha_60,
                "Return 1Y": ret_1y,
                "Alpha 1Y": alpha_1y,
                "Benchmark": bench_label,
                "Notes": notes,
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("Wave").sort_index()

    return metrics_df


# ------------------------------------------------------------
# Public entry point used by app.py
# ------------------------------------------------------------

def build_engine() -> pd.DataFrame:
    """
    Thin wrapper so app.py can import build_engine()
    and get a single metrics DataFrame.
    """
    return compute_all_wave_metrics()