"""
waves_engine.py

WAVES Intelligence™ — Live Engine

This script:

1. Loads wave_weights.csv (Primary / Secondary baskets for all Waves).
2. Auto-discovers all Waves.
3. Deduplicates tickers within each Wave and normalizes total weights.
4. Fetches live prices & daily returns via yfinance.
5. Writes positions logs for each Wave:
   logs/positions/<Wave>_positions_YYYYMMDD.csv

   Columns: ticker, weight, market_value

6. Writes performance logs for each Wave:
   logs/performance/<Wave>_performance_history.csv

   Columns: timestamp, nav, return, benchmark_nav, benchmark_return

   These are exactly what app.py expects to render:
   - Performance Curve
   - Metric strip
   - Alpha Dashboard (when benchmark_return is present)

Run manually:
    python waves_engine.py

Or on a schedule (cron, task scheduler, etc.).
"""

import os
import glob
from datetime import datetime, timezone, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAVE_WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")

# base notional per Wave (for market_value column)
BASE_PORTFOLIO_VALUE = 1_000_000.0

# benchmark mapping (easy to customize later)
# default is SPY if no match
WAVE_BENCHMARKS: Dict[str, str] = {
    # examples; you can extend or tweak these names anytime
    "Growth Wave": "SPY",
    "Income Wave": "SPY",
    "AI_Wave": "QQQ",
    "AI Wave": "QQQ",
    "SmallCapGrowth_Wave": "IWM",
    "Small Cap Growth Wave": "IWM",
    "CleanTransitInfra_Wave": "SPY",
    "Clean TransitInfra Wave": "SPY",
    "Clean Transit-Infra Wave": "SPY",
}

DEFAULT_BENCHMARK = "SPY"

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------


def ensure_dirs() -> None:
    """Make sure logs/performance and logs/positions exist."""
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)


def sanitize_wave_for_filename(wave: str) -> str:
    """Turn 'Growth Wave' into 'Growth_Wave' for filenames."""
    return wave.replace(" ", "_").replace("/", "_")


def load_wave_weights() -> pd.DataFrame:
    """Load and normalize wave_weights.csv."""
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        raise FileNotFoundError(f"wave_weights.csv not found at {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wave") or cols.get("wavename") or cols.get("wave_name")
    ticker_col = cols.get("ticker") or cols.get("symbol")
    weight_col = cols.get("weight") or cols.get("target_weight")
    basket_col = cols.get("basket")

    if not wave_col or not ticker_col:
        raise ValueError(
            "wave_weights.csv must have at least columns for 'Wave' and 'Ticker'."
        )

    df["__wave"] = df[wave_col].astype(str).str.strip()
    df["__ticker"] = df[ticker_col].astype(str).str.strip().str.upper()

    if weight_col:
        df["__weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        # if no weights, start equal and normalize later
        df["__weight"] = 1.0

    if basket_col:
        df["__basket"] = df[basket_col].astype(str).str.strip().str.title()
    else:
        df["__basket"] = "Primary"

    return df


def discover_waves(weights_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique Wave names."""
    return sorted(weights_df["__wave"].unique().tolist())


def build_wave_holdings(weights_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each Wave, deduplicate tickers and normalize weights to sum to 1.0.

    Returns:
        dict: {wave_name: holdings_df}

    holdings_df columns:
        ticker, weight, basket (Primary/Secondary), raw_weight
    """
    result: Dict[str, pd.DataFrame] = {}

    for wave in discover_waves(weights_df):
        sub = weights_df[weights_df["__wave"] == wave].copy()
        if sub.empty:
            continue

        # if same ticker appears multiple times (e.g., across baskets),
        # sum its raw weights
        grouped = (
            sub.groupby("__ticker", as_index=False)
            .agg(
                {
                    "__weight": "sum",
                    "__basket": lambda x: ", ".join(sorted(set(x))),
                }
            )
            .rename(columns={"__ticker": "ticker", "__weight": "raw_weight", "__basket": "basket"})
        )

        total_weight = grouped["raw_weight"].sum()
        if total_weight <= 0:
            grouped["weight"] = 1.0 / len(grouped)
        else:
            grouped["weight"] = grouped["raw_weight"] / total_weight

        result[wave] = grouped[["ticker", "weight", "basket", "raw_weight"]].copy()

    return result


def get_all_tickers(wave_holdings: Dict[str, pd.DataFrame]) -> List[str]:
    """Return sorted unique list of all tickers across all Waves."""
    tickers = set()
    for df in wave_holdings.values():
        for t in df["ticker"].unique():
            tickers.add(str(t).upper())
    return sorted(tickers)


def fetch_price_and_return(ticker: str) -> Tuple[float, float]:
    """
    Fetch last close and daily return for a ticker using yfinance.

    Returns:
        (last_price, daily_return)

    daily_return is (last / prev_close - 1).
    If previous close not available, return 0.0.
    """
    try:
        hist = yf.Ticker(ticker).history(period="2d", interval="1d")
        if hist.empty:
            return np.nan, 0.0

        closes = hist["Close"].astype(float)
        last_price = float(closes.iloc[-1])
        if len(closes) >= 2:
            prev = float(closes.iloc[-2])
        else:
            prev = last_price

        if prev == 0:
            ret = 0.0
        else:
            ret = last_price / prev - 1.0

        return last_price, ret
    except Exception:
        return np.nan, 0.0


def fetch_prices_for_universe(tickers: List[str]) -> pd.DataFrame:
    """Fetch price & daily return for a list of tickers, one-by-one (robust)."""
    rows = []
    for t in tickers:
        price, ret = fetch_price_and_return(t)
        rows.append({"ticker": t, "price": price, "return": ret})
    return pd.DataFrame(rows)


def get_benchmark_ticker_for_wave(wave: str) -> str:
    """Simple benchmark mapping (default SPY)."""
    # exact match first
    if wave in WAVE_BENCHMARKS:
        return WAVE_BENCHMARKS[wave]

    # case-insensitive contains match
    lower = wave.lower()
    if "sp500" in lower or "s&p" in lower or "s and p" in lower:
        return "^GSPC"
    if "nasdaq" in lower or "growth" in lower:
        return "QQQ"

    return DEFAULT_BENCHMARK


def load_existing_performance_log(path: str) -> pd.DataFrame:
    """Load performance history if it exists, else empty DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "nav",
                "return",
                "benchmark_nav",
                "benchmark_return",
            ]
        )

    try:
        df = pd.read_csv(path)
        # ensure columns exist
        for col in [
            "timestamp",
            "nav",
            "return",
            "benchmark_nav",
            "benchmark_return",
        ]:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "nav",
                "return",
                "benchmark_nav",
                "benchmark_return",
            ]
        )


# ---------------------------------------------------------------------
# ENGINE CORE
# ---------------------------------------------------------------------


def run_engine_once() -> None:
    """
    Main engine routine (one snapshot run).

    1. Load wave_weights.csv.
    2. Build per-wave holdings.
    3. Fetch prices & returns.
    4. For each Wave:
        - compute portfolio return
        - compute benchmark return
        - update NAV & benchmark NAV
        - write positions & performance logs.
    """
    ensure_dirs()

    print("Loading wave_weights.csv…")
    weights_df = load_wave_weights()
    wave_holdings = build_wave_holdings(weights_df)
    wave_names = list(wave_holdings.keys())

    if not wave_names:
        print("No Waves discovered. Exiting.")
        return

    print(f"Discovered Waves: {', '.join(wave_names)}")

    all_tickers = get_all_tickers(wave_holdings)
    print(f"Total tickers in universe: {len(all_tickers)}")

    print("Fetching prices & daily returns via yfinance…")
    prices_df = fetch_prices_for_universe(all_tickers)
    prices_df.set_index("ticker", inplace=True)

    now = datetime.now(timezone.utc)
    today_str = date.today().strftime("%Y%m%d")

    for wave in wave_names:
        print(f"\nProcessing Wave: {wave}")
        hdf = wave_holdings[wave].copy()

        # attach price & return
        hdf["ticker"] = hdf["ticker"].astype(str).str.upper()
        hdf = hdf.join(prices_df, on="ticker")

        # compute market_value based on normalized weight
        hdf["market_value"] = hdf["weight"] * BASE_PORTFOLIO_VALUE

        # portfolio return = sum(weight * return)
        hdf["return_contrib"] = hdf["weight"] * hdf["return"].fillna(0.0)
        portfolio_ret = float(hdf["return_contrib"].sum())

        # benchmark return
        bench_ticker = get_benchmark_ticker_for_wave(wave)
        bench_price, bench_ret = fetch_price_and_return(bench_ticker)

        print(
            f"  Wave daily return: {portfolio_ret:.4%}   "
            f"(benchmark {bench_ticker}: {bench_ret:.4%})"
        )

        # ----------------- write positions log -----------------
        wave_file_prefix = sanitize_wave_for_filename(wave)
        pos_filename = f"{wave_file_prefix}_positions_{today_str}.csv"
        pos_path = os.path.join(LOGS_POS_DIR, pos_filename)

        pos_df = hdf[["ticker", "weight", "market_value"]].copy()
        pos_df.to_csv(pos_path, index=False)
        print(f"  Wrote positions log: {pos_path}")

        # ----------------- update performance log -----------------
        perf_filename = f"{wave_file_prefix}_performance_history.csv"
        perf_path = os.path.join(LOGS_PERF_DIR, perf_filename)

        perf_df = load_existing_performance_log(perf_path)

        if perf_df.empty:
            prev_nav = 1.0
            prev_bench_nav = 1.0
        else:
            prev_nav = float(perf_df["nav"].iloc[-1])
            prev_bench_nav = float(perf_df["benchmark_nav"].iloc[-1])

        new_nav = prev_nav * (1.0 + portfolio_ret)
        new_bench_nav = prev_bench_nav * (1.0 + bench_ret)

        new_row = {
            "timestamp": now.isoformat(),
            "nav": new_nav,
            "return": portfolio_ret,
            "benchmark_nav": new_bench_nav,
            "benchmark_return": bench_ret,
        }

        perf_df = pd.concat(
            [perf_df, pd.DataFrame([new_row])], ignore_index=True
        )
        perf_df.to_csv(perf_path, index=False)
        print(f"  Updated performance log: {perf_path}")


# ---------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------


if __name__ == "__main__":
    print("=== WAVES Intelligence™ Live Engine ===")
    run_engine_once()
    print("\nRun complete.")
