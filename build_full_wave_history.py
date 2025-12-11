"""
build_full_wave_history.py — WAVES Intelligence™ History Engine (Option B)

Purpose
-------
Rebuilds DAILY NAV and alpha history for each Wave using:
- Current target weights from wave_weights.csv
- Constant weights (no drift) with optional monthly rebalance hook
- Custom blended benchmarks per Wave (editable below)
- yfinance daily prices

Outputs
-------
Writes a single CSV: Full_Wave_History.csv

Columns:
    Date           (YYYY-MM-DD)
    Wave           (e.g. "AI Wave")
    Position       (1, 2, 3, … within that Wave)
    Weight         (target portfolio weight of that ticker)
    Ticker         (stock / ETF symbol)
    Price          (daily close from yfinance, adjusted)
    MarketValue    (Weight * Wave NAV, with NAV starting at 1.0)
    NAV            (Wave NAV)
    WaveReturn     (daily Wave return)
    BenchReturn    (daily benchmark return for that Wave)
    Alpha          (WaveReturn - BenchReturn)

Run
---
From your repo root (same folder as app.py):

    python build_full_wave_history.py

Requires:
    pip install yfinance pandas numpy python-dateutil
"""

import os
import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit(
        "yfinance is not installed. Run: pip install yfinance"
    )


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

# Where to write the full history table
HISTORY_FILE = "Full_Wave_History.csv"

# Input weights file you just built (12 Waves, etc.)
WAVE_WEIGHTS_FILE = "wave_weights.csv"

# History window (you can change this later if you want)
YEARS_BACK = 10  # look back ~10 years

# --------------------------------------------------------------------
# BENCHMARK DEFINITIONS
# --------------------------------------------------------------------
# You can EDIT this dict to match the custom benchmarks
# you configured in the console. Each Wave maps to a dict of
# {benchmark_ticker: weight}.
#
# Default: sensible placeholders. Update to your final mixes later.
# Example:
#   "AI Wave": {"SMH": 0.4, "IGV": 0.3, "AIQ": 0.3}

WAVE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Tech / AI
    "AI Wave": {"SMH": 0.4, "IGV": 0.3, "AIQ": 0.3},

    # Core Equity
    "S&P Wave": {"SPY": 1.0},
    "Growth Wave": {"QQQ": 1.0},
    "Small-Mid Cap Growth Wave": {"IWM": 1.0},

    # Income / defensive
    "Income Wave": {"SCHD": 0.6, "BIL": 0.4},
    "SmartSafe Wave": {"BIL": 1.0},

    # Thematic — tweak as needed
    "Future Power & Energy Wave": {"ICLN": 0.5, "XLE": 0.5},
    "Clean Transit-Infrastructure Wave": {"ICLN": 0.5, "XLI": 0.5},
    "Cloud & Software Wave": {"IGV": 0.7, "QQQ": 0.3},
    "Quantum Computing Wave": {"QQQ": 0.7, "SMH": 0.3},
    "Crypto Wave": {"WGMI": 0.5, "BTC-USD": 0.5},
    "Crypto Income Wave": {"BTC-USD": 0.4, "ETH-USD": 0.3, "BIL": 0.3},
}

DEFAULT_BENCH = {"SPY": 1.0}  # used if a Wave is missing in WAVE_BENCHMARKS


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------


def load_wave_weights(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"wave_weights.csv not found at '{path}'. "
            "Make sure you committed it to the repo root."
        )

    df = pd.read_csv(path)

    # Normalize column names
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    expected = {"wave", "ticker", "weight"}
    missing = expected - set(cols)
    if missing:
        raise ValueError(
            f"wave_weights.csv is missing columns: {', '.join(sorted(missing))}"
        )

    # Clean up
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()]
        raise ValueError(
            f"Found non-numeric weights in wave_weights.csv:\n{bad_rows}"
        )

    # Re-normalize weights to sum 1.0 within each Wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda s: s / s.sum() if s.sum() != 0 else s
    )

    return df


def get_date_range(years_back: int = YEARS_BACK):
    today = dt.date.today()
    start = today - dt.timedelta(days=365 * years_back)
    return start, today


def collect_all_tickers(
    weights_df: pd.DataFrame, bench_map: Dict[str, Dict[str, float]]
) -> List[str]:
    tickers = set(weights_df["ticker"].unique())

    for bench_spec in bench_map.values():
        tickers.update(bench_spec.keys())

    # Crypto tickers like "BTC-USD" are also fine; yfinance handles them
    return sorted(tickers)


def download_price_history(tickers: List[str],
                           start: dt.date,
                           end: dt.date) -> pd.DataFrame:
    """
    Returns a DataFrame index=date, columns=ticker, values=Adj Close.
    """
    if not tickers:
        raise ValueError("No tickers provided to download_price_history().")

    print(f"Downloading history for {len(tickers)} tickers from {start} to {end}...")

    data = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )

    # yfinance returns different shapes depending on number of tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: level 0 = ticker, level 1 = field (Adj Close, etc.)
        # We'll take "Adj Close" if it exists, otherwise "Close".
        fields = set(level for _, level in data.columns)
        preferred = "Adj Close" if "Adj Close" in fields else "Close"
        wide = {}

        for ticker in tickers:
            if (ticker, preferred) in data.columns:
                wide[ticker] = data[(ticker, preferred)]
            else:
                # If missing entirely, just fill with NaN
                wide[ticker] = pd.Series(index=data.index, dtype="float64")

        prices = pd.DataFrame(wide)
    else:
        # Single ticker: data is a normal DataFrame with columns like
        # ['Open','High','Low','Close','Adj Close','Volume']
        fields = set(data.columns)
        col = "Adj Close" if "Adj Close" in fields else "Close"
        prices = data[[col]].copy()
        prices.columns = [tickers[0]]

    prices = prices.sort_index()
    prices = prices.ffill().bfill()  # forward/backward fill to clean small gaps

    return prices


def build_nav_from_prices(
    price_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    price_df: index=date, columns=ticker
    weights: dict[ticker] -> weight (must sum ~1.0)
    Returns NAV series starting at 1.0.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype="float64")

    # Normalize just in case
    if w.sum() == 0:
        raise ValueError("Weights sum to zero in build_nav_from_prices().")
    w = w / w.sum()

    sub = price_df[tickers].copy()

    # If any ticker is missing, fill with its column median price
    for t in tickers:
        if t not in sub.columns:
            sub[t] = np.nan
        sub[t] = sub[t].ffill().bfill()

    # Turn each ticker into a price RELATIVE series (start at 1.0)
    rel = sub / sub.iloc[0]
    # Portfolio NAV = weighted sum of rel prices; start at 1.0
    nav = rel.mul(w, axis=1).sum(axis=1)
    nav /= nav.iloc[0]

    return nav


def get_wave_benchmark_spec(wave_name: str) -> Dict[str, float]:
    """
    Returns the benchmark spec {ticker: weight} for a given Wave.
    Falls back to DEFAULT_BENCH if not defined.
    """
    spec = WAVE_BENCHMARKS.get(wave_name)
    if spec is None:
        print(f"[WARN] No benchmark defined for '{wave_name}', using SPY 100%.")
        spec = DEFAULT_BENCH
    # Normalize
    total = sum(spec.values())
    spec = {k: v / total for k, v in spec.items()}
    return spec


# --------------------------------------------------------------------
# MAIN BUILD
# --------------------------------------------------------------------


def build_full_history():
    start, end = get_date_range(YEARS_BACK)

    # 1) Load weights
    weights_df = load_wave_weights(WAVE_WEIGHTS_FILE)
    waves = sorted(weights_df["wave"].unique())
    print(f"Found {len(waves)} Waves in {WAVE_WEIGHTS_FILE}: {waves}")

    # 2) Download all prices (positions + benchmarks)
    all_tickers = collect_all_tickers(weights_df, WAVE_BENCHMARKS)
    prices = download_price_history(all_tickers, start, end)

    # 3) Prepare output rows
    records = []

    for wave in waves:
        wave_weights_df = weights_df[weights_df["wave"] == wave].copy()
        if wave_weights_df.empty:
            continue

        # Convert to dict[ticker] -> weight
        pos_weights = (
            wave_weights_df.set_index("ticker")["weight"].to_dict()
        )

        # Wave NAV
        wave_nav = build_nav_from_prices(prices, pos_weights)

        # Wave daily returns
        wave_ret = wave_nav.pct_change().fillna(0.0)

        # Benchmark NAV
        bench_spec = get_wave_benchmark_spec(wave)
        bench_nav = build_nav_from_prices(prices, bench_spec)
        bench_ret = bench_nav.pct_change().fillna(0.0)

        # Alpha
        alpha = wave_ret - bench_ret

        # For each date and each position, output a row
        # Position numbers are just 1..N by sort-order of ticker
        wave_weights_df = wave_weights_df.sort_values("ticker").reset_index(drop=True)
        wave_weights_df["position"] = wave_weights_df.index + 1

        for date, nav_val in wave_nav.items():
            w_ret = float(wave_ret.loc[date])
            b_ret = float(bench_ret.loc[date])
            a_val = float(alpha.loc[date])

            # Price snapshot for each ticker
            for _, row in wave_weights_df.iterrows():
                ticker = row["ticker"]
                weight = float(row["weight"])
                position_num = int(row["position"])

                price = float(prices.loc[date, ticker]) if ticker in prices.columns else np.nan
                market_val = nav_val * weight  # NAV * portfolio weight

                records.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Wave": wave,
                        "Position": position_num,
                        "Weight": weight,
                        "Ticker": ticker,
                        "Price": price,
                        "MarketValue": market_val,
                        "NAV": nav_val,
                        "WaveReturn": w_ret,
                        "BenchReturn": b_ret,
                        "Alpha": a_val,
                    }
                )

    # 4) Build DataFrame and write CSV
    if not records:
        raise RuntimeError("No records generated — check your weights/price data.")

    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values(["Wave", "Date", "Position"])
    out_df.to_csv(HISTORY_FILE, index=False)

    print(f"Wrote full wave history to: {HISTORY_FILE}")
    print(f"Rows: {len(out_df):,}")


if __name__ == "__main__":
    build_full_history()