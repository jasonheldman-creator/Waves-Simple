#!/usr/bin/env python3
"""
WAVES Intelligence™ – Local Engine
----------------------------------

Reads:
    - list.csv          : full universe from S&P 500 (and others)
    - wave_weights.csv  : Wave,Ticker,Weight for each Wave

Does:
    - For each Wave in wave_weights.csv:
        * join with universe to get Name / Sector / base Price
        * fetch latest prices from yfinance (fallback to universe Price)
        * compute dollar weights for NAV=1.0
        * write positions log: logs/<Wave>_positions_YYYYMMDD.csv
        * append daily performance log: logs/<Wave>_performance_daily.csv
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

# ---------- Files & directories ----------

UNIVERSE_FILE = "list.csv"          # Your master universe file
WEIGHTS_FILE = "wave_weights.csv"   # Wave, Ticker, Weight
LOG_DIR = "logs"

# Map each wave to a benchmark symbol (you can edit this any time)
BENCHMARK_BY_WAVE: Dict[str, str] = {
    "AI_Wave": "QQQ",
    "Growth_Wave": "QQQ",
    "Income_Wave": "DVY",
    "Future_Energy_Wave": "ICLN",
    "CleanTransitInfra_Wave": "IDRV",
    "RWA_Income_Wave": "VNQ",
    "SmallCap_Growth_Wave": "IWO",
    "SmallMid_Growth_Wave": "IJT",
    "SP500_Wave": "SPY",  # when you add the S&P engine later
}


# ---------- Helpers ----------

def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV with consistent options and nicer error message."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV '{path}': {e}")


# ---------- Load universe & weights ----------

def load_universe() -> pd.DataFrame:
    """
    Load list.csv.

    Expected columns (based on your file):
        Ticker, Company, Weight, Sector, Market Value, Price
    """
    df = _safe_read_csv(UNIVERSE_FILE)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Make sure required columns exist
    required = {"Ticker", "Company", "Sector", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{UNIVERSE_FILE} missing columns: {missing}")

    # Rename to consistent names we’ll use later
    df = df.rename(columns={"Company": "Name"})
    return df


def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv (Wave,Ticker,Weight) and clean it:
      - ignore lines starting with '#'
      - drop blank lines
      - enforce float weight
      - deduplicate (Wave, Ticker) by summing weights
    """
    df = _safe_read_csv(
        WEIGHTS_FILE,
        comment="#",          # ignore commented lines
        skip_blank_lines=True
    )

    df.columns = [c.strip() for c in df.columns]

    required = {"Wave", "Ticker", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{WEIGHTS_FILE} missing columns: {missing}")

    # Strip whitespace
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    # Convert weight to float, drop bad rows
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Wave", "Ticker", "Weight"])

    # Deduplicate tickers inside each wave (sum weights)
    df = df.groupby(["Wave", "Ticker"], as_index=False)["Weight"].sum()

    # Normalize so each wave sums to 1.0 (optional but nice)
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum()
    )

    return df


# ---------- Price fetching ----------

def fetch_latest_prices(
    tickers: List[str],
    benchmark_symbol: str
) -> Tuple[pd.DataFrame, float]:
    """
    Fetch latest prices for tickers + benchmark from yfinance.

    Returns:
        prices_df: DataFrame with columns ["Ticker", "Price"]
        bench_price: float or 0.0 on failure
    """
    all_symbols = list(set(tickers + [benchmark_symbol]))
    if not all_symbols:
        return pd.DataFrame(columns=["Ticker", "Price"]), 0.0

    try:
        data = yf.download(
            tickers=all_symbols,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False
        )
    except Exception as e:
        print(f"[WARN] yfinance download failed: {e}")
        return pd.DataFrame(columns=["Ticker", "Price"]), 0.0

    # For single/ multi ticker yfinance returns different shapes, handle both
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices_series = data["Adj Close"].iloc[-1]
    else:
        # fallback – nothing usable
        return pd.DataFrame(columns=["Ticker", "Price"]), 0.0

    if isinstance(prices_series, pd.Series):
        prices_df = prices_series.reset_index()
        prices_df.columns = ["Ticker", "Price"]
    else:
        # Single ticker case
        prices_df = pd.DataFrame({
            "Ticker": [all_symbols[0]],
            "Price": [float(prices_series)]
        })

    # Extract benchmark
    bench_rows = prices_df.loc[prices_df["Ticker"] == benchmark_symbol, "Price"]
    bench_price = float(bench_rows.values[0]) if not bench_rows.empty else 0.0

    # Keep only requested tickers
    prices_df = prices_df[prices_df["Ticker"].isin(tickers)].copy()

    return prices_df, bench_price


# ---------- Logging ----------

def write_positions_log(wave_name: str, df: pd.DataFrame) -> None:
    """
    Write positions snapshot for a wave:
        logs/<Wave>_positions_YYYYMMDD.csv
    """
    date_tag = datetime.utcnow().strftime("%Y%m%d")
    filename = f"{wave_name}_positions_{date_tag}.csv"
    path = os.path.join(LOG_DIR, filename)

    df.to_csv(path, index=False)
    print(f"[{wave_name}] Wrote positions log: {path}")


def append_performance_log(
    wave_name: str,
    nav: float,
    bench_symbol: str,
    bench_price: float
) -> None:
    """
    Append one row to logs/<Wave>_performance_daily.csv
    """
    filename = f"{wave_name}_performance_daily.csv"
    path = os.path.join(LOG_DIR, filename)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "wave": wave_name,
        "nav": nav,
        "benchmark": bench_symbol,
        "benchmark_price": bench_price,
    }

    if os.path.exists(path):
        df = _safe_read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print(f"[{wave_name}] Appended performance log: {path}")


# ---------- Core engine for one wave ----------

def run_wave_engine(
    wave_name: str,
    universe_df: pd.DataFrame,
    weights_df: pd.DataFrame
) -> None:
    """
    Build one wave using current universe & weights.
    """
    print(f"\n=== WAVES {wave_name} Engine Run ===")

    # Filter weights for this wave
    w = weights_df[weights_df["Wave"] == wave_name].copy()
    if w.empty:
        print(f"[{wave_name}] No weights found – skipping.")
        return

    # Merge with universe for Name / Sector / base Price
    df = pd.merge(
        w,
        universe_df,
        how="left",
        on="Ticker",
        validate="m:1"  # each ticker should map to at most one universe row
    )

    # Check for missing universe rows
    missing = df[df["Name"].isna()]
    if not missing.empty:
        missing_tickers = ", ".join(sorted(missing["Ticker"].unique()))
        print(f"[WARN] {wave_name}: tickers missing from universe: {missing_tickers}")
        # Drop them so they don't break calculations
        df = df[~df["Name"].isna()].copy()

    if df.empty:
        print(f"[{wave_name}] No valid holdings after joining universe – skipping.")
        return

    # Normalize weights again just in case we dropped something
    df["Weight"] = df["Weight"] / df["Weight"].sum()

    # Get benchmark
    bench_symbol = BENCHMARK_BY_WAVE.get(wave_name, "SPY")

    # Fetch latest prices
    tickers = df["Ticker"].dropna().unique().tolist()
    prices_df, bench_price = fetch_latest_prices(tickers, bench_symbol)

    # Attach live prices, fallback to universe Price if we miss any
    if not prices_df.empty:
        df = pd.merge(
            df,
            prices_df,
            on="Ticker",
            how="left",
            suffixes=("", "_live")
        )
        # Use live price where available, else universe Price
        df["Price"] = df["Price_live"].fillna(df["Price"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_live")])
    else:
        print(f"[WARN] {wave_name}: no live prices fetched, using universe Price only.")
        if "Price" not in df.columns:
            raise RuntimeError(f"{wave_name}: No price data available at all.")

    # Assume NAV = 1.0 for now (we’ll extend this later)
    portfolio_nav = 1.0
    df["TargetWeight"] = df["Weight"]
    df["DollarWeight"] = df["TargetWeight"] * portfolio_nav

    # Select columns for positions log
    positions_cols = [
        "Ticker",
        "Name",
        "Sector",
        "TargetWeight",
        "Price",
        "DollarWeight",
    ]

    # Some universe rows may not have Sector; keep column anyway
    for col in positions_cols:
        if col not in df.columns:
            df[col] = None

    positions_df = df[positions_cols].copy()

    # Write logs
    write_positions_log(wave_name, positions_df)
    append_performance_log(
        wave_name,
        nav=portfolio_nav,
        bench_symbol=bench_symbol,
        bench_price=bench_price,
    )

    print(f"[{wave_name}] Engine run completed.")


# ---------- Main ----------

def main() -> None:
    ensure_log_dir()
    universe_df = load_universe()
    weights_df = load_wave_weights()

    # Discover all waves dynamically
    waves = sorted(weights_df["Wave"].unique())
    print("Found waves:")
    for w in waves:
        print(f"  - {w}")

    # Run engine for each wave
    for wave_name in waves:
        try:
            run_wave_engine(wave_name, universe_df, weights_df)
        except Exception as e:
            print(f"[ERROR] Wave {wave_name} failed: {e}")


if __name__ == "__main__":
    main()
