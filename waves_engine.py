"""
WAVES Intelligence™ - Multi-Wave Engine

This script:
  1) Loads wave_weights.csv (Wave, Ticker, Weight).
  2) For EACH Wave:
       - Fetches latest prices via yfinance
       - Calculates position values & share counts for a notional portfolio
       - Writes a daily positions log: logs/<Wave>_positions_YYYYMMDD.csv
       - Appends daily summary row to logs/wave_performance_daily.csv
  3) Provides a clean console audit of what was run.

You can run it locally with:
    python3 waves_engine.py
"""

import os
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import yfinance as yf

# ---------- CONFIGURABLE CONSTANTS ----------

# Notional portfolio size per Wave (for share calc / performance normalization)
BASE_PORTFOLIO_VALUE = 1_000_000.0

WEIGHTS_FILE = "wave_weights.csv"
LOG_DIR = "logs"
PERF_LOG_FILE = os.path.join(LOG_DIR, "wave_performance_daily.csv")


# ---------- HELPER FUNCTIONS ----------

def ensure_log_dir() -> None:
    """Create logs/ directory if it does not exist."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def load_wave_weights(path: str) -> pd.DataFrame:
    """
    Load wave_weights.csv and normalize columns.

    Expected columns: Wave,Ticker,Weight
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wave weights file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    required = {"Wave", "Ticker", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    # Clean strings / whitespace
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Convert weights to float
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Drop rows with bad data
    df = df.dropna(subset=["Wave", "Ticker", "Weight"])

    return df


def fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch latest close price for each ticker using yfinance.

    Returns: dict[ticker] = price (float)
    """
    prices: Dict[str, float] = {}

    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2d")
            if hist.empty:
                continue
            # use last available close
            price = float(hist["Close"].iloc[-1])
            prices[t] = price
        except Exception:
            # Skip problem tickers but continue engine
            continue

    return prices


def compute_positions(
    wave_name: str,
    weights_df: pd.DataFrame,
    prices: Dict[str, float],
    as_of: dt.date,
) -> pd.DataFrame:
    """
    Build a positions dataframe for a single Wave.

    Columns:
      Wave, Date, Ticker, Weight, Price, Shares, PositionValue
    """
    df = weights_df.copy()

    df["Wave"] = wave_name
    df["Date"] = as_of.isoformat()

    df["Price"] = df["Ticker"].map(prices)

    # Drop tickers with no price
    df = df.dropna(subset=["Price"])

    # Normalize total weight per wave (in case it doesn't sum exactly to 1)
    total_weight = df["Weight"].sum()
    if total_weight <= 0:
        raise ValueError(f"Total weight for {wave_name} is non-positive.")

    df["NormWeight"] = df["Weight"] / total_weight

    df["PositionValue"] = df["NormWeight"] * BASE_PORTFOLIO_VALUE
    df["Shares"] = df["PositionValue"] / df["Price"]

    # Select and order columns for logging
    df_out = df[
        [
            "Wave",
            "Date",
            "Ticker",
            "NormWeight",
            "Price",
            "Shares",
            "PositionValue",
        ]
    ].rename(columns={"NormWeight": "Weight"})

    return df_out


def append_performance_log(
    wave_name: str,
    as_of: dt.date,
    positions_df: pd.DataFrame,
) -> None:
    """
    Append a summary row for this Wave to the combined performance log.

    Columns:
      Date, Wave, NumHoldings, TotalWeight, PortfolioValue
    """
    if positions_df.empty:
        return

    total_weight = positions_df["Weight"].sum()
    portfolio_value = positions_df["PositionValue"].sum()
    num_holdings = len(positions_df)

    row = {
        "Date": as_of.isoformat(),
        "Wave": wave_name,
        "NumHoldings": num_holdings,
        "TotalWeight": float(total_weight),
        "PortfolioValue": float(portfolio_value),
    }

    # If log exists, append; otherwise create
    if os.path.exists(PERF_LOG_FILE):
        perf_df = pd.read_csv(PERF_LOG_FILE)
        perf_df = pd.concat([perf_df, pd.DataFrame([row])], ignore_index=True)
    else:
        perf_df = pd.DataFrame([row])

    perf_df.to_csv(PERF_LOG_FILE, index=False)


def run_wave_engine_for(
    wave_name: str, all_weights: pd.DataFrame, as_of: dt.date
) -> None:
    """
    Run the engine for a single Wave:
      - Filter weights
      - Fetch prices
      - Compute positions
      - Write positions log
      - Append performance log
    """
    print(f"\n=== WAVES {wave_name} Engine Run ===")

    wave_weights = all_weights[all_weights["Wave"] == wave_name]
    if wave_weights.empty:
        print(f"  Skipping {wave_name}: no weights found.")
        return

    tickers = sorted(wave_weights["Ticker"].unique().tolist())
    print(f"  Holdings in {wave_name}: {len(tickers)} symbols")

    prices = fetch_latest_prices(tickers)
    if not prices:
        print(f"  No prices retrieved for {wave_name}. Skipping.")
        return

    positions_df = compute_positions(wave_name, wave_weights, prices, as_of)

    if positions_df.empty:
        print(f"  No valid positions for {wave_name}. Skipping.")
        return

    # Write positions log
    date_str = as_of.strftime("%Y%m%d")
    fname = f"{wave_name}_positions_{date_str}.csv"
    pos_path = os.path.join(LOG_DIR, fname)
    positions_df.to_csv(pos_path, index=False)
    print(f"  Wrote positions log: {pos_path}")

    # Append to performance log
    append_performance_log(wave_name, as_of, positions_df)
    print(f"  Appended performance log: {PERF_LOG_FILE}")

    print(f"=== {wave_name} engine run completed. ===")


# ---------- MAIN ENTRYPOINT ----------

def main() -> None:
    print("== WAVES Intelligence™ Multi-Wave Engine ==")

    ensure_log_dir()

    today = dt.date.today()

    try:
        weights_df = load_wave_weights(WEIGHTS_FILE)
    except Exception as e:
        print(f"Error loading weights file: {e}")
        return

    waves = sorted(weights_df["Wave"].unique().tolist())
    print(f"Discovered Waves in weights file: {waves}")

    if not waves:
        print("No waves found in weights file. Nothing to run.")
        return

    for wave_name in waves:
        try:
            run_wave_engine_for(wave_name, weights_df, today)
        except Exception as e:
            # Keep running other waves even if one fails
            print(f"  ERROR while running {wave_name}: {e}")

    print("\nAll wave engine runs completed.\n")


if __name__ == "__main__":
    main()
