# waves_engine.py — First-pass engine for Growth_Wave
#
# Purpose:
# - Loads universe (list.csv) and wave weights (wave_weights.csv)
# - Extracts Growth_Wave
# - Pulls live prices
# - Computes portfolio NAV & daily return vs SPY
# - Writes:
#     logs/growth_positions_YYYYMMDD.csv
#     logs/growth_performance_daily.csv
#
# This is a skeleton: we can later add VIX rules, secondary baskets, SmartSafe, etc.

import os
from datetime import datetime, date

import pandas as pd
import yfinance as yf

UNIVERSE_CSV = "list.csv"
WAVE_WEIGHTS_CSV = "wave_weights.csv"

GROWTH_WAVE_NAME = "Growth_Wave"
BENCHMARK_SYMBOL = "SPY"

LOG_DIR = "logs"
POSITIONS_PREFIX = "growth_positions_"
PERF_FILE = "growth_performance_daily.csv"


def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def load_universe():
    if not os.path.exists(UNIVERSE_CSV):
        raise FileNotFoundError(f"Universe file '{UNIVERSE_CSV}' not found.")

    df = pd.read_csv(UNIVERSE_CSV)

    col_map = {}
    if "Company" in df.columns:
        col_map["Company"] = "Name"
    if "Weight" in df.columns:
        col_map["Weight"] = "IndexWeight"
    if "Market Value" in df.columns:
        col_map["Market Value"] = "MarketValue"

    df = df.rename(columns=col_map)

    required = ["Ticker", "Name", "Sector", "IndexWeight", "MarketValue", "Price"]
    for col in required:
        if col not in df.columns:
            if col == "Sector":
                df["Sector"] = "Unclassified"
            elif col == "MarketValue":
                df["MarketValue"] = 0.0
            elif col == "Price":
                df["Price"] = 0.0
            else:
                raise ValueError(
                    f"Missing column '{col}' in '{UNIVERSE_CSV}'. "
                    f"Found columns: {list(df.columns)}"
                )

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str)
    df["Sector"] = df["Sector"].astype(str).str.strip()

    df["IndexWeight"] = pd.to_numeric(df["IndexWeight"], errors="coerce").fillna(0.0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    return df


def load_wave_weights():
    if not os.path.exists(WAVE_WEIGHTS_CSV):
        raise FileNotFoundError(f"Wave weights file '{WAVE_WEIGHTS_CSV}' not found.")

    df = pd.read_csv(WAVE_WEIGHTS_CSV, comment="#")

    required_cols = {"Ticker", "Wave", "Weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing column(s) {missing} in '{WAVE_WEIGHTS_CSV}'. "
            f"Found columns: {list(df.columns)}"
        )

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    df = df[df["Weight"] > 0]
    if df.empty:
        raise ValueError("All Growth_Wave weights are zero or invalid.")

    # Deduplicate (Wave, Ticker)
    df = (
        df.groupby(["Wave", "Ticker"], as_index=False)["Weight"]
        .sum()
    )

    # Filter to Growth_Wave only
    growth = df[df["Wave"] == GROWTH_WAVE_NAME].copy()
    if growth.empty:
        raise ValueError(f"No holdings found for '{GROWTH_WAVE_NAME}' in wave_weights.csv.")

    # Normalize weights to 1.0
    total = growth["Weight"].sum()
    if total <= 0:
        raise ValueError(f"Total weight for '{GROWTH_WAVE_NAME}' is non-positive.")
    growth["TargetWeight"] = growth["Weight"] / total
    growth = growth.drop(columns=["Weight"])

    return growth


def fetch_latest_prices(tickers):
    """Return DataFrame: Ticker, Price, BenchmarkPrice."""
    prices = []
    for t in tickers:
        try:
            yt = yf.Ticker(t)
            hist = yt.history(period="1d")
            if hist.empty:
                continue
            last_row = hist.iloc[-1]
            prices.append({"Ticker": t, "Price": float(last_row["Close"])})
        except Exception:
            continue

    df_prices = pd.DataFrame(prices)
    if not df_prices.empty:
        df_prices.set_index("Ticker", inplace=True)

    # Benchmark
    bench_price = None
    try:
        bench_hist = yf.Ticker(BENCHMARK_SYMBOL).history(period="1d")
        if not bench_hist.empty:
            bench_price = float(bench_hist.iloc[-1]["Close"])
    except Exception:
        pass

    return df_prices, bench_price


def load_yesterday_positions():
    """Optional: load yesterday's positions if they exist (for P&L / turnover later)."""
    today_str = date.today().strftime("%Y%m%d")
    # look for the most recent positions file before today
    if not os.path.exists(LOG_DIR):
        return None

    files = [
        f for f in os.listdir(LOG_DIR)
        if f.startswith(POSITIONS_PREFIX) and f.endswith(".csv")
    ]
    if not files:
        return None

    # sort and pick last
    files.sort()
    latest_file = files[-1]
    path = os.path.join(LOG_DIR, latest_file)
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def write_positions_log(df_positions):
    today_str = date.today().strftime("%Y%m%d")
    filename = f"{POSITIONS_PREFIX}{today_str}.csv"
    path = os.path.join(LOG_DIR, filename)
    df_positions.to_csv(path, index=False)
    print(f"Wrote positions log: {path}")


def append_performance_log(nav, bench_nav):
    """Append a row to growth_performance_daily.csv."""
    today_str = date.today().isoformat()
    row = {
        "date": today_str,
        "wave": GROWTH_WAVE_NAME,
        "nav": nav,
        "benchmark": BENCHMARK_SYMBOL,
        "benchmark_nav": bench_nav,
        "timestamp": datetime.utcnow().isoformat(),
    }

    path = os.path.join(LOG_DIR, PERF_FILE)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"Appended performance log: {path}")


def run_growth_engine():
    print("=== WAVES Growth_Wave Engine Run ===")
    ensure_log_dir()

    universe = load_universe()
    growth_weights = load_wave_weights()   # Ticker, Wave, TargetWeight

    # Join with universe for names/sectors
    df = growth_weights.merge(universe, on="Ticker", how="left")
    df = df.dropna(subset=["Name"])
    if df.empty:
        raise ValueError(
            f"After joining with list.csv, no valid universe rows for '{GROWTH_WAVE_NAME}'."
        )

    # Fetch latest prices
    tickers = df["Ticker"].dropna().unique().tolist()
    price_df, bench_price = fetch_latest_prices(tickers)

    # Attach live prices
    if not price_df.empty:
        df = df.merge(price_df.reset_index(), on="Ticker", how="left")
        df["Price"] = df["Price_y"].fillna(df["Price_x"])
        df = df.drop(columns=["Price_x", "Price_y"])
    else:
        # fall back to universe prices only
        if "Price" not in df.columns:
            raise ValueError("No live or universe prices available for Growth_Wave holdings.")

    # Assume portfolio NAV = 1.0 for now (we can change later to track over time)
    portfolio_nav = 1.0
    df["DollarWeight"] = df["TargetWeight"] * portfolio_nav

    # Save positions log
    positions_cols = [
        "Ticker",
        "Name",
        "Sector",
        "TargetWeight",
        "Price",
        "DollarWeight",
    ]
    positions_log = df[positions_cols].copy()
    write_positions_log(positions_log)

    # Append performance log
    # For now, we log NAV=1.0 and benchmark latest price.
    append_performance_log(nav=portfolio_nav, bench_nav=bench_price or 0.0)

    print("Growth_Wave engine run completed.")


if __name__ == "__main__":
    run_growth_engine()
