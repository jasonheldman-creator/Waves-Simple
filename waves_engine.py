# waves_engine.py
#
# WAVES Intelligence™ Multi-Wave Engine v0.3
# This version:
#   • Loads all Waves from wave_weights.csv
#   • Builds & logs EVERY wave EXCEPT SP500_Wave
#   • Writes positions + daily performance per wave
#
# Later:
#   • We'll add SP500_Wave as a special flagship engine
#   • We'll add custom strategy logic (VIX ladder, secondary baskets, etc.)

import os
from datetime import datetime, date

import pandas as pd
import yfinance as yf

# --------------- Config ---------------

UNIVERSE_CSV = "list.csv"
WAVE_WEIGHTS_CSV = "wave_weights.csv"

LOG_DIR = "logs"
PERF_FILE = "wave_performance_daily.csv"

# We'll skip this one for now and add it later as the flagship:
SP500_WAVE_NAME = "SP500_Wave"

# Default benchmark for now – we can customize per wave later
DEFAULT_BENCHMARK = "SPY"


# --------------- Helpers ---------------

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def load_universe():
    """
    Load the master universe from list.csv and normalize columns.
    We mainly need Ticker / Name / Sector; Price helps as a fallback.
    """
    if not os.path.exists(UNIVERSE_CSV):
        raise FileNotFoundError(f"Universe file '{UNIVERSE_CSV}' not found.")

    df = pd.read_csv(UNIVERSE_CSV)

    # Map columns from your current sheet into standard names where needed
    col_map = {}
    if "Company" in df.columns:
        col_map["Company"] = "Name"
    if "Weight" in df.columns and "IndexWeight" not in df.columns:
        col_map["Weight"] = "IndexWeight"
    if "Market Value" in df.columns and "MarketValue" not in df.columns:
        col_map["Market Value"] = "MarketValue"

    df = df.rename(columns=col_map)

    # Ensure minimal required columns exist
    required = ["Ticker", "Name", "Sector"]
    for col in required:
        if col not in df.columns:
            if col == "Sector":
                df["Sector"] = "Unclassified"
            else:
                raise ValueError(
                    f"Missing column '{col}' in '{UNIVERSE_CSV}'. "
                    f"Found columns: {list(df.columns)}"
                )

    # Optional numeric columns
    if "Price" not in df.columns:
        df["Price"] = 0.0

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str)
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    return df


def load_all_wave_weights():
    """
    Load the full wave_weights.csv and return a cleaned DataFrame
    with columns: Wave, Ticker, TargetWeight.
    """
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

    # Drop zero / negative weights
    df = df[df["Weight"] > 0.0]

    if df.empty:
        raise ValueError("wave_weights.csv has no positive weights.")

    # Collapse duplicates within each Wave/Ticker
    df = (
        df.groupby(["Wave", "Ticker"], as_index=False)["Weight"]
        .sum()
    )

    # Normalize weights *within each wave* to sum to 1.0
    df["TargetWeight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum()
    )
    df = df.drop(columns=["Weight"])

    return df  # columns: Wave, Ticker, TargetWeight


def fetch_latest_prices(tickers, benchmark_symbol=DEFAULT_BENCHMARK):
    """
    Fetch latest close prices for a list of tickers via yfinance.
    Returns: (prices_df, benchmark_price)
    prices_df index is Ticker, column is Price.
    """
    prices = []
    for t in tickers:
        try:
            yt = yf.Ticker(t)
            hist = yt.history(period="1d")
            if hist.empty:
                continue
            last_close = float(hist.iloc[-1]["Close"])
            prices.append({"Ticker": t, "Price": last_close})
        except Exception:
            # If one symbol fails, we just skip it for now
            continue

    df_prices = pd.DataFrame(prices)
    if not df_prices.empty:
        df_prices.set_index("Ticker", inplace=True)

    bench_price = None
    try:
        bh = yf.Ticker(benchmark_symbol).history(period="1d")
        if not bh.empty:
            bench_price = float(bh.iloc[-1]["Close"])
    except Exception:
        pass

    return df_prices, bench_price


def write_positions_log(wave_name: str, df_positions: pd.DataFrame):
    """
    Save today's positions for a wave to logs/<Wave>_positions_YYYYMMDD.csv
    """
    today_str = date.today().strftime("%Y%m%d")
    filename = f"{wave_name}_positions_{today_str}.csv"
    path = os.path.join(LOG_DIR, filename)
    df_positions.to_csv(path, index=False)
    print(f"[{wave_name}] wrote positions log → {path}")


def append_performance_log(wave_name: str, nav: float, bench_nav: float,
                           benchmark_symbol: str = DEFAULT_BENCHMARK):
    """
    Append a row to logs/wave_performance_daily.csv for a given wave.
    """
    today_str = date.today().isoformat()
    row = {
        "date": today_str,
        "wave": wave_name,
        "nav": nav,
        "benchmark": benchmark_symbol,
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
    print(f"[{wave_name}] appended performance row → {path}")


# --------------- Core wave runner ---------------

def run_wave(universe: pd.DataFrame, weights_df: pd.DataFrame, wave_name: str):
    """
    Build & log a single wave using its weights in weights_df.
    """
    print(f"\n=== Running engine for {wave_name} ===")

    # Filter to this wave
    subset = weights_df[weights_df["Wave"] == wave_name].copy()
    if subset.empty:
        print(f"[{wave_name}] No holdings found, skipping.")
        return

    # Join with universe for names / sectors / base prices
    df = subset.merge(universe, on="Ticker", how="left")

    # Drop any rows we absolutely cannot resolve
    df = df.dropna(subset=["Name"])
    if df.empty:
        print(f"[{wave_name}] No valid holdings after universe join, skipping.")
        return

    tickers = df["Ticker"].unique().tolist()
    price_df, bench_price = fetch_latest_prices(tickers)

    if not price_df.empty:
        df = df.merge(price_df.reset_index(), on="Ticker", how="left")
        # Price_x is from universe, Price_y from yfinance
        df["LivePrice"] = df["Price_y"].fillna(df["Price_x"])
        df = df.drop(columns=["Price_x", "Price_y"])
    else:
        df["LivePrice"] = df["Price"]

    # For now, NAV = 1.0 (we’ll evolve this over time)
    nav = 1.0
    df["DollarWeight"] = df["TargetWeight"] * nav

    positions_cols = [
        "Ticker",
        "Name",
        "Sector",
        "TargetWeight",
        "LivePrice",
        "DollarWeight",
    ]
    positions_log = df[positions_cols].copy()
    write_positions_log(wave_name, positions_log)

    append_performance_log(wave_name, nav, bench_price or 0.0)
    print(f"=== {wave_name} engine completed ===")


# --------------- Main ---------------

def main():
    print("=== WAVES Intelligence™ Multi-Wave Engine (non-SP500 waves) ===")
    ensure_log_dir()
    universe = load_universe()
    all_weights = load_all_wave_weights()

    waves = sorted(all_weights["Wave"].unique())
    # Skip SP500_Wave for now; we’ll add a special engine for it later.
    waves = [w for w in waves if w != SP500_WAVE_NAME]

    print(f"Discovered waves in weights file: {waves}")

    for w in waves:
        try:
            run_wave(universe, all_weights, w)
        except Exception as e:
            print(f"[{w}] ERROR while running engine: {e}")

    print("\nAll non-SP500 waves completed.\n")


if __name__ == "__main__":
    main()
