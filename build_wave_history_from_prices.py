import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta


# ------------------------
# Configuration
# ------------------------

PRICES_FILE = "prices.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"
OUTPUT_FILE = "wave_history.csv"

# Map each Wave to its benchmark ticker (edit this as needed)
BENCHMARK_BY_WAVE = {
    "S&P Wave": "SPY",
    "Growth Wave": "SPY",
    "Small Cap Growth Wave": "IWM",
    "Small-Mid Cap Growth Wave": "IJH",
    "Quantum Computing Wave": "QQQ",
    "Future Power & Energy Wave": "XLE",
    "Clean Transit-Infrastructure Wave": "ICLN",
    "Income Wave": "AGG",
    # add the rest of your 15 Waves here
}


# ------------------------
# Helper function to fetch prices
# ------------------------

def fetch_and_save_prices(tickers_to_fetch, prices_file):
    """
    Fetch historical prices for the given tickers using yfinance.
    Saves the data to prices_file in the format expected by this script.
    
    Args:
        tickers_to_fetch: List of ticker symbols
        prices_file: Path to save the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[ERROR] yfinance is not installed. Cannot fetch prices.")
        print("        Please install it with: pip install yfinance")
        return False
    
    try:
        print(f"Fetching historical prices for {len(tickers_to_fetch)} tickers...")
        
        # Fetch last 5 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        # Download data for all tickers
        data = yf.download(
            tickers=tickers_to_fetch,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
            group_by="ticker"
        )
        
        if data.empty:
            print("[ERROR] No data returned from yfinance.")
            return False
        
        # Convert to the expected format: date, ticker, close
        rows = []
        
        if len(tickers_to_fetch) == 1:
            # Single ticker case
            ticker = tickers_to_fetch[0]
            if "Close" in data.columns:
                for date, row in data.iterrows():
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "close": row["Close"]
                    })
        else:
            # Multiple tickers case
            for ticker in tickers_to_fetch:
                try:
                    if (ticker, "Close") in data.columns:
                        ticker_data = data[(ticker, "Close")]
                        for date, close_price in ticker_data.items():
                            if pd.notna(close_price):
                                rows.append({
                                    "date": date.strftime("%Y-%m-%d"),
                                    "ticker": ticker,
                                    "close": close_price
                                })
                except Exception as e:
                    print(f"[WARN] Could not process ticker {ticker}: {e}")
                    continue
        
        if not rows:
            print("[ERROR] No valid price data could be extracted.")
            return False
        
        # Create DataFrame and save
        prices_df = pd.DataFrame(rows)
        prices_df.to_csv(prices_file, index=False)
        print(f"Successfully fetched and saved {len(rows)} price records to {prices_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices: {e}")
        return False


# ------------------------
# Load data
# ------------------------

# First, load wave weights to determine what tickers we need
print("Loading wave weights...")
if not os.path.exists(WAVE_WEIGHTS_FILE):
    print(f"[ERROR] {WAVE_WEIGHTS_FILE} not found. Cannot proceed.")
    sys.exit(1)

weights = pd.read_csv(WAVE_WEIGHTS_FILE)

if not {"wave", "ticker", "weight"}.issubset(weights.columns):
    raise ValueError("wave_weights.csv must contain columns: wave, ticker, weight")

# Get unique tickers from wave_weights, plus benchmark tickers
required_tickers = set(weights["ticker"].dropna().unique())
for benchmark in BENCHMARK_BY_WAVE.values():
    required_tickers.add(benchmark)
required_tickers = sorted(list(required_tickers))

print(f"Found {len(required_tickers)} unique tickers in wave weights and benchmarks")

# Check if prices.csv exists, if not fetch the data
if not os.path.exists(PRICES_FILE):
    print(f"[WARN] {PRICES_FILE} not found. Attempting to fetch historical prices...")
    
    success = fetch_and_save_prices(required_tickers, PRICES_FILE)
    
    if not success:
        print("\n" + "="*70)
        print("[ERROR] Unable to fetch price data.")
        print("        The application cannot proceed without price data.")
        print("        Please ensure:")
        print("        1. You have internet connectivity")
        print("        2. yfinance is installed: pip install yfinance")
        print("        3. Or manually create prices.csv with columns: date, ticker, close")
        print("="*70)
        sys.exit(1)
    
    print(f"Successfully created {PRICES_FILE}")
else:
    print(f"Found existing {PRICES_FILE}")

print("Loading prices...")
prices = pd.read_csv(PRICES_FILE, parse_dates=["date"])
prices = prices.sort_values(["ticker", "date"])

if prices.empty:
    print(f"[ERROR] {PRICES_FILE} is empty. Cannot proceed.")
    sys.exit(1)

# Pivot prices to wide format and compute daily returns
print("Computing daily returns...")
price_wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
rets = price_wide.pct_change().dropna(how="all")  # daily returns, NaN where missing


# ------------------------
# Build wave history
# ------------------------

records = []

for wave, wdf in weights.groupby("wave"):
    if wave not in BENCHMARK_BY_WAVE:
        print(f"[WARN] No benchmark defined for wave '{wave}'. Skipping.")
        continue

    bench_ticker = BENCHMARK_BY_WAVE[wave]
    if bench_ticker not in rets.columns:
        print(f"[WARN] Benchmark ticker '{bench_ticker}' not found in prices. Skipping wave '{wave}'.")
        continue

    wdf = wdf.copy()
    wdf["weight"] = wdf["weight"].astype(float)

    # Normalize weights so they sum to 1 by absolute weight (per wave)
    total_abs = wdf["weight"].abs().sum()
    if total_abs == 0:
        print(f"[WARN] Wave '{wave}' has zero total weight. Skipping.")
        continue
    wdf["norm_weight"] = wdf["weight"] / total_abs

    tickers = list(wdf["ticker"])
    missing = [t for t in tickers if t not in rets.columns]
    if missing:
        print(f"[WARN] Some tickers for wave '{wave}' not in prices: {missing}")

    # Use only tickers that we actually have returns for
    tickers = [t for t in tickers if t in rets.columns]
    if not tickers:
        print(f"[WARN] No valid tickers for wave '{wave}'. Skipping.")
        continue

    wdf = wdf[wdf["ticker"].isin(tickers)].set_index("ticker")
    wave_rets = (rets[tickers] * wdf["norm_weight"]).sum(axis=1)

    bench_rets = rets[bench_ticker]

    df_wave = pd.DataFrame({
        "date": wave_rets.index,
        "wave": wave,
        "portfolio_return": wave_rets.values,
        "benchmark_return": bench_rets.reindex(wave_rets.index).values,
    }).dropna()

    records.append(df_wave)

if not records:
    raise RuntimeError("No wave history could be built. Check inputs and mappings.")

wave_history = pd.concat(records, ignore_index=True).sort_values(["wave", "date"])

print(f"Writing {OUTPUT_FILE} with {len(wave_history)} rows...")
wave_history.to_csv(OUTPUT_FILE, index=False)
print("Done.")
