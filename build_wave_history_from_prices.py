import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

# Import normalize_ticker directly from file to avoid helpers/__init__.py dependencies
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
_ticker_normalize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ticker_normalize)
normalize_ticker = _ticker_normalize.normalize_ticker


# ------------------------
# Configuration
# ------------------------

PRICES_FILE = "prices.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"
OUTPUT_FILE = "wave_history.csv"
SNAPSHOT_FILE = "wave_coverage_snapshot.json"

# Minimum coverage threshold: waves must have >= 90% weight coverage
MIN_COVERAGE_THRESHOLD = 0.90

# Map each Wave to its benchmark ticker (edit this as needed)
BENCHMARK_BY_WAVE = {
    # Equity Waves
    "S&P 500 Wave": "SPY",
    "Russell 3000 Wave": "VTHR",
    "US MegaCap Core Wave": "SPY",
    "Small Cap Growth Wave": "IWM",
    "Small to Mid Cap Growth Wave": "IJH",
    "US Mid/Small Growth & Semis Wave": "IJH",
    "US Small-Cap Disruptors Wave": "IWM",
    
    # Tech/Growth Waves
    "AI & Cloud MegaCap Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",
    "Next-Gen Compute & Semis Wave": "QQQ",
    
    # Energy/Infrastructure Waves
    "Future Power & Energy Wave": "XLE",
    "Future Energy & EV Wave": "XLE",
    "Clean Transit-Infrastructure Wave": "ICLN",
    "EV & Infrastructure Wave": "ICLN",
    
    # Income Waves
    "Income Wave": "AGG",
    "Vector Muni Ladder Wave": "MUB",
    "Vector Treasury Ladder Wave": "AGG",
    
    # Cash Waves
    "SmartSafe Tax-Free Money Market Wave": "SHV",
    "SmartSafe Treasury Cash Wave": "SHV",
    
    # Crypto Waves
    "Crypto Broad Growth Wave": "BTC-USD",
    "Crypto AI Growth Wave": "BTC-USD",
    "Crypto DeFi Growth Wave": "BTC-USD",
    "Crypto L1 Growth Wave": "BTC-USD",
    "Crypto L2 Growth Wave": "BTC-USD",
    "Crypto Income Wave": "BTC-USD",
    
    # Multi-Asset Waves
    "Infinity Multi-Asset Growth Wave": "SPY",
    "Demas Fund Wave": "SPY",
    
    # Commodity Waves
    "Gold Wave": "GLD",
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

# Apply ticker normalization
print("Normalizing tickers...")
weights["ticker_norm"] = weights["ticker"].apply(normalize_ticker)

# Get unique normalized tickers from wave_weights, plus benchmark tickers
required_tickers = set(weights["ticker_norm"].dropna().unique())
for benchmark in BENCHMARK_BY_WAVE.values():
    required_tickers.add(normalize_ticker(benchmark))
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

# Apply ticker normalization to prices
prices["ticker"] = prices["ticker"].apply(normalize_ticker)
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
coverage_metrics = []

for wave, wdf in weights.groupby("wave"):
    if wave not in BENCHMARK_BY_WAVE:
        print(f"[WARN] No benchmark defined for wave '{wave}'. Skipping.")
        continue

    bench_ticker = normalize_ticker(BENCHMARK_BY_WAVE[wave])
    if bench_ticker not in rets.columns:
        print(f"[WARN] Benchmark ticker '{bench_ticker}' not found in prices. Skipping wave '{wave}'.")
        continue

    wdf = wdf.copy()
    wdf["weight"] = wdf["weight"].astype(float)

    # Calculate total weight and identify available vs missing tickers
    total_weight = wdf["weight"].abs().sum()
    if total_weight == 0:
        print(f"[WARN] Wave '{wave}' has zero total weight. Skipping.")
        continue

    # Use normalized tickers for all operations
    tickers_norm = list(wdf["ticker_norm"])
    available_tickers = [t for t in tickers_norm if t in rets.columns]
    missing_tickers = [t for t in tickers_norm if t not in rets.columns]
    
    # Calculate coverage percentage
    available_weight = wdf[wdf["ticker_norm"].isin(available_tickers)]["weight"].abs().sum()
    coverage_pct = available_weight / total_weight if total_weight > 0 else 0.0
    
    # Track coverage metrics for snapshot
    coverage_metrics.append({
        "wave": wave,
        "total_tickers": len(tickers_norm),
        "available_tickers": len(available_tickers),
        "missing_tickers": len(missing_tickers),
        "missing_ticker_list": missing_tickers,
        "total_weight": float(total_weight),
        "available_weight": float(available_weight),
        "coverage_pct": float(coverage_pct),
        "meets_threshold": bool(coverage_pct >= MIN_COVERAGE_THRESHOLD)
    })
    
    if missing_tickers:
        print(f"[INFO] Wave '{wave}' missing {len(missing_tickers)} tickers: {missing_tickers}")
        print(f"       Coverage: {coverage_pct:.2%} (threshold: {MIN_COVERAGE_THRESHOLD:.2%})")
    
    # Check if wave meets minimum coverage threshold
    if coverage_pct < MIN_COVERAGE_THRESHOLD:
        print(f"[WARN] Wave '{wave}' coverage {coverage_pct:.2%} is below {MIN_COVERAGE_THRESHOLD:.2%} threshold. Skipping.")
        continue
    
    if not available_tickers:
        print(f"[WARN] No valid tickers for wave '{wave}'. Skipping.")
        continue

    # Filter to available tickers and reweight proportionally
    wdf_available = wdf[wdf["ticker_norm"].isin(available_tickers)].copy()
    wdf_available = wdf_available.set_index("ticker_norm")
    
    # Normalize weights so they sum to 1 by absolute weight (proportional reweighting)
    total_abs = wdf_available["weight"].abs().sum()
    wdf_available["norm_weight"] = wdf_available["weight"] / total_abs
    
    wave_rets = (rets[available_tickers] * wdf_available["norm_weight"]).sum(axis=1)

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

# Write coverage snapshot
snapshot = {
    "timestamp": datetime.now().isoformat(),
    "total_waves": len(coverage_metrics),
    "waves_meeting_threshold": sum(1 for m in coverage_metrics if m["meets_threshold"]),
    "waves_below_threshold": sum(1 for m in coverage_metrics if not m["meets_threshold"]),
    "min_coverage_threshold": MIN_COVERAGE_THRESHOLD,
    "waves": coverage_metrics
}

print(f"Writing coverage snapshot to {SNAPSHOT_FILE}...")
with open(SNAPSHOT_FILE, "w") as f:
    json.dump(snapshot, f, indent=2)

# Print summary
print("\n" + "="*70)
print("COVERAGE SUMMARY")
print("="*70)
print(f"Total waves processed: {snapshot['total_waves']}")
print(f"Waves meeting {MIN_COVERAGE_THRESHOLD:.0%} threshold: {snapshot['waves_meeting_threshold']}")
print(f"Waves below threshold: {snapshot['waves_below_threshold']}")
print("\nWaves below threshold:")
for metric in coverage_metrics:
    if not metric["meets_threshold"]:
        print(f"  - {metric['wave']}: {metric['coverage_pct']:.2%} coverage")
print("="*70)
print("Done.")
