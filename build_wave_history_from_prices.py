"""
Build wave history from prices with full strategy pipeline.

This script generates wave_history.csv with complete strategy-adjusted returns for equity waves.
For equity growth waves, it uses waves_engine.compute_history_nav() to apply the full strategy
pipeline including:
- Momentum overlay: Weight tilting based on 60-day momentum
- VIX overlay: Exposure scaling and safe allocation based on VIX regime
- Regime detection: Market regime-based exposure adjustments
- Volatility targeting: Dynamic volatility scaling
- Trend confirmation: Additional trend-based filters
- Safe allocation: Blending with safe assets (SGOV, BIL, etc.)

This ensures wave_history.csv contains returns that reflect the complete strategy stack,
matching the S&P 500 Wave reference implementation and providing full strategy stacking parity.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

# Import normalize_ticker directly from file to avoid helpers/__init__.py dependencies
# Note: helpers/__init__.py imports modules requiring streamlit, which may not be available
# in all environments. This direct import pattern avoids that dependency chain.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
_ticker_normalize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ticker_normalize)
normalize_ticker = _ticker_normalize.normalize_ticker

# Import waves_engine for full strategy pipeline
try:
    import waves_engine
    WAVES_ENGINE_AVAILABLE = True
    print("[INFO] waves_engine loaded - will use full strategy pipeline for equity waves")
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    print("[WARN] waves_engine not available - will use simplified VIX-only overlay")


# ------------------------
# Configuration
# ------------------------

PRICES_FILE = "prices.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"
OUTPUT_FILE = "wave_history.csv"
SNAPSHOT_FILE = "wave_coverage_snapshot.json"

# Minimum coverage threshold: waves must have >= 90% weight coverage
MIN_COVERAGE_THRESHOLD = 0.90

# VIX and regime thresholds (matching waves_engine.py)
VIX_TICKER = "VIX"
SPY_TICKER = "SPY"

# Regime classification thresholds (based on SPY 60-day return)
def classify_regime(ret_60d):
    """Classify market regime based on 60-day return."""
    if pd.isna(ret_60d):
        return "neutral"
    if ret_60d <= -0.12:
        return "panic"
    if ret_60d <= -0.04:
        return "downtrend"
    if ret_60d < 0.06:
        return "neutral"
    return "uptrend"

# VIX exposure factor calculation
def get_vix_exposure_factor(vix_level):
    """Calculate VIX-based exposure adjustment factor."""
    if pd.isna(vix_level) or vix_level <= 0:
        return 1.0  # Neutral if missing
    
    if vix_level < 15:
        return 1.15
    elif vix_level < 20:
        return 1.05
    elif vix_level < 25:
        return 0.95
    elif vix_level < 30:
        return 0.85
    elif vix_level < 40:
        return 0.75
    else:
        return 0.60

# Check if wave is equity (VIX overlay applies)
def is_equity_wave(wave_name):
    """Determine if a wave should use VIX overlay (equity waves only)."""
    crypto_keywords = ["Crypto", "Bitcoin"]
    income_keywords = ["Income", "Muni", "Treasury", "SmartSafe"]
    
    for keyword in crypto_keywords:
        if keyword in wave_name:
            return False
    for keyword in income_keywords:
        if keyword in wave_name:
            return False
    
    return True

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
# Compute VIX and SPY regime data
# ------------------------

print("Computing VIX and market regime data...")

# Extract VIX levels if available
vix_levels = None
if VIX_TICKER in price_wide.columns:
    vix_levels = price_wide[VIX_TICKER].copy()
    print(f"  Found VIX data: {len(vix_levels.dropna())} valid observations")
else:
    print(f"  [WARN] VIX ticker '{VIX_TICKER}' not found in prices. VIX overlay will be disabled.")

# Compute SPY 60-day returns for regime classification
spy_60d_returns = None
if SPY_TICKER in price_wide.columns:
    spy_prices = price_wide[SPY_TICKER].copy()
    spy_60d_returns = spy_prices.pct_change(periods=60)
    print(f"  Computed SPY 60-day returns for regime classification")
else:
    print(f"  [WARN] SPY ticker '{SPY_TICKER}' not found. Regime detection will default to 'neutral'.")


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

    # Build base wave dataframe
    df_wave = pd.DataFrame({
        "date": wave_rets.index,
        "wave": wave,
        "portfolio_return": wave_rets.values,
        "benchmark_return": bench_rets.reindex(wave_rets.index).values,
    }).dropna()
    
    # Apply strategy overlays for equity waves using waves_engine
    is_equity = is_equity_wave(wave)
    
    if is_equity and WAVES_ENGINE_AVAILABLE:
        # Use waves_engine.compute_history_nav() for full strategy pipeline
        # This applies: momentum, VIX, regime detection, volatility targeting, safe allocation, etc.
        print(f"[INFO] Computing full strategy pipeline for '{wave}' using waves_engine...")
        
        try:
            # Prepare price DataFrame for waves_engine (date index, ticker columns)
            # waves_engine expects prices, not returns
            price_df_for_engine = prices_pivot.copy()
            
            # Call waves_engine with full strategy pipeline
            result_df = waves_engine.compute_history_nav(
                wave_name=wave,
                mode="Standard",
                days=len(df_wave),  # Use same date range
                include_diagnostics=True,
                price_df=price_df_for_engine
            )
            
            if result_df is not None and not result_df.empty and 'wave_ret' in result_df.columns and 'bm_ret' in result_df.columns:
                # Success! Use strategy-adjusted returns from waves_engine
                # Align with our date range
                result_df = result_df.reindex(df_wave["date"])
                
                # Replace portfolio_return and benchmark_return with strategy-adjusted values
                df_wave["portfolio_return"] = result_df["wave_ret"].values
                df_wave["benchmark_return"] = result_df["bm_ret"].values
                
                # Extract diagnostics if available
                if hasattr(result_df, 'attrs') and 'diagnostics' in result_df.attrs:
                    diag_df = result_df.attrs['diagnostics']
                    if diag_df is not None and not diag_df.empty:
                        # Align diagnostics with wave dates
                        diag_aligned = diag_df.set_index('Date').reindex(df_wave["date"])
                        df_wave["vix_level"] = diag_aligned["vix"].values if "vix" in diag_aligned.columns else np.nan
                        df_wave["vix_regime"] = diag_aligned["regime"].values if "regime" in diag_aligned.columns else "neutral"
                        df_wave["exposure_used"] = diag_aligned["exposure"].values if "exposure" in diag_aligned.columns else 1.0
                    else:
                        # No diagnostics, use defaults
                        df_wave["vix_level"] = np.nan
                        df_wave["vix_regime"] = "neutral"
                        df_wave["exposure_used"] = 1.0
                else:
                    # No diagnostics, use defaults
                    df_wave["vix_level"] = np.nan
                    df_wave["vix_regime"] = "neutral"
                    df_wave["exposure_used"] = 1.0
                
                df_wave["overlay_active"] = True
                print(f"[SUCCESS] Full strategy pipeline applied for '{wave}' ({len(result_df)} days)")
            else:
                # waves_engine failed, fall back to simplified VIX overlay
                print(f"[WARN] waves_engine returned empty data for '{wave}', falling back to simplified VIX overlay")
                raise ValueError("waves_engine returned empty data")
                
        except Exception as e:
            # If waves_engine fails, fall back to simplified VIX-only overlay
            print(f"[WARN] waves_engine failed for '{wave}': {e}")
            print(f"[INFO] Falling back to simplified VIX overlay for '{wave}'")
            
            if vix_levels is not None and spy_60d_returns is not None:
                # Align VIX and SPY data with wave dates
                vix_aligned = vix_levels.reindex(df_wave["date"])
                spy_60d_aligned = spy_60d_returns.reindex(df_wave["date"])
                
                # Compute VIX regime for each date
                df_wave["vix_level"] = vix_aligned.values
                df_wave["vix_regime"] = spy_60d_aligned.apply(classify_regime).values
                df_wave["exposure_used"] = vix_aligned.apply(get_vix_exposure_factor).values
                df_wave["overlay_active"] = True
                
                # Apply VIX-only exposure adjustment (simplified)
                df_wave["portfolio_return"] = df_wave["portfolio_return"] * df_wave["exposure_used"]
            else:
                # No VIX data available
                df_wave["vix_level"] = np.nan
                df_wave["vix_regime"] = "neutral"
                df_wave["exposure_used"] = 1.0
                df_wave["overlay_active"] = False
    
    elif is_equity and not WAVES_ENGINE_AVAILABLE:
        # waves_engine not available, use simplified VIX-only overlay
        print(f"[INFO] Using simplified VIX overlay for '{wave}' (waves_engine not available)")
        
        if vix_levels is not None and spy_60d_returns is not None:
            # Align VIX and SPY data with wave dates
            vix_aligned = vix_levels.reindex(df_wave["date"])
            spy_60d_aligned = spy_60d_returns.reindex(df_wave["date"])
            
            # Compute VIX regime for each date
            df_wave["vix_level"] = vix_aligned.values
            df_wave["vix_regime"] = spy_60d_aligned.apply(classify_regime).values
            df_wave["exposure_used"] = vix_aligned.apply(get_vix_exposure_factor).values
            df_wave["overlay_active"] = True
            
            # Apply VIX-only exposure adjustment (simplified)
            df_wave["portfolio_return"] = df_wave["portfolio_return"] * df_wave["exposure_used"]
        else:
            # No VIX data available
            df_wave["vix_level"] = np.nan
            df_wave["vix_regime"] = "neutral"
            df_wave["exposure_used"] = 1.0
            df_wave["overlay_active"] = False
    else:
        # Non-equity waves (crypto, income): no strategy overlays
        df_wave["vix_level"] = np.nan
        df_wave["vix_regime"] = "neutral"
        df_wave["exposure_used"] = 1.0
        df_wave["overlay_active"] = False

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
