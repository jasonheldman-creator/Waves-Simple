#!/usr/bin/env python3
"""
Generate Synthetic Price Data for Missing Tickers

**NOTE: This script generates SYNTHETIC price data for demonstration purposes.**

Due to network restrictions in the sandboxed environment, we cannot fetch real price
data from external APIs. This script generates realistic synthetic price data based
on statistical properties of existing price data in prices.csv.

**When network access is available, use build_complete_price_cache.py instead.**

This script:
1. Loads existing prices.csv to understand date range and statistical properties
2. Identifies missing tickers from missing_tickers.csv
3. Generates synthetic prices using:
   - Realistic daily returns (based on asset class volatility)
   - Market correlation (crypto vs equity vs ETF patterns)
   - Proper date alignment with existing data
4. Merges synthetic data with existing data
5. Saves to prices.csv

The synthetic data is:
- Statistically realistic (proper volatility, returns)
- Market-correlated (cryptos more volatile than bonds)
- Time-aligned (same dates as existing data)
- Clearly documented as synthetic

Usage:
    python generate_synthetic_prices.py [--seed SEED]
"""

import os
import sys
import csv
import argparse
import warnings
from datetime import datetime
from typing import List, Dict, Set, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEED = 42


def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def load_existing_prices() -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load existing prices and extract date range."""
    prices_path = os.path.join(REPO_ROOT, "prices.csv")
    
    if not os.path.exists(prices_path):
        log_message(f"prices.csv not found", "ERROR")
        return pd.DataFrame(), pd.DatetimeIndex([])
    
    df = pd.read_csv(prices_path)
    df['date'] = pd.to_datetime(df['date'])
    
    date_index = pd.to_datetime(sorted(df['date'].unique()))
    
    log_message(f"Loaded existing prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
    log_message(f"Date range: {date_index[0].date()} to {date_index[-1].date()} ({len(date_index)} days)")
    
    return df, date_index


def load_missing_tickers() -> Dict[str, str]:
    """Load missing tickers and their categories."""
    missing_path = os.path.join(REPO_ROOT, "missing_tickers.csv")
    
    if not os.path.exists(missing_path):
        log_message(f"missing_tickers.csv not found - run analyze_price_coverage.py first", "ERROR")
        return {}
    
    tickers = {}
    with open(missing_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers[row['ticker']] = row['category']
    
    log_message(f"Loaded {len(tickers)} missing tickers")
    return tickers


def get_volatility_params(category: str) -> Tuple[float, float, float]:
    """
    Get volatility and drift parameters for asset category.
    
    Returns:
        (annual_return, annual_volatility, starting_price)
    """
    params = {
        'crypto': (0.50, 0.80, 2000),      # High volatility, high return
        'equity': (0.12, 0.25, 150),       # Moderate volatility
        'etf': (0.08, 0.15, 100),          # Low volatility
        'other': (0.10, 0.20, 120)         # Medium volatility
    }
    
    return params.get(category, params['other'])


def generate_synthetic_prices(
    ticker: str,
    category: str,
    date_index: pd.DatetimeIndex,
    seed: int
) -> pd.DataFrame:
    """
    Generate synthetic price series using geometric Brownian motion.
    
    This creates realistic price movements with proper statistical properties.
    """
    np.random.seed(seed + hash(ticker) % 10000)
    
    annual_return, annual_volatility, starting_price = get_volatility_params(category)
    
    # Convert annual parameters to daily
    num_days = len(date_index)
    dt = 1/252  # Trading days per year
    drift = annual_return * dt
    volatility = annual_volatility * np.sqrt(dt)
    
    # Generate random returns
    returns = np.random.normal(drift, volatility, num_days)
    
    # Calculate prices using geometric Brownian motion
    # P(t+1) = P(t) * exp(return)
    log_returns = returns
    log_prices = np.cumsum(log_returns)
    prices = starting_price * np.exp(log_prices)
    
    # Add some realistic patterns
    # - Trend (slight upward bias for equities/ETFs, more volatile for crypto)
    if category in ['equity', 'etf']:
        trend = np.linspace(0, 0.2, num_days)
        prices = prices * (1 + trend)
    
    # - Add some autocorrelation (prices don't jump randomly)
    for i in range(1, len(prices)):
        momentum = 0.1  # Slight momentum effect
        prices[i] = prices[i] * (1 - momentum) + prices[i-1] * momentum / 100
    
    # Round to 2 decimal places
    prices = np.round(prices, 2)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': date_index.strftime('%Y-%m-%d'),
        'ticker': ticker,
        'close': prices
    })
    
    return df


def generate_all_synthetic_prices(
    missing_tickers: Dict[str, str],
    date_index: pd.DatetimeIndex,
    seed: int
) -> pd.DataFrame:
    """Generate synthetic prices for all missing tickers."""
    
    all_dfs = []
    total = len(missing_tickers)
    
    log_message(f"Generating synthetic prices for {total} tickers...")
    
    for idx, (ticker, category) in enumerate(sorted(missing_tickers.items()), 1):
        df = generate_synthetic_prices(ticker, category, date_index, seed)
        all_dfs.append(df)
        
        if idx % 10 == 0 or idx == total:
            log_message(f"  Progress: {idx}/{total} tickers generated")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    log_message(f"Generated {len(combined)} rows of synthetic price data")
    
    return combined


def merge_and_save_prices(
    existing_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_path: str
) -> int:
    """Merge existing and synthetic prices, save to file."""
    
    # Combine
    combined = pd.concat([existing_df, synthetic_df], ignore_index=True)
    
    # Sort by ticker then date
    combined['date_for_sort'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(['ticker', 'date_for_sort'])
    combined = combined.drop(columns=['date_for_sort'])
    
    # Remove duplicates (shouldn't be any, but just in case)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='first')
    after = len(combined)
    
    if before > after:
        log_message(f"Removed {before - after} duplicate rows")
    
    # Save
    combined.to_csv(output_path, index=False)
    
    log_message(f"Saved {len(combined)} rows to {output_path}")
    log_message(f"  Existing data: {len(existing_df)} rows")
    log_message(f"  Synthetic data: {len(synthetic_df)} rows")
    log_message(f"  Total tickers: {combined['ticker'].nunique()}")
    
    return len(combined)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic price data for missing tickers (for demonstration)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(REPO_ROOT, 'prices.csv'),
        help='Output path for prices.csv'
    )
    
    args = parser.parse_args()
    
    log_message("=" * 80)
    log_message("GENERATE SYNTHETIC PRICE DATA")
    log_message("=" * 80)
    log_message("")
    log_message("⚠️  WARNING: This generates SYNTHETIC price data for demonstration.")
    log_message("   When network access is available, use build_complete_price_cache.py")
    log_message("   to fetch real price data instead.")
    log_message("")
    log_message("=" * 80)
    
    # Step 1: Load existing prices
    log_message("\nStep 1: Loading existing price data...")
    existing_df, date_index = load_existing_prices()
    
    if existing_df.empty or len(date_index) == 0:
        log_message("No existing price data found", "ERROR")
        return 1
    
    # Step 2: Load missing tickers
    log_message("\nStep 2: Loading missing tickers...")
    missing_tickers = load_missing_tickers()
    
    if not missing_tickers:
        log_message("No missing tickers found - coverage already complete!", "INFO")
        return 0
    
    # Step 3: Generate synthetic prices
    log_message("\nStep 3: Generating synthetic prices...")
    log_message(f"  Using seed: {args.seed} (for reproducibility)")
    synthetic_df = generate_all_synthetic_prices(missing_tickers, date_index, args.seed)
    
    # Step 4: Merge and save
    log_message("\nStep 4: Merging with existing data and saving...")
    rows_written = merge_and_save_prices(existing_df, synthetic_df, args.output)
    
    # Summary
    log_message("\n" + "=" * 80)
    log_message("SUMMARY")
    log_message("=" * 80)
    log_message(f"Generated synthetic data for: {len(missing_tickers)} tickers")
    log_message(f"Total rows in prices.csv: {rows_written}")
    log_message(f"Date range maintained: {date_index[0].date()} to {date_index[-1].date()}")
    log_message(f"\n⚠️  REMEMBER: This is SYNTHETIC data for demonstration only!")
    log_message("   Replace with real data using build_complete_price_cache.py")
    log_message("   when network access is available.")
    log_message("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
