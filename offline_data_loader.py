"""
Offline Data Loader for Wave Analytics

This module provides offline data loading capabilities using cached price data
from prices.csv. It ensures all 28 waves can be populated with data even when
live market data feeds are unavailable.

Key Features:
1. Load price data from prices.csv (long-format)
2. Distribute data to wave-specific directories
3. Generate missing benchmark and NAV files
4. Ensure all waves have operational status
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Constants
INITIAL_NAV_VALUE = 100.0  # Starting NAV value for portfolio calculations
SHARES_MULTIPLIER = 100  # Multiplier for converting weights to arbitrary share counts

# Import from waves_engine
from waves_engine import (
    get_all_wave_ids,
    get_display_name_from_wave_id,
    WAVE_WEIGHTS,
    Holding,
)

# Import from analytics_pipeline
from analytics_pipeline import (
    get_wave_analytics_dir,
    ensure_directory_exists,
    resolve_wave_tickers,
    resolve_wave_benchmarks,
    normalize_ticker,
)


def load_prices_csv(prices_csv_path: str = 'prices.csv') -> pd.DataFrame:
    """
    Load prices from long-format CSV.
    
    Args:
        prices_csv_path: Path to prices.csv file
        
    Returns:
        DataFrame in long format with columns: date, ticker, close
    """
    if not os.path.exists(prices_csv_path):
        raise FileNotFoundError(f"Price data file not found: {prices_csv_path}")
    
    # Load CSV
    df = pd.read_csv(prices_csv_path)
    
    # Ensure required columns exist
    if 'date' not in df.columns or 'ticker' not in df.columns or 'close' not in df.columns:
        raise ValueError(f"prices.csv must have columns: date, ticker, close")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove any NaN prices
    df = df.dropna(subset=['close'])
    
    return df


def convert_to_wide_format(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format price data to wide format (date x ticker).
    
    Args:
        df_long: DataFrame in long format (date, ticker, close)
        
    Returns:
        DataFrame in wide format with date index and ticker columns
    """
    df_wide = df_long.pivot(index='date', columns='ticker', values='close')
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide = df_wide.sort_index()
    
    return df_wide


def extract_ticker_prices(df_wide: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Extract prices for specific tickers from wide-format DataFrame.
    
    Args:
        df_wide: Wide-format price DataFrame
        tickers: List of tickers to extract
        
    Returns:
        DataFrame with only the requested tickers
    """
    # Normalize tickers
    normalized_tickers = [normalize_ticker(t) for t in tickers]
    
    # Find available tickers
    available = [t for t in normalized_tickers if t in df_wide.columns]
    
    # Extract available columns
    if not available:
        # Return empty DataFrame with proper index
        return pd.DataFrame(index=df_wide.index)
    
    return df_wide[available].copy()


def compute_simple_nav(prices_df: pd.DataFrame, holdings: List[Holding]) -> pd.DataFrame:
    """
    Compute simple NAV from prices and holdings (equal-weighted if no weights).
    
    Args:
        prices_df: Wide-format price DataFrame
        holdings: List of Holding objects with tickers and weights
        
    Returns:
        DataFrame with 'nav' column
    """
    if prices_df.empty:
        # Return empty NAV DataFrame
        return pd.DataFrame(columns=['nav'])
    
    # Build weights series
    weights = {}
    total_weight = sum(h.weight for h in holdings)
    
    for holding in holdings:
        ticker = normalize_ticker(holding.ticker)
        if ticker in prices_df.columns:
            weights[ticker] = holding.weight / total_weight if total_weight > 0 else 1.0 / len(holdings)
    
    if not weights:
        # No valid tickers found
        return pd.DataFrame(columns=['nav'], index=prices_df.index)
    
    # Compute weighted returns
    # Initialize NAV at starting value
    nav_series = pd.Series(INITIAL_NAV_VALUE, index=prices_df.index)
    
    # Compute daily returns for each ticker
    returns_df = prices_df[list(weights.keys())].pct_change()
    
    # Compute portfolio return as weighted average
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    for ticker, weight in weights.items():
        portfolio_returns += returns_df[ticker].fillna(0) * weight
    
    # Compound returns to get NAV
    nav_series = (1 + portfolio_returns).cumprod() * INITIAL_NAV_VALUE
    
    return pd.DataFrame({'nav': nav_series})


def populate_wave_data(wave_id: str, df_wide: pd.DataFrame, overwrite: bool = False) -> Dict[str, bool]:
    """
    Populate data files for a single wave from cached price data.
    
    Args:
        wave_id: Wave identifier
        df_wide: Wide-format price DataFrame
        overwrite: If True, overwrite existing files
        
    Returns:
        Dictionary with success status for each file type
    """
    result = {
        'prices': False,
        'benchmark_prices': False,
        'positions': False,
        'trades': False,
        'nav': False
    }
    
    # Get wave directory
    wave_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(wave_dir)
    
    # Get holdings and benchmarks
    display_name = get_display_name_from_wave_id(wave_id)
    holdings = WAVE_WEIGHTS.get(display_name, [])
    
    if not holdings:
        print(f"Warning: No holdings defined for {wave_id}")
        return result
    
    tickers = [h.ticker for h in holdings]
    benchmark_specs = resolve_wave_benchmarks(wave_id)
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    
    # 1. Generate prices.csv
    prices_path = os.path.join(wave_dir, 'prices.csv')
    if overwrite or not os.path.exists(prices_path):
        prices_df = extract_ticker_prices(df_wide, tickers)
        if not prices_df.empty:
            prices_df.to_csv(prices_path)
            result['prices'] = True
            print(f"✓ Generated prices.csv for {wave_id}: {len(prices_df)} days, {len(prices_df.columns)} tickers")
        else:
            print(f"⚠ No price data available for {wave_id}")
    else:
        result['prices'] = True  # File exists
    
    # 2. Generate benchmark_prices.csv
    benchmark_path = os.path.join(wave_dir, 'benchmark_prices.csv')
    if overwrite or not os.path.exists(benchmark_path):
        if benchmark_tickers:
            benchmark_df = extract_ticker_prices(df_wide, benchmark_tickers)
            if not benchmark_df.empty:
                benchmark_df.to_csv(benchmark_path)
                result['benchmark_prices'] = True
                print(f"✓ Generated benchmark_prices.csv for {wave_id}: {len(benchmark_df.columns)} tickers")
            else:
                print(f"⚠ No benchmark data available for {wave_id}")
    else:
        result['benchmark_prices'] = True  # File exists
    
    # 3. Generate positions.csv (simple snapshot)
    positions_path = os.path.join(wave_dir, 'positions.csv')
    if overwrite or not os.path.exists(positions_path):
        positions_data = []
        for holding in holdings:
            positions_data.append({
                'ticker': holding.ticker,
                'shares': SHARES_MULTIPLIER * holding.weight,  # Arbitrary shares for weight
                'weight': holding.weight,
                'name': holding.name or holding.ticker
            })
        
        if positions_data:
            positions_df = pd.DataFrame(positions_data)
            positions_df.to_csv(positions_path, index=False)
            result['positions'] = True
            print(f"✓ Generated positions.csv for {wave_id}: {len(positions_data)} positions")
    else:
        result['positions'] = True  # File exists
    
    # 4. Generate trades.csv (empty - no trades in static data)
    trades_path = os.path.join(wave_dir, 'trades.csv')
    if overwrite or not os.path.exists(trades_path):
        trades_df = pd.DataFrame(columns=['date', 'ticker', 'action', 'shares', 'price'])
        trades_df.to_csv(trades_path, index=False)
        result['trades'] = True
        print(f"✓ Generated trades.csv for {wave_id}: 0 trades")
    else:
        result['trades'] = True  # File exists
    
    # 5. Generate nav.csv
    nav_path = os.path.join(wave_dir, 'nav.csv')
    if overwrite or not os.path.exists(nav_path):
        # Load prices again to compute NAV
        prices_df = extract_ticker_prices(df_wide, tickers)
        if not prices_df.empty:
            nav_df = compute_simple_nav(prices_df, holdings)
            if not nav_df.empty:
                nav_df.to_csv(nav_path)
                result['nav'] = True
                print(f"✓ Generated nav.csv for {wave_id}: {len(nav_df)} days")
            else:
                print(f"⚠ Could not compute NAV for {wave_id}")
        else:
            print(f"⚠ No price data to compute NAV for {wave_id}")
    else:
        result['nav'] = True  # File exists
    
    return result


def populate_all_waves(prices_csv_path: str = 'prices.csv', overwrite: bool = False) -> Dict[str, Dict[str, bool]]:
    """
    Populate data files for all waves from cached price data.
    
    Args:
        prices_csv_path: Path to prices.csv file
        overwrite: If True, overwrite existing files
        
    Returns:
        Dictionary mapping wave_id to file generation results
    """
    print("=" * 80)
    print("OFFLINE DATA LOADER - Populating All Waves")
    print("=" * 80)
    print(f"Source: {prices_csv_path}")
    print(f"Overwrite existing files: {overwrite}")
    print("")
    
    # Load prices
    print("Loading price data...")
    df_long = load_prices_csv(prices_csv_path)
    df_wide = convert_to_wide_format(df_long)
    
    print(f"✓ Loaded {len(df_wide)} days of data")
    print(f"✓ Loaded {len(df_wide.columns)} unique tickers")
    print(f"  Date range: {df_wide.index[0].strftime('%Y-%m-%d')} to {df_wide.index[-1].strftime('%Y-%m-%d')}")
    print("")
    
    # Get all waves
    wave_ids = get_all_wave_ids()
    print(f"Processing {len(wave_ids)} waves...")
    print("")
    
    # Populate each wave
    results = {}
    for i, wave_id in enumerate(sorted(wave_ids), 1):
        display_name = get_display_name_from_wave_id(wave_id)
        print(f"[{i}/{len(wave_ids)}] {wave_id} ({display_name})")
        print("-" * 80)
        
        result = populate_wave_data(wave_id, df_wide, overwrite=overwrite)
        results[wave_id] = result
        
        print("")
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_waves = len(results)
    successful_waves = sum(1 for r in results.values() if r['prices'] and r['nav'])
    
    print(f"Total waves: {total_waves}")
    print(f"Waves with prices: {sum(1 for r in results.values() if r['prices'])}")
    print(f"Waves with benchmarks: {sum(1 for r in results.values() if r['benchmark_prices'])}")
    print(f"Waves with NAV: {sum(1 for r in results.values() if r['nav'])}")
    print(f"Fully populated: {successful_waves}")
    print("")
    
    return results


if __name__ == '__main__':
    """Run offline data loader to populate all waves."""
    import sys
    
    # Parse command line arguments
    overwrite = '--overwrite' in sys.argv or '-o' in sys.argv
    prices_path = 'prices.csv'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--prices='):
            prices_path = arg.split('=')[1]
    
    # Run loader
    try:
        results = populate_all_waves(prices_csv_path=prices_path, overwrite=overwrite)
        
        # Exit with success if at least some waves were populated
        successful = sum(1 for r in results.values() if r['prices'] and r['nav'])
        if successful > 0:
            print(f"✓ Successfully populated {successful}/{len(results)} waves")
            sys.exit(0)
        else:
            print(f"✗ Failed to populate any waves")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
