"""
offline_data_loader.py

Enhanced offline data loader with diagnostic functionality for Wave readiness.

This module provides functionality to:
- Load and merge data from prices.csv by ticker/date
- Normalize ticker variants (e.g., crypto symbols like BTC-USD)
- Return diagnostic information (missing_tickers, coverage_pct, stale_tickers)
- Generate data_coverage_summary.csv artifact
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np

# Import from waves_engine
from waves_engine import get_all_wave_ids, get_display_name_from_wave_id, WAVE_WEIGHTS

# Import from analytics_pipeline
from analytics_pipeline import (
    MIN_COVERAGE_FOR_ANALYTICS,
    MIN_DAYS_FOR_ANALYTICS,
    MAX_DAYS_STALE,
    compute_data_ready_status
)

# Ticker normalization mapping
TICKER_NORMALIZATION_MAP = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    'BTCUSD': 'BTC',
    'ETHUSD': 'ETH',
    # Add more mappings as needed
}


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbols to a standard format.
    
    Args:
        ticker: Original ticker symbol
        
    Returns:
        Normalized ticker symbol
    """
    if ticker in TICKER_NORMALIZATION_MAP:
        return TICKER_NORMALIZATION_MAP[ticker]
    return ticker


def load_prices_csv(prices_path: str = 'prices.csv') -> pd.DataFrame:
    """
    Load and parse prices.csv file.
    
    Args:
        prices_path: Path to prices.csv file
        
    Returns:
        DataFrame with date index and ticker columns
    """
    if not os.path.exists(prices_path):
        return pd.DataFrame()
    
    try:
        # Read prices.csv - expected format: date, ticker, close
        df = pd.read_csv(prices_path, parse_dates=['date'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to wide format: date index, ticker columns
        prices_wide = df.pivot_table(
            index='date',
            columns='ticker',
            values='close',
            aggfunc='first'  # Take first value if duplicates
        )
        
        # Sort by date
        prices_wide = prices_wide.sort_index()
        
        # Normalize ticker names
        normalized_columns = {col: normalize_ticker(col) for col in prices_wide.columns}
        prices_wide = prices_wide.rename(columns=normalized_columns)
        
        # If there are duplicate columns after normalization, combine them
        if prices_wide.columns.duplicated().any():
            prices_wide = prices_wide.groupby(level=0, axis=1).first()
        
        return prices_wide
        
    except Exception as e:
        print(f"Error loading prices.csv: {e}")
        return pd.DataFrame()


def get_wave_tickers(wave_id: str) -> List[str]:
    """
    Get list of tickers for a wave.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        List of ticker symbols
    """
    if wave_id not in WAVE_WEIGHTS:
        return []
    
    holdings = WAVE_WEIGHTS[wave_id]
    tickers = [normalize_ticker(h.ticker) for h in holdings]
    return tickers


def compute_wave_coverage_diagnostics(
    wave_id: str,
    prices_df: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Compute coverage diagnostics for a wave.
    
    Uses compute_data_ready_status from analytics_pipeline to get accurate diagnostics.
    
    Args:
        wave_id: Wave identifier
        prices_df: Optional DataFrame with date index and ticker columns (unused, for compatibility)
        
    Returns:
        Dictionary with diagnostic information:
        {
            'wave_id': str,
            'display_name': str,
            'coverage_pct': float,
            'history_days': int,
            'stale_days_max': int,
            'missing_tickers': list[str],
            'stale_tickers': list[str],
            'available_tickers': list[str]
        }
    """
    # Use the analytics_pipeline function for consistency
    diagnostics = compute_data_ready_status(wave_id)
    
    # Extract relevant fields
    result = {
        'wave_id': diagnostics['wave_id'],
        'display_name': diagnostics['display_name'],
        'coverage_pct': diagnostics.get('coverage_pct', 0.0),
        'history_days': diagnostics.get('history_days', 0),
        'stale_days_max': diagnostics.get('stale_days_max', 0),
        'missing_tickers': diagnostics.get('missing_tickers', []),
        'stale_tickers': diagnostics.get('stale_tickers', []),
        'available_tickers': []
    }
    
    # Calculate available tickers
    expected_tickers = get_wave_tickers(wave_id)
    missing_set = set(result['missing_tickers'])
    result['available_tickers'] = sorted([t for t in expected_tickers if t not in missing_set])
    
    return result


def generate_data_coverage_summary(
    output_path: str = 'data_coverage_summary.csv',
    prices_path: str = 'prices.csv'
) -> pd.DataFrame:
    """
    Generate data coverage summary CSV for all waves.
    
    Creates a diagnostic artifact with columns:
    - wave_id
    - display_name
    - coverage_pct
    - history_days
    - stale_days_max
    - missing_tickers
    - stale_tickers
    
    Args:
        output_path: Path to save the summary CSV
        prices_path: Path to prices.csv file (unused, for compatibility)
        
    Returns:
        DataFrame with coverage summary
    """
    # Get all waves
    wave_ids = get_all_wave_ids()
    
    # Compute diagnostics for each wave
    records = []
    for wave_id in wave_ids:
        diagnostics = compute_wave_coverage_diagnostics(wave_id)
        
        records.append({
            'wave_id': diagnostics['wave_id'],
            'display_name': diagnostics['display_name'],
            'coverage_pct': diagnostics['coverage_pct'],
            'history_days': diagnostics['history_days'],
            'stale_days_max': diagnostics['stale_days_max'],
            'missing_tickers': ', '.join(diagnostics['missing_tickers']),
            'stale_tickers': ', '.join(diagnostics['stale_tickers'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by coverage_pct (descending) then by wave_id
    df = df.sort_values(['coverage_pct', 'wave_id'], ascending=[False, True])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data coverage summary saved to {output_path}")
    
    return df


def merge_wave_data_from_prices(
    wave_id: str,
    prices_path: str = 'prices.csv'
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Merge data from prices.csv for a specific wave.
    
    Args:
        wave_id: Wave identifier
        prices_path: Path to prices.csv file
        
    Returns:
        Tuple of (prices_df, diagnostics_dict)
        - prices_df: DataFrame with date index and ticker columns for this wave
        - diagnostics_dict: Coverage diagnostics
    """
    # Load all prices
    all_prices = load_prices_csv(prices_path)
    
    # Get tickers for this wave
    wave_tickers = get_wave_tickers(wave_id)
    
    if not wave_tickers:
        return pd.DataFrame(), {
            'wave_id': wave_id,
            'error': 'No tickers defined for wave'
        }
    
    # Filter to only this wave's tickers
    available_tickers = [t for t in wave_tickers if t in all_prices.columns]
    
    if not available_tickers:
        diagnostics = compute_wave_coverage_diagnostics(wave_id, all_prices)
        return pd.DataFrame(), diagnostics
    
    # Get prices for this wave
    wave_prices = all_prices[available_tickers].copy()
    
    # Compute diagnostics
    diagnostics = compute_wave_coverage_diagnostics(wave_id, all_prices)
    
    return wave_prices, diagnostics


if __name__ == '__main__':
    """
    Generate data coverage summary when run as a script.
    """
    print("=" * 80)
    print("WAVE DATA COVERAGE DIAGNOSTICS")
    print("=" * 80)
    
    # Generate summary
    summary_df = generate_data_coverage_summary()
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Total waves: {len(summary_df)}")
    
    # Count waves meeting analytics thresholds
    analytics_ready = summary_df[
        (summary_df['coverage_pct'] >= MIN_COVERAGE_FOR_ANALYTICS * 100) &
        (summary_df['history_days'] >= MIN_DAYS_FOR_ANALYTICS)
    ]
    print(f"Analytics-ready waves: {len(analytics_ready)} ({len(analytics_ready)/len(summary_df)*100:.1f}%)")
    
    # Count by coverage bands
    high_coverage = len(summary_df[summary_df['coverage_pct'] >= 90])
    medium_coverage = len(summary_df[(summary_df['coverage_pct'] >= 70) & (summary_df['coverage_pct'] < 90)])
    low_coverage = len(summary_df[summary_df['coverage_pct'] < 70])
    
    print(f"\nCoverage distribution:")
    print(f"  High (≥90%): {high_coverage} waves")
    print(f"  Medium (70-90%): {medium_coverage} waves")
    print(f"  Low (<70%): {low_coverage} waves")
    
    # Show waves with stale data
    stale = summary_df[summary_df['stale_days_max'] > 0]
    if not stale.empty:
        print(f"\nWaves with stale data: {len(stale)}")
        for _, row in stale.head(5).iterrows():
            print(f"  - {row['wave_id']}: {row['stale_days_max']} days old")
    
    print("\n" + "=" * 80)
