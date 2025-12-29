#!/usr/bin/env python3
"""
Build Complete Price Data Cache

This script:
1. Extracts all tickers from Wave definitions (wave_weights.csv)
2. Extracts all benchmark tickers (wave_config.csv)
3. Adds safe asset tickers (treasuries, money market ETFs)
4. Downloads 1+ year of historical price data for each ticker
5. Creates consolidated prices.csv file
6. Generates comprehensive diagnostics

Usage:
    python build_complete_price_cache.py [--days DAYS] [--output OUTPUT]
"""

import os
import sys
import csv
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import yfinance as yf


# Configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOOKBACK_DAYS = 400  # ~13 months of trading data
BATCH_SIZE = 10  # Download in small batches to avoid rate limits
BATCH_DELAY = 1.5  # Delay between batches in seconds
MAX_RETRIES = 3  # Retry failed downloads


def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol (trim, uppercase)."""
    return ticker.strip().upper()


def extract_wave_holdings_tickers() -> Set[str]:
    """Extract all tickers from wave_weights.csv."""
    tickers = set()
    weights_path = os.path.join(REPO_ROOT, "wave_weights.csv")
    
    if not os.path.exists(weights_path):
        log_message(f"Warning: wave_weights.csv not found at {weights_path}", "WARN")
        return tickers
    
    try:
        with open(weights_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'ticker' in row and row['ticker']:
                    ticker = normalize_ticker(row['ticker'])
                    tickers.add(ticker)
        
        log_message(f"Extracted {len(tickers)} tickers from wave_weights.csv")
    except Exception as e:
        log_message(f"Error reading wave_weights.csv: {e}", "ERROR")
    
    return tickers


def extract_benchmark_tickers() -> Set[str]:
    """Extract all benchmark tickers from wave_config.csv."""
    tickers = set()
    config_path = os.path.join(REPO_ROOT, "wave_config.csv")
    
    if not os.path.exists(config_path):
        log_message(f"Warning: wave_config.csv not found at {config_path}", "WARN")
        return tickers
    
    try:
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Benchmark' in row and row['Benchmark']:
                    ticker = normalize_ticker(row['Benchmark'])
                    tickers.add(ticker)
        
        log_message(f"Extracted {len(tickers)} benchmark tickers from wave_config.csv")
    except Exception as e:
        log_message(f"Error reading wave_config.csv: {e}", "ERROR")
    
    return tickers


def get_safe_asset_tickers() -> Set[str]:
    """Get safe asset tickers (treasuries, money market, bonds)."""
    safe_assets = {
        # Treasury ETFs
        'BIL', 'SHY', 'IEF', 'TLT', 'SGOV',
        # Bond ETFs
        'AGG', 'BND', 'LQD', 'HYG',
        # Municipal bonds
        'MUB', 'SUB', 'SHM',
        # Market indices (used for analytics)
        'SPY', 'QQQ', 'IWM', 'IWV', 'DIA',
        # Volatility index
        '^VIX',
        # Commodities
        'GLD', 'IAU',
        # Stablecoins (for crypto waves)
        'USDC-USD', 'USDT-USD', 'DAI-USD',
    }
    
    log_message(f"Added {len(safe_assets)} safe asset tickers")
    return safe_assets


def build_complete_ticker_list() -> List[str]:
    """Build complete deduplicated ticker list from all sources."""
    all_tickers = set()
    
    # Step 1: Wave holdings
    holdings_tickers = extract_wave_holdings_tickers()
    all_tickers.update(holdings_tickers)
    
    # Step 2: Benchmarks
    benchmark_tickers = extract_benchmark_tickers()
    all_tickers.update(benchmark_tickers)
    
    # Step 3: Safe assets
    safe_tickers = get_safe_asset_tickers()
    all_tickers.update(safe_tickers)
    
    # Sort for consistent ordering
    ticker_list = sorted(list(all_tickers))
    
    log_message(f"Total unique tickers: {len(ticker_list)}")
    
    return ticker_list


def save_ticker_reference_list(tickers: List[str], output_path: str):
    """Save ticker reference list for visibility."""
    try:
        with open(output_path, 'w') as f:
            f.write("ticker,source,date_generated\n")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for ticker in tickers:
                f.write(f"{ticker},complete_build,{timestamp}\n")
        
        log_message(f"Saved ticker reference list to {output_path}")
    except Exception as e:
        log_message(f"Error saving ticker reference list: {e}", "ERROR")


def download_ticker_prices(
    ticker: str, 
    days: int,
    max_retries: int = MAX_RETRIES
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical prices for a single ticker.
    
    Returns:
        (DataFrame, error_message) tuple
        DataFrame has columns: date, ticker, close
        If successful, error_message is None
    """
    for attempt in range(max_retries):
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Download data
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True  # Use adjusted close
            )
            
            if hist.empty or len(hist) < 5:
                return None, f"Insufficient data: {len(hist)} rows"
            
            # Format data
            df = pd.DataFrame({
                'date': hist.index.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': hist['Close'].round(2)
            })
            
            return df, None
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                return None, error_msg
    
    return None, "Max retries exceeded"


def download_all_prices(
    tickers: List[str],
    days: int,
    batch_size: int = BATCH_SIZE
) -> Tuple[List[pd.DataFrame], Dict[str, str]]:
    """
    Download prices for all tickers in batches.
    
    Returns:
        (list of dataframes, failure dictionary)
    """
    successful_dfs = []
    failures = {}
    
    total = len(tickers)
    log_message(f"Starting download of {total} tickers (batch size: {batch_size})")
    
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        log_message(f"Processing batch {batch_num}/{total_batches}: {len(batch)} tickers")
        
        for ticker in batch:
            df, error = download_ticker_prices(ticker, days)
            
            if df is not None:
                successful_dfs.append(df)
                log_message(f"  ✓ {ticker}: {len(df)} rows")
            else:
                failures[ticker] = error
                log_message(f"  ✗ {ticker}: {error}", "WARN")
        
        # Delay between batches
        if i + batch_size < total:
            time.sleep(BATCH_DELAY)
    
    log_message(f"Download complete: {len(successful_dfs)} successful, {len(failures)} failed")
    
    return successful_dfs, failures


def build_consolidated_prices_csv(
    dataframes: List[pd.DataFrame],
    output_path: str
) -> int:
    """
    Build consolidated prices.csv from list of dataframes.
    
    Returns:
        Number of rows written
    """
    if not dataframes:
        log_message("No data to write", "ERROR")
        return 0
    
    try:
        # Concatenate all dataframes
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Sort by ticker then date
        combined = combined.sort_values(['ticker', 'date'])
        
        # Remove duplicates
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
        after_dedup = len(combined)
        
        if before_dedup > after_dedup:
            log_message(f"Removed {before_dedup - after_dedup} duplicate rows")
        
        # Write to CSV
        combined.to_csv(output_path, index=False)
        
        log_message(f"Wrote {len(combined)} rows to {output_path}")
        
        return len(combined)
        
    except Exception as e:
        log_message(f"Error building consolidated CSV: {e}", "ERROR")
        return 0


def generate_diagnostics(
    successful_tickers: Set[str],
    failures: Dict[str, str],
    total_tickers: int,
    prices_df: Optional[pd.DataFrame] = None
) -> Dict:
    """Generate comprehensive diagnostics report."""
    
    diagnostics = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_tickers_requested': total_tickers,
        'successful_downloads': len(successful_tickers),
        'failed_downloads': len(failures),
        'success_rate': f"{(len(successful_tickers) / total_tickers * 100):.1f}%",
        'failures': failures
    }
    
    # Calculate per-ticker statistics if we have the price data
    if prices_df is not None:
        ticker_stats = {}
        for ticker in successful_tickers:
            ticker_data = prices_df[prices_df['ticker'] == ticker]
            if not ticker_data.empty:
                ticker_stats[ticker] = {
                    'rows': len(ticker_data),
                    'date_range': f"{ticker_data['date'].min()} to {ticker_data['date'].max()}"
                }
        
        diagnostics['ticker_statistics'] = ticker_stats
    
    return diagnostics


def save_diagnostics(diagnostics: Dict, output_path: str):
    """Save diagnostics to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
        log_message(f"Saved diagnostics to {output_path}")
    except Exception as e:
        log_message(f"Error saving diagnostics: {e}", "ERROR")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Build complete price data cache for all Waves'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f'Days of historical data to fetch (default: {DEFAULT_LOOKBACK_DAYS})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(REPO_ROOT, 'prices.csv'),
        help='Output path for prices.csv'
    )
    
    args = parser.parse_args()
    
    log_message("=" * 60)
    log_message("BUILD COMPLETE PRICE DATA CACHE")
    log_message("=" * 60)
    
    # Step 1: Build ticker list
    log_message("\nStep 1: Building complete ticker list...")
    tickers = build_complete_ticker_list()
    
    # Save ticker reference list
    ticker_ref_path = os.path.join(REPO_ROOT, 'ticker_reference_list.csv')
    save_ticker_reference_list(tickers, ticker_ref_path)
    
    # Step 2: Download prices
    log_message(f"\nStep 2: Downloading {args.days} days of price data...")
    dataframes, failures = download_all_prices(tickers, args.days)
    
    # Step 3: Build consolidated CSV
    log_message("\nStep 3: Building consolidated prices.csv...")
    rows_written = build_consolidated_prices_csv(dataframes, args.output)
    
    # Step 4: Generate diagnostics
    log_message("\nStep 4: Generating diagnostics...")
    successful_tickers = set(df['ticker'].iloc[0] for df in dataframes)
    
    # Load the consolidated CSV for detailed stats
    prices_df = None
    if os.path.exists(args.output):
        try:
            prices_df = pd.read_csv(args.output)
        except Exception as e:
            log_message(f"Warning: Could not load prices.csv for diagnostics: {e}", "WARN")
    
    diagnostics = generate_diagnostics(
        successful_tickers,
        failures,
        len(tickers),
        prices_df
    )
    
    # Save diagnostics
    diagnostics_path = os.path.join(REPO_ROOT, 'price_cache_diagnostics.json')
    save_diagnostics(diagnostics, diagnostics_path)
    
    # Print summary
    log_message("\n" + "=" * 60)
    log_message("SUMMARY")
    log_message("=" * 60)
    log_message(f"Total tickers processed: {len(tickers)}")
    log_message(f"Successful downloads: {len(successful_tickers)}")
    log_message(f"Failed downloads: {len(failures)}")
    log_message(f"Success rate: {diagnostics['success_rate']}")
    log_message(f"Total price rows: {rows_written}")
    log_message(f"\nOutput files:")
    log_message(f"  - {args.output}")
    log_message(f"  - {ticker_ref_path}")
    log_message(f"  - {diagnostics_path}")
    
    if failures:
        log_message(f"\nFailed tickers ({len(failures)}):")
        for ticker, error in sorted(failures.items()):
            log_message(f"  {ticker}: {error}")
    
    log_message("=" * 60)
    
    return 0 if len(failures) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
