#!/usr/bin/env python3
"""
Extract statistics from price cache parquet file.

This script is used by the GitHub Actions workflow to extract
metadata from the generated price cache file and output it in
a format that can be consumed by GitHub Actions.

Usage:
    python extract_cache_stats.py <cache_path> <output_file>
    
    cache_path: Path to prices_cache.parquet file
    output_file: Path to GitHub Actions output file (usually $GITHUB_OUTPUT)
"""

import os
import sys
import pandas as pd
from datetime import datetime


def extract_statistics(cache_path):
    """
    Extract statistics from price cache parquet file.
    
    Returns:
        dict with keys: last_date, first_date, num_rows, num_cols, file_size_mb, age_days
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    try:
        # Load the cache
        df = pd.read_parquet(cache_path)
        
        # Get statistics
        num_rows = len(df)
        # Number of tickers = number of columns (since index is dates, columns are tickers)
        # This assumes the canonical price book format where index=dates, columns=tickers
        num_cols = len(df.columns)
        file_size_bytes = os.path.getsize(cache_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get last price date (most recent date in index)
        if isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index.max().strftime('%Y-%m-%d')
            first_date = df.index.min().strftime('%Y-%m-%d')
        else:
            last_date = "Unknown"
            first_date = "Unknown"
        
        # Calculate data age
        if last_date != "Unknown":
            last_dt = pd.to_datetime(last_date)
            today = pd.Timestamp.now().normalize()
            age_days = (today - last_dt).days
        else:
            age_days = -1
        
        return {
            'last_date': last_date,
            'first_date': first_date,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'file_size_mb': f"{file_size_mb:.2f}",
            'age_days': age_days
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract statistics: {e}")


def write_github_output(stats, output_file):
    """Write statistics to GitHub Actions output file."""
    with open(output_file, 'a') as f:
        for key, value in stats.items():
            f.write(f"{key}={value}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: extract_cache_stats.py <cache_path> <output_file>")
        sys.exit(1)
    
    cache_path = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        stats = extract_statistics(cache_path)
        
        # Print to console for workflow logs
        print(f"Last Price Date: {stats['last_date']}")
        print(f"Date Range: {stats['first_date']} to {stats['last_date']}")
        print(f"Dimensions: {stats['num_rows']} rows × {stats['num_cols']} columns")
        print(f"Tickers: {stats['num_cols']}")
        print(f"File Size: {stats['file_size_mb']} MB")
        print(f"Data Age: {stats['age_days']} days")
        
        # Write to GitHub output
        write_github_output(stats, output_file)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
