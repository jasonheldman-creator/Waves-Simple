#!/usr/bin/env python3
"""
Extract Cache Statistics

This utility script extracts and displays statistics from the price cache file.
Used by GitHub Actions workflow to verify cache updates and display metrics.

Usage:
    python extract_cache_stats.py [--cache-path PATH] [--format {text|json}]

Output:
    Displays cache dimensions, date range, data age, and other metrics
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)


def extract_cache_stats(cache_path: str) -> Dict[str, Any]:
    """
    Extract statistics from the price cache parquet file.
    
    Args:
        cache_path: Path to the prices_cache.parquet file
        
    Returns:
        Dictionary containing cache statistics
    """
    if not os.path.exists(cache_path):
        return {
            "exists": False,
            "error": f"Cache file not found: {cache_path}"
        }
    
    try:
        # Read parquet file
        df = pd.read_parquet(cache_path)
        
        # Calculate statistics
        file_size_bytes = os.path.getsize(cache_path)
        file_size_kb = file_size_bytes / 1024
        file_size_mb = file_size_kb / 1024
        
        # Get date range
        min_date = df.index.min()
        max_date = df.index.max()
        
        # Calculate data age
        data_age_days = (datetime.now() - max_date.to_pydatetime()).days
        
        # Count tickers and trading days
        num_tickers = len(df.columns)
        num_trading_days = len(df)
        
        # Calculate data completeness
        total_cells = num_tickers * num_trading_days
        non_null_cells = df.notna().sum().sum()
        completeness_pct = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Get ticker list
        tickers = sorted(df.columns.tolist())
        
        stats = {
            "exists": True,
            "path": cache_path,
            "file_size": {
                "bytes": int(file_size_bytes),
                "kb": round(file_size_kb, 2),
                "mb": round(file_size_mb, 2)
            },
            "dimensions": {
                "rows": int(num_trading_days),
                "columns": int(num_tickers),
                "description": f"{num_trading_days} rows × {num_tickers} columns"
            },
            "date_range": {
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "description": f"{min_date.date()} to {max_date.date()}"
            },
            "last_price_date": str(max_date.date()),
            "data_age": {
                "days": int(data_age_days),
                "description": f"{data_age_days} day{'s' if data_age_days != 1 else ''}"
            },
            "completeness": {
                "total_cells": int(total_cells),
                "non_null_cells": int(non_null_cells),
                "percentage": round(completeness_pct, 2)
            },
            "tickers": tickers,
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        return {
            "exists": True,
            "error": f"Error reading cache file: {str(e)}"
        }


def format_text_output(stats: Dict[str, Any]) -> str:
    """
    Format statistics as human-readable text.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Formatted text output
    """
    if not stats.get("exists"):
        return f"Error: {stats.get('error', 'Unknown error')}"
    
    if "error" in stats:
        return f"Error: {stats['error']}"
    
    lines = [
        "=" * 70,
        "PRICE CACHE STATISTICS",
        "=" * 70,
        "",
        f"Output Path:      {stats['path']}",
        f"File Size:        {stats['file_size']['kb']:.2f} KB ({stats['file_size']['mb']:.2f} MB)",
        "",
        f"Dimensions:       {stats['dimensions']['description']}",
        f"  - Trading Days: {stats['dimensions']['rows']}",
        f"  - Tickers:      {stats['dimensions']['columns']}",
        "",
        f"Date Range:       {stats['date_range']['description']}",
        f"  - Min Date:     {stats['date_range']['min_date']}",
        f"  - Max Date:     {stats['date_range']['max_date']}",
        "",
        f"Last Price Date:  {stats['last_price_date']}",
        f"Data Age:         {stats['data_age']['description']}",
        "",
        f"Data Completeness: {stats['completeness']['percentage']:.2f}%",
        f"  - Total Cells:   {stats['completeness']['total_cells']:,}",
        f"  - Non-Null:      {stats['completeness']['non_null_cells']:,}",
        "",
        f"Sample Tickers ({min(10, len(stats['tickers']))}/{len(stats['tickers'])}):",
    ]
    
    # Add first 10 tickers
    for ticker in stats['tickers'][:10]:
        lines.append(f"  - {ticker}")
    
    if len(stats['tickers']) > 10:
        lines.append(f"  ... and {len(stats['tickers']) - 10} more")
    
    lines.extend([
        "",
        "=" * 70
    ])
    
    return "\n".join(lines)


def format_json_output(stats: Dict[str, Any]) -> str:
    """
    Format statistics as JSON.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        JSON formatted output
    """
    return json.dumps(stats, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and display price cache statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display stats in text format (default)
  python extract_cache_stats.py
  
  # Display stats in JSON format
  python extract_cache_stats.py --format json
  
  # Use custom cache path
  python extract_cache_stats.py --cache-path /path/to/prices_cache.parquet
        """
    )
    
    parser.add_argument(
        "--cache-path",
        default="data/cache/prices_cache.parquet",
        help="Path to the prices_cache.parquet file (default: data/cache/prices_cache.parquet)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    # Extract statistics
    stats = extract_cache_stats(args.cache_path)
    
    # Format and print output
    if args.format == "json":
        print(format_json_output(stats))
    else:
        print(format_text_output(stats))
    
    # Exit with error code if cache doesn't exist or has errors
    if not stats.get("exists") or "error" in stats:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
