#!/usr/bin/env python3
"""
Ticker Master File Generator

Creates a verified and canonical ticker master file (ticker_master_clean.csv)
by extracting tickers from wave definitions, normalizing them, and validating
them against the yfinance data provider.

This serves as the source of truth for all ticker references in the system.
"""

import os
import sys
import time
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import yfinance as yf

# Import wave definitions and ticker normalization
from waves_engine import WAVE_WEIGHTS, _normalize_ticker


def extract_all_tickers_from_waves() -> Set[str]:
    """
    Extract all unique tickers from wave definitions.
    
    Returns:
        Set of raw ticker symbols (before normalization)
    """
    all_tickers = set()
    
    for wave_name, holdings in WAVE_WEIGHTS.items():
        for holding in holdings:
            all_tickers.add(holding.ticker)
    
    print(f"📊 Extracted {len(all_tickers)} unique tickers from {len(WAVE_WEIGHTS)} waves")
    return all_tickers


def normalize_ticker_list(tickers: Set[str]) -> Dict[str, str]:
    """
    Normalize all tickers using the system's normalization rules.
    
    Args:
        tickers: Set of raw ticker symbols
        
    Returns:
        Dict mapping original ticker to normalized ticker
    """
    ticker_map = {}
    
    for ticker in tickers:
        normalized = _normalize_ticker(ticker)
        ticker_map[ticker] = normalized
        
        if ticker != normalized:
            print(f"  🔄 Normalized: {ticker} → {normalized}")
    
    return ticker_map


def validate_ticker(ticker: str, retries: int = 3) -> Tuple[bool, str]:
    """
    Validate a ticker against yfinance data provider.
    
    Args:
        ticker: Ticker symbol to validate
        retries: Number of retry attempts
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    for attempt in range(retries):
        try:
            # Try to fetch recent data
            stock = yf.Ticker(ticker)
            
            # Try to get basic info first (fast check)
            info = stock.info
            if info and len(info) > 1:  # yfinance returns {symbol: ticker} even for invalid tickers
                return True, ""
            
            # Fallback: Try to get historical data
            hist = stock.history(period="5d")
            if not hist.empty and len(hist) >= 1:
                return True, ""
            
            # If we get here, ticker might be invalid
            return False, "No data available"
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_msg or "rate limit" in error_msg.lower():
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"  ⏳ Rate limit hit for {ticker}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, "Rate limit exceeded"
            
            # Other errors
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False, f"Validation error: {error_msg}"
    
    return False, "Validation failed after retries"


def validate_all_tickers(normalized_tickers: Dict[str, str], batch_delay: float = 0.5) -> Tuple[List[str], Dict[str, str]]:
    """
    Validate all normalized tickers against the data provider.
    
    Args:
        normalized_tickers: Dict mapping original to normalized tickers
        batch_delay: Delay between batches to avoid rate limiting
        
    Returns:
        Tuple of (valid_tickers, failed_tickers_with_reasons)
    """
    valid_tickers = []
    failed_tickers = {}
    
    # Get unique normalized tickers (some originals might normalize to the same ticker)
    unique_normalized = sorted(set(normalized_tickers.values()))
    
    print(f"\n🔍 Validating {len(unique_normalized)} unique normalized tickers...")
    print("=" * 60)
    
    batch_size = 10
    for i, ticker in enumerate(unique_normalized):
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(unique_normalized):
            print(f"Progress: {i + 1}/{len(unique_normalized)} tickers validated")
        
        # Add delay between batches
        if i > 0 and i % batch_size == 0:
            time.sleep(batch_delay)
        
        # Validate ticker
        is_valid, error_msg = validate_ticker(ticker)
        
        if is_valid:
            valid_tickers.append(ticker)
            print(f"  ✅ {ticker}")
        else:
            failed_tickers[ticker] = error_msg
            print(f"  ❌ {ticker}: {error_msg}")
    
    print("=" * 60)
    print(f"\n✅ Valid: {len(valid_tickers)}/{len(unique_normalized)}")
    print(f"❌ Failed: {len(failed_tickers)}/{len(unique_normalized)}")
    
    return valid_tickers, failed_tickers


def write_ticker_master_file(valid_tickers: List[str], output_path: str) -> None:
    """
    Write the canonical ticker master file.
    
    Args:
        valid_tickers: List of validated ticker symbols
        output_path: Path to output CSV file
    """
    # Ensure tickers are uppercase, trimmed, and sorted
    clean_tickers = sorted(set(ticker.strip().upper() for ticker in valid_tickers))
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'validated_date'])
        
        validation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for ticker in clean_tickers:
            writer.writerow([ticker, validation_date])
    
    print(f"\n📝 Written {len(clean_tickers)} validated tickers to: {output_path}")


def write_failure_report(failed_tickers: Dict[str, str], output_path: str) -> None:
    """
    Write a report of failed ticker validations.
    
    Args:
        failed_tickers: Dict mapping failed tickers to error messages
        output_path: Path to output report file
    """
    if not failed_tickers:
        print("ℹ️  No failed tickers to report")
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'error_message', 'report_date'])
        
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for ticker, error_msg in sorted(failed_tickers.items()):
            writer.writerow([ticker, error_msg, report_date])
    
    print(f"📋 Written failure report to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("TICKER MASTER FILE GENERATOR")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Extract tickers from wave definitions
    print("Step 1: Extracting tickers from wave definitions...")
    raw_tickers = extract_all_tickers_from_waves()
    
    # Step 2: Normalize tickers
    print("\nStep 2: Normalizing tickers...")
    normalized_map = normalize_ticker_list(raw_tickers)
    
    # Step 3: Validate all tickers
    print("\nStep 3: Validating tickers against yfinance...")
    valid_tickers, failed_tickers = validate_all_tickers(normalized_map)
    
    # Step 4: Write ticker master file
    print("\nStep 4: Writing ticker master file...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    master_file_path = os.path.join(base_dir, 'ticker_master_clean.csv')
    write_ticker_master_file(valid_tickers, master_file_path)
    
    # Step 5: Write failure report (if any failures)
    if failed_tickers:
        print("\nStep 5: Writing failure report...")
        os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
        failure_report_path = os.path.join(base_dir, 'reports', 'ticker_validation_failures.csv')
        write_failure_report(failed_tickers, failure_report_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✅ Valid tickers: {len(valid_tickers)}")
    print(f"❌ Failed tickers: {len(failed_tickers)}")
    print(f"📊 Success rate: {len(valid_tickers)/(len(valid_tickers)+len(failed_tickers))*100:.1f}%")
    print(f"📝 Master file: ticker_master_clean.csv")
    if failed_tickers:
        print(f"📋 Failure report: reports/ticker_validation_failures.csv")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Exit with appropriate code
    if failed_tickers:
        print("\n⚠️  Some tickers failed validation. Review the failure report.")
        return 1
    else:
        print("\n✅ All tickers validated successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
