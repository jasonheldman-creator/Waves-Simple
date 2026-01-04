#!/usr/bin/env python3
"""
Ticker Master File Generator - Safe Mode

Creates a canonical ticker master file (ticker_master_clean.csv)
by extracting and normalizing tickers from wave definitions.

This version does not require network validation since these tickers
are already in use by the waves and are assumed to be valid.
"""

import os
import sys
import csv
from datetime import datetime
from typing import Dict, List, Set, Tuple

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


def normalize_and_validate_tickers(tickers: Set[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalize all tickers and check for duplicates.
    
    Args:
        tickers: Set of raw ticker symbols
        
    Returns:
        Tuple of (unique_normalized_tickers, normalization_map)
    """
    normalization_map = {}
    normalized_set = set()
    duplicates = {}
    
    print("\n🔄 Normalizing tickers...")
    print("=" * 60)
    
    for ticker in sorted(tickers):
        normalized = _normalize_ticker(ticker)
        normalization_map[ticker] = normalized
        
        if ticker != normalized:
            print(f"  🔄 {ticker} → {normalized}")
        
        # Track for duplicates
        if normalized in normalized_set:
            if normalized not in duplicates:
                duplicates[normalized] = []
            duplicates[normalized].append(ticker)
        else:
            normalized_set.add(normalized)
    
    print("=" * 60)
    
    if duplicates:
        print("\n⚠️  Found duplicate normalized tickers:")
        for norm_ticker, orig_tickers in duplicates.items():
            print(f"  {norm_ticker}: {orig_tickers}")
    
    unique_tickers = sorted(normalized_set)
    print(f"\n✅ Normalized to {len(unique_tickers)} unique tickers")
    
    return unique_tickers, normalization_map


def write_ticker_master_file(valid_tickers: List[str], normalization_map: Dict[str, str], output_path: str) -> None:
    """
    Write the canonical ticker master file.
    
    Args:
        valid_tickers: List of normalized ticker symbols
        normalization_map: Mapping from original to normalized tickers
        output_path: Path to output CSV file
    """
    # Create reverse map to find original tickers
    normalized_to_original = {}
    for orig, norm in normalization_map.items():
        if norm not in normalized_to_original:
            normalized_to_original[norm] = []
        normalized_to_original[norm].append(orig)
    
    # Ensure tickers are uppercase, trimmed, and sorted
    clean_tickers = sorted(set(ticker.strip().upper() for ticker in valid_tickers))
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'original_forms', 'created_date', 'source'])
        
        creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for ticker in clean_tickers:
            original_forms = normalized_to_original.get(ticker, [ticker])
            original_str = ';'.join(sorted(set(original_forms)))
            writer.writerow([ticker, original_str, creation_date, 'WAVE_WEIGHTS'])
    
    print(f"\n📝 Written {len(clean_tickers)} validated tickers to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("TICKER MASTER FILE GENERATOR (Safe Mode)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Extract tickers from wave definitions
    print("Step 1: Extracting tickers from wave definitions...")
    raw_tickers = extract_all_tickers_from_waves()
    
    # Step 2: Normalize tickers and check for duplicates
    print("\nStep 2: Normalizing and deduplicating tickers...")
    unique_tickers, normalization_map = normalize_and_validate_tickers(raw_tickers)
    
    # Step 3: Write ticker master file
    print("\nStep 3: Writing ticker master file...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    master_file_path = os.path.join(base_dir, 'ticker_master_clean.csv')
    write_ticker_master_file(unique_tickers, normalization_map, master_file_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✅ Total unique tickers: {len(unique_tickers)}")
    print(f"📝 Master file: ticker_master_clean.csv")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
