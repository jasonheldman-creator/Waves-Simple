#!/usr/bin/env python3
"""
Demonstration of snapshot generation workflow.

This script demonstrates the complete snapshot generation process
and validates the output structure. In production, this would fetch
live market data, but for demonstration we show the structure.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
from datetime import datetime
from analytics_truth import (
    load_weights,
    expected_waves,
    _convert_wave_name_to_id,
)


def demo_snapshot_structure():
    """
    Demonstrate the snapshot generation structure.
    
    In production with network access, this would call:
        generate_live_snapshot_csv()
    
    For demonstration, we show what the output would look like.
    """
    print("\n" + "=" * 80)
    print("SNAPSHOT GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Load weights
    print("\n[Step 1] Loading wave_weights.csv...")
    weights_df = load_weights('wave_weights.csv')
    print(f"✓ Loaded {len(weights_df)} weight entries")
    print(f"✓ Found {weights_df['wave'].nunique()} unique waves")
    print(f"✓ Found {weights_df['ticker'].nunique()} unique tickers")
    
    # Step 2: Get expected waves
    print("\n[Step 2] Determining expected waves...")
    waves = expected_waves(weights_df)
    print(f"✓ Expected exactly {len(waves)} waves")
    
    # Step 3: Build snapshot structure (simulated)
    print("\n[Step 3] Building snapshot structure...")
    print("Note: In production, this would fetch live market data")
    print("      from yfinance (equities) and CoinGecko (crypto)")
    
    rows = []
    current_time = datetime.now()
    current_utc = current_time.isoformat()
    
    for i, wave_name in enumerate(waves, 1):
        wave_id = _convert_wave_name_to_id(wave_name)
        
        # Get tickers for this wave
        wave_tickers = weights_df[weights_df['wave'] == wave_name]['ticker'].tolist()
        
        # Simulate: In production, returns would be computed from live data
        # For demo, we show the structure with placeholder values
        row = {
            'wave_id': wave_id,
            'Wave': wave_name,
            'Return_1D': np.nan,  # Would be computed from live prices
            'Return_30D': np.nan,
            'Return_60D': np.nan,
            'Return_365D': np.nan,
            'status': 'SIMULATED',  # In production: 'OK' or 'NO DATA'
            'coverage_pct': 100.0,  # In production: % of successful tickers
            'missing_tickers': '',  # In production: comma-separated failed tickers
            'tickers_ok': len(wave_tickers),  # In production: count of successful
            'tickers_total': len(wave_tickers),
            'asof_utc': current_utc
        }
        
        rows.append(row)
        
        if i <= 3:  # Show first 3 waves as examples
            print(f"\n  Wave {i}: {wave_name}")
            print(f"    - wave_id: {wave_id}")
            print(f"    - tickers: {len(wave_tickers)} ({', '.join(wave_tickers[:3])}...)")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    print(f"\n✓ Built snapshot DataFrame with {len(df)} rows")
    
    # Step 4: Validate structure
    print("\n[Step 4] Validating snapshot structure...")
    
    # Check row count
    assert len(df) == 28, f"Expected 28 rows, got {len(df)}"
    print("✓ Exactly 28 rows")
    
    # Check unique wave_ids
    assert df['wave_id'].nunique() == 28, f"Expected 28 unique wave_ids"
    print("✓ All 28 wave_ids are unique")
    
    # Check required columns
    required_cols = [
        'wave_id', 'Wave', 'Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
        'status', 'coverage_pct', 'missing_tickers', 'tickers_ok', 'tickers_total',
        'asof_utc'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"✓ All {len(required_cols)} required columns present")
    
    # Step 5: Show summary
    print("\n[Step 5] Summary Statistics...")
    print(f"  Total Waves: {len(df)}")
    print(f"  Unique wave_ids: {df['wave_id'].nunique()}")
    print(f"  Total tickers across all waves: {df['tickers_total'].sum()}")
    print(f"  Average tickers per wave: {df['tickers_total'].mean():.1f}")
    
    # Show sample rows
    print("\n[Step 6] Sample Output (first 5 waves)...")
    sample_cols = ['wave_id', 'Wave', 'status', 'tickers_ok', 'tickers_total', 'coverage_pct']
    print(df[sample_cols].head().to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nIn production with network access:")
    print("1. Call: generate_live_snapshot_csv()")
    print("2. This will fetch live market data for all 120 tickers")
    print("3. Compute weighted returns for each wave (1D, 30D, 60D, 365D)")
    print("4. Write exactly 28 rows to data/live_snapshot.csv")
    print("5. Each wave will have status='OK' (with data) or 'NO DATA' (all tickers failed)")
    print("\n" + "=" * 80)
    
    return df


def show_expected_api_calls():
    """Show what API calls would be made in production"""
    print("\n" + "=" * 80)
    print("EXPECTED API CALLS IN PRODUCTION")
    print("=" * 80)
    
    weights_df = load_weights('wave_weights.csv')
    all_tickers = sorted(weights_df['ticker'].unique())
    
    equity_tickers = [t for t in all_tickers if not t.endswith('-USD')]
    crypto_tickers = [t for t in all_tickers if t.endswith('-USD')]
    
    print(f"\nTotal unique tickers: {len(all_tickers)}")
    print(f"  - Equity tickers (yfinance): {len(equity_tickers)}")
    print(f"  - Crypto tickers (CoinGecko): {len(crypto_tickers)}")
    
    print(f"\nSample equity tickers (first 10):")
    for ticker in equity_tickers[:10]:
        print(f"  - fetch_prices_equity_yf('{ticker}', days=400)")
    
    print(f"\nSample crypto tickers (first 10):")
    for ticker in crypto_tickers[:10]:
        print(f"  - fetch_prices_crypto_coingecko('{ticker}', days=400)")
    
    print("\nEach API call will:")
    print("  1. Fetch ~400 days of daily close prices")
    print("  2. Return a pandas Series indexed by date")
    print("  3. Raise exception on failure (tracked as missing ticker)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstration
    df = demo_snapshot_structure()
    
    # Show API call structure
    show_expected_api_calls()
    
    print("\nDemonstration completed successfully! ✓")
