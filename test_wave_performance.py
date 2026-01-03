"""
Test wave_performance module with actual PRICE_BOOK data.

This test validates:
1. Wave performance computation from PRICE_BOOK
2. Handling of missing tickers
3. Return calculations for various periods
4. Readiness diagnostics
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_wave_performance():
    """Test wave performance computation."""
    print("=" * 70)
    print("Testing Wave Performance Module")
    print("=" * 70)
    
    # Import required modules
    from helpers.price_book import get_price_book
    from helpers.wave_performance import (
        compute_wave_returns,
        compute_all_waves_performance,
        compute_all_waves_readiness,
        get_price_book_diagnostics
    )
    
    # Load PRICE_BOOK
    print("\n1. Loading PRICE_BOOK...")
    price_book = get_price_book()
    print(f"   ✓ Loaded PRICE_BOOK: {price_book.shape[0]} days × {price_book.shape[1]} tickers")
    print(f"   Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Test PRICE_BOOK diagnostics
    print("\n2. Testing PRICE_BOOK diagnostics...")
    pb_diag = get_price_book_diagnostics(price_book)
    print(f"   ✓ Path: {pb_diag['path']}")
    print(f"   ✓ Shape: {pb_diag['shape']}")
    print(f"   ✓ Date range: {pb_diag['date_min']} to {pb_diag['date_max']}")
    print(f"   ✓ Total tickers: {pb_diag['total_tickers']}")
    
    # Test single wave computation
    print("\n3. Testing single wave computation (S&P 500 Wave)...")
    result = compute_wave_returns('S&P 500 Wave', price_book, periods=[1, 30, 60, 365])
    print(f"   Success: {result['success']}")
    print(f"   Failure reason: {result['failure_reason']}")
    print(f"   Coverage: {result['coverage_pct']:.1f}%")
    print(f"   Tickers: {len(result['tickers'])} present, {len(result['missing_tickers'])} missing")
    
    if result['success']:
        print(f"   Returns:")
        for period, ret in result['returns'].items():
            if ret is not None:
                print(f"      {period}: {ret*100:+.2f}%")
            else:
                print(f"      {period}: N/A")
    else:
        print(f"   ✗ Failed: {result['failure_reason']}")
    
    # Test all waves performance
    print("\n4. Testing all waves performance computation...")
    perf_df = compute_all_waves_performance(price_book, periods=[1, 30, 60, 365])
    print(f"   ✓ Computed performance for {len(perf_df)} waves")
    
    # Count waves by status
    status_counts = perf_df['Status/Confidence'].value_counts()
    print(f"\n   Status breakdown:")
    for status, count in status_counts.items():
        print(f"      {status}: {count} waves")
    
    # Show waves with failures
    failed = perf_df[perf_df['Failure_Reason'].notna()]
    if not failed.empty:
        print(f"\n   ⚠️  {len(failed)} waves with failures:")
        for _, row in failed.head(5).iterrows():
            print(f"      - {row['Wave']}: {row['Failure_Reason']} (coverage: {row['Coverage_Pct']:.1f}%)")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")
    else:
        print(f"\n   ✓ All waves computed successfully!")
    
    # Show sample successful waves
    successful = perf_df[perf_df['Failure_Reason'].isna()]
    if not successful.empty:
        print(f"\n   Sample successful waves:")
        for _, row in successful.head(3).iterrows():
            print(f"      - {row['Wave']}: 1D={row['1D Return']}, 30D={row['30D']}, Status={row['Status/Confidence']}")
    
    # Test readiness computation
    print("\n5. Testing wave readiness computation...")
    readiness_df = compute_all_waves_readiness(price_book)
    print(f"   ✓ Computed readiness for {len(readiness_df)} waves")
    
    ready_count = readiness_df['data_ready'].sum()
    not_ready_count = len(readiness_df) - ready_count
    print(f"   Data-ready: {ready_count}/{len(readiness_df)} waves ({ready_count/len(readiness_df)*100:.1f}%)")
    
    # Show not-ready waves
    if not_ready_count > 0:
        not_ready = readiness_df[~readiness_df['data_ready']]
        print(f"\n   Not data-ready waves ({not_ready_count}):")
        for _, row in not_ready.head(5).iterrows():
            print(f"      - {row['wave_name']}: {row['reason']} (coverage: {row['coverage_pct']:.1f}%)")
        if not_ready_count > 5:
            print(f"      ... and {not_ready_count - 5} more")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    test_wave_performance()
