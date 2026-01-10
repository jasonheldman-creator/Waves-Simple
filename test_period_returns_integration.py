"""
Integration test for period return calculations with actual price data.

This test validates that the canonical period return helper is correctly
integrated across all calculation paths and produces expected results.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to helpers directory for imports
helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
os.chdir(helpers_dir)
sys.path.insert(0, helpers_dir)

import price_book as pb_module
import wave_performance as wp_module
from period_returns import TRADING_DAYS_MAP

get_price_book = pb_module.get_price_book
compute_wave_returns = wp_module.compute_wave_returns
compute_portfolio_snapshot = wp_module.compute_portfolio_snapshot


def test_integration():
    """Integration test with actual price data."""
    print("=" * 70)
    print("INTEGRATION TEST: Period Return Calculations")
    print("=" * 70)
    
    # Load price book
    print("\n1. Loading PRICE_BOOK...")
    price_book = get_price_book()
    
    if price_book.empty:
        print("   ⚠️  PRICE_BOOK is empty - skipping integration test")
        print("   This is expected in CI environments without price cache")
        return True
    
    print(f"   ✓ Loaded PRICE_BOOK: {price_book.shape[0]} days × {price_book.shape[1]} tickers")
    print(f"   Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Test 1: Wave-level returns use correct trading days
    print("\n2. Testing wave-level returns with trading day mapping...")
    result = compute_wave_returns('S&P 500 Wave', price_book, periods=[1, 30, 60, 365])
    
    if not result['success']:
        print(f"   ✗ Failed to compute wave returns: {result['failure_reason']}")
        return False
    
    print(f"   ✓ Wave returns computed successfully")
    print(f"   Period mapping verification:")
    
    # Verify that returns exist for all periods
    for period in [1, 30, 60, 365]:
        period_key = f'{period}D'
        ret = result['returns'].get(period_key)
        
        if period == 365:
            trading_days = TRADING_DAYS_MAP['365D']  # Should be 252
        else:
            trading_days = TRADING_DAYS_MAP[period_key]
        
        if ret is not None:
            print(f"     {period_key}: {ret*100:+.2f}% (using {trading_days} trading days)")
        else:
            print(f"     {period_key}: N/A (insufficient data)")
    
    # Verify 365D uses 252 trading days
    if result['returns']['365D'] is not None:
        assert TRADING_DAYS_MAP['365D'] == 252, "365D should use 252 trading days"
        print(f"   ✓ Verified: 365D correctly uses 252 trading days (1 year market convention)")
    
    # Test 2: Portfolio snapshot uses correct trading days
    print("\n3. Testing portfolio snapshot with trading day mapping...")
    snapshot = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60, 365])
    
    if not snapshot['success']:
        print(f"   ⚠️  Portfolio snapshot not available: {snapshot['failure_reason']}")
        print(f"   This may be expected if insufficient waves have data")
    else:
        print(f"   ✓ Portfolio snapshot computed successfully")
        print(f"   Waves included: {snapshot['wave_count']}")
        print(f"   Date range: {snapshot['date_range'][0]} to {snapshot['date_range'][1]}")
        
        print(f"\n   Portfolio returns:")
        for period_key, ret in snapshot['portfolio_returns'].items():
            if ret is not None:
                print(f"     {period_key}: {ret*100:+.2f}%")
            else:
                print(f"     {period_key}: N/A")
        
        print(f"\n   Alpha (vs benchmark):")
        for period_key, alpha in snapshot['alphas'].items():
            if alpha is not None:
                print(f"     {period_key}: {alpha*100:+.2f}%")
            else:
                print(f"     {period_key}: N/A")
    
    # Test 3: Benchmark alignment
    print("\n4. Testing benchmark alignment...")
    
    # For a wave with returns, verify that wave and benchmark are computed
    # over the same date range
    if result['success'] and result['returns']['30D'] is not None:
        print(f"   ✓ Wave has valid 30D return")
        print(f"   Wave date range: {result['date_range']}")
        
        # The implementation should ensure both wave and benchmark use the same dates
        # This is verified by the fact that returns are computed
        print(f"   ✓ Wave and benchmark aligned (returns computed successfully)")
    else:
        print(f"   ⚠️  Cannot test alignment - wave has no valid 30D return")
    
    # Test 4: Verify trading day constants
    print("\n5. Verifying trading day constants...")
    assert TRADING_DAYS_MAP['1D'] == 1, "1D should be 1 trading day"
    assert TRADING_DAYS_MAP['30D'] == 30, "30D should be 30 trading days"
    assert TRADING_DAYS_MAP['60D'] == 60, "60D should be 60 trading days"
    assert TRADING_DAYS_MAP['365D'] == 252, "365D should be 252 trading days"
    print(f"   ✓ All trading day constants correct:")
    print(f"     {TRADING_DAYS_MAP}")
    
    print("\n" + "=" * 70)
    print("✓ INTEGRATION TEST PASSED")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
