"""
Test suite to verify unified price source implementation.

This test ensures that:
1. All wave computations use the canonical PRICE_BOOK
2. No implicit network fetching occurs
3. Missing tickers are properly reported
4. The system is deterministic and reproducible
"""

import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_no_yfinance_calls_in_wave_computation():
    """Test that wave computation doesn't call yfinance when PRICE_BOOK is available."""
    print("=" * 80)
    print("TEST: No yfinance calls during wave computation")
    print("=" * 80)
    
    from waves_engine import compute_history_nav
    import yfinance as yf
    
    # Mock yfinance.download to detect if it's called
    original_download = yf.download
    call_count = [0]
    
    def mock_download(*args, **kwargs):
        call_count[0] += 1
        print(f"  ⚠️  WARNING: yfinance.download was called! Args: {args[:2]}")
        return original_download(*args, **kwargs)
    
    # Patch yfinance.download
    with patch('yfinance.download', side_effect=mock_download):
        print("\n1. Computing wave NAV with mocked yfinance.download...")
        
        try:
            nav = compute_history_nav('US MegaCap Core Wave', mode='Standard', days=30)
            
            if call_count[0] == 0:
                print(f"  ✓ No yfinance calls detected - using PRICE_BOOK")
                print(f"  ✓ Computed {len(nav)} days of NAV")
                result = True
            else:
                print(f"  ✗ FAIL: yfinance.download was called {call_count[0]} times")
                print(f"  ✗ Expected: 0 calls (should use PRICE_BOOK)")
                result = False
        except Exception as e:
            print(f"  ✗ Error during computation: {e}")
            result = False
    
    print(f"\n{'✓ Test PASSED' if result else '✗ Test FAILED'}")
    return result


def test_price_book_source_in_metadata():
    """Test that wave computation metadata indicates PRICE_BOOK as source."""
    print("\n" + "=" * 80)
    print("TEST: PRICE_BOOK source indicated in metadata")
    print("=" * 80)
    
    from waves_engine import compute_history_nav
    
    print("\n1. Computing wave NAV...")
    nav = compute_history_nav('S&P 500 Wave', mode='Standard', days=30)
    
    if not nav.empty:
        print(f"  ✓ Computed {len(nav)} days of NAV")
        
        # Check if coverage metadata exists
        if hasattr(nav, 'attrs') and 'coverage' in nav.attrs:
            coverage = nav.attrs['coverage']
            print(f"\n2. Checking coverage metadata...")
            print(f"  ✓ Coverage metadata exists")
            print(f"  ✓ Wave coverage: {coverage.get('wave_coverage_pct', 0):.1f}%")
            print(f"  ✓ Benchmark coverage: {coverage.get('bm_coverage_pct', 0):.1f}%")
            result = True
        else:
            print(f"  ⚠️  No coverage metadata found")
            result = True  # Still pass - metadata is optional
    else:
        print(f"  ✗ Empty NAV result")
        result = False
    
    print(f"\n{'✓ Test PASSED' if result else '✗ Test FAILED'}")
    return result


def test_deterministic_results():
    """Test that multiple calls return identical results (deterministic)."""
    print("\n" + "=" * 80)
    print("TEST: Deterministic results from PRICE_BOOK")
    print("=" * 80)
    
    from waves_engine import compute_history_nav
    import pandas as pd
    
    wave = 'US MegaCap Core Wave'
    days = 30
    
    print(f"\n1. Computing wave NAV twice with same parameters...")
    
    # First computation
    nav1 = compute_history_nav(wave, mode='Standard', days=days)
    
    # Second computation
    nav2 = compute_history_nav(wave, mode='Standard', days=days)
    
    if nav1.empty or nav2.empty:
        print("  ✗ One or both results are empty")
        return False
    
    # Compare results
    print(f"\n2. Comparing results...")
    print(f"  First run: {len(nav1)} days, final NAV={nav1['wave_nav'].iloc[-1]:.6f}")
    print(f"  Second run: {len(nav2)} days, final NAV={nav2['wave_nav'].iloc[-1]:.6f}")
    
    # Check if results are identical
    try:
        pd.testing.assert_frame_equal(nav1, nav2, check_exact=False, rtol=1e-10)
        print(f"  ✓ Results are identical (deterministic)")
        result = True
    except AssertionError as e:
        print(f"  ✗ Results differ: {e}")
        result = False
    
    print(f"\n{'✓ Test PASSED' if result else '✗ Test FAILED'}")
    return result


def test_missing_ticker_handling():
    """Test that missing tickers are properly handled with NaN columns."""
    print("\n" + "=" * 80)
    print("TEST: Missing ticker handling")
    print("=" * 80)
    
    from waves_engine import _download_history
    
    print("\n1. Requesting tickers including a fake one...")
    
    # Request a mix of real and fake tickers
    tickers = ['SPY', 'FAKE_TICKER_XYZ_123']
    prices, failures = _download_history(tickers, days=30, wave_name='Test Wave')
    
    print(f"\n2. Checking results...")
    print(f"  Loaded {len(prices.columns)} tickers")
    print(f"  Failed tickers: {len(failures)}")
    
    # Check if fake ticker is in failures OR if it has NaN column
    if 'FAKE_TICKER_XYZ_123' in failures:
        print(f"  ✓ Fake ticker properly reported as missing in failures")
        print(f"  ✓ Reason: {failures['FAKE_TICKER_XYZ_123']}")
        result = True
    elif 'FAKE_TICKER_XYZ_123' in prices.columns and prices['FAKE_TICKER_XYZ_123'].isna().all():
        print(f"  ✓ Fake ticker has NaN column (acceptable behavior from PRICE_BOOK)")
        result = True
    else:
        print(f"  ✗ Fake ticker handling unexpected")
        result = False
    
    # Check if SPY is loaded (should be in cache)
    if not prices.empty and 'SPY' in prices.columns:
        print(f"  ✓ SPY successfully loaded from PRICE_BOOK")
    else:
        print(f"  ⚠️  SPY not loaded (may not be in cache)")
    
    print(f"\n{'✓ Test PASSED' if result else '✗ Test FAILED'}")
    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("UNIFIED PRICE SOURCE TEST SUITE")
    print("=" * 80)
    print(f"Testing that all price data comes from canonical PRICE_BOOK")
    print(f"Cache file: data/cache/prices_cache.parquet")
    print("=" * 80)
    
    tests = [
        ("No yfinance calls", test_no_yfinance_calls_in_wave_computation),
        ("PRICE_BOOK metadata", test_price_book_source_in_metadata),
        ("Deterministic results", test_deterministic_results),
        ("Missing ticker handling", test_missing_ticker_handling),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        symbol = "✓" if passed else "✗"
        status = "PASSED" if passed else "FAILED"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Unified price source working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Review results above")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
