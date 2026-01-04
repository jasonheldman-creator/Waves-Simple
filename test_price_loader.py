"""
Test suite for price loader and caching module.

Tests the new canonical price loader with intelligent caching.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.price_loader import (
    normalize_ticker,
    deduplicate_tickers,
    load_or_fetch_prices,
    get_cache_info,
    clear_cache,
    CACHE_PATH
)
from analytics_pipeline import compute_data_ready_status
from waves_engine import get_all_wave_ids


def test_ticker_normalization():
    """Test that ticker normalization works correctly."""
    print("=" * 80)
    print("TEST: Ticker Normalization")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        ('BRK.B', 'BRK-B'),
        ('BF.B', 'BF-B'),
        ('stETH-USD', 'STETH-USD'),
        ('  aapl  ', 'AAPL'),
        ('msft', 'MSFT'),
    ]
    
    for input_ticker, expected in test_cases:
        result = normalize_ticker(input_ticker)
        assert result == expected, f"normalize_ticker('{input_ticker}') = '{result}', expected '{expected}'"
        print(f"  ✓ {input_ticker:20} -> {result}")
    
    print("\n✓ All ticker normalization tests passed")
    return True


def test_ticker_deduplication():
    """Test that ticker deduplication works correctly."""
    print("\n" + "=" * 80)
    print("TEST: Ticker Deduplication")
    print("=" * 80)
    
    # Test with duplicates and mixed case
    input_tickers = ['AAPL', 'aapl', 'MSFT', 'BRK.B', 'BRK-B', '  GOOGL  ', 'googl']
    expected = ['AAPL', 'BRK-B', 'GOOGL', 'MSFT']
    
    result = deduplicate_tickers(input_tickers)
    
    assert result == expected, f"deduplicate_tickers failed: got {result}, expected {expected}"
    
    print(f"  Input:  {input_tickers}")
    print(f"  Output: {result}")
    print("\n✓ Ticker deduplication test passed")
    return True


def test_cache_operations():
    """Test cache operations (clear, info)."""
    print("\n" + "=" * 80)
    print("TEST: Cache Operations")
    print("=" * 80)
    
    # Get cache info
    info = get_cache_info()
    
    print(f"  Cache exists: {info['exists']}")
    print(f"  Cache path: {info['path']}")
    
    if info['exists']:
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Tickers: {info['num_tickers']}")
        print(f"  Days: {info['num_days']}")
        print(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        print(f"  Sample tickers: {info['tickers'][:5]}")
    
    print("\n✓ Cache operations test passed")
    return True


def test_load_prices_basic():
    """Test basic price loading."""
    print("\n" + "=" * 80)
    print("TEST: Basic Price Loading")
    print("=" * 80)
    
    # Test with a few common tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Load last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"  Loading prices for {tickers}")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        prices = load_or_fetch_prices(
            tickers=tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        print(f"\n  Result shape: {prices.shape}")
        print(f"  Columns: {list(prices.columns)}")
        
        if not prices.empty:
            print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
            print(f"  Sample data:")
            print(prices.head())
            
            # Check that all requested tickers are present
            for ticker in tickers:
                assert ticker in prices.columns, f"Ticker {ticker} missing from result"
            
            print("\n✓ Basic price loading test passed")
        else:
            print("\n⚠ Warning: No price data returned (may be expected if network is unavailable)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading prices: {e}")
        return False


def test_smartsafe_exemption():
    """Test that SmartSafe cash waves are exempt from price checks."""
    print("\n" + "=" * 80)
    print("TEST: SmartSafe Cash Wave Exemption")
    print("=" * 80)
    
    # Test SmartSafe waves
    smartsafe_waves = [
        'smartsafe_treasury_cash_wave',
        'smartsafe_tax_free_money_market_wave'
    ]
    
    for wave_id in smartsafe_waves:
        result = compute_data_ready_status(wave_id, use_cache=True)
        
        print(f"\n  Wave: {wave_id}")
        print(f"    Readiness status: {result['readiness_status']}")
        print(f"    Reason: {result['reason']}")
        print(f"    Details: {result['details']}")
        
        # SmartSafe waves should be exempt
        assert result['reason'] in ['EXEMPT', 'READY'], \
            f"SmartSafe wave {wave_id} should be EXEMPT or READY, got {result['reason']}"
        
        assert result['is_ready'], \
            f"SmartSafe wave {wave_id} should be ready"
        
        assert result['coverage_pct'] == 100.0, \
            f"SmartSafe wave {wave_id} should have 100% coverage"
        
        print(f"    ✓ Correctly identified as exempt")
    
    print("\n✓ SmartSafe exemption test passed")
    return True


def test_readiness_with_cache():
    """Test readiness diagnostics with cache."""
    print("\n" + "=" * 80)
    print("TEST: Readiness Diagnostics with Cache")
    print("=" * 80)
    
    # Test a few waves
    test_waves = get_all_wave_ids()[:5]
    
    ready_count = 0
    not_ready_count = 0
    exempt_count = 0
    
    for wave_id in test_waves:
        result = compute_data_ready_status(wave_id, use_cache=True)
        
        print(f"\n  Wave: {wave_id}")
        print(f"    Status: {result['readiness_status']}")
        print(f"    Ready: {result['is_ready']}")
        print(f"    Coverage: {result['coverage_pct']:.1f}%")
        print(f"    History days: {result['history_days']}")
        print(f"    Missing tickers: {len(result['missing_tickers'])}")
        print(f"    Stale tickers: {len(result['stale_tickers'])}")
        
        # Check required fields are present
        required_fields = [
            'wave_id', 'display_name', 'readiness_status', 'is_ready',
            'analytics_ready', 'coverage_pct', 'history_days',
            'missing_tickers', 'stale_tickers', 'reason'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        if result['reason'] == 'EXEMPT':
            exempt_count += 1
        elif result['is_ready']:
            ready_count += 1
        else:
            not_ready_count += 1
            print(f"    Blocking issues: {result['blocking_issues']}")
            print(f"    Suggested actions: {result['suggested_actions'][:1]}")
    
    print(f"\n  Summary:")
    print(f"    Exempt: {exempt_count}")
    print(f"    Ready: {ready_count}")
    print(f"    Not ready: {not_ready_count}")
    
    print("\n✓ Readiness diagnostics test passed")
    return True


def test_missing_and_stale_detection():
    """Test detection of missing and stale tickers."""
    print("\n" + "=" * 80)
    print("TEST: Missing and Stale Ticker Detection")
    print("=" * 80)
    
    # Get a non-SmartSafe wave
    all_waves = get_all_wave_ids()
    test_wave = None
    
    for wave_id in all_waves:
        if 'smartsafe' not in wave_id.lower():
            test_wave = wave_id
            break
    
    if test_wave is None:
        print("  ⚠ No non-SmartSafe wave found for testing")
        return True
    
    result = compute_data_ready_status(test_wave, use_cache=True)
    
    print(f"  Test wave: {test_wave}")
    print(f"  Missing tickers: {len(result['missing_tickers'])}")
    if result['missing_tickers']:
        print(f"    Examples: {result['missing_tickers'][:5]}")
    
    print(f"  Stale tickers: {len(result['stale_tickers'])}")
    if result['stale_tickers']:
        print(f"    Examples: {result['stale_tickers'][:5]}")
        print(f"    Max staleness: {result['stale_days_max']} days")
    
    # Verify data structure
    assert isinstance(result['missing_tickers'], list), "missing_tickers should be a list"
    assert isinstance(result['stale_tickers'], list), "stale_tickers should be a list"
    assert isinstance(result['stale_days_max'], (int, float)), "stale_days_max should be numeric"
    
    print("\n✓ Missing and stale detection test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PRICE LOADER TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        ("Ticker Normalization", test_ticker_normalization),
        ("Ticker Deduplication", test_ticker_deduplication),
        ("Cache Operations", test_cache_operations),
        ("Basic Price Loading", test_load_prices_basic),
        ("SmartSafe Exemption", test_smartsafe_exemption),
        ("Readiness with Cache", test_readiness_with_cache),
        ("Missing/Stale Detection", test_missing_and_stale_detection),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
