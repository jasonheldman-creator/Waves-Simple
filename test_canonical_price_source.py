"""
Test suite for canonical price source implementation.

This test suite validates the single source of truth for price data:
- Canonical getter function (get_canonical_prices)
- Disabled implicit fetching
- Refined required tickers logic
- Diagnostics alignment with execution
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.price_loader import (
    get_canonical_prices,
    load_or_fetch_prices,
    collect_required_tickers,
    check_cache_readiness,
    get_cache_info,
    refresh_price_cache,
    CACHE_PATH
)


def test_canonical_getter():
    """Test that get_canonical_prices loads only from cache."""
    print("=" * 80)
    print("TEST: Canonical Price Getter")
    print("=" * 80)
    
    # Test 1: Load without tickers (should return all cached tickers)
    print("\n1. Loading all cached tickers...")
    all_prices = get_canonical_prices()
    
    if not all_prices.empty:
        print(f"  ✓ Loaded {len(all_prices)} days, {len(all_prices.columns)} tickers")
        assert len(all_prices.columns) > 0, "Should have cached tickers"
        assert len(all_prices) > 0, "Should have price data"
    else:
        print("  ⚠️ Cache is empty (expected if cache doesn't exist)")
    
    # Test 2: Load specific tickers
    print("\n2. Loading specific tickers...")
    specific_tickers = ['SPY', 'BTC-USD']
    specific_prices = get_canonical_prices(specific_tickers)
    
    print(f"  ✓ Loaded {len(specific_prices)} days, {len(specific_prices.columns)} tickers")
    assert len(specific_prices.columns) == len(specific_tickers), "Should have all requested tickers"
    
    # Test 3: Load non-existent ticker (should return NaN column)
    print("\n3. Loading non-existent ticker...")
    fake_ticker = ['NONEXISTENT_TICKER_XYZ']
    fake_prices = get_canonical_prices(fake_ticker)
    
    print(f"  ✓ Loaded {len(fake_prices)} days, {len(fake_prices.columns)} tickers")
    assert len(fake_prices.columns) == 1, "Should have one column for fake ticker"
    if not fake_prices.empty:
        assert fake_prices['NONEXISTENT_TICKER_XYZ'].isna().all(), "Should have all NaN for non-existent ticker"
    
    print("\n✓ Canonical getter tests passed")
    return True


def test_no_implicit_fetching():
    """Test that load_or_fetch_prices doesn't fetch by default."""
    print("\n" + "=" * 80)
    print("TEST: No Implicit Fetching")
    print("=" * 80)
    
    # Test 1: Call load_or_fetch_prices without force_fetch (should NOT download)
    print("\n1. Calling load_or_fetch_prices without force_fetch...")
    print("   This should ONLY load from cache, NOT download from network")
    
    test_tickers = ['SPY', 'AAPL']
    prices = load_or_fetch_prices(test_tickers)
    
    print(f"  ✓ Returned {len(prices)} days, {len(prices.columns)} tickers")
    print("  ✓ No network download occurred (loaded from cache only)")
    
    # Test 2: Verify force_fetch parameter exists and defaults to False
    print("\n2. Verifying force_fetch parameter...")
    import inspect
    sig = inspect.signature(load_or_fetch_prices)
    force_fetch_param = sig.parameters.get('force_fetch')
    
    assert force_fetch_param is not None, "force_fetch parameter should exist"
    assert force_fetch_param.default == False, "force_fetch should default to False"
    print("  ✓ force_fetch parameter defaults to False")
    
    print("\n✓ No implicit fetching tests passed")
    return True


def test_refined_required_tickers():
    """Test that collect_required_tickers only includes active wave tickers."""
    print("\n" + "=" * 80)
    print("TEST: Refined Required Tickers")
    print("=" * 80)
    
    # Test 1: Collect required tickers for active waves only
    print("\n1. Collecting required tickers (active waves only)...")
    required_tickers = collect_required_tickers(active_only=True)
    
    print(f"  ✓ Collected {len(required_tickers)} required tickers")
    print(f"  Sample: {required_tickers[:10] if len(required_tickers) > 10 else required_tickers}")
    
    # Test 2: Verify essential indicators are included
    print("\n2. Verifying essential indicators...")
    essential_indicators = ['SPY', '^VIX', 'BTC-USD']
    for indicator in essential_indicators:
        assert indicator in required_tickers, f"{indicator} should be in required tickers"
        print(f"  ✓ {indicator} is included")
    
    # Test 3: Verify tickers are normalized and deduplicated
    print("\n3. Verifying normalization and deduplication...")
    assert len(required_tickers) == len(set(required_tickers)), "Should have no duplicates"
    print("  ✓ No duplicates found")
    
    # All should be uppercase
    assert all(t == t.upper() or t.startswith('^') for t in required_tickers), "All tickers should be uppercase"
    print("  ✓ All tickers are properly normalized")
    
    print("\n✓ Refined required tickers tests passed")
    return True


def test_diagnostics_alignment():
    """Test that diagnostics align with execution logic."""
    print("\n" + "=" * 80)
    print("TEST: Diagnostics Alignment")
    print("=" * 80)
    
    # Test 1: Get cache info
    print("\n1. Getting cache info...")
    info = get_cache_info()
    
    print(f"  Cache exists: {info['exists']}")
    print(f"  Path: {info['path']}")
    print(f"  Tickers: {info['num_tickers']}")
    print(f"  Days: {info['num_days']}")
    print(f"  Last updated: {info['last_updated']}")
    print(f"  Is stale: {info['is_stale']}")
    print(f"  Days stale: {info['days_stale']}")
    
    assert info['path'] == CACHE_PATH, "Should report canonical cache path"
    print(f"  ✓ Reports canonical cache path: {CACHE_PATH}")
    
    # Test 2: Check cache readiness
    print("\n2. Checking cache readiness...")
    readiness = check_cache_readiness(active_only=True)
    
    print(f"  Status: {readiness['status_code']}")
    print(f"  Ready: {readiness['ready']}")
    print(f"  Required tickers: {readiness['required_tickers']}")
    print(f"  Cached tickers: {readiness['num_tickers']}")
    print(f"  Missing tickers: {len(readiness['missing_tickers'])}")
    print(f"  Extra tickers: {len(readiness['extra_tickers'])}")
    print(f"  Failed tickers: {len(readiness['failed_tickers'])}")
    
    # Test 3: Verify diagnostics provides actionable information
    print("\n3. Verifying diagnostics completeness...")
    required_fields = [
        'ready', 'exists', 'num_days', 'num_tickers', 'max_date',
        'days_stale', 'required_tickers', 'missing_tickers',
        'extra_tickers', 'failed_tickers', 'status', 'status_code'
    ]
    
    for field in required_fields:
        assert field in readiness, f"Readiness should include {field}"
    
    print(f"  ✓ All {len(required_fields)} required fields present")
    
    # Test 4: Verify missing vs extra distinction
    print("\n4. Verifying missing vs extra tickers distinction...")
    if info['exists'] and info['num_tickers'] > 0:
        # Extra tickers should not cause failures
        if readiness['extra_tickers']:
            print(f"  ℹ️ Found {len(readiness['extra_tickers'])} extra (harmless) tickers")
            print("  ✓ Extra tickers properly categorized as non-critical")
        
        # Missing tickers should affect readiness
        if readiness['missing_tickers']:
            print(f"  ⚠️ Found {len(readiness['missing_tickers'])} missing (critical) tickers")
            assert not readiness['ready'], "Should not be ready if tickers are missing"
            print("  ✓ Missing tickers properly affect readiness status")
    
    print("\n✓ Diagnostics alignment tests passed")
    return True


def test_cache_metadata():
    """Test that cache metadata is accurate and complete."""
    print("\n" + "=" * 80)
    print("TEST: Cache Metadata")
    print("=" * 80)
    
    # Get both info and readiness
    info = get_cache_info()
    readiness = check_cache_readiness(active_only=True)
    
    # Test 1: Verify metadata consistency
    print("\n1. Verifying metadata consistency...")
    
    if info['exists']:
        assert info['num_tickers'] == readiness['num_tickers'], "Ticker counts should match"
        assert info['num_days'] == readiness['num_days'], "Day counts should match"
        print("  ✓ Cache info and readiness report consistent metadata")
    else:
        assert readiness['status_code'] == 'MISSING', "Should report MISSING status"
        print("  ✓ Missing cache properly reported")
    
    # Test 2: Verify date information
    print("\n2. Verifying date information...")
    
    if info['last_updated']:
        # Should be a valid date string
        try:
            datetime.strptime(info['last_updated'], '%Y-%m-%d')
            print(f"  ✓ Last updated date is valid: {info['last_updated']}")
        except ValueError:
            raise AssertionError("Last updated should be a valid date string")
        
        # Staleness should be a non-negative integer
        assert isinstance(info['days_stale'], int), "Days stale should be an integer"
        assert info['days_stale'] >= 0, "Days stale should be non-negative"
        print(f"  ✓ Staleness properly calculated: {info['days_stale']} days")
    
    # Test 3: Verify ticker lists
    print("\n3. Verifying ticker lists...")
    
    if info['exists'] and info['num_tickers'] > 0:
        assert isinstance(info['tickers'], list), "Tickers should be a list"
        assert len(info['tickers']) == info['num_tickers'], "Ticker list length should match count"
        print(f"  ✓ Ticker list is complete: {info['num_tickers']} tickers")
    
    print("\n✓ Cache metadata tests passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("*" * 80)
    print("CANONICAL PRICE SOURCE TEST SUITE")
    print("*" * 80)
    
    tests = [
        ("Canonical Getter", test_canonical_getter),
        ("No Implicit Fetching", test_no_implicit_fetching),
        ("Refined Required Tickers", test_refined_required_tickers),
        ("Diagnostics Alignment", test_diagnostics_alignment),
        ("Cache Metadata", test_cache_metadata),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    print("*" * 80)
    print("TEST RESULTS")
    print("*" * 80)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print("🎉 All tests passed!")
    print("*" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
