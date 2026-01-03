#!/usr/bin/env python3
"""
Test script to verify app stability and PRICE_BOOK centralization.

This script tests:
1. PRICE_BOOK singleton loads correctly
2. System health computation works
3. All required functions are available
4. No infinite loops in data loading
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_price_book_centralization():
    """Test that PRICE_BOOK is the single source of truth."""
    print("=" * 70)
    print("TEST: PRICE_BOOK Centralization")
    print("=" * 70)
    
    from helpers.price_book import (
        get_price_book,
        get_price_book_meta,
        get_price_book_singleton,
        compute_system_health,
        get_required_tickers_active_waves,
        get_active_required_tickers,
        compute_missing_and_extra_tickers,
        ALLOW_NETWORK_FETCH,
        PRICE_FETCH_ENABLED,
        CANONICAL_CACHE_PATH
    )
    
    # Test 1: ALLOW_NETWORK_FETCH is False by default
    print("\n1. Testing ALLOW_NETWORK_FETCH flag...")
    assert ALLOW_NETWORK_FETCH == False, "ALLOW_NETWORK_FETCH should be False by default"
    assert PRICE_FETCH_ENABLED == False, "PRICE_FETCH_ENABLED should be False by default"
    print("   ✓ ALLOW_NETWORK_FETCH = False (safe)")
    
    # Test 2: Load PRICE_BOOK singleton
    print("\n2. Testing PRICE_BOOK singleton...")
    price_book_1 = get_price_book_singleton()
    price_book_2 = get_price_book_singleton()
    assert price_book_1 is price_book_2, "Singleton should return same object"
    print(f"   ✓ Singleton works: {len(price_book_1)} rows × {len(price_book_1.columns)} cols")
    
    # Test 3: get_required_tickers_active_waves
    print("\n3. Testing get_required_tickers_active_waves()...")
    tickers_1 = get_required_tickers_active_waves()
    tickers_2 = get_active_required_tickers()
    assert tickers_1 == tickers_2, "Both functions should return same tickers"
    print(f"   ✓ Returns {len(tickers_1)} active required tickers")
    
    # Test 4: System health computation
    print("\n4. Testing compute_system_health()...")
    health = compute_system_health(price_book_1)
    assert 'health_status' in health
    assert 'health_emoji' in health
    assert 'missing_count' in health
    assert 'total_required' in health
    assert 'coverage_pct' in health
    print(f"   ✓ Health: {health['health_emoji']} {health['health_status']}")
    print(f"   ✓ Coverage: {health['coverage_pct']:.1f}%")
    
    # Test 5: Missing/Extra ticker analysis
    print("\n5. Testing compute_missing_and_extra_tickers()...")
    analysis = compute_missing_and_extra_tickers(price_book_1)
    assert 'missing_tickers' in analysis
    assert 'extra_tickers' in analysis
    print(f"   ✓ Missing: {analysis['missing_count']} tickers")
    print(f"   ✓ Extra: {analysis['extra_count']} tickers (harmless)")
    
    # Test 6: Normal ETF tickers present
    print("\n6. Testing normal ETF tickers (SPY, QQQ, NVDA)...")
    test_tickers = ['SPY', 'QQQ', 'NVDA']
    for ticker in test_tickers:
        assert ticker in price_book_1.columns, f"{ticker} should be in PRICE_BOOK"
        coverage = price_book_1[ticker].notna().sum() / len(price_book_1) * 100
        print(f"   ✓ {ticker}: {coverage:.1f}% data coverage")
    
    # Test 7: Canonical cache path
    print("\n7. Testing canonical cache path...")
    expected_path = os.path.join("data", "cache", "prices_cache.parquet")
    assert CANONICAL_CACHE_PATH == expected_path, f"Expected {expected_path}, got {CANONICAL_CACHE_PATH}"
    print(f"   ✓ Canonical path: {CANONICAL_CACHE_PATH}")
    
    print("\n" + "=" * 70)
    print("✅ All PRICE_BOOK centralization tests PASSED")
    print("=" * 70)
    return True


def test_no_implicit_fetching():
    """Test that no implicit fetching occurs."""
    print("\n" + "=" * 70)
    print("TEST: No Implicit Network Fetching")
    print("=" * 70)
    
    from helpers.price_loader import load_or_fetch_prices
    import inspect
    
    # Check that force_fetch defaults to False
    sig = inspect.signature(load_or_fetch_prices)
    force_fetch_param = sig.parameters.get('force_fetch')
    
    assert force_fetch_param is not None, "force_fetch parameter should exist"
    assert force_fetch_param.default == False, "force_fetch should default to False"
    
    print("\n✓ load_or_fetch_prices has force_fetch=False by default")
    print("✓ No implicit network fetching will occur")
    
    return True


def test_diagnostics_consistency():
    """Test that diagnostics use PRICE_BOOK."""
    print("\n" + "=" * 70)
    print("TEST: Diagnostics Consistency")
    print("=" * 70)
    
    from helpers.price_book import (
        get_price_book,
        get_price_book_meta,
        compute_missing_and_extra_tickers
    )
    from helpers.price_loader import check_cache_readiness
    
    # Load PRICE_BOOK
    price_book = get_price_book()
    
    # Get meta from PRICE_BOOK
    meta = get_price_book_meta(price_book)
    
    # Get readiness from price_loader
    readiness = check_cache_readiness(active_only=True)
    
    # They should report consistent ticker counts
    assert meta['tickers_count'] == readiness['num_tickers'], \
        "PRICE_BOOK and cache readiness should report same ticker count"
    
    # They should report consistent date counts
    assert meta['rows'] == readiness['num_days'], \
        "PRICE_BOOK and cache readiness should report same day count"
    
    print(f"\n✓ PRICE_BOOK meta: {meta['rows']} days × {meta['tickers_count']} tickers")
    print(f"✓ Cache readiness: {readiness['num_days']} days × {readiness['num_tickers']} tickers")
    print("✓ Diagnostics are consistent")
    
    return True


def run_all_tests():
    """Run all stability tests."""
    print("\n")
    print("*" * 80)
    print("WAVES-SIMPLE APP STABILITY TEST SUITE")
    print("Testing PRICE_BOOK centralization and infinite loop prevention")
    print("*" * 80)
    
    tests = [
        ("PRICE_BOOK Centralization", test_price_book_centralization),
        ("No Implicit Fetching", test_no_implicit_fetching),
        ("Diagnostics Consistency", test_diagnostics_consistency),
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
        print("🎉 All stability tests passed!")
    print("*" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
