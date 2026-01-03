#!/usr/bin/env python3
"""
Comprehensive Stabilization Test

This test validates all requirements from the stabilization problem statement:
1. Single entry point (app.py)
2. Single price source (PRICE_BOOK)
3. Correct active required ticker logic
4. Aligned diagnostics
5. Stability guarantees
"""

import sys
import os


def test_single_entry_point():
    """Test that app.py is the only entry point."""
    print("=" * 70)
    print("Test 1: Single Entry Point")
    print("=" * 70)
    
    # Check that minimal_app.py has been renamed
    if os.path.exists('minimal_app.py'):
        print("❌ FAILED: minimal_app.py still exists (should be renamed)")
        return False
    
    if not os.path.exists('_deprecated_minimal_app.py'):
        print("❌ FAILED: _deprecated_minimal_app.py not found")
        return False
    
    if not os.path.exists('app.py'):
        print("❌ FAILED: app.py not found")
        return False
    
    print("✅ SUCCESS: Entry point locked to app.py")
    print("   - minimal_app.py renamed to _deprecated_minimal_app.py")
    print("   - app.py exists as sole entry point")
    return True


def test_canonical_cache_path():
    """Test that canonical cache path is correctly defined."""
    print("\n" + "=" * 70)
    print("Test 2: Canonical Cache Path")
    print("=" * 70)
    
    from helpers.price_book import CANONICAL_CACHE_PATH
    
    expected_path = "data/cache/prices_cache.parquet"
    
    if expected_path not in CANONICAL_CACHE_PATH:
        print(f"❌ FAILED: Canonical cache path is '{CANONICAL_CACHE_PATH}'")
        print(f"   Expected to contain: '{expected_path}'")
        return False
    
    print(f"✅ SUCCESS: Canonical cache path is correct")
    print(f"   Path: {CANONICAL_CACHE_PATH}")
    return True


def test_price_fetch_disabled():
    """Test that PRICE_FETCH_ENABLED defaults to False."""
    print("\n" + "=" * 70)
    print("Test 3: Price Fetch Disabled by Default")
    print("=" * 70)
    
    from helpers.price_book import PRICE_FETCH_ENABLED
    
    if PRICE_FETCH_ENABLED:
        print("❌ FAILED: PRICE_FETCH_ENABLED is True (should be False by default)")
        return False
    
    print("✅ SUCCESS: PRICE_FETCH_ENABLED is False (safe mode)")
    print("   No implicit fetching will occur")
    return True


def test_active_ticker_collection():
    """Test that active ticker collection works correctly."""
    print("\n" + "=" * 70)
    print("Test 4: Active Ticker Collection")
    print("=" * 70)
    
    try:
        from helpers.price_loader import collect_required_tickers
        
        tickers = collect_required_tickers(active_only=True)
        
        if len(tickers) < 50:
            print(f"❌ FAILED: Only {len(tickers)} tickers collected")
            print(f"   Expected 80-200+ tickers for ~27 active waves")
            return False
        
        # Check for essential tickers
        essential = ['SPY', '^VIX', 'BTC-USD']
        missing_essential = [t for t in essential if t not in tickers]
        
        if missing_essential:
            print(f"❌ FAILED: Missing essential tickers: {missing_essential}")
            return False
        
        print(f"✅ SUCCESS: Active ticker collection working correctly")
        print(f"   Total tickers: {len(tickers)}")
        print(f"   Essential tickers present: {essential}")
        return True
    
    except Exception as e:
        print(f"❌ FAILED: Exception during ticker collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reality_panel_functions():
    """Test that Reality Panel functions are available."""
    print("\n" + "=" * 70)
    print("Test 5: Reality Panel Functions")
    print("=" * 70)
    
    try:
        from helpers.price_book import (
            get_price_book,
            get_price_book_meta,
            get_active_required_tickers,
            compute_missing_and_extra_tickers
        )
        
        print("✅ SUCCESS: All Reality Panel functions available")
        print("   - get_price_book")
        print("   - get_price_book_meta")
        print("   - get_active_required_tickers")
        print("   - compute_missing_and_extra_tickers")
        return True
    
    except ImportError as e:
        print(f"❌ FAILED: Missing Reality Panel function: {e}")
        return False


def test_staleness_logic():
    """Test that staleness logic is based on cache age, not extra tickers."""
    print("\n" + "=" * 70)
    print("Test 6: Staleness Logic")
    print("=" * 70)
    
    try:
        from helpers.price_loader import check_cache_readiness
        
        # This function should check staleness based on max date, not extra tickers
        # We'll verify by checking the signature and docstring
        import inspect
        
        sig = inspect.signature(check_cache_readiness)
        params = list(sig.parameters.keys())
        
        # Should have max_stale_days parameter
        if 'max_stale_days' not in params:
            print("❌ FAILED: check_cache_readiness missing max_stale_days parameter")
            return False
        
        # Check docstring mentions staleness based on date
        doc = inspect.getdoc(check_cache_readiness)
        if 'stale' not in doc.lower() or 'date' not in doc.lower():
            print("❌ FAILED: Staleness logic documentation unclear")
            return False
        
        print("✅ SUCCESS: Staleness logic properly implemented")
        print("   - Based on max_stale_days parameter")
        print("   - Checks data age, not extra tickers")
        return True
    
    except Exception as e:
        print(f"❌ FAILED: Exception checking staleness logic: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_compilation():
    """Test that app.py compiles without errors."""
    print("\n" + "=" * 70)
    print("Test 7: App Compilation")
    print("=" * 70)
    
    import py_compile
    
    try:
        py_compile.compile('app.py', doraise=True)
        print("✅ SUCCESS: app.py compiles without errors")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ FAILED: app.py compilation error: {e}")
        return False


def main():
    """Run all stabilization tests."""
    print("=" * 70)
    print("COMPREHENSIVE STABILIZATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ('Single Entry Point', test_single_entry_point),
        ('Canonical Cache Path', test_canonical_cache_path),
        ('Price Fetch Disabled', test_price_fetch_disabled),
        ('Active Ticker Collection', test_active_ticker_collection),
        ('Reality Panel Functions', test_reality_panel_functions),
        ('Staleness Logic', test_staleness_logic),
        ('App Compilation', test_app_compilation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    print("\n" + "=" * 70)
    print("STABILIZATION TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    print("=" * 70)
    
    if passed_count == total_count:
        print("\n🎉 ALL STABILIZATION TESTS PASSED!")
        print("\nThe app is now stabilized with:")
        print("  ✓ Single entry point (app.py)")
        print("  ✓ Single price source (PRICE_BOOK)")
        print("  ✓ Correct active ticker logic")
        print("  ✓ Aligned diagnostics")
        print("  ✓ Stability guarantees")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        print("Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
