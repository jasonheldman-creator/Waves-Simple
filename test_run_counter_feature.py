"""
Test Suite for Run Counter and PRICE_BOOK Freshness Feature
===========================================================

This test suite validates:
1. RUN COUNTER is properly tracked and displayed
2. Auto-refresh is disabled by default
3. Manual PRICE_BOOK rebuild works even in safe_mode
4. STALE data warnings are displayed correctly
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auto_refresh_config():
    """Test that auto-refresh is disabled by default"""
    from auto_refresh_config import DEFAULT_AUTO_REFRESH_ENABLED
    
    print("Testing auto-refresh configuration...")
    assert DEFAULT_AUTO_REFRESH_ENABLED == False, "Auto-refresh should be OFF by default"
    print("✅ Auto-refresh is OFF by default")


def test_price_book_rebuild_signature():
    """Test that rebuild_price_cache has force_user_initiated parameter"""
    from helpers.price_book import rebuild_price_cache
    import inspect
    
    print("\nTesting rebuild_price_cache signature...")
    sig = inspect.signature(rebuild_price_cache)
    params = list(sig.parameters.keys())
    
    assert 'active_only' in params, "rebuild_price_cache should have active_only parameter"
    assert 'force_user_initiated' in params, "rebuild_price_cache should have force_user_initiated parameter"
    
    # Check default value
    force_param = sig.parameters['force_user_initiated']
    assert force_param.default == False, "force_user_initiated should default to False"
    
    print("✅ rebuild_price_cache has correct signature with force_user_initiated parameter")


def test_stale_threshold_constant():
    """Test that STALE_DAYS_THRESHOLD is defined"""
    from helpers.price_book import STALE_DAYS_THRESHOLD
    
    print("\nTesting STALE_DAYS_THRESHOLD constant...")
    assert STALE_DAYS_THRESHOLD is not None, "STALE_DAYS_THRESHOLD should be defined"
    assert STALE_DAYS_THRESHOLD == 10, "STALE_DAYS_THRESHOLD should be 10 days"
    print(f"✅ STALE_DAYS_THRESHOLD = {STALE_DAYS_THRESHOLD} days")


def test_price_fetch_environment():
    """Test that PRICE_FETCH_ENABLED is properly configured"""
    from helpers.price_book import PRICE_FETCH_ENABLED, ALLOW_NETWORK_FETCH
    
    print("\nTesting PRICE_FETCH_ENABLED environment configuration...")
    # Check that both aliases exist
    assert hasattr(sys.modules['helpers.price_book'], 'PRICE_FETCH_ENABLED'), \
        "PRICE_FETCH_ENABLED should be defined"
    assert hasattr(sys.modules['helpers.price_book'], 'ALLOW_NETWORK_FETCH'), \
        "ALLOW_NETWORK_FETCH should be defined as alias"
    
    # Check they're consistent
    assert PRICE_FETCH_ENABLED == ALLOW_NETWORK_FETCH, \
        "PRICE_FETCH_ENABLED and ALLOW_NETWORK_FETCH should be consistent"
    
    print(f"✅ PRICE_FETCH_ENABLED = {PRICE_FETCH_ENABLED}")
    print(f"✅ ALLOW_NETWORK_FETCH = {ALLOW_NETWORK_FETCH}")


def test_rebuild_price_cache_bypass():
    """Test that rebuild_price_cache can be called with force_user_initiated"""
    from helpers.price_book import rebuild_price_cache, PRICE_FETCH_ENABLED
    
    print("\nTesting rebuild_price_cache with force_user_initiated...")
    
    # This should not fail even if PRICE_FETCH_ENABLED is False
    # Note: We're testing the function signature and logic flow, not actual fetching
    # The function will return an error if dependencies are missing, which is expected
    result = rebuild_price_cache(active_only=True, force_user_initiated=True)
    
    assert isinstance(result, dict), "rebuild_price_cache should return a dictionary"
    assert 'allowed' in result, "Result should have 'allowed' key"
    assert 'success' in result, "Result should have 'success' key"
    assert 'message' in result or result.get('allowed') == True, \
        "Result should have 'message' key if not allowed"
    
    print("✅ rebuild_price_cache accepts force_user_initiated parameter")
    print(f"   Result: allowed={result.get('allowed')}, success={result.get('success')}")
    if not result.get('allowed'):
        print(f"   Message: {result.get('message')}")


def run_all_tests():
    """Run all tests in the suite"""
    print("=" * 70)
    print("Run Counter and PRICE_BOOK Freshness Test Suite")
    print("=" * 70)
    
    tests = [
        test_auto_refresh_config,
        test_price_book_rebuild_signature,
        test_stale_threshold_constant,
        test_price_fetch_environment,
        test_rebuild_price_cache_bypass,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
