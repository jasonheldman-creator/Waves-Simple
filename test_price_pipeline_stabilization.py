"""
Test Price Pipeline Stabilization Features

Tests for the new price cache infrastructure including:
- collect_required_tickers function
- Explicit cache refresh
- Cache readiness checks
- Failed ticker tracking
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_collect_required_tickers_imports():
    """Test that we can import the collect_required_tickers function."""
    try:
        from helpers.price_loader import collect_required_tickers
        print("✅ Successfully imported collect_required_tickers")
        return True
    except Exception as e:
        print(f"❌ Failed to import collect_required_tickers: {e}")
        return False


def test_check_cache_readiness_imports():
    """Test that we can import the check_cache_readiness function."""
    try:
        from helpers.price_loader import check_cache_readiness
        print("✅ Successfully imported check_cache_readiness")
        return True
    except Exception as e:
        print(f"❌ Failed to import check_cache_readiness: {e}")
        return False


def test_refresh_price_cache_imports():
    """Test that we can import the refresh_price_cache function."""
    try:
        from helpers.price_loader import refresh_price_cache
        print("✅ Successfully imported refresh_price_cache")
        return True
    except Exception as e:
        print(f"❌ Failed to import refresh_price_cache: {e}")
        return False


def test_save_failed_tickers_imports():
    """Test that we can import the save_failed_tickers function."""
    try:
        from helpers.price_loader import save_failed_tickers
        print("✅ Successfully imported save_failed_tickers")
        return True
    except Exception as e:
        print(f"❌ Failed to import save_failed_tickers: {e}")
        return False


def test_constants_updated():
    """Test that the constants have been updated correctly."""
    try:
        from helpers.price_loader import (
            MIN_REQUIRED_DAYS,
            MAX_STALE_DAYS,
            MAX_FORWARD_FILL_DAYS,
            RETRY_ATTEMPTS,
            BATCH_SIZE
        )
        
        print(f"MIN_REQUIRED_DAYS: {MIN_REQUIRED_DAYS}")
        print(f"MAX_STALE_DAYS: {MAX_STALE_DAYS}")
        print(f"MAX_FORWARD_FILL_DAYS: {MAX_FORWARD_FILL_DAYS}")
        print(f"RETRY_ATTEMPTS: {RETRY_ATTEMPTS}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        
        assert MIN_REQUIRED_DAYS == 60, f"Expected MIN_REQUIRED_DAYS=60, got {MIN_REQUIRED_DAYS}"
        assert MAX_STALE_DAYS == 5, f"Expected MAX_STALE_DAYS=5, got {MAX_STALE_DAYS}"
        assert MAX_FORWARD_FILL_DAYS == 3, f"Expected MAX_FORWARD_FILL_DAYS=3, got {MAX_FORWARD_FILL_DAYS}"
        assert RETRY_ATTEMPTS == 1, f"Expected RETRY_ATTEMPTS=1, got {RETRY_ATTEMPTS}"
        assert BATCH_SIZE == 50, f"Expected BATCH_SIZE=50, got {BATCH_SIZE}"
        
        print("✅ All constants updated correctly")
        return True
    except AssertionError as e:
        print(f"❌ Constant validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test constants: {e}")
        return False


def test_force_cache_refresh_env_var():
    """Test that FORCE_CACHE_REFRESH environment variable is checked."""
    try:
        # Set environment variable
        os.environ['FORCE_CACHE_REFRESH'] = '1'
        
        # Re-import to get new value
        import importlib
        import helpers.price_loader as pl
        importlib.reload(pl)
        
        assert pl.FORCE_CACHE_REFRESH, "FORCE_CACHE_REFRESH should be True when env var is '1'"
        
        # Reset environment variable
        os.environ['FORCE_CACHE_REFRESH'] = '0'
        importlib.reload(pl)
        
        assert not pl.FORCE_CACHE_REFRESH, "FORCE_CACHE_REFRESH should be False when env var is '0'"
        
        print("✅ FORCE_CACHE_REFRESH environment variable handling works correctly")
        return True
    except AssertionError as e:
        print(f"❌ Environment variable test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test environment variable: {e}")
        return False
    finally:
        # Clean up
        if 'FORCE_CACHE_REFRESH' in os.environ:
            del os.environ['FORCE_CACHE_REFRESH']


def test_failed_tickers_path():
    """Test that failed tickers path is defined."""
    try:
        from helpers.price_loader import FAILED_TICKERS_PATH
        
        print(f"FAILED_TICKERS_PATH: {FAILED_TICKERS_PATH}")
        
        assert 'failed_tickers.csv' in FAILED_TICKERS_PATH, "Path should contain 'failed_tickers.csv'"
        assert 'data/cache' in FAILED_TICKERS_PATH, "Path should be in 'data/cache'"
        
        print("✅ FAILED_TICKERS_PATH defined correctly")
        return True
    except AssertionError as e:
        print(f"❌ Path validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test path: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Price Pipeline Stabilization Tests")
    print("=" * 70)
    
    tests = [
        ("Import collect_required_tickers", test_collect_required_tickers_imports),
        ("Import check_cache_readiness", test_check_cache_readiness_imports),
        ("Import refresh_price_cache", test_refresh_price_cache_imports),
        ("Import save_failed_tickers", test_save_failed_tickers_imports),
        ("Constants updated", test_constants_updated),
        ("FORCE_CACHE_REFRESH env var", test_force_cache_refresh_env_var),
        ("Failed tickers path", test_failed_tickers_path),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        print("-" * 70)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
