"""
Test suite for price cache pipeline enhancements.

Tests the new features:
1. Trading-day aware freshness validation
2. Required symbol validation (volatility, benchmarks, cash proxies)
3. No-change rule behavior (stale + unchanged = error)
"""

import os
import sys
import json
import tempfile
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from build_price_cache
from build_price_cache import (
    validate_required_symbols,
    is_cache_fresh,
    REQUIRED_VOLATILITY_REGIME,
    REQUIRED_BENCHMARKS,
    REQUIRED_CASH_PROXIES,
    MAX_STALE_CALENDAR_DAYS
)


def test_required_symbols_validation():
    """Test validation of required symbols in cache."""
    print("=" * 80)
    print("TEST: Required Symbols Validation")
    print("=" * 80)
    
    # Test case 1: All required symbols present
    cache_data = {
        '^VIX': [100, 101, 102],
        'SPY': [400, 401, 402],
        'QQQ': [300, 301, 302],
        'IWM': [200, 201, 202],
        'BIL': [90, 90.1, 90.2],
        'SHY': [80, 80.1, 80.2],
        'AAPL': [150, 151, 152]
    }
    df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
    
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is True, "Expected validation to pass with all required symbols"
    assert len(missing) == 0, f"Expected no missing symbols, got {missing}"
    print("  ✓ All required symbols present → VALID")
    
    # Test case 2: Missing all volatility regime symbols
    cache_data = {
        'SPY': [400, 401, 402],
        'QQQ': [300, 301, 302],
        'IWM': [200, 201, 202],
        'BIL': [90, 90.1, 90.2],
        'SHY': [80, 80.1, 80.2],
    }
    df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
    
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is False, "Expected validation to fail without volatility symbols"
    assert 'volatility' in missing, "Expected missing volatility symbols"
    assert missing['volatility'] == REQUIRED_VOLATILITY_REGIME
    print("  ✓ Missing all volatility symbols → INVALID")
    
    # Test case 3: Have one volatility symbol (^VIX) - should pass
    cache_data = {
        '^VIX': [100, 101, 102],
        'SPY': [400, 401, 402],
        'QQQ': [300, 301, 302],
        'IWM': [200, 201, 202],
        'BIL': [90, 90.1, 90.2],
        'SHY': [80, 80.1, 80.2],
    }
    df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
    
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is True, "Expected validation to pass with ^VIX present"
    print("  ✓ Only ^VIX present (not VIXY or VXX) → VALID")
    
    # Test case 4: Missing benchmark symbol (SPY)
    cache_data = {
        '^VIX': [100, 101, 102],
        'QQQ': [300, 301, 302],
        'IWM': [200, 201, 202],
        'BIL': [90, 90.1, 90.2],
        'SHY': [80, 80.1, 80.2],
    }
    df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
    
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is False, "Expected validation to fail without SPY"
    assert 'benchmarks' in missing
    assert 'SPY' in missing['benchmarks']
    print("  ✓ Missing SPY → INVALID")
    
    # Test case 5: Missing cash proxy (BIL)
    cache_data = {
        '^VIX': [100, 101, 102],
        'SPY': [400, 401, 402],
        'QQQ': [300, 301, 302],
        'IWM': [200, 201, 202],
        'SHY': [80, 80.1, 80.2],
    }
    df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
    
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is False, "Expected validation to fail without BIL"
    assert 'cash_proxies' in missing
    assert 'BIL' in missing['cash_proxies']
    print("  ✓ Missing BIL → INVALID")
    
    # Test case 6: Empty cache
    df = pd.DataFrame()
    is_valid, missing = validate_required_symbols(df)
    assert is_valid is False, "Expected validation to fail with empty cache"
    assert 'volatility' in missing
    assert 'benchmarks' in missing
    assert 'cash_proxies' in missing
    print("  ✓ Empty cache → INVALID")
    
    print("\n✓ All required symbols validation tests passed")
    return True


def test_cache_freshness_validation():
    """Test trading-day aware cache freshness validation."""
    print("=" * 80)
    print("TEST: Cache Freshness Validation")
    print("=" * 80)
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Test case 1: Cache updated today - should be fresh
    max_date = today
    is_fresh, reason = is_cache_fresh(max_date)
    assert is_fresh is True, f"Expected fresh for today's date: {reason}"
    print(f"  ✓ Today's date → FRESH")
    
    # Test case 2: Cache updated 1 day ago - should be fresh
    max_date = today - timedelta(days=1)
    is_fresh, reason = is_cache_fresh(max_date)
    assert is_fresh is True, f"Expected fresh for 1 day old: {reason}"
    print(f"  ✓ 1 day old → FRESH")
    
    # Test case 3: Cache updated 5 days ago - should be fresh (at threshold)
    max_date = today - timedelta(days=MAX_STALE_CALENDAR_DAYS)
    is_fresh, reason = is_cache_fresh(max_date)
    assert is_fresh is True, f"Expected fresh for {MAX_STALE_CALENDAR_DAYS} days old: {reason}"
    print(f"  ✓ {MAX_STALE_CALENDAR_DAYS} days old (at threshold) → FRESH")
    
    # Test case 4: Cache updated 6 days ago - should be stale
    max_date = today - timedelta(days=MAX_STALE_CALENDAR_DAYS + 1)
    is_fresh, reason = is_cache_fresh(max_date)
    # Note: Without network access, this will be stale based on calendar days
    # In production with network, it might be fresh if it matches last trading day
    print(f"  ✓ {MAX_STALE_CALENDAR_DAYS + 1} days old → {is_fresh} (reason: {reason})")
    
    # Test case 5: Cache updated 30 days ago - should definitely be stale
    max_date = today - timedelta(days=30)
    is_fresh, reason = is_cache_fresh(max_date)
    assert is_fresh is False, f"Expected stale for 30 days old: {reason}"
    print(f"  ✓ 30 days old → STALE")
    
    # Test case 6: None/empty max_date - should be stale
    is_fresh, reason = is_cache_fresh(None)
    assert is_fresh is False, f"Expected stale for None: {reason}"
    print(f"  ✓ None/empty date → STALE")
    
    # Test case 7: String date format
    max_date_str = today.strftime('%Y-%m-%d')
    is_fresh, reason = is_cache_fresh(max_date_str)
    assert is_fresh is True, f"Expected fresh for string date today: {reason}"
    print(f"  ✓ String date format → handled correctly")
    
    print("\n✓ All cache freshness validation tests passed")
    return True


def test_no_change_rule_behavior():
    """Test the no-change rule: fail only if stale AND unchanged."""
    print("=" * 80)
    print("TEST: No-Change Rule Behavior")
    print("=" * 80)
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Scenario 1: Fresh cache + no changes = OK (debug log)
    fresh_date = today - timedelta(days=1)
    is_fresh, reason = is_cache_fresh(fresh_date)
    has_changes = False
    
    should_fail = (not is_fresh) and (not has_changes)
    assert should_fail is False, "Should not fail when cache is fresh, even without changes"
    print("  ✓ Fresh cache + No changes = OK (workflow passes with debug log)")
    
    # Scenario 2: Stale cache + has changes = OK (data updated)
    stale_date = today - timedelta(days=10)
    is_fresh, reason = is_cache_fresh(stale_date)
    has_changes = True
    
    should_fail = (not is_fresh) and (not has_changes)
    assert should_fail is False, "Should not fail when changes are present"
    print("  ✓ Stale cache + Has changes = OK (workflow commits updates)")
    
    # Scenario 3: Stale cache + no changes = FAIL (unable to update)
    stale_date = today - timedelta(days=10)
    is_fresh, reason = is_cache_fresh(stale_date)
    has_changes = False
    
    should_fail = (not is_fresh) and (not has_changes)
    assert should_fail is True, "Should fail when cache is stale AND unchanged"
    print("  ✓ Stale cache + No changes = FAIL (error: unable to fetch new data)")
    
    # Scenario 4: Fresh cache + has changes = OK (normal update)
    fresh_date = today
    is_fresh, reason = is_cache_fresh(fresh_date)
    has_changes = True
    
    should_fail = (not is_fresh) and (not has_changes)
    assert should_fail is False, "Should not fail when fresh with changes"
    print("  ✓ Fresh cache + Has changes = OK (workflow commits updates)")
    
    print("\n✓ All no-change rule behavior tests passed")
    return True


def test_metadata_with_missing_symbols():
    """Test that metadata correctly records missing required symbols."""
    print("=" * 80)
    print("TEST: Metadata with Missing Symbols")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock cache with missing symbols
        cache_data = {
            'SPY': [400, 401, 402],
            'QQQ': [300, 301, 302],
            # Missing: IWM, ^VIX/VIXY/VXX, BIL, SHY
        }
        df = pd.DataFrame(cache_data, index=pd.date_range('2024-01-01', periods=3))
        
        is_valid, missing = validate_required_symbols(df)
        
        # Create metadata
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "success_rate": 0.95,
            "min_success_rate": 0.90,
            "tickers_total": 100,
            "tickers_successful": 95,
            "tickers_failed": 5,
            "max_price_date": "2024-01-03",
            "cache_file": "test_cache.parquet"
        }
        
        if not is_valid:
            metadata["missing_required_symbols"] = missing
        
        # Validate metadata structure
        assert "missing_required_symbols" in metadata, "Expected missing_required_symbols in metadata"
        assert "volatility" in metadata["missing_required_symbols"]
        assert "benchmarks" in metadata["missing_required_symbols"]
        assert "cash_proxies" in metadata["missing_required_symbols"]
        
        print(f"  ✓ Metadata includes missing_required_symbols field")
        print(f"  ✓ Missing volatility: {metadata['missing_required_symbols']['volatility']}")
        print(f"  ✓ Missing benchmarks: {metadata['missing_required_symbols']['benchmarks']}")
        print(f"  ✓ Missing cash_proxies: {metadata['missing_required_symbols']['cash_proxies']}")
    
    print("\n✓ All metadata with missing symbols tests passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING PRICE CACHE ENHANCEMENT TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        test_required_symbols_validation,
        test_cache_freshness_validation,
        test_no_change_rule_behavior,
        test_metadata_with_missing_symbols,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
