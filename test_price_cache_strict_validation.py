"""
Test suite for price cache strict validation features.

Tests the new strict validation logic to ensure:
1. Cache freshness validation (max date within threshold)
2. Required symbol coverage checks
3. No-change detection logic
4. Exit code behavior for all validation scenarios
"""

import os
import sys
import tempfile
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_price_cache import (
    validate_required_symbols,
    validate_cache_freshness,
    detect_cache_changes,
    REQUIRED_SYMBOLS,
    MAX_STALE_DAYS,
)


def test_required_symbols_definition():
    """Test that required symbols are properly defined."""
    print("=" * 80)
    print("TEST: Required Symbols Definition")
    print("=" * 80)
    
    # Check all categories exist
    assert "volatility_proxies" in REQUIRED_SYMBOLS
    assert "benchmark_indices" in REQUIRED_SYMBOLS
    assert "cash_safe_instruments" in REQUIRED_SYMBOLS
    assert "crypto_benchmarks" in REQUIRED_SYMBOLS
    
    print(f"✓ All required categories defined: {list(REQUIRED_SYMBOLS.keys())}")
    
    # Check each category has symbols
    for category, symbols in REQUIRED_SYMBOLS.items():
        assert len(symbols) > 0, f"Category {category} has no symbols"
        print(f"  {category}: {symbols}")
    
    print("\n✓ Required symbols definition test passed")
    return True


def test_validate_required_symbols_all_present():
    """Test validation when all required symbols are present."""
    print("=" * 80)
    print("TEST: Validate Required Symbols - All Present")
    print("=" * 80)
    
    # Create cache with all required symbols
    all_symbols = []
    for symbols in REQUIRED_SYMBOLS.values():
        all_symbols.extend(symbols)
    
    # Add some extra symbols
    all_symbols.extend(['AAPL', 'MSFT', 'GOOGL'])
    
    # Create sample cache
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    cache_df = pd.DataFrame(
        np.random.randn(len(dates), len(all_symbols)),
        index=dates,
        columns=all_symbols
    )
    
    # Validate
    success, missing = validate_required_symbols(cache_df)
    
    assert success is True, "Expected validation to succeed"
    assert len(missing) == 0, f"Expected no missing symbols, got: {missing}"
    
    print(f"✓ Validation passed for cache with {len(all_symbols)} symbols")
    print(f"  Required categories covered: {list(REQUIRED_SYMBOLS.keys())}")
    print("\n✓ All required symbols present test passed")
    return True


def test_validate_required_symbols_missing():
    """Test validation when some required symbols are missing."""
    print("=" * 80)
    print("TEST: Validate Required Symbols - Some Missing")
    print("=" * 80)
    
    # Create cache missing some required symbols
    all_symbols = []
    for symbols in REQUIRED_SYMBOLS.values():
        all_symbols.extend(symbols)
    
    # Remove VIX and BIL to test missing symbols
    all_symbols.remove('^VIX')
    all_symbols.remove('BIL')
    
    # Create sample cache
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    cache_df = pd.DataFrame(
        np.random.randn(len(dates), len(all_symbols)),
        index=dates,
        columns=all_symbols
    )
    
    # Validate
    success, missing = validate_required_symbols(cache_df)
    
    assert success is False, "Expected validation to fail"
    assert len(missing) > 0, "Expected missing symbols to be reported"
    assert "volatility_proxies" in missing, "Expected volatility_proxies to be in missing"
    assert "cash_safe_instruments" in missing, "Expected cash_safe_instruments to be in missing"
    
    print(f"✓ Validation correctly detected missing symbols:")
    for category, symbols in missing.items():
        print(f"  {category}: {symbols}")
    
    print("\n✓ Missing required symbols detection test passed")
    return True


def test_validate_cache_freshness_recent_data():
    """Test freshness validation with recent data."""
    print("=" * 80)
    print("TEST: Cache Freshness - Recent Data")
    print("=" * 80)
    
    # Create cache with data up to yesterday
    end_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=365)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    cache_df = pd.DataFrame(
        np.random.randn(len(dates), 5),
        index=dates,
        columns=['SPY', 'QQQ', '^VIX', 'BTC-USD', 'ETH-USD']
    )
    
    # Validate
    is_fresh, max_date, days_old = validate_cache_freshness(cache_df, MAX_STALE_DAYS)
    
    assert is_fresh is True, f"Expected cache to be fresh (max_date={max_date}, days_old={days_old})"
    assert days_old <= MAX_STALE_DAYS, f"Expected days_old ({days_old}) <= threshold ({MAX_STALE_DAYS})"
    
    print(f"✓ Cache is fresh:")
    print(f"  Max date: {max_date.date()}")
    print(f"  Days old: {days_old}")
    print(f"  Threshold: {MAX_STALE_DAYS} days")
    
    print("\n✓ Recent data freshness test passed")
    return True


def test_validate_cache_freshness_stale_data():
    """Test freshness validation with stale data."""
    print("=" * 80)
    print("TEST: Cache Freshness - Stale Data")
    print("=" * 80)
    
    # Create cache with old data (10 days ago)
    end_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
    start_date = end_date - pd.Timedelta(days=365)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    cache_df = pd.DataFrame(
        np.random.randn(len(dates), 5),
        index=dates,
        columns=['SPY', 'QQQ', '^VIX', 'BTC-USD', 'ETH-USD']
    )
    
    # Validate
    is_fresh, max_date, days_old = validate_cache_freshness(cache_df, MAX_STALE_DAYS)
    
    assert is_fresh is False, f"Expected cache to be stale (max_date={max_date}, days_old={days_old})"
    assert days_old > MAX_STALE_DAYS, f"Expected days_old ({days_old}) > threshold ({MAX_STALE_DAYS})"
    
    print(f"✓ Cache correctly identified as stale:")
    print(f"  Max date: {max_date.date()}")
    print(f"  Days old: {days_old}")
    print(f"  Threshold: {MAX_STALE_DAYS} days")
    
    print("\n✓ Stale data detection test passed")
    return True


def test_detect_cache_changes_new_file():
    """Test change detection when cache is newly created."""
    print("=" * 80)
    print("TEST: Change Detection - New File")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        new_cache = os.path.join(tmpdir, "new_cache.parquet")
        old_cache = os.path.join(tmpdir, "old_cache.parquet")  # Doesn't exist
        
        # Create new cache
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_parquet(new_cache)
        
        # Detect changes
        has_changes, reason = detect_cache_changes(new_cache, old_cache)
        
        assert has_changes is True, "Expected changes to be detected for new file"
        assert "New cache file" in reason or "created" in reason.lower(), f"Unexpected reason: {reason}"
        
        print(f"✓ New file detected: {reason}")
    
    print("\n✓ New file detection test passed")
    return True


def test_detect_cache_changes_modified_file():
    """Test change detection when cache is modified."""
    print("=" * 80)
    print("TEST: Change Detection - Modified File")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        new_cache = os.path.join(tmpdir, "new_cache.parquet")
        old_cache = os.path.join(tmpdir, "old_cache.parquet")
        
        # Create old cache
        df_old = pd.DataFrame({'A': [1, 2, 3]})
        df_old.to_parquet(old_cache)
        
        # Create new cache with different data
        df_new = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_new.to_parquet(new_cache)
        
        # Detect changes
        has_changes, reason = detect_cache_changes(new_cache, old_cache)
        
        assert has_changes is True, "Expected changes to be detected"
        
        print(f"✓ Changes detected: {reason}")
    
    print("\n✓ Modified file detection test passed")
    return True


def test_detect_cache_changes_no_changes():
    """Test change detection when cache is unchanged."""
    print("=" * 80)
    print("TEST: Change Detection - No Changes")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "cache.parquet")
        
        # Create cache
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_parquet(cache_file)
        
        # Copy to simulate old cache
        old_cache = os.path.join(tmpdir, "old_cache.parquet")
        shutil.copy2(cache_file, old_cache)
        
        # Detect changes (comparing file with its copy)
        has_changes, reason = detect_cache_changes(cache_file, old_cache)
        
        assert has_changes is False, "Expected no changes to be detected"
        assert "No changes" in reason or "same" in reason.lower(), f"Unexpected reason: {reason}"
        
        print(f"✓ No changes detected: {reason}")
    
    print("\n✓ No changes detection test passed")
    return True


def test_freshness_with_different_thresholds():
    """Test freshness validation with different thresholds."""
    print("=" * 80)
    print("TEST: Cache Freshness - Different Thresholds")
    print("=" * 80)
    
    # Create cache with data 3 days old
    end_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
    start_date = end_date - pd.Timedelta(days=30)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    cache_df = pd.DataFrame(
        np.random.randn(len(dates), 2),
        index=dates,
        columns=['SPY', 'QQQ']
    )
    
    # Test with threshold = 2 days (should fail)
    is_fresh_2, _, days_old_2 = validate_cache_freshness(cache_df, max_stale_days=2)
    assert is_fresh_2 is False, "Expected stale with 2-day threshold"
    print(f"  ✓ With 2-day threshold: stale (days_old={days_old_2})")
    
    # Test with threshold = 5 days (should pass)
    is_fresh_5, _, days_old_5 = validate_cache_freshness(cache_df, max_stale_days=5)
    assert is_fresh_5 is True, "Expected fresh with 5-day threshold"
    print(f"  ✓ With 5-day threshold: fresh (days_old={days_old_5})")
    
    print("\n✓ Different thresholds test passed")
    return True


def test_empty_cache_validation():
    """Test validation behavior with empty cache."""
    print("=" * 80)
    print("TEST: Empty Cache Validation")
    print("=" * 80)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    
    # Test required symbols validation
    success, missing = validate_required_symbols(empty_df)
    assert success is False, "Expected validation to fail for empty cache"
    assert len(missing) > 0, "Expected all categories to be missing"
    print(f"  ✓ Required symbols validation correctly fails for empty cache")
    
    # Test freshness validation
    is_fresh, max_date, days_old = validate_cache_freshness(empty_df)
    assert is_fresh is False, "Expected freshness check to fail for empty cache"
    assert max_date is None, "Expected max_date to be None for empty cache"
    print(f"  ✓ Freshness validation correctly fails for empty cache")
    
    print("\n✓ Empty cache validation test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING PRICE CACHE STRICT VALIDATION TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        test_required_symbols_definition,
        test_validate_required_symbols_all_present,
        test_validate_required_symbols_missing,
        test_validate_cache_freshness_recent_data,
        test_validate_cache_freshness_stale_data,
        test_detect_cache_changes_new_file,
        test_detect_cache_changes_modified_file,
        test_detect_cache_changes_no_changes,
        test_freshness_with_different_thresholds,
        test_empty_cache_validation,
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
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
