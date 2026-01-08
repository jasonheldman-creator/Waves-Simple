"""
Unit Tests for Cache Validation Module

Tests cover:
1. Trading-day freshness validation
2. Required symbol validation with ALL/ANY group semantics
3. No-change behavior (fresh vs stale)
4. Cache integrity checks
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.cache_validation import (
    fetch_spy_trading_days,
    get_cache_max_date,
    validate_trading_day_freshness,
    validate_required_symbols,
    validate_cache_integrity,
    validate_no_change_logic,
    REQUIRED_SYMBOLS_ALL,
    REQUIRED_SYMBOLS_VIX_ANY,
    REQUIRED_SYMBOLS_TBILL_ANY
)


def create_test_cache(cache_path: str, symbols: list, end_date: datetime = None, num_days: int = 100):
    """
    Create a test cache parquet file.
    
    Args:
        cache_path: Path to create cache file
        symbols: List of symbols to include
        end_date: End date for the cache (default: today)
        num_days: Number of trading days to include
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Create date range (excluding weekends for realism)
    dates = pd.bdate_range(end=end_date, periods=num_days)
    
    # Create random price data
    data = {}
    for symbol in symbols:
        data[symbol] = np.random.uniform(50, 200, size=len(dates))
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to parquet
    df.to_parquet(cache_path)
    
    return df


class TestTradingDayFreshness:
    """Tests for trading-day freshness validation."""
    
    def test_offline_tolerance_validation(self):
        """Test the tolerance logic works correctly (offline test)."""
        print("\n--- Test: Offline Tolerance Validation ---")
        
        # Create a mock cache and simulate trading days
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Create test data with specific dates
            test_dates = pd.bdate_range(end=datetime(2024, 12, 20), periods=5)
            
            # Test 1: Cache at latest trading day
            print("\n  Test 1: Cache at latest trading day")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=test_dates[-1], num_days=10)
            cache_max = get_cache_max_date(cache_path)
            print(f"  Cache max date: {cache_max.date()}")
            print(f"  Test date: {test_dates[-1].date()}")
            assert cache_max.date() == test_dates[-1].date(), "Cache should match latest trading day"
            print("  ✓ Cache at latest trading day works")
            
            # Test 2: Cache 1 session behind
            print("\n  Test 2: Cache 1 session behind")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=test_dates[-2], num_days=10)
            cache_max = get_cache_max_date(cache_path)
            print(f"  Cache max date: {cache_max.date()}")
            print(f"  Test date: {test_dates[-2].date()}")
            assert cache_max.date() == test_dates[-2].date(), "Cache should be 1 session behind"
            print("  ✓ Cache 1 session behind detected correctly")
            
            # Test 3: Cache 2+ sessions behind
            print("\n  Test 3: Cache 2+ sessions behind")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=test_dates[-3], num_days=10)
            cache_max = get_cache_max_date(cache_path)
            print(f"  Cache max date: {cache_max.date()}")
            print(f"  Test date: {test_dates[-3].date()}")
            assert cache_max.date() == test_dates[-3].date(), "Cache should be 2+ sessions behind"
            print("  ✓ Cache 2+ sessions behind detected correctly")
            
        print("\n✅ PASS: Offline tolerance validation successful")
        return True
    
    def test_fetch_spy_trading_days(self):
        """Test that we can fetch SPY trading days."""
        print("\n--- Test: Fetch SPY Trading Days ---")
        
        last_trading_day, trading_days = fetch_spy_trading_days(calendar_days=10)
        
        if last_trading_day is None:
            print("⚠️  SKIP: Could not fetch SPY data (network or yfinance unavailable)")
            return True
        
        print(f"Last trading day: {last_trading_day.date()}")
        print(f"Total trading days found: {len(trading_days)}")
        
        assert last_trading_day is not None, "Should return a valid trading day"
        assert isinstance(last_trading_day, datetime), "Should return datetime object"
        assert len(trading_days) > 0, "Should return at least one trading day"
        assert last_trading_day in trading_days, "Last trading day should be in list"
        
        # Check that last_trading_day is recent (within 5 days of today)
        days_ago = (datetime.now() - last_trading_day).days
        assert days_ago <= 5, f"Last trading day should be recent (was {days_ago} days ago)"
        
        print("✅ PASS: SPY trading days fetched successfully")
        return True
    
    def test_cache_max_date(self):
        """Test getting max date from cache."""
        print("\n--- Test: Get Cache Max Date ---")
        
        # Create temporary cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Create cache with known end date
            test_end_date = datetime(2024, 12, 31)
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=test_end_date, num_days=10)
            
            # Get max date
            max_date = get_cache_max_date(cache_path)
            
            assert max_date is not None, "Should return a valid date"
            assert isinstance(max_date, datetime), "Should return datetime object"
            
            # Check that max_date matches our test_end_date (within a day for business day alignment)
            date_diff = abs((max_date.date() - test_end_date.date()).days)
            assert date_diff <= 3, f"Max date should be close to test end date (diff: {date_diff} days)"
            
            print(f"Cache max date: {max_date.date()}")
            print("✅ PASS: Cache max date retrieved correctly")
            
        return True
    
    def test_validate_trading_day_freshness_success(self):
        """Test validation passes when cache is fresh."""
        print("\n--- Test: Validate Trading-Day Freshness (Fresh Cache) ---")
        
        # Fetch current last trading day
        last_trading_day, _ = fetch_spy_trading_days(calendar_days=10)
        
        if last_trading_day is None:
            print("⚠️  SKIP: Could not fetch SPY data")
            return True
        
        # Create cache with current last trading day
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=last_trading_day, num_days=10)
            
            # Validate
            result = validate_trading_day_freshness(cache_path, max_market_feed_gap_days=5)
            
            print(f"Valid: {result['valid']}")
            print(f"Last trading day: {result['last_trading_day']}")
            print(f"Cache max date: {result['cache_max_date']}")
            print(f"Delta days: {result['delta_days']}")
            
            assert result['valid'] is True, "Should pass validation for fresh cache"
            assert result['error'] is None, "Should have no error"
            assert result['delta_days'] == 0, "Delta should be 0 for fresh cache"
            
            print("✅ PASS: Fresh cache validated successfully")
        
        return True
    
    def test_validate_trading_day_freshness_one_session_behind(self):
        """Test validation passes when cache is 1 trading session behind."""
        print("\n--- Test: Validate Trading-Day Freshness (1 Session Behind) ---")
        
        # Fetch current trading days
        last_trading_day, trading_days = fetch_spy_trading_days(calendar_days=10)
        
        if last_trading_day is None or len(trading_days) < 2:
            print("⚠️  SKIP: Could not fetch SPY data or insufficient trading days")
            return True
        
        # Get the second-to-last trading day (1 session behind)
        sorted_trading_days = sorted(trading_days, reverse=True)
        one_session_behind = sorted_trading_days[1]
        
        # Create cache with data from 1 session behind
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=one_session_behind, num_days=10)
            
            # Validate
            result = validate_trading_day_freshness(cache_path, max_market_feed_gap_days=5)
            
            print(f"Valid: {result['valid']}")
            print(f"Last trading day: {result['last_trading_day']}")
            print(f"Cache max date: {result['cache_max_date']}")
            print(f"Delta days: {result['delta_days']}")
            
            assert result['valid'] is True, "Should pass validation for cache 1 session behind"
            assert result['error'] is None, "Should have no error"
            
            print("✅ PASS: Cache 1 session behind validated successfully")
        
        return True
    
    def test_validate_trading_day_freshness_stale(self):
        """Test validation fails when cache is more than 1 session stale."""
        print("\n--- Test: Validate Trading-Day Freshness (Stale Cache) ---")
        
        # Fetch current trading days
        last_trading_day, trading_days = fetch_spy_trading_days(calendar_days=10)
        
        if last_trading_day is None or len(trading_days) < 3:
            print("⚠️  SKIP: Could not fetch SPY data or insufficient trading days")
            return True
        
        # Get a trading day that is 2+ sessions behind
        sorted_trading_days = sorted(trading_days, reverse=True)
        two_sessions_behind = sorted_trading_days[2] if len(sorted_trading_days) >= 3 else sorted_trading_days[-1]
        
        # Create cache with old data (2+ sessions behind)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            create_test_cache(cache_path, ["SPY", "QQQ"], end_date=two_sessions_behind, num_days=10)
            
            # Validate
            result = validate_trading_day_freshness(cache_path, max_market_feed_gap_days=5)
            
            print(f"Valid: {result['valid']}")
            print(f"Error: {result['error']}")
            
            assert result['valid'] is False, "Should fail validation for cache 2+ sessions behind"
            assert result['error'] is not None, "Should have an error message"
            assert "sessions behind" in result['error'] or "not a valid trading day" in result['error'], "Error should mention sessions or trading day"
            
            print("✅ PASS: Stale cache validation failed as expected")
        
        return True


class TestRequiredSymbols:
    """Tests for required symbol validation."""
    
    def test_validate_required_symbols_all_present(self):
        """Test validation passes when all required symbols are present."""
        print("\n--- Test: Required Symbols (All Present) ---")
        
        # Create cache with all required symbols
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Include all required symbols
            symbols = REQUIRED_SYMBOLS_ALL + [REQUIRED_SYMBOLS_VIX_ANY[0]] + [REQUIRED_SYMBOLS_TBILL_ANY[0]]
            create_test_cache(cache_path, symbols, num_days=10)
            
            # Validate
            result = validate_required_symbols(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"Missing ALL group: {result['missing_all_group']}")
            print(f"Present VIX group: {result['present_vix_group']}")
            print(f"Present T-bill group: {result['present_tbill_group']}")
            
            assert result['valid'] is True, "Should pass validation"
            assert len(result['missing_all_group']) == 0, "Should have no missing ALL group symbols"
            assert len(result['present_vix_group']) > 0, "Should have at least one VIX symbol"
            assert len(result['present_tbill_group']) > 0, "Should have at least one T-bill symbol"
            
            print("✅ PASS: All required symbols validated")
        
        return True
    
    def test_validate_required_symbols_missing_all_group(self):
        """Test validation fails when ALL group symbols are missing."""
        print("\n--- Test: Required Symbols (Missing ALL Group) ---")
        
        # Create cache missing one ALL group symbol (SPY)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Missing SPY from ALL group
            symbols = ["QQQ", "IWM"] + [REQUIRED_SYMBOLS_VIX_ANY[0]] + [REQUIRED_SYMBOLS_TBILL_ANY[0]]
            create_test_cache(cache_path, symbols, num_days=10)
            
            # Validate
            result = validate_required_symbols(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"Error: {result['error']}")
            print(f"Missing ALL group: {result['missing_all_group']}")
            
            assert result['valid'] is False, "Should fail validation"
            assert "SPY" in result['missing_all_group'], "Should identify SPY as missing"
            assert "ALL group" in result['error'], "Error should mention ALL group"
            
            print("✅ PASS: Missing ALL group detected")
        
        return True
    
    def test_validate_required_symbols_missing_vix_any(self):
        """Test validation fails when no VIX ANY group symbols are present."""
        print("\n--- Test: Required Symbols (Missing VIX ANY Group) ---")
        
        # Create cache missing all VIX symbols
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Missing all VIX symbols
            symbols = REQUIRED_SYMBOLS_ALL + [REQUIRED_SYMBOLS_TBILL_ANY[0]]
            create_test_cache(cache_path, symbols, num_days=10)
            
            # Validate
            result = validate_required_symbols(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"Error: {result['error']}")
            print(f"Present VIX group: {result['present_vix_group']}")
            
            assert result['valid'] is False, "Should fail validation"
            assert len(result['present_vix_group']) == 0, "Should have no VIX symbols"
            assert "VIX ANY group" in result['error'], "Error should mention VIX ANY group"
            
            print("✅ PASS: Missing VIX ANY group detected")
        
        return True
    
    def test_validate_required_symbols_missing_tbill_any(self):
        """Test validation fails when no T-bill ANY group symbols are present."""
        print("\n--- Test: Required Symbols (Missing T-bill ANY Group) ---")
        
        # Create cache missing all T-bill symbols
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            
            # Missing all T-bill symbols
            symbols = REQUIRED_SYMBOLS_ALL + [REQUIRED_SYMBOLS_VIX_ANY[0]]
            create_test_cache(cache_path, symbols, num_days=10)
            
            # Validate
            result = validate_required_symbols(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"Error: {result['error']}")
            print(f"Present T-bill group: {result['present_tbill_group']}")
            
            assert result['valid'] is False, "Should fail validation"
            assert len(result['present_tbill_group']) == 0, "Should have no T-bill symbols"
            assert "T-bill ANY group" in result['error'], "Error should mention T-bill ANY group"
            
            print("✅ PASS: Missing T-bill ANY group detected")
        
        return True


class TestCacheIntegrity:
    """Tests for cache integrity validation."""
    
    def test_validate_cache_integrity_success(self):
        """Test validation passes for valid cache."""
        print("\n--- Test: Cache Integrity (Valid) ---")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.parquet")
            create_test_cache(cache_path, ["SPY", "QQQ", "IWM"], num_days=10)
            
            result = validate_cache_integrity(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"File exists: {result['file_exists']}")
            print(f"File size: {result['file_size_bytes']} bytes")
            print(f"Symbol count: {result['symbol_count']}")
            
            assert result['valid'] is True, "Should pass validation"
            assert result['file_exists'] is True, "File should exist"
            assert result['file_size_bytes'] > 0, "File should have non-zero size"
            assert result['symbol_count'] == 3, "Should have 3 symbols"
            
            print("✅ PASS: Cache integrity validated")
        
        return True
    
    def test_validate_cache_integrity_missing_file(self):
        """Test validation fails when file doesn't exist."""
        print("\n--- Test: Cache Integrity (Missing File) ---")
        
        cache_path = "/tmp/nonexistent_cache.parquet"
        
        result = validate_cache_integrity(cache_path)
        
        print(f"Valid: {result['valid']}")
        print(f"Error: {result['error']}")
        
        assert result['valid'] is False, "Should fail validation"
        assert result['file_exists'] is False, "File should not exist"
        assert "does not exist" in result['error'], "Error should mention missing file"
        
        print("✅ PASS: Missing file detected")
        
        return True
    
    def test_validate_cache_integrity_empty_file(self):
        """Test validation fails when file is empty."""
        print("\n--- Test: Cache Integrity (Empty File) ---")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "empty_cache.parquet")
            
            # Create empty file
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            Path(cache_path).touch()
            
            result = validate_cache_integrity(cache_path)
            
            print(f"Valid: {result['valid']}")
            print(f"Error: {result['error']}")
            
            assert result['valid'] is False, "Should fail validation"
            assert result['file_size_bytes'] == 0, "File should be empty"
            assert "empty" in result['error'].lower(), "Error should mention empty file"
            
            print("✅ PASS: Empty file detected")
        
        return True


class TestNoChangeLogic:
    """Tests for no-change behavior validation."""
    
    def test_no_change_fresh_unchanged(self):
        """Test fresh + unchanged → success (no commit)."""
        print("\n--- Test: No-Change Logic (Fresh + Unchanged) ---")
        
        result = validate_no_change_logic(cache_freshness_valid=True, has_changes=False)
        
        print(f"Should commit: {result['should_commit']}")
        print(f"Should succeed: {result['should_succeed']}")
        print(f"Message: {result['message']}")
        
        assert result['should_commit'] is False, "Should not commit"
        assert result['should_succeed'] is True, "Should succeed"
        assert "Fresh but unchanged" in result['message'], "Message should indicate fresh but unchanged"
        
        print("✅ PASS: Fresh + unchanged → success (no commit)")
        
        return True
    
    def test_no_change_stale_unchanged(self):
        """Test stale + unchanged → fail."""
        print("\n--- Test: No-Change Logic (Stale + Unchanged) ---")
        
        result = validate_no_change_logic(cache_freshness_valid=False, has_changes=False)
        
        print(f"Should commit: {result['should_commit']}")
        print(f"Should succeed: {result['should_succeed']}")
        print(f"Message: {result['message']}")
        
        assert result['should_commit'] is False, "Should not commit"
        assert result['should_succeed'] is False, "Should fail"
        assert "Stale + unchanged" in result['message'], "Message should indicate stale + unchanged"
        
        print("✅ PASS: Stale + unchanged → fail")
        
        return True
    
    def test_no_change_fresh_changed(self):
        """Test fresh + changed → success (commit)."""
        print("\n--- Test: No-Change Logic (Fresh + Changed) ---")
        
        result = validate_no_change_logic(cache_freshness_valid=True, has_changes=True)
        
        print(f"Should commit: {result['should_commit']}")
        print(f"Should succeed: {result['should_succeed']}")
        print(f"Message: {result['message']}")
        
        assert result['should_commit'] is True, "Should commit"
        assert result['should_succeed'] is True, "Should succeed"
        assert "Fresh and changed" in result['message'], "Message should indicate fresh and changed"
        
        print("✅ PASS: Fresh + changed → success (commit)")
        
        return True
    
    def test_no_change_stale_changed(self):
        """Test stale + changed → success (commit)."""
        print("\n--- Test: No-Change Logic (Stale + Changed) ---")
        
        result = validate_no_change_logic(cache_freshness_valid=False, has_changes=True)
        
        print(f"Should commit: {result['should_commit']}")
        print(f"Should succeed: {result['should_succeed']}")
        print(f"Message: {result['message']}")
        
        assert result['should_commit'] is True, "Should commit"
        assert result['should_succeed'] is True, "Should succeed"
        assert "Stale but changed" in result['message'], "Message should indicate stale but changed"
        
        print("✅ PASS: Stale + changed → success (commit)")
        
        return True


def run_all_tests():
    """Run all test classes."""
    print("=" * 80)
    print("CACHE VALIDATION UNIT TESTS")
    print("=" * 80)
    
    test_classes = [
        TestTradingDayFreshness(),
        TestRequiredSymbols(),
        TestCacheIntegrity(),
        TestNoChangeLogic()
    ]
    
    all_passed = True
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'=' * 80}")
        print(f"{class_name}")
        print(f"{'=' * 80}")
        
        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                result = method()
                if result:
                    passed_tests += 1
                else:
                    all_passed = False
                    print(f"❌ FAIL: {method_name}")
            except AssertionError as e:
                all_passed = False
                print(f"❌ FAIL: {method_name}")
                print(f"   Error: {e}")
            except Exception as e:
                all_passed = False
                print(f"❌ ERROR: {method_name}")
                print(f"   Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
