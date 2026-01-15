"""
Unit Tests for validate_cache_metadata.py

Tests the trading-day aware cache validation logic.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from validate_cache_metadata import (
    is_valid_spy_date,
    fetch_spy_latest_trading_day,
    validate_cache_metadata
)


class TestIsValidSpyDate:
    """Tests for is_valid_spy_date function."""
    
    def test_valid_date_string(self):
        """Test that valid date string passes."""
        print("\n--- Test: Valid date string ---")
        assert is_valid_spy_date("2026-01-09") is True
        print("✓ PASS")
    
    def test_none_fails(self):
        """Test that None fails validation."""
        print("\n--- Test: None fails ---")
        assert is_valid_spy_date(None) is False
        print("✓ PASS")
    
    def test_empty_string_fails(self):
        """Test that empty string fails validation."""
        print("\n--- Test: Empty string fails ---")
        assert is_valid_spy_date("") is False
        print("✓ PASS")
    
    def test_whitespace_fails(self):
        """Test that whitespace-only string fails validation."""
        print("\n--- Test: Whitespace fails ---")
        assert is_valid_spy_date("  ") is False
        print("✓ PASS")


class TestFetchSpyLatestTradingDay:
    """Tests for fetch_spy_latest_trading_day function."""
    
    def test_mock_successful_fetch(self):
        """Test successful SPY data fetch with mocked yfinance."""
        print("\n--- Test: Mock successful fetch ---")
        
        # Create mock SPY data - use dates that are actual business days
        # Jan 9, 2026 is a Friday, so this will work
        dates = pd.date_range(end='2026-01-09', periods=5, freq='B')  # 5 business days
        mock_data = pd.DataFrame(
            {'Close': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        # Mock yfinance download - need to patch at the module level where it's imported
        with patch('yfinance.download', return_value=mock_data):
            latest, trading_days = fetch_spy_latest_trading_day(calendar_days=10)
            
            assert latest is not None, "Should return a trading day"
            assert len(trading_days) == 5, "Should return 5 trading days"
            assert latest == date(2026, 1, 9), f"Latest should be 2026-01-09, got {latest}"
            
            print(f"Latest trading day: {latest}")
            print(f"Trading days: {trading_days}")
            print("✓ PASS")
    
    def test_empty_data_returns_none(self):
        """Test that empty SPY data returns None."""
        print("\n--- Test: Empty data returns None ---")
        
        # Mock yfinance with empty data
        with patch('yfinance.download', return_value=pd.DataFrame()):
            latest, trading_days = fetch_spy_latest_trading_day(calendar_days=10)
            
            assert latest is None, "Should return None for empty data"
            assert trading_days == [], "Should return empty list"
            
            print("✓ PASS")
    
    def test_network_error_returns_none(self):
        """Test that network errors return None gracefully."""
        print("\n--- Test: Network error returns None ---")
        
        # Mock yfinance to raise an exception
        with patch('yfinance.download', side_effect=Exception("Network error")):
            latest, trading_days = fetch_spy_latest_trading_day(calendar_days=10)
            
            assert latest is None, "Should return None on network error"
            assert trading_days == [], "Should return empty list"
            
            print("✓ PASS")


class TestValidateCacheMetadata:
    """Tests for validate_cache_metadata function."""
    
    def test_missing_file_fails(self):
        """Test that missing metadata file fails validation."""
        print("\n--- Test: Missing file fails ---")
        
        result = validate_cache_metadata('/nonexistent/file.json')
        assert result is False, "Should fail for missing file"
        
        print("✓ PASS")
    
    def test_missing_spy_max_date_fails(self):
        """Test that missing spy_max_date fails validation."""
        print("\n--- Test: Missing spy_max_date fails ---")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metadata = {
                "tickers_total": 100
                # spy_max_date is missing
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            result = validate_cache_metadata(temp_file)
            assert result is False, "Should fail for missing spy_max_date"
            print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_tickers_total_below_threshold_fails(self):
        """Test that tickers_total < 50 fails validation."""
        print("\n--- Test: tickers_total < 50 fails ---")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metadata = {
                "spy_max_date": "2026-01-10",
                "tickers_total": 30  # Below threshold
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            result = validate_cache_metadata(temp_file)
            assert result is False, "Should fail for tickers_total < 50"
            print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_exact_match_passes(self):
        """Test that exact match with latest trading day passes."""
        print("\n--- Test: Exact match passes ---")
        
        # Create mock SPY data with latest trading day = 2026-01-09 (Friday)
        dates = pd.date_range(end='2026-01-09', periods=5, freq='B')
        mock_data = pd.DataFrame(
            {'Close': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metadata = {
                "spy_max_date": "2026-01-09",  # Matches latest trading day
                "tickers_total": 100
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            with patch('yfinance.download', return_value=mock_data):
                result = validate_cache_metadata(temp_file, grace_period_days=1)
                assert result is True, "Should pass for exact match"
                print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_one_day_behind_with_grace_period_passes(self):
        """Test that 1 trading day behind passes with grace_period=1."""
        print("\n--- Test: 1 day behind with grace period passes ---")
        
        # Create mock SPY data with latest trading day = 2026-01-09 (Friday)
        dates = pd.date_range(end='2026-01-09', periods=5, freq='B')
        mock_data = pd.DataFrame(
            {'Close': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Get the second-to-last trading day
            second_last = dates[-2].date()
            metadata = {
                "spy_max_date": str(second_last),  # 1 trading day behind
                "tickers_total": 100
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            with patch('yfinance.download', return_value=mock_data):
                result = validate_cache_metadata(temp_file, grace_period_days=1)
                assert result is True, "Should pass for 1 day behind with grace_period=1"
                print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_one_day_behind_without_grace_period_fails(self):
        """Test that 1 trading day behind fails with grace_period=0."""
        print("\n--- Test: 1 day behind without grace period fails ---")
        
        # Create mock SPY data with latest trading day = 2026-01-09 (Friday)
        dates = pd.date_range(end='2026-01-09', periods=5, freq='B')
        mock_data = pd.DataFrame(
            {'Close': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Get the second-to-last trading day
            second_last = dates[-2].date()
            metadata = {
                "spy_max_date": str(second_last),  # 1 trading day behind
                "tickers_total": 100
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            with patch('yfinance.download', return_value=mock_data):
                result = validate_cache_metadata(temp_file, grace_period_days=0)
                assert result is False, "Should fail for 1 day behind with grace_period=0"
                print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_two_days_behind_fails(self):
        """Test that 2 trading days behind fails even with grace_period=1."""
        print("\n--- Test: 2 days behind fails ---")
        
        # Create mock SPY data with latest trading day = 2026-01-09 (Friday)
        dates = pd.date_range(end='2026-01-09', periods=5, freq='B')
        mock_data = pd.DataFrame(
            {'Close': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Get the third-to-last trading day
            third_last = dates[-3].date()
            metadata = {
                "spy_max_date": str(third_last),  # 2 trading days behind
                "tickers_total": 100
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            with patch('yfinance.download', return_value=mock_data):
                result = validate_cache_metadata(temp_file, grace_period_days=1)
                assert result is False, "Should fail for 2 days behind with grace_period=1"
                print("✓ PASS")
        finally:
            os.unlink(temp_file)
    
    def test_fallback_to_calendar_days_when_network_unavailable(self):
        """Test fallback to calendar day check when SPY data unavailable."""
        print("\n--- Test: Fallback to calendar day check ---")
        
        # Create metadata with recent date (within 5 calendar days)
        recent_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metadata = {
                "spy_max_date": recent_date,
                "tickers_total": 100
            }
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            # Mock yfinance to return empty data (simulating network failure)
            with patch('yfinance.download', return_value=pd.DataFrame()):
                result = validate_cache_metadata(temp_file, grace_period_days=1)
                assert result is True, "Should pass fallback check for recent date"
                print("✓ PASS")
        finally:
            os.unlink(temp_file)


def run_all_tests():
    """Run all test classes."""
    print("=" * 80)
    print("VALIDATE_CACHE_METADATA UNIT TESTS")
    print("=" * 80)
    
    test_classes = [
        TestIsValidSpyDate(),
        TestFetchSpyLatestTradingDay(),
        TestValidateCacheMetadata()
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
                method()
                passed_tests += 1
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


def test_bootstrap_override():
    """Test ALLOW_METADATA_BOOTSTRAP override functionality."""
    print("\n" + "=" * 80)
    print("TEST: Bootstrap Override (ALLOW_METADATA_BOOTSTRAP)")
    print("=" * 80)
    
    # Create a stale metadata file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        metadata = {
            "spy_max_date": "2026-01-01",  # Intentionally stale
            "tickers_total": 100,
            "generated_at_utc": "2026-01-01T00:00:00Z"
        }
        json.dump(metadata, f)
        temp_path = f.name
    
    try:
        # Test 1: Without override, should fail
        print("\n1. Testing without override (should fail)...")
        os.environ.pop('ALLOW_METADATA_BOOTSTRAP', None)
        
        with patch('validate_cache_metadata.fetch_spy_latest_trading_day') as mock_fetch:
            # Mock a current trading day
            today = date.today()
            mock_fetch.return_value = (today, [today])
            
            result = validate_cache_metadata(temp_path, grace_period_days=1)
            
            if result is False:
                print("   ✓ Validation correctly failed without override")
            else:
                print("   ✗ Validation should have failed but passed")
                return False
        
        # Test 2: With override, should pass with warning
        print("\n2. Testing with override (should pass with warning)...")
        os.environ['ALLOW_METADATA_BOOTSTRAP'] = '1'
        
        with patch('validate_cache_metadata.fetch_spy_latest_trading_day') as mock_fetch:
            # Mock a current trading day
            today = date.today()
            mock_fetch.return_value = (today, [today])
            
            result = validate_cache_metadata(temp_path, grace_period_days=1)
            
            if result is True:
                print("   ✓ Validation correctly passed with override")
            else:
                print("   ✗ Validation should have passed with override but failed")
                return False
        
        # Test 3: Other validations still enforced with override
        print("\n3. Testing that other validations still fail with override...")
        os.environ['ALLOW_METADATA_BOOTSTRAP'] = '1'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            bad_metadata = {
                "spy_max_date": None,  # Missing spy_max_date
                "tickers_total": 100,
                "generated_at_utc": "2026-01-01T00:00:00Z"
            }
            json.dump(bad_metadata, f)
            bad_path = f.name
        
        result = validate_cache_metadata(bad_path, grace_period_days=1)
        
        if result is False:
            print("   ✓ Validation correctly failed on missing spy_max_date even with override")
        else:
            print("   ✗ Validation should have failed on missing spy_max_date")
            return False
        
        os.unlink(bad_path)
        
        print("\n✅ ALL BOOTSTRAP OVERRIDE TESTS PASSED")
        return True
        
    finally:
        os.unlink(temp_path)
        os.environ.pop('ALLOW_METADATA_BOOTSTRAP', None)


if __name__ == '__main__':
    # Run original tests
    result = run_all_tests()
    
    # Run bootstrap override test
    print("\n")
    if test_bootstrap_override():
        print("\n✅ ALL TESTS (INCLUDING BOOTSTRAP OVERRIDE) PASSED")
        sys.exit(result)
    else:
        print("\n❌ BOOTSTRAP OVERRIDE TEST FAILED")
        sys.exit(1)
