"""
Test suite for SPY validation in build_price_cache.

Tests the new get_spy_series() function to ensure:
1. SPY presence is validated correctly
2. At least 2 valid entries are required
3. spy_max_date is extracted correctly
4. Proper exception is raised when SPY is missing or invalid
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_price_cache import get_spy_series


def test_spy_series_valid():
    """Test that get_spy_series works with valid SPY data."""
    print("=" * 80)
    print("TEST: Valid SPY Data")
    print("=" * 80)
    
    # Create test DataFrame with valid SPY data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'SPY': [100 + i for i in range(10)],
        'AAPL': [150 + i for i in range(10)],
        'MSFT': [200 + i for i in range(10)]
    }
    cache_df = pd.DataFrame(data, index=dates)
    
    try:
        spy_series, spy_max_date = get_spy_series(cache_df)
        
        # Verify results
        assert len(spy_series) == 10, f"Expected 10 entries, got {len(spy_series)}"
        assert spy_max_date == dates[-1], f"Expected {dates[-1]}, got {spy_max_date}"
        
        print(f"  ✓ SPY series has {len(spy_series)} entries")
        print(f"  ✓ SPY max date: {spy_max_date}")
        print("  ✓ Test passed")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_spy_series_with_nans():
    """Test that get_spy_series handles NaN values correctly."""
    print("\n" + "=" * 80)
    print("TEST: SPY Data with NaN Values")
    print("=" * 80)
    
    # Create test DataFrame with some NaN values in SPY
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'SPY': [100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109],
        'AAPL': [150 + i for i in range(10)]
    }
    cache_df = pd.DataFrame(data, index=dates)
    
    try:
        spy_series, spy_max_date = get_spy_series(cache_df)
        
        # Should have 8 valid entries (10 - 2 NaN)
        assert len(spy_series) == 8, f"Expected 8 entries, got {len(spy_series)}"
        
        # Max date should be the last date (even though it has a valid value)
        assert spy_max_date == dates[-1], f"Expected {dates[-1]}, got {spy_max_date}"
        
        print(f"  ✓ SPY series has {len(spy_series)} valid entries (2 NaN removed)")
        print(f"  ✓ SPY max date: {spy_max_date}")
        print("  ✓ Test passed")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_spy_series_missing():
    """Test that get_spy_series raises exception when SPY is missing."""
    print("\n" + "=" * 80)
    print("TEST: Missing SPY Column")
    print("=" * 80)
    
    # Create test DataFrame without SPY
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'AAPL': [150 + i for i in range(10)],
        'MSFT': [200 + i for i in range(10)]
    }
    cache_df = pd.DataFrame(data, index=dates)
    
    try:
        spy_series, spy_max_date = get_spy_series(cache_df)
        print("  ✗ Test failed: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = "SPY ticker not found"
        if expected_msg in str(e):
            print(f"  ✓ Correctly raised ValueError: {e}")
            print("  ✓ Test passed")
            return True
        else:
            print(f"  ✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ✗ Wrong exception type: {e}")
        return False


def test_spy_series_insufficient_data():
    """Test that get_spy_series raises exception with insufficient SPY data."""
    print("\n" + "=" * 80)
    print("TEST: Insufficient SPY Data (< 2 entries)")
    print("=" * 80)
    
    # Create test DataFrame with only 1 valid SPY entry
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = {
        'SPY': [100, np.nan, np.nan, np.nan, np.nan],
        'AAPL': [150 + i for i in range(5)]
    }
    cache_df = pd.DataFrame(data, index=dates)
    
    try:
        spy_series, spy_max_date = get_spy_series(cache_df)
        print("  ✗ Test failed: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = "insufficient valid data"
        if expected_msg in str(e).lower():
            print(f"  ✓ Correctly raised ValueError: {e}")
            print("  ✓ Test passed")
            return True
        else:
            print(f"  ✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ✗ Wrong exception type: {e}")
        return False


def test_spy_series_case_sensitive():
    """Test that get_spy_series requires exact case-sensitive match."""
    print("\n" + "=" * 80)
    print("TEST: Case-Sensitive SPY Match")
    print("=" * 80)
    
    # Create test DataFrame with lowercase 'spy'
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'spy': [100 + i for i in range(10)],  # lowercase
        'AAPL': [150 + i for i in range(10)]
    }
    cache_df = pd.DataFrame(data, index=dates)
    
    try:
        spy_series, spy_max_date = get_spy_series(cache_df)
        print("  ✗ Test failed: Should have raised ValueError (case-sensitive)")
        return False
    except ValueError as e:
        expected_msg = "SPY ticker not found"
        if expected_msg in str(e):
            print(f"  ✓ Correctly raised ValueError for lowercase 'spy': {e}")
            print("  ✓ Test passed")
            return True
        else:
            print(f"  ✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ✗ Wrong exception type: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SPY VALIDATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_spy_series_valid,
        test_spy_series_with_nans,
        test_spy_series_missing,
        test_spy_series_insufficient_data,
        test_spy_series_case_sensitive
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")
    
    if passed == total:
        print("  ✓ All tests passed")
        return 0
    else:
        print(f"  ✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
