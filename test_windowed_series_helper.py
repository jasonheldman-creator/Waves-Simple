#!/usr/bin/env python3
"""
Test suite for slice_windowed_series helper function.

Tests validate:
1. Correct slicing with sufficient data
2. Unavailable flag when insufficient data
3. Correct start_date and end_date
4. Diagnostic fields (rows_used, available)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_windowed_series_sufficient_data():
    """Test slicing with sufficient data."""
    print("\n=== Test: Windowed Series - Sufficient Data ===")
    
    try:
        # Import directly from module file to avoid streamlit dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wave_performance", 
            "/home/runner/work/Waves-Simple/Waves-Simple/helpers/wave_performance.py"
        )
        wave_performance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wave_performance)
        slice_windowed_series = wave_performance.slice_windowed_series
        
        # Create test data: 100 days of returns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        
        # Request 30-day window
        result = slice_windowed_series(returns, window_days=30)
        
        # Validate
        assert result['available'] == True, "Should be available with 100 days of data"
        assert result['rows_used'] == 30, f"Should use exactly 30 rows, got {result['rows_used']}"
        assert len(result['windowed_series']) == 30, "Windowed series should have 30 rows"
        assert result['start_date'] is not None, "Start date should not be None"
        assert result['end_date'] is not None, "End date should not be None"
        
        # Check dates are correct (last 30 days)
        expected_start = dates[-30].strftime('%Y-%m-%d')
        expected_end = dates[-1].strftime('%Y-%m-%d')
        
        assert result['start_date'] == expected_start, f"Start date mismatch: {result['start_date']} != {expected_start}"
        assert result['end_date'] == expected_end, f"End date mismatch: {result['end_date']} != {expected_end}"
        
        print(f"✓ Window available: {result['available']}")
        print(f"✓ Rows used: {result['rows_used']}/30")
        print(f"✓ Start date: {result['start_date']}")
        print(f"✓ End date: {result['end_date']}")
        print("✅ PASS")
        return True
        
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_windowed_series_insufficient_data():
    """Test slicing with insufficient data."""
    print("\n=== Test: Windowed Series - Insufficient Data ===")
    
    try:
        # Import directly from module file to avoid streamlit dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wave_performance", 
            "/home/runner/work/Waves-Simple/Waves-Simple/helpers/wave_performance.py"
        )
        wave_performance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wave_performance)
        slice_windowed_series = wave_performance.slice_windowed_series
        
        # Create test data: only 20 days of returns
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        returns = pd.Series(np.random.randn(20) * 0.01, index=dates)
        
        # Request 30-day window (more than available)
        result = slice_windowed_series(returns, window_days=30)
        
        # Validate
        assert result['available'] == False, "Should NOT be available with only 20 days of data"
        assert result['rows_used'] == 20, f"Should report 20 available rows, got {result['rows_used']}"
        assert len(result['windowed_series']) == 0, "Windowed series should be empty when unavailable"
        assert result['start_date'] is None, "Start date should be None when unavailable"
        assert result['end_date'] is None, "End date should be None when unavailable"
        
        print(f"✓ Window available: {result['available']}")
        print(f"✓ Rows available: {result['rows_used']} (requested 30)")
        print(f"✓ Start date: {result['start_date']}")
        print(f"✓ End date: {result['end_date']}")
        print("✅ PASS")
        return True
        
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_windowed_series_exact_boundary():
    """Test slicing at exact boundary (data = window)."""
    print("\n=== Test: Windowed Series - Exact Boundary ===")
    
    try:
        # Import directly from module file to avoid streamlit dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wave_performance", 
            "/home/runner/work/Waves-Simple/Waves-Simple/helpers/wave_performance.py"
        )
        wave_performance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wave_performance)
        slice_windowed_series = wave_performance.slice_windowed_series
        
        # Create test data: exactly 30 days of returns
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        returns = pd.Series(np.random.randn(30) * 0.01, index=dates)
        
        # Request 30-day window (exact match)
        result = slice_windowed_series(returns, window_days=30)
        
        # Validate
        assert result['available'] == True, "Should be available with exactly 30 days"
        assert result['rows_used'] == 30, f"Should use exactly 30 rows, got {result['rows_used']}"
        assert len(result['windowed_series']) == 30, "Windowed series should have 30 rows"
        
        print(f"✓ Window available: {result['available']}")
        print(f"✓ Rows used: {result['rows_used']}/30")
        print("✅ PASS")
        return True
        
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_windowed_series_empty_series():
    """Test slicing with empty series."""
    print("\n=== Test: Windowed Series - Empty Series ===")
    
    try:
        # Import directly from module file to avoid streamlit dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wave_performance", 
            "/home/runner/work/Waves-Simple/Waves-Simple/helpers/wave_performance.py"
        )
        wave_performance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wave_performance)
        slice_windowed_series = wave_performance.slice_windowed_series
        
        # Create empty series
        returns = pd.Series(dtype=float)
        
        # Request 30-day window
        result = slice_windowed_series(returns, window_days=30)
        
        # Validate
        assert result['available'] == False, "Should NOT be available with empty series"
        assert result['rows_used'] == 0, f"Should report 0 rows, got {result['rows_used']}"
        
        print(f"✓ Window available: {result['available']}")
        print(f"✓ Rows used: {result['rows_used']}")
        print("✅ PASS")
        return True
        
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("WINDOWED SERIES HELPER FUNCTION TESTS")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(test_windowed_series_sufficient_data())
    results.append(test_windowed_series_insufficient_data())
    results.append(test_windowed_series_exact_boundary())
    results.append(test_windowed_series_empty_series())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        sys.exit(1)
