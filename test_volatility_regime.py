"""
Test suite for compute_volatility_regime_and_exposure function.

This test validates the volatility regime calculation logic for the S&P Flagship Wave,
including VIX proxy selection, rolling median smoothing, and regime mapping.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.wave_performance import compute_volatility_regime_and_exposure


def test_compute_volatility_regime_with_vix():
    """Test volatility regime computation with ^VIX ticker (highest priority)."""
    # Create test price book with ^VIX data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    # VIX values that span all three regimes
    vix_values = [15.0, 16.0, 17.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    
    price_book = pd.DataFrame({
        '^VIX': vix_values,
        'SPY': [400 + i for i in range(10)]  # Dummy SPY data
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    # Assertions
    assert result['success'] is True
    assert result['available'] is True
    assert result['reason'] is None
    assert result['vix_ticker_used'] == '^VIX'
    assert result['last_vix_date'] is not None
    assert result['last_vix_value'] is not None
    assert result['current_regime'] in ['Low Volatility', 'Moderate Volatility', 'High Volatility']
    assert result['current_exposure'] in [0.25, 0.65, 1.00]
    
    # Check data frame structure
    df = result['data_frame']
    assert df is not None
    assert 'date' in df.columns
    assert 'vix_value_used' in df.columns
    assert 'vix_ticker_used' in df.columns
    assert 'regime' in df.columns
    assert 'exposure' in df.columns
    assert len(df) == 10
    
    # Check series
    assert result['exposure_series'] is not None
    assert result['regime_series'] is not None
    assert len(result['exposure_series']) == 10
    assert len(result['regime_series']) == 10


def test_compute_volatility_regime_with_vixy():
    """Test volatility regime computation with VIXY ticker (fallback priority)."""
    # Create test price book with VIXY but no ^VIX
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    price_book = pd.DataFrame({
        'VIXY': [20.0, 21.0, 22.0, 23.0, 24.0],
        'SPY': [400 + i for i in range(5)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    # Should use VIXY since ^VIX is not available
    assert result['success'] is True
    assert result['available'] is True
    assert result['vix_ticker_used'] == 'VIXY'


def test_compute_volatility_regime_with_vxx():
    """Test volatility regime computation with VXX ticker (lowest priority)."""
    # Create test price book with VXX but no ^VIX or VIXY
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    price_book = pd.DataFrame({
        'VXX': [25.0, 26.0, 27.0, 28.0, 29.0],
        'SPY': [400 + i for i in range(5)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    # Should use VXX since ^VIX and VIXY are not available
    assert result['success'] is True
    assert result['available'] is True
    assert result['vix_ticker_used'] == 'VXX'


def test_compute_volatility_regime_priority():
    """Test that ^VIX is preferred over VIXY and VXX when all are available."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    price_book = pd.DataFrame({
        '^VIX': [20.0, 21.0, 22.0, 23.0, 24.0],
        'VIXY': [30.0, 31.0, 32.0, 33.0, 34.0],
        'VXX': [40.0, 41.0, 42.0, 43.0, 44.0],
        'SPY': [400 + i for i in range(5)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    # Should use ^VIX (highest priority)
    assert result['vix_ticker_used'] == '^VIX'
    # Last VIX value should be from ^VIX series (with 3-day median smoothing)
    # The smoothed value should be around 23.0 (median of [22, 23, 24])
    assert 22.0 <= result['last_vix_value'] <= 24.0


def test_compute_volatility_regime_missing_vix_proxy():
    """Test behavior when no VIX proxy is available."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    price_book = pd.DataFrame({
        'SPY': [400 + i for i in range(5)],
        'QQQ': [300 + i for i in range(5)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    # Should return unavailable with reason
    assert result['success'] is False
    assert result['available'] is False
    assert result['reason'] == 'missing_vix_proxy'
    assert result['vix_ticker_used'] is None
    assert result['data_frame'] is None


def test_compute_volatility_regime_mapping():
    """Test that VIX values map correctly to regimes and exposure levels."""
    dates = pd.date_range(start='2024-01-01', periods=6, freq='D')
    
    # Test boundary values for regime mapping
    # Low: < 18, Moderate: 18-25, High: >= 25
    vix_values = [15.0, 18.0, 22.0, 25.0, 30.0, 17.0]  # Mix of all regimes
    
    price_book = pd.DataFrame({
        '^VIX': vix_values,
        'SPY': [400 + i for i in range(6)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book, smooth_window=1)  # No smoothing for direct test
    
    # Check regime mapping (without smoothing)
    assert result['regime_series'].iloc[0] == 'Low Volatility'  # VIX=15
    assert result['regime_series'].iloc[1] == 'Moderate Volatility'  # VIX=18
    assert result['regime_series'].iloc[2] == 'Moderate Volatility'  # VIX=22
    assert result['regime_series'].iloc[3] == 'High Volatility'  # VIX=25
    assert result['regime_series'].iloc[4] == 'High Volatility'  # VIX=30
    
    # Check exposure mapping
    assert result['exposure_series'].iloc[0] == 1.00  # VIX=15
    assert result['exposure_series'].iloc[1] == 0.65  # VIX=18
    assert result['exposure_series'].iloc[2] == 0.65  # VIX=22
    assert result['exposure_series'].iloc[3] == 0.25  # VIX=25
    assert result['exposure_series'].iloc[4] == 0.25  # VIX=30


def test_compute_volatility_regime_smoothing():
    """Test that 3-day rolling median smoothing is applied correctly."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    # Create VIX with spike that should be smoothed
    vix_values = [20.0, 20.0, 50.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]  # Spike at day 3
    
    price_book = pd.DataFrame({
        '^VIX': vix_values,
        'SPY': [400 + i for i in range(10)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book, smooth_window=3)
    
    # Check that smoothing reduced the spike impact
    # Day 3: median([20, 20, 50]) = 20 (not 50)
    smoothed_values = result['vix_series_smoothed']
    assert smoothed_values.iloc[2] == 20.0  # Smoothed spike value
    
    # Raw series should still have the spike
    assert result['vix_series_raw'].iloc[2] == 50.0


def test_compute_volatility_regime_empty_price_book():
    """Test behavior with empty price book."""
    price_book = pd.DataFrame()
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    assert result['success'] is False
    assert result['available'] is False
    assert result['reason'] == 'PRICE_BOOK is empty'


def test_compute_volatility_regime_none_price_book():
    """Test behavior with None price book."""
    result = compute_volatility_regime_and_exposure(None)
    
    assert result['success'] is False
    assert result['available'] is False
    assert result['reason'] == 'PRICE_BOOK is empty'


def test_compute_volatility_regime_data_frame_format():
    """Test that the output data frame has the correct format."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    price_book = pd.DataFrame({
        '^VIX': [20.0, 21.0, 22.0, 23.0, 24.0],
        'SPY': [400 + i for i in range(5)]
    }, index=dates)
    
    result = compute_volatility_regime_and_exposure(price_book)
    
    df = result['data_frame']
    
    # Check columns
    expected_columns = ['date', 'vix_value_used', 'vix_ticker_used', 'regime', 'exposure']
    assert list(df.columns) == expected_columns
    
    # Check data types
    assert df['vix_value_used'].dtype in [np.float64, float]
    assert df['vix_ticker_used'].dtype == object  # string
    assert df['regime'].dtype == object  # string
    assert df['exposure'].dtype in [np.float64, float]
    
    # Check all vix_ticker_used values are the same
    assert (df['vix_ticker_used'] == '^VIX').all()


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Volatility Regime and Exposure Computation")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Test with ^VIX ticker", test_compute_volatility_regime_with_vix),
        ("Test with VIXY ticker", test_compute_volatility_regime_with_vixy),
        ("Test with VXX ticker", test_compute_volatility_regime_with_vxx),
        ("Test VIX priority", test_compute_volatility_regime_priority),
        ("Test missing VIX proxy", test_compute_volatility_regime_missing_vix_proxy),
        ("Test regime mapping", test_compute_volatility_regime_mapping),
        ("Test smoothing", test_compute_volatility_regime_smoothing),
        ("Test empty price book", test_compute_volatility_regime_empty_price_book),
        ("Test None price book", test_compute_volatility_regime_none_price_book),
        ("Test data frame format", test_compute_volatility_regime_data_frame_format),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
