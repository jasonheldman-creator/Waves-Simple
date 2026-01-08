"""
Test suite for crypto volatility overlay module.

This test validates:
1. Volatility ratio computation from BTC/ETH price data
2. Regime classification (calm, normal, elevated, stress)
3. Exposure scaling for growth and income strategies
4. Data safety under edge conditions (missing data, insufficient history)
5. Isolation from non-crypto wave logic
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.crypto_volatility_overlay import (
    compute_volatility_ratio,
    classify_regime,
    compute_exposure_scaling,
    get_crypto_signal_prices,
    compute_crypto_volatility_regime,
    get_crypto_wave_exposure,
    CRYPTO_VOL_SHORT_WINDOW,
    CRYPTO_VOL_LONG_WINDOW,
    MIN_DATA_POINTS,
    DEFAULT_EXPOSURE,
    GROWTH_EXPOSURE_MAP,
    INCOME_EXPOSURE_MAP
)


def create_test_price_series(days: int, volatility: float = 0.02, base_price: float = 40000.0) -> pd.Series:
    """
    Create a synthetic price series with specified volatility.
    
    Args:
        days: Number of days
        volatility: Daily volatility (standard deviation of returns)
        base_price: Starting price
        
    Returns:
        Price series indexed by date
    """
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Generate random returns with specified volatility
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility, days)
    
    # Compute prices from returns
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices, index=dates)


def test_volatility_ratio_computation():
    """Test volatility ratio computation with different market conditions."""
    print("\n=== Testing Volatility Ratio Computation ===")
    
    # Test 1: Normal volatility (ratio ~1.0)
    prices = create_test_price_series(60, volatility=0.02)
    ratio = compute_volatility_ratio(prices, short_window=10, long_window=30)
    assert ratio is not None, "Should compute ratio with sufficient data"
    assert 0.5 <= ratio <= 2.0, f"Ratio should be reasonable, got {ratio}"
    print(f"  ✓ Normal volatility: ratio={ratio:.3f}")
    
    # Test 2: Insufficient data
    short_prices = create_test_price_series(20, volatility=0.02)
    ratio = compute_volatility_ratio(short_prices, short_window=10, long_window=30)
    assert ratio is None, "Should return None with insufficient data"
    print("  ✓ Insufficient data: returns None")
    
    # Test 3: None input
    ratio = compute_volatility_ratio(None)
    assert ratio is None, "Should return None with None input"
    print("  ✓ None input: returns None")
    
    print("✓ Volatility ratio computation tests passed")


def test_regime_classification():
    """Test market regime classification."""
    print("\n=== Testing Regime Classification ===")
    
    test_cases = [
        (0.5, 'calm'),           # Low volatility
        (0.6, 'calm'),           # Low volatility
        (0.7, 'normal'),         # Normal volatility (boundary)
        (1.0, 'normal'),         # Normal volatility
        (1.2, 'normal'),         # Normal volatility
        (1.3, 'elevated'),       # Elevated volatility (boundary)
        (1.5, 'elevated'),       # Elevated volatility
        (1.9, 'elevated'),       # Elevated volatility
        (2.0, 'stress'),         # Stress (boundary)
        (2.5, 'stress'),         # Stress
        (3.0, 'stress'),         # High stress
    ]
    
    for vol_ratio, expected_regime in test_cases:
        regime = classify_regime(vol_ratio)
        assert regime == expected_regime, f"Ratio {vol_ratio} should be {expected_regime}, got {regime}"
        print(f"  ✓ Volatility ratio {vol_ratio:.1f} → {regime}")
    
    # Test edge cases
    regime = classify_regime(None)
    assert regime == 'normal', "None input should default to 'normal'"
    print("  ✓ None input → normal (safe default)")
    
    regime = classify_regime(np.nan)
    assert regime == 'normal', "NaN input should default to 'normal'"
    print("  ✓ NaN input → normal (safe default)")
    
    print("✓ Regime classification tests passed")


def test_exposure_scaling_growth():
    """Test exposure scaling for growth waves."""
    print("\n=== Testing Exposure Scaling (Growth) ===")
    
    test_cases = [
        ('calm', GROWTH_EXPOSURE_MAP['calm']),
        ('normal', GROWTH_EXPOSURE_MAP['normal']),
        ('elevated', GROWTH_EXPOSURE_MAP['elevated']),
        ('stress', GROWTH_EXPOSURE_MAP['stress']),
    ]
    
    for regime, expected_exposure in test_cases:
        exposure = compute_exposure_scaling(regime, is_growth=True)
        assert exposure == expected_exposure, f"Growth exposure for {regime} should be {expected_exposure}, got {exposure}"
        print(f"  ✓ {regime} → {exposure:.2f}")
    
    # Verify progressive reduction
    assert GROWTH_EXPOSURE_MAP['calm'] > GROWTH_EXPOSURE_MAP['normal']
    assert GROWTH_EXPOSURE_MAP['normal'] > GROWTH_EXPOSURE_MAP['elevated']
    assert GROWTH_EXPOSURE_MAP['elevated'] > GROWTH_EXPOSURE_MAP['stress']
    print("  ✓ Progressive exposure reduction verified")
    
    print("✓ Growth exposure scaling tests passed")


def test_exposure_scaling_income():
    """Test exposure scaling for income waves (more conservative)."""
    print("\n=== Testing Exposure Scaling (Income) ===")
    
    test_cases = [
        ('calm', INCOME_EXPOSURE_MAP['calm']),
        ('normal', INCOME_EXPOSURE_MAP['normal']),
        ('elevated', INCOME_EXPOSURE_MAP['elevated']),
        ('stress', INCOME_EXPOSURE_MAP['stress']),
    ]
    
    for regime, expected_exposure in test_cases:
        exposure = compute_exposure_scaling(regime, is_growth=False)
        assert exposure == expected_exposure, f"Income exposure for {regime} should be {expected_exposure}, got {exposure}"
        print(f"  ✓ {regime} → {exposure:.2f}")
    
    # Verify progressive reduction
    assert INCOME_EXPOSURE_MAP['calm'] >= INCOME_EXPOSURE_MAP['normal']
    assert INCOME_EXPOSURE_MAP['normal'] > INCOME_EXPOSURE_MAP['elevated']
    assert INCOME_EXPOSURE_MAP['elevated'] > INCOME_EXPOSURE_MAP['stress']
    print("  ✓ Progressive exposure reduction verified")
    
    # Verify income is more conservative than growth in stress
    assert INCOME_EXPOSURE_MAP['stress'] > GROWTH_EXPOSURE_MAP['stress']
    print("  ✓ Income more conservative than growth in stress")
    
    print("✓ Income exposure scaling tests passed")


def test_get_crypto_signal_prices():
    """Test extraction of BTC and ETH prices from price book."""
    print("\n=== Testing Crypto Signal Price Extraction ===")
    
    # Test 1: Both BTC and ETH available
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    price_book = pd.DataFrame({
        'BTC-USD': np.random.uniform(40000, 45000, 30),
        'ETH-USD': np.random.uniform(2000, 2500, 30),
        'SPY': np.random.uniform(400, 410, 30)
    }, index=dates)
    
    btc_series, eth_series = get_crypto_signal_prices(price_book)
    assert btc_series is not None, "Should find BTC-USD"
    assert eth_series is not None, "Should find ETH-USD"
    assert len(btc_series) == 30, "Should have 30 BTC data points"
    assert len(eth_series) == 30, "Should have 30 ETH data points"
    print("  ✓ Both BTC and ETH found")
    
    # Test 2: Only BTC available
    price_book = pd.DataFrame({
        'BTC-USD': np.random.uniform(40000, 45000, 30),
        'SPY': np.random.uniform(400, 410, 30)
    }, index=dates)
    
    btc_series, eth_series = get_crypto_signal_prices(price_book)
    assert btc_series is not None, "Should find BTC-USD"
    assert eth_series is None, "Should not find ETH-USD"
    print("  ✓ Only BTC found")
    
    # Test 3: Neither available
    price_book = pd.DataFrame({
        'SPY': np.random.uniform(400, 410, 30)
    }, index=dates)
    
    btc_series, eth_series = get_crypto_signal_prices(price_book)
    assert btc_series is None, "Should not find BTC-USD"
    assert eth_series is None, "Should not find ETH-USD"
    print("  ✓ Neither BTC nor ETH found")
    
    print("✓ Crypto signal price extraction tests passed")


def test_compute_crypto_volatility_regime_full_data():
    """Test crypto volatility regime computation with full data."""
    print("\n=== Testing Crypto Volatility Regime (Full Data) ===")
    
    # Create price book with both BTC and ETH
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    price_book = pd.DataFrame({
        'BTC-USD': create_test_price_series(60, volatility=0.02, base_price=40000.0).values,
        'ETH-USD': create_test_price_series(60, volatility=0.02, base_price=2000.0).values,
        'SPY': create_test_price_series(60, volatility=0.01, base_price=400.0).values
    }, index=dates)
    
    result = compute_crypto_volatility_regime(price_book)
    
    # Assertions
    assert result['success'] is True, "Should succeed with full data"
    assert result['available'] is True, "Data should be available"
    assert result['reason'] is None, "No failure reason"
    assert result['regime'] in ['calm', 'normal', 'elevated', 'stress'], f"Invalid regime: {result['regime']}"
    assert 0.0 <= result['growth_exposure'] <= 1.0, "Growth exposure should be in [0, 1]"
    assert 0.0 <= result['income_exposure'] <= 1.0, "Income exposure should be in [0, 1]"
    assert result['data_quality'] == 'good', "Should have good data quality with both signals"
    assert 'BTC' in result['signals_used'], "Should use BTC"
    assert 'ETH' in result['signals_used'], "Should use ETH"
    assert result['btc_vol_ratio'] is not None, "Should compute BTC volatility ratio"
    assert result['eth_vol_ratio'] is not None, "Should compute ETH volatility ratio"
    assert result['combined_vol_ratio'] is not None, "Should compute combined volatility ratio"
    
    print(f"  ✓ Regime: {result['regime']}")
    print(f"  ✓ Growth exposure: {result['growth_exposure']:.2f}")
    print(f"  ✓ Income exposure: {result['income_exposure']:.2f}")
    print(f"  ✓ Data quality: {result['data_quality']}")
    print(f"  ✓ Signals used: {result['signals_used']}")
    
    print("✓ Crypto volatility regime (full data) tests passed")


def test_compute_crypto_volatility_regime_partial_data():
    """Test crypto volatility regime computation with partial data (BTC only)."""
    print("\n=== Testing Crypto Volatility Regime (Partial Data) ===")
    
    # Create price book with only BTC
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    price_book = pd.DataFrame({
        'BTC-USD': create_test_price_series(60, volatility=0.02, base_price=40000.0).values,
        'SPY': create_test_price_series(60, volatility=0.01, base_price=400.0).values
    }, index=dates)
    
    result = compute_crypto_volatility_regime(price_book)
    
    # Assertions
    assert result['success'] is True, "Should succeed with partial data"
    assert result['available'] is True, "Data should be available"
    assert result['regime'] in ['calm', 'normal', 'elevated', 'stress'], f"Invalid regime: {result['regime']}"
    assert result['data_quality'] == 'partial', "Should have partial data quality"
    assert 'BTC' in result['signals_used'], "Should use BTC"
    assert 'ETH' not in result['signals_used'], "Should not use ETH"
    assert result['btc_vol_ratio'] is not None, "Should compute BTC volatility ratio"
    assert result['eth_vol_ratio'] is None, "Should not compute ETH volatility ratio"
    
    print(f"  ✓ Regime: {result['regime']}")
    print(f"  ✓ Data quality: {result['data_quality']}")
    print(f"  ✓ Signals used: {result['signals_used']}")
    
    print("✓ Crypto volatility regime (partial data) tests passed")


def test_compute_crypto_volatility_regime_insufficient_data():
    """Test crypto volatility regime with insufficient data (data safety)."""
    print("\n=== Testing Crypto Volatility Regime (Insufficient Data) ===")
    
    # Create price book with short history
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    price_book = pd.DataFrame({
        'BTC-USD': create_test_price_series(20, volatility=0.02, base_price=40000.0).values,
        'SPY': create_test_price_series(20, volatility=0.01, base_price=400.0).values
    }, index=dates)
    
    result = compute_crypto_volatility_regime(price_book)
    
    # Should return safe defaults
    assert result['success'] is True, "Should succeed with safe defaults"
    assert result['available'] is False, "Data should be marked unavailable"
    assert result['reason'] == 'Insufficient data for volatility computation', "Should report reason"
    assert result['growth_exposure'] == DEFAULT_EXPOSURE, "Should use default growth exposure"
    assert result['income_exposure'] == DEFAULT_EXPOSURE, "Should use default income exposure"
    assert result['regime'] == 'normal', "Should default to normal regime"
    
    print(f"  ✓ Default growth exposure: {result['growth_exposure']:.2f}")
    print(f"  ✓ Default income exposure: {result['income_exposure']:.2f}")
    print(f"  ✓ Reason: {result['reason']}")
    
    print("✓ Crypto volatility regime (insufficient data) tests passed")


def test_compute_crypto_volatility_regime_no_crypto_data():
    """Test crypto volatility regime with no crypto data (data safety)."""
    print("\n=== Testing Crypto Volatility Regime (No Crypto Data) ===")
    
    # Create price book without BTC or ETH
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    price_book = pd.DataFrame({
        'SPY': create_test_price_series(60, volatility=0.01, base_price=400.0).values,
        'QQQ': create_test_price_series(60, volatility=0.01, base_price=350.0).values
    }, index=dates)
    
    result = compute_crypto_volatility_regime(price_book)
    
    # Should return safe defaults
    assert result['success'] is True, "Should succeed with safe defaults"
    assert result['available'] is False, "Data should be marked unavailable"
    assert result['reason'] == 'No BTC or ETH data available', "Should report reason"
    assert result['growth_exposure'] == DEFAULT_EXPOSURE, "Should use default growth exposure"
    assert result['income_exposure'] == DEFAULT_EXPOSURE, "Should use default income exposure"
    
    print(f"  ✓ Default exposure returned: {result['growth_exposure']:.2f}")
    print(f"  ✓ Reason: {result['reason']}")
    
    print("✓ Crypto volatility regime (no crypto data) tests passed")


def test_compute_crypto_volatility_regime_empty_price_book():
    """Test crypto volatility regime with empty price book (data safety)."""
    print("\n=== Testing Crypto Volatility Regime (Empty Price Book) ===")
    
    # Test with empty DataFrame
    price_book = pd.DataFrame()
    result = compute_crypto_volatility_regime(price_book)
    
    assert result['success'] is False, "Should fail with empty price book"
    assert result['reason'] == 'PRICE_BOOK is empty', "Should report empty price book"
    assert result['growth_exposure'] == DEFAULT_EXPOSURE, "Should use default exposure"
    
    # Test with None
    result = compute_crypto_volatility_regime(None)
    assert result['success'] is False, "Should fail with None price book"
    assert result['reason'] == 'PRICE_BOOK is empty', "Should report empty price book"
    
    print("  ✓ Empty DataFrame handled safely")
    print("  ✓ None input handled safely")
    
    print("✓ Crypto volatility regime (empty price book) tests passed")


def test_get_crypto_wave_exposure():
    """Test convenience function for getting wave-specific exposure."""
    print("\n=== Testing Get Crypto Wave Exposure ===")
    
    # Create test price book
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    price_book = pd.DataFrame({
        'BTC-USD': create_test_price_series(60, volatility=0.02, base_price=40000.0).values,
        'ETH-USD': create_test_price_series(60, volatility=0.02, base_price=2000.0).values,
        'SPY': create_test_price_series(60, volatility=0.01, base_price=400.0).values
    }, index=dates)
    
    # Test growth wave
    result = get_crypto_wave_exposure("Crypto L1 Growth Wave", price_book)
    assert result['wave_name'] == "Crypto L1 Growth Wave", "Wave name should match"
    assert result['is_growth'] is True, "Should detect as growth wave"
    assert 0.0 <= result['exposure'] <= 1.0, "Exposure should be in [0, 1]"
    assert result['regime'] in ['calm', 'normal', 'elevated', 'stress'], "Valid regime"
    print(f"  ✓ Growth wave: {result['wave_name']}")
    print(f"    Exposure: {result['exposure']:.2f}, Regime: {result['regime']}")
    
    # Test income wave
    result = get_crypto_wave_exposure("Crypto Income Wave", price_book)
    assert result['wave_name'] == "Crypto Income Wave", "Wave name should match"
    assert result['is_growth'] is False, "Should detect as income wave"
    assert 0.0 <= result['exposure'] <= 1.0, "Exposure should be in [0, 1]"
    print(f"  ✓ Income wave: {result['wave_name']}")
    print(f"    Exposure: {result['exposure']:.2f}, Regime: {result['regime']}")
    
    # Test explicit is_growth parameter
    result = get_crypto_wave_exposure("Test Wave", price_book, is_growth=True)
    assert result['is_growth'] is True, "Should use explicit is_growth"
    print("  ✓ Explicit is_growth parameter works")
    
    print("✓ Get crypto wave exposure tests passed")


def test_exposure_limits():
    """Test that exposure values stay within defined limits."""
    print("\n=== Testing Exposure Limits ===")
    
    # Test all regimes for growth waves
    for regime in ['calm', 'normal', 'elevated', 'stress']:
        exposure = compute_exposure_scaling(regime, is_growth=True)
        assert 0.0 <= exposure <= 1.0, f"Growth exposure for {regime} out of bounds: {exposure}"
        print(f"  ✓ Growth {regime}: {exposure:.2f} in [0.0, 1.0]")
    
    # Test all regimes for income waves
    for regime in ['calm', 'normal', 'elevated', 'stress']:
        exposure = compute_exposure_scaling(regime, is_growth=False)
        assert 0.0 <= exposure <= 1.0, f"Income exposure for {regime} out of bounds: {exposure}"
        print(f"  ✓ Income {regime}: {exposure:.2f} in [0.0, 1.0]")
    
    print("✓ Exposure limits tests passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 80)
    print("CRYPTO VOLATILITY OVERLAY TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Volatility Ratio Computation", test_volatility_ratio_computation),
        ("Regime Classification", test_regime_classification),
        ("Exposure Scaling (Growth)", test_exposure_scaling_growth),
        ("Exposure Scaling (Income)", test_exposure_scaling_income),
        ("Crypto Signal Price Extraction", test_get_crypto_signal_prices),
        ("Crypto Volatility Regime (Full Data)", test_compute_crypto_volatility_regime_full_data),
        ("Crypto Volatility Regime (Partial Data)", test_compute_crypto_volatility_regime_partial_data),
        ("Crypto Volatility Regime (Insufficient Data)", test_compute_crypto_volatility_regime_insufficient_data),
        ("Crypto Volatility Regime (No Crypto Data)", test_compute_crypto_volatility_regime_no_crypto_data),
        ("Crypto Volatility Regime (Empty Price Book)", test_compute_crypto_volatility_regime_empty_price_book),
        ("Get Crypto Wave Exposure", test_get_crypto_wave_exposure),
        ("Exposure Limits", test_exposure_limits),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
