#!/usr/bin/env python3
"""
Test suite for crypto volatility overlay module (Phase 1B.2).

Validates:
1. Volatility computation from price data
2. Drawdown calculation over time windows
3. Regime classification logic (LOW/MED/HIGH/CRISIS)
4. Exposure scaling based on regime
5. Integration with cached price data (no live fetch)
6. Edge cases and error handling
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from helpers.crypto_volatility_overlay import (
        compute_crypto_overlay,
        _compute_realized_volatility,
        _compute_max_drawdown,
        _classify_volatility_regime,
        _classify_drawdown_severity,
        _determine_regime,
        VOL_THRESHOLDS,
        DD_THRESHOLDS,
        EXPOSURE_BY_REGIME
    )
    print("✓ Successfully imported crypto_volatility_overlay module")
except Exception as e:
    print(f"✗ Failed to import crypto_volatility_overlay: {e}")
    sys.exit(1)


def create_synthetic_price_series(
    days: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.5,
    trend: float = 0.0,
    seed: int = 42
) -> pd.Series:
    """
    Create synthetic price series for testing.
    
    Args:
        days: Number of days
        start_price: Starting price
        volatility: Annualized volatility
        trend: Annualized drift/trend
        seed: Random seed for reproducibility
        
    Returns:
        Series with DatetimeIndex and prices
    """
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Generate returns using geometric Brownian motion
    daily_vol = volatility / np.sqrt(252)
    daily_trend = trend / 252
    
    returns = np.random.normal(daily_trend, daily_vol, days)
    
    # Convert to prices
    prices = start_price * np.exp(np.cumsum(returns))
    
    return pd.Series(prices, index=dates)


def create_drawdown_price_series(days: int = 100, drawdown_pct: float = -0.30) -> pd.Series:
    """
    Create price series with specific drawdown.
    
    Args:
        days: Number of days
        drawdown_pct: Target drawdown (e.g., -0.30 for -30%)
        
    Returns:
        Series with prices showing drawdown
    """
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    prices = []
    for i in range(days):
        if i < days // 3:
            # Rising phase
            price = 100.0 * (1 + i * 0.01)
        elif i < 2 * days // 3:
            # Drawdown phase
            progress = (i - days // 3) / (days // 3)
            peak = 100.0 * (1 + (days // 3) * 0.01)
            price = peak * (1 + drawdown_pct * progress)
        else:
            # Recovery phase
            trough = 100.0 * (1 + (days // 3) * 0.01) * (1 + drawdown_pct)
            progress = (i - 2 * days // 3) / (days // 3)
            price = trough * (1 + 0.5 * abs(drawdown_pct) * progress)
        prices.append(price)
    
    return pd.Series(prices, index=dates)


def test_volatility_computation():
    """Test realized volatility calculation."""
    print("\n=== Testing Volatility Computation ===")
    
    # Test with low volatility series
    low_vol_prices = create_synthetic_price_series(days=100, volatility=0.20, seed=42)
    vol = _compute_realized_volatility(low_vol_prices, window=30)
    print(f"  Low volatility series: {vol:.2%}")
    assert not np.isnan(vol), "Volatility should not be NaN"
    assert vol > 0, "Volatility should be positive"
    assert vol < 0.50, f"Expected low vol < 50%, got {vol:.2%}"
    
    # Test with high volatility series
    high_vol_prices = create_synthetic_price_series(days=100, volatility=1.50, seed=43)
    vol = _compute_realized_volatility(high_vol_prices, window=30)
    print(f"  High volatility series: {vol:.2%}")
    assert vol > 0.50, f"Expected high vol > 50%, got {vol:.2%}"
    
    # Test with insufficient data
    short_prices = pd.Series([100.0])
    vol = _compute_realized_volatility(short_prices, window=30)
    assert np.isnan(vol), "Should return NaN for insufficient data"
    print(f"  Insufficient data: NaN (correct)")
    
    # Test with None/empty
    vol = _compute_realized_volatility(None, window=30)
    assert np.isnan(vol), "Should return NaN for None input"
    
    vol = _compute_realized_volatility(pd.Series([]), window=30)
    assert np.isnan(vol), "Should return NaN for empty series"
    print(f"  Edge cases: NaN (correct)")
    
    print("✓ Volatility computation tests passed")


def test_drawdown_computation():
    """Test maximum drawdown calculation."""
    print("\n=== Testing Drawdown Computation ===")
    
    # Test with no drawdown (rising prices)
    rising_prices = pd.Series([100, 105, 110, 115, 120])
    dd = _compute_max_drawdown(rising_prices, window=60)
    print(f"  Rising prices: {dd:.2%}")
    assert dd >= -0.01, f"Rising prices should have minimal drawdown, got {dd:.2%}"
    
    # Test with moderate drawdown
    moderate_dd_prices = create_drawdown_price_series(days=100, drawdown_pct=-0.25)
    dd = _compute_max_drawdown(moderate_dd_prices, window=60)
    print(f"  Moderate drawdown series: {dd:.2%}")
    assert dd < -0.15, f"Should detect significant drawdown, got {dd:.2%}"
    assert dd > -0.35, f"Drawdown should be around -25%, got {dd:.2%}"
    
    # Test with severe drawdown
    severe_dd_prices = create_drawdown_price_series(days=100, drawdown_pct=-0.55)
    dd = _compute_max_drawdown(severe_dd_prices, window=60)
    print(f"  Severe drawdown series: {dd:.2%}")
    assert dd < -0.40, f"Should detect severe drawdown, got {dd:.2%}"
    
    # Test with insufficient data
    short_prices = pd.Series([100.0])
    dd = _compute_max_drawdown(short_prices, window=60)
    assert dd == 0.0, "Should return 0 for insufficient data"
    print(f"  Insufficient data: 0.0 (correct)")
    
    print("✓ Drawdown computation tests passed")


def test_volatility_regime_classification():
    """Test volatility regime classification."""
    print("\n=== Testing Volatility Regime Classification ===")
    
    test_cases = [
        (0.30, "low"),
        (0.50, "medium"),
        (0.90, "high"),
        (1.50, "extreme"),
        (np.nan, "medium"),  # Default for invalid
    ]
    
    for vol, expected_regime in test_cases:
        regime = _classify_volatility_regime(vol)
        vol_str = f"{vol:.2%}" if not np.isnan(vol) else "NaN"
        assert regime == expected_regime, f"Vol {vol_str} should be {expected_regime}, got {regime}"
        print(f"  ✓ Vol {vol_str:>6} → {regime}")
    
    print("✓ Volatility regime classification tests passed")


def test_drawdown_severity_classification():
    """Test drawdown severity classification."""
    print("\n=== Testing Drawdown Severity Classification ===")
    
    test_cases = [
        (0.00, "none"),
        (-0.10, "none"),
        (-0.20, "minor"),
        (-0.35, "moderate"),
        (-0.55, "severe"),
        (-0.70, "critical"),
    ]
    
    for dd, expected_severity in test_cases:
        severity = _classify_drawdown_severity(dd)
        assert severity == expected_severity, f"DD {dd:.2%} should be {expected_severity}, got {severity}"
        print(f"  ✓ DD {dd:>6.2%} → {severity}")
    
    print("✓ Drawdown severity classification tests passed")


def test_regime_determination():
    """Test overall regime determination logic."""
    print("\n=== Testing Regime Determination ===")
    
    test_cases = [
        # (vol_regime, dd_severity, expected_regime)
        ("low", "none", "LOW"),
        ("medium", "none", "MED"),
        ("high", "none", "HIGH"),
        ("low", "minor", "LOW"),
        ("medium", "moderate", "HIGH"),
        ("high", "severe", "HIGH"),
        ("extreme", "severe", "CRISIS"),
        ("low", "critical", "CRISIS"),  # Critical DD always triggers crisis
    ]
    
    for vol_regime, dd_severity, expected_regime in test_cases:
        regime = _determine_regime(vol_regime, dd_severity)
        assert regime == expected_regime, \
            f"Vol={vol_regime}, DD={dd_severity} should give {expected_regime}, got {regime}"
        print(f"  ✓ Vol={vol_regime:>8}, DD={dd_severity:>10} → {expected_regime}")
    
    print("✓ Regime determination tests passed")


def test_exposure_by_regime():
    """Test that exposure values are correct for each regime."""
    print("\n=== Testing Exposure by Regime ===")
    
    expected_exposures = {
        "LOW": 1.00,
        "MED": 0.75,
        "HIGH": 0.50,
        "CRISIS": 0.20,
    }
    
    for regime, expected_exposure in expected_exposures.items():
        actual_exposure = EXPOSURE_BY_REGIME[regime]
        assert actual_exposure == expected_exposure, \
            f"Regime {regime} should have exposure {expected_exposure}, got {actual_exposure}"
        print(f"  ✓ {regime:>6} → {actual_exposure:.2%} exposure")
    
    # Verify crisis regime has minimum exposure (≤20%)
    assert EXPOSURE_BY_REGIME["CRISIS"] <= 0.20, "Crisis exposure must be ≤20%"
    print(f"  ✓ Crisis regime has minimum exposure ≤20%")
    
    print("✓ Exposure by regime tests passed")


def test_compute_crypto_overlay_integration():
    """Test the full compute_crypto_overlay function."""
    print("\n=== Testing compute_crypto_overlay Integration ===")
    
    # Create synthetic price data
    btc_prices = create_synthetic_price_series(days=100, volatility=0.60, seed=42)
    eth_prices = create_synthetic_price_series(days=100, volatility=0.70, seed=43)
    
    price_data = pd.DataFrame({
        'BTC-USD': btc_prices,
        'ETH-USD': eth_prices
    })
    
    # Test normal operation
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    # Verify output structure
    assert 'overlay_label' in overlay, "Missing overlay_label"
    assert 'regime' in overlay, "Missing regime"
    assert 'exposure' in overlay, "Missing exposure"
    assert 'volatility' in overlay, "Missing volatility"
    assert 'max_drawdown' in overlay, "Missing max_drawdown"
    assert 'vol_regime' in overlay, "Missing vol_regime"
    assert 'dd_severity' in overlay, "Missing dd_severity"
    
    assert overlay['overlay_label'] == "Crypto Vol", "Incorrect overlay label"
    assert overlay['regime'] in ["LOW", "MED", "HIGH", "CRISIS"], f"Invalid regime: {overlay['regime']}"
    assert 0.2 <= overlay['exposure'] <= 1.0, f"Exposure out of range: {overlay['exposure']}"
    
    print(f"  ✓ Overlay computed successfully:")
    print(f"    - Label: {overlay['overlay_label']}")
    print(f"    - Regime: {overlay['regime']}")
    print(f"    - Exposure: {overlay['exposure']:.2%}")
    print(f"    - Volatility: {overlay['volatility']:.2%}")
    print(f"    - Max Drawdown: {overlay['max_drawdown']:.2%}")
    
    print("✓ Integration test passed")


def test_compute_crypto_overlay_crisis_scenario():
    """Test overlay function with crisis scenario."""
    print("\n=== Testing Crisis Scenario ===")
    
    # Create severe drawdown scenario
    btc_crisis = create_drawdown_price_series(days=100, drawdown_pct=-0.65)
    eth_crisis = create_drawdown_price_series(days=100, drawdown_pct=-0.60)
    
    price_data = pd.DataFrame({
        'BTC-USD': btc_crisis,
        'ETH-USD': eth_crisis
    })
    
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    # In crisis, exposure should be at minimum (0.2)
    print(f"  Crisis scenario detected:")
    print(f"    - Regime: {overlay['regime']}")
    print(f"    - Exposure: {overlay['exposure']:.2%}")
    print(f"    - Max Drawdown: {overlay['max_drawdown']:.2%}")
    
    # With -60%+ drawdown, should trigger CRISIS regime
    assert overlay['regime'] in ["HIGH", "CRISIS"], \
        f"Severe drawdown should trigger HIGH or CRISIS regime, got {overlay['regime']}"
    assert overlay['exposure'] <= 0.50, \
        f"Severe drawdown should reduce exposure to ≤50%, got {overlay['exposure']:.2%}"
    
    print("✓ Crisis scenario test passed")


def test_compute_crypto_overlay_edge_cases():
    """Test overlay function with edge cases."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with empty benchmarks
    overlay = compute_crypto_overlay(
        benchmarks=[],
        price_data=pd.DataFrame(),
        vol_window=30,
        dd_window=60
    )
    assert overlay['regime'] == "MED", "Empty benchmarks should default to MED regime"
    assert overlay['exposure'] == EXPOSURE_BY_REGIME["MED"], "Should use MED exposure"
    print(f"  ✓ Empty benchmarks → MED regime with {overlay['exposure']:.2%} exposure")
    
    # Test with missing benchmark in price data
    price_data = pd.DataFrame({
        'SPY': create_synthetic_price_series(days=100, volatility=0.20)
    })
    
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    # Should handle gracefully and return default
    assert overlay['regime'] in ["LOW", "MED", "HIGH", "CRISIS"], "Should return valid regime"
    print(f"  ✓ Missing benchmarks in data → {overlay['regime']} regime (graceful handling)")
    
    # Test with None price data
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD'],
        price_data=None,
        vol_window=30,
        dd_window=60
    )
    assert overlay['regime'] == "MED", "None price_data should default to MED regime"
    print(f"  ✓ None price_data → MED regime (graceful handling)")
    
    print("✓ Edge case tests passed")


def test_deterministic_behavior():
    """Test that overlay computation is deterministic."""
    print("\n=== Testing Deterministic Behavior ===")
    
    # Create identical price data
    btc_prices = create_synthetic_price_series(days=100, volatility=0.60, seed=42)
    eth_prices = create_synthetic_price_series(days=100, volatility=0.70, seed=43)
    
    price_data = pd.DataFrame({
        'BTC-USD': btc_prices,
        'ETH-USD': eth_prices
    })
    
    # Compute overlay twice
    overlay1 = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    overlay2 = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    # Results should be identical
    assert overlay1['regime'] == overlay2['regime'], "Regime should be deterministic"
    assert overlay1['exposure'] == overlay2['exposure'], "Exposure should be deterministic"
    assert overlay1['volatility'] == overlay2['volatility'], "Volatility should be deterministic"
    assert overlay1['max_drawdown'] == overlay2['max_drawdown'], "Drawdown should be deterministic"
    
    print(f"  ✓ Deterministic behavior verified:")
    print(f"    - Same inputs → same outputs")
    print(f"    - Regime: {overlay1['regime']}")
    print(f"    - Exposure: {overlay1['exposure']:.2%}")
    
    print("✓ Deterministic behavior test passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("CRYPTO VOLATILITY OVERLAY TEST SUITE (Phase 1B.2)")
    print("=" * 70)
    
    tests = [
        ("Volatility Computation", test_volatility_computation),
        ("Drawdown Computation", test_drawdown_computation),
        ("Volatility Regime Classification", test_volatility_regime_classification),
        ("Drawdown Severity Classification", test_drawdown_severity_classification),
        ("Regime Determination", test_regime_determination),
        ("Exposure by Regime", test_exposure_by_regime),
        ("compute_crypto_overlay Integration", test_compute_crypto_overlay_integration),
        ("Crisis Scenario", test_compute_crypto_overlay_crisis_scenario),
        ("Edge Cases", test_compute_crypto_overlay_edge_cases),
        ("Deterministic Behavior", test_deterministic_behavior),
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
    
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
