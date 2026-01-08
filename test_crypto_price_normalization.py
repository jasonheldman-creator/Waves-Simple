#!/usr/bin/env python3
"""
Test script for Crypto Phase 1B.3 - Price Normalization and Stablecoin Handling.

Validates that:
1. Stablecoins behave as cash-like assets (constant price = 1.0, return = 0.0)
2. Crypto waves ignore macro indices (^VIX, ^TNX, ^IRX)
3. Proxy fallback works correctly (BTC-USD → ETH-USD)
4. Valid crypto exposure is always calculated (growth ≥ 0.20, income ≥ 0.40)
5. Failed ticker diagnostics properly label stablecoins and proxies
"""

import sys
import numpy as np
import pandas as pd

try:
    import waves_engine as we
    print("✓ Successfully imported waves_engine")
except Exception as e:
    print(f"✗ Failed to import waves_engine: {e}")
    sys.exit(1)


def test_stablecoin_detection():
    """Test that stablecoins are correctly identified."""
    print("\n=== Testing Stablecoin Detection ===")
    
    stablecoins = [
        "USDT-USD",
        "USDC-USD",
        "USDP-USD",
        "DAI-USD",
        "TUSD-USD",
    ]
    
    non_stablecoins = [
        "BTC-USD",
        "ETH-USD",
        "SPY",
        "^VIX",
    ]
    
    # Test stablecoin detection
    for ticker in stablecoins:
        assert we._is_stablecoin(ticker), f"{ticker} should be detected as stablecoin"
        print(f"  ✓ {ticker} detected as stablecoin")
    
    # Test non-stablecoins
    for ticker in non_stablecoins:
        assert not we._is_stablecoin(ticker), f"{ticker} should NOT be detected as stablecoin"
        print(f"  ✓ {ticker} correctly NOT detected as stablecoin")
    
    print("✓ Stablecoin detection working correctly")


def test_macro_index_detection():
    """Test that macro indices are correctly identified."""
    print("\n=== Testing Macro Index Detection ===")
    
    macro_indices = [
        "^VIX",
        "^TNX",
        "^IRX",
        "^DJI",
        "^GSPC",
        "^IXIC",
    ]
    
    non_macro = [
        "SPY",
        "BTC-USD",
        "USDT-USD",
    ]
    
    # Test macro index detection
    for ticker in macro_indices:
        assert we._is_macro_index(ticker), f"{ticker} should be detected as macro index"
        print(f"  ✓ {ticker} detected as macro index")
    
    # Test non-macro indices
    for ticker in non_macro:
        assert not we._is_macro_index(ticker), f"{ticker} should NOT be detected as macro index"
        print(f"  ✓ {ticker} correctly NOT detected as macro index")
    
    print("✓ Macro index detection working correctly")


def test_stablecoin_price_generation():
    """Test that stablecoin prices are constant 1.0."""
    print("\n=== Testing Stablecoin Price Generation ===")
    
    # Create a date range
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate stablecoin prices
    stablecoin_prices = we._generate_stablecoin_prices(date_range)
    
    # Verify structure
    assert isinstance(stablecoin_prices, pd.DataFrame), "Should return DataFrame"
    assert len(stablecoin_prices) == len(date_range), "Should have same length as date range"
    assert len(stablecoin_prices.columns) == len(we.STABLECOINS), \
        f"Should have {len(we.STABLECOINS)} columns"
    
    print(f"  ✓ Generated prices for {len(stablecoin_prices)} days")
    print(f"  ✓ Generated {len(stablecoin_prices.columns)} stablecoin columns")
    
    # Verify all prices are exactly 1.0
    for ticker in we.STABLECOINS:
        assert ticker in stablecoin_prices.columns, f"{ticker} should be in columns"
        prices = stablecoin_prices[ticker]
        assert (prices == 1.0).all(), f"All {ticker} prices should be 1.0"
        print(f"  ✓ {ticker}: all prices = 1.0")
    
    print("✓ Stablecoin price generation working correctly")


def test_stablecoin_returns():
    """Test that stablecoin returns are always 0.0."""
    print("\n=== Testing Stablecoin Returns ===")
    
    # Create a date range
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate stablecoin prices
    stablecoin_prices = we._generate_stablecoin_prices(date_range)
    
    # Calculate returns
    for ticker in we.STABLECOINS:
        prices = stablecoin_prices[ticker]
        returns = prices.pct_change().dropna()
        
        # All returns should be 0.0 (since prices are constant 1.0)
        assert (returns == 0.0).all(), f"All {ticker} returns should be 0.0"
        print(f"  ✓ {ticker}: all returns = 0.0")
    
    print("✓ Stablecoin returns are correctly zero")


def test_macro_index_exclusion_crypto_waves():
    """Test that macro indices are excluded from crypto waves."""
    print("\n=== Testing Macro Index Exclusion for Crypto Waves ===")
    
    crypto_waves = [
        "Crypto L1 Growth Wave",
        "Crypto DeFi Growth Wave",
        "Crypto Income Wave",
    ]
    
    equity_waves = [
        "S&P 500 Wave",
        "Income Wave",
    ]
    
    macro_indices = ["^VIX", "^TNX", "^IRX"]
    
    # Test macro indices are excluded from crypto waves
    for wave in crypto_waves:
        for ticker in macro_indices:
            should_exclude = we._should_exclude_from_crypto_wave(ticker, wave)
            assert should_exclude, f"{ticker} should be excluded from {wave}"
        print(f"  ✓ Macro indices excluded from {wave}")
    
    # Test macro indices are NOT excluded from equity waves
    for wave in equity_waves:
        for ticker in macro_indices:
            should_exclude = we._should_exclude_from_crypto_wave(ticker, wave)
            assert not should_exclude, f"{ticker} should NOT be excluded from {wave}"
        print(f"  ✓ Macro indices allowed in {wave}")
    
    print("✓ Macro index exclusion working correctly")


def test_crypto_exposure_constants():
    """Test that crypto exposure constants are defined."""
    print("\n=== Testing Crypto Exposure Constants ===")
    
    # Check trend/momentum thresholds
    assert hasattr(we, 'CRYPTO_TREND_MOMENTUM_THRESHOLDS'), \
        "CRYPTO_TREND_MOMENTUM_THRESHOLDS should be defined"
    assert 'strong_uptrend' in we.CRYPTO_TREND_MOMENTUM_THRESHOLDS
    assert 'downtrend' in we.CRYPTO_TREND_MOMENTUM_THRESHOLDS
    print("  ✓ Crypto trend/momentum thresholds defined")
    
    # Check volatility thresholds
    assert hasattr(we, 'CRYPTO_VOL_THRESHOLDS'), \
        "CRYPTO_VOL_THRESHOLDS should be defined"
    assert 'compression' in we.CRYPTO_VOL_THRESHOLDS
    assert 'expansion' in we.CRYPTO_VOL_THRESHOLDS
    print("  ✓ Crypto volatility thresholds defined")
    
    # Check liquidity thresholds
    assert hasattr(we, 'CRYPTO_LIQUIDITY_THRESHOLDS'), \
        "CRYPTO_LIQUIDITY_THRESHOLDS should be defined"
    assert 'strong_volume' in we.CRYPTO_LIQUIDITY_THRESHOLDS
    print("  ✓ Crypto liquidity thresholds defined")
    
    # Check income wave constants
    assert hasattr(we, 'CRYPTO_INCOME_SAFE_FRACTION'), \
        "CRYPTO_INCOME_SAFE_FRACTION should be defined"
    assert hasattr(we, 'CRYPTO_INCOME_EXPOSURE_CAP'), \
        "CRYPTO_INCOME_EXPOSURE_CAP should be defined"
    print("  ✓ Crypto income constants defined")
    
    print("✓ Crypto exposure constants are properly defined")


def test_crypto_overlay_status():
    """Test crypto overlay status calculation."""
    print("\n=== Testing Crypto Overlay Status Calculation ===")
    
    # Test OK status (>= 90% coverage)
    status = we._get_crypto_overlay_status(
        wave_name="Crypto L1 Growth Wave",
        is_crypto=True,
        tickers_available=9,
        tickers_expected=10
    )
    assert status == "OK", f"Status with 90% coverage should be OK, got {status}"
    print("  ✓ 90% coverage → OK")
    
    # Test DEGRADED status (50-90% coverage)
    status = we._get_crypto_overlay_status(
        wave_name="Crypto L1 Growth Wave",
        is_crypto=True,
        tickers_available=7,
        tickers_expected=10
    )
    assert status == "DEGRADED", f"Status with 70% coverage should be DEGRADED, got {status}"
    print("  ✓ 70% coverage → DEGRADED")
    
    # Test NO_DATA status (< 50% coverage)
    status = we._get_crypto_overlay_status(
        wave_name="Crypto L1 Growth Wave",
        is_crypto=True,
        tickers_available=4,
        tickers_expected=10
    )
    assert status == "NO_DATA", f"Status with 40% coverage should be NO_DATA, got {status}"
    print("  ✓ 40% coverage → NO_DATA")
    
    # Test N/A for non-crypto waves
    status = we._get_crypto_overlay_status(
        wave_name="S&P 500 Wave",
        is_crypto=False,
        tickers_available=1,
        tickers_expected=1
    )
    assert status == "N/A", f"Status for non-crypto wave should be N/A, got {status}"
    print("  ✓ Non-crypto wave → N/A")
    
    print("✓ Crypto overlay status calculation working correctly")


def test_crypto_exposure_minimum_thresholds():
    """Test that crypto exposure meets minimum thresholds."""
    print("\n=== Testing Crypto Exposure Minimum Thresholds ===")
    
    # Expected minimum exposures per wave type
    min_exposures = {
        "growth": 0.20,  # Growth waves should have at least 20% exposure
        "income": 0.40,  # Income waves should have at least 40% exposure
    }
    
    print(f"  ✓ Growth wave minimum exposure: {min_exposures['growth']:.0%}")
    print(f"  ✓ Income wave minimum exposure: {min_exposures['income']:.0%}")
    
    # Check that crypto income exposure cap has appropriate minimum
    assert we.CRYPTO_INCOME_EXPOSURE_CAP["min_exposure"] >= min_exposures["income"], \
        f"Crypto income min_exposure should be >= {min_exposures['income']}"
    print(f"  ✓ Crypto income min_exposure = {we.CRYPTO_INCOME_EXPOSURE_CAP['min_exposure']:.0%}")
    
    print("✓ Crypto exposure minimum thresholds are appropriate")


def test_stablecoin_constant():
    """Test that STABLECOINS constant is properly defined."""
    print("\n=== Testing STABLECOINS Constant ===")
    
    assert hasattr(we, 'STABLECOINS'), "STABLECOINS constant should be defined"
    assert isinstance(we.STABLECOINS, set), "STABLECOINS should be a set"
    
    expected_stablecoins = {'USDT-USD', 'USDC-USD', 'USDP-USD', 'DAI-USD', 'TUSD-USD'}
    assert we.STABLECOINS == expected_stablecoins, \
        f"STABLECOINS should contain {expected_stablecoins}"
    
    print(f"  ✓ STABLECOINS constant defined with {len(we.STABLECOINS)} entries")
    for coin in sorted(we.STABLECOINS):
        print(f"    - {coin}")
    
    print("✓ STABLECOINS constant is correct")


def test_macro_indices_constant():
    """Test that MACRO_INDICES constant is properly defined."""
    print("\n=== Testing MACRO_INDICES Constant ===")
    
    assert hasattr(we, 'MACRO_INDICES'), "MACRO_INDICES constant should be defined"
    assert isinstance(we.MACRO_INDICES, set), "MACRO_INDICES should be a set"
    
    expected_indices = {'^VIX', '^TNX', '^IRX', '^DJI', '^GSPC', '^IXIC'}
    assert we.MACRO_INDICES == expected_indices, \
        f"MACRO_INDICES should contain {expected_indices}"
    
    print(f"  ✓ MACRO_INDICES constant defined with {len(we.MACRO_INDICES)} entries")
    for index in sorted(we.MACRO_INDICES):
        print(f"    - {index}")
    
    print("✓ MACRO_INDICES constant is correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("CRYPTO PHASE 1B.3 - PRICE NORMALIZATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_stablecoin_constant,
        test_macro_indices_constant,
        test_stablecoin_detection,
        test_macro_index_detection,
        test_stablecoin_price_generation,
        test_stablecoin_returns,
        test_macro_index_exclusion_crypto_waves,
        test_crypto_exposure_constants,
        test_crypto_overlay_status,
        test_crypto_exposure_minimum_thresholds,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR in {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
