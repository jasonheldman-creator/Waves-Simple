"""
Integration test for crypto volatility overlay with wave_performance module.

This test validates:
1. Integration of crypto overlay into wave_performance diagnostics
2. Crypto regime computation for crypto waves
3. N/A handling for equity waves
4. Proper isolation between crypto and equity waves
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.wave_performance import (
    compute_crypto_regime_diagnostics,
    compute_all_waves_performance_with_crypto_diagnostics
)


def create_test_price_book() -> pd.DataFrame:
    """Create a test price book with BTC, ETH, and equity tickers."""
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    
    # Simulate crypto prices with some volatility
    np.random.seed(42)
    btc_prices = 40000 * (1 + np.random.normal(0, 0.02, 90)).cumprod()
    eth_prices = 2000 * (1 + np.random.normal(0, 0.02, 90)).cumprod()
    
    # Simulate equity prices with lower volatility
    spy_prices = 400 * (1 + np.random.normal(0, 0.01, 90)).cumprod()
    qqq_prices = 350 * (1 + np.random.normal(0, 0.01, 90)).cumprod()
    
    price_book = pd.DataFrame({
        'BTC-USD': btc_prices,
        'ETH-USD': eth_prices,
        'SPY': spy_prices,
        'QQQ': qqq_prices,
    }, index=dates)
    
    return price_book


def test_compute_crypto_regime_diagnostics_crypto_wave():
    """Test crypto regime diagnostics for a crypto wave."""
    print("\n=== Testing Crypto Regime Diagnostics (Crypto Wave) ===")
    
    price_book = create_test_price_book()
    
    # Test with a crypto growth wave
    diagnostics = compute_crypto_regime_diagnostics(price_book, "Crypto L1 Growth Wave")
    
    # Assertions
    assert diagnostics['applicable'] is True, "Should be applicable for crypto wave"
    assert diagnostics['regime'] in ['calm', 'normal', 'elevated', 'stress'], f"Invalid regime: {diagnostics['regime']}"
    assert 0.0 <= diagnostics['exposure'] <= 1.0, "Exposure should be in [0, 1]"
    assert diagnostics['overlay_status'] != 'Not Applicable', "Should have overlay status"
    
    print(f"  ✓ Regime: {diagnostics['regime']}")
    print(f"  ✓ Exposure: {diagnostics['exposure']:.2f}")
    print(f"  ✓ Overlay Status: {diagnostics['overlay_status']}")
    print(f"  ✓ Data Quality: {diagnostics['data_quality']}")
    
    if diagnostics['combined_vol_ratio'] is not None:
        print(f"  ✓ Combined Vol Ratio: {diagnostics['combined_vol_ratio']:.3f}")
    
    print("✓ Crypto regime diagnostics for crypto wave test passed")


def test_compute_crypto_regime_diagnostics_equity_wave():
    """Test crypto regime diagnostics for an equity wave (should return N/A)."""
    print("\n=== Testing Crypto Regime Diagnostics (Equity Wave) ===")
    
    price_book = create_test_price_book()
    
    # Test with an equity wave
    diagnostics = compute_crypto_regime_diagnostics(price_book, "US MegaCap Core Wave")
    
    # Assertions
    assert diagnostics['applicable'] is False, "Should not be applicable for equity wave"
    assert diagnostics['regime'] == 'N/A', "Regime should be N/A for equity wave"
    assert diagnostics['exposure'] == 1.00, "Exposure should be 1.00 (full) for equity wave"
    assert diagnostics['overlay_status'] == 'Not Applicable', "Overlay status should be Not Applicable"
    
    print("  ✓ Applicable: False (equity wave)")
    print("  ✓ Regime: N/A")
    print("  ✓ Exposure: 1.00 (default)")
    print("  ✓ Overlay Status: Not Applicable")
    
    print("✓ Crypto regime diagnostics for equity wave test passed")


def test_compute_crypto_regime_diagnostics_income_wave():
    """Test crypto regime diagnostics for crypto income wave."""
    print("\n=== Testing Crypto Regime Diagnostics (Crypto Income Wave) ===")
    
    price_book = create_test_price_book()
    
    # Test with crypto income wave
    diagnostics = compute_crypto_regime_diagnostics(price_book, "Crypto Income Wave")
    
    # Assertions
    assert diagnostics['applicable'] is True, "Should be applicable for crypto income wave"
    assert diagnostics['regime'] in ['calm', 'normal', 'elevated', 'stress'], f"Invalid regime: {diagnostics['regime']}"
    assert 0.0 <= diagnostics['exposure'] <= 1.0, "Exposure should be in [0, 1]"
    
    print(f"  ✓ Regime: {diagnostics['regime']}")
    print(f"  ✓ Exposure: {diagnostics['exposure']:.2f} (income strategy)")
    print(f"  ✓ Overlay Status: {diagnostics['overlay_status']}")
    
    print("✓ Crypto regime diagnostics for crypto income wave test passed")


def test_isolation_crypto_vs_equity():
    """Test that crypto and equity waves are properly isolated."""
    print("\n=== Testing Crypto vs Equity Wave Isolation ===")
    
    price_book = create_test_price_book()
    
    # Get diagnostics for crypto wave
    crypto_diag = compute_crypto_regime_diagnostics(price_book, "Crypto L1 Growth Wave")
    
    # Get diagnostics for equity wave
    equity_diag = compute_crypto_regime_diagnostics(price_book, "US MegaCap Core Wave")
    
    # Assertions - they should be completely different
    assert crypto_diag['applicable'] is True, "Crypto wave should use overlay"
    assert equity_diag['applicable'] is False, "Equity wave should NOT use overlay"
    
    assert crypto_diag['regime'] != 'N/A', "Crypto wave should have real regime"
    assert equity_diag['regime'] == 'N/A', "Equity wave should have N/A regime"
    
    print("  ✓ Crypto wave uses crypto overlay")
    print("  ✓ Equity wave does NOT use crypto overlay")
    print("  ✓ Complete isolation verified")
    
    print("✓ Crypto vs equity wave isolation test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 80)
    print("CRYPTO VOLATILITY OVERLAY INTEGRATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Crypto Regime Diagnostics (Crypto Wave)", test_compute_crypto_regime_diagnostics_crypto_wave),
        ("Crypto Regime Diagnostics (Equity Wave)", test_compute_crypto_regime_diagnostics_equity_wave),
        ("Crypto Regime Diagnostics (Crypto Income Wave)", test_compute_crypto_regime_diagnostics_income_wave),
        ("Crypto vs Equity Wave Isolation", test_isolation_crypto_vs_equity),
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
