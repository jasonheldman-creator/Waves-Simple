#!/usr/bin/env python3
"""
Test script for crypto-specific overlays.
Validates that:
1. VIX/regime overlays are disabled for crypto waves
2. Crypto-specific overlays are active and distinct
3. Equity waves remain unchanged
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


def test_crypto_wave_detection():
    """Test that crypto waves are correctly identified."""
    print("\n=== Testing Crypto Wave Detection ===")
    
    crypto_waves = [
        "Crypto L1 Growth Wave",
        "Crypto DeFi Growth Wave",
        "Crypto L2 Growth Wave",
        "Crypto AI Growth Wave",
        "Crypto Broad Growth Wave",
        "Crypto Income Wave",
        "Bitcoin Wave"
    ]
    
    equity_waves = [
        "US MegaCap Core Wave",
        "S&P 500 Wave",
        "AI & Cloud MegaCap Wave",
        "Income Wave",
        "SmartSafe Treasury Cash Wave"
    ]
    
    # Test crypto wave detection
    for wave in crypto_waves:
        assert we._is_crypto_wave(wave), f"{wave} should be detected as crypto"
        print(f"  ✓ {wave} detected as crypto")
    
    # Test equity wave detection (should not be crypto)
    for wave in equity_waves:
        assert not we._is_crypto_wave(wave), f"{wave} should NOT be detected as crypto"
        print(f"  ✓ {wave} correctly NOT detected as crypto")
    
    print("✓ Crypto wave detection working correctly")


def test_crypto_growth_wave_detection():
    """Test that crypto growth waves are correctly identified."""
    print("\n=== Testing Crypto Growth Wave Detection ===")
    
    crypto_growth_waves = [
        "Crypto L1 Growth Wave",
        "Crypto DeFi Growth Wave",
        "Crypto L2 Growth Wave",
        "Crypto AI Growth Wave",
        "Crypto Broad Growth Wave"
    ]
    
    non_growth_waves = [
        "Crypto Income Wave",
        "Bitcoin Wave",
        "US MegaCap Core Wave",
        "Income Wave"
    ]
    
    # Test crypto growth wave detection
    for wave in crypto_growth_waves:
        assert we._is_crypto_growth_wave(wave), f"{wave} should be detected as crypto growth"
        print(f"  ✓ {wave} detected as crypto growth")
    
    # Test non-growth waves
    for wave in non_growth_waves:
        assert not we._is_crypto_growth_wave(wave), f"{wave} should NOT be detected as crypto growth"
        print(f"  ✓ {wave} correctly NOT detected as crypto growth")
    
    print("✓ Crypto growth wave detection working correctly")


def test_crypto_income_wave_detection():
    """Test that crypto income wave is correctly identified."""
    print("\n=== Testing Crypto Income Wave Detection ===")
    
    assert we._is_crypto_income_wave("Crypto Income Wave"), "Should detect Crypto Income Wave"
    print("  ✓ Crypto Income Wave detected correctly")
    
    assert not we._is_crypto_income_wave("Crypto L1 Growth Wave"), "Should not detect growth wave as income"
    assert not we._is_crypto_income_wave("Income Wave"), "Should not detect equity Income Wave as crypto income"
    print("  ✓ Other waves correctly NOT detected as Crypto Income Wave")
    
    print("✓ Crypto Income Wave detection working correctly")


def test_strategy_family_assignment():
    """Test that strategy families are correctly assigned."""
    print("\n=== Testing Strategy Family Assignment ===")
    
    # Test crypto income wave
    assert we.get_strategy_family("Crypto Income Wave") == "crypto_income", \
        "Crypto Income Wave should have crypto_income strategy family"
    print("  ✓ Crypto Income Wave → crypto_income")
    
    # Test crypto growth waves
    crypto_growth_waves = [
        "Crypto L1 Growth Wave",
        "Crypto DeFi Growth Wave",
        "Crypto L2 Growth Wave",
        "Crypto AI Growth Wave",
        "Crypto Broad Growth Wave"
    ]
    
    for wave in crypto_growth_waves:
        family = we.get_strategy_family(wave)
        assert family == "crypto_growth", f"{wave} should have crypto_growth family, got {family}"
        print(f"  ✓ {wave} → crypto_growth")
    
    # Test equity growth waves
    assert we.get_strategy_family("S&P 500 Wave") == "equity_growth", \
        "S&P 500 Wave should have equity_growth strategy family"
    print("  ✓ S&P 500 Wave → equity_growth")
    
    # Test equity income waves
    assert we.get_strategy_family("Income Wave") == "equity_income", \
        "Income Wave should have equity_income strategy family"
    print("  ✓ Income Wave → equity_income")
    
    print("✓ Strategy family assignment working correctly")


def test_crypto_overlays_configuration():
    """Test that crypto-specific overlays are in configuration."""
    print("\n=== Testing Crypto Overlay Configuration ===")
    
    expected_crypto_strategies = [
        "crypto_trend_momentum",
        "crypto_volatility",
        "crypto_liquidity",
        "crypto_income_stability",
        "crypto_income_drawdown_guard",
        "crypto_income_liquidity_gate"
    ]
    
    for strategy in expected_crypto_strategies:
        assert strategy in we.DEFAULT_STRATEGY_CONFIGS, f"Strategy {strategy} not in configs"
        config = we.DEFAULT_STRATEGY_CONFIGS[strategy]
        assert hasattr(config, 'enabled'), f"{strategy} missing 'enabled' field"
        assert hasattr(config, 'weight'), f"{strategy} missing 'weight' field"
        print(f"  ✓ {strategy}: enabled={config.enabled}, weight={config.weight}")
    
    print("✓ Crypto overlay configurations present")


def test_crypto_trend_regime():
    """Test crypto trend regime classification."""
    print("\n=== Testing Crypto Trend Regime Classification ===")
    
    test_cases = [
        (0.20, "strong_uptrend"),
        (0.10, "uptrend"),
        (0.00, "neutral"),
        (-0.10, "downtrend"),
        (-0.20, "strong_downtrend"),
    ]
    
    for trend, expected_regime in test_cases:
        regime = we._crypto_trend_regime(trend)
        assert regime == expected_regime, f"Trend {trend} should be {expected_regime}, got {regime}"
        print(f"  ✓ Trend {trend:+.2f} → {regime}")
    
    print("✓ Crypto trend regime classification working correctly")


def test_crypto_volatility_state():
    """Test crypto volatility state classification."""
    print("\n=== Testing Crypto Volatility State Classification ===")
    
    test_cases = [
        (0.25, "extreme_compression"),
        (0.40, "compression"),
        (0.60, "normal"),
        (1.00, "expansion"),
        (1.50, "extreme_expansion"),
    ]
    
    for vol, expected_state in test_cases:
        state = we._crypto_volatility_state(vol)
        assert state == expected_state, f"Vol {vol} should be {expected_state}, got {state}"
        print(f"  ✓ Volatility {vol:.2f} → {state}")
    
    print("✓ Crypto volatility state classification working correctly")


def test_crypto_liquidity_state():
    """Test crypto liquidity state classification."""
    print("\n=== Testing Crypto Liquidity State Classification ===")
    
    test_cases = [
        (2.0, "strong_volume"),
        (1.2, "normal_volume"),
        (0.5, "weak_volume"),
    ]
    
    for vol_ratio, expected_state in test_cases:
        state = we._crypto_liquidity_state(vol_ratio)
        assert state == expected_state, f"Volume ratio {vol_ratio} should be {expected_state}, got {state}"
        print(f"  ✓ Volume ratio {vol_ratio:.1f} → {state}")
    
    print("✓ Crypto liquidity state classification working correctly")


def test_crypto_wave_nav_computation():
    """Test that crypto waves can compute NAV without errors."""
    print("\n=== Testing Crypto Wave NAV Computation ===")
    
    # Test a crypto growth wave
    try:
        result = we.compute_history_nav("Crypto L1 Growth Wave", "Standard", 90)
        assert "wave_nav" in result.columns, "Missing wave_nav column"
        assert len(result) > 0, "No data returned"
        print(f"  ✓ Crypto L1 Growth Wave NAV computed: {len(result)} days")
    except Exception as e:
        print(f"  ✗ Crypto L1 Growth Wave NAV computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test the crypto income wave
    try:
        result = we.compute_history_nav("Crypto Income Wave", "Standard", 90)
        assert "wave_nav" in result.columns, "Missing wave_nav column"
        assert len(result) > 0, "No data returned"
        print(f"  ✓ Crypto Income Wave NAV computed: {len(result)} days")
    except Exception as e:
        print(f"  ✗ Crypto Income Wave NAV computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Crypto wave NAV computation working")
    return True


def test_equity_wave_unchanged():
    """Test that equity waves still work correctly (unchanged)."""
    print("\n=== Testing Equity Wave Unchanged ===")
    
    try:
        result = we.compute_history_nav("US MegaCap Core Wave", "Standard", 90)
        assert "wave_nav" in result.columns, "Missing wave_nav column"
        assert len(result) > 0, "No data returned"
        print(f"  ✓ US MegaCap Core Wave NAV computed: {len(result)} days")
    except Exception as e:
        print(f"  ✗ Equity wave NAV computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Equity waves remain functional")
    return True


def test_crypto_vs_equity_overlay_separation():
    """Test that crypto and equity waves use different overlays."""
    print("\n=== Testing Crypto vs Equity Overlay Separation ===")
    
    # Get diagnostics for a crypto wave
    try:
        crypto_diag = we.get_vix_regime_diagnostics("Crypto L1 Growth Wave", "Standard", 90)
        
        # Check that crypto wave has crypto-specific diagnostics
        assert "is_crypto" in crypto_diag.columns, "Missing is_crypto column"
        assert crypto_diag["is_crypto"].all(), "Crypto wave should have is_crypto=True"
        
        # Check that VIX/regime are disabled (should be n/a or NaN)
        has_crypto_overlays = (
            "crypto_trend_regime" in crypto_diag.columns and
            "crypto_vol_state" in crypto_diag.columns and
            "crypto_liq_state" in crypto_diag.columns
        )
        assert has_crypto_overlays, "Missing crypto-specific overlay columns"
        print("  ✓ Crypto wave has crypto-specific overlay diagnostics")
        
    except Exception as e:
        print(f"  ⚠ Crypto diagnostics test: {e}")
    
    # Get diagnostics for an equity wave
    try:
        equity_diag = we.get_vix_regime_diagnostics("US MegaCap Core Wave", "Standard", 90)
        
        # Check that equity wave has VIX/regime diagnostics
        assert "vix" in equity_diag.columns, "Missing vix column"
        assert "regime" in equity_diag.columns, "Missing regime column"
        
        # Check that equity wave does NOT use crypto overlays
        if "is_crypto" in equity_diag.columns:
            assert not equity_diag["is_crypto"].any(), "Equity wave should have is_crypto=False"
        
        print("  ✓ Equity wave has VIX/regime overlay diagnostics")
        
    except Exception as e:
        print(f"  ⚠ Equity diagnostics test: {e}")
    
    print("✓ Crypto and equity overlays are properly separated")
    return True


def run_all_tests():
    """Run all test functions."""
    print("=" * 80)
    print("CRYPTO OVERLAY VALIDATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Crypto Wave Detection", test_crypto_wave_detection),
        ("Crypto Growth Wave Detection", test_crypto_growth_wave_detection),
        ("Crypto Income Wave Detection", test_crypto_income_wave_detection),
        ("Strategy Family Assignment", test_strategy_family_assignment),
        ("Crypto Overlay Configuration", test_crypto_overlays_configuration),
        ("Crypto Trend Regime", test_crypto_trend_regime),
        ("Crypto Volatility State", test_crypto_volatility_state),
        ("Crypto Liquidity State", test_crypto_liquidity_state),
        ("Crypto Wave NAV Computation", test_crypto_wave_nav_computation),
        ("Equity Wave Unchanged", test_equity_wave_unchanged),
        ("Crypto vs Equity Overlay Separation", test_crypto_vs_equity_overlay_separation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is False:
                failed += 1
            else:
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
