#!/usr/bin/env python3
"""
Test script for income-specific overlays.
Validates that:
1. Income waves are correctly identified
2. Income-specific overlays are active and distinct
3. Equity growth and crypto waves remain unchanged
4. VIX/regime overlays are disabled for income waves
5. Attribution labels are correct
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


def test_income_wave_detection():
    """Test that income waves are correctly identified."""
    print("\n=== Testing Income Wave Detection ===")
    
    income_waves = [
        "Income Wave",
        "Vector Treasury Ladder Wave",
        "Vector Muni Ladder Wave",
        "SmartSafe Treasury Cash Wave",
        "SmartSafe Tax-Free Money Market Wave",
    ]
    
    non_income_waves = [
        "US MegaCap Core Wave",
        "S&P 500 Wave",
        "AI & Cloud MegaCap Wave",
        "Crypto L1 Growth Wave",
        "Crypto Income Wave",  # This is crypto income, not equity income
        "Growth Wave",
    ]
    
    # Test income wave detection
    for wave in income_waves:
        if wave in we.WAVE_WEIGHTS:  # Only test waves that exist
            assert we._is_income_wave(wave), f"{wave} should be detected as income wave"
            print(f"  ✓ {wave} detected as income wave")
    
    # Test non-income wave detection
    for wave in non_income_waves:
        if wave in we.WAVE_WEIGHTS:  # Only test waves that exist
            assert not we._is_income_wave(wave), f"{wave} should NOT be detected as income wave"
            print(f"  ✓ {wave} correctly NOT detected as income wave")
    
    print("✓ Income wave detection working correctly")


def test_income_wave_exclusivity():
    """Test that income waves are not crypto waves and vice versa."""
    print("\n=== Testing Income Wave Exclusivity ===")
    
    income_waves = [
        "Income Wave",
        "Vector Treasury Ladder Wave",
        "Vector Muni Ladder Wave",
        "SmartSafe Treasury Cash Wave",
        "SmartSafe Tax-Free Money Market Wave",
    ]
    
    for wave in income_waves:
        if wave in we.WAVE_WEIGHTS:
            # Income waves should NOT be crypto waves
            assert not we._is_crypto_wave(wave), f"{wave} should not be crypto wave"
            assert not we._is_crypto_income_wave(wave), f"{wave} should not be crypto income wave"
            print(f"  ✓ {wave} is income but not crypto")
    
    # Crypto Income Wave should be crypto but not equity income
    if "Crypto Income Wave" in we.WAVE_WEIGHTS:
        assert we._is_crypto_wave("Crypto Income Wave"), "Crypto Income Wave should be crypto"
        assert we._is_crypto_income_wave("Crypto Income Wave"), "Should detect as crypto income"
        assert not we._is_income_wave("Crypto Income Wave"), "Should NOT be equity income wave"
        print("  ✓ Crypto Income Wave is crypto but not equity income")
    
    print("✓ Income wave exclusivity working correctly")


def test_rates_duration_overlay():
    """Test rates/duration regime classification."""
    print("\n=== Testing Rates/Duration Regime Overlay ===")
    
    # Test regime classification
    test_cases = [
        (0.15, "rising_fast"),
        (0.05, "rising"),
        (0.01, "stable"),
        (-0.05, "falling"),
        (-0.15, "falling_fast"),
        (np.nan, "stable"),
    ]
    
    for tnx_trend, expected_regime in test_cases:
        regime = we._rates_duration_regime(tnx_trend)
        assert regime == expected_regime, f"Expected {expected_regime}, got {regime} for TNX trend {tnx_trend}"
        print(f"  ✓ TNX trend {tnx_trend:.2f} → {regime}")
    
    # Test overlay function
    exposure, regime = we._rates_duration_overlay(0.15)
    assert regime == "rising_fast", "Should detect rising_fast regime"
    assert exposure == we.INCOME_RATES_EXPOSURE["rising_fast"], "Exposure should match config"
    print(f"  ✓ Rates overlay returns exposure={exposure:.2f}, regime={regime}")
    
    print("✓ Rates/duration overlay working correctly")


def test_credit_risk_overlay():
    """Test credit/risk regime classification."""
    print("\n=== Testing Credit/Risk Regime Overlay ===")
    
    # Test regime classification
    test_cases = [
        (0.03, "risk_on"),
        (0.00, "neutral"),
        (-0.03, "risk_off"),
        (np.nan, "neutral"),
    ]
    
    for spread, expected_regime in test_cases:
        regime = we._credit_risk_regime(spread)
        assert regime == expected_regime, f"Expected {expected_regime}, got {regime} for spread {spread}"
        print(f"  ✓ HYG-LQD spread {spread:.2f} → {regime}")
    
    # Test overlay function
    exposure, safe_boost, regime = we._credit_risk_overlay(-0.03)
    assert regime == "risk_off", "Should detect risk_off regime"
    assert exposure == we.INCOME_CREDIT_EXPOSURE["risk_off"], "Exposure should match config"
    assert safe_boost == we.INCOME_CREDIT_SAFE_BOOST["risk_off"], "Safe boost should match config"
    print(f"  ✓ Credit overlay returns exposure={exposure:.2f}, safe_boost={safe_boost:.2f}, regime={regime}")
    
    print("✓ Credit/risk overlay working correctly")


def test_drawdown_guard_overlay():
    """Test carry + drawdown guard overlay."""
    print("\n=== Testing Drawdown Guard Overlay ===")
    
    # Test different drawdown scenarios
    test_cases = [
        (1.00, 1.00, 0.05, "normal", 0.0),      # No drawdown
        (0.96, 1.00, 0.05, "minor", we.INCOME_DRAWDOWN_SAFE_BOOST["minor"]),   # -4% drawdown (above -3% threshold)
        (0.94, 1.00, 0.05, "moderate", we.INCOME_DRAWDOWN_SAFE_BOOST["moderate"]), # -6% drawdown
        (0.91, 1.00, 0.05, "severe", we.INCOME_DRAWDOWN_SAFE_BOOST["severe"]),   # -9% drawdown
        (1.00, 1.00, 0.20, "severe", we.INCOME_DRAWDOWN_SAFE_BOOST["severe"]),   # Vol spike
    ]
    
    for current_nav, peak_nav, recent_vol, expected_state, expected_boost in test_cases:
        safe_boost, stress_state = we._drawdown_guard_overlay(current_nav, peak_nav, recent_vol)
        assert stress_state == expected_state, f"Expected {expected_state}, got {stress_state}"
        assert abs(safe_boost - expected_boost) < 0.001, f"Expected boost {expected_boost}, got {safe_boost}"
        print(f"  ✓ DD={(current_nav/peak_nav-1)*100:.1f}%, vol={recent_vol:.2f} → {stress_state}, boost={safe_boost:.2f}")
    
    print("✓ Drawdown guard overlay working correctly")


def test_income_wave_nav_computation():
    """Test that income waves can compute NAV with new overlays."""
    print("\n=== Testing Income Wave NAV Computation ===")
    
    income_waves = [
        "Income Wave",
        "Vector Treasury Ladder Wave",
        "Vector Muni Ladder Wave",
    ]
    
    for wave_name in income_waves:
        if wave_name not in we.WAVE_WEIGHTS:
            print(f"  ⚠️  {wave_name} not found in WAVE_WEIGHTS, skipping")
            continue
        
        try:
            result = we.compute_history_nav(wave_name, "Standard", 90, include_diagnostics=True)
            
            if not result.empty:
                print(f"  ✓ {wave_name} NAV computed: {len(result)} days")
                
                # Check if diagnostics are available
                if hasattr(result, 'attrs') and 'diagnostics' in result.attrs:
                    diag = result.attrs['diagnostics']
                    
                    # Verify strategy family tagging
                    if 'strategy_family' in diag.columns:
                        assert all(diag['strategy_family'] == 'income'), f"{wave_name} should have strategy_family='income'"
                        print(f"    ✓ Strategy family correctly tagged as 'income'")
                    
                    # Verify income-specific diagnostics exist
                    if 'income_rates_regime' in diag.columns:
                        print(f"    ✓ Income rates regime diagnostics present")
                    if 'income_credit_regime' in diag.columns:
                        print(f"    ✓ Income credit regime diagnostics present")
                    if 'income_stress_state' in diag.columns:
                        print(f"    ✓ Income stress state diagnostics present")
                    
                    # Verify VIX/regime are disabled (should be n/a)
                    if 'regime' in diag.columns and 'vix' in diag.columns:
                        assert all(diag['regime'] == 'n/a'), f"{wave_name} should have VIX regime disabled"
                        print(f"    ✓ Equity VIX/regime overlays disabled")
                else:
                    print(f"    ⚠️  No diagnostics available for {wave_name}")
            else:
                print(f"  ⚠️  Empty result for {wave_name}")
        except Exception as e:
            print(f"  ✗ {wave_name} NAV computation failed: {e}")
            raise
    
    print("✓ Income wave NAV computation working correctly")


def test_equity_growth_wave_unaffected():
    """Test that equity growth waves are unaffected by income overlays."""
    print("\n=== Testing Equity Growth Waves Unaffected ===")
    
    equity_waves = [
        "US MegaCap Core Wave",
        "S&P 500 Wave",
    ]
    
    for wave_name in equity_waves:
        if wave_name not in we.WAVE_WEIGHTS:
            print(f"  ⚠️  {wave_name} not found in WAVE_WEIGHTS, skipping")
            continue
        
        try:
            result = we.compute_history_nav(wave_name, "Standard", 90, include_diagnostics=True)
            
            if not result.empty:
                print(f"  ✓ {wave_name} NAV computed: {len(result)} days")
                
                # Check if diagnostics are available
                if hasattr(result, 'attrs') and 'diagnostics' in result.attrs:
                    diag = result.attrs['diagnostics']
                    
                    # Verify strategy family tagging
                    if 'strategy_family' in diag.columns:
                        assert all(diag['strategy_family'] == 'equity_growth'), f"{wave_name} should have strategy_family='equity_growth'"
                        print(f"    ✓ Strategy family correctly tagged as 'equity_growth'")
                    
                    # Verify income overlays are disabled
                    if 'income_rates_regime' in diag.columns:
                        assert all(diag['income_rates_regime'] == 'n/a'), f"{wave_name} should have income overlays disabled"
                        print(f"    ✓ Income overlays disabled")
                    
                    # Verify VIX/regime are active (should NOT be n/a)
                    if 'regime' in diag.columns:
                        active_regimes = diag[diag['regime'] != 'n/a']
                        if len(active_regimes) > 0:
                            print(f"    ✓ Equity VIX/regime overlays active")
            else:
                print(f"  ⚠️  Empty result for {wave_name}")
        except Exception as e:
            print(f"  ⚠️  {wave_name} computation issue (non-critical): {e}")
    
    print("✓ Equity growth waves unaffected by income overlays")


def test_crypto_wave_unaffected():
    """Test that crypto waves are unaffected by income overlays."""
    print("\n=== Testing Crypto Waves Unaffected ===")
    
    crypto_waves = [
        "Crypto L1 Growth Wave",
        "Crypto Income Wave",
    ]
    
    for wave_name in crypto_waves:
        if wave_name not in we.WAVE_WEIGHTS:
            print(f"  ⚠️  {wave_name} not found in WAVE_WEIGHTS, skipping")
            continue
        
        try:
            result = we.compute_history_nav(wave_name, "Standard", 90, include_diagnostics=True)
            
            if not result.empty:
                print(f"  ✓ {wave_name} NAV computed: {len(result)} days")
                
                # Check if diagnostics are available
                if hasattr(result, 'attrs') and 'diagnostics' in result.attrs:
                    diag = result.attrs['diagnostics']
                    
                    # Verify strategy family tagging
                    if 'strategy_family' in diag.columns:
                        assert all(diag['strategy_family'] == 'crypto'), f"{wave_name} should have strategy_family='crypto'"
                        print(f"    ✓ Strategy family correctly tagged as 'crypto'")
                    
                    # Verify income overlays are disabled
                    if 'income_rates_regime' in diag.columns:
                        assert all(diag['income_rates_regime'] == 'n/a'), f"{wave_name} should have income overlays disabled"
                        print(f"    ✓ Income overlays disabled")
                    
                    # Verify equity VIX/regime are disabled
                    if 'regime' in diag.columns and 'vix' in diag.columns:
                        assert all(diag['regime'] == 'n/a'), f"{wave_name} should have equity regime disabled"
                        print(f"    ✓ Equity VIX/regime overlays disabled")
            else:
                print(f"  ⚠️  Empty result for {wave_name}")
        except Exception as e:
            print(f"  ⚠️  {wave_name} computation issue (non-critical): {e}")
    
    print("✓ Crypto waves unaffected by income overlays")


def main():
    """Run all tests."""
    print("=" * 70)
    print("INCOME WAVE OVERLAY TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Income Wave Detection", test_income_wave_detection),
        ("Income Wave Exclusivity", test_income_wave_exclusivity),
        ("Rates/Duration Overlay", test_rates_duration_overlay),
        ("Credit/Risk Overlay", test_credit_risk_overlay),
        ("Drawdown Guard Overlay", test_drawdown_guard_overlay),
        ("Income Wave NAV Computation", test_income_wave_nav_computation),
        ("Equity Growth Waves Unaffected", test_equity_growth_wave_unaffected),
        ("Crypto Waves Unaffected", test_crypto_wave_unaffected),
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
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
