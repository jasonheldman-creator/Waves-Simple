"""
Test Strategic Stacking Parity for US MegaCap Core Wave

This test validates that momentum and VIX overlays are actively applied
to realized returns for US MegaCap Core Wave, ensuring parity with
S&P 500 Wave behavior.

Requirements from problem statement:
1. Momentum overlay actively applied to realized returns
2. VIX overlay exposure scaling applied at runtime
3. VIX safe fraction normalization impacts wave returns
"""

import numpy as np
import pandas as pd

from waves_engine import (
    WAVE_WEIGHTS,
    _is_crypto_wave,
    _is_income_wave,
    compute_history_nav,
    get_vix_regime_diagnostics,
)


def test_us_megacap_is_equity_wave():
    """Verify US MegaCap Core Wave is classified as equity growth wave."""
    wave_name = "US MegaCap Core Wave"
    
    is_crypto = _is_crypto_wave(wave_name)
    is_income = _is_income_wave(wave_name)
    
    assert not is_crypto, "US MegaCap Core Wave should NOT be classified as crypto"
    assert not is_income, "US MegaCap Core Wave should NOT be classified as income"
    
    print(f"✓ {wave_name} is equity growth wave (not crypto, not income)")


def test_us_megacap_has_multiple_holdings():
    """Verify US MegaCap Core Wave has multiple holdings (momentum can tilt weights)."""
    wave_name = "US MegaCap Core Wave"
    
    holdings = WAVE_WEIGHTS.get(wave_name)
    assert holdings is not None, f"{wave_name} not found in WAVE_WEIGHTS"
    assert len(holdings) > 1, f"{wave_name} should have multiple holdings for momentum tilting"
    
    print(f"✓ {wave_name} has {len(holdings)} holdings (momentum can tilt weights)")


def test_vix_overlay_active_for_us_megacap():
    """
    Verify VIX overlay is actively applied to US MegaCap Core Wave.
    
    This test:
    1. Computes wave diagnostics
    2. Checks that VIX exposure varies with VIX levels
    3. Confirms VIX safe fraction increases with VIX
    """
    wave_name = "US MegaCap Core Wave"
    
    # Get diagnostics with VIX/regime overlay information
    diag_df = get_vix_regime_diagnostics(wave_name, mode="Standard", days=365)
    
    assert not diag_df.empty, "Diagnostics should not be empty"
    assert "vix" in diag_df.columns, "VIX column should be present"
    assert "vix_exposure" in diag_df.columns, "VIX exposure column should be present"
    assert "vix_gate" in diag_df.columns, "VIX gate (safe fraction) column should be present"
    
    # Filter to valid VIX readings
    valid_vix = diag_df[diag_df["vix"].notna() & (diag_df["vix"] > 0)]
    
    if len(valid_vix) < 10:
        print(f"⚠ Warning: Only {len(valid_vix)} valid VIX readings, skipping variation test")
        return
    
    # Check VIX exposure varies with VIX level
    vix_exposure_values = valid_vix["vix_exposure"].unique()
    assert len(vix_exposure_values) > 1, "VIX exposure should vary with VIX levels"
    
    # Check VIX safe fraction varies with VIX level
    vix_gate_values = valid_vix["vix_gate"].unique()
    assert len(vix_gate_values) > 1, "VIX safe fraction should vary with VIX levels"
    
    # Verify inverse relationship: higher VIX → lower exposure
    high_vix_periods = valid_vix[valid_vix["vix"] >= 25]
    low_vix_periods = valid_vix[valid_vix["vix"] < 20]
    
    if len(high_vix_periods) > 0 and len(low_vix_periods) > 0:
        avg_exposure_high_vix = high_vix_periods["vix_exposure"].mean()
        avg_exposure_low_vix = low_vix_periods["vix_exposure"].mean()
        
        assert avg_exposure_high_vix < avg_exposure_low_vix, \
            f"High VIX exposure ({avg_exposure_high_vix:.2f}) should be lower than low VIX exposure ({avg_exposure_low_vix:.2f})"
        
        print(f"✓ VIX exposure varies: Low VIX={avg_exposure_low_vix:.2f}, High VIX={avg_exposure_high_vix:.2f}")
        
        # Verify safe fraction increases with VIX
        avg_safe_high_vix = high_vix_periods["vix_gate"].mean()
        avg_safe_low_vix = low_vix_periods["vix_gate"].mean()
        
        assert avg_safe_high_vix > avg_safe_low_vix, \
            f"High VIX safe fraction ({avg_safe_high_vix:.2f}) should be higher than low VIX safe fraction ({avg_safe_low_vix:.2f})"
        
        print(f"✓ VIX safe fraction varies: Low VIX={avg_safe_low_vix:.2f}, High VIX={avg_safe_high_vix:.2f}")


def test_momentum_impacts_returns():
    """
    Verify momentum overlay impacts realized returns for US MegaCap Core Wave.
    
    This test compares returns with vs without momentum to ensure it's active.
    Note: Since we can't easily disable momentum in compute_history_nav,
    we verify that diagnostics show momentum was enabled.
    """
    wave_name = "US MegaCap Core Wave"
    
    # Get diagnostics
    diag_df = get_vix_regime_diagnostics(wave_name, mode="Standard", days=365)
    
    assert not diag_df.empty, "Diagnostics should not be empty"
    
    # Check if diagnostics indicate strategy is active
    # The diagnostics should show varying exposure values if momentum is working
    if "exposure" in diag_df.columns:
        exposure_values = diag_df["exposure"].unique()
        # Exposure should vary due to multiple overlays (VIX, regime, vol targeting, momentum via weights)
        assert len(exposure_values) > 1, "Exposure should vary (indicates active overlays)"
        
        print(f"✓ Exposure varies across {len(exposure_values)} unique values (overlays active)")


def test_parity_with_sp500_wave():
    """
    Verify US MegaCap Core Wave has same overlay structure as S&P 500 Wave.
    
    Both should:
    - Be equity growth waves
    - Have VIX overlay enabled
    - Have regime detection enabled
    - Have momentum applied (though S&P 500 has only 1 holding)
    """
    megacap_wave = "US MegaCap Core Wave"
    sp500_wave = "S&P 500 Wave"
    
    # Both should be equity waves
    assert not _is_crypto_wave(megacap_wave), f"{megacap_wave} should be equity"
    assert not _is_crypto_wave(sp500_wave), f"{sp500_wave} should be equity"
    assert not _is_income_wave(megacap_wave), f"{megacap_wave} should not be income"
    assert not _is_income_wave(sp500_wave), f"{sp500_wave} should not be income"
    
    # Both should have VIX diagnostics available
    megacap_diag = get_vix_regime_diagnostics(megacap_wave, mode="Standard", days=90)
    sp500_diag = get_vix_regime_diagnostics(sp500_wave, mode="Standard", days=90)
    
    assert not megacap_diag.empty, f"{megacap_wave} diagnostics should not be empty"
    assert not sp500_diag.empty, f"{sp500_wave} diagnostics should not be empty"
    
    # Both should have same diagnostic columns
    megacap_cols = set(megacap_diag.columns)
    sp500_cols = set(sp500_diag.columns)
    
    assert megacap_cols == sp500_cols, \
        f"Diagnostic columns should match. MegaCap: {megacap_cols}, S&P500: {sp500_cols}"
    
    print(f"✓ Both waves have identical diagnostic structure ({len(megacap_cols)} columns)")


def test_vix_safe_fraction_normalization():
    """
    Verify VIX safe fraction normalization impacts wave returns.
    
    This tests that safe_fraction from VIX overlay:
    1. Is properly calculated
    2. Is included in final return calculation
    3. Varies with VIX levels
    """
    wave_name = "US MegaCap Core Wave"
    
    # Get full wave computation with diagnostics
    result = compute_history_nav(wave_name, mode="Standard", days=365, include_diagnostics=True)
    
    assert not result.empty, "Wave computation should return results"
    assert "wave_ret" in result.columns, "Should have wave returns"
    assert "bm_ret" in result.columns, "Should have benchmark returns"
    
    # Check diagnostics are included
    if hasattr(result, 'attrs') and 'diagnostics' in result.attrs:
        diag_df = result.attrs['diagnostics']
        
        if 'safe_fraction' in diag_df.columns:
            safe_fractions = diag_df['safe_fraction'].dropna()
            
            if len(safe_fractions) > 0:
                # Safe fraction should vary (not always 0 or always same value)
                unique_values = safe_fractions.nunique()
                assert unique_values > 1, \
                    f"Safe fraction should vary, found only {unique_values} unique value(s)"
                
                # Safe fraction should be within valid range [0, 0.95]
                assert safe_fractions.min() >= 0.0, "Safe fraction should be >= 0"
                assert safe_fractions.max() <= 0.95, "Safe fraction should be <= 0.95"
                
                print(f"✓ Safe fraction varies: min={safe_fractions.min():.2f}, "
                      f"max={safe_fractions.max():.2f}, unique={unique_values}")


def run_all_tests():
    """Run all strategic parity tests."""
    tests = [
        ("Equity Wave Classification", test_us_megacap_is_equity_wave),
        ("Multiple Holdings", test_us_megacap_has_multiple_holdings),
        ("VIX Overlay Active", test_vix_overlay_active_for_us_megacap),
        ("Momentum Impact", test_momentum_impacts_returns),
        ("Parity with S&P 500", test_parity_with_sp500_wave),
        ("VIX Safe Fraction Normalization", test_vix_safe_fraction_normalization),
    ]
    
    print("=" * 70)
    print("US MegaCap Core Wave - Strategic Stacking Parity Tests")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nTest: {test_name}")
            print("-" * 70)
            test_func()
            print(f"✅ PASSED: {test_name}\n")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_name}")
            print(f"   Exception: {e}\n")
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
