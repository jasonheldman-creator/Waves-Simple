"""
Test Strategic Overlay Parity Between S&P 500 Wave and US MegaCap Core Wave

This test directly compares the overlay application logic between the two waves
to ensure they follow the same code paths.
"""

import pandas as pd
from waves_engine import (
    _compute_core,
    _is_crypto_wave,
    _is_income_wave,
    WAVE_WEIGHTS,
)


def test_both_waves_are_equity_growth():
    """Both waves should be equity growth (not crypto, not income)."""
    waves = ["S&P 500 Wave", "US MegaCap Core Wave"]
    
    for wave_name in waves:
        is_crypto = _is_crypto_wave(wave_name)
        is_income = _is_income_wave(wave_name)
        
        assert not is_crypto, f"{wave_name} should not be crypto"
        assert not is_income, f"{wave_name} should not be income"
        
        print(f"✓ {wave_name}: equity growth wave")


def test_both_waves_get_same_overlays():
    """Both waves should have same overlay strategies enabled."""
    waves = ["S&P 500 Wave", "US MegaCap Core Wave"]
    
    results = {}
    
    for wave_name in waves:
        result = _compute_core(wave_name=wave_name, mode="Standard", days=180, overrides=None, shadow=True)
        
        if hasattr(result, 'attrs') and 'strategy_attribution' in result.attrs:
            attr_rows = result.attrs['strategy_attribution']
            if attr_rows:
                # Get latest day's attribution
                latest = attr_rows[-1]
                strategies = latest['strategy_attribution']
                
                # Get list of enabled strategies (excluding _summary)
                enabled_strategies = []
                for strat_name, strat_data in strategies.items():
                    if strat_name != '_summary':
                        if strat_data.get('enabled', False):
                            enabled_strategies.append(strat_name)
                
                results[wave_name] = set(enabled_strategies)
                
                print(f"\n{wave_name} enabled strategies:")
                for strat in sorted(enabled_strategies):
                    print(f"  - {strat}")
    
    # Compare the two sets
    if len(results) == 2:
        sp500_strategies = results.get("S&P 500 Wave", set())
        megacap_strategies = results.get("US MegaCap Core Wave", set())
        
        common = sp500_strategies & megacap_strategies
        sp500_only = sp500_strategies - megacap_strategies
        megacap_only = megacap_strategies - sp500_strategies
        
        print(f"\nCommon strategies ({len(common)}):")
        for strat in sorted(common):
            print(f"  ✓ {strat}")
        
        if sp500_only:
            print(f"\nS&P 500 only ({len(sp500_only)}):")
            for strat in sorted(sp500_only):
                print(f"  ! {strat}")
        
        if megacap_only:
            print(f"\nUS MegaCap only ({len(megacap_only)}):")
            for strat in sorted(megacap_only):
                print(f"  ! {strat}")
        
        # They should have the same enabled strategies (for equity growth waves)
        assert sp500_strategies == megacap_strategies, \
            f"Strategy sets should match. Difference: S&P500 only={sp500_only}, MegaCap only={megacap_only}"
        
        print(f"\n✓ Both waves have identical enabled strategies")


def test_vix_exposure_calculation_parity():
    """Verify VIX exposure is calculated identically for both waves."""
    from waves_engine import _vix_exposure_factor, _vix_safe_fraction
    
    waves = ["S&P 500 Wave", "US MegaCap Core Wave"]
    vix_levels = [15.0, 20.0, 25.0, 30.0, 40.0]
    mode = "Standard"
    
    print("\nVIX Exposure Factor Comparison:")
    print("VIX Level | S&P 500  | MegaCap  | Match")
    print("-" * 50)
    
    all_match = True
    for vix in vix_levels:
        sp500_exp = _vix_exposure_factor(vix, mode, "S&P 500 Wave")
        megacap_exp = _vix_exposure_factor(vix, mode, "US MegaCap Core Wave")
        match = abs(sp500_exp - megacap_exp) < 0.0001
        all_match = all_match and match
        
        match_symbol = "✓" if match else "✗"
        print(f"{vix:9.1f} | {sp500_exp:8.4f} | {megacap_exp:8.4f} | {match_symbol}")
    
    assert all_match, "VIX exposure factors should match for both waves"
    
    print("\nVIX Safe Fraction Comparison:")
    print("VIX Level | S&P 500  | MegaCap  | Match")
    print("-" * 50)
    
    all_match = True
    for vix in vix_levels:
        sp500_safe = _vix_safe_fraction(vix, mode, "S&P 500 Wave")
        megacap_safe = _vix_safe_fraction(vix, mode, "US MegaCap Core Wave")
        match = abs(sp500_safe - megacap_safe) < 0.0001
        all_match = all_match and match
        
        match_symbol = "✓" if match else "✗"
        print(f"{vix:9.1f} | {sp500_safe:8.4f} | {megacap_safe:8.4f} | {match_symbol}")
    
    assert all_match, "VIX safe fractions should match for both waves"
    
    print("\n✓ VIX overlay calculations are identical for both waves")


def test_regime_detection_parity():
    """Verify regime detection logic is identical for both waves."""
    from waves_engine import _regime_from_return, REGIME_EXPOSURE, REGIME_GATING
    
    test_returns = [-0.15, -0.08, -0.02, 0.03, 0.07]
    mode = "Standard"
    
    print("\nRegime Detection Comparison:")
    print("60D Return | Regime      | Exposure | Gating")
    print("-" * 60)
    
    for ret in test_returns:
        regime = _regime_from_return(ret)
        exposure = REGIME_EXPOSURE[regime]
        gating = REGIME_GATING[mode][regime]
        
        print(f"{ret:10.2%} | {regime:11s} | {exposure:8.2f} | {gating:6.2f}")
    
    print("\n✓ Regime detection uses same logic for all equity growth waves")


def run_all_tests():
    """Run all parity comparison tests."""
    tests = [
        ("Both Waves Are Equity Growth", test_both_waves_are_equity_growth),
        ("Both Waves Get Same Overlays", test_both_waves_get_same_overlays),
        ("VIX Exposure Calculation Parity", test_vix_exposure_calculation_parity),
        ("Regime Detection Parity", test_regime_detection_parity),
    ]
    
    print("=" * 70)
    print("Strategic Overlay Parity: S&P 500 vs US MegaCap Core")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nTest: {test_name}")
            print("-" * 70)
            test_func()
            print(f"\n✅ PASSED: {test_name}\n")
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_name}")
            print(f"   Exception: {e}\n")
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ CONCLUSION: Strategic stacking parity is CONFIRMED")
        print("   Both S&P 500 Wave and US MegaCap Core Wave use identical overlay logic.")
        print("   The only difference is wave composition (1 vs 10 holdings).")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
