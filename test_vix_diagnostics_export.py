#!/usr/bin/env python3
"""
Test VIX diagnostics export functionality.

This test verifies that:
1. VIX_Level is extracted from strategy_state
2. VIX_Regime is extracted from strategy_state
3. VIX_Adjustment_Pct is parsed from trigger_reasons patterns
"""

import sys
from snapshot_ledger import _extract_vix_diagnostics_from_strategy_state

def test_vix_diagnostics_extraction():
    """Test VIX diagnostics extraction from strategy_state."""
    
    print("=" * 80)
    print("TEST: VIX Diagnostics Extraction")
    print("=" * 80)
    
    # Test Case 1: Full strategy state with VIX data (equity wave)
    print("\n[Test 1] Equity wave with VIX overlay adjustment")
    strategy_state_equity = {
        'vix_level': 18.5,
        'vix_regime': 'normal',
        'trigger_reasons': [
            'Uptrend regime (+10% exposure)',
            'vix_overlay: -5% exposure'  # Should parse to -0.05
        ]
    }
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_equity)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == 18.5, f"Expected 18.5, got {vix_level}"
    assert vix_regime == 'normal', f"Expected 'normal', got {vix_regime}"
    assert vix_adj == -0.05, f"Expected -0.05, got {vix_adj}"
    print("  ✓ PASSED")
    
    # Test Case 2: Crypto wave (VIX should be n/a)
    print("\n[Test 2] Crypto wave (no VIX overlay)")
    strategy_state_crypto = {
        'vix_level': None,
        'vix_regime': 'n/a (crypto)',
        'trigger_reasons': [
            'Crypto uptrend (+15% exposure)'
        ]
    }
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_crypto)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == "", f"Expected blank, got {vix_level}"
    assert vix_regime == 'n/a (crypto)', f"Expected 'n/a (crypto)', got {vix_regime}"
    assert vix_adj == "", f"Expected blank, got {vix_adj}"
    print("  ✓ PASSED")
    
    # Test Case 3: Elevated VIX with positive adjustment
    print("\n[Test 3] Elevated VIX with +3% adjustment")
    strategy_state_elevated = {
        'vix_level': 25.2,
        'vix_regime': 'elevated',
        'trigger_reasons': [
            'Elevated VIX (25.2): reduced exposure, +30% cash',
            'vix_overlay: +3% exposure'  # Should parse to +0.03
        ]
    }
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_elevated)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == 25.2, f"Expected 25.2, got {vix_level}"
    assert vix_regime == 'elevated', f"Expected 'elevated', got {vix_regime}"
    assert vix_adj == 0.03, f"Expected 0.03, got {vix_adj}"
    print("  ✓ PASSED")
    
    # Test Case 4: No VIX adjustment in trigger_reasons
    print("\n[Test 4] No VIX adjustment in trigger_reasons")
    strategy_state_no_adj = {
        'vix_level': 15.0,
        'vix_regime': 'low',
        'trigger_reasons': [
            'Uptrend regime (+10% exposure)',
            'Momentum strong'
        ]
    }
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_no_adj)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == 15.0, f"Expected 15.0, got {vix_level}"
    assert vix_regime == 'low', f"Expected 'low', got {vix_regime}"
    assert vix_adj == "", f"Expected blank, got {vix_adj}"
    print("  ✓ PASSED")
    
    # Test Case 5: Empty strategy_state
    print("\n[Test 5] Empty strategy_state")
    strategy_state_empty = {}
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_empty)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == "", f"Expected blank, got {vix_level}"
    assert vix_regime == 'unknown', f"Expected 'unknown', got {vix_regime}"
    assert vix_adj == "", f"Expected blank, got {vix_adj}"
    print("  ✓ PASSED")
    
    # Test Case 6: Decimal percentage
    print("\n[Test 6] Decimal percentage (-2.5%)")
    strategy_state_decimal = {
        'vix_level': 22.0,
        'vix_regime': 'elevated',
        'trigger_reasons': [
            'vix_overlay: -2.5% exposure'
        ]
    }
    
    vix_level, vix_regime, vix_adj = _extract_vix_diagnostics_from_strategy_state(strategy_state_decimal)
    print(f"  VIX_Level: {vix_level}")
    print(f"  VIX_Regime: {vix_regime}")
    print(f"  VIX_Adjustment_Pct: {vix_adj}")
    
    assert vix_level == 22.0, f"Expected 22.0, got {vix_level}"
    assert vix_regime == 'elevated', f"Expected 'elevated', got {vix_regime}"
    assert vix_adj == -0.025, f"Expected -0.025, got {vix_adj}"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    return 0

if __name__ == '__main__':
    sys.exit(test_vix_diagnostics_extraction())
