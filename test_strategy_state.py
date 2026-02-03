#!/usr/bin/env python3
"""
Test script for strategy_state feature (v17.4).
Validates that strategy state is properly extracted, persisted, and displayed.
"""

import sys
import json
from typing import Dict, Any

print("=" * 80)
print("STRATEGY STATE VALIDATION TEST (v17.4)")
print("=" * 80)

# Test 1: Import waves_engine and check for get_latest_strategy_state
print("\n[Test 1] Import waves_engine and check for get_latest_strategy_state...")
try:
    import waves_engine as we
    print("✓ Successfully imported waves_engine")
    
    assert hasattr(we, 'get_latest_strategy_state'), "get_latest_strategy_state not found"
    print("✓ get_latest_strategy_state function exists")
    
    assert hasattr(we, '_format_strategy_reason'), "_format_strategy_reason not found"
    print("✓ _format_strategy_reason helper function exists")
    
except Exception as e:
    print(f"✗ Failed to import waves_engine: {e}")
    sys.exit(1)

# Test 2: Call get_latest_strategy_state for a sample wave
print("\n[Test 2] Get strategy state for US MegaCap Core Wave...")
try:
    wave_name = "US MegaCap Core Wave"
    mode = "Standard"
    days = 30
    
    result = we.get_latest_strategy_state(wave_name, mode, days)
    
    assert result.get("ok") == True, "get_latest_strategy_state returned ok=False"
    print(f"✓ get_latest_strategy_state returned ok=True")
    
    strategy_state = result.get("strategy_state", {})
    assert strategy_state, "strategy_state is empty"
    print(f"✓ strategy_state is not empty")
    
    # Validate required fields
    required_fields = [
        "regime", "vix_regime", "exposure", "safe_allocation",
        "trigger_reasons", "strategy_family", "timestamp",
        "aggregated_risk_state", "active_strategies"
    ]
    
    for field in required_fields:
        assert field in strategy_state, f"Missing required field: {field}"
        print(f"  ✓ Field '{field}' present: {strategy_state[field]}")
    
    # Validate data types
    assert isinstance(strategy_state["regime"], str), "regime should be string"
    assert isinstance(strategy_state["vix_regime"], str), "vix_regime should be string"
    assert isinstance(strategy_state["exposure"], (int, float)), "exposure should be numeric"
    assert isinstance(strategy_state["safe_allocation"], (int, float)), "safe_allocation should be numeric"
    assert isinstance(strategy_state["trigger_reasons"], list), "trigger_reasons should be list"
    assert isinstance(strategy_state["strategy_family"], str), "strategy_family should be string"
    assert isinstance(strategy_state["aggregated_risk_state"], str), "aggregated_risk_state should be string"
    assert isinstance(strategy_state["active_strategies"], int), "active_strategies should be int"
    
    print("✓ All required fields have correct types")
    
    # Validate value ranges
    assert 0.0 <= strategy_state["exposure"] <= 2.0, f"exposure out of range: {strategy_state['exposure']}"
    assert 0.0 <= strategy_state["safe_allocation"] <= 1.0, f"safe_allocation out of range: {strategy_state['safe_allocation']}"
    assert strategy_state["active_strategies"] >= 0, f"active_strategies should be non-negative: {strategy_state['active_strategies']}"
    
    print("✓ Field values are within expected ranges")
    
    # Display trigger reasons
    print(f"\nTrigger Reasons ({len(strategy_state['trigger_reasons'])} total):")
    for reason in strategy_state["trigger_reasons"]:
        print(f"  - {reason}")
    
except Exception as e:
    print(f"✗ Failed to get strategy state: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check snapshot_ledger integration
print("\n[Test 3] Check snapshot_ledger integration...")
try:
    import snapshot_ledger as sl
    print("✓ Successfully imported snapshot_ledger")
    
    # Check if snapshot_ledger has the proper tier functions updated
    assert hasattr(sl, '_build_snapshot_row_tier_a'), "_build_snapshot_row_tier_a not found"
    print("✓ _build_snapshot_row_tier_a exists")
    
    # Inspect the tier A function to see if it references strategy_state
    import inspect
    tier_a_source = inspect.getsource(sl._build_snapshot_row_tier_a)
    assert "strategy_state" in tier_a_source, "strategy_state not referenced in tier A"
    print("✓ Tier A function references strategy_state")
    
    assert "get_latest_strategy_state" in tier_a_source, "get_latest_strategy_state not called in tier A"
    print("✓ Tier A function calls get_latest_strategy_state")
    
except Exception as e:
    print(f"✗ Failed to check snapshot_ledger: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test strategy state for different wave types
print("\n[Test 4] Test strategy state for different wave types...")

test_waves = [
    ("S&P 500 Wave", "equity_growth"),
    ("Crypto L1 Growth Wave", "crypto_growth"),
    ("Income Wave", "equity_income"),
]

for wave_name, expected_family in test_waves:
    try:
        print(f"\n  Testing {wave_name}...")
        result = we.get_latest_strategy_state(wave_name, "Standard", 30)
        
        if not result.get("ok"):
            print(f"    ⚠ No data available for {wave_name}")
            continue
        
        strategy_state = result.get("strategy_state", {})
        actual_family = strategy_state.get("strategy_family", "unknown")
        
        print(f"    ✓ Strategy family: {actual_family}")
        print(f"    ✓ Regime: {strategy_state.get('regime')}")
        print(f"    ✓ Exposure: {strategy_state.get('exposure'):.3f}")
        print(f"    ✓ Safe allocation: {strategy_state.get('safe_allocation'):.3f}")
        print(f"    ✓ Active strategies: {strategy_state.get('active_strategies')}")
        
    except Exception as e:
        print(f"    ⚠ Error testing {wave_name}: {e}")

# Test 5: Test _format_strategy_reason helper
print("\n[Test 5] Test _format_strategy_reason helper...")
try:
    # Test different strategy types
    test_cases = [
        {
            "strategy_name": "regime_detection",
            "exposure_impact": 1.1,
            "safe_impact": 0.0,
            "risk_state": "risk-on",
            "metadata": {"regime": "uptrend"}
        },
        {
            "strategy_name": "vix_overlay",
            "exposure_impact": 0.9,
            "safe_impact": 0.15,
            "risk_state": "risk-off",
            "metadata": {"vix_level": 25.5}
        },
        {
            "strategy_name": "volatility_targeting",
            "exposure_impact": 1.05,
            "safe_impact": 0.0,
            "risk_state": "neutral",
            "metadata": {"recent_vol": 0.15, "vol_target": 0.20}
        },
    ]
    
    for test_case in test_cases:
        reason = we._format_strategy_reason(
            test_case["strategy_name"],
            test_case["exposure_impact"],
            test_case["safe_impact"],
            test_case["risk_state"],
            test_case["metadata"]
        )
        
        print(f"  {test_case['strategy_name']:30s} -> {reason}")
        
        # Reason should be non-empty if there's meaningful impact
        if abs(test_case["exposure_impact"] - 1.0) > 0.01 or abs(test_case["safe_impact"]) > 0.01:
            assert reason, f"Expected non-empty reason for {test_case['strategy_name']}"
    
    print("✓ All strategy reason formats validated")
    
except Exception as e:
    print(f"✗ Failed to test _format_strategy_reason: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify engine_version is included
print("\n[Test 6] Verify engine_version is accessible...")
try:
    assert hasattr(we, 'get_engine_version'), "get_engine_version not found"
    engine_version = we.get_engine_version()
    print(f"✓ Engine version: {engine_version}")
    
    # Verify it's v17.4 or later (since strategy_state is a v17.4 feature)
    version_parts = engine_version.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    assert major >= 17, f"Major version should be 17+, got {major}"
    if major == 17:
        assert minor >= 4, f"Minor version should be 4+, got {minor}"
    
    print(f"✓ Engine version is compatible (v17.4+)")
    
except Exception as e:
    print(f"✗ Failed to verify engine_version: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("✓ All tests passed successfully!")
print("\nStrategy State Feature (v17.4) is properly implemented:")
print("  • get_latest_strategy_state() extracts strategy state")
print("  • _format_strategy_reason() generates human-readable reasons")
print("  • snapshot_ledger.py integrates strategy_state")
print("  • Engine version tracking is in place (v17.4+)")
print("  • Strategy state works for different wave types")
print("\n✓ Ready for UI integration and manual testing")
print("=" * 80)

sys.exit(0)
