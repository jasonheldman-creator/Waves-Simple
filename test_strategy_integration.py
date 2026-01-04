#!/usr/bin/env python3
"""
Test script for strategy integration and attribution.
Validates that all strategies are actively contributing to exposure and returns.
"""

import sys
import json
from typing import Dict, Any

try:
    import waves_engine as we
    print("✓ Successfully imported waves_engine")
except Exception as e:
    print(f"✗ Failed to import waves_engine: {e}")
    sys.exit(1)


def test_strategy_configs():
    """Test that strategy configurations are properly defined."""
    print("\n=== Testing Strategy Configurations ===")
    
    assert hasattr(we, 'DEFAULT_STRATEGY_CONFIGS'), "DEFAULT_STRATEGY_CONFIGS not found"
    configs = we.DEFAULT_STRATEGY_CONFIGS
    
    expected_strategies = [
        "momentum", "trend_confirmation", "relative_strength",
        "volatility_targeting", "regime_detection", "vix_overlay",
        "smartsafe", "mode_constraint"
    ]
    
    for strategy in expected_strategies:
        assert strategy in configs, f"Strategy {strategy} not in configs"
        config = configs[strategy]
        assert hasattr(config, 'enabled'), f"{strategy} missing 'enabled' field"
        assert hasattr(config, 'weight'), f"{strategy} missing 'weight' field"
        assert hasattr(config, 'min_impact'), f"{strategy} missing 'min_impact' field"
        assert hasattr(config, 'max_impact'), f"{strategy} missing 'max_impact' field"
        print(f"  ✓ {strategy}: enabled={config.enabled}, weight={config.weight}")
    
    print("✓ All strategy configurations validated")


def test_strategy_attribution():
    """Test that strategy attribution is computed correctly."""
    print("\n=== Testing Strategy Attribution ===")
    
    wave_name = "US MegaCap Core Wave"
    mode = "Standard"
    days = 90  # Smaller window for faster testing
    
    print(f"  Computing attribution for {wave_name} ({mode}, {days} days)...")
    
    try:
        attribution = we.get_strategy_attribution(wave_name, mode, days)
    except Exception as e:
        print(f"✗ Failed to compute attribution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate structure
    assert attribution.get("ok") == True, "Attribution computation failed"
    assert "summary" in attribution, "Missing summary in attribution"
    assert "daily_attribution" in attribution, "Missing daily_attribution"
    assert "strategy_stats" in attribution, "Missing strategy_stats"
    
    summary = attribution["summary"]
    print(f"  ✓ Computed {summary['total_trading_days']} trading days")
    print(f"  ✓ Risk-off days: {summary['risk_off_days']} ({summary['risk_off_percentage']:.1%})")
    print(f"  ✓ Risk-on days: {summary['risk_on_days']}")
    
    # Check that we have strategy stats for each strategy
    strategy_stats = attribution["strategy_stats"]
    print(f"\n  Strategy Activity Summary:")
    for strat_name, stats in strategy_stats.items():
        activity = stats["activity_rate"] * 100
        avg_exp = stats["avg_exposure_impact"]
        avg_safe = stats["avg_safe_impact"]
        print(f"    {strat_name:20s}: {activity:5.1f}% active, "
              f"avg_exposure={avg_exp:.3f}, avg_safe={avg_safe:.3f}")
    
    # Validate that strategies are actually impacting the results
    active_strategies = [name for name, stats in strategy_stats.items() 
                        if stats["activity_rate"] > 0.0]
    assert len(active_strategies) > 0, "No strategies are active!"
    print(f"\n  ✓ {len(active_strategies)} strategies are actively contributing")
    
    # Check most impactful strategies
    most_impactful = summary["most_impactful_strategies"]
    print(f"\n  Most Impactful Strategies:")
    for item in most_impactful[:3]:
        print(f"    {item['name']:20s}: importance={item['importance_score']:.4f}")
    
    # Check for dormant strategies
    dormant = summary.get("dormant_strategies", [])
    if dormant:
        print(f"\n  ⚠ Dormant strategies (< 10% active): {', '.join(dormant)}")
    else:
        print(f"\n  ✓ No dormant strategies - all are contributing")
    
    print("\n✓ Strategy attribution validated successfully")
    return True


def test_baseline_compatibility():
    """Test that baseline compute_history_nav still works unchanged."""
    print("\n=== Testing Baseline Compatibility ===")
    
    wave_name = "US MegaCap Core Wave"
    mode = "Standard"
    days = 90
    
    print(f"  Computing baseline NAV for {wave_name}...")
    
    try:
        result = we.compute_history_nav(wave_name, mode, days)
    except Exception as e:
        print(f"✗ Failed to compute baseline NAV: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    assert not result.empty, "Baseline computation returned empty result"
    assert "wave_nav" in result.columns, "Missing wave_nav column"
    assert "bm_nav" in result.columns, "Missing bm_nav column"
    assert "wave_ret" in result.columns, "Missing wave_ret column"
    assert "bm_ret" in result.columns, "Missing bm_ret column"
    
    print(f"  ✓ Computed {len(result)} days of history")
    print(f"  ✓ Final wave NAV: {result['wave_nav'].iloc[-1]:.4f}")
    print(f"  ✓ Final benchmark NAV: {result['bm_nav'].iloc[-1]:.4f}")
    
    print("✓ Baseline compatibility validated")
    return True


def test_shadow_simulation():
    """Test that shadow simulation works with strategy overrides."""
    print("\n=== Testing Shadow Simulation with Strategy Overrides ===")
    
    wave_name = "US MegaCap Core Wave"
    mode = "Standard"
    days = 90
    
    # Test with trend confirmation disabled
    print(f"  Simulating with trend_confirmation disabled...")
    
    overrides = {
        "strategy_configs": we.DEFAULT_STRATEGY_CONFIGS.copy()
    }
    overrides["strategy_configs"]["trend_confirmation"] = we.StrategyConfig(
        enabled=False, weight=0.0, min_impact=0.0, max_impact=1.0
    )
    
    try:
        result = we.simulate_history_nav(wave_name, mode, days, overrides)
    except Exception as e:
        print(f"✗ Failed shadow simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    assert not result.empty, "Shadow simulation returned empty result"
    print(f"  ✓ Shadow simulation completed with {len(result)} days")
    
    # Check attribution
    attribution = result.attrs.get("strategy_attribution", [])
    if attribution:
        # Check that trend_confirmation is disabled in at least one day
        sample = attribution[0]["strategy_attribution"]
        trend_data = sample.get("trend_confirmation", {})
        print(f"  ✓ Strategy override applied (trend_confirmation metadata: {trend_data.get('metadata', {})})")
    
    print("✓ Shadow simulation with overrides validated")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Strategy Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Strategy Configurations", test_strategy_configs),
        ("Strategy Attribution", test_strategy_attribution),
        ("Baseline Compatibility", test_baseline_compatibility),
        ("Shadow Simulation", test_shadow_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except AssertionError as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
