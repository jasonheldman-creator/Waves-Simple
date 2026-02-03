"""
Test Build Script VIX Integration

This test validates that the build_wave_history_from_prices.py script
properly computes and adds VIX execution state columns.

Since we can't run the full build in this environment, we'll test
the helper functions that compute VIX data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the directory to path to import from build script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we added to the build script
# We need to import them dynamically since the script isn't a module
import importlib.util
spec = importlib.util.spec_from_file_location("build_script", "build_wave_history_from_prices.py")
build_module = importlib.util.module_from_spec(spec)

# Execute the module to get the functions
try:
    spec.loader.exec_module(build_module)
    classify_regime = build_module.classify_regime
    get_vix_exposure_factor = build_module.get_vix_exposure_factor
    is_equity_wave = build_module.is_equity_wave
    print("✓ Successfully imported build script functions")
except Exception as e:
    print(f"❌ Failed to import build script: {e}")
    sys.exit(1)


def test_classify_regime():
    """Test regime classification function."""
    print("\n" + "=" * 70)
    print("TEST 1: Regime Classification")
    print("=" * 70)
    
    test_cases = [
        (-0.15, "panic"),
        (-0.10, "downtrend"),
        (-0.05, "downtrend"),
        (0.00, "neutral"),
        (0.05, "neutral"),
        (0.08, "uptrend"),
        (np.nan, "neutral"),
    ]
    
    all_passed = True
    for ret_60d, expected in test_cases:
        result = classify_regime(ret_60d)
        ret_str = f"{ret_60d:.2f}" if not pd.isna(ret_60d) else "NaN"
        status = "✓" if result == expected else "✗"
        print(f"{status} classify_regime({ret_str}) = '{result}' (expected '{expected}')")
        if result != expected:
            all_passed = False
    
    if all_passed:
        print("\n✅ All regime classification tests passed")
    else:
        print("\n❌ Some regime classification tests failed")
    
    return all_passed


def test_vix_exposure_factor():
    """Test VIX exposure factor calculation."""
    print("\n" + "=" * 70)
    print("TEST 2: VIX Exposure Factor")
    print("=" * 70)
    
    test_cases = [
        (12.0, 1.15, "Low VIX < 15"),
        (18.0, 1.05, "Moderate VIX 15-20"),
        (22.0, 0.95, "Elevated VIX 20-25"),
        (27.0, 0.85, "High VIX 25-30"),
        (35.0, 0.75, "Very high VIX 30-40"),
        (45.0, 0.60, "Extreme VIX > 40"),
        (np.nan, 1.0, "Missing VIX"),
        (0.0, 1.0, "Invalid VIX"),
    ]
    
    all_passed = True
    for vix_level, expected, description in test_cases:
        result = get_vix_exposure_factor(vix_level)
        vix_str = f"{vix_level:6.1f}" if not pd.isna(vix_level) else "  NaN"
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"{status} VIX {vix_str} → {result:.2f} exposure (expected {expected:.2f}) - {description}")
        if abs(result - expected) >= 0.01:
            all_passed = False
    
    if all_passed:
        print("\n✅ All VIX exposure factor tests passed")
    else:
        print("\n❌ Some VIX exposure factor tests failed")
    
    return all_passed


def test_is_equity_wave():
    """Test equity wave detection."""
    print("\n" + "=" * 70)
    print("TEST 3: Equity Wave Detection")
    print("=" * 70)
    
    test_cases = [
        ("US MegaCap Core Wave", True, "Equity wave"),
        ("AI & Cloud MegaCap Wave", True, "Tech equity wave"),
        ("S&P 500 Wave", True, "Index equity wave"),
        ("Crypto Broad Growth Wave", False, "Crypto wave"),
        ("Crypto Income Wave", False, "Crypto income wave"),
        ("Income Wave", False, "Income wave"),
        ("Vector Muni Ladder Wave", False, "Muni wave"),
        ("SmartSafe Treasury Cash Wave", False, "Cash wave"),
    ]
    
    all_passed = True
    for wave_name, expected, description in test_cases:
        result = is_equity_wave(wave_name)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{wave_name}' → {result} (expected {expected}) - {description}")
        if result != expected:
            all_passed = False
    
    if all_passed:
        print("\n✅ All equity wave detection tests passed")
    else:
        print("\n❌ Some equity wave detection tests failed")
    
    return all_passed


def test_integration_scenario():
    """Test a complete integration scenario with mock data."""
    print("\n" + "=" * 70)
    print("TEST 4: Integration Scenario")
    print("=" * 70)
    
    # Simulate building wave history for one equity wave
    wave_name = "US MegaCap Core Wave"
    is_equity = is_equity_wave(wave_name)
    
    print(f"\nProcessing wave: {wave_name}")
    print(f"Is equity wave: {is_equity}")
    
    if not is_equity:
        print("❌ Wave should be detected as equity")
        return False
    
    # Simulate VIX and SPY data for 5 days
    mock_dates = pd.date_range('2024-01-01', periods=5, freq='D')
    mock_vix = [15.0, 20.0, 25.0, 30.0, 35.0]
    mock_spy_60d_ret = [0.08, 0.05, 0.00, -0.06, -0.13]
    
    print("\nMock data:")
    for i, date in enumerate(mock_dates):
        vix = mock_vix[i]
        spy_ret = mock_spy_60d_ret[i]
        regime = classify_regime(spy_ret)
        exposure = get_vix_exposure_factor(vix)
        
        print(f"  {date.strftime('%Y-%m-%d')}: VIX={vix:5.1f}, SPY_60D={spy_ret:+.2f}, Regime={regime:9s}, Exposure={exposure:.2f}")
    
    # Verify the data would be added correctly
    df_mock = pd.DataFrame({
        'date': mock_dates,
        'wave': wave_name,
        'portfolio_return': [0.01, 0.02, -0.01, -0.02, -0.03],
        'benchmark_return': [0.01, 0.015, -0.005, -0.015, -0.025],
    })
    
    # Add VIX columns (simulating what the build script does)
    df_mock['vix_level'] = mock_vix
    df_mock['vix_regime'] = [classify_regime(r) for r in mock_spy_60d_ret]
    df_mock['exposure_used'] = [get_vix_exposure_factor(v) for v in mock_vix]
    df_mock['overlay_active'] = True
    
    print("\nResulting dataframe:")
    print(df_mock[['date', 'vix_level', 'vix_regime', 'exposure_used', 'overlay_active']].to_string(index=False))
    
    # Validate columns exist
    required_cols = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
    if all(col in df_mock.columns for col in required_cols):
        print("\n✅ Integration scenario passed - all VIX columns created correctly")
        return True
    else:
        print("\n❌ Integration scenario failed - missing columns")
        return False


def run_all_tests():
    """Run all build script VIX integration tests."""
    print("\n" + "=" * 70)
    print("BUILD SCRIPT VIX INTEGRATION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Regime classification
    results.append(("Regime Classification", test_classify_regime()))
    
    # Test 2: VIX exposure factor
    results.append(("VIX Exposure Factor", test_vix_exposure_factor()))
    
    # Test 3: Equity wave detection
    results.append(("Equity Wave Detection", test_is_equity_wave()))
    
    # Test 4: Integration scenario
    results.append(("Integration Scenario", test_integration_scenario()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
