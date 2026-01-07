"""
Test VIX Execution State Integration

This test validates that:
1. build_wave_history_from_prices.py can generate VIX execution state columns
2. wave_history.csv includes the required VIX columns
3. VIX data is properly computed for equity waves
4. Non-equity waves have disabled VIX overlay
"""

import pandas as pd
import numpy as np
import os
import sys


def test_vix_columns_structure():
    """Test that wave_history.csv has the required VIX execution state columns."""
    print("=" * 70)
    print("TEST 1: VIX Execution State Column Structure")
    print("=" * 70)
    
    wave_history_path = 'wave_history.csv'
    
    if not os.path.exists(wave_history_path):
        print(f"⚠️  wave_history.csv not found. Run build_wave_history_from_prices.py first.")
        return False
    
    df = pd.read_csv(wave_history_path)
    print(f"\n✓ Loaded wave_history.csv: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for required VIX columns
    required_vix_cols = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
    
    missing_cols = [col for col in required_vix_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠️  Missing VIX columns: {missing_cols}")
        print("  These columns will be added when you rebuild wave_history.csv")
        return False
    
    print(f"\n✓ All required VIX columns present: {required_vix_cols}")
    return True


def test_equity_wave_vix_data():
    """Test that equity waves have VIX execution state data."""
    print("\n" + "=" * 70)
    print("TEST 2: Equity Wave VIX Data")
    print("=" * 70)
    
    wave_history_path = 'wave_history.csv'
    
    if not os.path.exists(wave_history_path):
        print(f"⚠️  wave_history.csv not found.")
        return False
    
    df = pd.read_csv(wave_history_path)
    
    required_vix_cols = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
    if not all(col in df.columns for col in required_vix_cols):
        print(f"⚠️  VIX columns not present. Skipping test.")
        return False
    
    # Look for equity waves (those with overlay_active=True)
    equity_waves = df[df['overlay_active'] == True]
    
    if len(equity_waves) == 0:
        print("⚠️  No equity waves found with active VIX overlay.")
        print("  This may be expected if VIX/SPY data is not available in prices.csv")
        return False
    
    print(f"\n✓ Found {len(equity_waves)} rows with active VIX overlay")
    
    # Check for valid VIX data
    valid_vix = equity_waves[equity_waves['vix_level'].notna()]
    print(f"✓ {len(valid_vix)} rows have valid VIX levels")
    
    # Show sample VIX data
    sample = equity_waves.head(5)[['date', 'wave', 'vix_level', 'vix_regime', 'exposure_used', 'overlay_active']]
    print("\nSample VIX execution state data:")
    print(sample.to_string(index=False))
    
    # Validate regime values
    valid_regimes = ['panic', 'downtrend', 'neutral', 'uptrend']
    unique_regimes = equity_waves['vix_regime'].unique()
    print(f"\n✓ VIX regimes found: {list(unique_regimes)}")
    
    for regime in unique_regimes:
        if regime not in valid_regimes:
            print(f"⚠️  Unexpected regime value: {regime}")
            return False
    
    # Validate exposure_used is in reasonable range
    exposure_min = equity_waves['exposure_used'].min()
    exposure_max = equity_waves['exposure_used'].max()
    print(f"✓ Exposure range: {exposure_min:.2f} to {exposure_max:.2f}")
    
    if exposure_min < 0.5 or exposure_max > 1.5:
        print(f"⚠️  Exposure values outside expected range [0.5, 1.5]")
        return False
    
    print("\n✅ Equity wave VIX data validation passed")
    return True


def test_non_equity_wave_vix_disabled():
    """Test that non-equity waves have VIX overlay disabled."""
    print("\n" + "=" * 70)
    print("TEST 3: Non-Equity Wave VIX Disabled")
    print("=" * 70)
    
    wave_history_path = 'wave_history.csv'
    
    if not os.path.exists(wave_history_path):
        print(f"⚠️  wave_history.csv not found.")
        return False
    
    df = pd.read_csv(wave_history_path)
    
    required_vix_cols = ['overlay_active']
    if not all(col in df.columns for col in required_vix_cols):
        print(f"⚠️  VIX columns not present. Skipping test.")
        return False
    
    # Look for non-equity waves (crypto, income, cash)
    non_equity_keywords = ['Crypto', 'Income', 'Muni', 'Treasury', 'SmartSafe']
    
    non_equity_waves = df[df['wave'].str.contains('|'.join(non_equity_keywords), na=False)]
    
    if len(non_equity_waves) == 0:
        print("⚠️  No non-equity waves found in wave_history.csv")
        return True  # Not a failure, just no data to test
    
    print(f"\n✓ Found {len(non_equity_waves)} non-equity wave rows")
    
    # Check that VIX overlay is disabled
    disabled_count = (non_equity_waves['overlay_active'] == False).sum()
    print(f"✓ {disabled_count}/{len(non_equity_waves)} have overlay_active=False")
    
    if disabled_count < len(non_equity_waves):
        active_non_equity = non_equity_waves[non_equity_waves['overlay_active'] == True]
        print(f"\n⚠️  Found {len(active_non_equity)} non-equity waves with active overlay:")
        print(active_non_equity[['wave', 'overlay_active']].drop_duplicates())
        return False
    
    print("\n✅ Non-equity wave VIX disabled validation passed")
    return True


def test_latest_date_vix_state():
    """Test that latest trading day has VIX execution state."""
    print("\n" + "=" * 70)
    print("TEST 4: Latest Trading Day VIX State")
    print("=" * 70)
    
    wave_history_path = 'wave_history.csv'
    
    if not os.path.exists(wave_history_path):
        print(f"⚠️  wave_history.csv not found.")
        return False
    
    df = pd.read_csv(wave_history_path)
    
    required_vix_cols = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
    if not all(col in df.columns for col in required_vix_cols):
        print(f"⚠️  VIX columns not present. Skipping test.")
        return False
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    
    print(f"\n✓ Latest trading day: {latest_date.strftime('%Y-%m-%d')}")
    
    latest_data = df[df['date'] == latest_date]
    print(f"✓ Found {len(latest_data)} waves for latest date")
    
    # Check for active VIX overlays
    active_vix = latest_data[(latest_data['overlay_active'] == True) & (latest_data['vix_level'].notna())]
    
    if len(active_vix) == 0:
        print("⚠️  No active VIX overlays for latest trading day")
        print("  This is expected if VIX/SPY data is not available")
        return False
    
    print(f"✓ {len(active_vix)} waves have active VIX overlay for latest date")
    
    # Show sample
    print("\nLatest VIX execution state:")
    sample = active_vix[['wave', 'vix_level', 'vix_regime', 'exposure_used']].head(3)
    print(sample.to_string(index=False))
    
    print("\n✅ Latest trading day VIX state validation passed")
    return True


def run_all_tests():
    """Run all VIX execution state tests."""
    print("\n" + "=" * 70)
    print("VIX EXECUTION STATE INTEGRATION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Column structure
    results.append(("Column Structure", test_vix_columns_structure()))
    
    # Test 2: Equity wave VIX data
    results.append(("Equity Wave VIX Data", test_equity_wave_vix_data()))
    
    # Test 3: Non-equity wave VIX disabled
    results.append(("Non-Equity VIX Disabled", test_non_equity_wave_vix_disabled()))
    
    # Test 4: Latest date VIX state
    results.append(("Latest Date VIX State", test_latest_date_vix_state()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "⚠️  SKIP/FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
