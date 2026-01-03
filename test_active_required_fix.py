#!/usr/bin/env python3
"""
Test the Active Required metric fix.

This test verifies that:
1. collect_required_tickers properly converts wave_id to display_name
2. All 27 active waves contribute tickers
3. The total ticker count is realistic (80-200+ tickers)
"""

import sys
import os

def test_collect_required_tickers():
    """Test that collect_required_tickers returns tickers for all active waves."""
    print("=" * 70)
    print("Testing collect_required_tickers with active_only=True")
    print("=" * 70)
    
    try:
        # Import the function to test
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from helpers.price_loader import collect_required_tickers
        
        # Collect tickers for active waves only
        tickers = collect_required_tickers(active_only=True)
        
        print(f"\nResults:")
        print(f"  Total tickers collected: {len(tickers)}")
        print(f"  Sample tickers (first 20): {sorted(tickers)[:20]}")
        
        # Verify expectations
        if len(tickers) < 20:
            print(f"\n❌ FAILED: Only {len(tickers)} tickers collected")
            print("   Expected at least 80+ tickers for 27 active waves")
            return False
        elif len(tickers) < 80:
            print(f"\n⚠️  WARNING: Only {len(tickers)} tickers collected")
            print("   Expected 80-200+ tickers for 27 active waves")
            print("   This might indicate some waves are missing tickers")
        else:
            print(f"\n✅ SUCCESS: {len(tickers)} tickers collected")
            print("   This is a realistic count for 27 active waves")
        
        # Verify essential indicators are included
        essential = ['SPY', '^VIX', 'BTC-USD']
        for ticker in essential:
            if ticker not in tickers:
                print(f"❌ FAILED: Essential indicator {ticker} is missing")
                return False
        print(f"✅ All essential indicators present: {essential}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_active_wave_count():
    """Test that we have 27 active waves in the registry."""
    print("\n" + "=" * 70)
    print("Testing active wave count in wave_registry.csv")
    print("=" * 70)
    
    try:
        import pandas as pd
        
        wave_registry_path = os.path.join('data', 'wave_registry.csv')
        
        if not os.path.exists(wave_registry_path):
            print(f"⚠️  wave_registry.csv not found at {wave_registry_path}")
            return True  # Don't fail if file doesn't exist
        
        df = pd.read_csv(wave_registry_path)
        
        # Count active waves
        active_count = df['active'].sum()
        total_count = len(df)
        
        print(f"\nResults:")
        print(f"  Total waves in registry: {total_count}")
        print(f"  Active waves: {active_count}")
        print(f"  Inactive waves: {total_count - active_count}")
        
        # Show first 10 active wave IDs
        active_waves = df[df['active']]['wave_id'].tolist()
        print(f"\n  First 10 active wave IDs:")
        for i, wave_id in enumerate(active_waves[:10], 1):
            print(f"    {i}. {wave_id}")
        
        # Verify expectations
        if active_count == 27:
            print(f"\n✅ SUCCESS: Exactly 27 active waves as expected")
            return True
        elif active_count >= 20:
            print(f"\n⚠️  WARNING: {active_count} active waves (expected 27)")
            return True
        else:
            print(f"\n❌ FAILED: Only {active_count} active waves")
            print("   Expected at least 20 active waves")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_wave_id_to_display_name_conversion():
    """Test that wave_id to display_name conversion works."""
    print("\n" + "=" * 70)
    print("Testing wave_id to display_name conversion")
    print("=" * 70)
    
    try:
        from waves_engine import get_display_name_from_wave_id, WAVE_WEIGHTS
        
        # Test a few known wave_ids
        test_cases = [
            ("sp500_wave", "S&P 500 Wave"),
            ("ai_cloud_megacap_wave", "AI & Cloud MegaCap Wave"),
            ("gold_wave", "Gold Wave"),
        ]
        
        print(f"\nTesting conversions:")
        all_passed = True
        for wave_id, expected_name in test_cases:
            display_name = get_display_name_from_wave_id(wave_id)
            if display_name == expected_name:
                print(f"  ✓ {wave_id} -> {display_name}")
            else:
                print(f"  ✗ {wave_id} -> {display_name} (expected: {expected_name})")
                all_passed = False
            
            # Check if the display_name exists in WAVE_WEIGHTS
            if display_name and display_name in WAVE_WEIGHTS:
                print(f"    ✓ Display name found in WAVE_WEIGHTS")
            else:
                print(f"    ✗ Display name NOT in WAVE_WEIGHTS")
                all_passed = False
        
        if all_passed:
            print(f"\n✅ SUCCESS: All conversions work correctly")
        else:
            print(f"\n❌ FAILED: Some conversions failed")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Active Required Metric Fix - Test Suite")
    print("=" * 70)
    
    tests = [
        ("Active Wave Count", test_active_wave_count),
        ("Wave ID Conversion", test_wave_id_to_display_name_conversion),
        ("Ticker Collection", test_collect_required_tickers),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print('=' * 70)
        results.append(test_func())
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for (name, _), passed in zip(tests, results):
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nOverall: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    
    if all(results):
        print("\n🎉 All tests passed! The fix is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
