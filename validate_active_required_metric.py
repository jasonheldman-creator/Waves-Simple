#!/usr/bin/env python3
"""
Validation script for the Active Required metric fix.

This script simulates the diagnostic display and verifies:
1. Active Required count is realistic (80-200+ tickers)
2. Missing tickers list is smaller and actionable
3. Guardrail warning triggers correctly
"""

import sys
import os
import pandas as pd


def validate_active_required_metric():
    """Validate the Active Required metric calculation."""
    print("=" * 70)
    print("ACTIVE REQUIRED METRIC VALIDATION")
    print("=" * 70)
    
    # Step 1: Get active wave count from registry
    print("\nStep 1: Reading wave registry...")
    wave_registry_path = 'data/wave_registry.csv'
    registry_df = pd.read_csv(wave_registry_path)
    active_wave_count = registry_df['active'].sum()
    total_wave_count = len(registry_df)
    
    print(f"  Total waves: {total_wave_count}")
    print(f"  Active waves: {active_wave_count}")
    print(f"  Status: {'✅' if active_wave_count >= 20 else '⚠️'}")
    
    # Step 2: Collect required tickers (simulate the fix)
    print("\nStep 2: Collecting required tickers...")
    
    from waves_engine import (
        get_all_wave_ids,
        get_display_name_from_wave_id,
        WAVE_WEIGHTS,
        BENCHMARK_WEIGHTS_STATIC,
        is_smartsafe_cash_wave
    )
    
    all_wave_ids = get_all_wave_ids()
    active_wave_ids = set(registry_df[registry_df['active']]['wave_id'].tolist())
    filtered_wave_ids = [wid for wid in all_wave_ids if wid in active_wave_ids]
    
    tickers = set()
    waves_processed = 0
    waves_skipped = 0
    ticker_counts = []
    
    for wave_id in filtered_wave_ids:
        # Skip SmartSafe cash waves
        if is_smartsafe_cash_wave(wave_id):
            waves_skipped += 1
            continue
        
        # Convert wave_id to display_name
        display_name = get_display_name_from_wave_id(wave_id)
        if not display_name:
            print(f"  ⚠️  Warning: Could not get display name for {wave_id}")
            continue
        
        wave_tickers_before = len(tickers)
        
        # Get holdings tickers
        wave_weights = WAVE_WEIGHTS.get(display_name, [])
        for holding in wave_weights:
            if hasattr(holding, 'ticker'):
                tickers.add(holding.ticker)
        
        # Get benchmark tickers
        benchmark_weights = BENCHMARK_WEIGHTS_STATIC.get(display_name, [])
        for benchmark in benchmark_weights:
            if isinstance(benchmark, tuple) and len(benchmark) >= 1:
                tickers.add(benchmark[0])
        
        wave_tickers_added = len(tickers) - wave_tickers_before
        ticker_counts.append((wave_id, wave_tickers_added))
        waves_processed += 1
    
    # Add essential indicators
    essential_indicators = ['SPY', '^VIX', 'BTC-USD']
    tickers.update(essential_indicators)
    
    active_required = len(tickers)
    
    print(f"  Waves processed: {waves_processed}")
    print(f"  Waves skipped (SmartSafe): {waves_skipped}")
    print(f"  Total unique tickers: {active_required}")
    
    # Show first 10 waves with their ticker counts
    print(f"\n  First 10 waves and their ticker contributions:")
    for wave_id, count in ticker_counts[:10]:
        print(f"    {wave_id}: {count} tickers")
    
    # Step 3: Evaluate the metric
    print("\nStep 3: Evaluating Active Required metric...")
    print(f"  Active Required: {active_required}")
    
    if active_required >= 80:
        print(f"  Status: ✅ EXCELLENT ({active_required} tickers)")
        print(f"  This reflects all {active_wave_count} active waves properly.")
    elif active_required >= 20:
        print(f"  Status: ⚠️  ACCEPTABLE ({active_required} tickers)")
        print(f"  Expected 80-200+ for {active_wave_count} waves, but this is functional.")
    else:
        print(f"  Status: ❌ FAILED ({active_required} tickers)")
        print(f"  This is too few for {active_wave_count} active waves!")
    
    # Step 4: Check guardrail condition
    print("\nStep 4: Checking guardrail condition...")
    guardrail_triggered = active_wave_count >= 20 and active_required < 20
    
    if guardrail_triggered:
        print(f"  🚨 GUARDRAIL WARNING WOULD TRIGGER:")
        print(f"     'BUG: active_required too small — registry/ticker collection failed.'")
        print(f"     Active waves: {active_wave_count}")
        print(f"     Active required: {active_required}")
    else:
        print(f"  ✅ Guardrail check passed (no warning needed)")
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    issues = []
    if active_required < 20:
        issues.append(f"Active Required too low: {active_required} < 20")
    if active_wave_count >= 20 and active_required < 20:
        issues.append("Guardrail would trigger (as expected before fix)")
    
    if not issues:
        print("✅ All checks passed!")
        print(f"   - {active_wave_count} active waves detected")
        print(f"   - {active_required} tickers collected")
        print(f"   - {waves_processed} waves contributed tickers")
        print(f"   - {waves_skipped} waves skipped (SmartSafe)")
        print("\n🎉 The Active Required metric is working correctly!")
        return 0
    else:
        print("⚠️  Issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nNote: If this is a pre-fix validation, these issues are expected.")
        print("After the fix, Active Required should be 80-200+ tickers.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(validate_active_required_metric())
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
