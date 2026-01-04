#!/usr/bin/env python3
"""
Demo script to visualize the Wave Data Ready diagnostics output.
This shows what operators will see in the Overview tab.
"""

from analytics_pipeline import compute_data_ready_status
from waves_engine import get_all_wave_ids

def print_diagnostics_summary():
    """Print a formatted summary of wave readiness diagnostics."""
    
    print("\n" + "=" * 100)
    print("WAVE DATA READY DIAGNOSTICS - OVERVIEW TAB PREVIEW")
    print("=" * 100)
    
    wave_ids = get_all_wave_ids()
    
    # Collect statistics
    ready_count = 0
    degraded_count = 0
    missing_count = 0
    
    ready_waves = []
    degraded_waves = []
    missing_waves = []
    
    for wave_id in sorted(wave_ids):
        result = compute_data_ready_status(wave_id)
        
        if result['is_ready']:
            ready_count += 1
            ready_waves.append((wave_id, result))
        elif result['reason'] in ['STALE_DATA', 'INSUFFICIENT_HISTORY']:
            degraded_count += 1
            degraded_waves.append((wave_id, result))
        else:
            missing_count += 1
            missing_waves.append((wave_id, result))
    
    # Print summary metrics (as shown in UI)
    print("\n📊 SUMMARY METRICS")
    print("-" * 100)
    print(f"  Total Waves:      {len(wave_ids)}")
    print(f"  ✅ Ready:         {ready_count:2d}  (All checks passed, data fresh)")
    print(f"  ⚠️  Degraded:     {degraded_count:2d}  (Stale or insufficient data)")
    print(f"  ❌ Missing:       {missing_count:2d}  (Missing configuration or data files)")
    
    # Print detailed status for each category
    if ready_waves:
        print("\n✅ READY WAVES")
        print("-" * 100)
        print(f"{'Wave ID':<35} {'Display Name':<40} {'Details':<25}")
        print("-" * 100)
        for wave_id, result in ready_waves:
            print(f"{wave_id:<35} {result['display_name']:<40} {result['details']:<25}")
    
    if degraded_waves:
        print("\n⚠️  DEGRADED WAVES")
        print("-" * 100)
        print(f"{'Wave ID':<35} {'Reason':<20} {'Details':<45}")
        print("-" * 100)
        for wave_id, result in degraded_waves:
            print(f"{wave_id:<35} {result['reason']:<20} {result['details']:<45}")
    
    if missing_waves:
        print("\n❌ MISSING INPUTS (Sample - showing first 10)")
        print("-" * 100)
        print(f"{'Wave ID':<35} {'Reason':<20} {'Details':<45}")
        print("-" * 100)
        for wave_id, result in missing_waves[:10]:
            print(f"{wave_id:<35} {result['reason']:<20} {result['details']:<45}")
        if len(missing_waves) > 10:
            print(f"... and {len(missing_waves) - 10} more waves with missing inputs")
    
    # Print example of detailed checks
    print("\n🔍 EXAMPLE: Detailed Checks for a Ready Wave")
    print("-" * 100)
    if ready_waves:
        wave_id, result = ready_waves[0]
        print(f"Wave: {wave_id} ({result['display_name']})")
        print(f"Status: ✅ Ready")
        print(f"Details: {result['details']}")
        print(f"\nChecks:")
        for check_name, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name.replace('_', ' ').title()}: {passed}")
    
    # Print example of missing inputs
    print("\n🔍 EXAMPLE: Detailed Checks for a Wave with Missing Inputs")
    print("-" * 100)
    if missing_waves:
        wave_id, result = missing_waves[0]
        print(f"Wave: {wave_id} ({result['display_name']})")
        print(f"Status: ❌ Missing Inputs")
        print(f"Reason: {result['reason']}")
        print(f"Details: {result['details']}")
        print(f"\nChecks:")
        for check_name, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name.replace('_', ' ').title()}: {passed}")
    
    print("\n" + "=" * 100)
    print("END OF DIAGNOSTICS")
    print("=" * 100)
    print("\nNote: In the actual UI, this data is displayed in an interactive Streamlit table")
    print("with filtering options and CSV export capability.")
    print()


if __name__ == "__main__":
    print_diagnostics_summary()
