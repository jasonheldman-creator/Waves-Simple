#!/usr/bin/env python3
"""
Demonstration of Active Wave Count Validation

This script demonstrates the new active wave filtering functionality
that eliminates false "Expected 28, found 27" validation alerts.
"""

import pandas as pd
from wave_registry_manager import get_active_wave_registry, get_wave_registry


def demo_active_wave_filtering():
    """Demonstrate active wave filtering."""
    print("\n" + "=" * 80)
    print("ACTIVE WAVE COUNT DEMONSTRATION")
    print("=" * 80)
    
    # Load full registry
    print("\n📊 Loading Wave Registry...")
    full_registry = get_wave_registry()
    active_registry = get_active_wave_registry()
    
    total_waves = len(full_registry)
    active_waves = len(active_registry)
    inactive_waves = total_waves - active_waves
    
    print(f"✓ Total waves in registry: {total_waves}")
    print(f"✓ Active waves: {active_waves}")
    print(f"✓ Inactive waves: {inactive_waves}")
    
    # Show inactive waves
    if inactive_waves > 0:
        print(f"\n📋 Inactive Waves:")
        inactive_df = full_registry[full_registry['active'] == False]
        for _, wave in inactive_df.iterrows():
            print(f"   - {wave['wave_name']}")
            print(f"     Wave ID: {wave['wave_id']}")
            print(f"     Category: {wave.get('category', 'N/A')}")
    
    # Validation simulation
    print(f"\n" + "=" * 80)
    print("VALIDATION SIMULATION")
    print("=" * 80)
    
    # Simulate old behavior (hard-coded 28)
    print(f"\n❌ OLD BEHAVIOR (Hard-coded):")
    print(f"   Expected: 28 waves")
    print(f"   Found: {active_waves} waves")
    if active_waves != 28:
        print(f"   ⚠️  Alert: Expected 28 waves, found {active_waves} ← FALSE ALERT!")
    
    # Simulate new behavior (dynamic count)
    print(f"\n✅ NEW BEHAVIOR (Dynamic):")
    print(f"   Expected: {active_waves} active waves")
    print(f"   Found: {active_waves} active waves")
    if active_waves == active_waves:  # Obviously true, just for demonstration
        print(f"   🎉 Success: Wave Universe Validated: {active_waves}/{active_waves} active waves")
    
    if inactive_waves > 0:
        inactive_names = ', '.join(inactive_df['wave_name'].tolist())
        print(f"   ℹ️  Info: Inactive waves excluded: {inactive_names}")
    
    # Show success banner
    print(f"\n" + "=" * 80)
    print("SUCCESS BANNER (NEW)")
    print("=" * 80)
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  ✅ Wave Universe Validated                             │
    │                                                          │
    │     Universe: {active_waves}                                        │
    │     Waves Live: {active_waves}/{active_waves}                                  │
    │                                                          │
    │     ℹ️  Inactive waves excluded: {inactive_waves}                     │
    │        {', '.join(inactive_df['wave_name'].tolist()) if inactive_waves > 0 else 'None'}{'               ' if inactive_waves > 0 else '                  '}  │
    └─────────────────────────────────────────────────────────┘
    """)
    
    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS")
    print("=" * 80)
    print("""
    ✅ No more false "Expected 28, found 27" alerts
    ✅ Validation reflects actual active wave count
    ✅ Clear indication of inactive waves
    ✅ Success banner shows X/X (both values match)
    ✅ Dynamic computation from wave_registry.csv
    """)
    
    print("=" * 80)
    print("\n✅ Demonstration Complete\n")


if __name__ == '__main__':
    demo_active_wave_filtering()
