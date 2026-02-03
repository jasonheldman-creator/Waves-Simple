#!/usr/bin/env python3
"""
Update wave status based on data readiness.

This script checks each wave for complete dynamic benchmark and volatility overlay data,
and marks waves without complete data as STAGING in the registry.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.wave_registry import update_wave_status_based_on_readiness, get_wave_registry

def main():
    print("=" * 80)
    print("WAVE STATUS UPDATE - Mark waves as STAGING based on data readiness")
    print("=" * 80)
    
    # Show current status
    print("\nCurrent wave status in registry:")
    registry = get_wave_registry()
    if not registry.empty:
        status_counts = registry['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} waves")
    
    # Update wave status based on readiness
    print("\nUpdating wave status based on data readiness...")
    updated_count = update_wave_status_based_on_readiness()
    
    # Show updated status
    print("\nUpdated wave status in registry:")
    registry = get_wave_registry()
    if not registry.empty:
        status_counts = registry['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} waves")
    
    print("\n" + "=" * 80)
    print(f"✓ Updated {updated_count} waves to STAGING status")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
