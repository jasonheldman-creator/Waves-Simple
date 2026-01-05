#!/usr/bin/env python3
"""
Validation script for auto-refresh configuration changes.

This script validates that:
1. Auto-refresh is enabled by default (DEFAULT_AUTO_REFRESH_ENABLED = True)
2. Default interval is 30 seconds (DEFAULT_REFRESH_INTERVAL_MS = 30000)
3. 30 seconds option exists in REFRESH_INTERVAL_OPTIONS
"""

import sys

def validate_auto_refresh_config():
    """Validate auto_refresh_config.py settings."""
    print("=" * 60)
    print("Validating Auto-Refresh Configuration")
    print("=" * 60)
    
    # Import the config module
    try:
        from auto_refresh_config import (
            DEFAULT_AUTO_REFRESH_ENABLED,
            DEFAULT_REFRESH_INTERVAL_MS,
            REFRESH_INTERVAL_OPTIONS
        )
    except ImportError as e:
        print(f"❌ Failed to import auto_refresh_config: {e}")
        return False
    
    success = True
    
    # Check 1: Auto-refresh enabled by default
    print("\n1. Checking DEFAULT_AUTO_REFRESH_ENABLED...")
    if DEFAULT_AUTO_REFRESH_ENABLED:
        print("   ✅ Auto-refresh is enabled by default (True)")
    else:
        print(f"   ❌ Expected True, got {DEFAULT_AUTO_REFRESH_ENABLED}")
        success = False
    
    # Check 2: Default interval is 30 seconds
    print("\n2. Checking DEFAULT_REFRESH_INTERVAL_MS...")
    if DEFAULT_REFRESH_INTERVAL_MS == 30000:
        print("   ✅ Default interval is 30 seconds (30000ms)")
    else:
        print(f"   ❌ Expected 30000ms, got {DEFAULT_REFRESH_INTERVAL_MS}ms")
        success = False
    
    # Check 3: 30 seconds option exists
    print("\n3. Checking REFRESH_INTERVAL_OPTIONS...")
    if "30 seconds" in REFRESH_INTERVAL_OPTIONS:
        if REFRESH_INTERVAL_OPTIONS["30 seconds"] == 30000:
            print("   ✅ '30 seconds' option exists with correct value (30000ms)")
        else:
            print(f"   ❌ '30 seconds' option has incorrect value: {REFRESH_INTERVAL_OPTIONS['30 seconds']}ms")
            success = False
    else:
        print("   ❌ '30 seconds' option not found in REFRESH_INTERVAL_OPTIONS")
        success = False
    
    # Display all available options
    print("\n   Available refresh interval options:")
    for name, value_ms in sorted(REFRESH_INTERVAL_OPTIONS.items(), key=lambda x: x[1]):
        print(f"      - {name}: {value_ms}ms ({value_ms/1000}s)")
    
    return success


def validate_beta_computation():
    """Validate beta computation functions are importable."""
    print("\n" + "=" * 60)
    print("Validating Beta Computation Functions")
    print("=" * 60)
    
    # Import the functions
    try:
        from helpers.wave_performance import compute_beta, compute_wave_beta
        print("\n✅ Beta computation functions imported successfully")
        print("   - compute_beta()")
        print("   - compute_wave_beta()")
        return True
    except ImportError as e:
        print(f"\n❌ Failed to import beta functions: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("AUTO-REFRESH AND BETA CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Run validations
    config_ok = validate_auto_refresh_config()
    beta_ok = validate_beta_computation()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if config_ok and beta_ok:
        print("\n✅ All validations passed!")
        print("\nChanges implemented:")
        print("  1. Auto-refresh enabled by default")
        print("  2. Default interval set to 30 seconds")
        print("  3. Beta computation functions available")
        return 0
    else:
        print("\n❌ Some validations failed!")
        if not config_ok:
            print("  - Auto-refresh configuration issues detected")
        if not beta_ok:
            print("  - Beta computation functions not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
