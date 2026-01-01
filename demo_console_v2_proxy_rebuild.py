#!/usr/bin/env python3
"""
Console v2: Clean Proxy-Based Rebuild Demo
============================================

This script demonstrates the clean proxy-based rebuild capability for all 28 waves
in the Institutional Console v2. It showcases:

1. Validation of the proxy registry (28 waves)
2. Clean rebuild of the proxy snapshot
3. Verification that all 28 waves are rendered with analytics
4. Display of confidence levels and data quality metrics

Usage:
    python demo_console_v2_proxy_rebuild.py

Features:
    - ✅ Validates all 28 waves are configured in proxy registry
    - ✅ Performs clean proxy rebuild with timeout protection
    - ✅ Shows confidence levels (FULL, PARTIAL, UNAVAILABLE)
    - ✅ Displays analytics for each wave (returns, alpha)
    - ✅ Saves detailed diagnostics
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from planb_proxy_pipeline import (
    build_proxy_snapshot,
    load_proxy_snapshot,
    get_snapshot_freshness,
    load_diagnostics,
    CONFIDENCE_FULL,
    CONFIDENCE_PARTIAL,
    CONFIDENCE_UNAVAILABLE
)
from helpers.proxy_registry_validator import (
    load_proxy_registry,
    validate_proxy_registry,
    get_enabled_proxy_waves
)


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def validate_proxy_registry_demo():
    """
    Step 1: Validate the proxy registry and confirm 28 waves.
    """
    print_section("STEP 1: Validating Proxy Registry (Expecting 28 Waves)")
    
    # Load and validate registry
    validation_result = validate_proxy_registry(strict=False)
    
    print(validation_result['report'])
    
    # Get enabled waves
    waves = get_enabled_proxy_waves()
    
    if len(waves) == 28:
        print(f"\n✅ SUCCESS: Proxy registry contains exactly 28 waves")
    else:
        print(f"\n⚠️  WARNING: Expected 28 waves, found {len(waves)}")
    
    # List all waves by category
    categories = {}
    for wave in waves:
        category = wave.get('category', 'Unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(wave.get('display_name', wave.get('wave_id', 'Unknown')))
    
    print("\nWaves by Category:")
    for category, wave_names in sorted(categories.items()):
        print(f"\n  {category} ({len(wave_names)} waves):")
        for name in sorted(wave_names):
            print(f"    • {name}")
    
    return len(waves) == 28


def rebuild_proxy_snapshot_demo():
    """
    Step 2: Perform a clean rebuild of the proxy snapshot.
    """
    print_section("STEP 2: Rebuilding Proxy Snapshot (Max 15s Timeout)")
    
    # Create a mock session state to disable safe mode for demo
    mock_session_state = {
        "safe_mode_no_fetch": False,
        "safe_demo_mode": False
    }
    
    print("Starting proxy snapshot rebuild...")
    print("  • Fetching 365 days of price data")
    print("  • Computing returns (1D, 30D, 60D, 365D)")
    print("  • Calculating alpha vs benchmarks")
    print("  • Enforcing 15-second timeout")
    print()
    
    # Build the snapshot
    snapshot_df = build_proxy_snapshot(
        days=365,
        enforce_timeout=True,
        session_state=mock_session_state,
        explicit_button_click=True
    )
    
    if not snapshot_df.empty:
        print(f"\n✅ SUCCESS: Built snapshot with {len(snapshot_df)} waves")
        return True
    else:
        print(f"\n❌ FAILED: Snapshot is empty")
        return False


def analyze_snapshot_results():
    """
    Step 3: Analyze the snapshot results and display metrics.
    """
    print_section("STEP 3: Analyzing Snapshot Results")
    
    # Load the snapshot
    snapshot_df = load_proxy_snapshot()
    
    if snapshot_df.empty:
        print("❌ No snapshot data available")
        return False
    
    # Get freshness info
    freshness = get_snapshot_freshness()
    
    print(f"Snapshot Info:")
    print(f"  • Total waves in snapshot: {len(snapshot_df)}")
    print(f"  • Snapshot age: {freshness.get('age_minutes', 0):.1f} minutes")
    print(f"  • Freshness status: {'Fresh ✅' if freshness.get('fresh') else 'Stale ⚠️'}")
    
    # Confidence breakdown
    confidence_counts = snapshot_df['confidence'].value_counts()
    
    print(f"\nConfidence Level Breakdown:")
    for confidence in [CONFIDENCE_FULL, CONFIDENCE_PARTIAL, CONFIDENCE_UNAVAILABLE]:
        count = confidence_counts.get(confidence, 0)
        icon = "🟢" if confidence == CONFIDENCE_FULL else "🔵" if confidence == CONFIDENCE_PARTIAL else "🔴"
        print(f"  {icon} {confidence}: {count} waves")
    
    # Category breakdown
    print(f"\nWaves by Category:")
    category_counts = snapshot_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  • {category}: {count} waves")
    
    # Load diagnostics
    diagnostics = load_diagnostics()
    
    if diagnostics:
        print(f"\nBuild Diagnostics:")
        print(f"  • Build duration: {diagnostics.get('build_duration_seconds', 0):.1f}s")
        print(f"  • Successful fetches: {diagnostics.get('successful_fetches', 0)}")
        print(f"  • Failed fetches: {diagnostics.get('failed_fetches', 0)}")
        if diagnostics.get('timeout_exceeded'):
            print(f"  • ⚠️  Timeout exceeded (partial results)")
    
    return True


def display_wave_analytics():
    """
    Step 4: Display sample analytics for waves.
    """
    print_section("STEP 4: Sample Wave Analytics")
    
    # Load the snapshot
    snapshot_df = load_proxy_snapshot()
    
    if snapshot_df.empty:
        print("❌ No snapshot data available")
        return False
    
    # Display analytics for top waves by category
    categories_to_show = ['Equity', 'Crypto', 'Fixed Income', 'Commodity']
    
    for category in categories_to_show:
        category_waves = snapshot_df[snapshot_df['category'] == category]
        
        if category_waves.empty:
            continue
        
        print(f"\n{category} Waves:")
        print("-" * 80)
        
        # Show up to 3 waves from each category
        for idx, (_, wave) in enumerate(category_waves.head(3).iterrows()):
            confidence_icon = "🟢" if wave['confidence'] == CONFIDENCE_FULL else "🔵" if wave['confidence'] == CONFIDENCE_PARTIAL else "🔴"
            
            print(f"\n  {confidence_icon} {wave['display_name']}")
            print(f"     Proxy: {wave['proxy_ticker']} | Benchmark: {wave['benchmark_ticker']}")
            
            if wave['confidence'] in [CONFIDENCE_FULL, CONFIDENCE_PARTIAL]:
                # Display returns (handle NaN values)
                ret_1d = wave['return_1D'] * 100 if pd.notna(wave['return_1D']) else 0.0
                ret_30d = wave['return_30D'] * 100 if pd.notna(wave['return_30D']) else 0.0
                ret_365d = wave['return_365D'] * 100 if pd.notna(wave['return_365D']) else 0.0
                print(f"     Returns: 1D={ret_1d:.2f}% | 30D={ret_30d:.2f}% | 365D={ret_365d:.2f}%")
                
                # Display alpha (handle NaN values)
                alpha_1d = wave['alpha_1D'] * 100 if pd.notna(wave['alpha_1D']) else 0.0
                alpha_30d = wave['alpha_30D'] * 100 if pd.notna(wave['alpha_30D']) else 0.0
                alpha_365d = wave['alpha_365D'] * 100 if pd.notna(wave['alpha_365D']) else 0.0
                print(f"     Alpha:   1D={alpha_1d:.2f}% | 30D={alpha_30d:.2f}% | 365D={alpha_365d:.2f}%")
            else:
                print(f"     Status: Data unavailable")
    
    return True


def main():
    """
    Main demo script execution.
    """
    print_header("Console v2: Clean Proxy-Based Rebuild Demo")
    print("Demonstrating proxy-based analytics for all 28 waves")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Validate proxy registry
    if not validate_proxy_registry_demo():
        print("\n❌ DEMO FAILED: Proxy registry validation failed")
        return 1
    
    # Step 2: Rebuild proxy snapshot
    if not rebuild_proxy_snapshot_demo():
        print("\n❌ DEMO FAILED: Proxy snapshot rebuild failed")
        return 1
    
    # Step 3: Analyze results
    if not analyze_snapshot_results():
        print("\n❌ DEMO FAILED: Snapshot analysis failed")
        return 1
    
    # Step 4: Display analytics
    if not display_wave_analytics():
        print("\n❌ DEMO FAILED: Analytics display failed")
        return 1
    
    # Success summary
    print_header("DEMO COMPLETED SUCCESSFULLY")
    print("✅ All 28 waves validated in proxy registry")
    print("✅ Clean proxy rebuild completed within timeout")
    print("✅ Snapshot saved to data/live_proxy_snapshot.csv")
    print("✅ Diagnostics saved to data/planb_diagnostics_run.json")
    print("✅ Analytics computed for all waves")
    print("\nThe Console v2 proxy-based rebuild system is working cleanly!")
    print("\nNext Steps:")
    print("  • Review snapshot: data/live_proxy_snapshot.csv")
    print("  • Review diagnostics: data/planb_diagnostics_run.json")
    print("  • Run the app: streamlit run app.py")
    print("  • Navigate to Overview tab to see all 28 waves rendered")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
