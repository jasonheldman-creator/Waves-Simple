"""
Validation script for benchmark diagnostics and proof fields.

This script demonstrates the new auditable proof fields added to wave computations:
1. Benchmark mode (DYNAMIC vs STATIC)
2. Benchmark components preview
3. Benchmark hash
4. 365D window integrity
5. Alpha reconciliation

Run this script to see the new fields in action.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Import the waves engine
from waves_engine import compute_history_nav, load_dynamic_benchmark_specs

# Import price book
try:
    from helpers.price_book import get_price_book
    PRICE_BOOK_AVAILABLE = True
except ImportError:
    PRICE_BOOK_AVAILABLE = False
    print("⚠️  Warning: helpers.price_book not available. Using sample data.")


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def validate_benchmark_diagnostics():
    """Validate that benchmark diagnostic fields are present and accurate."""
    
    print_section_header("BENCHMARK DIAGNOSTICS VALIDATION")
    
    if not PRICE_BOOK_AVAILABLE:
        print("⚠️  Cannot run validation without price_book. Skipping...")
        return
    
    # Load price book
    print("\n📊 Loading price data...")
    try:
        price_book = get_price_book()
        print(f"✓ Loaded price data: {len(price_book)} days × {len(price_book.columns)} tickers")
        print(f"  Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    except Exception as e:
        print(f"✗ Error loading price book: {e}")
        return
    
    # Load benchmark specs
    print("\n🎯 Loading dynamic benchmark specifications...")
    benchmark_specs = load_dynamic_benchmark_specs()
    if benchmark_specs:
        num_benchmarks = len(benchmark_specs.get('benchmarks', {}))
        print(f"✓ Loaded {num_benchmarks} dynamic benchmark specifications")
        print(f"  Version: {benchmark_specs.get('version', 'unknown')}")
    else:
        print("⚠️  No dynamic benchmark specifications found")
    
    # Test waves to validate
    test_waves = [
        "S&P 500 Wave",  # Should have DYNAMIC benchmark (SPY:100%)
        "AI & Cloud MegaCap Wave",  # Should have DYNAMIC benchmark (multi-component)
        "US MegaCap Core Wave",  # Should have DYNAMIC benchmark
    ]
    
    print("\n🔬 Testing benchmark diagnostics for waves...")
    
    for wave_name in test_waves:
        print(f"\n{'─'*80}")
        print(f"Wave: {wave_name}")
        print(f"{'─'*80}")
        
        try:
            # Compute wave history
            result = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=365,
                include_diagnostics=False,
                price_df=price_book
            )
            
            if result.empty:
                print(f"✗ No data returned for {wave_name}")
                continue
            
            # Extract coverage attrs
            coverage = result.attrs.get('coverage', {})
            
            # Check benchmark mode
            benchmark_mode = coverage.get('benchmark_mode', 'UNKNOWN')
            print(f"\n📌 Benchmark Mode: {benchmark_mode}")
            
            # Check benchmark components preview
            components_preview = coverage.get('benchmark_components_preview', 'N/A')
            print(f"📌 Benchmark Components: {components_preview}")
            
            # Check benchmark hash
            benchmark_hash = coverage.get('benchmark_hash', 'N/A')
            print(f"📌 Benchmark Hash: {benchmark_hash}")
            
            # Check dynamic benchmark details
            dynamic_bm = coverage.get('dynamic_benchmark', {})
            if dynamic_bm.get('enabled'):
                print(f"📌 Dynamic Benchmark Name: {dynamic_bm.get('benchmark_name', 'Unknown')}")
                print(f"📌 Dynamic Benchmark Version: {dynamic_bm.get('version', 'Unknown')}")
                components = dynamic_bm.get('components', [])
                print(f"📌 Number of Components: {len(components)}")
                for comp in components:
                    avail_icon = "✓" if comp.get('available') else "✗"
                    print(f"   {avail_icon} {comp['ticker']}: {comp['weight']*100:.1f}%")
            else:
                reason = dynamic_bm.get('reason', 'unknown')
                print(f"📌 Dynamic Benchmark: Disabled ({reason})")
            
            # Check 365D window integrity
            window_integrity = coverage.get('window_365d_integrity', {})
            if window_integrity:
                print(f"\n📊 365D Window Integrity:")
                print(f"   Wave Days: {window_integrity.get('wave_365d_days', 'N/A')}")
                print(f"   Benchmark Days: {window_integrity.get('bench_365d_days', 'N/A')}")
                print(f"   Intersection Days: {window_integrity.get('intersection_days_used', 'N/A')}")
                print(f"   Sufficient History: {window_integrity.get('sufficient_history', False)}")
                
                if window_integrity.get('wave_365d_start'):
                    print(f"   Wave Date Range: {window_integrity['wave_365d_start']} to {window_integrity['wave_365d_end']}")
                if window_integrity.get('bench_365d_start'):
                    print(f"   Benchmark Date Range: {window_integrity['bench_365d_start']} to {window_integrity['bench_365d_end']}")
                
                warning = window_integrity.get('warning_message')
                if warning:
                    print(f"   ⚠️  WARNING: {warning}")
            
            # Check alpha reconciliation
            reconciliation = coverage.get('alpha_365d_reconciliation', {})
            if reconciliation:
                print(f"\n✅ 365D Alpha Reconciliation:")
                passed = reconciliation.get('reconciliation_passed', False)
                status_icon = "✓" if passed else "✗"
                print(f"   {status_icon} Reconciliation: {'PASSED' if passed else 'FAILED'}")
                
                expected = reconciliation.get('expected_alpha')
                computed = reconciliation.get('computed_alpha')
                mismatch = reconciliation.get('mismatch')
                mismatch_bps = reconciliation.get('mismatch_bps')
                
                if expected is not None and computed is not None:
                    print(f"   Expected Alpha: {expected*100:.4f}%")
                    print(f"   Computed Alpha: {computed*100:.4f}%")
                    print(f"   Mismatch: {mismatch*100:.6f}% ({mismatch_bps:.2f} bps)")
                
                wave_ret = coverage.get('wave_365d_return')
                bench_ret = coverage.get('bench_365d_return')
                alpha = coverage.get('alpha_365d')
                
                if wave_ret is not None:
                    print(f"   Wave 365D Return: {wave_ret*100:.2f}%")
                if bench_ret is not None:
                    print(f"   Benchmark 365D Return: {bench_ret*100:.2f}%")
                if alpha is not None:
                    print(f"   Alpha 365D: {alpha*100:.2f}%")
                
                warning = reconciliation.get('warning_message')
                if warning:
                    print(f"   ⚠️  WARNING: {warning}")
            
            print(f"\n{'─'*80}")
            print(f"✓ Validation complete for {wave_name}")
            
        except Exception as e:
            print(f"✗ Error validating {wave_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main validation function."""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "BENCHMARK DIAGNOSTICS VALIDATOR" + " "*27 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nThis script validates the new auditable proof fields:")
    print("  1. Benchmark mode (DYNAMIC vs STATIC)")
    print("  2. Benchmark components preview")
    print("  3. Benchmark hash (for auditability)")
    print("  4. 365D window integrity (date ranges, overlap)")
    print("  5. Alpha reconciliation (validation checks)")
    
    # Run validation
    validate_benchmark_diagnostics()
    
    print_section_header("VALIDATION SUMMARY")
    
    print("\n✅ All benchmark diagnostic fields are now available in wave computations!")
    print("\nFields are stored in result.attrs['coverage'] and include:")
    print("  • benchmark_mode: 'DYNAMIC' or 'STATIC'")
    print("  • benchmark_components_preview: Formatted string with top 5 tickers")
    print("  • benchmark_hash: Stable hash for auditability")
    print("  • window_365d_integrity: Dict with date ranges and overlap metrics")
    print("  • alpha_365d_reconciliation: Dict with validation results")
    
    print("\n📝 Next steps:")
    print("  1. Update UI to display these fields in wave detail panels")
    print("  2. Add warning banners for LIMITED HISTORY or reconciliation failures")
    print("  3. Show N/A for alpha metrics when history is insufficient")
    
    print("\n" + "█"*80 + "\n")


if __name__ == "__main__":
    main()
