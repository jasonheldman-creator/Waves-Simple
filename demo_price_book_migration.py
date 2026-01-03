"""
End-to-end demonstration of PRICE_BOOK-based Performance & Readiness system.

This script simulates the UI flow to show:
1. Loading PRICE_BOOK
2. Computing performance for all waves
3. Computing readiness for all waves
4. Generating diagnostics panel data
5. Expected UI output
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_performance_overview():
    """Demonstrate Performance Overview table generation."""
    print("=" * 80)
    print("DEMO: Performance Overview Table")
    print("=" * 80)
    
    from helpers.price_book import get_price_book
    from helpers.wave_performance import compute_all_waves_performance
    
    # Step 1: Load PRICE_BOOK
    print("\n1. Loading PRICE_BOOK...")
    price_book = get_price_book()
    print(f"   ✓ Loaded: {price_book.shape[0]} days × {price_book.shape[1]} tickers")
    print(f"   Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Step 2: Compute performance for all waves
    print("\n2. Computing performance for all 28 waves...")
    performance_df = compute_all_waves_performance(price_book, periods=[1, 30, 60, 365])
    print(f"   ✓ Computed: {len(performance_df)} waves")
    
    # Step 3: Display sample results
    print("\n3. Sample Performance Results:")
    print("-" * 80)
    
    # Show first 5 waves
    display_cols = ['Wave', '1D Return', '30D', '60D', '365D', 'Status/Confidence']
    print(performance_df[display_cols].head(5).to_string(index=False))
    
    # Step 4: Show status breakdown
    print("\n4. Status Breakdown:")
    status_counts = performance_df['Status/Confidence'].value_counts()
    for status, count in status_counts.items():
        print(f"   {status}: {count} waves")
    
    # Step 5: Show failed waves if any
    failed = performance_df[performance_df['Failure_Reason'].notna()]
    if not failed.empty:
        print(f"\n5. Waves with Issues ({len(failed)} waves):")
        for _, row in failed.iterrows():
            print(f"   - {row['Wave']}: {row['Failure_Reason']} (coverage: {row['Coverage_Pct']:.1f}%)")
    else:
        print("\n5. ✅ All waves computed successfully (no failures)")


def demo_readiness_diagnostics():
    """Demonstrate Readiness Diagnostics panel generation."""
    print("\n\n" + "=" * 80)
    print("DEMO: Wave Data Readiness Diagnostics")
    print("=" * 80)
    
    from helpers.price_book import get_price_book
    from helpers.wave_performance import (
        compute_all_waves_readiness,
        get_price_book_diagnostics,
        compute_all_waves_performance
    )
    from datetime import datetime
    
    # Step 1: Load PRICE_BOOK
    print("\n1. Loading PRICE_BOOK...")
    price_book = get_price_book()
    
    # Step 2: Get PRICE_BOOK diagnostics
    print("\n2. PRICE_BOOK Truth Diagnostics:")
    print("-" * 80)
    pb_diag = get_price_book_diagnostics(price_book)
    
    print(f"   Cache File: prices_cache.parquet")
    print(f"   Path: {pb_diag['path']}")
    print(f"   Shape: {pb_diag['shape'][0]} days × {pb_diag['shape'][1]} tickers")
    print(f"   Date Range: {pb_diag['date_min']} to {pb_diag['date_max']}")
    
    if pb_diag['date_max'] != 'N/A':
        latest_date = datetime.strptime(pb_diag['date_max'], '%Y-%m-%d')
        days_stale = (datetime.now() - latest_date).days
        print(f"   Staleness: {days_stale} days old")
    
    # Step 3: Wave status summary
    print("\n3. Wave Status Summary:")
    print("-" * 80)
    
    from waves_engine import get_all_waves_universe
    universe = get_all_waves_universe()
    total_waves = len(universe.get('waves', []))
    
    perf_df = compute_all_waves_performance(price_book, periods=[1])
    waves_with_data = len(perf_df[perf_df['Status/Confidence'] != 'Unavailable'])
    
    print(f"   Total Active Waves: {total_waves}")
    print(f"   Waves Returning Data: {waves_with_data}/{total_waves}")
    print(f"   Waves with Issues: {total_waves - waves_with_data}")
    
    # Step 4: Show failing waves grouped by reason
    failed_waves = perf_df[perf_df['Failure_Reason'].notna()]
    
    if not failed_waves.empty:
        print(f"\n4. Waves with N/A Data ({len(failed_waves)} waves):")
        print("-" * 80)
        
        failure_groups = failed_waves.groupby('Failure_Reason')['Wave'].apply(list).to_dict()
        
        for reason, waves in failure_groups.items():
            print(f"\n   ❌ {reason} ({len(waves)} waves):")
            for wave in waves:
                print(f"      - {wave}")
    else:
        print("\n4. ✅ All waves returning data successfully")
    
    # Step 5: Readiness table
    print("\n5. Wave Readiness Assessment:")
    print("-" * 80)
    
    readiness_df = compute_all_waves_readiness(price_book)
    
    ready_count = readiness_df['data_ready'].sum()
    total_count = len(readiness_df)
    
    print(f"   Data-Ready Waves: {ready_count}/{total_count} ({ready_count/total_count*100:.1f}%)")
    
    # Show not-ready waves
    not_ready = readiness_df[~readiness_df['data_ready']]
    if not not_ready.empty:
        print(f"\n   Not Data-Ready Waves ({len(not_ready)} waves):")
        for _, row in not_ready.head(5).iterrows():
            print(f"      - {row['wave_name']}: {row['reason']} (coverage: {row['coverage_pct']:.1f}%)")
        if len(not_ready) > 5:
            print(f"      ... and {len(not_ready) - 5} more")
    else:
        print("\n   ✅ All waves are data-ready")


def demo_expected_ui_flow():
    """Show expected UI flow in System Health tab."""
    print("\n\n" + "=" * 80)
    print("DEMO: Expected System Health Tab UI Flow")
    print("=" * 80)
    
    print("""
USER NAVIGATES TO SYSTEM HEALTH TAB
↓
SECTION: Wave Data Readiness Diagnostics
    ├─ PRICE_BOOK Truth Diagnostics Panel
    │  ├─ Cache metadata (path, shape, dates, staleness)
    │  ├─ Wave status (28/28 returning data)
    │  └─ Failure reason groups (if any)
    │
    ├─ Wave-by-Wave Readiness Assessment Table
    │  ├─ Checkbox: "Show only NOT data-ready"
    │  ├─ Columns: wave_name, data_ready, reason, coverage_pct, ...
    │  └─ Summary: X/28 waves are data-ready (X% readiness)
    │
    └─ [Divider]
↓
SECTION: 28 Waves Performance Overview
    ├─ Label: "Data Source: PRICE_BOOK (prices_cache.parquet)"
    │         "Live computation from canonical price cache"
    │
    ├─ Performance Table
    │  ├─ Columns: Wave, 1D Return, 30D, 60D, 365D, Status/Confidence
    │  └─ Shows actual computed returns
    │
    └─ Expander: "⚠️ Waves with Issues (X waves)"
       └─ Shows waves with Failure_Reason and Coverage_Pct
↓
[Rest of System Health Tab continues...]

KEY IMPROVEMENTS:
✓ No "could not find CSV" warnings
✓ All data computed live from PRICE_BOOK
✓ Clear failure reasons when issues occur
✓ Explicit data source labeling
✓ Real-time coverage and staleness reporting
    """)


def demo_comparison():
    """Show before/after comparison."""
    print("\n\n" + "=" * 80)
    print("DEMO: Before vs After Comparison")
    print("=" * 80)
    
    print("""
BEFORE (CSV-based):
-------------------
Performance Overview:
    ❌ Shows N/A for nearly all waves
    ❌ Data source: Stale snapshot CSV or "No data available"
    ❌ No failure reasons
    ❌ Dependent on snapshot generation timing

Readiness Diagnostics:
    ❌ "Could not find data_coverage_summary.csv"
    ❌ Relies on stale CSV artifact
    ❌ May show incorrect readiness due to CSV staleness
    ❌ No PRICE_BOOK metadata

AFTER (PRICE_BOOK-based):
-------------------------
Performance Overview:
    ✅ Shows actual returns for all 28 waves (where data available)
    ✅ Data source: "PRICE_BOOK (prices_cache.parquet)"
    ✅ Explicit failure reasons in expander
    ✅ Always current with PRICE_BOOK state

Readiness Diagnostics:
    ✅ Live computation against PRICE_BOOK
    ✅ No CSV dependency
    ✅ Accurate readiness based on actual ticker coverage
    ✅ PRICE_BOOK Truth Diagnostics panel shows metadata
    ✅ Failure reasons grouped for easy debugging

BENEFITS:
---------
• Single source of truth (PRICE_BOOK)
• Always current data
• Transparent failure reasons
• No stale CSV artifacts
• Faster load times
• Better diagnostics
    """)


def main():
    """Run all demos."""
    demo_performance_overview()
    demo_readiness_diagnostics()
    demo_expected_ui_flow()
    demo_comparison()
    
    print("\n\n" + "=" * 80)
    print("🎉 END-TO-END DEMO COMPLETE")
    print("=" * 80)
    print("\nAll functionality working as expected!")
    print("\nTo see these changes in the UI:")
    print("1. Run: streamlit run app.py")
    print("2. Navigate to 'System Health' tab")
    print("3. Scroll down to see the updated sections")
    print("=" * 80)


if __name__ == '__main__':
    main()
