#!/usr/bin/env python
"""
Demonstration script for Wave Readiness improvements.
Shows the analytics_ready flag, coverage diagnostics, and readiness panel data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_pipeline import (
    compute_data_ready_status,
    MIN_COVERAGE_FOR_ANALYTICS,
    MIN_DAYS_FOR_ANALYTICS
)
from waves_engine import get_all_wave_ids
from offline_data_loader import generate_data_coverage_summary

def demo_analytics_ready_flag():
    """Demonstrate the analytics_ready flag for different waves."""
    print("=" * 80)
    print("WAVE ANALYTICS READINESS DEMONSTRATION")
    print("=" * 80)
    print(f"\nAnalytics Ready Criteria:")
    print(f"  - Coverage: >= {MIN_COVERAGE_FOR_ANALYTICS*100:.0f}%")
    print(f"  - History: >= {MIN_DAYS_FOR_ANALYTICS} days")
    print("\n" + "=" * 80)
    
    wave_ids = get_all_wave_ids()
    
    # Categorize waves
    analytics_ready = []
    analytics_limited = []
    
    for wave_id in wave_ids:
        diagnostics = compute_data_ready_status(wave_id)
        
        if diagnostics.get('analytics_ready'):
            analytics_ready.append(diagnostics)
        else:
            analytics_limited.append(diagnostics)
    
    # Show analytics-ready waves
    print(f"\n✅ ANALYTICS READY WAVES ({len(analytics_ready)}):")
    if analytics_ready:
        for diag in analytics_ready[:5]:  # Show first 5
            print(f"\n  🟢 {diag['display_name']}")
            print(f"     Coverage: {diag['coverage_pct']:.1f}%")
            print(f"     History: {diag['history_days']} days")
            print(f"     Status: {diag['readiness_status'].title()}")
    else:
        print("\n  No waves currently meet analytics-ready criteria.")
        print(f"  Need: Coverage >= {MIN_COVERAGE_FOR_ANALYTICS*100:.0f}% AND History >= {MIN_DAYS_FOR_ANALYTICS} days")
    
    # Show analytics-limited waves
    print(f"\n⚠️ ANALYTICS LIMITED WAVES ({len(analytics_limited)}):")
    
    # Group by reason
    low_coverage = []
    low_history = []
    both = []
    unavailable = []
    
    for diag in analytics_limited:
        coverage_pct = diag['coverage_pct']
        history_days = diag['history_days']
        
        if diag['readiness_status'] == 'unavailable':
            unavailable.append(diag)
        elif coverage_pct < MIN_COVERAGE_FOR_ANALYTICS * 100 and history_days < MIN_DAYS_FOR_ANALYTICS:
            both.append(diag)
        elif coverage_pct < MIN_COVERAGE_FOR_ANALYTICS * 100:
            low_coverage.append(diag)
        elif history_days < MIN_DAYS_FOR_ANALYTICS:
            low_history.append(diag)
    
    if low_history:
        print(f"\n  📅 Insufficient History ({len(low_history)} waves):")
        for diag in low_history[:3]:
            print(f"     - {diag['display_name']}: {diag['history_days']} days (need {MIN_DAYS_FOR_ANALYTICS})")
            print(f"       Coverage: {diag['coverage_pct']:.1f}% ✓")
            print(f"       💡 Run: python analytics_pipeline.py --wave {diag['wave_id']} --lookback=60")
    
    if low_coverage:
        print(f"\n  📊 Insufficient Coverage ({len(low_coverage)} waves):")
        for diag in low_coverage[:3]:
            print(f"     - {diag['display_name']}: {diag['coverage_pct']:.1f}% (need {MIN_COVERAGE_FOR_ANALYTICS*100:.0f}%)")
            print(f"       History: {diag['history_days']} days ✓")
            missing = diag['missing_tickers']
            if missing:
                print(f"       Missing {len(missing)} ticker(s): {', '.join(missing[:5])}")
    
    if both:
        print(f"\n  ⚠️ Both Issues ({len(both)} waves):")
        for diag in both[:3]:
            print(f"     - {diag['display_name']}")
            print(f"       Coverage: {diag['coverage_pct']:.1f}% (need {MIN_COVERAGE_FOR_ANALYTICS*100:.0f}%)")
            print(f"       History: {diag['history_days']} days (need {MIN_DAYS_FOR_ANALYTICS})")
    
    if unavailable:
        print(f"\n  🔴 Unavailable ({len(unavailable)} waves):")
        for diag in unavailable[:3]:
            print(f"     - {diag['display_name']}: {diag['readiness_status']}")
            blocking = diag.get('blocking_issues', [])
            if blocking:
                print(f"       Blocking: {', '.join(blocking[:2])}")
    
    print("\n" + "=" * 80)


def demo_stale_ticker_detection():
    """Demonstrate stale ticker detection."""
    print("\n" + "=" * 80)
    print("STALE TICKER DETECTION DEMONSTRATION")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    waves_with_stale = []
    
    for wave_id in wave_ids:
        diagnostics = compute_data_ready_status(wave_id)
        
        stale_tickers = diagnostics.get('stale_tickers', [])
        stale_days_max = diagnostics.get('stale_days_max', 0)
        
        if stale_tickers:
            waves_with_stale.append({
                'name': diagnostics['display_name'],
                'stale_tickers': stale_tickers,
                'stale_days_max': stale_days_max
            })
    
    if waves_with_stale:
        print(f"\n⏰ Found {len(waves_with_stale)} waves with stale data (> 7 days old):")
        for wave in waves_with_stale[:5]:
            print(f"\n  - {wave['name']}")
            print(f"    Max Age: {wave['stale_days_max']} days")
            print(f"    Stale Tickers: {', '.join(wave['stale_tickers'][:5])}")
            if len(wave['stale_tickers']) > 5:
                print(f"    ... and {len(wave['stale_tickers'])-5} more")
    else:
        print("\n✅ No stale data detected - All data is current (< 7 days old)")
    
    print("\n" + "=" * 80)


def demo_coverage_summary():
    """Demonstrate the data_coverage_summary.csv output."""
    print("\n" + "=" * 80)
    print("DATA COVERAGE SUMMARY DEMONSTRATION")
    print("=" * 80)
    
    print("\nGenerating data_coverage_summary.csv...")
    summary_df = generate_data_coverage_summary(output_path='demo_coverage_summary.csv')
    
    print(f"\n✓ Generated summary for {len(summary_df)} waves")
    
    # Show top waves by coverage
    print("\n📊 Top Waves by Coverage:")
    top_waves = summary_df.head(10)
    for _, row in top_waves.iterrows():
        status_emoji = "🟢" if row['coverage_pct'] >= 90 else "🟡" if row['coverage_pct'] >= 70 else "🟠" if row['coverage_pct'] >= 50 else "🔴"
        analytics_emoji = "✅" if row['coverage_pct'] >= 85 and row['history_days'] >= 30 else "⚠️"
        
        print(f"  {status_emoji} {analytics_emoji} {row['display_name']}")
        print(f"     Coverage: {row['coverage_pct']:.1f}% | History: {row['history_days']} days")
        if row['missing_tickers']:
            missing_count = len(row['missing_tickers'].split(', '))
            print(f"     Missing: {missing_count} ticker(s)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    demo_analytics_ready_flag()
    demo_stale_ticker_detection()
    demo_coverage_summary()
    
    print("\n✅ All demonstrations complete!")
    print("\nSee 'demo_coverage_summary.csv' for full coverage report.")
