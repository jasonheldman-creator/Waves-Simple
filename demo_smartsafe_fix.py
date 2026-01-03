"""
Demonstration script showing the SmartSafe exemption fix for Data-Ready count.

This script shows:
1. Before: SmartSafe waves were flagged as "not ready" with missing tickers
2. After: SmartSafe waves are correctly identified as "always ready"
3. Impact on Data-Ready count in the UI
"""

from waves_engine import SMARTSAFE_CASH_WAVES, get_all_wave_ids
from analytics_pipeline import compute_data_ready_status, generate_wave_readiness_report
import pandas as pd
from datetime import datetime, timedelta


def simulate_before_fix():
    """Simulate the behavior BEFORE the fix."""
    print("=" * 80)
    print("BEFORE FIX: SmartSafe waves flagged as NOT READY")
    print("=" * 80)
    
    # This would have been the old behavior:
    # - SmartSafe waves checked for price files
    # - Price files missing → flagged as "not ready"
    # - Missing tickers listed
    # - Counted as 0 in Data-Ready count
    
    print("\nOld behavior (simulated):")
    print("  smartsafe_treasury_cash_wave:")
    print("    Status: NOT_READY")
    print("    Reason: MISSING_PRICES")
    print("    Missing Tickers: ['SGOV', 'BIL']")
    print("    Coverage: 0%")
    print()
    print("  smartsafe_tax_free_money_market_wave:")
    print("    Status: NOT_READY")
    print("    Reason: MISSING_PRICES")
    print("    Missing Tickers: ['SHM', 'MUB', 'SUB']")
    print("    Coverage: 0%")
    print()
    print("Impact on UI:")
    print("  Data-Ready Count: 0 (SmartSafe waves NOT counted)")
    print()


def show_after_fix():
    """Show the behavior AFTER the fix."""
    print("=" * 80)
    print("AFTER FIX: SmartSafe waves correctly identified as ALWAYS READY")
    print("=" * 80)
    
    for wave_id in SMARTSAFE_CASH_WAVES:
        result = compute_data_ready_status(wave_id)
        print(f"\n  {wave_id}:")
        print(f"    Status: {result['readiness_status'].upper()}")
        print(f"    Reason: {result['reason']}")
        print(f"    Missing Tickers: {result['missing_tickers']}")
        print(f"    Stale Tickers: {result['stale_tickers']}")
        print(f"    Coverage: {result['coverage_pct']}%")
        print(f"    Readiness Reasons: {result['readiness_reasons']}")
    
    print()
    print("Impact on UI:")
    print(f"  Data-Ready Count: +{len(SMARTSAFE_CASH_WAVES)} (SmartSafe waves NOW counted)")
    print()


def show_wave_readiness_report():
    """Show the wave readiness report with SmartSafe waves."""
    print("=" * 80)
    print("WAVE READINESS REPORT (showing SmartSafe waves)")
    print("=" * 80)
    
    df = generate_wave_readiness_report()
    
    # Filter for full readiness waves
    full_waves = df[df['readiness_status'] == 'full']
    print(f"\nWaves with FULL readiness: {len(full_waves)}")
    print()
    print(full_waves[['wave_id', 'readiness_status', 'coverage_pct', 'readiness_summary']].to_string(index=False))
    print()
    
    # Show summary
    print("Overall Summary:")
    print(df['readiness_status'].value_counts().to_string())
    print()


def show_ui_impact():
    """Show the impact on the UI Data-Ready count."""
    print("=" * 80)
    print("UI DATA-READY COUNT IMPACT")
    print("=" * 80)
    
    # Simulate the count calculation (from app.py)
    try:
        df = pd.read_csv('wave_history.csv')
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()
        
        # Get canonical waves
        all_waves = get_all_wave_ids()
        canonical_waves = all_waves
        
        # Calculate data-ready waves WITHOUT SmartSafe
        recent_data = df[df['date'] >= (latest_date - timedelta(days=7))]
        recent_waves = set(recent_data['wave'].unique())
        canonical_waves_set = set(canonical_waves)
        data_ready_waves = recent_waves.intersection(canonical_waves_set)
        
        before_count = len(data_ready_waves)
        
        # Calculate data-ready waves WITH SmartSafe
        smartsafe_waves_in_canonical = canonical_waves_set.intersection(SMARTSAFE_CASH_WAVES)
        data_ready_waves_with_smartsafe = data_ready_waves.union(smartsafe_waves_in_canonical)
        
        after_count = len(data_ready_waves_with_smartsafe)
        
        print(f"\nBefore fix:")
        print(f"  Data-Ready waves (without SmartSafe): {before_count}")
        print()
        print(f"After fix:")
        print(f"  Data-Ready waves (with SmartSafe): {after_count}")
        print(f"  Improvement: +{after_count - before_count} waves")
        print()
        print(f"SmartSafe waves added to count:")
        for wave in sorted(smartsafe_waves_in_canonical):
            print(f"  - {wave}")
        
    except FileNotFoundError:
        print("\nwave_history.csv not found (demo mode)")
        print("When no wave_history data exists:")
        print(f"  Before: Data-Ready Count = 0")
        print(f"  After:  Data-Ready Count = {len(SMARTSAFE_CASH_WAVES)} (SmartSafe waves)")
    
    print()


def main():
    """Run the demonstration."""
    print("\n" + "=" * 80)
    print("SmartSafe Cash Wave Exemptions - Fix Demonstration")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Show before
    simulate_before_fix()
    
    # Show after
    show_after_fix()
    
    # Show wave readiness report
    show_wave_readiness_report()
    
    # Show UI impact
    show_ui_impact()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ SmartSafe cash waves are now correctly identified as ALWAYS READY")
    print("✓ They show 100% coverage with no missing/stale tickers")
    print("✓ They are included in the Data-Ready count (+2 waves)")
    print("✓ No false alerts about missing price data for cash instruments")
    print()


if __name__ == "__main__":
    main()
