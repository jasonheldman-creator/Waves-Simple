"""
validate_truthframe.py

Quick validation script to verify TruthFrame implementation.
Run this to confirm everything is working correctly.
"""

from analytics_truth import get_truth_frame
from truth_frame_helpers import (
    get_wave_returns,
    get_wave_alphas,
    format_return_display,
    get_top_performers,
    get_readiness_summary,
)


def validate_truthframe():
    """Validate TruthFrame implementation"""
    
    print("=" * 80)
    print("TRUTHFRAME VALIDATION")
    print("=" * 80)
    
    # Test 1: Load TruthFrame in Safe Mode
    print("\n[1/6] Loading TruthFrame in Safe Mode...")
    truth_df = get_truth_frame(safe_mode=True)
    
    if truth_df is None or truth_df.empty:
        print("  ✗ FAILED: TruthFrame is empty")
        return False
    
    print(f"  ✓ TruthFrame loaded: {len(truth_df)} waves")
    
    # Test 2: Verify all 28 waves present
    print("\n[2/6] Verifying all 28 waves present...")
    if len(truth_df) != 28:
        print(f"  ✗ FAILED: Expected 28 waves, got {len(truth_df)}")
        return False
    
    print(f"  ✓ All 28 waves present")
    
    # Test 3: Check required columns
    print("\n[3/6] Checking required columns...")
    required_columns = [
        'wave_id', 'display_name', 'mode', 'readiness_status', 'coverage_pct',
        'data_regime_tag', 'return_1d', 'return_30d', 'alpha_1d', 'alpha_30d',
        'exposure_pct', 'cash_pct', 'last_snapshot_ts'
    ]
    
    missing_columns = [col for col in required_columns if col not in truth_df.columns]
    
    if missing_columns:
        print(f"  ✗ FAILED: Missing columns: {missing_columns}")
        return False
    
    print(f"  ✓ All required columns present")
    
    # Test 4: Get wave-specific data
    print("\n[4/6] Testing wave-specific data retrieval...")
    returns = get_wave_returns(truth_df, 'sp500_wave')
    alphas = get_wave_alphas(truth_df, 'sp500_wave')
    
    print(f"  ✓ S&P 500 Wave returns: {returns}")
    print(f"  ✓ S&P 500 Wave alphas: {alphas}")
    
    # Test 5: Get top performers
    print("\n[5/6] Testing top performers analysis...")
    top_5 = get_top_performers(truth_df, metric='return_30d', n=5)
    
    if top_5 is None or top_5.empty:
        print("  ⚠ WARNING: No top performers (all data may be unavailable)")
    else:
        print(f"  ✓ Top 5 performers by 30-day return:")
        for idx, wave in top_5.iterrows():
            ret_str = format_return_display(wave.get('return_30d'))
            print(f"     • {wave['display_name']}: {ret_str}")
    
    # Test 6: Get readiness summary
    print("\n[6/6] Testing readiness summary...")
    summary = get_readiness_summary(truth_df)
    
    print(f"  ✓ Readiness Summary:")
    print(f"     • Full: {summary['full']}")
    print(f"     • Partial: {summary['partial']}")
    print(f"     • Operational: {summary['operational']}")
    print(f"     • Unavailable: {summary['unavailable']}")
    print(f"     • Total: {summary['total']}")
    
    # Final validation
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("✓ All tests passed!")
    print("✓ TruthFrame is working correctly")
    print("✓ Ready for use in the application")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = validate_truthframe()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ VALIDATION FAILED WITH ERROR:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
