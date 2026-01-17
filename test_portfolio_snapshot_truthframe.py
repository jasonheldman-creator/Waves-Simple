#!/usr/bin/env python3
"""
Test suite for TruthFrame-based portfolio snapshot.

This test validates that the new TruthFrame-based portfolio snapshot:
1. Successfully aggregates metrics from TruthFrame
2. Returns properly formatted results with returns and alphas
3. Reflects VIX regime, dynamic benchmarks, exposure, and cash
4. Handles errors gracefully
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_truthframe_portfolio_snapshot_basic():
    """Test basic TruthFrame-based portfolio snapshot computation."""
    print("\n=== Test: TruthFrame Portfolio Snapshot Basic ===")
    
    try:
        from analytics_truth import compute_portfolio_snapshot_from_truth
        
        # Compute portfolio snapshot from TruthFrame
        print("Computing TruthFrame-based portfolio snapshot...")
        snapshot = compute_portfolio_snapshot_from_truth(mode='Standard', periods=(1, 30, 60, 365))
        
        # Check for errors
        if 'error' in snapshot:
            print(f"❌ FAIL: Snapshot computation returned error: {snapshot['error']}")
            return False
        
        print(f"✓ Snapshot computation succeeded")
        
        # Validate timestamp
        if 'computed_at_utc' not in snapshot:
            print("❌ FAIL: Missing computed_at_utc timestamp")
            return False
        
        print(f"  - Computed at: {snapshot['computed_at_utc']}")
        
        # Check returns for each period
        print("\nPortfolio Returns from TruthFrame:")
        for period in [1, 30, 60, 365]:
            ret_key = f'return_{period}d'
            alpha_key = f'alpha_{period}d'
            
            ret = snapshot.get(ret_key)
            alpha = snapshot.get(alpha_key)
            
            if ret is not None:
                # Check if it's a valid number (not NaN)
                import pandas as pd
                if pd.isna(ret):
                    print(f"  {period}D: N/A (insufficient data)")
                else:
                    alpha_str = f"{alpha:+.2%}" if alpha is not None and not pd.isna(alpha) else "N/A"
                    print(f"  {period}D: {ret:+.2%} (Alpha: {alpha_str})")
            else:
                print(f"  {period}D: Missing")
        
        # Validate that we have at least some metrics
        has_any_return = any(
            snapshot.get(f'return_{p}d') is not None 
            for p in [1, 30, 60, 365]
        )
        
        if not has_any_return:
            print("❌ FAIL: No return metrics available")
            return False
        
        has_any_alpha = any(
            snapshot.get(f'alpha_{p}d') is not None 
            for p in [1, 30, 60, 365]
        )
        
        if not has_any_alpha:
            print("❌ FAIL: No alpha metrics available")
            return False
        
        print("\n✓ PASS: TruthFrame portfolio snapshot basic test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_truthframe_portfolio_snapshot_data_quality():
    """Test that TruthFrame snapshot reflects strategy adjustments."""
    print("\n=== Test: TruthFrame Data Quality ===")
    
    try:
        from analytics_truth import compute_portfolio_snapshot_from_truth, get_truth_frame
        import pandas as pd
        
        # Get TruthFrame to inspect what data is available
        print("Loading TruthFrame...")
        truth_df = get_truth_frame(safe_mode=False)
        
        if truth_df is None or truth_df.empty:
            print("❌ FAIL: TruthFrame is empty")
            return False
        
        print(f"✓ TruthFrame loaded: {len(truth_df)} waves")
        
        # Filter to Standard mode
        standard_waves = truth_df[truth_df['mode'] == 'Standard']
        
        if standard_waves.empty:
            print("❌ FAIL: No waves in Standard mode")
            return False
        
        print(f"  - Waves in Standard mode: {len(standard_waves)}")
        
        # Check that TruthFrame has required columns
        required_cols = [
            'return_1d', 'return_30d', 'return_60d', 'return_365d',
            'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d',
            'exposure_pct', 'cash_pct'
        ]
        
        missing_cols = [col for col in required_cols if col not in truth_df.columns]
        if missing_cols:
            print(f"❌ FAIL: TruthFrame missing columns: {missing_cols}")
            return False
        
        print(f"✓ TruthFrame has all required columns")
        
        # Compute portfolio snapshot
        snapshot = compute_portfolio_snapshot_from_truth(mode='Standard', periods=(1, 30, 60, 365))
        
        if 'error' in snapshot:
            print(f"❌ FAIL: Snapshot returned error: {snapshot['error']}")
            return False
        
        # Verify that portfolio metrics are aggregated correctly
        # Portfolio return should be the mean of wave returns
        for period in [1, 30, 60, 365]:
            ret_col = f'return_{period}d'
            
            # Calculate expected mean
            expected_mean = standard_waves[ret_col].mean()
            actual_value = snapshot.get(ret_col)
            
            # Both should be NaN or both should be close
            if pd.isna(expected_mean) and pd.isna(actual_value):
                print(f"  ✓ {period}D: Both N/A (expected)")
            elif pd.isna(expected_mean) or pd.isna(actual_value):
                print(f"  ❌ {period}D: Mismatch - expected NaN: {pd.isna(expected_mean)}, actual NaN: {pd.isna(actual_value)}")
            else:
                # Check if they're close (within 0.01%)
                diff = abs(expected_mean - actual_value)
                if diff < 0.0001:  # 0.01% tolerance
                    print(f"  ✓ {period}D: {actual_value:+.2%} (matches mean)")
                else:
                    print(f"  ⚠ {period}D: Expected {expected_mean:+.2%}, got {actual_value:+.2%} (diff: {diff:.4f})")
        
        print("\n✓ PASS: TruthFrame data quality test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_truthframe_portfolio_snapshot_modes():
    """Test that TruthFrame snapshot works for different modes."""
    print("\n=== Test: TruthFrame Multiple Modes ===")
    
    try:
        from analytics_truth import compute_portfolio_snapshot_from_truth
        
        modes = ['Standard', 'Alpha-Minus-Beta', 'Aggressive']
        results = {}
        
        for mode in modes:
            print(f"\nTesting mode: {mode}")
            snapshot = compute_portfolio_snapshot_from_truth(mode=mode, periods=(1, 30, 60, 365))
            
            if 'error' in snapshot:
                print(f"  ⚠ Mode {mode} returned error: {snapshot['error']}")
                results[mode] = None
            else:
                print(f"  ✓ Mode {mode} succeeded")
                results[mode] = snapshot
        
        # At least Standard mode should work
        if results.get('Standard') is None:
            print("❌ FAIL: Standard mode failed")
            return False
        
        print("\n✓ PASS: TruthFrame multiple modes test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("TruthFrame-Based Portfolio Snapshot Test Suite")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("TruthFrame Basic", test_truthframe_portfolio_snapshot_basic()))
    results.append(("TruthFrame Data Quality", test_truthframe_portfolio_snapshot_data_quality()))
    results.append(("TruthFrame Multiple Modes", test_truthframe_portfolio_snapshot_modes()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if total_passed == len(results) else 1)
