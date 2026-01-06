#!/usr/bin/env python3
"""
Verification script for portfolio alpha attribution rolling window fix.

This script verifies that:
1. 60D diagnostics show correct date range (~60 trading days)
2. No silent fallback to inception
3. Invalid windows are properly marked
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_60d_window():
    """Verify 60D attribution uses strict rolling window."""
    print("=" * 70)
    print("VERIFICATION: 60D Attribution Rolling Window")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("\nLoading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
        print(f"  Date range: {price_book.index[0].strftime('%Y-%m-%d')} to {price_book.index[-1].strftime('%Y-%m-%d')}")
        
        # Compute attribution with 60D period
        print("\nComputing 60D attribution...")
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[60]
        )
        
        if not result['success']:
            print(f"⚠️  Attribution computation failed: {result['failure_reason']}")
            return False
        
        print("✓ Attribution computation succeeded")
        
        # Check 60D period summary
        if '60D' not in result['period_summaries']:
            print("❌ FAIL: 60D period not in summaries")
            return False
        
        summary = result['period_summaries']['60D']
        print("\n" + "=" * 70)
        print("60D PERIOD SUMMARY")
        print("=" * 70)
        
        # Display all diagnostic fields
        print(f"\nStatus: {summary.get('status', 'N/A')}")
        print(f"Reason: {summary.get('reason', 'N/A')}")
        print(f"Requested Period Days: {summary.get('requested_period_days', 'N/A')}")
        print(f"Actual Rows Used: {summary.get('actual_rows_used', 'N/A')}")
        print(f"Is Exact Window: {summary.get('is_exact_window', 'N/A')}")
        print(f"Window Type: {summary.get('window_type', 'N/A')}")
        print(f"Start Date: {summary.get('start_date', 'N/A')}")
        print(f"End Date: {summary.get('end_date', 'N/A')}")
        
        # Check if valid or invalid
        if summary.get('status') == 'valid':
            print("\n" + "=" * 70)
            print("CUMULATIVE RETURNS (60D WINDOW)")
            print("=" * 70)
            print(f"Cumulative Realized: {summary.get('cum_real', 0):+.4%}")
            print(f"Cumulative Unoverlay: {summary.get('cum_sel', 0):+.4%}")
            print(f"Cumulative Benchmark: {summary.get('cum_bm', 0):+.4%}")
            print(f"\nTotal Alpha: {summary.get('total_alpha', 0):+.4%}")
            print(f"Selection Alpha: {summary.get('selection_alpha', 0):+.4%}")
            print(f"Overlay Alpha: {summary.get('overlay_alpha', 0):+.4%}")
            print(f"Residual: {summary.get('residual', 0):+.6%}")
            
            # Verify date range is not from 2021
            start_date_str = summary.get('start_date', '')
            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                if start_date.year <= 2021:
                    print(f"\n❌ FAIL: Start date is from {start_date.year} (inception era)")
                    return False
                else:
                    print(f"\n✅ PASS: Start date is from {start_date.year} (not inception)")
            
            # Verify window is exactly 60 days
            if summary.get('actual_rows_used') != 60:
                print(f"❌ FAIL: actual_rows_used={summary.get('actual_rows_used')}, expected 60")
                return False
            else:
                print("✅ PASS: Window is exactly 60 trading days")
            
            # Verify is_exact_window is True
            if not summary.get('is_exact_window'):
                print("❌ FAIL: is_exact_window should be True")
                return False
            else:
                print("✅ PASS: is_exact_window is True")
            
            print("\n" + "=" * 70)
            print("✅ ALL CHECKS PASSED")
            print("=" * 70)
            return True
            
        elif summary.get('status') == 'invalid':
            print("\n" + "=" * 70)
            print("INVALID PERIOD (AS EXPECTED WITH INSUFFICIENT DATA)")
            print("=" * 70)
            print(f"Reason: {summary.get('reason', 'unknown')}")
            print("\n✅ PASS: Invalid period properly handled (no silent fallback)")
            return True
        else:
            print(f"\n❌ FAIL: Unknown status: {summary.get('status')}")
            return False
        
    except Exception as e:
        print(f"\n❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification."""
    success = verify_60d_window()
    
    print("\n" + "=" * 70)
    if success:
        print("VERIFICATION COMPLETE: ✅ ALL CHECKS PASSED")
    else:
        print("VERIFICATION FAILED: ❌ SOME CHECKS FAILED")
    print("=" * 70)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
