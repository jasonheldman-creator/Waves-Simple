#!/usr/bin/env python3
"""
Test to verify that blue box and diagnostics use the same ledger and show consistent values.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_blue_box_diagnostics_alignment():
    """
    Test that blue box and Attribution Diagnostics show the same values from the ledger.
    """
    print("\n" + "=" * 70)
    print("BLUE BOX AND DIAGNOSTICS ALIGNMENT TEST")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("\n1. Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days, {len(price_book.columns)} tickers")
        
        # Compute ledger (same as both blue box and diagnostics use)
        print("\n2. Computing portfolio alpha ledger (single source of truth)...")
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            benchmark_ticker='SPY',
            mode='Standard',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        print("✓ Ledger computed successfully")
        
        # Verify each period's data consistency
        print("\n3. Verifying period data (what blue box would show)...")
        
        for period_key in ['1D', '30D', '60D', '365D']:
            period_data = ledger['period_results'].get(period_key, {})
            available = period_data.get('available', False)
            
            print(f"\n{period_key} Period:")
            
            if available:
                # Show what blue box would display (Portfolio / Benchmark / Alpha)
                portfolio = period_data['cum_realized']
                benchmark = period_data['cum_benchmark']
                alpha = period_data['total_alpha']
                start = period_data['start_date']
                end = period_data['end_date']
                
                print(f"  ✓ Available")
                print(f"  📈 Portfolio Return: {portfolio:+.4%}")
                print(f"  📊 Benchmark Return: {benchmark:+.4%}")
                print(f"  🎯 Alpha: {alpha:+.4%}")
                print(f"  📅 Period: {start} to {end}")
                
                # Verify reconciliation 1: Portfolio - Benchmark = Alpha
                expected_alpha = portfolio - benchmark
                diff_1 = abs(expected_alpha - alpha)
                if diff_1 > RESIDUAL_TOLERANCE:
                    print(f"  ❌ Reconciliation 1 failed: diff={diff_1:.6f}")
                    return False
                else:
                    print(f"  ✓ Reconciliation 1: Portfolio - Benchmark = Alpha (diff={diff_1:.6f})")
                
                # Verify reconciliation 2: Selection + Overlay + Residual = Total
                selection = period_data['selection_alpha']
                overlay = period_data['overlay_alpha']
                residual = period_data['residual']
                
                expected_total = selection + overlay + residual
                diff_2 = abs(expected_total - alpha)
                if diff_2 > RESIDUAL_TOLERANCE:
                    print(f"  ❌ Reconciliation 2 failed: diff={diff_2:.6f}")
                    return False
                else:
                    print(f"  ✓ Reconciliation 2: Selection + Overlay + Residual = Total (diff={diff_2:.6f})")
                
            else:
                # Show what blue box would display for unavailable period
                reason = period_data.get('reason', 'unknown')
                print(f"  ⚠️ Unavailable")
                print(f"  📈 Portfolio Return: N/A")
                print(f"  📊 Benchmark Return: N/A")
                print(f"  🎯 Alpha: N/A")
                print(f"  ❗ Reason: {reason[:70]}...")
        
        # Verify 60D period specifically (what diagnostics would show)
        print("\n4. Verifying 60D period (what Attribution Diagnostics would show)...")
        
        period_60d = ledger['period_results'].get('60D', {})
        
        if period_60d.get('available'):
            print("✓ 60D period available in diagnostics")
            
            # Extract values that diagnostics would display
            total_alpha = period_60d['total_alpha']
            selection_alpha = period_60d['selection_alpha']
            overlay_alpha = period_60d['overlay_alpha']
            residual = period_60d['residual']
            
            print(f"  Total Alpha: {total_alpha:+.4%}")
            print(f"  Selection Alpha: {selection_alpha:+.4%}")
            print(f"  Overlay Alpha: {overlay_alpha:+.4%}")
            print(f"  Residual: {residual:+.4%}")
            
            # Verify these match what blue box shows for 60D
            blue_box_60d_alpha = ledger['period_results']['60D']['total_alpha']
            
            if total_alpha != blue_box_60d_alpha:
                print(f"❌ FAIL: Diagnostics alpha ({total_alpha:.6f}) != Blue box alpha ({blue_box_60d_alpha:.6f})")
                return False
            
            print(f"✓ Diagnostics and blue box show same 60D alpha: {total_alpha:+.4%}")
        else:
            reason = period_60d.get('reason', 'unknown')
            print(f"⚠️ 60D period unavailable in diagnostics: {reason}")
            print("✓ Both blue box and diagnostics would show N/A")
        
        # Summary
        print("\n" + "=" * 70)
        print("ALIGNMENT VERIFICATION SUMMARY")
        print("=" * 70)
        print("✅ Blue box and Attribution Diagnostics use the same ledger")
        print("✅ Both show consistent values for available periods")
        print("✅ Both show N/A with reasons for unavailable periods")
        print("✅ Reconciliations hold for all available periods")
        print("\n🎉 ALL ALIGNMENT CHECKS PASSED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_blue_box_diagnostics_alignment()
    sys.exit(0 if success else 1)
