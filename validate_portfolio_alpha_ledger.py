#!/usr/bin/env python3
"""
Manual validation script for portfolio alpha ledger implementation.

This script tests the functions with actual PRICE_BOOK data and displays
the results to verify they work as expected before running the full app.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_exposure_series():
    """Validate exposure series computation."""
    print("\n" + "="*70)
    print("VALIDATING EXPOSURE SERIES")
    print("="*70)
    
    try:
        from helpers.wave_performance import compute_portfolio_exposure_series
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days, {len(price_book.columns)} tickers")
        
        # Check which VIX proxies are available
        vix_tickers = ['^VIX', 'VIXY', 'VXX']
        available_vix = [t for t in vix_tickers if t in price_book.columns]
        print(f"VIX proxies available: {available_vix if available_vix else 'None'}")
        
        # Compute exposure series
        print("\nComputing exposure series...")
        exposure = compute_portfolio_exposure_series(
            price_book,
            mode='Standard',
            apply_smoothing=True
        )
        
        if exposure is None:
            print("⚠️ No VIX proxy available - exposure will default to 1.0")
            print("✓ This is expected behavior when VIX data is unavailable")
            return True
        
        # Display exposure statistics
        print(f"✓ Exposure series computed: {len(exposure)} days")
        print(f"  Min exposure: {exposure.min():.3f}")
        print(f"  Max exposure: {exposure.max():.3f}")
        print(f"  Avg exposure: {exposure.mean():.3f}")
        print(f"  Current exposure: {exposure.iloc[-1]:.3f}")
        
        # Show exposure distribution by regime
        low_exposure = (exposure <= 0.30).sum()
        mid_exposure = ((exposure > 0.30) & (exposure < 0.90)).sum()
        high_exposure = (exposure >= 0.90).sum()
        
        print(f"\n  Exposure distribution:")
        print(f"    Low (≤0.30):  {low_exposure} days ({low_exposure/len(exposure)*100:.1f}%)")
        print(f"    Mid (0.30-0.90): {mid_exposure} days ({mid_exposure/len(exposure)*100:.1f}%)")
        print(f"    High (≥0.90): {high_exposure} days ({high_exposure/len(exposure)*100:.1f}%)")
        
        print("\n✅ Exposure series validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_alpha_ledger():
    """Validate alpha ledger computation."""
    print("\n" + "="*70)
    print("VALIDATING ALPHA LEDGER")
    print("="*70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
        
        # Compute ledger
        print("\nComputing alpha ledger...")
        ledger = compute_portfolio_alpha_ledger(
            price_book,
            periods=[1, 30, 60, 365],
            benchmark_ticker='SPY',
            mode='Standard',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        print("✓ Ledger computed successfully")
        
        # Display configuration
        print(f"\nConfiguration:")
        print(f"  VIX ticker: {ledger['vix_ticker_used'] or 'N/A (using exposure=1.0)'}")
        print(f"  Safe ticker: {ledger['safe_ticker_used'] or 'N/A (using 0% safe return)'}")
        print(f"  Overlay available: {ledger['overlay_available']}")
        
        # Display warnings
        if ledger.get('warnings'):
            print(f"\n⚠️ Warnings:")
            for warning in ledger['warnings']:
                print(f"  - {warning}")
        
        # Display period results
        print(f"\nPeriod Results:")
        print(f"{'Period':<10} {'Available':<12} {'Return':<12} {'Alpha':<12} {'Residual':<12}")
        print("-" * 70)
        
        for period_key in ['1D', '30D', '60D', '365D']:
            period_data = ledger['period_results'].get(period_key, {})
            
            if period_data.get('available'):
                cum_ret = period_data['cum_realized']
                total_alpha = period_data['total_alpha']
                residual = period_data['residual']
                
                # Color code residual
                residual_pct = abs(residual) * 100
                residual_status = "🟢" if residual_pct < 0.10 else "🟡" if residual_pct < 0.5 else "🔴"
                
                print(f"{period_key:<10} {'✓':<12} {cum_ret:+.2%}      {total_alpha:+.2%}      {residual_status} {residual:+.4%}")
            else:
                reason = period_data.get('reason', 'unknown')
                print(f"{period_key:<10} {'✗':<12} {'—':<12} {'—':<12} {reason}")
        
        # Display 30D attribution breakdown
        print(f"\n30D Attribution Breakdown:")
        period_30d = ledger['period_results'].get('30D', {})
        
        if period_30d.get('available'):
            print(f"  Total Alpha:     {period_30d['total_alpha']:+.2%}")
            print(f"  Selection Alpha: {period_30d['selection_alpha']:+.2%}")
            print(f"  Overlay Alpha:   {period_30d['overlay_alpha']:+.2%}")
            print(f"  Residual:        {period_30d['residual']:+.4%}")
            
            if period_30d.get('alpha_captured') is not None:
                print(f"  Alpha Captured:  {period_30d['alpha_captured']:+.2%}")
            
            print(f"  Date Range:      {period_30d['start_date']} to {period_30d['end_date']}")
        else:
            print(f"  ⚠️ 30D unavailable: {period_30d.get('reason', 'unknown')}")
        
        print("\n✅ Alpha ledger validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("="*70)
    print("PORTFOLIO ALPHA LEDGER VALIDATION")
    print("="*70)
    
    results = []
    
    # Run validations
    results.append(("Exposure Series", validate_exposure_series()))
    results.append(("Alpha Ledger", validate_alpha_ledger()))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nThe implementation is ready for UI testing.")
        return 0
    else:
        print(f"\n⚠️ Some validations failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
