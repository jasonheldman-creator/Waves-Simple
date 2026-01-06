#!/usr/bin/env python3
"""
Simple validation script for portfolio alpha attribution.
This tests the function directly and displays the results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 70)
    print("PORTFOLIO ALPHA ATTRIBUTION VALIDATION")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("\n1. Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days × {len(price_book.columns)} tickers")
        print(f"  Date range: {price_book.index[0].strftime('%Y-%m-%d')} to {price_book.index[-1].strftime('%Y-%m-%d')}")
        
        # Compute attribution
        print("\n2. Computing portfolio alpha attribution...")
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        if not result['success']:
            print(f"❌ FAIL: Attribution computation failed: {result['failure_reason']}")
            return False
        
        print("✓ Attribution computation succeeded")
        
        # Display warnings
        if result['warnings']:
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        # Display daily series info
        print("\n3. Daily series:")
        print(f"  - daily_realized_return: {len(result['daily_realized_return'])} days")
        print(f"  - daily_unoverlay_return: {len(result['daily_unoverlay_return'])} days")
        print(f"  - daily_benchmark_return: {len(result['daily_benchmark_return'])} days")
        print(f"  - daily_exposure: {len(result['daily_exposure'])} days")
        
        # Display period summaries
        print("\n4. Period Summaries:")
        print("-" * 70)
        
        for period_key in ['30D', '60D', '365D']:
            if period_key in result['period_summaries']:
                summary = result['period_summaries'][period_key]
                print(f"\n{period_key} Period:")
                print(f"  Cumulative Realized Return: {summary['cum_real']:+.4%}")
                print(f"  Cumulative Selection Return: {summary['cum_sel']:+.4%}")
                print(f"  Cumulative Benchmark Return: {summary['cum_bm']:+.4%}")
                print(f"  ---")
                print(f"  Total Alpha:      {summary['total_alpha']:+.4%}")
                print(f"  Selection Alpha:  {summary['selection_alpha']:+.4%}")
                print(f"  Overlay Alpha:    {summary['overlay_alpha']:+.4%}")
                print(f"  Residual:         {summary['residual']:+.6%}")
                
                # Check residual
                if abs(summary['residual']) > 0.001:
                    print(f"  ⚠️  WARNING: Residual exceeds tolerance!")
                else:
                    print(f"  ✓ Residual within tolerance")
        
        # Display since inception summary
        if result['since_inception_summary']:
            inception = result['since_inception_summary']
            print(f"\nSince Inception ({inception['days']} days):")
            print(f"  Cumulative Realized Return: {inception['cum_real']:+.4%}")
            print(f"  Cumulative Selection Return: {inception['cum_sel']:+.4%}")
            print(f"  Cumulative Benchmark Return: {inception['cum_bm']:+.4%}")
            print(f"  ---")
            print(f"  Total Alpha:      {inception['total_alpha']:+.4%}")
            print(f"  Selection Alpha:  {inception['selection_alpha']:+.4%}")
            print(f"  Overlay Alpha:    {inception['overlay_alpha']:+.4%}")
            print(f"  Residual:         {inception['residual']:+.6%}")
            
            if abs(inception['residual']) > 0.001:
                print(f"  ⚠️  WARNING: Residual exceeds tolerance!")
            else:
                print(f"  ✓ Residual within tolerance")
        
        print("\n" + "=" * 70)
        print("✅ VALIDATION COMPLETE - All checks passed")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
